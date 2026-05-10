# -*- coding: utf-8 -*-
"""
desertification.features
========================
Feature engineering, LST anomaly computation, pre-PySR audit, and
feature selection.

Extracted from V6 Cell 24 with V5 documentation.
"""

import numpy as np
import pandas as pd

from .config import (
    K_REGION, FEATURES, FEATURES_HLS, NDVI_DERIVED_FEATURES,
    LOW_CORR_THRESHOLD, LOW_CORR_THRESHOLD_BY_REGION,
    FLAT_THRESHOLD, ZERO_FRAC_THRESHOLD,
    FEATURES_TO_EXCLUDE_BY_REGION,
)


def add_lst_anomaly(df):
    """
    Compute monthly climatological LST anomalies.

    For each LST column, subtracts the per-calendar-month mean to remove
    the seasonal cycle, exposing interannual thermal anomalies.

    Parameters
    ----------
    df : pd.DataFrame
        Must have DatetimeIndex and optionally LST_Day, LST_Night columns.

    Returns
    -------
    pd.DataFrame
        Copy with added *_anom columns.
    """
    df = df.copy()
    for col in ['LST_Day', 'LST_Night']:
        if col in df.columns:
            monthly_mean      = df[col].groupby(df.index.month).transform('mean')
            df[col + '_anom'] = (df[col] - 273.15) - (monthly_mean - 273.15)
    if 'LST_Day_anom' in df and 'LST_Night_anom' in df:
        df['Temp_Delta_anom'] = df['LST_Day_anom'] - df['LST_Night_anom']
    return df


def build_features(df, region_name='Rajasthan Canal'):
    """
    Add engineered features for ODE discovery.

    Computes:
    - NDVI_sq: NDVI squared (polynomial term; excluded for Rajasthan by FIX-BA).
    - NDVI_poly: NDVI^1.5 (FIX-BB). In the Rajasthan range [0.20, 0.55]:
        NDVI_sq  → [0.04, 0.30]  (std ≈ 0.025, very flat, excluded)
        NDVI_poly → [0.09, 0.41] (std ≈ 0.045, usable nonlinear signal)
      NDVI_poly provides the curvature term PySR needs without the flatness
      problem that caused NDVI_sq to be dropped.
    - NDVI_logistic: NDVI × (1 − NDVI/K), for FIX-AE post-hoc injection only.
      NOT included in PySR features (FIX-AG).

    Parameters
    ----------
    df : pd.DataFrame
    region_name : str
        Key into K_REGION for carrying capacity.

    Returns
    -------
    pd.DataFrame
        Copy with added columns.
    """
    df = df.copy()
    K  = K_REGION.get(region_name, 0.50)

    is_hls_data = 'RedEdge_NDVI' in df.columns

    if 'NDVI' in df:
        df['NDVI_sq']       = df['NDVI'] ** 2
        # FIX-BB: NDVI^1.5 — intermediate nonlinearity between NDVI and NDVI_sq.
        # Included in FEATURES; NOT excluded for Rajasthan (unlike NDVI_sq).
        df['NDVI_poly']     = df['NDVI'] ** 1.5
        # FIX-AG: computed for FIX-AE post-hoc use, NOT in FEATURES
        df['NDVI_logistic'] = df['NDVI'] * (1.0 - df['NDVI'] / K)

    # For HLS data, add Rain_norm and Aridity_safe if not present
    # (they may not have been computed if MODIS climate data was merged in)
    if is_hls_data:
        if 'Rain' in df.columns and 'Rain_norm' not in df.columns:
            rain_95 = df['Rain'].quantile(0.95)
            df['Rain_norm'] = (df['Rain'] / (rain_95 + 1e-6)).clip(0, 1)
        if 'Rain_norm' in df.columns and 'NDVI' in df.columns and 'Aridity_safe' not in df.columns:
            df['Aridity_safe'] = (
                df['Rain_norm'] / (df['NDVI'].clip(lower=0.05) + 0.10)
            ).clip(0, 5)
        if 'GroundWater' in df.columns and 'Groundwater_Anomaly' not in df.columns:
            df['Groundwater_Anomaly'] = df['GroundWater']

    return df


def feature_audit(df, region_name):
    """
    Pre-PySR audit: compute per-feature correlation with NDVI, flag
    outliers and flat-value features.

    (FIX-G, FIX-H, FIX-AI)

    Parameters
    ----------
    df : pd.DataFrame
    region_name : str

    Returns
    -------
    dict
        Mapping feature_name → {'r': Pearson r, 'flat': flat fraction,
        'zero': exact-zero fraction}.
    """
    is_pixel_level = 'pixel_id' in df.columns
    feat_list = FEATURES_HLS if is_pixel_level else FEATURES
    print(f"\n  Feature audit for {region_name}"
          f"{' (pixel-level HLS)' if is_pixel_level else ''}:")
    corrs = {}
    for f in feat_list:
        if f not in df.columns:
            continue
        if f == 'NDVI':
            # FIX-H: skip NDVI self-correlation (always 1.0)
            corrs[f] = {'r': 1.0, 'flat': 0.0, 'zero': 0.0}
            continue
        valid = df[['NDVI', f]].dropna()
        if len(valid) < 5:
            print(f"    {f:<25} insufficient data ({len(valid)} rows)")
            continue
        r = float(valid['NDVI'].corr(valid[f]))
        mode_vals = df[f].mode()
        flat_frac = float((df[f] == mode_vals.iloc[0]).mean()) if len(mode_vals) > 0 else 0.0
        zero_frac = float((df[f].fillna(0) == 0).mean())
        outlier_frac = float((df[f].abs() > df[f].quantile(0.99) * 3).mean())
        corrs[f] = {
            'r': r,
            'flat': flat_frac,
            'zero': zero_frac,
            'outlier': outlier_frac,
            'n_valid': len(valid),
        }
        flags = []
        if outlier_frac > 0.05:
            flags.append(f"outliers={outlier_frac:.1%}")
        if flat_frac > 0.3:
            flags.append(f"flat={flat_frac:.1%}")
        if zero_frac > 0.5:
            flags.append(f"zeros={zero_frac:.1%}")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"    {f:<25} r={r:+.3f}{flag_str}")
    return corrs


def safe_features(df, region_corrs=None, region_name=None, return_dropped=False):
    """
    Return the subset of FEATURES that exist in df and pass quality checks.

    (FIX-Q: drop LOW-CORR features, FIX-AI: drop flat features,
     FIX-BA: region-specific exclusions, FIX-BB: NDVI_poly included)

    Parameters
    ----------
    df : pd.DataFrame
    region_corrs : dict, optional
        Feature → {'r': corr, 'flat': flat_frac} (from feature_audit).
    region_name : str, optional
        Region name for region-specific filtering.
    return_dropped : bool, optional
        If True, return (features, dropped_info) tuple.

    Returns
    -------
    list of str or (list, dict)
        Feature names safe to use for PySR, optionally with dropped info dict.
    """
    feat = [f for f in FEATURES if f in df.columns and df[f].std() > 1e-8]

    # FIX-BA: Apply region-specific exclusions (zero-variance or dead features).
    # Note: NDVI_poly is intentionally NOT in the Rajasthan exclusion set (FIX-BB).
    dropped_excluded = []
    if region_name:
        exclude_set = FEATURES_TO_EXCLUDE_BY_REGION.get(region_name, set())
        for f in feat[:]:
            if f in exclude_set:
                feat.remove(f)
                dropped_excluded.append(f"{f} (region-specific exclusion)")
        if dropped_excluded:
            print(f"  FIX-BA: Excluded region-specific dead features: {dropped_excluded}")

    corr_threshold = LOW_CORR_THRESHOLD
    if region_name:
        corr_threshold = LOW_CORR_THRESHOLD_BY_REGION.get(region_name, LOW_CORR_THRESHOLD)

    dropped_corr = []
    dropped_flat = []
    dropped_zero = []

    if region_corrs:
        for f in feat[:]:
            info = region_corrs.get(f, None)
            if info is None or f in NDVI_DERIVED_FEATURES:
                continue
            r    = info['r']    if isinstance(info, dict) else info
            flat = info.get('flat', 0.0) if isinstance(info, dict) else 0.0
            zero = info.get('zero', 0.0) if isinstance(info, dict) else 0.0
            if zero > ZERO_FRAC_THRESHOLD:
                feat.remove(f)
                dropped_zero.append(f"{f} (zeros={zero:.0%})")
                continue
            if flat > FLAT_THRESHOLD:
                feat.remove(f)
                dropped_flat.append(f"{f} (flat={flat:.0%})")
                continue
            if not np.isnan(r) and abs(r) < corr_threshold:
                feat.remove(f)
                dropped_corr.append(f"{f} (r={r:+.3f})")
        if dropped_zero:
            print(f"  Dropped zero-heavy features: {dropped_zero}")
        if dropped_flat:
            print(f"  FIX-AI: Dropped flat features: {dropped_flat}")
        if dropped_corr:
            print(f"  FIX-Q: Dropped low-correlation features: {dropped_corr}")

    dropped = {
        'excluded': dropped_excluded,
        'zero':     dropped_zero,
        'flat':     dropped_flat,
        'low_corr': dropped_corr,
        'corr_threshold': corr_threshold,
    }
    if return_dropped:
        return feat, dropped
    return feat