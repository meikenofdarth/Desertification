# -*- coding: utf-8 -*-
"""
desertification.data
====================
Google Earth Engine data extraction, scaling, quality checks, and CSV caching.

Extracted from V6 Cell 24 (most evolved version):
  - fetch_data_failsafe: includes GRACE groundwater, habitat patchiness,
    sum-based rain aggregation (FIX-1), Evapo masking (FIX-2/FIX-P),
    SoilMoist sanity check (FIX-3), and full scientific scaling.
  - plot_data_quality: data quality diagnostic figure.
  - load_or_fetch: CSV cache wrapper to avoid repeated GEE calls.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import vars_map


def fetch_data_failsafe(roi, start_date, end_date):
    """
    Extract all variables from GEE for the given ROI and date range.

    Performs server-side monthly aggregation, scientific unit scaling,
    derived feature engineering, and gap interpolation.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest.
    start_date, end_date : str
        ISO date strings, e.g. '2005-01-01', '2023-12-31'.

    Returns
    -------
    pd.DataFrame
        Monthly time series indexed by date with all extracted and
        engineered features.
    """
    import ee

    date_list = ee.List.sequence(
        0, ee.Date(end_date).difference(ee.Date(start_date), 'month').subtract(1))
    start_ee = ee.Date(start_date)

    def process_month(n):
        m_start = start_ee.advance(n, 'month')
        m_end   = m_start.advance(1, 'month')
        fd      = {'date': m_start.format('YYYY-MM-dd')}

        for name, info in vars_map.items():
            col      = ee.ImageCollection(info['id']).filterDate(m_start, m_end).select(info['band'])
            fallback = ee.Image.constant(-9999).rename(info['band'])
            img      = ee.Image(ee.Algorithms.If(
                col.size().gt(0),
                col.sum() if info['agg'] == 'sum' else col.median(),
                fallback))

            reducers = ee.Reducer.mean().combine(
                reducer2=ee.Reducer.stdDev(), sharedInputs=True)
            stats    = img.reduceRegion(
                reducer=reducers, geometry=roi,
                scale=info['res'], maxPixels=1e9)

            fd[name] = ee.Dictionary(stats).get(info['band'] + '_mean', -9999)

            # Spatial heterogeneity (patchiness) from NDVI std-dev
            if name == 'NDVI':
                fd['Habitat_Patchiness'] = ee.Dictionary(stats).get(
                    info['band'] + '_stdDev', -9999)

        # Groundwater: average the three GRACE solutions
        gw_col  = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/LAND").filterDate(m_start, m_end)
        gw_fall = ee.Image.constant(-9999).rename('mean')
        gw_img  = ee.Image(ee.Algorithms.If(
            gw_col.size().gt(0),
            gw_col.median().select(
                ['lwe_thickness_csr', 'lwe_thickness_gfz', 'lwe_thickness_jpl']
            ).reduce(ee.Reducer.mean()),
            gw_fall))
        gw_stat = gw_img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=roi, scale=100000, maxPixels=1e9)
        fd['GroundWater'] = gw_stat.get('mean', -9999)

        return ee.Feature(None, fd)

    print("  Requesting server-side computation...")
    raw = ee.FeatureCollection(date_list.map(process_month)).getInfo()
    df  = pd.DataFrame([f['properties'] for f in raw['features']])
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').replace(-9999, np.nan).apply(pd.to_numeric, errors='coerce')

    # ── Evapo masking (FIX-2, FIX-P) ─────────────────────────
    if 'Evapo' in df:
        df['Evapo'] = df['Evapo'].where(df['Evapo'] < 30000, np.nan)
        suspicious  = df['Evapo'].value_counts(normalize=True)
        suspicious  = suspicious[suspicious > 0.30].index.tolist()
        if suspicious:
            df['Evapo'] = df['Evapo'].replace(suspicious, np.nan)

    # ── Scientific scaling (MODIS unit conversions) ───────────
    scale_map = {
        'NDVI': 0.0001, 'Habitat_Patchiness': 0.0001,
        'EVI': 0.0001, 'LAI': 0.1,
        'FPAR': 0.01, 'GPP': 0.0001, 'Albedo': 0.001,
        'LST_Day': 0.02, 'LST_Night': 0.02,
        'Evapo': 0.1, 'AOD': 0.001,
    }
    for col, factor in scale_map.items():
        if col in df:
            df[col] *= factor

    # ── SoilMoist sanity check (FIX-3) ────────────────────────
    if 'SoilMoist' in df:
        sm = df['SoilMoist'].mean()
        print(f"  SoilMoist mean: {sm:.4f} m3/m3  {'ok' if sm > 0.01 else 'WARNING'}")

    # ── Derived / engineered features ─────────────────────────
    if 'LST_Day' in df and 'LST_Night' in df:
        df['Temp_Delta'] = df['LST_Day'] - df['LST_Night']

    if 'GPP' in df and 'FPAR' in df:
        df['Carbon_Flux'] = df['GPP'] * df['FPAR']

    if 'AOD' in df:
        df['Dust_Stress'] = df['AOD']

    if 'Fire' in df:
        df['Fire_Pressure'] = (
            (df['Fire'] > 0).astype(float)
            .rolling(window=12, min_periods=1).sum()
        )

    if 'Rain' in df:
        rain_95         = df['Rain'].quantile(0.95)
        df['Rain_norm'] = (df['Rain'] / (rain_95 + 1e-6)).clip(0, 1)

    if 'Rain' in df and 'NDVI' in df:
        df['Aridity_safe'] = (
            df['Rain_norm'] / (df['NDVI'].clip(lower=0.05) + 0.10)
        ).clip(0, 5)

    if 'GroundWater' in df:
        df['Groundwater_Anomaly'] = df['GroundWater']

    # ── Interpolation ─────────────────────────────────────────
    df = df.interpolate(method='linear').ffill().bfill()
    return df


def plot_data_quality(df_raj, df_gobi, save_path='images/data_quality.png'):
    """Generate data quality diagnostic plots for both regions."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100)
    fig.patch.set_facecolor('white')

    df_raj[['NDVI', 'LAI']].plot(ax=axes[0], title="Vegetation — Rajasthan")

    df_raj[['SoilMoist']].plot(ax=axes[1], title="Hydrology — Rajasthan")
    ax_r = axes[1].twinx()
    df_raj['Rain'].plot(ax=ax_r, color='steelblue', alpha=0.7, label='Rain (mm/month)')
    ax_r.set_ylabel('Rain (mm/month)')
    ax_r.legend(loc='upper left')

    cols_new = [c for c in ['Carbon_Flux', 'FPAR', 'Dust_Stress', 'Aridity_safe']
                if c in df_raj.columns]
    df_raj[cols_new].plot(ax=axes[2], title="Engineered Variables — Rajasthan")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close('all')
    print(f"  Data quality plot saved: {save_path}")


def load_or_fetch(roi, start_date, end_date, cache_path, force_refresh=False):
    """
    Load data from CSV cache if available, otherwise fetch from GEE.

    Parameters
    ----------
    roi : ee.Geometry
        Region of interest (only used if fetching).
    start_date, end_date : str
        Date range.
    cache_path : str
        Path to the CSV cache file.
    force_refresh : bool
        If True, always re-fetch from GEE regardless of cache.

    Returns
    -------
    pd.DataFrame
    """
    if not force_refresh and os.path.exists(cache_path):
        print(f"  Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=['date'], index_col='date')
        return df

    df = fetch_data_failsafe(roi, start_date, end_date)
    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
    df.to_csv(cache_path)
    print(f"  Saved to {cache_path}")
    return df
