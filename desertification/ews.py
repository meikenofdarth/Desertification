# -*- coding: utf-8 -*-
"""
desertification.ews
===================
Early Warning Signals (EWS) for ecosystem collapse detection.

Restored from Cell 13 (v1 Expanded pipeline) — this module was
absent from both V5 and V6.

Based on the theory of Critical Slowing Down (CSD): as a system
approaches a tipping point its resilience decreases, manifest as:
  • Rising variance        — the system takes longer to recover from shocks
  • Rising lag-1 AR(1)     — stronger "memory" / slower return to equilibrium
  • Rising skewness        — asymmetric fluctuations near tipping point
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau


def compute_ews(df, ndvi_col='NDVI', window=24, detrend=True):
    """
    Compute statistical early-warning signals of ecosystem collapse.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the `ndvi_col` column.
    ndvi_col : str
        Column name for the vegetation index (default 'NDVI').
    window : int
        Rolling window in months (default 24 = 2 years).
    detrend : bool
        If True, removes a rolling-mean trend before computing
        statistics (removes seasonality confound).

    Returns
    -------
    ews : pd.DataFrame
        With columns: NDVI_detrended, EWS_Variance, EWS_AC1,
        EWS_Skewness, EWS_Composite (z-score average of all three).
    """
    series = df[ndvi_col].copy()

    if detrend:
        trend  = series.rolling(window=window, center=True, min_periods=1).mean()
        series = series - trend

    ews = pd.DataFrame(index=df.index)
    ews['NDVI_detrended'] = series

    # Rolling variance
    ews['EWS_Variance'] = series.rolling(
        window=window, min_periods=window // 2
    ).var()

    # Rolling lag-1 autocorrelation
    def rolling_ac1(x):
        if len(x) < 4:
            return np.nan
        r, _ = pearsonr(x[:-1], x[1:])
        return r

    ews['EWS_AC1'] = series.rolling(
        window=window, min_periods=window // 2
    ).apply(rolling_ac1, raw=True)

    # Rolling skewness (asymmetric fluctuations near tipping point)
    ews['EWS_Skewness'] = series.rolling(
        window=window, min_periods=window // 2
    ).skew()

    # Composite EWS: z-score each signal, average them
    def zscore(s):
        return (s - s.mean()) / (s.std() + 1e-9)

    ews['EWS_Composite'] = (
        zscore(ews['EWS_Variance'].fillna(0)) +
        zscore(ews['EWS_AC1'].fillna(0)) +
        zscore(ews['EWS_Skewness'].fillna(0))
    ) / 3.0

    # Kendall-tau trend statistic for each signal
    valid = ews.dropna(subset=['EWS_Variance', 'EWS_AC1'])
    if len(valid) > 10:
        tau_var, p_var = kendalltau(range(len(valid)), valid['EWS_Variance'])
        tau_ac1, p_ac1 = kendalltau(range(len(valid)), valid['EWS_AC1'])
        print(f"  EWS Kendall-τ  Variance: {tau_var:+.3f}  (p={p_var:.3f})")
        print(f"  EWS Kendall-τ  AC(1):    {tau_ac1:+.3f}  (p={p_ac1:.3f})")
        if tau_var > 0.2 and p_var < 0.1:
            print("  ⚠  Rising variance detected — potential early warning signal.")
        if tau_ac1 > 0.2 and p_ac1 < 0.1:
            print("  ⚠  Rising AC(1) detected — potential early warning signal.")

    return ews
