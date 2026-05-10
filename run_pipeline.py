# -*- coding: utf-8 -*-
"""
run_pipeline.py — Updated entry point with USE_HLS toggle
==========================================================
Drop-in replacement for ReversedDesertification.ipynb that supports
both the original MODIS 500 m pipeline and the new HLS/Sentinel-2
30 m pixel-level pipeline.

Usage (Colab or local):
    %run run_pipeline.py              # MODIS mode (default)
    %run run_pipeline.py --hls        # HLS pixel-level mode
    %run run_pipeline.py --hls --pixels 500  # HLS with 500 pixels
"""

import argparse
import os
import sys

# ── Parse args (safe for both CLI and Colab %run) ─────────────
parser = argparse.ArgumentParser(description="Desertification SciML Pipeline")
parser.add_argument('--hls', action='store_true',
                    help='Use HLS/Sentinel-2 30 m pixel-level pipeline')
parser.add_argument('--pixels', type=int, default=1000,
                    help='Number of pixels to sample per ROI (HLS mode only)')
args, _ = parser.parse_known_args()

USE_HLS  = args.hls
N_PIXELS = args.pixels

# ── Imports ───────────────────────────────────────────────────
import ee
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100.0
import matplotlib.pyplot as plt

from desertification.config import (
    K_REGION, CEIL_CONFIG, THRESH, FEATURES, FEATURES_HLS,
    NITER_RAJ, NITER_GOBI,
    EQ_SLOPE_RAJ, EQ_SLOPE_GOBI,
    EQ_BOUNDS_RAJ, EQ_BOUNDS_GOBI,
    SIGN_CHANGE_RANGE_RAJ, SIGN_CHANGE_RANGE_GOBI,
    YEARS, DT, N_SIMS, FORECAST_START,
)
from desertification.data import load_or_fetch, plot_data_quality
from desertification.features import add_lst_anomaly, build_features, feature_audit
from desertification.ode_discovery import discover_and_compile_ode
from desertification.dynamics import make_drivers, lyapunov_with_sanity, estimate_noise_sigma
from desertification.dynamics import sde_euler_maruyama, run_monte_carlo
from desertification.interventions import (
    check_intervention_signs, auto_correct_plan,
    GOBI_IRRIGATION_BASE, GOBI_RESTORATION_BASE,
    RAJ_CANAL_BOOST_BASE, RAJ_DROUGHT_BASE,
)
from desertification.ews import compute_ews
from desertification.plotting import (
    plot_main_figure, plot_diagnostics,
    plot_feature_timeseries, plot_collapse_risk,
)

print(f"Pipeline modules loaded (USE_HLS={USE_HLS}, N_PIXELS={N_PIXELS}).")


# ═════════════════════════════════════════════════════════════
# 1. Data Extraction
# ═════════════════════════════════════════════════════════════
PROJECT_ID = 'reversing-desertification'
try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

rajasthan_roi = ee.Geometry.BBox(73.8, 29.3, 74.3, 29.8)
gobi_roi      = ee.Geometry.BBox(108.2, 40.2, 108.8, 40.8)

print("\n[1/6] Extracting satellite data...")
if USE_HLS:
    print(f"  Mode: HLS/Sentinel-2 pixel-level ({N_PIXELS} pixels @ 30 m)")
    df_rajasthan = load_or_fetch(rajasthan_roi, '2005-01-01', '2023-12-31',
                                 'rajasthan_hls.csv', use_hls=True, n_pixels=N_PIXELS)
    df_gobi      = load_or_fetch(gobi_roi,      '2005-01-01', '2023-12-31',
                                 'gobi_hls.csv',      use_hls=True, n_pixels=N_PIXELS)
else:
    print("  Mode: MODIS spatial-average (500 m)")
    df_rajasthan = load_or_fetch(rajasthan_roi, '2005-01-01', '2023-12-31', 'rajasthan_fixed.csv')
    df_gobi      = load_or_fetch(gobi_roi,      '2005-01-01', '2023-12-31', 'gobi_fixed.csv')

print(f"  Rajasthan shape: {df_rajasthan.shape}")
print(f"  Gobi shape:      {df_gobi.shape}")

if not USE_HLS:
    plot_data_quality(df_rajasthan, df_gobi)


# ═════════════════════════════════════════════════════════════
# 2. Feature Engineering
# ═════════════════════════════════════════════════════════════
print("\n[2/6] Feature engineering...")
if not USE_HLS:
    df_rajasthan = add_lst_anomaly(df_rajasthan)
    df_gobi      = add_lst_anomaly(df_gobi)
else:
    # For HLS data, LST anomaly is computed on the merged MODIS columns
    if 'LST_Day' in df_rajasthan.columns:
        df_rajasthan = add_lst_anomaly(df_rajasthan)
    if 'LST_Day' in df_gobi.columns:
        df_gobi = add_lst_anomaly(df_gobi)

df_rajasthan = build_features(df_rajasthan, 'Rajasthan Canal')
df_gobi      = build_features(df_gobi,      'Gobi Green Wall')

print(f"  Rajasthan: {df_rajasthan.shape}, Gobi: {df_gobi.shape}")
active_features = FEATURES_HLS if USE_HLS else FEATURES
print(f"  Available features: {[f for f in active_features if f in df_rajasthan.columns]}")


# ═════════════════════════════════════════════════════════════
# 3. ODE Discovery (Symbolic Regression)
# ═════════════════════════════════════════════════════════════
print("\n[3/6] Running PySR symbolic regression...")

ode_raj, feat_raj, eq_raj = discover_and_compile_ode(
    df_rajasthan, "Rajasthan Canal",
    niterations=NITER_RAJ, eq_slope_max=EQ_SLOPE_RAJ,
    eq_bounds=EQ_BOUNDS_RAJ, sign_change_range=SIGN_CHANGE_RANGE_RAJ,
    ceil_config=CEIL_CONFIG['Rajasthan Canal'],
    K_default=K_REGION['Rajasthan Canal'],
)

ode_gobi, feat_gobi, eq_gobi = discover_and_compile_ode(
    df_gobi, "Gobi Green Wall",
    niterations=NITER_GOBI, eq_slope_max=EQ_SLOPE_GOBI,
    eq_bounds=EQ_BOUNDS_GOBI, sign_change_range=SIGN_CHANGE_RANGE_GOBI,
    ceil_config=CEIL_CONFIG['Gobi Green Wall'],
    K_default=K_REGION['Gobi Green Wall'],
)


# ═════════════════════════════════════════════════════════════
# 4. Stability Analysis
# ═════════════════════════════════════════════════════════════
print("\n[4/6] Computing Lyapunov exponents + Early Warning Signals...")

# For pixel-level data, use median NDVI across pixels for stability analysis
if USE_HLS and 'pixel_id' in df_rajasthan.columns:
    # Aggregate to region-level for dynamics analysis
    _df_raj_agg = df_rajasthan.groupby(df_rajasthan.index).median(numeric_only=True)
    _df_gobi_agg = df_gobi.groupby(df_gobi.index).median(numeric_only=True)
else:
    _df_raj_agg = df_rajasthan
    _df_gobi_agg = df_gobi

drivers_raj  = make_drivers(_df_raj_agg, feat_raj,  K_REGION['Rajasthan Canal'])
drivers_gobi = make_drivers(_df_gobi_agg, feat_gobi, K_REGION['Gobi Green Wall'])

v0_raj  = float(_df_raj_agg['NDVI'].iloc[-1])
v0_gobi = float(_df_gobi_agg['NDVI'].iloc[-1])

lambda_raj  = lyapunov_with_sanity(ode_raj,  drivers_raj,  v0_raj,  "Rajasthan Canal")
lambda_gobi = lyapunov_with_sanity(ode_gobi, drivers_gobi, v0_gobi, "Gobi Green Wall")

print("\n  >> Rajasthan EWS:")
ews_raj  = compute_ews(_df_raj_agg)
print("\n  >> Gobi EWS:")
ews_gobi = compute_ews(_df_gobi_agg)


# ═════════════════════════════════════════════════════════════
# 5. Intervention Scenarios & Monte Carlo
# ═════════════════════════════════════════════════════════════
print("\n[5/6] Running intervention sign audits + Monte Carlo simulations...")

noise_raj  = estimate_noise_sigma(_df_raj_agg['NDVI'].values, region_name='Rajasthan Canal')
noise_gobi = estimate_noise_sigma(_df_gobi_agg['NDVI'].values, region_name='Gobi Green Wall')
print(f"  Calibrated noise sigma: Rajasthan={noise_raj:.4f}, Gobi={noise_gobi:.4f}")
STEPS      = int(YEARS / DT)
TIME_AXIS  = np.linspace(0, YEARS, STEPS)

# Sign audits
gobi_irr_s = check_intervention_signs(ode_gobi, drivers_gobi, feat_gobi, GOBI_IRRIGATION_BASE,  ndvi_test=v0_gobi)
gobi_rst_s = check_intervention_signs(ode_gobi, drivers_gobi, feat_gobi, GOBI_RESTORATION_BASE, ndvi_test=v0_gobi)
raj_bst_s  = check_intervention_signs(ode_raj,  drivers_raj,  feat_raj,  RAJ_CANAL_BOOST_BASE,  ndvi_test=v0_raj)
raj_dry_s  = check_intervention_signs(ode_raj,  drivers_raj,  feat_raj,  RAJ_DROUGHT_BASE,       ndvi_test=v0_raj)

# Auto-correct (FIX-N + FIX-AA)
GOBI_IRRIGATION  = auto_correct_plan(GOBI_IRRIGATION_BASE,  gobi_irr_s, 'positive', ode_gobi.selection_tier)
GOBI_RESTORATION = auto_correct_plan(GOBI_RESTORATION_BASE, gobi_rst_s, 'positive', ode_gobi.selection_tier)
RAJ_CANAL_BOOST  = auto_correct_plan(RAJ_CANAL_BOOST_BASE,  raj_bst_s,  'positive', ode_raj.selection_tier)
RAJ_DROUGHT      = auto_correct_plan(RAJ_DROUGHT_BASE,      raj_dry_s,  'negative', ode_raj.selection_tier)

# Monte Carlo simulations
print("\n  Running Monte Carlo...")
sims_raj        = run_monte_carlo(ode_raj,  v0_raj,  drivers_raj,  DT, STEPS, noise_raj,  N_SIMS, ode_features=feat_raj, region_name='Rajasthan Canal')
sims_raj_boost  = run_monte_carlo(ode_raj,  v0_raj,  drivers_raj,  DT, STEPS, noise_raj,  N_SIMS, RAJ_CANAL_BOOST,  feat_raj, region_name='Rajasthan Canal')
sims_raj_drought= run_monte_carlo(ode_raj,  v0_raj,  drivers_raj,  DT, STEPS, noise_raj,  N_SIMS, RAJ_DROUGHT,      feat_raj, region_name='Rajasthan Canal')
sims_gobi_base  = run_monte_carlo(ode_gobi, v0_gobi, drivers_gobi, DT, STEPS, noise_gobi, N_SIMS, ode_features=feat_gobi, region_name='Gobi Green Wall')
sims_gobi_irr   = run_monte_carlo(ode_gobi, v0_gobi, drivers_gobi, DT, STEPS, noise_gobi, N_SIMS, GOBI_IRRIGATION,  feat_gobi, region_name='Gobi Green Wall')
sims_gobi_rest  = run_monte_carlo(ode_gobi, v0_gobi, drivers_gobi, DT, STEPS, noise_gobi, N_SIMS, GOBI_RESTORATION, feat_gobi, region_name='Gobi Green Wall')

# Collapse probabilities
def pc_terminal(sims):
    return float(np.mean(sims[:, -1] < THRESH))

def pc_ever(sims):
    return float(np.mean(np.any(sims < THRESH, axis=1)))

def pc_persistent(sims, years=2.0, dt=DT):
    """Fraction of runs that stay below threshold for >= given consecutive years."""
    win = max(1, int(round(years / dt)))
    below = sims < THRESH
    kernel = np.ones(win, dtype=int)
    hits = np.zeros(sims.shape[0], dtype=bool)
    for i in range(sims.shape[0]):
        run = np.convolve(below[i].astype(int), kernel, mode='valid')
        hits[i] = np.any(run >= win)
    return float(np.mean(hits))

pc_raj_end = pc_terminal(sims_raj);       pc_rb_end = pc_terminal(sims_raj_boost);   pc_rd_end = pc_terminal(sims_raj_drought)
pc_gb_end  = pc_terminal(sims_gobi_base); pc_irr_end = pc_terminal(sims_gobi_irr);   pc_rst_end = pc_terminal(sims_gobi_rest)

pc_raj = pc_ever(sims_raj);       pc_rb = pc_ever(sims_raj_boost);   pc_rd = pc_ever(sims_raj_drought)
pc_gb  = pc_ever(sims_gobi_base); pc_irr = pc_ever(sims_gobi_irr);   pc_rst = pc_ever(sims_gobi_rest)

pc_raj_p2y = pc_persistent(sims_raj, years=2.0);       pc_rb_p2y = pc_persistent(sims_raj_boost, years=2.0);   pc_rd_p2y = pc_persistent(sims_raj_drought, years=2.0)
pc_gb_p2y  = pc_persistent(sims_gobi_base, years=2.0); pc_irr_p2y = pc_persistent(sims_gobi_irr, years=2.0);   pc_rst_p2y = pc_persistent(sims_gobi_rest, years=2.0)

print(f"\n  Collapse probabilities (50 yr, ever below threshold):")
print(f"    Rajasthan: baseline={pc_raj:.1%}, boost={pc_rb:.1%}, drought={pc_rd:.1%}")
print(f"    Gobi:      baseline={pc_gb:.1%}, irrigation={pc_irr:.1%}, restoration={pc_rst:.1%}")


# ═════════════════════════════════════════════════════════════
# 6. Visualisation
# ═════════════════════════════════════════════════════════════
print("\n[6/6] Generating figures...")
os.makedirs('images', exist_ok=True)

plot_main_figure(
    TIME_AXIS, sims_raj, sims_gobi_base,
    sims_gobi_irr, sims_gobi_rest,
    sims_raj_boost, sims_raj_drought,
    pc_raj, pc_gb, pc_irr, pc_rst, pc_rb, pc_rd,
    ode_raj, ode_gobi, drivers_raj, drivers_gobi,
    v0_raj, v0_gobi, lambda_raj, lambda_gobi,
    RAJ_CANAL_BOOST, RAJ_DROUGHT,
    GOBI_IRRIGATION, GOBI_RESTORATION,
    ews_gobi=ews_gobi,
)
plot_diagnostics(
    ode_raj, ode_gobi, drivers_raj, drivers_gobi,
    v0_raj, v0_gobi, lambda_raj, lambda_gobi,
    _df_raj_agg if USE_HLS else df_rajasthan,
    _df_gobi_agg if USE_HLS else df_gobi,
    feat_raj, feat_gobi,
)
plot_feature_timeseries(
    _df_raj_agg if USE_HLS else df_rajasthan,
    _df_gobi_agg if USE_HLS else df_gobi,
)
plot_collapse_risk(pc_raj, pc_gb, pc_irr, pc_rst, pc_rb, pc_rd)


# ═════════════════════════════════════════════════════════════
# 7. Summary
# ═════════════════════════════════════════════════════════════
print("\n" + "="*64)
print("  MODULAR PIPELINE — FINAL SUMMARY")
print(f"  Data source: {'HLS/Sentinel-2 (30 m, pixel-level)' if USE_HLS else 'MODIS (500 m, spatial average)'}")
print("="*64)
print(f"  Rajasthan collapse prob (50 yr, ever below threshold):")
print(f"    Baseline              : {pc_raj:.1%}")
print(f"    + Canal boost (20-yr) : {pc_rb:.1%}  (delta: {pc_raj-pc_rb:+.1%})")
print(f"    + Canal failure       : {pc_rd:.1%}  (delta: {pc_rd-pc_raj:+.1%})")
print(f"\n  Gobi collapse prob (50 yr, ever below threshold):")
print(f"    Baseline              : {pc_gb:.1%}")
print(f"    + Irrigation boost    : {pc_irr:.1%}  (delta: {pc_gb-pc_irr:+.1%})")
print(f"    + Full restoration    : {pc_rst:.1%}  (delta: {pc_gb-pc_rst:+.1%})")
print()
print(f"  lambda Rajasthan : {lambda_raj:.4f} ({'STABLE' if lambda_raj<0 else 'UNSTABLE'}) [P{ode_raj.selection_tier}]")
print(f"  lambda Gobi      : {lambda_gobi:.4f} ({'STABLE' if lambda_gobi<0 else 'UNSTABLE'}) [P{ode_gobi.selection_tier}]")
print()
print("  Discovered ODEs:")
print(f"    Rajasthan: {eq_raj}")
if ode_raj.alpha_inject != 0.0:
    print(f"               + {ode_raj.alpha_inject:.4f}*NDVI_logistic(K={ode_raj.K_default}) [FIX-AE]")
print(f"    Gobi:      {eq_gobi}")
if ode_gobi.alpha_inject != 0.0:
    print(f"               + {ode_gobi.alpha_inject:.4f}*NDVI_logistic(K={ode_gobi.K_default}) [FIX-AE]")

fit_r2_raj = getattr(ode_raj, 'fit_r2', None)
fit_r2_gobi = getattr(ode_gobi, 'fit_r2', None)
fit_nmae_raj = getattr(ode_raj, 'fit_nmae_iqr', None)
fit_nmae_gobi = getattr(ode_gobi, 'fit_nmae_iqr', None)
fmt_num = lambda v, spec: format(v, spec) if isinstance(v, (int, float)) else 'NA'
print()
print(f"  Fit diagnostics: Rajasthan (R^2={fmt_num(fit_r2_raj, '+.3f')}, nMAE(IQR)={fmt_num(fit_nmae_raj, '.3f')}), "
      f"Gobi (R^2={fmt_num(fit_r2_gobi, '+.3f')}, nMAE(IQR)={fmt_num(fit_nmae_gobi, '.3f')})")

if USE_HLS:
    print(f"\n  ▶ HLS mode: {df_rajasthan.shape[0]} Rajasthan rows, "
          f"{df_gobi.shape[0]} Gobi rows (pixel-level)")
    print(f"  ▶ Compare R² with MODIS baseline to quantify improvement")

# Save machine-readable run summary
import json
os.makedirs('images', exist_ok=True)
run_summary = {
    'data_source': 'HLS_Sentinel2_30m' if USE_HLS else 'MODIS_500m',
    'n_pixels': N_PIXELS if USE_HLS else 1,
    'collapse_probability_ever': {
        'rajasthan': {'baseline': pc_raj, 'canal_boost': pc_rb, 'drought': pc_rd},
        'gobi': {'baseline': pc_gb, 'irrigation_boost': pc_irr, 'restoration': pc_rst},
    },
    'lyapunov': {'rajasthan': lambda_raj, 'gobi': lambda_gobi},
    'fit_r2': {'rajasthan': fit_r2_raj, 'gobi': fit_r2_gobi},
    'fit_nmae_iqr': {'rajasthan': fit_nmae_raj, 'gobi': fit_nmae_gobi},
    'selection_tier': {'rajasthan': ode_raj.selection_tier, 'gobi': ode_gobi.selection_tier},
    'equations': {'rajasthan': eq_raj, 'gobi': eq_gobi},
}
suffix = '_hls' if USE_HLS else ''
with open(f'images/run_summary{suffix}.json', 'w', encoding='utf-8') as f:
    json.dump(run_summary, f, indent=2)
print(f"  Saved run summary: images/run_summary{suffix}.json")
print("="*64)
