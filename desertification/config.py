# -*- coding: utf-8 -*-
"""
desertification.config
======================
All constants, hyperparameters, variable maps, and metadata.

Merged from:
  - V6 Cell 24: vars_map (with agg field), CEIL_CONFIG (gradient drift_base),
    PYSR constants, EQ_BOUNDS, SIGN_CHANGE_RANGE
  - V5 Cell 23: VARIABLE_META (with rich descriptions), K_REGION,
    CEIL_CONFIG (region-specific ndvi_points)
  - V3 Cell 20: FIX-A through FIX-P constants

Cumulative fix provenance
-------------------------
FIX-A   Aridity_safe replaces raw Aridity_Index (div-by-zero spikes)
FIX-B   Drift magnitude guard DRIFT_CLIP
FIX-C   Slope check at equilibrium
FIX-D   Multi-driver preference within priority tiers
FIX-E   Lyapunov sanity check |λ| > 10
FIX-F   Post-run lever effectiveness audit
FIX-G   Per-feature pre-PySR audit
FIX-H   feature_audit NDVI self-corr crash fix
FIX-I   No-sign-change ODE rejection all tiers
FIX-J   Slope threshold tightened 5 → 2
FIX-K   Drift clip relaxed 0.30 → 0.40
FIX-L   PySR hyperparameters tuned
FIX-M   Nested-trig rejection
FIX-N   Auto-correct inverted lever signs
FIX-O   SDE burn-in drift damping (2 yr ramp)
FIX-P   Evapo mode threshold 0.40 → 0.30
FIX-Q   Drop LOW-CORR features per-region
FIX-R   P4 requires sign-change; P5 = fallback
FIX-S   Per-region PySR iterations
FIX-T   IQR ribbon (p25–p75) vs p10–p90
FIX-U   Per-region eq bounds and slope
FIX-V   Synthetic ceiling augmentation
FIX-W   Per-region sign-change scan range
FIX-X   has_stable_sign_change() + P2.5 priority tier
FIX-Y   NDVI_logistic feature + carrying capacity K
FIX-Z   Region-specific ceiling augmentation (CEIL_CONFIG)
FIX-AA  Physical direction metadata + derived variable protection
FIX-AB  K-modulating SDE for attractor-shifting interventions
FIX-AC  Phase portrait panel
FIX-AD  Deseasonalise dNDVI/dt training target
FIX-AE  Post-hoc NDVI_logistic injection when NDVI absent
FIX-AF  Gradient ceiling rows
FIX-AG  NDVI_logistic removed from PySR features
FIX-AH  PySR ^ constraint removed
FIX-BA  Per-region dead-feature exclusions (NDVI_sq/Carbon_Flux/GPP/Fire_Pressure for Rajasthan)
FIX-BB  NDVI_poly (NDVI^1.5) replaces NDVI_sq for Rajasthan nonlinearity
FIX-BC  Reject sin/cos applied to bounded [0,1] non-oscillatory variables
FIX-BD  NITER_RAJ increased 260 → 320 for deeper Rajasthan search
"""

# ============================================================
# GEE Variable Map
# ============================================================
vars_map = {
    # Vegetation
    'NDVI':      {'id': "MODIS/061/MOD13Q1",             'band': 'NDVI',                'res': 500,   'agg': 'median'},
    'EVI':       {'id': "MODIS/061/MOD13Q1",             'band': 'EVI',                 'res': 500,   'agg': 'median'},
    'LAI':       {'id': "MODIS/061/MCD15A3H",            'band': 'Lai',                 'res': 500,   'agg': 'median'},
    'FPAR':      {'id': "MODIS/061/MCD15A3H",            'band': 'Fpar',                'res': 500,   'agg': 'median'},
    # Carbon
    'GPP':       {'id': "MODIS/061/MOD17A2H",            'band': 'Gpp',                 'res': 500,   'agg': 'median'},
    # Temperature
    'LST_Day':   {'id': "MODIS/061/MOD11A2",             'band': 'LST_Day_1km',         'res': 1000,  'agg': 'median'},
    'LST_Night': {'id': "MODIS/061/MOD11A2",             'band': 'LST_Night_1km',       'res': 1000,  'agg': 'median'},
    # Water
    'Rain':      {'id': "UCSB-CHG/CHIRPS/DAILY",         'band': 'precipitation',       'res': 5000,  'agg': 'sum'},
    'SoilMoist': {'id': "NASA/FLDAS/NOAH01/C/GL/M/V001", 'band': 'SoilMoi00_10cm_tavg', 'res': 10000, 'agg': 'median'},
    'Evapo':     {'id': "MODIS/061/MOD16A2",             'band': 'ET',                  'res': 500,   'agg': 'median'},
    # Surface reflectance
    'Albedo':    {'id': "MODIS/061/MCD43A3",             'band': 'Albedo_WSA_shortwave', 'res': 500,  'agg': 'median'},
    # Disturbance
    'Fire':      {'id': "MODIS/061/MCD64A1",             'band': 'BurnDate',            'res': 500,   'agg': 'median'},
    # Atmosphere
    'AOD':       {'id': "MODIS/061/MCD19A2_GRANULES",    'band': 'Optical_Depth_055',   'res': 1000,  'agg': 'median'},
}

# ============================================================
# HLS / Sentinel-2 High-Resolution Vegetation Data
# ============================================================
# NASA Harmonized Landsat-Sentinel (HLS) products provide 30 m
# vegetation indices with ~3-day revisit (Landsat 8/9 + Sentinel-2).
# These replace MODIS 500 m NDVI/EVI when USE_HLS = True.

# HLS Landsat component (30 m, 2013-present via GEE)
hls_vars_map = {
    'NDVI_HLS': {
        'collection': 'NASA/HLS/HLSL30/v002',
        'bands_red': 'B4',       # Red
        'bands_nir': 'B5',       # NIR
        'res': 30,
        'formula': 'ndvi',       # (NIR - Red) / (NIR + Red)
    },
    'EVI_HLS': {
        'collection': 'NASA/HLS/HLSL30/v002',
        'bands_blue': 'B2',      # Blue
        'bands_red':  'B4',      # Red
        'bands_nir':  'B5',      # NIR
        'res': 30,
        'formula': 'evi',        # 2.5 * (NIR-Red) / (NIR + 6*Red - 7.5*Blue + 1)
    },
}

# Sentinel-2 specific products (10-20 m, 2015-present)
sentinel2_vars_map = {
    'NDVI_S2': {
        'collection': 'COPERNICUS/S2_SR_HARMONIZED',
        'bands_red': 'B4',       # Red (10 m)
        'bands_nir': 'B8',       # NIR (10 m)
        'res': 10,
        'qa_band': 'SCL',        # Scene Classification Layer for cloud masking
    },
    'RedEdge_NDVI': {
        'collection': 'COPERNICUS/S2_SR_HARMONIZED',
        'bands_red': 'B5',       # Red-Edge 1 (20 m)
        'bands_nir': 'B8A',      # NIR narrow (20 m)
        'res': 20,
        'qa_band': 'SCL',
    },
    'NDMI': {
        'collection': 'COPERNICUS/S2_SR_HARMONIZED',
        'bands_nir':  'B8',      # NIR (10 m)
        'bands_swir': 'B11',     # SWIR 1 (20 m)
        'res': 20,
        'qa_band': 'SCL',
        'formula': 'ndmi',       # (NIR - SWIR) / (NIR + SWIR)
    },
}

# ── Pixel Sampling Configuration ─────────────────────────────
# Instead of spatial averaging (which compresses NDVI range and
# destroys pixel-level dynamics), sample individual 30 m pixels.
PIXEL_SAMPLE_COUNT       = 1000      # pixels per ROI
PIXEL_SAMPLE_SEED        = 42
TEMPORAL_COMPOSITE_DAYS  = 16        # Match Landsat-8/9 revisit cycle
USE_HLS                  = False     # Toggle: True = HLS pixel-level, False = MODIS average

# Sentinel-2 SCL cloud-mask classes to reject (cloud, shadow, snow)
S2_MASK_VALUES = [0, 1, 2, 3, 8, 9, 10, 11]  # No data, Defective, Shadows, Cloud*

# ============================================================
# Per-Region Carrying Capacity
# ============================================================
K_REGION = {
    'Rajasthan Canal': 0.65,
    'Gobi Green Wall': 0.25,
}

# ============================================================
# PySR Hyperparameters  (FIX-L, FIX-S, FIX-AH, FIX-BD)
# ============================================================
# FIX-BD: Increased from 260 → 320 to give PySR more time to find
# higher-R² candidates in Rajasthan's weak-signal landscape.
NITER_RAJ  = 320
NITER_GOBI = 240

PYSR_BINARY_OPS      = ["+", "*", "-", "/"]
PYSR_BINARY_OPS_NO_DIV = ["+", "*", "-"]
PYSR_UNARY_OPS       = ["exp", "sin", "cos"]
PYSR_MAXSIZE         = 12
PYSR_POPULATIONS     = 30

RAJASTHAN_DENOM_PRONE_FEATURES = (
    'LST_Day_anom',
    'Temp_Delta_anom',
)

# FIX-BA: Per-region feature exclusions based on data quality.
# Rajasthan NDVI ∈ [0.2, 0.55] → NDVI_sq ∈ [0.04, 0.30]: near-zero variance, acts as noise.
# Carbon_Flux, GPP, Fire_Pressure are dead/sparse signals for semi-arid Rajasthan.
# FIX-BB: NDVI_poly (NDVI^1.5) is intentionally NOT excluded — it has higher variance
# than NDVI_sq in the Rajasthan range ([0.09, 0.41] vs [0.04, 0.30]) and provides
# the nonlinear vegetation term PySR needs without the flatness problem.
FEATURES_TO_EXCLUDE_BY_REGION = {
    'Rajasthan Canal': {'NDVI_sq', 'Carbon_Flux', 'GPP', 'Fire_Pressure'},
    'Gobi Green Wall': set(),
}

# ============================================================
# Drift & Equilibrium Constants  (FIX-K, FIX-J, FIX-U, FIX-W)
# ============================================================
DRIFT_CLIP = 0.40

EQ_SLOPE_RAJ   = 3.0
EQ_SLOPE_GOBI  = 2.0
EQ_BOUNDS_RAJ  = (0.08, 0.95)
EQ_BOUNDS_GOBI = (0.05, 0.90)

SIGN_CHANGE_RANGE_RAJ  = (0.10, 0.80)
SIGN_CHANGE_RANGE_GOBI = (0.05, 0.25)

# ============================================================
# Ceiling Augmentation Config  (FIX-Z, FIX-AF)
# ============================================================
CEIL_CONFIG = {
    'Rajasthan Canal': {'ndvi_points': [0.62, 0.66, 0.70, 0.74, 0.78], 'drift_base': -0.035, 'weight': 5},
    'Gobi Green Wall': {'ndvi_points': [0.21, 0.23, 0.25, 0.27, 0.29], 'drift_base': -0.012, 'weight': 5},
}

FLOOR_CONFIG = {
    'Rajasthan Canal': {'ndvi_points': [0.05, 0.08, 0.10, 0.12, 0.15], 'drift_base': 0.020, 'weight': 5},
    'Gobi Green Wall': {'ndvi_points': [0.02, 0.03, 0.04, 0.05, 0.06], 'drift_base': 0.008, 'weight': 5},
}

# ============================================================
# Physical Direction Metadata  (FIX-AA)
# ============================================================
VARIABLE_META = {
    'NDVI':              {'physical_good': +1, 'lever_role': 'diagnostic_only',
                          'description': 'state variable, not a lever'},
    'NDVI_sq':           {'physical_good': +1, 'lever_role': 'diagnostic_only'},
    'NDVI_logistic':     {'physical_good': +1, 'lever_role': 'diagnostic_only'},
    # FIX-BB: NDVI_poly = NDVI^1.5; higher variance than NDVI_sq in Rajasthan range.
    # Diagnostic only — depends on current NDVI state, not an independent lever.
    'NDVI_poly':         {'physical_good': +1, 'lever_role': 'diagnostic_only',
                          'description': 'NDVI^1.5: nonlinear vegetation term with better '
                                         'variance than NDVI_sq in constrained NDVI ranges'},
    'Rain_norm':         {'physical_good': +1, 'lever_role': 'direct',
                          'description': 'rainfall — more helps vegetation'},
    'SoilMoist':         {'physical_good': +1, 'lever_role': 'direct',
                          'description': 'soil water — more helps vegetation'},
    'FPAR':              {'physical_good': +1, 'lever_role': 'direct',
                          'description': 'canopy light use — more = healthier'},
    'Dust_Stress':       {'physical_good': -1, 'lever_role': 'direct',
                          'description': 'dust/aerosol load — less is better'},
    'Fire_Pressure':     {'physical_good': -1, 'lever_role': 'direct',
                          'description': 'fire disturbance — less is better'},
    'Carbon_Flux':       {'physical_good': +1, 'lever_role': 'direct'},
    'Albedo':            {'physical_good': -1, 'lever_role': 'direct',
                          'description': 'surface reflectance — lower = more vegetation'},
    'LST_Day_anom':      {'physical_good': -1, 'lever_role': 'direct',
                          'description': 'heat anomaly — lower is better'},
    'Temp_Delta_anom':   {'physical_good': -1, 'lever_role': 'direct'},
    'GPP':               {'physical_good': +1, 'lever_role': 'direct'},
    'Groundwater_Anomaly': {'physical_good': +1, 'lever_role': 'direct'},
    'Aridity_safe':      {'physical_good': -1, 'lever_role': 'diagnostic_only',
                          'derived_from': ['Rain_norm', 'NDVI'],
                          'description': 'DERIVED: Rain_norm/(NDVI+0.10) — '
                                         'diagnostic only, NOT a valid lever'},
}

# ============================================================
# Feature Sets  (FIX-AG, FIX-BB)
# ============================================================
# NDVI_logistic is computed in build_features for FIX-AE post-hoc injection
# but is NOT included in the PySR feature set (FIX-AG).
#
# NDVI_poly = NDVI^1.5 IS included (FIX-BB). It provides the nonlinear
# vegetation term for Rajasthan after NDVI_sq is excluded by FIX-BA.
# In the Rajasthan range (NDVI ∈ [0.20, 0.55]):
#   NDVI_sq  → [0.04, 0.30]  std ≈ 0.025  (very flat — excluded)
#   NDVI_poly → [0.09, 0.41]  std ≈ 0.045  (usable signal)
FEATURES = [
    'NDVI', 'NDVI_sq', 'NDVI_poly',
    'Rain_norm', 'SoilMoist',
    'LST_Day_anom', 'Temp_Delta_anom',
    'Albedo', 'Carbon_Flux', 'FPAR', 'GPP',
    'Dust_Stress', 'Fire_Pressure',
    'Groundwater_Anomaly',
]

# Extended feature set when using HLS/Sentinel-2 pixel-level data.
# RedEdge_NDVI uses Sentinel-2 B5/B8A (20 m) — not available from MODIS.
# NDMI (Normalised Difference Moisture Index) uses B8/B11 (20 m).
# NDVI_spatial_std captures intra-ROI vegetation heterogeneity at 30 m.
FEATURES_HLS = FEATURES + [
    'RedEdge_NDVI',       # S2-only: red-edge chlorophyll-sensitive NDVI
    'NDMI',               # S2-only: normalised difference moisture index
    'NDVI_spatial_std',   # Pixel-level: spatial heterogeneity within ROI
    'NDVI_p10',           # S2-only: 10th percentile of NDVI across pixels
    'NDVI_p90',           # S2-only: 90th percentile of NDVI across pixels
    'NDVI_std',           # S2-only: standard deviation of NDVI across pixels
]

# NDVI-like features that must be recomputed dynamically during simulation.
# FIX-BB: NDVI_poly added — it depends on current NDVI state.
NDVI_DERIVED_FEATURES = {'NDVI', 'NDVI_sq', 'NDVI_logistic', 'NDVI_poly'}

LOW_CORR_THRESHOLD = 0.05
LOW_CORR_THRESHOLD_BY_REGION = {
    'Rajasthan Canal': 0.08,
    'Gobi Green Wall': 0.05,
}
FLAT_THRESHOLD       = 0.50
ZERO_FRAC_THRESHOLD  = 0.85

RAJASTHAN_MAX_FEATURES  = 8
DIAGNOSTIC_EQ_LIMIT     = 30

RELIABILITY_MIN_R2          = 0.10
RAJASTHAN_P1_MIN_R2         = 0.03
RAJASTHAN_P2_MIN_R2         = 0.03
RELIABILITY_MAX_TIER        = 3
RELIABILITY_MAX_CLIP_SAT    = 0.25
RELIABILITY_MAX_NMAE_IQR    = 0.90
RELIABILITY_RELAXED_MAX_TIER = 5

# ============================================================
# Simulation Constants
# ============================================================
THRESH          = 0.10
YEARS           = 50
DT              = 0.1
N_SIMS          = 150
FORECAST_START  = 2024.0

NOISE_CALIBRATION_SCALE   = 0.70
NOISE_MIN_SIGMA           = 0.0025
NOISE_MAX_SIGMA_DEFAULT   = 0.015
NOISE_MAX_SIGMA_BY_REGION = {
    'Rajasthan Canal': 0.018,
    'Gobi Green Wall': 0.012,
}
NOISE_REGION_MULTIPLIER = {
    'Rajasthan Canal': 1.00,
    'Gobi Green Wall': 0.85,
}

K_SHIFT_TRANSLATION_GAIN_DEFAULT    = 0.75
K_SHIFT_TRANSLATION_GAIN_BY_REGION  = {
    'Rajasthan Canal': 0.55,
    'Gobi Green Wall': 0.80,
}
K_SHIFT_TRANSLATION_CLIP = 0.10

SIM_DRIVER_NOISE_REL = {
    'Rain_norm': 0.05,
    'SoilMoist': 0.04,
    'FPAR':      0.03,
}
SIM_DRIVER_NOISE_MIN_ABS = {
    'Rain_norm': 0.010,
    'SoilMoist': 0.003,
    'FPAR':      0.005,
}
SIM_DRIVER_NOISE_REGION_SCALE = {
    'Rajasthan Canal': 1.20,
    'Gobi Green Wall': 0.90,
}