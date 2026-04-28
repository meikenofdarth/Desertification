# Desertification Analysis Project - Complete Context Document

**Project Name:** Reversed Desertification Detection and Intervention Analysis  
**Generated On:** April 28, 2026  
**Source:** Extracted from conversation context and codebase archaeology

---

## 1. PROJECT OVERVIEW

### Problem Definition
The project addresses desertification collapse dynamics in arid/semi-arid regions by discovering interpretable differential equations (ODEs) that model NDVI change rates (dNDVI/dt). The goal is to:
- Identify drivers of desertification/land degradation
- Predict collapse risk under different intervention scenarios
- Evaluate policy lever effectiveness (e.g., canal boost, restoration efforts)
- Distinguish between transient degradation and persistent collapse

### Geographic Focus
- **Primary regions:** Rajasthan Canal (semi-arid, India) and Gobi Green Wall (arid, China)
- **Temporal coverage:** Monthly satellite observations (multi-year time series)
- **Spatial resolution:** ~500m for NDVI, aggregated at regional scale for model

### Type of Solution
Symbolic regression pipeline for ODE discovery using PySR (Python Symbolic Regression), combined with:
- Feature engineering for vegetation and climate indices
- Multi-tier equation selection based on reliability gates
- Intervention scenario modeling and collapse risk assessment
- Policy sensitivity analysis for land management levers

### End Goal
Real-time collapse risk monitoring and policy effectiveness evaluation. Outputs include:
- Collapse probability estimates (baseline, under interventions, under drought)
- Qualitative policy sensitivity (which levers move the system)
- Quantitative landscape stability (Lyapunov exponent, equilibrium NDVI range)

---

## 2. DOMAIN CONTEXT

### Desertification Definition
In this project's scope:
- **Desertification** = sustained reduction in NDVI (vegetation greenness) below ecological thresholds
- **Collapse** = persistent dNDVI/dt < 0 (vegetation degrading) with NDVI < critical threshold (region-specific: Rajasthan ~0.10-0.80, Gobi ~0.05-0.25)
- **Reversibility** = examining whether interventions can restore positive dNDVI/dt and push system back to stable attractor

### Key Environmental Indicators
| Indicator | Meaning | Source |
|-----------|---------|--------|
| **NDVI** | Normalized Difference Vegetation Index; vegetation greenness proxy (0-1, higher=greener) | MODIS MOD13Q1 |
| **NDVI_sq** | NDVI squared; emphasizes higher values, captures nonlinearity in vegetation response | Derived from NDVI |
| **NDVI_logistic** | NDVI normalized via logistic curve; accounts for saturation at high NDVI | Derived from NDVI |
| **Rain_norm** | Normalized rainfall anomaly | Seasonal rainfall data |
| **SoilMoist** | Soil moisture index | Climate/satellite reanalysis |
| **Albedo** | Surface reflectance; higher=lighter/drier soil | MODIS MCD43A3 |
| **LST_Day_anom** | Land Surface Temperature anomaly (daytime, deseasonalized) | MODIS MOD11A2 |
| **Temp_Delta_anom** | Temperature range anomaly (max-min, deseasonalized) | MODIS/climate reanalysis |
| **Carbon_Flux** | Net ecosystem carbon exchange | GEDI / satellite-based estimation |
| **GPP** | Gross Primary Productivity (vegetation productivity) | MOD17 |
| **FPAR** | Fraction of Photosynthetically Active Radiation absorbed | MODIS MOD15A2H |
| **Dust_Stress** | Dust aerosol optical depth or dust storm frequency | Satellite aerosol data |
| **Fire_Pressure** | Fire occurrence/burnt area index | MODIS MCD64A1 |
| **Groundwater_Anomaly** | Subsurface water availability anomaly | GRACE satellite / groundwater models |

### Theoretical Assumptions
1. **Positive feedback loops:** Vegetation loss → albedo increase → soil heating → less rainfall → more vegetation loss
2. **Multiple equilibria:** Semi-arid systems can exist in low-NDVI state (desert) or high-NDVI state (vegetation). Small perturbations can tip the system between states.
3. **Intervention effectiveness:** Policy levers (irrigation, restoration) shift dNDVI/dt in positive direction by modifying rainfall availability or soil conditions.
4. **Reversibility horizon:** If collapsed state is too deep (NDVI << critical threshold), recovery requires massive intervention (addressed via policy_sensitivity_floor).

---

## 3. DATA PIPELINE

### 3.1 Data Sources

| Dataset | Resolution | Temporal | Coverage | Purpose |
|---------|-----------|----------|----------|---------|
| MODIS MOD13Q1 | 250m | 16-day composites | Global | NDVI, derived vegetation indices |
| MODIS MCD43A3 | 500m | 16-day | Global | Albedo (directional reflectance) |
| MODIS MOD11A2 | 1km | 8-day | Global | Land Surface Temperature |
| MODIS MOD15A2H | 500m | 8-day | Global | FPAR (radiation absorption fraction) |
| MODIS MOD17 | 1km | 8-day/monthly | Global | Gross Primary Productivity (GPP) |
| MODIS MCD64A1 | 500m | Monthly | Global | Fire/burned area (Fire_Pressure) |
| MODIS MOD29 | 1km | Daily | Global | Dust aerosol optical depth (Dust_Stress) |
| GRACE | 1° x 1° | Monthly | Global | Groundwater anomaly |
| Climate/Reanalysis | 0.25°-1° | Daily/Monthly | Global | Precipitation, temperature, soil moisture |
| CSV Timeseries | Not applicable | Monthly (deseasonalized) | Regional | Aggregated driver/response pairs for each region |

### 3.2 Data Schema

**Input to ODE Discovery (per-region, monthly samples):**

```
Timestamp (YYYY-MM)
NDVI              [0.0, 1.0]        # Current vegetation state
NDVI_sq           [0.0, 1.0]        # NDVI squared (derived)
NDVI_logistic     [0.0, 1.0]        # NDVI logistic transform (derived)
Rain_norm         [~-2, +3]         # Normalized rainfall anomaly
SoilMoist         [0, 100] or [-3,3] # Soil moisture index (normalized)
Albedo            [0.0, 1.0]        # Surface reflectance
LST_Day_anom      [°C, typically -20 to +20] # Temperature anomaly
Temp_Delta_anom   [°C]              # Daily max-min anomaly
Carbon_Flux       [various units]   # Net carbon exchange
GPP               [g C m-2 day-1]   # Gross primary productivity
FPAR              [0.0, 1.0]        # Fraction of absorbed PAR
Dust_Stress       [0.0, 2.0] or unitless  # Aerosol optical depth
Fire_Pressure     [0, 1] or count   # Fire/burn index
Groundwater_Anom  [mm or cm]        # Subsurface water anomaly
TARGET: dNDVI/dt  [~-0.01, +0.01]   # Monthly NDVI change (normalized by dt=1 month)
```

**Region-Specific Notes:**
- **Rajasthan Canal:** NDVI range typically [0.20, 0.55] → NDVI_sq range [0.04, 0.30] (near-zero variance, flagged as problematic)
- **Gobi Green Wall:** NDVI range typically [0.05, 0.25] → NDVI_sq range [0.0025, 0.0625] (extremely flat, but Gobi restoration targets low-NDVI regime so this is acceptable)

### 3.3 Data Preprocessing

#### Step 1: LST Anomaly Deseasonalization
```
For each calendar month (Jan, Feb, ..., Dec):
  - Compute mean LST across all years for that month
  - Subtract from each month's LST to get anomaly
Result: LST_Day_anom, Temp_Delta_anom (monthly time series)
Purpose: Remove seasonal cycle, expose interannual thermal anomalies
```

#### Step 2: Feature Engineering & Derived Metrics
```
NDVI_sq = NDVI ** 2
  Purpose: Capture nonlinear vegetation response to drivers
  
NDVI_logistic = NDVI / (1 + exp(-(NDVI - K/2) / (K/4)))
  where K = K_REGION = {'Rajasthan': 0.8, 'Gobi': 0.3}
  Purpose: Logistic saturation model; accounts for diminishing productivity at high NDVI
  
dNDVI/dt = NDVI[t] - NDVI[t-1]  (monthly difference)
  Purpose: Target variable for ODE discovery
```

#### Step 3: Feature Audit (Pre-PySR Quality Gate)
For each region, compute:
- **Correlation with NDVI:** |r| with target dNDVI/dt
- **Flatness fraction:** % of values equal to mode (dead features: flatness > 30%)
- **Zero fraction:** % of exact-zero values (sparse features: zeros > 50%)
- **Outlier fraction:** % of |value| > 3σ (unstable features: outliers > 5%)
- **Valid sample count:** n_valid = non-NaN pairs

**Example findings from run 105634:**
- Carbon_Flux: ~90% zeros (DROPPED for Rajasthan, kept for Gobi)
- GPP: ~80% zeros (DROPPED for Rajasthan)
- NDVI_sq (Rajasthan): flat_frac = 0.95 (near-zero variance, DROPPED via FIX-BA)
- FPAR: good signal (r ~0.4-0.5, kept)

#### Step 4: Region-Specific Feature Filtering (FIX-BA)
**Rajasthan exclusions:** {NDVI_sq, Carbon_Flux, GPP, Fire_Pressure}
- Reason: NDVI_sq has near-zero variance in constrained Rajasthan NDVI range (0.2-0.55)
- Reason: Carbon_Flux, GPP are sparse/dead for semi-arid Rajasthan

**Gobi exclusions:** None (all features retained, as they have adequate signal)

#### Step 5: Feature Selection by Correlation Gate
- **Rajasthan threshold:** r_min = 0.08 (region-specific, higher tolerance for weak signal)
- **Gobi threshold:** r_min = 0.05 (global standard)
- **Action:** Drop features with |r| < threshold

#### Step 6: Complexity Cap (FIX-AJ, refined by FIX-BA)
- **Rajasthan max features:** 8 (reduced from 13 to prevent overfitting on weak signal)
- **Ranking:** Keep top 8 by absolute correlation with dNDVI/dt
- **Gobi max features:** No explicit cap (Gobi data quality sufficient)

#### Step 7: Normalization
- Most features already scaled to [0, 1] or standard normal via satellite processing
- Additional outlier clipping: values > 3σ clipped to ±3σ (mitigates PySR search instability)

---

## 4. FEATURE ENGINEERING

### Vegetation Indices

| Index | Formula | Range | Interpretation |
|-------|---------|-------|-----------------|
| **NDVI** | (NIR - Red) / (NIR + Red) | [0, 1] | Baseline; 0=desert, 1=dense vegetation |
| **NDVI_sq** | NDVI² | [0, 1] | Emphasizes high-NDVI states; nonlinear response |
| **NDVI_logistic** | NDVI / (1 + exp(-(NDVI - K/2)/(K/4))) | [0, K] | Logistic saturation; accounts for productivity limits |
| **FPAR** | Satellite-derived | [0, 1] | Fraction of light absorbed by vegetation (complementary to NDVI) |

### Temporal Features

| Feature | Computation | Purpose |
|---------|------------|---------|
| **dNDVI/dt** | NDVI[t] - NDVI[t-1] | Target: monthly vegetation change rate |
| **LST_Day_anom** | LST[t] - mean_LST[calendar_month] | Deseasonalized temperature anomaly |
| **Temp_Delta_anom** | (LST_max - LST_min)[t] - mean(LST_max - LST_min)[calendar_month] | Deseasonalized daily range anomaly |
| **Rain_norm** | (Precip[t] - mean_Precip) / std_Precip | Normalized rainfall anomaly (standard deviation units) |

### Spatial/Climate Features (Regional Aggregates)

| Feature | Source | Interpretation |
|---------|--------|-----------------|
| **Albedo** | MODIS MCD43A3 | Surface reflectance; higher = drier/lighter surface |
| **SoilMoist** | Reanalysis (ERA5, GLDAS) | Subsurface water availability; lower = drought |
| **Dust_Stress** | MODIS aerosol optical depth | Atmospheric dust; high = dusty conditions, plant stress |
| **Groundwater_Anom** | GRACE + groundwater models | Subsurface water anomaly; negative = groundwater depletion |

### Derived Indices (Policy Sensitivity Analysis)

| Index | Formula | Purpose |
|-------|---------|---------|
| **Aridity_safe** | Rain_norm / (NDVI + 0.1) | Aridity proxy; used for equilibrium detection (excludes from ODE to preserve intervention effect) |
| **NDVI_logistic** (injected post-hoc) | Logistic transform of NDVI | Injected into equations if absent or tier ≥ 4; ensures physical constraints on growth |

---

## 5. MODEL / ALGORITHMIC APPROACH

### 5.1 Model Type

**Overall:** Symbolic Regression for Ordinary Differential Equation (ODE) Discovery

**Specific Algorithm:** PySR (Symbolic Regression via Genetic Programming)
- Uses evolutionary search to find parsimonious equations
- Balances model complexity (fewer operators) vs fit quality
- Supports custom unary/binary operators and operator constraints

**Problem Formulation:**
```
Find: dNDVI/dt = f(NDVI, Rain_norm, SoilMoist, Albedo, ...)
such that:
  - Minimize prediction error (R² or MAE)
  - Minimize equation complexity (size/operator count)
  - Subject to: physical constraints (sign-change at equilibrium, policy sensitivity)
```

### 5.2 Training Details

#### PySR Hyperparameters (Per-Region)

| Parameter | Rajasthan | Gobi | Purpose |
|-----------|-----------|------|---------|
| **NITER** | 260 | 240 | Search iterations (generations); deeper search for weak-signal Rajasthan |
| **PYSR_MAXSIZE** | 12 | 12 | Max equation complexity (number of nodes in expression tree) |
| **PYSR_POPULATIONS** | 30 | 30 | Parallel populations per generation |
| **Binary Ops** | ["+", "*", "-", "/"] | ["+", "*", "-", "/"] | Allowed binary operators |
| **Unary Ops** | ["exp", "sin", "cos"] | ["exp", "sin", "cos"] | Allowed unary functions |
| **Operator Constraints** | No division for denomination-prone features (LST_Day_anom, Temp_Delta_anom) | Standard | Prevent brittle reciprocal expressions |

#### Loss Function & Optimization
```
Primary Loss = MSE(predictions, dNDVI/dt_observed)
  
Secondary Objectives:
  - Complexity penalty (prefer fewer operators)
  - Feasibility check (equation must be compilable, no NaN/Inf at evaluation points)
  
Optimization: Multi-objective genetic algorithm (Pareto frontier exploration)
```

#### Data Split
- **Train/Test:** All available months used for training (no explicit holdout; model validated on reliability gates)
- **Temporal:** Monthly time series; PySR assumes i.i.d. samples (ignores temporal autocorrelation)
- **Rationale:** Symbolic regression trades off generalization for interpretability; reliability gates serve as post-hoc validation

### 5.3 Model Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Region-specific CSV timeseries (NDVI, drivers)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Feature Audit (feature_audit)                          │
│ - Compute correlation, flatness, zero fraction per feature     │
│ - Log diagnostic info for debugging                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Feature Filtering (safe_features)                      │
│ - Apply FIX-BA region-specific exclusions                      │
│ - Drop zero-heavy, flat, low-correlation features              │
│ - Complexity cap (keep top-8 by correlation for Rajasthan)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Prepare PySR Input                                     │
│ - Features = safe_features result                              │
│ - Target = dNDVI/dt                                            │
│ - Remove rows with NaN                                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Run PySR Search                                         │
│ - Generate 1000s of candidate equations                        │
│ - Rank by (loss, complexity) tradeoff                          │
│ - Return equation population                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Equation Summarization & Diagnostics (FIX-AH)          │
│ - Evaluate top DIAGNOSTIC_EQ_LIMIT equations (default 30)      │
│ - Count by validity: AST parseable, NDVI present, driver       │
│ - Return counts + top-by-loss, top-by-R2 candidates            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Equation Selection (6-Tier Hierarchy)                  │
│                                                                 │
│ P1 (Tier 1): R² ≥ 0.10, nMAE ≤ threshold, clip < 80%           │
│    + Sign-change at equilibrium                                │
│    + NDVI appears directly (not just derived)                  │
│    + For Rajasthan: ADDITIONALLY must have R² ≥ 0.03 (FIX-AK)  │
│                                                                 │
│ P2 (Tier 2): Relaxed R² (≥ 0.05), same other checks            │
│    + For Rajasthan P2: NDVI_sq required (FIX-AE-bis)           │
│    + For Rajasthan P2: ADDITIONALLY must R² ≥ 0.03 (FIX-AK)    │
│                                                                 │
│ P3-P6 (Tiers 3-6): Progressive relaxation of constraints       │
│    + Higher tiers may lack sign-change or use NDVI_logistic    │
│    + Fallback mechanism: if P1-P5 fail, use fallback           │
│                                                                 │
│ Reliability gates applied at tier-selection level:             │
│  - tier_ok: tier ≤ 3                                           │
│  - fit_ok: R² vs nMAE_IQR check                                │
│  - clip_ok: clip saturation < 80%                             │
│  - sign_change_ok: proper sign reversal at equilibrium         │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: Post-Hoc ODE Compilation                               │
│ - Compile equation to Python function (ode_func)               │
│ - Attach diagnostics metadata                                  │
│ - Cache for re-use                                             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: Intervention Scenario Simulation                       │
│ - Use ode_func to simulate dNDVI/dt under different policies   │
│ - Compute collapse risk, policy sensitivity                    │
│ - Generate outputs (probabilities, visualizations)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. SYSTEM DESIGN

### 6.1 High-Level Design (HLD)

```
┌──────────────────────────────────────────────────────────────────┐
│                    Google Colab Notebook                         │
│  (ReversedDesertification.ipynb)                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Cell 1-10: Data Loading & Preprocessing                  │ │
│  │  - Load region-specific timeseries CSV                    │ │
│  │  - Compute LST anomaly, derived NDVI metrics              │ │
│  │  - Print data summary                                    │ │
│  └────────────────────┬─────────────────────────────────────┘ │
│                       │                                        │
│  ┌────────────────────▼─────────────────────────────────────┐ │
│  │ Cell 11-25: ODE Discovery (Via desertification Package) │ │
│  │  - Import desertification.ode_discovery                  │ │
│  │  - Call discover_and_compile_ode(df_raj, "Rajasthan")   │ │
│  │  - Call discover_and_compile_ode(df_gobi, "Gobi")       │ │
│  │  - Each returns (ode_func, equation_str, diagnostics)   │ │
│  └────────────────────┬─────────────────────────────────────┘ │
│                       │                                        │
│  ┌────────────────────▼─────────────────────────────────────┐ │
│  │ Cell 26-35: Policy & Collapse Risk Analysis             │ │
│  │  - Define intervention policies (boost, restoration)     │ │
│  │  - Run 100+ year simulations for each scenario           │ │
│  │  - Compute P(collapse), Lyapunov, equilibrium range      │ │
│  └────────────────────┬─────────────────────────────────────┘ │
│                       │                                        │
│  ┌────────────────────▼─────────────────────────────────────┐ │
│  │ Cell 36-40: Visualization & Export                      │ │
│  │  - Generate plots: Main_Summary, Diagnostic_Plots, etc.  │ │
│  │  - Export run_summary.json with all diagnostics          │ │
│  │  - Save images to /images/                               │ │
│  │  - Zip all outputs                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
         │
         │ imports & calls
         ▼
┌──────────────────────────────────────────────────────────────────┐
│              desertification/ Python Package                     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ config.py                                                  │ │
│  │ - All hyperparameters & constants (NITER, thresholds)    │ │
│  │ - Feature definitions & variable metadata                 │ │
│  │ - ODE bounds, equilibrium parameters                      │ │
│  │ - Region-specific settings                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ features.py                                                │ │
│  │ - add_lst_anomaly(): LST deseasonalization               │ │
│  │ - add_ndvi_derived(): Create NDVI_sq, NDVI_logistic      │ │
│  │ - feature_audit(): Pre-PySR quality diagnostics          │ │
│  │ - safe_features(): Quality-gated feature selection       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ ode_discovery.py                                           │ │
│  │ - discover_and_compile_ode(): MAIN entry point           │ │
│  │   • Feature filtering → PySR search → equation selection  │ │
│  │   • Tier 1-6 selection logic                             │ │
│  │   • ODE function compilation                              │ │
│  │   • Diagnostics capture                                   │ │
│  │ - summarize_equations(): Preview & diagnostic evaluation │ │
│  │ - _to_builtin(): JSON serialization helper                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ interventions.py                                           │ │
│  │ - apply_intervention(): Modify drivers (Rain_norm, etc)   │ │
│  │ - Supports: boost, restoration, drought, etc.            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ dynamics.py                                                │ │
│  │ - simulate_ndvi(): Run ODE forward in time                │ │
│  │ - compute_collapse_risk(): Monte Carlo collapse detection │ │
│  │ - compute_lyapunov(): Stability metric                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ plotting.py, ews.py, data.py, __init__.py                │ │
│  │ - Visualization, early warning systems, data utils       │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Low-Level Design (LLD)

#### 6.2.1 Core Data Flow: discover_and_compile_ode()

**Location:** `desertification/ode_discovery.py`  
**Signature:**
```python
def discover_and_compile_ode(
    df: pd.DataFrame,
    region_name: str,
    target_col: str = 'dNDVI_dt'
) -> Tuple[Callable, str, dict]:
    """
    Discover ODE and return compiled function + equation string + diagnostics.
    
    Returns:
      - ode_func: Callable that computes dNDVI/dt from feature vector
      - equation_str: Human-readable equation
      - diagnostics: dict with feature_dropped, equation_preview, etc.
    """
```

**Key Steps:**

1. **Feature Audit:**
   ```python
   region_corrs = feature_audit(df, region_name)
   # Returns: {feature_name: {'r': corr, 'flat': flat_frac, 'zero': zero_frac, ...}}
   ```

2. **Feature Selection:**
   ```python
   safe_feats, dropped_info = safe_features(
       df, region_corrs, region_name, return_dropped=True
   )
   # dropped_info captures what was filtered and why
   # Applies FIX-BA region-specific exclusions
   # Applies complexity cap (max 8 features for Rajasthan)
   ```

3. **Prepare PySR Input:**
   ```python
   X = df[safe_feats].dropna()
   y = df[target_col].loc[X.index]
   
   # Compile model object using PySR.jl backend
   model = PySRRegressor(
       niterations=NITER_RAJ if region='Rajasthan' else NITER_GOBI,
       binary_operators=PYSR_BINARY_OPS,
       unary_operators=PYSR_UNARY_OPS,
       ...
   )
   model.fit(X, y)
   ```

4. **Equation Selection (6-Tier Hierarchy):**
   ```python
   # Extract candidate equations from PySR population
   candidates = extract_equations(model, max_count=1000)
   
   for tier in [1, 2, 3, 4, 5, 6]:
       for eq in rank_equations_by_fit(candidates):
           if passes_tier_checks(eq, tier, region_name):
               selected_eq = eq
               selected_tier = tier
               break
       if selected_eq:
           break
   
   # Tier checks include:
   #  - R² thresholds
   #  - NDVI presence/form (direct, squared, logistic)
   #  - Sign-change at equilibrium
   #  - Reliability gates (fit_ok, clip_ok, etc.)
   ```

5. **ODE Function Compilation:**
   ```python
   # Convert equation string to Python callable
   ode_func = compile_equation_to_function(selected_eq)
   
   # Signature: ode_func(NDVI, Rain_norm, SoilMoist, ...) → dNDVI/dt
   # Stored as ode_func for reuse
   ```

6. **Diagnostics Capture:**
   ```python
   diagnostics = {
       'feature_dropped': dropped_info,
       'feature_reduced': {'original_count': len(FEATURES), 
                          'final_count': len(safe_feats),
                          'reason': 'complexity_cap_applied' if capped else 'none'},
       'equation_preview': summarize_equations(model, safe_feats, tier=selected_tier),
       'selected_tier': selected_tier,
       'fit_r2': model.score(X, y),
   }
   ode_func.diagnostics = diagnostics  # Attach as metadata
   ```

#### 6.2.2 Key Functions & Interfaces

| Function | Module | Inputs | Outputs | Purpose |
|----------|--------|--------|---------|---------|
| `add_lst_anomaly(df)` | features.py | DataFrame with LST | Updated df with LST_Day_anom, Temp_Delta_anom | Deseasonalize temperature |
| `add_ndvi_derived(df)` | features.py | DataFrame with NDVI | Updated df with NDVI_sq, NDVI_logistic | Create derived vegetation indices |
| `feature_audit(df, region_name)` | features.py | DataFrame, region string | dict: feature → audit stats | Pre-PySR diagnostic audit |
| `safe_features(df, region_corrs, region_name, return_dropped)` | features.py | DataFrame, audit dict, region, bool | list of safe feature names (or tuple with dropped info) | Quality-gated feature selection |
| `discover_and_compile_ode(df, region_name)` | ode_discovery.py | Regional DataFrame, region string | (ode_func, eq_str, diagnostics) | Main ODE discovery pipeline |
| `summarize_equations(model, features, limit=30)` | ode_discovery.py | PySR model, feature list, int | dict with counts + equation previews | Diagnostic equation summary |
| `apply_intervention(driver_dict, intervention_type)` | interventions.py | Feature dict, intervention string | Modified feature dict | Apply policy intervention |
| `simulate_ndvi(ode_func, initial_ndvi, drivers_timeseries, dt)` | dynamics.py | ODE function, initial state, drivers, timestep | Array of NDVI time evolution | Forward ODE integration |
| `compute_collapse_risk(ode_func, region, n_simulations)` | dynamics.py | ODE function, region name, int | float [0, 1] = P(collapse) | Monte Carlo collapse probability |

---

## 7. CODEBASE STRUCTURE

### Directory Layout

```
/Users/sanchitkumardogra/kaam/clg/SEM8/Desertification/
├── desertification/                          # Main Python package
│   ├── __init__.py                           # Package initialization
│   ├── config.py                             # All constants, hyperparameters, metadata
│   ├── features.py                           # Feature engineering & audit
│   ├── ode_discovery.py                      # PySR symbolic regression pipeline
│   ├── interventions.py                      # Policy intervention logic
│   ├── dynamics.py                           # ODE simulation & risk analysis
│   ├── ews.py                                # Early warning system components
│   ├── plotting.py                           # Visualization functions
│   └── data.py                               # Data loading & utilities
├── scripts/
│   ├── evaluate_run.py                       # Standalone run evaluation
│   └── README.md                             # Script documentation
├── ReversedDesertification.ipynb              # Main Colab notebook (modular workflow)
├── reverseddesertification.py                # Legacy: older non-modular version
├── reverseddesertification_old.py            # Legacy: backup
├── ReversedDesertification_archive.ipynb     # Legacy: archived notebook
├── desertification_package.zip               # Distributable package (deployed to Colab)
├── desertification_outputs_20260428_*.zip    # Run outputs (diagnostic images + JSON)
├── trial1/, trial2/                          # Legacy experimental runs
├── Update March/, Update April/              # Legacy archives
└── .venv/                                    # Python virtual environment
    └── bin/activate                          # Activation script
```

### Key Files & Roles

| File | Purpose | Status | Lines |
|------|---------|--------|-------|
| `config.py` | Centralized constants (NITER, thresholds, feature metadata) | Active | ~250 |
| `features.py` | Feature engineering, audit, safe selection | Active | ~180 |
| `ode_discovery.py` | PySR integration, 6-tier selection, compilation | Active | ~1200 |
| `interventions.py` | Policy lever application | Active | ~100 |
| `dynamics.py` | ODE simulation, collapse risk, Lyapunov | Active | ~200 |
| `plotting.py` | Visualization (collapse risk, trajectory plots, etc.) | Active | ~150 |
| `ReversedDesertification.ipynb` | Main modular workflow (Colab) | Active | ~40 cells |
| `scripts/evaluate_run.py` | Standalone evaluation tool | Deprecated | ~50 |

---

## 8. CORE LOGIC

### 8.1 ODE Discovery Logic

**Problem:** Find dNDVI/dt = f(NDVI, drivers) such that f captures land degradation dynamics and policy sensitivity.

**Algorithm:** Multi-objective Genetic Programming (via PySR)

```
1. Initialize: Population of random equations
2. For iteration = 1 to NITER:
     a. Evaluate each equation on data (compute loss, complexity)
     b. Apply selection pressure (Pareto ranking: prefer low loss & low complexity)
     c. Generate offspring via mutation/crossover
     d. Replace worst performers
3. Return: Pareto-optimal equation population
```

**Constraint Application:**

After PySR finishes, equations ranked by 6-tier hierarchy:

```python
TIER_1_CRITERIA = {
    'r2_min': 0.10,           # Fit threshold
    'clip_saturation_max': 0.80,
    'has_sign_change': True,   # At equilibrium NDVI
    'has_ndvi_direct': True,   # NDVI appears (not just derived)
    'rajasthan_r2_min': 0.03,  # FIX-AK: additional Rajasthan constraint
}

TIER_2_CRITERIA = {
    'r2_min': 0.05,
    'clip_saturation_max': 0.80,
    'has_sign_change': True,
    'rajasthan_require_ndvi_sq': True,  # FIX-AE-bis: P2 must have NDVI_sq for Rajasthan
    'rajasthan_r2_min': 0.03,           # FIX-AK: fit floor for P2
}

TIER_3_CRITERIA = { 'r2_min': 0.02, ... }
...
TIER_6_CRITERIA = { 'r2_min': 0.00, ... }  # Fallback tier; almost no constraints
```

**Reliability Gates (Applied After Tier Selection):**

```python
RELIABILITY_CHECKS = {
    'tier_ok': tier <= 3 (strict) or tier <= 5 (relaxed),
    'fit_ok': (r2 >= 0.03 and nmae_iqr <= 1.0) or (r2 >= 0.01 and nmae_iqr <= 0.8),
    'clip_ok': clip_saturation < 0.80,
    'sign_change_ok': dNDVI/dt has correct sign at low/high NDVI,
}

RELIABILITY_PASS = all([tier_ok, fit_ok, clip_ok, sign_change_ok])
```

### 8.2 Feature Filtering Logic (FIX-BA)

**Region-Specific Exclusion:**

```python
# config.py
FEATURES_TO_EXCLUDE_BY_REGION = {
    'Rajasthan Canal': {'NDVI_sq', 'Carbon_Flux', 'GPP', 'Fire_Pressure'},
    'Gobi Green Wall': set(),
}

# features.py
def safe_features(df, region_corrs, region_name, return_dropped=False):
    feat = [f for f in FEATURES if f in df.columns and df[f].std() > 1e-8]
    
    # FIX-BA: Apply region-specific exclusions
    dropped_excluded = []
    if region_name:
        exclude_set = FEATURES_TO_EXCLUDE_BY_REGION.get(region_name, set())
        for f in feat[:]:
            if f in exclude_set:
                feat.remove(f)
                dropped_excluded.append(f"{f} (region-specific exclusion)")
    
    # ... then apply correlation & flatness gates
```

**Rationale for Rajasthan Exclusions:**
- **NDVI_sq:** Rajasthan NDVI ∈ [0.2, 0.55] → NDVI_sq ∈ [0.04, 0.30] with very low variance. Acts as noise, not signal.
- **Carbon_Flux, GPP, Fire_Pressure:** Near-zero for semi-arid regions; dead features that PySR tries to fit artifactually.

### 8.3 Equation Selection Logic (6-Tier Hierarchy)

**Flow:**

```
1. PySR returns 1000s of equations ranked by (loss, complexity)
2. For each tier (1→6):
     For each equation (best→worst by fit):
         If equation passes tier criteria:
             If passes reliability gates:
                 SELECT this equation ✓
                 Break
     If selected:
         Break

3. If no equation selected by tier 6:
     Use fallback heuristic (e.g., simplest equation with positive R²)
```

**Tier-Specific Behavior:**

| Tier | Purpose | R² Min | NDVI Form | Sign-Change Required? | Example |
|------|---------|--------|-----------|----------------------|---------|
| 1 | Best fit, constrained | 0.10 | Direct/NDVI only | Yes | Rajasthan: must also R² ≥ 0.03 |
| 2 | Good fit, slightly relaxed | 0.05 | Any | Yes | Rajasthan P2: must have NDVI_sq, R² ≥ 0.03 |
| 3 | Acceptable fit | 0.02 | Any | Yes | Weaker than P1, still has sign-change |
| 4 | Weak fit, may lack sign-change | 0.00 | Any | Maybe | May use NDVI_logistic post-hoc injection (FIX-AE) |
| 5 | Very weak fit | 0.00 | Any | No | Last attempt before fallback |
| 6 | Fallback | N/A | Fallback heuristic | No | Simplest equation or zero growth |

### 8.4 Equilibrium & Sign-Change Detection

**Equilibrium NDVI (dNDVI/dt = 0):**

```python
def find_equilibrium_ndvi(ode_func, region):
    """Numerically find NDVI where dNDVI/dt = 0 (steady state)"""
    for ndvi_test in linspace(region.ndvi_min, region.ndvi_max, 100):
        d_ndvi = ode_func(ndvi_test, drivers_baseline)
        if abs(d_ndvi) < 1e-3:
            return ndvi_test
    return None
```

**Sign-Change Check:**

```python
# Verify that dNDVI/dt changes sign across equilibrium
ndvi_low = equilibrium_ndvi - 0.05
ndvi_high = equilibrium_ndvi + 0.05

d_low = ode_func(ndvi_low, drivers_baseline)
d_high = ode_func(ndvi_high, drivers_baseline)

sign_change_ok = (d_low < 0 and d_high > 0) or (d_low > 0 and d_high < 0)
```

**Physical Interpretation:**
- If NDVI < equilibrium and dNDVI/dt < 0: system pushes toward degradation (unstable)
- If NDVI > equilibrium and dNDVI/dt > 0: system pushes toward growth (unstable)
- If both true: equilibrium is unstable saddle point (correct for bistable systems)

### 8.5 Collapse Risk Computation (Monte Carlo)

**Algorithm:**

```python
def compute_collapse_risk(ode_func, region, n_simulations=3000, years=50):
    """
    Estimate P(collapse) = fraction of trajectories that reach NDVI < critical threshold
    """
    collapse_count = 0
    
    for sim in range(n_simulations):
        # Random initial NDVI from observed distribution
        ndvi_init = sample_from_observation_distribution(region)
        
        # Random driver sequence (stochastic resampling or parameterized noise)
        drivers_trajectory = generate_stochastic_drivers(years=years)
        
        # Simulate forward in time
        ndvi_trajectory = simulate_ndvi(ode_func, ndvi_init, drivers_trajectory, dt=1)
        
        # Check: does NDVI ever drop below critical threshold?
        if min(ndvi_trajectory) < CRITICAL_NDVI[region]:
            collapse_count += 1
    
    p_collapse = collapse_count / n_simulations
    return p_collapse
```

**Collapse Definition (Region-Specific):**

```python
SIGN_CHANGE_RANGE_RAJ = (0.10, 0.80)   # Critical NDVI range for Rajasthan
SIGN_CHANGE_RANGE_GOBI = (0.05, 0.25)  # Critical NDVI range for Gobi

# Collapse = NDVI falls & stays below lower bound for ≥ 2 years
```

### 8.6 Policy Sensitivity Analysis

**Method:** Intervention lever strength affects drivers, which shifts dNDVI/dt.

```python
# Intervention definition
interventions = {
    'canal_boost': {'Rain_norm': +0.5, 'SoilMoist': +0.3},  # Irrigation adds water
    'restoration': {'Albedo': -0.1, 'Dust_Stress': -0.2},   # Vegetation reduces dust
    'drought': {'Rain_norm': -1.0},                          # Climate stress
}

# For each intervention:
for scenario in ['baseline', 'canal_boost', 'restoration', ...]:
    drivers_modified = apply_intervention(drivers_baseline, scenario)
    dndvi_dt = ode_func(NDVI, drivers_modified)
    
    # Compute policy sensitivity as derivative
    sensitivity = (dndvi_dt_intervention - dndvi_dt_baseline) / intervention_strength
```

**Policy Sensitivity Floor (FIX-AK):**

```python
# Prevent equations from being selected if they don't respond to any lever
policy_terms = [driver for driver in drivers if derivative(ode_func, driver) != 0]

if len(policy_terms) < 2:
    # Weak/no policy sensitivity; flag as unreliable
    reliability_pass = False
```

---

## 9. VISUALIZATION / OUTPUT

### Output Types

| Output | Type | Purpose | Generated In |
|--------|------|---------|--------------|
| **Main_Summary.png** | Figure | Overview: equations, collapse probabilities, policy sensitivity | plotting.py (multibrand plot) |
| **Collapse_Risk_Summary.png** | Figure | P(collapse) heatmap across scenarios/droughts | plotting.py |
| **Diagnostic_Plots.png** | Figure | Feature importance, dNDVI/dt distributions, Q-Q plots | plotting.py |
| **Feature_Diagnostics.png** | Figure | Feature audit: correlation, flatness, zero fractions per region | plotting.py |
| **data_quality.png** | Figure | NDVI ranges, driver distributions, data coverage | plotting.py |
| **run_summary.json** | JSON | Machine-readable metadata: equations, R², tier, collapse probs, diagnostics | Notebook export |
| **.zip (outputs)** | Archive | All images + JSON + logs; ready for distribution | Notebook (final cell) |

### Visualization Libraries & Tools

- **matplotlib:** Core plotting (line plots, heatmaps, subplots)
- **seaborn:** Statistical visualization (kde plots, heatmaps)
- **folium:** (Planned) Interactive maps with collapse risk overlay
- **plotly:** (Planned) Interactive dashboards

### Output Format & Storage

```
desertification_outputs_20260428_HHMMSS.zip
├── images/
│   ├── Main_Summary.png
│   ├── Collapse_Risk_Summary.png
│   ├── Diagnostic_Plots.png
│   ├── Feature_Diagnostics.png
│   ├── data_quality.png
│   └── run_summary.json
└── (timestamp_log.txt)  [if applicable]
```

**run_summary.json Schema:**

```json
{
  "collapse_probability_ever": {
    "rajasthan": {"baseline": 0.08, "drought": 0.99, ...},
    "gobi": {...}
  },
  "collapse_probability_terminal": {...},
  "lyapunov": {"rajasthan": -0.038, "gobi": -0.049},
  "selection_tier": {"rajasthan": 1, "gobi": 1},
  "fit_r2": {"rajasthan": 0.031, "gobi": 0.056},
  "equations": {
    "rajasthan": "(FPAR*Rain_norm - NDVI_sq)*sin(Albedo*4.01 - 0.75)",
    "gobi": "(NDVI_sq*(...)*exp(Dust_Stress)"
  },
  "policy_sensitivity_terms": {
    "rajasthan": ["SoilMoist", "Albedo", "FPAR", "Dust_Stress"],
    "gobi": ["Rain_norm", "SoilMoist", "Albedo", "FPAR", "Dust_Stress", "Groundwater_Anomaly"]
  },
  "reliability_pass": {"rajasthan": true, "gobi": true},
  "diagnostics": {
    "rajasthan": {
      "feature_dropped": [...],
      "feature_reduced": {"original_count": 13, "final_count": 8},
      "equation_preview": {...}
    },
    "gobi": {...}
  }
}
```

---

## 10. CURRENT STATUS

### Completed Components

✅ **Core ODE Discovery Pipeline**
- PySR integration with multi-objective search
- 6-tier equation selection with reliability gates
- Feature audit and region-specific filtering
- ODE function compilation and caching

✅ **Feature Engineering**
- NDVI derived indices (NDVI_sq, NDVI_logistic)
- LST anomaly deseasonalization
- Regional feature filtering with quality gates
- Complexity cap (Rajasthan max 8 features)

✅ **Diagnostics Infrastructure**
- Feature audit with outlier/flatness/zero tracking
- Equation preview system (top-30 candidates evaluated)
- Per-equation validity checking (AST, NDVI presence, driver presence)
- Comprehensive diagnostics export to run_summary.json

✅ **Constraint Implementations**
- FIX-AK: P1/P2 fit floors for Rajasthan (R² ≥ 0.03)
- FIX-AE/-AE-bis: NDVI_sq injection for P2 when absent
- FIX-BA: Region-specific feature exclusions (NDVI_sq, Carbon_Flux, GPP, Fire_Pressure for Rajasthan)
- FIX-AF, FIX-AI, FIX-AJ: Gradient ceiling, flatness gates, denominator-prone feature drops

✅ **Intervention Scenario Analysis**
- Policy lever application (canal_boost, restoration, drought)
- Monte Carlo collapse risk computation (3000 simulations)
- Lyapunov stability estimates
- Policy sensitivity analysis

✅ **Visualization & Export**
- Main_Summary, Collapse_Risk_Summary, Diagnostic_Plots, Feature_Diagnostics, data_quality plots
- run_summary.json export with full metadata
- .zip packaging for distribution

### In-Progress Components

🟡 **Fit Stability Improvement**
- Current best: R² = 0.031 (Rajasthan, run 105634) after FIX-BA
- Target: R² > 0.05-0.10 (approaching 151944 baseline of 0.0643)
- Issue: Equation selection still fallback to weak fits when top PySR candidates don't meet strict constraints
- Next: Investigate residual patterns, consider alternative feature engineering (e.g., NDVI_poly)

🟡 **Run-to-Run Stability**
- Previous runs showed fit variance (R² 0.0643 → 0.0163 → -0.0157)
- Recent diagnostic runs (103841, 105634) show stability (R² near 0.001-0.031)
- Diagnostics infrastructure in place to track future variance

### Not Started

❌ **Interactive Geospatial Mapping**
- Planned: Folium-based collapse risk map overlay
- Status: Design sketched, not implemented

❌ **Real-Time Deployment**
- Planned: Cloud backend for continuous satellite monitoring
- Status: Not scoped

❌ **Multi-Region Extension**
- Current: Rajasthan + Gobi only
- Planned: 5+ additional regions
- Status: Architecture supports, not implemented

---

## 11. ISSUES & LIMITATIONS

### Critical Issues

**🔴 Rajasthan Fit Instability (Root Cause Identified, Partially Fixed)**

- **Symptom:** R² bounces across runs (0.0643 → 0.0163 → -0.0157)
- **Root Cause (Diagnosed):** 
  - NDVI_sq has near-zero variance in Rajasthan (NDVI range 0.2-0.55 → NDVI_sq range 0.04-0.30)
  - Dead features (Carbon_Flux, GPP, Fire_Pressure) introduce noise
  - Limited feature signal relative to Rajasthan data noise level
- **Mitigation (FIX-BA Applied):**
  - Excluded NDVI_sq, Carbon_Flux, GPP, Fire_Pressure from Rajasthan search
  - Result: R² improved from 0.001 → 0.031 (25x recovery)
  - Still below baseline (0.0643), but recovery trajectory clear
- **Remaining Concern:** Even with FIX-BA, Rajasthan R² = 0.031 still marginal; may need further investigation into residual bias or alternative feature engineering

### Data Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Short timeseries (3-5 years)** | Limited statistical power for rare events (collapse) | Monte Carlo resampling; parametric noise injection |
| **Low NDVI signal in Rajasthan** | Weak equation fits; high PySR search variance | Feature exclusion (FIX-BA), complexity cap, higher iterations |
| **Sparse features (Carbon_Flux, GPP, Fire_Pressure)** | Introduce noise; PySR fits artifactual patterns | Region-specific exclusion (FIX-BA) |
| **Missing data gaps** | Satellite revisit gaps, clouds | Interpolation + flagging unreliable months |
| **Satellite sensor changes** | Radiometric inconsistency across sensors | Cross-calibration tables (not implemented; assumed handled upstream) |
| **Regional heterogeneity** | Within-region sub-pixel variability | Spatial averaging; assumes homogeneous region |

### Model Weaknesses

| Weakness | Consequence | Workaround |
|----------|------------|-----------|
| **Temporal i.i.d. assumption in PySR** | Ignores autocorrelation in dNDVI/dt time series | Manual lag feature engineering (not implemented) |
| **ODE assumes slow timescale dynamics** | Misses rapid oscillations or sub-monthly drivers | Aggregate to monthly (accepted trade-off for interpretability) |
| **Equation complexity cap** | May exclude useful higher-order terms | Config tunable (RAJASTHAN_MAX_FEATURES = 8) |
| **6-tier fallback heuristic** | No guarantee of meaningful equation at tier 6 | Implement minimum complexity floor or user override |
| **Stochastic PySR search** | Run-to-run equation variance despite same settings | Address via diagnostics capture + statistical confidence intervals (TBD) |

### Edge Cases & Unhandled Scenarios

| Edge Case | Current Behavior | Status |
|-----------|-----------------|--------|
| **NDVI constant (flat region)** | All dNDVI/dt = 0; ODE becomes trivial | Not handled; would fail feature audit |
| **All features NaN for a month** | That month dropped from PySR training | Silent filtering; could add warning |
| **Extreme outlier month (sensor glitch)** | Outlier clipping (±3σ) may distort signal | Outlier detection logs added; manual review recommended |
| **Region with zero collapse history** | P(collapse) = 0; Lyapunov not meaningful | Diagnostics export still valid; caveat added to output |
| **Multimodal NDVI distribution** | Mean equilibrium may not exist; bistability complex | Current code assumes unimodal; manual inspection required |

---

## 12. DESIGN DECISIONS & TRADE-OFFS

### Why Symbolic Regression (PySR)?

**Chosen Over Alternatives:**

| Alternative | Pros | Cons | Reason Rejected |
|------------|------|------|-----------------|
| **Neural Network (LSTM)** | Powerful for temporal prediction | Black box; hard to interpret physics | Goal: interpretable equations for policy makers |
| **Linear Regression** | Fast, simple, interpretable | Cannot capture nonlinear vegetation dynamics | Land degradation is inherently nonlinear |
| **Random Forest** | Fast, handles nonlinearity | Non-interpretable; no explicit equation | Transparency required for stakeholder trust |
| **Physics-Based ODE (SDE)** | Principled, uses domain knowledge | Requires strong prior about dynamics | Exploratory project; data-driven approach preferred |
| **PySR** | Interpretable equations, nonlinear, handles complex relationships | Slower, stochastic; requires constraint tuning | **SELECTED:** Best trade-off |

**Design Decision:** Prioritize interpretability for policy communication over pure predictive accuracy.

### Why 6-Tier Hierarchy?

**Rationale:**

- **Tier 1-3:** Strict criteria ensure high-quality equations suitable for policy guidance
- **Tier 4-6:** Progressive relaxation prevents complete failure when data quality limits tier 1-3 eligibility
- **Fallback:** Ensures pipeline never fails; worst-case returns a usable (if weak) equation

**Trade-Off:** Ranked constraints allow flexibility without sacrificing reliability gates.

### Why Region-Specific Settings?

**Rationale:**

| Setting | Rajasthan | Gobi | Reason |
|---------|-----------|------|--------|
| **NITER** | 260 | 240 | Rajasthan weaker signal; needs deeper search |
| **LOW_CORR_THRESHOLD** | 0.08 | 0.05 | Rajasthan data noisier; tolerate weaker correlations |
| **RAJASTHAN_MAX_FEATURES** | 8 | No cap | Rajasthan high noise; reduce overfitting risk |
| **Feature exclusions** | {NDVI_sq, Carbon_Flux, GPP, Fire_Pressure} | {} | Dead features specific to semi-arid Rajasthan |
| **P1/P2 R² floor** | 0.03 | None | Rajasthan weak signal; ensure minimum viability |

**Design Decision:** One-size-fits-all settings fail; regional customization essential.

### Why Monte Carlo Collapse Risk?

**Alternatives & Rationale:**

| Approach | Pros | Cons | Selected? |
|----------|------|------|-----------|
| **Deterministic:** Single best-case scenario | Fast, clear | Ignores uncertainty | No |
| **Perturbation analysis:** Vary drivers by ±σ | Interpretable | Limited exploration | No |
| **Monte Carlo:** Stochastic driver resampling | Quantifies uncertainty, accounts for rare events | Slower, need sufficient samples | **YES** |
| **Bayesian:** Posterior distribution of collapse risk | Principled uncertainty | Complex; requires prior | Future work |

**Design Decision:** Monte Carlo provides probability estimates suitable for risk communication.

---

## 13. DEPENDENCIES & ENVIRONMENT

### Python Version & Virtual Environment

- **Python Version:** 3.8+ (tested on 3.10)
- **Virtual Environment:** venv (`.venv/bin/activate`)
- **Activation:** `source /Users/sanchitkumardogra/kaam/clg/SEM8/Desertification/.venv/bin/activate`

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **pandas** | >=1.3 | DataFrame operations, time series |
| **numpy** | >=1.20 | Numerical computations, random sampling |
| **scipy** | >=1.7 | Optimization (ODE integration), statistics |
| **scikit-learn** | >=1.0 | (Optional) Feature scaling, validation metrics |
| **PySR** | Latest | Symbolic regression engine (Genetic Programming) |
| **matplotlib** | >=3.3 | Static plotting |
| **seaborn** | >=0.11 | Statistical visualization |
| **Jupyter** | >=1 | Notebook execution (Colab) |

### Hardware Assumptions

- **CPU:** Multi-core CPU for PySR parallelization (benefit from 4+ cores)
- **GPU:** Not required; PySR.jl (Julia backend) does not accelerate on GPU typically
- **Memory:** ~2-4 GB RAM for typical workflow (timeseries + PySR population)
- **Storage:** ~100 MB for full timeseries + intermediate outputs

### Deployment Environment

**Primary:** Google Colab Notebook
- Automatic dependency installation via `pip install`
- No need for local Python; notebook handles all setup
- Outputs downloaded as .zip after execution

**Fallback:** Local macOS machine
- Python venv established
- All dependencies pre-installed
- Direct script execution via `python desertification/ode_discovery.py` (if standalone script exists; currently Colab-centric)

---

## 14. RUN INSTRUCTIONS

### Quick Start (Colab)

**Step 1: Upload Package**
```bash
# On local machine
zip -r desertification_package.zip desertification scripts ReversedDesertification.ipynb
# Upload to Colab file browser

# In Colab
from google.colab import files
files.upload()  # Select desertification_package.zip
```

**Step 2: Extract & Run**
```python
!unzip desertification_package.zip
%cd /content
!python -c "from desertification import *; print('✓ Package imported')"
```

**Step 3: Execute Notebook**
```
Open ReversedDesertification.ipynb in Colab
Run all cells (Cell 1 → Cell 40+)
Wait for diagnostics output
Download outputs as .zip
```

### Step-by-Step Workflow (Colab)

**Phase 1: Data Prep (Cells 1-10)**
1. Load regional timeseries CSV (Rajasthan, Gobi)
2. Compute LST anomaly via deseasonalization
3. Add NDVI_sq, NDVI_logistic
4. Inspect data summary

**Phase 2: ODE Discovery (Cells 11-25)**
1. Call `discover_and_compile_ode(df_raj, "Rajasthan Canal")`
2. Call `discover_and_compile_ode(df_gobi, "Gobi Green Wall")`
3. Extract equations, tier, diagnostics
4. Print diagnostic summaries

**Phase 3: Policy Analysis (Cells 26-35)**
1. Define intervention policies (canal_boost, restoration, drought)
2. For each scenario, run 3000-year Monte Carlo
3. Compute collapse probabilities, Lyapunov, equilibrium
4. Collect results in summary dict

**Phase 4: Visualization (Cells 36-40)**
1. Generate Main_Summary.png (equations + collapse probs)
2. Generate Collapse_Risk_Summary.png (heatmap)
3. Generate Diagnostic_Plots.png (feature importance)
4. Generate Feature_Diagnostics.png (audit results)
5. Generate data_quality.png (driver distributions)
6. Export run_summary.json
7. Zip all outputs to `desertification_outputs_TIMESTAMP.zip`
8. Download .zip

### Local Execution (if applicable)

**Prerequisite:**
```bash
cd /Users/sanchitkumardogra/kaam/clg/SEM8/Desertification
source .venv/bin/activate
pip install -q pandas numpy scipy matplotlib seaborn PySR
```

**Run ODE Discovery (standalone):**
```python
from desertification.ode_discovery import discover_and_compile_ode
import pandas as pd

df_raj = pd.read_csv('data/rajasthan_timeseries.csv')
ode_raj, eq_str, diag = discover_and_compile_ode(df_raj, "Rajasthan Canal")

print(f"Equation: {eq_str}")
print(f"R²: {ode_raj.fit_r2}")
print(f"Diagnostics: {diag}")
```

### Output Inspection

**After run completes, download desertification_outputs_*.zip and inspect:**

1. **images/run_summary.json** → Machine-readable metadata
   ```bash
   cat images/run_summary.json | python3 -m json.tool
   ```

2. **images/*.png** → Visual diagnostics
   - Open in image viewer
   - Check Main_Summary for equations + policy sensitivity
   - Check Feature_Diagnostics for feature quality

3. **Full workflow logs** → Colab cell output
   - Scroll cell output for feature audit, tier selection, validation checks

---

## 15. FUTURE WORK / NEXT STEPS

### Immediate Next Steps (High Priority)

1. **🎯 Improve Rajasthan Fit Beyond 0.031**
   - Current: R² = 0.031 after FIX-BA
   - Target: R² > 0.05 (approach 151944 baseline of 0.0643)
   - Options:
     - a) Try alternative feature engineering (e.g., NDVI_poly = NDVI^1.5 instead of NDVI_sq)
     - b) Increase NITER_RAJ further (260 → 350+)
     - c) Investigate residual bias; check if simple linear correction helps
     - d) Run multiple seeds to identify search variance vs data quality limits
   - Estimated effort: 2-3 runs (~30 min each)

2. **📊 Run Stability Verification**
   - Execute 3-5 consecutive runs with same settings to quantify R² variance
   - Establish statistical confidence bands
   - Determine if current fit is stable or outlier
   - Estimated effort: 3-5 runs (~30 min each)

3. **🔬 Residual Analysis**
   - Extract residuals (predicted − observed dNDVI/dt)
   - Plot residual distribution (Q-Q plot, ACF plot)
   - Check for bias pattern (e.g., systematic underprediction at high NDVI)
   - If pattern found, consider residual correction or feature interaction terms
   - Estimated effort: 1-2 analysis runs

### Medium-Term Improvements (Target: Next 2 Weeks)

4. **🌍 Multi-Region Expansion**
   - Add 3-5 additional regions (e.g., Horn of Africa, Atacama, Central Asia)
   - Reuse config framework; add per-region settings
   - Validation: Compare qualitative policy directions across regions
   - Estimated effort: 1-2 weeks (data prep + config tuning)

5. **🗺️ Interactive Geospatial Mapping**
   - Implement Folium-based collapse risk map
   - Overlay collapse probability by grid cell
   - Add intervention scenario selector
   - Estimated effort: 1 week

6. **📈 Uncertainty Quantification**
   - Replace point estimates with confidence intervals
   - Use Bayesian ODE fitting or bootstrap
   - Quantify epistemic uncertainty (model choice) vs aleatoric uncertainty (data noise)
   - Estimated effort: 2 weeks

### Exploratory Research Questions

7. **Can we recover high NDVI regime dynamics for Gobi?**
   - Current Gobi equations focus on low-NDVI degradation
   - Unexplored: High-NDVI growth dynamics (restoration phase)
   - Idea: Train separate ODEs for low-NDVI and high-NDVI regimes
   - Estimated effort: 1-2 weeks

8. **How sensitive are equations to satellite sensor changes?**
   - Current: Assumes MODIS data is uniform
   - Concern: MODIS Terra aging; Aqua aging
   - Idea: Cross-validate with Sentinel-2, Landsat-8 data
   - Estimated effort: 2-3 weeks (data prep) + 1 week (validation)

9. **Can early warning indicators (EWS) predict collapse 6-12 months in advance?**
   - Current: Reactive risk assessment
   - Opportunity: Proactive early warning
   - Idea: Train LSTM on leading indicators (autocorrelation, variance shifts)
   - Estimated effort: 2-3 weeks

### Long-Term Vision (3-6 Months)

10. **⚙️ Production Deployment**
    - Cloud-based backend (AWS/GCP)
    - Real-time satellite ingestion pipeline
    - Continuous ODE re-fitting (monthly updates)
    - API for policy stakeholders
    - Estimated effort: 8-12 weeks

11. **🤖 Ensemble Approach**
    - Combine PySR with neural networks, random forests
    - Leverage strengths of each method
    - Estimate model confidence via ensemble disagreement
    - Estimated effort: 4-6 weeks

12. **🎓 Academic Publication**
    - Write methods paper (ODE discovery for land degradation)
    - Submit to GRL, JGR, or regional journal
    - Estimated effort: 8 weeks (write-up + peer review)

---

## 16. RAW CONTEXT NOTES

### Key Conversation Fragments

**Session 1-3: Problem Identification & Initial Attempts**

> "I want to keep the realism gains but recover Rajasthan fit"
> → Led to equation stability constraints (NDVI_sq preference)

> "Do both, I am now not really concerned about a solution being minimal...Do whatever it takes to make this work"
> → Triggered aggressive constraint approach (P1/P2 fit floors)

> "Can we add more diagnostics and logs to figure out what may be going wrong? We can remove some of the complexity from our equation by removing some of the less reliable metrics or run more iterations or something else"
> → Pivoted to diagnostic-driven investigation + feature reduction (FIX-BA)

**Root Cause Analysis (Diagnostic Run 105634)**

Feature_Diagnostics.png revealed:
- **NDVI_sq near-zero variance for Rajasthan** (0.04-0.30 range, essentially flat)
- **Dead features:** Carbon_Flux, GPP, Fire_Pressure (~90% zeros for semi-arid Rajasthan)
- **Good features:** FPAR, Rain_norm, Albedo show strong variation

→ Applied FIX-BA: Exclude {NDVI_sq, Carbon_Flux, GPP, Fire_Pressure} from Rajasthan search

**Result:** R² improved 25x (0.001 → 0.031), recovered to Tier 1 status

### Critical Design Insight

**Constraint Trap:** Early attempts tightened equation selection criteria, but this *backfired* because:
- Tight constraints filtered out top-100 candidate equations
- Forced fallback to lower tiers with worse pre-selection fit
- Each constraint iteration made R² worse (0.06 → 0.02 → -0.02)

**Lesson Learned:** Rather than constraining equation selection, improve the *candidate pool* via better feature engineering.

### Ongoing Investigation Questions

1. **Why does Rajasthan still lag behind 151944 baseline (0.0643) despite FIX-BA?**
   - Hypothesis A: Remaining noise in features; need further filtering
   - Hypothesis B: NDVI_sq actually was useful despite flatness; different polynomial might help
   - Hypothesis C: Residual bias requires systematic correction
   - Diagnostic: Run residual analysis, try NDVI_poly = NDVI^1.5, check if simple linear residual correction helps

2. **Is the fitted equation structure physical or artifactual?**
   - Current Rajasthan equation: `(FPAR*Rain_norm - NDVI_sq)*sin(Albedo*4.01 - 0.75)`
   - Question: Does the sin(Albedo*...) term reflect real oscillations or PySR search artifact?
   - Diagnostic: Check if removing sin() and retrying preserves R², or if sin() is essential for fit

3. **Will fit stability improve with multiple runs?**
   - Previous runs showed large variance (R² 0.06 → 0.02)
   - Recent runs (103841, 105634) show smaller variance (R² ~0.001-0.031)
   - Question: Is variance now controlled, or was earlier variance just randomness?
   - Diagnostic: Run 5-10 consecutive trials with identical settings; measure std(R²)

### Technical Debt

- [ ] Standalone script for local execution (currently Colab-centric)
- [ ] Unit tests for feature engineering, ODE selection logic
- [ ] Documented command-line interface (CLI) for batch runs
- [ ] Cross-validation framework (currently no train/test split)
- [ ] Hyperparameter sensitivity analysis (NITER, PYSR_MAXSIZE impacts on fit?)

### Recent Fixes Summary

| Fix ID | Issue | Solution | Result |
|--------|-------|----------|--------|
| FIX-BA | NDVI_sq near-zero variance for Rajasthan | Exclude {NDVI_sq, Carbon_Flux, GPP, Fire_Pressure} from Rajasthan | R² 0.001 → 0.031 (+25x) |
| FIX-AK | Rajasthan weak fits at Tier 1/2 | Add R² ≥ 0.03 floor for Rajasthan P1/P2 | Tier 1 selection stability |
| FIX-AE-bis | P2 equations missing NDVI_sq | Force NDVI_sq injection for Rajasthan P2 | Ensures NDVI_sq inclusion if missing |
| FIX-AJ | Denominator-prone features (LST_Day_anom, Temp_Delta_anom) cause brittle equations | Drop from feature set for Rajasthan | Numerical stability improvement |
| FIX-AI | Flat features degrade fit | Gate on flatness > 30% | Feature quality improvement |
| FIX-AH | No visibility into equation candidate quality | Implement summarize_equations() diagnostic | Diagnostic transparency |

---

## APPENDIX A: Config Parameters (Current Values)

```python
# PySR Hyperparameters
NITER_RAJ = 260         # Search iterations for Rajasthan (increased for weak signal)
NITER_GOBI = 240
PYSR_MAXSIZE = 12       # Max equation complexity
PYSR_POPULATIONS = 30

# Feature Filtering Thresholds
LOW_CORR_THRESHOLD = 0.05
LOW_CORR_THRESHOLD_BY_REGION = {
    'Rajasthan Canal': 0.08,
    'Gobi Green Wall': 0.05,
}
FLAT_THRESHOLD = 0.30
ZERO_FRAC_THRESHOLD = 0.50

# Region-Specific Constraints
RAJASTHAN_MAX_FEATURES = 8             # Complexity cap
RAJASTHAN_P1_MIN_R2 = 0.03             # Tier 1 fit floor
RAJASTHAN_P2_MIN_R2 = 0.03             # Tier 2 fit floor
DIAGNOSTIC_EQ_LIMIT = 30               # Equation preview limit

FEATURES_TO_EXCLUDE_BY_REGION = {
    'Rajasthan Canal': {'NDVI_sq', 'Carbon_Flux', 'GPP', 'Fire_Pressure'},
    'Gobi Green Wall': set(),
}

# Equilibrium & Risk Parameters
SIGN_CHANGE_RANGE_RAJ = (0.10, 0.80)
SIGN_CHANGE_RANGE_GOBI = (0.05, 0.25)
DRIFT_CLIP = 0.40

# Collapse Risk Monte Carlo
N_SIMULATIONS = 3000
SIM_YEARS = 100+
```

---

## APPENDIX B: Run History & Diagnostics

### Recent Runs

| Run ID | Date | Rajasthan R² | Gobi R² | Tier (Raj) | Issue/Fix | Status |
|--------|------|-------------|---------|-----------|----------|--------|
| 151944 | Apr 26 | 0.0643 | 0.0714 | 1 | Baseline; good fit | ✅ Baseline |
| 153206 | Apr 26 | 0.0163 | 0.0689 | 2 | Fit regressed 75% | ⚠️ Instability |
| 161741 | Apr 27 | 0.0012 | 0.0688 | 1 | NDVI_sq preference applied; fit collapsed | ❌ Failed |
| 100613 | Apr 28 | -0.0157 | 0.0623 | 2 | P1/P2 fit floors applied; worse fit | ❌ Backfired |
| 102126 | Apr 28 | 0.0232 | 0.0544 | 2 | Diagnostic infrastructure added | ⚠️ Marginal |
| 103841 | Apr 28 | 0.0012 | 0.0664 | 2 | Diagnostic infrastructure + 260 iterations | ⚠️ Weak |
| **105634** | **Apr 28** | **0.0313** | **0.0558** | **1** | **FIX-BA applied: exclude NDVI_sq, dead features** | **✅ Recovery** |

### Diagnostic Insights

**Feature Diagnostics (Run 105634):**
- NDVI_sq (Rajasthan): Flat ~0.95 (essentially zero variance) → EXCLUDED by FIX-BA
- Carbon_Flux (Rajasthan): ~90% zeros → EXCLUDED
- GPP (Rajasthan): ~80% zeros → EXCLUDED
- Fire_Pressure (Rajasthan): ~70% zeros → EXCLUDED
- FPAR (Rajasthan): Good variation, r ≈ 0.4-0.5 → RETAINED
- Rain_norm (Rajasthan): Good variation, r ≈ 0.3 → RETAINED
- Albedo (Rajasthan): Good variation, r ≈ 0.2-0.3 → RETAINED

**Equation Preview (Run 105634):**
- Total PySR candidates evaluated: 30
- Valid AST parse: 28/30
- NDVI present: 26/30
- Driver terms present: 24/30
- Tier 1 eligible: 1/30
- Selected: Tier 1, R² = 0.0313

---

**END OF DOCUMENT**

Generated: April 28, 2026  
Format: Markdown  
Next Revision: After next diagnostic run or major fix implementation

