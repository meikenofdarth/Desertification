Scientific Machine Learning for Desertification Reversal Stability

> A modular SciML pipeline that discovers governing differential equations from satellite data to assess whether large-scale desertification reversal efforts produce **stable ecological attractors** or **fragile states** vulnerable to collapse under future climate stress.

![Python](https://img.shields.io/badge/python-3.8+-blue?logo=python&logoColor=white)
![PySR](https://img.shields.io/badge/PySR-symbolic_regression-green)
![GEE](https://img.shields.io/badge/Google_Earth_Engine-satellite_data-orange?logo=google-earth)
![License](https://img.shields.io/badge/license-academic-lightgrey)

---

## Table of Contents

- [Overview](#overview)
- [Study Regions](#study-regions)
- [Pipeline Architecture](#pipeline-architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Results](#key-results)
- [Methodology](#methodology)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Overview

Desertification — the gradual transformation of productive land into barren desert — is one of the most pressing environmental challenges globally. Governments have invested heavily in reversal programmes, but a critical question remains: **are these reversals truly stable, or could they collapse under future climate stress?**

This project addresses that question by:

1. **Extracting** 19 years (2005–2023) of multi-sensor satellite data via Google Earth Engine
2. **Discovering** governing ODEs through symbolic regression (PySR)
3. **Analysing** ecosystem stability using Lyapunov exponents and phase portraits
4. **Forecasting** 50-year futures via stochastic differential equation (SDE) Monte Carlo simulation
5. **Evaluating** policy intervention scenarios with an auto-correcting lever engine
6. **Detecting** early warning signals of ecological tipping points (Critical Slowing Down)

---

## Study Regions

| Region | Location | Bounding Box | Strategy | Period |
|--------|----------|--------------|----------|--------|
| **Rajasthan Canal** | Indira Gandhi Canal, Rajasthan, India | 73.8–74.3°E, 29.3–29.8°N | Irrigation-driven greening | 2005–2023 |
| **Gobi Green Wall** | Gobi Desert, China | 108.2–108.8°E, 40.2–40.8°N | Afforestation (Three-North Shelter Forest) | 2005–2023 |

---

## Pipeline Architecture

```
Google Earth Engine → Data Extraction → Feature Engineering → PySR Symbolic Regression
                                                                      │
                                                              Discovered ODE
                                                                      │
                                    ┌─────────────────────────────────┼───────────────────┐
                                    ▼                                 ▼                   ▼
                           Lyapunov Stability              SDE Monte Carlo         Early Warning
                              Analysis                    (150 simulations)         Signals (CSD)
                                                                  │
                                                    ┌─────────────┼──────────────┐
                                                    ▼             ▼              ▼
                                                Baseline     Intervention    Stress Test
                                                Forecast      Scenarios      Scenarios
                                                    └─────────────┼──────────────┘
                                                                  ▼
                                                          Collapse Risk
                                                          Assessment
```

**Modules** (8-file Python package):

| Module | Purpose |
|--------|---------|
| `config.py` | Constants, hyperparameters, variable maps, fix provenance (54 named fixes) |
| `data.py` | GEE data extraction, MODIS scaling, quality checks, CSV caching |
| `features.py` | Feature engineering (LST anomaly, NDVI_poly, quality audit) |
| `ode_discovery.py` | PySR symbolic regression, 6-tier equation selection, post-hoc corrections |
| `dynamics.py` | Lyapunov exponents, Euler–Maruyama SDE integrator, Monte Carlo |
| `interventions.py` | Policy scenario engine, lever sign audit, auto-correction, K-shift |
| `ews.py` | Early Warning Signals (rolling variance, lag-1 autocorrelation, Kendall τ) |
| `plotting.py` | All figure generation (main summary, diagnostics, collapse risk) |

---

## Setup

### Prerequisites

- Python 3.8+
- A [Google Earth Engine](https://earthengine.google.com/) account (for fresh data extraction)
- [Google Colab](https://colab.research.google.com/) (recommended) or a local environment

### Option A: Google Colab (Recommended)

This is the primary execution environment. PySR requires Julia, which Colab handles automatically.

1. **Upload** the `desertification_package.zip` to your Colab working directory
2. **Upload** cached data files (optional, to skip GEE extraction):
   - `rajasthan_fixed.csv`
   - `gobi_fixed.csv`
3. **Open** the notebook `ReversedDesertification.ipynb` in Colab
4. **Install PySR** (first run only — requires runtime restart):
   ```python
   !pip install -q pysr
   import pysr; pysr.install()
   import os; os.kill(os.getpid(), 9)  # restart runtime
   ```
5. **Run all cells** from top to bottom after runtime restart

### Option B: Local Environment

```bash
# Clone the repository
git clone git@github.com:meikenofdarth/Desertification.git
cd Desertification

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy pandas matplotlib sympy scipy
pip install pysr            # requires Julia — see https://astroautomata.com/PySR/
pip install earthengine-api  # only needed for fresh data extraction

# Authenticate GEE (only needed for fresh data extraction)
earthengine authenticate
```

### Data

The pipeline supports two data modes:

| Mode | When to use | Requires |
|------|-------------|----------|
| **Cached CSV** | Default — uses pre-extracted `rajasthan_fixed.csv` and `gobi_fixed.csv` | CSV files in working directory |
| **Live GEE fetch** | First run or to refresh data | GEE authentication + project ID |

If running from cached CSVs, no GEE authentication is needed.

---

## Usage

### Full Pipeline Run

Open `ReversedDesertification.ipynb` and execute sequentially. The notebook runs through 6 stages:

```
[1/6] Data Extraction      — Load from CSV cache or fetch from GEE
[2/6] Feature Engineering   — Derived features, LST anomaly, quality audit
[3/6] ODE Discovery         — PySR symbolic regression for both regions
[4/6] Stability Analysis    — Lyapunov exponents + Early Warning Signals
[5/6] Interventions & MC    — Policy scenarios + 150-sim Monte Carlo
[6/6] Visualisation          — Generate all figures and run_summary.json
```

### Outputs

Each run produces a timestamped output bundle:

```
desertification_outputs_YYYYMMDD_HHMMSS/
├── images/
│   ├── Main_Summary.png           # 5-panel summary figure
│   ├── Diagnostic_Plots.png       # ODE drift + R² scatter
│   ├── Feature_Diagnostics.png    # Feature time-series comparison
│   ├── Collapse_Risk_Summary.png  # Collapse probability bar chart
│   ├── data_quality.png           # Data quality diagnostics
│   ├── run_summary.json           # Machine-readable results
│   └── cell_outputs_*.log         # Full cell output log
├── rajasthan_fixed.csv            # Cached satellite data
└── gobi_fixed.csv                 # Cached satellite data
```

### Run Evaluation

Use the evaluation script to generate a report-ready summary from any run:

```bash
# Evaluate a single run
python3 scripts/evaluate_run.py desertification_outputs_20260428_113226.zip

# Compare with a previous run
python3 scripts/evaluate_run.py desertification_outputs_20260428_113226.zip \
  --previous output_review/20260418_142206/images/run_summary.json
```

This generates:
- `evaluation_summary.md` — human-readable verdict
- `evaluation_tables.tex` — LaTeX tables for the report

---

## Project Structure

```
Desertification/
├── README.md                        # ← you are here
├── ReversedDesertification.ipynb     # Main pipeline notebook (Colab)
├── ReversedDesertification_archive.ipynb  # Legacy monolithic notebook (~11 MB)
│
├── desertification/                  # Core Python package
│   ├── __init__.py
│   ├── config.py                     # All constants and hyperparameters
│   ├── data.py                       # GEE extraction + CSV caching
│   ├── features.py                   # Feature engineering + audit
│   ├── ode_discovery.py              # PySR + tiered equation selection
│   ├── dynamics.py                   # Lyapunov + SDE + Monte Carlo
│   ├── interventions.py              # Policy scenario engine
│   ├── ews.py                        # Early Warning Signals
│   └── plotting.py                   # All visualisation
│
├── scripts/
│   ├── evaluate_run.py               # Post-run evaluation & comparison
│   └── README.md                     # Evaluation script docs
│
├── desertification_outputs_*/        # Timestamped run outputs
├── output_review/                    # Archived evaluated runs
│
├── rajasthan_fixed.csv               # Cached satellite data (Rajasthan)
├── gobi_fixed.csv                    # Cached satellite data (Gobi)
│
├── Final_Report.tex                  # LaTeX final report source
├── Final_Report_on_Reversing_Desertification.pdf
├── InterimLatex.tex                  # Interim report source
│
├── reverseddesertification.py        # Standalone script version
└── desertification_package.zip       # Deployable package archive
```

---

## Key Results

### Collapse Risk (50-year SDE Monte Carlo, 150 simulations)

| Scenario | Ever-Collapse | Terminal | Persistent (≥2 yr) |
|----------|:------------:|:--------:|:------------------:|
| **Rajasthan — Baseline** | 8.7% | 2.0% | 2.7% |
| **Rajasthan — Canal Boost** | 0.0% | 0.0% | 0.0% |
| **Rajasthan — Drought** | 96.7% | 88.0% | 93.3% |
| **Gobi — Baseline** | 93.3% | 36.0% | 67.3% |
| **Gobi — Irrigation** | 78.0% | 29.3% | 47.3% |
| **Gobi — Full Restoration** | 10.0% | 0.0% | 0.7% |

### Stability

| Region | Lyapunov λ | Stability | EWS Signal |
|--------|:----------:|:---------:|:----------:|
| Rajasthan | −0.036 | Stable | Rising variance only |
| Gobi | −0.045 | Stable | Rising variance only |

## Methodology

### Theoretical Pillars

1. **Dynamical systems theory** — ecosystems as nonlinear systems with attractors and tipping points
2. **Symbolic regression** (PySR) — discovers interpretable mathematical formulae from data
3. **Critical Slowing Down theory** — statistical early warning signals preceding regime shifts

### Satellite Data Sources

11 variables from 7 satellite product families:

- **Vegetation:** NDVI, EVI, LAI, FPAR (MODIS)
- **Temperature:** LST Day/Night (MODIS)
- **Water:** Precipitation (CHIRPS), Soil Moisture (FLDAS), Groundwater (GRACE-FO)
- **Surface:** Albedo (MODIS)
- **Atmosphere:** Aerosol Optical Depth (MODIS)
- **Carbon:** Gross Primary Productivity (MODIS)

### Iterative Development

The pipeline evolved through **54 named corrective fixes** (FIX-A through FIX-BD) and **33 complete pipeline runs**, systematically addressing data quality issues, equation selection failures, simulation instabilities, and intervention singularities.

---

## Limitations

- **Low R²:** Near-zero R² on deseasonalised dNDVI/dt (Rajasthan: 0.001, Gobi: 0.087) — a fundamental signal-to-noise limitation, not an engineering failure
- **Equation instability:** PySR's stochastic search produces different equations across runs
- **Missing irrigation data:** Rajasthan's primary driver (canal water delivery) is not directly measured in satellite data
- **Monthly resolution:** Rapid ecological responses are averaged out at monthly time steps

The model is best understood as a **decision-support framework for comparative risk ranking**, not a precise deterministic predictor.

