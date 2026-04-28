# -*- coding: utf-8 -*-
"""
desertification — Modular SciML pipeline for ecosystem stability analysis
=========================================================================

Merged from ReversedDesertification notebook versions v1–v6.
Analyses 19 years (2005–2023) of satellite data for:
  • Rajasthan Canal Region (India) — irrigation-driven greening
  • Gobi Desert (China) — afforestation-based restoration

Modules
-------
config          Constants, hyperparameters, variable maps
data            GEE data extraction, scaling, quality checks
features        Feature engineering, LST anomaly, audit
ode_discovery   PySR symbolic regression, tiered ODE selection
dynamics        Lyapunov exponents, SDE integrator, Monte Carlo
interventions   Policy scenario engine, sign audit, auto-correct
ews             Early Warning Signals (Critical Slowing Down)
plotting        All figure generation
"""

__version__ = "1.0.0"

# ── Always-available imports (no heavy dependencies) ──────────
from .config import (
    vars_map, K_REGION, FEATURES, NDVI_DERIVED_FEATURES,
    VARIABLE_META, CEIL_CONFIG, THRESH,
)
from .interventions import InterventionPlan, apply_interventions
from .interventions import check_intervention_signs, auto_correct_plan
from .ews import compute_ews

# ── Lazy imports for modules that require sympy/pysr/ee ───────
# These are available when the full scientific stack is installed
# (e.g. in Colab). Import them explicitly in your notebook:
#
#   from desertification.data import fetch_data_failsafe, load_or_fetch
#   from desertification.features import add_lst_anomaly, build_features
#   from desertification.ode_discovery import discover_and_compile_ode
#   from desertification.dynamics import make_drivers, calculate_lyapunov
#   from desertification.plotting import plot_main_figure, plot_diagnostics
