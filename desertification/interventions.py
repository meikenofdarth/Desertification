# -*- coding: utf-8 -*-
"""
desertification.interventions
=============================
Policy intervention engine: scenario definition, driver modification,
sign audit, and auto-correction.

Extracted from V6 Cell 24 with V5's detailed documentation:
  - FIX-AB: K-modulating SDE for attractor-shifting interventions
  - FIX-AA: Physical direction metadata + derived variable protection
  - FIX-N:  Auto-correct inverted lever signs
  - FIX-F:  Post-run lever effectiveness audit
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .config import VARIABLE_META


@dataclass
class InterventionPlan:
    """
    Defines a policy intervention scenario.

    Parameters
    ----------
    name : str
        Human-readable scenario name.
    color : str
        Hex color for plotting.
    adjustments : dict
        {variable_name: [(start_year, end_year, delta), ...]}
        Each tuple shifts the variable by delta during [start, end).
    k_shifts : list of tuple
        FIX-AB: [(start_year, end_year, delta_K), ...]
        A positive delta_K raises the carrying capacity (attractor moves up);
        a negative delta_K lowers it (ecosystem degrades).
        Physical interpretation: sustained irrigation raises the water-limited
        carrying capacity; drought permanently lowers it.
    """
    name: str
    color: str
    adjustments: Dict[str, List[Tuple[float, float, float]]] = field(default_factory=dict)
    k_shifts: List[Tuple[float, float, float]] = field(default_factory=list)


def apply_interventions(drivers, current_year, plan, ode_features):
    """
    Apply time-varying driver modifications from an InterventionPlan.

    Parameters
    ----------
    drivers : dict
        Baseline driver values.
    current_year : float
        Current simulation year.
    plan : InterventionPlan
        Active policy scenario.
    ode_features : list of str
        Feature names in the ODE.

    Returns
    -------
    dict
        Modified driver values.
    """
    modified = drivers.copy()

    # Standard driver adjustments
    for variable, windows in plan.adjustments.items():
        if variable not in modified:
            continue
        for (start_yr, end_yr, delta) in windows:
            if start_yr <= current_year < end_yr:
                modified[variable] = max(modified[variable] + delta, 0.001)

    # FIX-AB: K-shift → attractor modulation
    if plan.k_shifts:
        k_total = sum(dk for (sy, ey, dk) in plan.k_shifts
                      if sy <= current_year < ey)
        if k_total != 0:
            k_base = modified.get('K_base', 0.50)
            modified['K_override'] = k_base + k_total

    return modified


def check_intervention_signs(ode_func, drivers, feat, plan, ndvi_test=None):
    """
    FIX-F: Audit whether each lever in a plan helps or hurts vegetation.

    For each adjusted variable, perturbs the driver and checks whether
    drift increases (helps) or decreases (hurts) at a test NDVI value.

    Parameters
    ----------
    ode_func : callable
    drivers : dict
    feat : list of str
    plan : InterventionPlan
    ndvi_test : float, optional
        NDVI value to test at (default 0.20).

    Returns
    -------
    dict
        {variable_name: drift_difference (positive = helps)}
    """
    if ndvi_test is None:
        ndvi_test = 0.20
    base = ode_func(ndvi_test, **drivers)
    print(f"\n  Intervention sign audit — {plan.name}")
    print(f"  Base drift at NDVI={ndvi_test:.3f}: {base:.4f}")
    results = {}
    for var, windows in plan.adjustments.items():
        if var not in feat:
            print(f"    {var:<25} NOT IN ODE — skipped")
            continue
        if VARIABLE_META.get(var, {}).get('lever_role') == 'diagnostic_only':
            print(f"    {var:<25} DIAGNOSTIC ONLY — skipped by FIX-AA")
            continue
        td = drivers.copy()
        delta = windows[0][2]
        td[var] = max(drivers.get(var, 0.0) + delta, 0.001)
        new = ode_func(ndvi_test, **td)
        diff = new - base
        dirn = ('helps (drift up)' if diff > 1e-6
            else 'hurts (drift down)' if diff < -1e-6
                else 'no measurable effect')
        print(f"    {var:<25} delta={delta:+.3f}  drift: {base:.6f} → {new:.6f}  "
              f"({diff:+.6f})  {dirn}")
        results[var] = diff
    return results


def auto_correct_plan(plan, sign_results, intent='positive', ode_tier=None):
    """
    FIX-N + FIX-AA: Tier-aware auto-correction of intervention lever signs.

    For P1–P4 ODEs: physical_good override is active (trusts physical
    direction over ODE sensitivity, since the ODE is structurally sound).
    For P5–P6 fallback ODEs: physical override is disabled (fallback ODEs
    may have structurally inverted coefficients).

    Parameters
    ----------
    plan : InterventionPlan
        Original plan to correct.
    sign_results : dict
        Output of check_intervention_signs.
    intent : str
        'positive' = this plan should help vegetation.
        'negative' = this plan should hurt vegetation (e.g. drought).
    ode_tier : int, optional
        Selection tier of the ODE (1–6). Affects override behaviour.

    Returns
    -------
    InterventionPlan
        Corrected plan.
    """
    tier = ode_tier if ode_tier is not None else 1
    corrected = InterventionPlan(
        name=plan.name + " [auto-corrected]",
        color=plan.color,
        adjustments={},
        k_shifts=list(getattr(plan, 'k_shifts', [])),
    )
    flipped = []
    skipped = []

    for var, windows in plan.adjustments.items():
        meta = VARIABLE_META.get(var, {})
        if meta.get('lever_role') == 'diagnostic_only':
            skipped.append(var)
            continue

        new_windows = []
        for (sy, ey, delta) in windows:
            diff = sign_results.get(var, None)
            if diff is not None and abs(diff) > 1e-6:
                helps       = diff > 0
                should_help = (intent == 'positive')
                phys_good   = meta.get('physical_good', None)

                # Check physical consistency
                phys_correct = (phys_good is None or
                                (phys_good > 0 and delta > 0 and should_help) or
                                (phys_good < 0 and delta < 0 and should_help) or
                                (phys_good > 0 and delta < 0 and not should_help) or
                                (phys_good < 0 and delta > 0 and not should_help))

                if helps != should_help:
                    if phys_correct and tier <= 4:
                        # Physical direction overrides ODE sensitivity
                        new_windows.append((sy, ey, delta))
                        print(f"  FIX-AA: Keeping '{var}' at {delta:+.3f} "
                              f"(physical direction overrides ODE sensitivity="
                              f"{diff:+.4f}, P{tier})")
                    else:
                        new_windows.append((sy, ey, -delta))
                        flipped.append(
                            f"{var}: {delta:+.3f} → {-delta:+.3f} "
                            f"(sensitivity={diff:+.4f}"
                            f"{', P5+ fallback' if tier >= 5 else ''})")
                else:
                    new_windows.append((sy, ey, delta))
            else:
                new_windows.append((sy, ey, delta))
        corrected.adjustments[var] = new_windows

    if skipped:
        print(f"  FIX-AA: Skipped derived levers: {skipped}")
    if flipped:
        print(f"  FIX-N: Auto-corrected levers in '{plan.name}':")
        for msg in flipped:
            print(f"    {msg}")
    else:
        print(f"  FIX-N: No corrections needed for '{plan.name}'.")
    return corrected


# ============================================================
# Predefined Intervention Plans
# ============================================================
# FIX-AA: Aridity_safe removed from all adjustments — it is a derived variable.
# FIX-AB: k_shifts encode carrying-capacity modulation.

GOBI_IRRIGATION_BASE = InterventionPlan(
    name="Gobi: 10-yr irrigation boost",
    color="#0277BD",
    adjustments={
        "Rain_norm":  [(2024, 2034, 0.20)],
        "SoilMoist":  [(2024, 2029, 0.04)],
    },
    k_shifts=[],
)

GOBI_RESTORATION_BASE = InterventionPlan(
    name="Gobi: full restoration package",
    color="#2E7D32",
    adjustments={
        "Rain_norm":      [(2024, 2074, 0.10)],
        "SoilMoist":      [(2024, 2039, 0.03)],
        "Dust_Stress":    [(2024, 2074, -0.03)],
        "Fire_Pressure":  [(2024, 2074, -1.00)],
        "FPAR":           [(2024, 2074, 0.05)],
        "Carbon_Flux":    [(2024, 2074, 0.005)],
    },
    k_shifts=[(2024, 2074, 0.06)],
)

RAJ_CANAL_BOOST_BASE = InterventionPlan(
    name="Rajasthan: canal water boost (20-yr)",
    color="#1565C0",
    adjustments={
        "SoilMoist":  [(2024, 2044, 0.05)],
        "Rain_norm":  [(2024, 2044, 0.15)],
        "FPAR":       [(2024, 2044, 0.05)],
    },
    k_shifts=[(2024, 2044, 0.10)],
)

RAJ_DROUGHT_BASE = InterventionPlan(
    name="Rajasthan: canal failure / drought",
    color="#B71C1C",
    adjustments={
        # Calibrated severity: keep drought clearly harmful while reducing
        # deterministic saturation in Monte Carlo outcomes.
        "SoilMoist":  [(2024, 2074, -0.03)],
        "Rain_norm":  [(2024, 2074, -0.09)],
        "FPAR":       [(2024, 2074, -0.02)],
    },
    k_shifts=[(2024, 2074, -0.045)],
)
