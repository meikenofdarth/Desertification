# -*- coding: utf-8 -*-
"""
desertification.dynamics
========================
Lyapunov exponent computation, SDE integrator (Euler-Maruyama),
and Monte Carlo ensemble runner.

Extracted from V6 Cell 24 with FIX-E (Lyapunov sanity check)
and FIX-O (SDE burn-in drift damping).
"""

import numpy as np

from .config import (
    NDVI_DERIVED_FEATURES,
    NOISE_CALIBRATION_SCALE,
    NOISE_MIN_SIGMA,
    NOISE_MAX_SIGMA_DEFAULT,
    NOISE_MAX_SIGMA_BY_REGION,
    NOISE_REGION_MULTIPLIER,
    SIM_DRIVER_NOISE_REL,
    SIM_DRIVER_NOISE_MIN_ABS,
    SIM_DRIVER_NOISE_REGION_SCALE,
)


def make_drivers(df, feat, K_default=None):
    """
    Build driver dict from feature means for simulation.

    Excludes NDVI-derived features (they are recomputed dynamically
    during simulation from the current NDVI state).

    Parameters
    ----------
    df : pd.DataFrame
    feat : list of str
        Feature names used during ODE discovery.
    K_default : float, optional
        If provided, added as 'K_base' entry.

    Returns
    -------
    dict
        Feature name → mean value.
    """
    d = {f: float(df[f].mean())
         for f in feat
         if f not in NDVI_DERIVED_FEATURES and f in df.columns}
    if K_default is not None:
        d['K_base'] = K_default
    return d


def estimate_noise_sigma(ndvi_series, region_name=None):
    """
    Estimate stochastic noise amplitude from NDVI history (robustly).

    Uses robust spread estimators on first differences and applies
    region-aware caps to avoid outlier-driven collapse inflation.

    Parameters
    ----------
    ndvi_series : array-like
        Historical NDVI time series.
    region_name : str, optional
        Region key for region-specific calibration caps.

    Returns
    -------
    float
        Calibrated sigma for Euler-Maruyama noise term.
    """
    arr = np.asarray(ndvi_series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return float(NOISE_MIN_SIGMA)

    diffs = np.diff(arr)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 5:
        return float(NOISE_MIN_SIGMA)

    raw_std = float(np.std(diffs))

    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    robust_std = float(1.4826 * mad)

    q90 = float(np.quantile(np.abs(diffs), 0.90))
    clipped = np.clip(diffs, -q90, q90)
    core_std = float(np.std(clipped))

    candidates = [v for v in (raw_std, robust_std, core_std) if np.isfinite(v) and v > 0.0]
    if not candidates:
        sigma = float(NOISE_MIN_SIGMA)
    else:
        sigma = float(np.median(candidates))

    sigma *= float(NOISE_CALIBRATION_SCALE)
    sigma *= float(NOISE_REGION_MULTIPLIER.get(region_name, 1.0))
    sigma_max = float(NOISE_MAX_SIGMA_BY_REGION.get(region_name, NOISE_MAX_SIGMA_DEFAULT))
    sigma = float(np.clip(sigma, NOISE_MIN_SIGMA, sigma_max))
    return sigma


def _apply_driver_process_noise(driver_values, rng, region_name=None):
    """Apply mild stochastic variability to key climate/biophysical drivers."""
    if not isinstance(driver_values, dict):
        return driver_values

    jittered = dict(driver_values)
    region_scale = float(SIM_DRIVER_NOISE_REGION_SCALE.get(region_name, 1.0))
    upper_bounds = {
        'Rain_norm': 1.5,
        'SoilMoist': 1.0,
        'FPAR': 1.0,
    }

    for var, rel in SIM_DRIVER_NOISE_REL.items():
        if var not in jittered:
            continue
        base = abs(float(jittered[var]))
        sigma = max(base * float(rel), float(SIM_DRIVER_NOISE_MIN_ABS.get(var, 0.0)))
        sigma *= region_scale
        new_val = float(jittered[var]) + float(rng.normal(0.0, sigma))
        lo = 0.001
        hi = float(upper_bounds.get(var, max(1.0, lo + 10.0 * sigma)))
        jittered[var] = float(np.clip(new_val, lo, hi))

    return jittered


def calculate_lyapunov(ode_func, drivers, v_start, iterations=1000, dt=0.01):
    """
    Estimate the maximal Lyapunov exponent via trajectory divergence.

    Uses a twin-trajectory method: one reference trajectory and one
    perturbed trajectory, rescaling the perturbation at each step.

    Parameters
    ----------
    ode_func : callable
        f(ndvi_val, **drivers) → drift.
    drivers : dict
        Fixed driver values for simulation.
    v_start : float
        Initial NDVI value.
    iterations : int
    dt : float

    Returns
    -------
    float
        Estimated Lyapunov exponent.
    """
    v = v_start
    v_perturbed = v_start + 1e-6
    lyapunov_sum = 0.0

    for _ in range(iterations):
        # RK4-like step for both trajectories
        def rk4(val):
            k1 = ode_func(val, **drivers)
            k2 = ode_func(val + k1*dt/2, **drivers)
            k3 = ode_func(val + k2*dt/2, **drivers)
            k4 = ode_func(val + k3*dt, **drivers)
            return val + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

        v_next   = rk4(v)
        v_p_next = rk4(v_perturbed)

        distance = abs(v_next - v_p_next)
        if distance < 1e-15:
            distance = 1e-15

        lyapunov_sum += np.log(distance / 1e-6)

        # Rescale perturbation
        v = v_next
        v_perturbed = v_next + (v_p_next - v_next) * (1e-6 / distance)

    return lyapunov_sum / (iterations * dt)


def lyapunov_with_sanity(ode_func, drivers, v0, region_name):
    """
    Compute Lyapunov exponent with sanity checks (FIX-E).

    If |λ| > 10, warns and falls back to the P2 equation if available.

    Parameters
    ----------
    ode_func : callable
        Must have .p2_idx attribute.
    drivers : dict
    v0 : float
    region_name : str

    Returns
    -------
    float
        Lyapunov exponent.
    """
    lam = calculate_lyapunov(ode_func, drivers, v0)
    tier = getattr(ode_func, 'selection_tier', '?')
    print(f"\n  {region_name}: λ = {lam:.4f}  "
          f"({'STABLE' if lam < 0 else 'UNSTABLE'}) [P{tier}]")

    if abs(lam) > 10:
        print(f"  FIX-E WARNING: |λ| = {abs(lam):.1f} > 10 — suspiciously large.")
        p2_idx = getattr(ode_func, 'p2_idx', None)
        if p2_idx is not None:
            print(f"  FIX-E: Falling back to P2 equation (idx={p2_idx})")
            ev_p2 = ode_func.make_eval_fn(ode_func.model.sympy(index=p2_idx))
            # Recompute with P2 eval
            def p2_ode(v, **kw):
                r = ev_p2(v)
                return r if r is not None else 0.0
            lam2 = calculate_lyapunov(p2_ode, drivers, v0)
            if abs(lam2) < abs(lam):
                print(f"  FIX-E: P2 λ = {lam2:.4f} (using this instead)")
                return lam2
        if tier >= 5:
            print(f"  FIX-F: P{tier} fallback — results may be K-shift-only, "
                  f"treat λ as qualitative.")
    return lam


def sde_euler_maruyama(ode_func, v0, drivers, dt, steps, noise,
                       plan=None, ode_features=None,
                       forecast_start=2024.0, seed=42, burnin_years=2.0,
                       region_name=None, driver_noise=True):
    """
    Euler-Maruyama SDE integrator with optional intervention support.

    dy = f(y, drivers_t) dt  +  σ dW

    Parameters
    ----------
    ode_func : callable
        f(ndvi_val, **drivers) → drift.
    v0 : float
        Initial NDVI value.
    drivers : dict
        Baseline driver values.
    dt : float
        Timestep in years.
    steps : int
        Number of integration steps.
    noise : float
        Stochastic noise amplitude σ.
    plan : InterventionPlan, optional
        Policy scenario to apply.
    ode_features : list of str, optional
        Feature names (needed for apply_interventions).
    forecast_start : float
        Start year for the forecast.
    seed : int
        Random seed for reproducibility.
    burnin_years : float
        FIX-O: drift damping ramp-up period to suppress artefact
        collapse at t=0.
    region_name : str, optional
        Region name for region-specific process-noise scaling.
    driver_noise : bool
        If True, apply mild stochastic variability to selected drivers.

    Returns
    -------
    np.ndarray
        Trajectory of NDVI values, shape (steps,).
    """
    # Import here to avoid circular dependency
    from .interventions import apply_interventions

    rng  = np.random.default_rng(seed)
    traj = np.zeros(steps)
    v    = v0
    feat = ode_features or []
    sq_dt = np.sqrt(dt)
    burnin_steps = int(burnin_years / dt)

    for i in range(steps):
        active = (apply_interventions(drivers, forecast_start + i*dt, plan, feat)
                  if plan else drivers)
        if driver_noise:
            active = _apply_driver_process_noise(active, rng, region_name=region_name)
        drift  = ode_func(v, **active)
        # FIX-O: burn-in drift damping
        if i < burnin_steps:
            drift = drift * (i+1) / burnin_steps
        v = float(np.clip(v + drift*dt + noise*sq_dt*rng.standard_normal(), 0.0, 1.0))
        traj[i] = v
    return traj


def run_monte_carlo(ode_func, v0, drivers, dt, steps, noise, n,
                    plan=None, ode_features=None, forecast_start=2024.0,
                    region_name=None, driver_noise=True):
    """
    Run n independent SDE trajectories and return a (n, steps) array.

    Parameters
    ----------
    ode_func : callable
    v0 : float
    drivers : dict
    dt, steps, noise : float/int
    n : int
        Number of Monte Carlo simulations.
    plan : InterventionPlan, optional
    ode_features : list of str, optional
    forecast_start : float
    region_name : str, optional
    driver_noise : bool

    Returns
    -------
    np.ndarray
        Shape (n, steps).
    """
    return np.array([
        sde_euler_maruyama(
            ode_func, v0, drivers, dt, steps, noise,
            plan=plan,
            ode_features=ode_features,
            forecast_start=forecast_start,
            seed=i,
            region_name=region_name,
            driver_noise=driver_noise,
        )
        for i in range(n)
    ])
