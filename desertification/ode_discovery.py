# -*- coding: utf-8 -*-
"""
desertification.ode_discovery
=============================
PySR symbolic regression, tiered equation selection, and ODE compilation.

Fixes applied in this version
------------------------------
FIX-BB  NDVI_poly (NDVI^1.5) supported throughout: augmentation rows,
        make_eval_fn, ode_func, tier selection, ndvi-presence checks,
        and ndvi_sq_preference extended to prefer either NDVI_sq or
        NDVI_poly (whichever is in feat).
FIX-BC  is_invalid_equation now rejects sin/cos applied to bounded
        [0,1] non-oscillatory variables (Albedo, NDVI, NDVI_poly,
        NDVI_sq, FPAR, SoilMoist).  This eliminates artifacts like
        sin(Albedo*4.01 − 0.75) that pass all other checks but have
        no ecological justification.

Earlier fixes retained unchanged
---------------------------------
FIX-AD  deseasonalised dNDVI/dt training target
FIX-AE  post-hoc NDVI_logistic injection when NDVI absent
FIX-AE-bis  injection when NDVI present but no sign-change + tier≥4
FIX-AF  gradient ceiling rows
FIX-AG  NDVI_logistic removed from PySR features
FIX-AH  PySR ^ constraint removed
FIX-I   no-sign-change rejection in all tiers
FIX-M   nested-trig rejection
FIX-D   multi-driver preference within tiers
FIX-C/J slope check at equilibrium
FIX-X   has_stable_sign_change + P2.5 tier
FIX-R   P4 requires sign-change; P5/P6 = fallbacks
"""

import numpy as np

from .config import (
    K_REGION, FEATURES, NDVI_DERIVED_FEATURES, DRIFT_CLIP,
    PYSR_BINARY_OPS, PYSR_BINARY_OPS_NO_DIV,
    PYSR_UNARY_OPS, PYSR_MAXSIZE, PYSR_POPULATIONS,
    RELIABILITY_MIN_R2, RELIABILITY_MAX_TIER, RELIABILITY_MAX_CLIP_SAT,
    RELIABILITY_MAX_NMAE_IQR, RELIABILITY_RELAXED_MAX_TIER,
    VARIABLE_META, RAJASTHAN_DENOM_PRONE_FEATURES,
    K_SHIFT_TRANSLATION_GAIN_DEFAULT, K_SHIFT_TRANSLATION_GAIN_BY_REGION,
    K_SHIFT_TRANSLATION_CLIP,
    RAJASTHAN_P1_MIN_R2,
    RAJASTHAN_P2_MIN_R2,
    RAJASTHAN_MAX_FEATURES,
    DIAGNOSTIC_EQ_LIMIT,
)
from .features import feature_audit, safe_features


def _has_nested_trig(expr):
    """
    FIX-M: True if sin/cos is applied to another sin/cos.
    """
    import sympy
    trig_funcs = (sympy.sin, sympy.cos)
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, tuple(trig_funcs)):
            for child in sympy.preorder_traversal(node.args[0]):
                if isinstance(child, tuple(trig_funcs)):
                    return True
    return False


def discover_and_compile_ode(df, region_name, niterations=100, eq_slope_max=2.0,
                              eq_bounds=(0.05, 0.90), sign_change_range=(0.05, 0.95),
                              ceil_config=None, K_default=0.50):
    """
    Discover the governing ODE for dNDVI/dt using PySR symbolic regression.

    Performs feature audit → safe feature selection → deseasonalised target
    computation → gradient ceiling augmentation → PySR fit → 6-tier
    priority selection → post-hoc NDVI_logistic injection → ODE compilation.

    Parameters
    ----------
    df : pd.DataFrame
    region_name : str
    niterations : int
    eq_slope_max : float
    eq_bounds : tuple
    sign_change_range : tuple
    ceil_config : dict, optional
    K_default : float

    Returns
    -------
    ode_func, feat, eq_str
    """
    print(f"\n{'='*60}\n  Discovering ODE for: {region_name}\n{'='*60}")

    import sympy
    from pysr import PySRRegressor

    # ── Feature selection ─────────────────────────────────────
    region_corrs = feature_audit(df, region_name)
    feat, dropped_info = safe_features(
        df,
        region_corrs,
        region_name=region_name,
        return_dropped=True,
    )
    diagnostics = {
        'feature_audit':   region_corrs,
        'feature_dropped': dropped_info,
    }

    # Rajasthan: drop near-zero anomaly features that attract brittle reciprocals.
    if region_name == 'Rajasthan Canal':
        drop_feats = []
        for f in RAJASTHAN_DENOM_PRONE_FEATURES:
            if f in feat:
                mu = float(df[f].mean()) if f in df.columns else 0.0
                if abs(mu) < 0.20:
                    drop_feats.append(f)
        if drop_feats and (len(feat) - len(drop_feats) >= 5):
            feat = [f for f in feat if f not in drop_feats]
            print(f"  FIX-AJ: Dropped denominator-prone anomaly features: {drop_feats}")

    # Rajasthan complexity cap.
    if region_name == 'Rajasthan Canal' and len(feat) > RAJASTHAN_MAX_FEATURES:
        # FIX-BB: treat NDVI_poly as a core feature (like NDVI_sq was before FIX-BA).
        core = [f for f in ['NDVI', 'NDVI_sq', 'NDVI_poly'] if f in feat]
        others = [f for f in feat if f not in core]
        ranked = sorted(
            others,
            key=lambda f: abs(region_corrs.get(f, {}).get('r', 0.0)),
            reverse=True,
        )
        keep    = core + ranked[: max(0, RAJASTHAN_MAX_FEATURES - len(core))]
        dropped = [f for f in feat if f not in keep]
        feat    = keep
        diagnostics['feature_reduced'] = {
            'max_features': RAJASTHAN_MAX_FEATURES,
            'kept': keep,
            'dropped': dropped,
        }
        print(f"  Rajasthan feature cap: kept {len(keep)} features, dropped {dropped}")

    print(f"  Features for PySR ({len(feat)}): {feat}")
    diagnostics['feature_final'] = feat

    df_s = df[feat].rolling(window=3, min_periods=1).mean()

    # ── FIX-AD: deseasonalised target ─────────────────────────
    y_raw = df_s['NDVI'].diff().fillna(0)
    y     = (y_raw - y_raw.groupby(y_raw.index.month).transform('mean')).values
    X     = df_s[feat].fillna(0).values
    print(f"  FIX-AD: Deseasonalised target — seasonal variance removed")

    # ── Helper: set NDVI-derived columns in a feature row ─────
    def _set_ndvi_derived(row, ndvi_val):
        """Update NDVI, NDVI_sq, and NDVI_poly slots in a feature row (in-place)."""
        if 'NDVI'      in feat: row[feat.index('NDVI')]      = ndvi_val
        if 'NDVI_sq'   in feat: row[feat.index('NDVI_sq')]   = ndvi_val ** 2
        if 'NDVI_poly' in feat: row[feat.index('NDVI_poly')] = ndvi_val ** 1.5

    # ── FIX-AF + FIX-Z: gradient ceiling rows ─────────────────
    cfg           = ceil_config or {
        'ndvi_points': [0.70, 0.75, 0.80, 0.85, 0.90],
        'drift_base':  -0.04,
        'weight':      5,
    }
    ndvi_max_ceil = cfg['ndvi_points'][-1]
    X_mean        = X.mean(axis=0)
    aug_rows_X    = []
    aug_rows_y    = []
    for ndvi_c in cfg['ndvi_points']:
        row     = X_mean.copy()
        drift_c = cfg['drift_base'] * (ndvi_c / ndvi_max_ceil)
        _set_ndvi_derived(row, ndvi_c)            # FIX-BB: sets NDVI_poly too
        for _ in range(cfg['weight']):
            aug_rows_X.append(row)
            aug_rows_y.append(drift_c)
    print(f"  FIX-AF: Gradient ceiling: {len(aug_rows_y)} rows, "
          f"NDVI [{cfg['ndvi_points'][0]:.2f}–{ndvi_max_ceil:.2f}], "
          f"drift [{cfg['drift_base']*(cfg['ndvi_points'][0]/ndvi_max_ceil):.4f} "
          f"to {cfg['drift_base']:.4f}]")

    # ── FIX-AI: growth floor rows ─────────────────────────────
    from .config import FLOOR_CONFIG
    floor_cfg = FLOOR_CONFIG.get(region_name, None)
    if floor_cfg:
        ndvi_min_floor = floor_cfg['ndvi_points'][0]
        ndvi_max_floor = floor_cfg['ndvi_points'][-1]
        for ndvi_f in floor_cfg['ndvi_points']:
            row     = X_mean.copy()
            drift_f = floor_cfg['drift_base'] * (1.0 - ndvi_f / ndvi_max_floor)
            drift_f = max(drift_f, 0.001)
            _set_ndvi_derived(row, ndvi_f)        # FIX-BB: sets NDVI_poly too
            for _ in range(floor_cfg['weight']):
                aug_rows_X.append(row)
                aug_rows_y.append(drift_f)
        print(f"  FIX-AI: Growth floor: {len(floor_cfg['ndvi_points'])*floor_cfg['weight']} rows, "
              f"NDVI [{ndvi_min_floor:.2f}–{ndvi_max_floor:.2f}], "
              f"drift [+{floor_cfg['drift_base']:.4f} gradient]")

    X_aug = np.vstack([X, np.array(aug_rows_X)])
    y_aug = np.concatenate([y, np.array(aug_rows_y)])
    print(f"  Total augmentation: {len(aug_rows_y)} synthetic rows + {len(X)} real = {len(X_aug)}")

    # ── PySR fit ───────────────────────────────────────────────
    binary_ops = PYSR_BINARY_OPS
    if region_name == 'Rajasthan Canal':
        binary_ops = PYSR_BINARY_OPS_NO_DIV
        print("  FIX-AJ: Using division-free binary operators for Rajasthan search")

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=binary_ops,
        unary_operators=PYSR_UNARY_OPS,
        model_selection="best",
        variable_names=feat,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        maxsize=PYSR_MAXSIZE,
        populations=PYSR_POPULATIONS,
    )
    model.fit(X_aug, y_aug, variable_names=feat)

    # ── Sympy symbols ─────────────────────────────────────────
    eq_df        = model.equations_.copy()
    ndvi_sym     = sympy.Symbol('NDVI')
    ndvi_sq_sym  = sympy.Symbol('NDVI_sq')
    ndvi_poly_sym = sympy.Symbol('NDVI_poly')     # FIX-BB
    sym_vars     = [sympy.Symbol(f) for f in feat]

    test_drivers = {f: float(df_s[f].mean())
                    for f in feat if f not in NDVI_DERIVED_FEATURES}

    # FIX-BB: prefer whichever nonlinear NDVI term is in feat.
    # After FIX-BA removes NDVI_sq for Rajasthan, NDVI_poly takes its role.
    prefer_ndvi_nonlinear = (
        region_name == 'Rajasthan Canal'
        and ('NDVI_poly' in feat or 'NDVI_sq' in feat)
    )
    require_ndvi_sq = (region_name == 'Rajasthan Canal' and 'NDVI_sq' in feat)

    def _is_ndvi_present(expr):
        """True if expr contains any NDVI-family symbol."""
        return bool(
            ndvi_sym      in expr.free_symbols or
            ndvi_sq_sym   in expr.free_symbols or
            ndvi_poly_sym in expr.free_symbols
        )

    def ndvi_sq_preference(expr):
        """
        FIX-BB: score bonus for any nonlinear NDVI term (NDVI_poly or NDVI_sq).

        Rajasthan dynamics are curvature-dominated; NDVI_poly now serves the
        role previously played by NDVI_sq before FIX-BA excluded it.
        """
        if not prefer_ndvi_nonlinear:
            return 1
        has_nonlin = (
            ndvi_sq_sym   in expr.free_symbols or
            ndvi_poly_sym in expr.free_symbols
        )
        return 1 if has_nonlin else 0

    def make_eval_fn(expr, K_val=K_default, clip=True):
        fn = sympy.lambdify(sym_vars, expr, modules=["numpy"])
        def _eval(ndvi_val, K_override=None):
            K_eff = K_override if K_override is not None else K_val
            args  = [test_drivers.get(f, 0.0) for f in feat]
            if 'NDVI'      in feat: args[feat.index('NDVI')]      = ndvi_val
            if 'NDVI_sq'   in feat: args[feat.index('NDVI_sq')]   = ndvi_val ** 2
            if 'NDVI_poly' in feat: args[feat.index('NDVI_poly')] = ndvi_val ** 1.5  # FIX-BB
            try:
                r = float(fn(*args))
                if not np.isfinite(r):
                    return None
                if clip:
                    return float(np.clip(r, -DRIFT_CLIP, DRIFT_CLIP))
                return r
            except Exception:
                return None
        return _eval

    def has_nonndvi_driver(expr):
        """FIX-D: check if equation uses at least one non-NDVI driver."""
        return bool(expr.free_symbols & {sympy.Symbol(f)
                                         for f in feat
                                         if f not in NDVI_DERIVED_FEATURES})

    sc_lo, sc_hi = sign_change_range

    def has_sign_change(ev):
        vals  = [ev(v) for v in np.linspace(sc_lo, sc_hi, 100)]
        valid = [v for v in vals if v is not None and np.isfinite(v)]
        return len(valid) >= 10 and min(valid) < 0 and max(valid) > 0

    def has_stable_sign_change(ev):
        scan   = np.linspace(sc_lo, sc_hi, 150)
        drifts = [ev(v) for v in scan]
        for k in range(len(drifts) - 1):
            d0, d1 = drifts[k], drifts[k + 1]
            if d0 is None or d1 is None or not (np.isfinite(d0) and np.isfinite(d1)):
                continue
            if d0 > 0 and d1 < 0:
                return True
        return False

    def clip_saturation_fraction(ev):
        vals  = [ev(v) for v in np.linspace(sc_lo, sc_hi, 200)]
        valid = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        if valid.size < 20:
            return 1.0
        return float(np.mean(np.abs(valid) >= DRIFT_CLIP))

    def has_reasonable_drift_scale(ev):
        vals  = [ev(v) for v in np.linspace(sc_lo, sc_hi, 200)]
        valid = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        if valid.size < 20:
            return False
        return float(np.max(np.abs(valid))) <= (DRIFT_CLIP * 6.0)

    def has_numerical_singularity(ev):
        scan = np.linspace(sc_lo, sc_hi, 400)
        raw  = []
        for v in scan:
            r = ev(v)
            raw.append(np.nan if (r is None or not np.isfinite(r)) else float(r))
        vals  = np.array(raw, dtype=float)
        valid = vals[np.isfinite(vals)]
        if valid.size < 50:
            return True
        jumps = np.abs(np.diff(valid))
        if jumps.size == 0:
            return True
        return float(np.quantile(jumps, 0.99)) > (DRIFT_CLIP * 0.75)

    def _prepare_prediction(pred_like):
        pred_arr = pred_like
        if np.isscalar(pred_arr):
            pred_arr = np.full_like(y, float(pred_arr), dtype=float)
        pred_arr = np.asarray(pred_arr, dtype=float)
        if pred_arr.shape[0] != y.shape[0]:
            pred_arr = np.resize(pred_arr, y.shape[0])
        pred_arr = np.nan_to_num(pred_arr, nan=0.0, posinf=DRIFT_CLIP, neginf=-DRIFT_CLIP)
        return np.clip(pred_arr, -DRIFT_CLIP, DRIFT_CLIP)

    def fit_stats(pred_arr):
        resid    = y - pred_arr
        var_y    = float(np.var(y))
        r2       = float(1.0 - np.var(resid) / (var_y + 1e-12))
        mae      = float(np.mean(np.abs(resid)))
        iqr      = float(np.percentile(y, 75) - np.percentile(y, 25))
        nmae_iqr = float(mae / (iqr + 1e-6))
        return {'r2': r2, 'mae': mae, 'nmae_iqr': nmae_iqr}

    def fit_r2_for_expr(expr):
        try:
            fn   = sympy.lambdify(sym_vars, expr, modules=["numpy"])
            cols = [df_s[f].fillna(0).values for f in feat]
            pred = fn(*cols)
            pred = _prepare_prediction(pred)
            return fit_stats(pred)['r2']
        except Exception:
            return float('-inf')

    def scan_equilibrium(ev):
        lo, hi = eq_bounds
        scan   = np.linspace(lo, hi, 150)
        drifts = [ev(v) for v in scan]
        for k in range(len(drifts) - 1):
            d0, d1 = drifts[k], drifts[k + 1]
            if d0 is None or d1 is None or not (np.isfinite(d0) and np.isfinite(d1)):
                continue
            if d0 > 0 and d1 < 0:
                eq_v = scan[k] + d0 * (scan[k + 1] - scan[k]) / (d0 - d1)
                if not (lo < eq_v < hi):
                    continue
                ev_lo = ev(eq_v - 0.01)
                ev_hi = ev(eq_v + 0.01)
                if ev_lo is None or ev_hi is None:
                    continue
                slope = (ev_hi - ev_lo) / 0.02
                if abs(slope) > eq_slope_max:
                    continue
                return eq_v, slope
        return None, None

    # FIX-BC: Bounded [0,1] non-oscillatory variables for which sin/cos is
    # ecologically meaningless.  Albedo/NDVI/FPAR/SoilMoist are constrained
    # proportions or indices — they do not oscillate in a physical sense.
    # PySR occasionally wraps them in sin/cos to fit noise; we reject those.
    _BOUNDED_SYMS = frozenset(
        sympy.Symbol(f)
        for f in ['Albedo', 'NDVI', 'NDVI_sq', 'NDVI_poly', 'FPAR', 'SoilMoist']
        if f in feat
    )

    def is_invalid_equation(expr):
        """FIX-M + FIX-BC: reject nested trigs, bad denominators, and
        sin/cos of bounded [0,1] non-oscillatory variables."""
        estr = str(expr).replace(' ', '')

        # FIX-M: nested trig or trig in denominator
        if '/sin(' in estr or '/cos(' in estr or 'sin(sin(' in estr or 'cos(cos(' in estr:
            return True

        # FIX-BC: sin/cos of bounded vars (e.g. sin(Albedo*4.01 - 0.75))
        for node in sympy.preorder_traversal(expr):
            if isinstance(node, (sympy.sin, sympy.cos)):
                if node.free_symbols & _BOUNDED_SYMS:
                    return True

        # Reject reciprocals of non-NDVI drivers
        non_ndvi_syms = {sympy.Symbol(f) for f in feat if f not in NDVI_DERIVED_FEATURES}
        for node in sympy.preorder_traversal(expr):
            if isinstance(node, sympy.Pow):
                base, exp = node.as_base_exp()
                if exp.is_number:
                    try:
                        exp_val = float(exp)
                    except Exception:
                        exp_val = 0.0
                    if exp_val < 0 and (base.free_symbols & non_ndvi_syms):
                        return True

        # String heuristic: direct division by non-NDVI feature
        for f in feat:
            if f in NDVI_DERIVED_FEATURES:
                continue
            if f"/{f}" in estr:
                return True

        return False

    def _trim_expr(expr_str, limit=160):
        if len(expr_str) <= limit:
            return expr_str
        return expr_str[:limit] + "..."

    def summarize_equations(eq_frame, limit=DIAGNOSTIC_EQ_LIMIT):
        preview = []
        counts  = {
            'total': int(len(eq_frame)),
            'evaluated': 0, 'valid_ast': 0,
            'ndvi_present': 0, 'ndvi_sq_present': 0, 'ndvi_poly_present': 0,
            'has_driver': 0, 'sign_change': 0, 'stable_sign_change': 0, 'clip_ok': 0,
        }
        for idx in eq_frame.sort_values('loss').index[:limit]:
            try:
                expr = model.sympy(index=idx)
            except Exception:
                continue
            counts['evaluated'] += 1
            if is_invalid_equation(expr):
                continue
            counts['valid_ast'] += 1

            ndvi_present     = _is_ndvi_present(expr)
            ndvi_sq_present  = bool(ndvi_sq_sym   in expr.free_symbols)
            ndvi_poly_present = bool(ndvi_poly_sym in expr.free_symbols)  # FIX-BB
            has_driver       = has_nonndvi_driver(expr)
            counts['ndvi_present']      += 1 if ndvi_present      else 0
            counts['ndvi_sq_present']   += 1 if ndvi_sq_present   else 0
            counts['ndvi_poly_present'] += 1 if ndvi_poly_present  else 0
            counts['has_driver']        += 1 if has_driver         else 0

            ev_raw   = make_eval_fn(expr, clip=False)
            sc       = has_sign_change(ev_raw)
            ssc      = has_stable_sign_change(ev_raw)
            clip_sat = clip_saturation_fraction(ev_raw)
            counts['sign_change']        += 1 if sc       else 0
            counts['stable_sign_change'] += 1 if ssc      else 0
            counts['clip_ok']            += 1 if clip_sat <= RELIABILITY_MAX_CLIP_SAT else 0

            r2       = fit_r2_for_expr(expr)
            loss_val = float(eq_frame.loc[idx, 'loss']) if 'loss' in eq_frame else 0.0
            complexity = None
            if 'complexity' in eq_frame.columns:
                try:
                    complexity = float(eq_frame.loc[idx, 'complexity'])
                except Exception:
                    pass

            preview.append({
                'idx': int(idx), 'loss': float(loss_val), 'complexity': complexity,
                'r2': float(r2) if np.isfinite(r2) else None,
                'ndvi_sq': bool(ndvi_sq_present), 'ndvi_poly': bool(ndvi_poly_present),
                'has_driver': bool(has_driver), 'sign_change': bool(sc),
                'stable_sign_change': bool(ssc), 'clip_saturation': float(clip_sat),
                'expr': _trim_expr(str(expr)),
            })

        by_loss = preview[:8]
        by_r2   = sorted(preview,
                         key=lambda row: (row['r2'] is not None, row['r2']),
                         reverse=True)[:8]
        return {'counts': counts, 'by_loss': by_loss, 'by_r2': by_r2}

    diagnostics['equation_preview'] = summarize_equations(eq_df)
    if diagnostics['equation_preview']['counts']['evaluated']:
        c = diagnostics['equation_preview']['counts']
        print(
            "  Eq preview: total={total}, eval={evaluated}, ndvi={ndvi_present}, "
            "ndvi_poly={ndvi_poly_present}, drivers={has_driver}, "
            "stable_sc={stable_sign_change}, clip_ok={clip_ok}".format(**c)
        )

    # ── 6-tier priority selection ─────────────────────────────
    best_idx       = None
    best_label     = ""
    selection_tier = 6
    p2_idx         = None

    # P1: stable equilibrium
    for req_drv in [True, False]:
        if best_idx is not None:
            break
        local_best  = None
        local_score = None
        for idx in eq_df.sort_values('loss').index:
            try:
                expr = model.sympy(index=idx)
                if is_invalid_equation(expr):
                    continue
                if not _is_ndvi_present(expr):       # FIX-BB: checks NDVI_poly too
                    continue
                if require_ndvi_sq and (ndvi_sq_sym not in expr.free_symbols):
                    continue
                if req_drv and not has_nonndvi_driver(expr):
                    continue
                ev_raw = make_eval_fn(expr, clip=False)
                if not has_reasonable_drift_scale(ev_raw):
                    continue
                if has_numerical_singularity(ev_raw):
                    continue
                if clip_saturation_fraction(ev_raw) > RELIABILITY_MAX_CLIP_SAT:
                    continue
                if not has_sign_change(ev_raw):
                    continue
                tv = [ev_raw(v) for v in [0.1, 0.3, 0.5, 0.7, 0.9]]
                if any(v is None for v in tv) or max(abs(v) for v in tv) > (DRIFT_CLIP * 4.0):
                    continue
                eq_v, slope = scan_equilibrium(ev_raw)
                if eq_v is None:
                    continue
                r2 = fit_r2_for_expr(expr)
                if not np.isfinite(r2):
                    continue
                if (region_name == 'Rajasthan Canal') and (r2 < RAJASTHAN_P1_MIN_R2):
                    continue
                try:
                    loss_val = float(eq_df.loc[idx, 'loss'])
                except Exception:
                    loss_val = 0.0
                score = (ndvi_sq_preference(expr), r2, -loss_val)
                if local_score is None or score > local_score:
                    local_score = score
                    local_best  = (idx, eq_v, slope, r2)
            except Exception:
                continue

        if local_best is not None:
            best_idx       = int(local_best[0])
            selection_tier = 1
            best_label     = (
                f"stable equilibrium at NDVI={local_best[1]:.3f} "
                f"(slope={local_best[2]:.2f}), P1 [fit-aware r2={local_best[3]:+.3f}]"
            )

    # P2: monotone decreasing + sign-change
    req_drv_values = [True, False]
    if region_name == 'Rajasthan Canal':
        req_drv_values = [True]
    for req_drv in req_drv_values:
        if best_idx is not None:
            break
        local_best  = None
        local_score = None
        for idx in eq_df.sort_values('loss').index:
            try:
                expr = model.sympy(index=idx)
                if is_invalid_equation(expr):
                    continue
                if not _is_ndvi_present(expr):       # FIX-BB
                    continue
                if require_ndvi_sq and (ndvi_sq_sym not in expr.free_symbols):
                    continue
                if req_drv and not has_nonndvi_driver(expr):
                    continue
                ev_raw = make_eval_fn(expr, clip=False)
                if not has_reasonable_drift_scale(ev_raw):
                    continue
                if has_numerical_singularity(ev_raw):
                    continue
                if clip_saturation_fraction(ev_raw) > RELIABILITY_MAX_CLIP_SAT:
                    continue
                if not has_sign_change(ev_raw):
                    continue
                vals = [ev_raw(v) for v in np.linspace(sc_lo, sc_hi, 8)]
                if any(v is None for v in vals):
                    continue
                if sum(1 for i in range(len(vals) - 1) if vals[i] > vals[i + 1]) < 5:
                    continue
                r2 = fit_r2_for_expr(expr)
                if not np.isfinite(r2):
                    continue
                if (region_name == 'Rajasthan Canal') and (r2 < RAJASTHAN_P2_MIN_R2):
                    continue
                try:
                    loss_val = float(eq_df.loc[idx, 'loss'])
                except Exception:
                    loss_val = 0.0
                score = (ndvi_sq_preference(expr), r2, -loss_val)
                if local_score is None or score > local_score:
                    local_score = score
                    local_best  = (idx, r2)
            except Exception:
                continue

        if local_best is not None:
            if p2_idx is None:
                p2_idx = int(local_best[0])
            best_idx       = int(local_best[0])
            selection_tier = 2
            best_label     = f"monotone decreasing + sign-change, P2 [fit-aware r2={local_best[1]:+.3f}]"

    # P2.5: stable sign-change (FIX-X)
    if best_idx is None:
        for req_drv in [True, False]:
            if best_idx is not None:
                break
            for idx in eq_df.sort_values('loss').index:
                try:
                    expr = model.sympy(index=idx)
                    if is_invalid_equation(expr):
                        continue
                    if not _is_ndvi_present(expr):   # FIX-BB
                        continue
                    if req_drv and not has_nonndvi_driver(expr):
                        continue
                    if _has_nested_trig(expr):
                        continue
                    ev_raw = make_eval_fn(expr, clip=False)
                    if not has_reasonable_drift_scale(ev_raw):
                        continue
                    if has_numerical_singularity(ev_raw):
                        continue
                    if clip_saturation_fraction(ev_raw) > RELIABILITY_MAX_CLIP_SAT:
                        continue
                    if not has_stable_sign_change(ev_raw):
                        continue
                    best_idx       = idx
                    selection_tier = 3
                    best_label     = f"stable sign-change P2.5, NDVI∈[{sc_lo:.2f},{sc_hi:.2f}]"
                    break
                except Exception:
                    continue

    # P3: any sign-change + no nested trig
    if best_idx is None:
        for idx in eq_df.sort_values('loss').index:
            try:
                expr = model.sympy(index=idx)
                if is_invalid_equation(expr):
                    continue
                if not _is_ndvi_present(expr):       # FIX-BB
                    continue
                if _has_nested_trig(expr):
                    continue
                ev_raw = make_eval_fn(expr, clip=False)
                if not has_reasonable_drift_scale(ev_raw):
                    continue
                if has_numerical_singularity(ev_raw):
                    continue
                if clip_saturation_fraction(ev_raw) > RELIABILITY_MAX_CLIP_SAT:
                    continue
                if has_sign_change(ev_raw):
                    best_idx       = idx
                    selection_tier = 3
                    best_label     = "NDVI-present, sign-change, no-nested-trig (P3)"
                    break
            except Exception:
                continue

    # P4: sign-change (FIX-R)
    if best_idx is None:
        for idx in eq_df.sort_values('loss').index:
            try:
                expr = model.sympy(index=idx)
                if is_invalid_equation(expr):
                    continue
                if not _is_ndvi_present(expr):       # FIX-BB
                    continue
                ev_raw = make_eval_fn(expr, clip=False)
                if not has_reasonable_drift_scale(ev_raw):
                    continue
                if has_numerical_singularity(ev_raw):
                    continue
                if clip_saturation_fraction(ev_raw) > RELIABILITY_MAX_CLIP_SAT:
                    continue
                if has_sign_change(ev_raw):
                    best_idx       = idx
                    selection_tier = 4
                    best_label     = "NDVI-present + sign-change (P4)"
                    break
            except Exception:
                continue

    # P5: NDVI present (fallback)
    if best_idx is None:
        for req_drv in [True, False]:
            if best_idx is not None:
                break
            for idx in eq_df.sort_values('loss').index:
                try:
                    expr = model.sympy(index=idx)
                    if is_invalid_equation(expr):
                        continue
                    if not _is_ndvi_present(expr):   # FIX-BB
                        continue
                    if req_drv and not has_nonndvi_driver(expr):
                        continue
                    ev_raw = make_eval_fn(expr, clip=False)
                    if not has_reasonable_drift_scale(ev_raw):
                        continue
                    if has_numerical_singularity(ev_raw):
                        continue
                    if clip_saturation_fraction(ev_raw) > RELIABILITY_MAX_CLIP_SAT:
                        continue
                    best_idx       = idx
                    selection_tier = 5
                    best_label     = ("NDVI + driver fallback (P5)"
                                      if req_drv else "NDVI-present fallback (P5)")
                    break
                except Exception:
                    continue

    # P6: absolute fallback
    if best_idx is None:
        for idx in eq_df.sort_values('loss').index:
            try:
                expr = model.sympy(index=idx)
                if not is_invalid_equation(expr):
                    best_idx       = int(idx)
                    selection_tier = 6
                    best_label     = "absolute fallback — best-loss equation (P6)"
                    break
            except Exception:
                continue
        if best_idx is None:
            best_idx       = int(eq_df.sort_values('loss').index[0])
            selection_tier = 6
            best_label     = "extreme fallback — ignoring AST checks (P6)"

    # Fit-recovery search
    provisional_expr = model.sympy(index=best_idx)
    provisional_r2   = fit_r2_for_expr(provisional_expr)
    if provisional_r2 < RELIABILITY_MIN_R2:
        print(f"  Fit-recovery search: provisional R^2={provisional_r2:+.3f} "
              f"< {RELIABILITY_MIN_R2:.2f}")
        rescue_idx   = None
        rescue_r2    = provisional_r2
        rescue_stable = False
        rescue_score = None

        for idx in eq_df.sort_values('loss').index:
            try:
                expr = model.sympy(index=idx)
            except Exception:
                continue
            if is_invalid_equation(expr):
                continue
            if not _is_ndvi_present(expr):           # FIX-BB
                continue
            if _has_nested_trig(expr):
                continue
            ev_raw   = make_eval_fn(expr, clip=False)
            if not has_reasonable_drift_scale(ev_raw):
                continue
            if has_numerical_singularity(ev_raw):
                continue
            clip_sat = clip_saturation_fraction(ev_raw)
            if clip_sat > RELIABILITY_MAX_CLIP_SAT:
                continue
            if not has_sign_change(ev_raw):
                continue
            r2 = fit_r2_for_expr(expr)
            if not np.isfinite(r2):
                continue
            stable = has_stable_sign_change(ev_raw)
            try:
                loss = float(eq_df.loc[idx, 'loss'])
            except Exception:
                loss = 0.0
            score = (1 if stable else 0, ndvi_sq_preference(expr), r2, -loss)

            if rescue_score is None or score > rescue_score:
                rescue_score  = score
                rescue_idx    = idx
                rescue_r2     = r2
                rescue_stable = stable

        if rescue_idx is not None and rescue_r2 > (provisional_r2 + 0.005):
            best_idx       = rescue_idx
            selection_tier = min(selection_tier, 3 if rescue_stable else 4)
            best_label     = (f"{best_label}; fit-recovery "
                              f"(R^2 {provisional_r2:+.3f} -> {rescue_r2:+.3f})")

    diagnostics['selection'] = {
        'tier': int(selection_tier),
        'label': best_label,
        'best_idx': int(best_idx),
    }

    best_sympy    = model.sympy(index=best_idx)
    eq_str        = str(best_sympy)
    ev_best_raw   = make_eval_fn(best_sympy, clip=False)
    ev_best       = make_eval_fn(best_sympy)
    clip_sat_best = clip_saturation_fraction(ev_best_raw)

    fallback_driverless = (selection_tier >= 5) and (not has_nonndvi_driver(best_sympy))
    if fallback_driverless:
        print("  FIX-AJ WARNING: Fallback equation is NDVI-only; intervention sensitivity may be weak.")

    sym_vars           = [sympy.Symbol(f) for f in feat]
    selected_raw_func  = sympy.lambdify(sym_vars, best_sympy, modules=["numpy"])

    def eval_selected_vector(df_like):
        """Vector prediction on smoothed design matrix (includes NDVI_poly if present)."""
        cols = [df_like[f].fillna(0).values for f in feat]
        pred = selected_raw_func(*cols)
        if np.isscalar(pred):
            pred = np.full(len(df_like), float(pred), dtype=float)
        pred = np.asarray(pred, dtype=float)
        if pred.shape[0] != len(df_like):
            pred = np.resize(pred, len(df_like))
        pred = np.nan_to_num(pred, nan=0.0, posinf=DRIFT_CLIP, neginf=-DRIFT_CLIP)
        return np.clip(pred, -DRIFT_CLIP, DRIFT_CLIP)

    print(f"\n  Selected [{best_label}]:")
    print(f"  dNDVI/dt = {eq_str}")

    # ── FIX-AE / FIX-AE-bis: post-hoc NDVI_logistic injection ──
    ndvi_present    = _is_ndvi_present(best_sympy)   # FIX-BB: checks NDVI_poly too
    alpha_inject    = 0.0
    needs_injection = False

    if not ndvi_present:
        needs_injection = True
        print(f"\n  FIX-AE: NDVI absent from equation — will attempt injection.")
    elif selection_tier >= 4 and not has_sign_change(ev_best_raw):
        needs_injection = True
        print(f"\n  FIX-AE-bis: NDVI present but NO sign-change at P{selection_tier} — "
              f"will attempt corrective injection.")

    if needs_injection:
        logistic_col = df_s['NDVI'] * (1.0 - df_s['NDVI'] / K_default)
        if logistic_col.std() > 1e-8:
            y_real   = y
            pred     = eval_selected_vector(df_s)
            residual = y_real - pred
            log_vals = logistic_col.values[:len(residual)]
            alpha_inject = float(np.clip(
                np.dot(residual, log_vals) / (np.dot(log_vals, log_vals) + 1e-12),
                -2.0, 2.0))
            alpha_before = alpha_inject
            if alpha_inject < 0.05:
                alpha_inject = max(alpha_inject, 0.15)
                print(f"  FIX-AE-bis: α too small ({alpha_before:.4f}), overriding to {alpha_inject:.4f}")

            def ev_alpha(v, a):
                b = ev_best_raw(v)
                return (b + a * v * (1 - v / K_default)) if b is not None else None

            if not has_stable_sign_change(lambda v: ev_alpha(v, alpha_inject)):
                grid_start       = max(0.05, alpha_inject)
                alpha_recovered  = None
                for a_try in np.linspace(grid_start, 2.0, 40):
                    if has_stable_sign_change(lambda v, aa=a_try: ev_alpha(v, aa)):
                        alpha_recovered = float(a_try)
                        break
                if alpha_recovered is not None and abs(alpha_recovered - alpha_inject) > 1e-9:
                    print(f"  FIX-AE-bis: No stable crossing at α={alpha_inject:.4f}; "
                          f"using α={alpha_recovered:.4f} (grid recovered)")
                    alpha_inject = alpha_recovered

            print(f"  Injecting α={alpha_inject:.4f} × NDVI_logistic(K={K_default})")
        else:
            alpha_inject = 0.0
            print("\n  FIX-AE: NDVI_logistic zero-variance — cannot inject.")

    sign_change_raw        = has_sign_change(ev_best_raw)
    stable_sign_change_raw = has_stable_sign_change(ev_best_raw)
    if alpha_inject != 0.0:
        def ev_effective(v):
            b = ev_best_raw(v)
            return (b + alpha_inject * v * (1 - v / K_default)) if b is not None else None
        sign_change_effective        = has_sign_change(ev_effective)
        stable_sign_change_effective = has_stable_sign_change(ev_effective)
        clip_sat_effective           = clip_saturation_fraction(ev_effective)
    else:
        sign_change_effective        = sign_change_raw
        stable_sign_change_effective = stable_sign_change_raw
        clip_sat_effective           = clip_sat_best

    # ── Compile ode_func closure ──────────────────────────────
    raw_func = selected_raw_func
    _alpha   = alpha_inject

    cols       = [df_s[f].fillna(0).values for f in feat]
    base_pred  = _prepare_prediction(raw_func(*cols))
    if _alpha != 0.0:
        ndvi_vals = df_s['NDVI'].fillna(0).values
        base_pred = np.clip(
            base_pred + _alpha * ndvi_vals * (1.0 - ndvi_vals / K_default),
            -DRIFT_CLIP, DRIFT_CLIP)
    fit_stats_base = fit_stats(base_pred)

    def build_residual_correction(base_prediction):
        candidates = [f for f in feat if f not in NDVI_DERIVED_FEATURES and f in df_s.columns]
        if len(candidates) < 1:
            return [], base_prediction, {'applied': False,
                                          'candidate_count': len(candidates),
                                          'selected_terms': 0}

        residual = y - base_prediction
        X_raw    = np.column_stack([df_s[f].fillna(float(df_s[f].mean())).values
                                    for f in candidates])
        means    = X_raw.mean(axis=0)
        stds     = X_raw.std(axis=0)
        valid_std = stds > 1e-8
        if int(np.sum(valid_std)) < 1:
            return [], base_prediction, {'applied': False,
                                          'candidate_count': len(candidates),
                                          'selected_terms': 0}

        X_raw = X_raw[:, valid_std]
        means = means[valid_std]
        stds  = stds[valid_std]
        names = [n for n, ok in zip(candidates, valid_std) if ok]
        X_z   = (X_raw - means) / (stds + 1e-12)

        if float(np.std(residual)) > 1e-10:
            corrs = []
            for j in range(X_z.shape[1]):
                xj = X_z[:, j]
                if float(np.std(xj)) <= 1e-10:
                    corrs.append(0.0)
                    continue
                c = np.corrcoef(xj, residual)[0, 1]
                corrs.append(float(c) if np.isfinite(c) else 0.0)
            corrs = np.asarray(corrs, dtype=float)
            keep  = np.abs(corrs) >= 0.015
            if int(np.sum(keep)) >= 1:
                X_z   = X_z[:, keep]
                X_raw = X_raw[:, keep]
                means = means[keep]
                stds  = stds[keep]
                names = [n for n, ok in zip(names, keep) if ok]

        if X_z.shape[1] == 0:
            return [], base_prediction, {'applied': False,
                                          'candidate_count': len(candidates),
                                          'selected_terms': 0}

        lam  = 0.15
        gram = X_z.T @ X_z + lam * np.eye(X_z.shape[1])
        rhs  = X_z.T @ residual
        try:
            weights = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(gram, rhs, rcond=None)[0]

        weights = np.clip(weights, -0.45, 0.45)

        for j, name in enumerate(names):
            phys = VARIABLE_META.get(name, {}).get('physical_good')
            if phys is None:
                continue
            if phys > 0 and weights[j] < 0:
                weights[j] *= 0.20
            if phys < 0 and weights[j] > 0:
                weights[j] *= 0.20

        order = np.argsort(np.abs(weights))[::-1]
        terms = []
        for j in order:
            coef = float(weights[j])
            if abs(coef) < 0.004:
                continue
            terms.append({
                'var':    names[j],
                'coef':   coef,
                'mean':   float(means[j]),
                'std':    float(stds[j]),
                'clip_z': 3.0,
            })
            if len(terms) >= 8:
                break

        if not terms:
            return [], base_prediction, {'applied': False,
                                          'candidate_count': len(candidates),
                                          'selected_terms': 0}

        corrected = base_prediction.copy()
        for term in terms:
            col = df_s[term['var']].fillna(term['mean']).values
            z   = np.clip((col - term['mean']) / (term['std'] + 1e-12),
                          -term['clip_z'], term['clip_z'])
            corrected += term['coef'] * z
        corrected = np.clip(corrected, -DRIFT_CLIP, DRIFT_CLIP)

        base_s   = fit_stats(base_prediction)
        corr_s   = fit_stats(corrected)
        score_base = base_s['r2'] - 0.25 * base_s['nmae_iqr']
        score_corr = corr_s['r2'] - 0.25 * corr_s['nmae_iqr']

        if base_s['r2'] < RELIABILITY_MIN_R2:
            improved = (
                corr_s['r2'] > (base_s['r2'] + 0.002) or
                corr_s['nmae_iqr'] < (base_s['nmae_iqr'] - 0.01) or
                score_corr > (score_base + 0.003)
            )
            if not improved and (corr_s['nmae_iqr'] <= base_s['nmae_iqr']) and \
                    (corr_s['r2'] >= base_s['r2'] - 0.002):
                improved = True
        else:
            improved = (
                corr_s['r2'] > (base_s['r2'] + 0.01) or
                corr_s['nmae_iqr'] < (base_s['nmae_iqr'] - 0.05)
            )

        meta = {
            'applied':             bool(improved),
            'candidate_count':     len(candidates),
            'selected_terms':      len(terms),
            'base_r2':             base_s['r2'],
            'base_nmae_iqr':       base_s['nmae_iqr'],
            'corrected_r2':        corr_s['r2'],
            'corrected_nmae_iqr':  corr_s['nmae_iqr'],
            'base_score':          score_base,
            'corrected_score':     score_corr,
        }
        if not improved:
            return [], base_prediction, meta
        return terms, corrected, meta

    correction_terms, pred_sel, correction_meta = build_residual_correction(base_pred)
    fit_stats_final = fit_stats(pred_sel)
    fit_r2          = fit_stats_final['r2']
    fit_mae         = fit_stats_final['mae']
    fit_nmae_iqr    = fit_stats_final['nmae_iqr']

    def build_policy_sensitivity_terms(expr):
        used_syms   = set(expr.free_symbols)
        terms       = []
        y_std       = float(np.std(y))
        driver_means = {}
        for f in feat:
            if f in NDVI_DERIVED_FEATURES or f not in df_s.columns:
                continue
            vals = df_s[f].fillna(float(df_s[f].mean())).values
            driver_means[f] = float(np.mean(vals))

        ndvi_anchor = float(np.clip(np.median(df_s['NDVI'].fillna(0).values), sc_lo, sc_hi))
        ndvi_anchor = float(np.clip(ndvi_anchor, sc_lo, K_default * 0.95))

        def local_delta_for_driver(var_name, step):
            args0 = [driver_means.get(f, 0.0) for f in feat]
            if 'NDVI'      in feat: args0[feat.index('NDVI')]      = ndvi_anchor
            if 'NDVI_sq'   in feat: args0[feat.index('NDVI_sq')]   = ndvi_anchor ** 2
            if 'NDVI_poly' in feat: args0[feat.index('NDVI_poly')] = ndvi_anchor ** 1.5  # FIX-BB
            args1 = list(args0)
            if var_name in feat:
                idx       = feat.index(var_name)
                args1[idx] = args0[idx] + step
            try:
                r0 = float(raw_func(*args0))
                r1 = float(raw_func(*args1))
                if not (np.isfinite(r0) and np.isfinite(r1)):
                    return 0.0
                return float(r1 - r0)
            except Exception:
                return 0.0

        for f in feat:
            if f in NDVI_DERIVED_FEATURES:
                continue
            meta_f = VARIABLE_META.get(f, {})
            if meta_f.get('lever_role') != 'direct':
                continue
            if f not in df_s.columns:
                continue

            vals  = df_s[f].fillna(float(df_s[f].mean())).values
            base  = float(np.mean(vals))
            std   = float(np.std(vals))
            if std < 1e-8:
                continue

            phys = meta_f.get('physical_good')
            if phys not in (-1, 1):
                continue

            corr = 0.0
            if y_std > 1e-10:
                try:
                    c    = np.corrcoef(vals, y)[0, 1]
                    corr = float(c) if np.isfinite(c) else 0.0
                except Exception:
                    corr = 0.0

            is_used     = sympy.Symbol(f) in used_syms
            step        = max(0.20 * std, 1e-4)
            local_delta = local_delta_for_driver(f, step)
            aligned     = bool(local_delta > 0.0) if phys > 0 else bool(local_delta < 0.0)
            weak        = abs(local_delta) < 5e-4

            if is_used and aligned and not weak:
                continue

            gain = 0.0040 + 0.0040 * abs(corr)
            if not is_used:  gain += 0.0030
            if weak:         gain += 0.0030
            if not aligned:  gain += 0.0040
            gain = float(min(gain, 0.0180))
            coef = float(gain if phys > 0 else -gain)

            terms.append({
                'var':         f,
                'coef':        coef,
                'base':        base,
                'std':         std,
                'clip_z':      3.0,
                'is_used':     bool(is_used),
                'aligned':     bool(aligned),
                'weak':        bool(weak),
                'local_delta': float(local_delta),
            })

        return terms

    policy_sensitivity_terms = build_policy_sensitivity_terms(best_sympy)

    def ode_func(ndvi_val, **driver_kwargs):
        """Evaluate drift dNDVI/dt at given NDVI and driver values."""
        K_eff = driver_kwargs.get('K_override', K_default)
        k_gain = float(K_SHIFT_TRANSLATION_GAIN_BY_REGION.get(
            region_name, K_SHIFT_TRANSLATION_GAIN_DEFAULT))
        k_shift_translation = float(np.clip(
            (K_eff - K_default) * k_gain,
            -K_SHIFT_TRANSLATION_CLIP,
            K_SHIFT_TRANSLATION_CLIP,
        ))
        eval_ndvi = max(ndvi_val - k_shift_translation, 0.0)

        args = [driver_kwargs.get(f, 0.0) for f in feat]
        if 'NDVI'      in feat: args[feat.index('NDVI')]      = eval_ndvi
        if 'NDVI_sq'   in feat: args[feat.index('NDVI_sq')]   = eval_ndvi ** 2
        if 'NDVI_poly' in feat: args[feat.index('NDVI_poly')] = eval_ndvi ** 1.5  # FIX-BB

        try:
            r = float(raw_func(*args))
            if _alpha != 0.0:
                r += _alpha * eval_ndvi * (1.0 - eval_ndvi / K_eff)
            if correction_terms:
                for term in correction_terms:
                    x_val = float(driver_kwargs.get(term['var'], term['mean']))
                    z_val = (x_val - term['mean']) / (term['std'] + 1e-12)
                    z_val = float(np.clip(z_val, -term['clip_z'], term['clip_z']))
                    r += term['coef'] * z_val
            if policy_sensitivity_terms:
                gate_raw = max(eval_ndvi * (1.0 - eval_ndvi / max(K_eff, 1e-6)), 0.0)
                gate     = float(np.clip(0.20 + 1.5 * gate_raw, 0.20, 0.50))
                for term in policy_sensitivity_terms:
                    x_val = float(driver_kwargs.get(term['var'], term['base']))
                    z_val = (x_val - term['base']) / (term['std'] + 1e-12)
                    z_val = float(np.clip(z_val, -term['clip_z'], term['clip_z']))
                    if abs(z_val) < 1e-12:
                        continue
                    r += term['coef'] * z_val * gate
            return float(np.clip(r, -DRIFT_CLIP, DRIFT_CLIP)) if np.isfinite(r) else 0.0
        except Exception:
            return 0.0

    # ── Attach metadata ───────────────────────────────────────
    ode_func.__doc__                       = eq_str
    ode_func.feat_names                    = feat
    ode_func.sympy_expr                    = best_sympy
    ode_func.model                         = model
    ode_func.p2_idx                        = p2_idx
    ode_func.K_default                     = K_default
    ode_func.selection_tier                = selection_tier
    ode_func.alpha_inject                  = _alpha
    ode_func.clip_saturation               = clip_sat_effective
    ode_func.sign_change_raw               = sign_change_raw
    ode_func.sign_change_effective         = sign_change_effective
    ode_func.stable_sign_change_effective  = stable_sign_change_effective
    ode_func.residual_correction_terms     = correction_terms
    ode_func.residual_correction_applied   = correction_meta.get('applied', False)
    ode_func.fit_r2_base                   = correction_meta.get('base_r2', fit_stats_base['r2'])
    ode_func.fit_nmae_iqr_base             = correction_meta.get('base_nmae_iqr', fit_stats_base['nmae_iqr'])
    ode_func.policy_sensitivity_terms      = policy_sensitivity_terms
    ode_func.policy_sensitivity_active     = bool(policy_sensitivity_terms)
    ode_func.make_eval_fn                  = make_eval_fn
    ode_func.fit_r2                        = fit_r2
    ode_func.fit_mae                       = fit_mae
    ode_func.fit_nmae_iqr                  = fit_nmae_iqr

    r2_ok          = fit_r2 >= RELIABILITY_MIN_R2
    nmae_ok        = fit_nmae_iqr <= RELIABILITY_MAX_NMAE_IQR
    fit_ok         = bool(r2_ok or nmae_ok)
    strict_tier_ok = selection_tier <= RELIABILITY_MAX_TIER
    relaxed_tier_ok = (
        selection_tier <= RELIABILITY_RELAXED_MAX_TIER
        and fit_ok
        and clip_sat_effective <= RELIABILITY_MAX_CLIP_SAT
        and bool(stable_sign_change_effective)
    )
    tier_ok        = bool(strict_tier_ok or relaxed_tier_ok)

    reliability_gate = {
        'tier_ok':       tier_ok,
        'fit_ok':        fit_ok,
        'clip_ok':       clip_sat_effective <= RELIABILITY_MAX_CLIP_SAT,
        'sign_change_ok': bool(stable_sign_change_effective),
    }
    reliability_checks = {
        **reliability_gate,
        'tier_strict_ok':  bool(strict_tier_ok),
        'tier_relaxed_ok': bool(relaxed_tier_ok),
        'r2_ok':           bool(r2_ok),
        'nmae_ok':         bool(nmae_ok),
    }
    reliability_pass              = all(reliability_gate.values())
    ode_func.reliability_checks   = reliability_checks
    ode_func.reliability_pass     = reliability_pass

    def _to_builtin(value):
        if isinstance(value, np.generic):      return value.item()
        if isinstance(value, dict):            return {k: _to_builtin(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):   return [_to_builtin(v) for v in value]
        return value

    ode_func.diagnostics = _to_builtin(diagnostics)

    # ── Diagnostics ───────────────────────────────────────────
    used = [f for f in feat if sympy.Symbol(f) in best_sympy.free_symbols]
    print(f"  Variables used: {used}")
    print(f"  Selection tier: P{selection_tier}")
    print(f"  Fit quality: R^2={fit_r2:+.3f}, nMAE(IQR)={fit_nmae_iqr:.3f}")
    if correction_meta.get('selected_terms', 0) > 0:
        status = 'applied' if correction_meta.get('applied', False) else 'rejected'
        print(
            f"  Residual correction {status}: terms={correction_meta['selected_terms']}, "
            f"R^2 {correction_meta.get('base_r2', fit_stats_base['r2']):+.3f} -> "
            f"{correction_meta.get('corrected_r2', fit_r2):+.3f}, "
            f"nMAE(IQR) {correction_meta.get('base_nmae_iqr', fit_stats_base['nmae_iqr']):.3f} -> "
            f"{correction_meta.get('corrected_nmae_iqr', fit_nmae_iqr):.3f}"
        )
    if policy_sensitivity_terms:
        names         = [t['var'] for t in policy_sensitivity_terms]
        weak_vars     = [t['var'] for t in policy_sensitivity_terms if t.get('weak')]
        misaligned_vars = [t['var'] for t in policy_sensitivity_terms if not t.get('aligned', True)]
        print(f"  FIX-AK: Policy sensitivity floor active for direct levers: {names}")
        if weak_vars or misaligned_vars:
            print(f"  FIX-AK: Corrective coverage — weak={weak_vars}, misaligned={misaligned_vars}")
    if fit_r2 < 0.0:
        print("  WARNING: Negative R^2 — equation has poor predictive skill on real data.")
    if (fit_r2 < RELIABILITY_MIN_R2) and nmae_ok:
        print("  NOTE: R^2 below threshold but absolute error acceptable for low-variance target.")
    if clip_sat_effective > RELIABILITY_MAX_CLIP_SAT:
        print(f"  WARNING: Drift clipping saturation is high ({clip_sat_effective:.0%}).")
    if _alpha != 0.0:
        print(f"  NDVI_logistic injection: α={_alpha:.4f}, K={K_default}")

    sc_msg = ('OK: Stable sign change confirmed'
              if stable_sign_change_effective
              else 'WARNING: No stable sign change')
    print(f"  {sc_msg}")
    print(f"  Reliability gate: {'PASS' if reliability_pass else 'FAIL'}")
    if not reliability_pass:
        for key, ok in reliability_gate.items():
            if not ok:
                print(f"    - failed: {key}")

    if _alpha != 0.0:
        def ev_inj(v):
            b = ev_best_raw(v)
            return (b + _alpha * v * (1 - v / K_default)) if b is not None else None
        if has_stable_sign_change(ev_inj):
            print("  FIX-AE+FIX-X: Stable crossing confirmed after injection.")

    if sign_change_effective:
        scan = np.linspace(sc_lo, sc_hi, 150)
        for k in range(len(scan) - 1):
            if _alpha != 0.0:
                d0 = ev_best_raw(scan[k])
                d1 = ev_best_raw(scan[k + 1])
                d0 = (d0 + _alpha * scan[k] * (1 - scan[k] / K_default)) if d0 is not None else None
                d1 = (d1 + _alpha * scan[k + 1] * (1 - scan[k + 1] / K_default)) if d1 is not None else None
            else:
                d0 = ev_best_raw(scan[k])
                d1 = ev_best_raw(scan[k + 1])
            if d0 is not None and d1 is not None and d0 < 0 and d1 > 0:
                print(f"  FIX-X WARNING: UNSTABLE crossing (− to +) at "
                      f"NDVI≈{scan[k]:.3f} — Allee-effect repeller")
                break

    return ode_func, feat, eq_str