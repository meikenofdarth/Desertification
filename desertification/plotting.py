# -*- coding: utf-8 -*-
"""
desertification.plotting
========================
All figure generation for the pipeline.

Merged from:
  - V5 Cell 23: Phase portrait layout (FIX-AC), integrated main figure
  - V6 Cell 24: ribbon, plot_phase_portrait (fixed f-string),
    diagnostic figure, feature time-series
  - Cell 13: EWS panel, correlation panel, safe_corr
  - Cell 18: Collapse risk bar chart
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .config import FEATURES, THRESH, DRIFT_CLIP


# ============================================================
# Utilities
# ============================================================

def ribbon(ax, t, sims, color, label, alpha=0.20):
    """Plot Monte Carlo ensemble as mean line + IQR ribbon (FIX-T)."""
    mean = sims.mean(axis=0)
    p25  = np.percentile(sims, 25, axis=0)
    p75  = np.percentile(sims, 75, axis=0)
    ax.fill_between(t, p25, p75, color=color, alpha=alpha)
    ax.plot(t, mean, color=color, lw=2.0, label=label)
    return mean


def find_zero_crossings(ndvi_arr, drift_arr):
    """Find NDVI values where drift crosses zero (equilibrium points)."""
    crossings = []
    for k in range(len(drift_arr) - 1):
        d0 = drift_arr[k]
        d1 = drift_arr[k + 1]
        if d0 is None or d1 is None or not (np.isfinite(d0) and np.isfinite(d1)):
            continue
        if (d0 > 0 and d1 < 0) or (d0 < 0 and d1 > 0):
            # Linear interpolation to crossing point
            frac = d0 / (d0 - d1)
            eq_ndvi = ndvi_arr[k] + frac * (ndvi_arr[k + 1] - ndvi_arr[k])
            stable = d0 > 0 and d1 < 0  # + to − = stable attractor
            crossings.append((eq_ndvi, stable))
    return crossings


def safe_corr(df, var):
    """Safely compute Pearson correlation between NDVI and a variable."""
    if var not in df.columns:
        return np.nan
    valid = df[['NDVI', var]].dropna()
    if len(valid) < 5:
        return np.nan
    return valid['NDVI'].corr(valid[var])


# ============================================================
# Phase Portrait
# ============================================================

def plot_phase_portrait(ax, ode_f, drivers_d, v0_val, lambda_val,
                        region_label='', region_color='#2E7D32',
                        intervention_plans=None,
                        ndvi_range=(0.03, 0.90), sc_range=(0.05, 0.95)):
    """
    Plot drift curve with attractors, basins, and intervention overlays.

    Parameters
    ----------
    ax : matplotlib Axes
    ode_f : callable
        ODE function.
    drivers_d : dict
        Baseline driver values.
    v0_val : float
        Current NDVI value.
    lambda_val : float
        Lyapunov exponent.
    region_label : str
    region_color : str
    intervention_plans : list of tuple, optional
        [(plan, K_shifted, color, label), ...]
    ndvi_range : tuple
    sc_range : tuple
    """
    ndvi_arr = np.linspace(*ndvi_range, 200)

    # Baseline drift
    drift_arr = [ode_f(v, **drivers_d) for v in ndvi_arr]

    # Shade basins
    for i in range(len(ndvi_arr) - 1):
        c = '#81C784' if drift_arr[i] is not None and drift_arr[i] > 0 else '#EF9A9A'
        ax.axvspan(ndvi_arr[i], ndvi_arr[i + 1], alpha=0.06, color=c, lw=0)

    # Baseline drift curve
    ax.plot(ndvi_arr, drift_arr, color=region_color, lw=2.5, label='Baseline drift')

    # Zero line
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)

    # Baseline zero-crossings
    crossings = find_zero_crossings(ndvi_arr, drift_arr)
    for eq_v, stable in crossings:
        ax.plot(eq_v, 0,
                marker='o' if stable else 'o',
                markersize=10 if stable else 8,
                markerfacecolor=region_color if stable else 'white',
                markeredgecolor=region_color, markeredgewidth=2,
                zorder=10,
                label=f"{'Stable' if stable else 'Unstable'} eq: {eq_v:.2f}")

    # Intervention-shifted curves
    if intervention_plans:
        for plan_obj, k_shifted, color, label in intervention_plans:
            # Build modified drivers for mid-intervention state
            mod_drv = drivers_d.copy()
            for var, wins in plan_obj.adjustments.items():
                if var in mod_drv and wins:
                    mod_drv[var] = max(mod_drv[var] + wins[0][2], 0.001)
            if k_shifted is not None:
                mod_drv['K_override'] = k_shifted
            d_shifted = [ode_f(v, **mod_drv) for v in ndvi_arr]
            ax.plot(ndvi_arr, d_shifted, color=color, lw=1.5, ls='--',
                    alpha=0.8, label=label)
            # Mark shifted attractor
            cross_s = find_zero_crossings(ndvi_arr, d_shifted)
            for eq_v, stable in cross_s:
                if stable:
                    ax.plot(eq_v, 0, marker='D', markersize=7,
                            markerfacecolor=color, markeredgecolor='white',
                            markeredgewidth=1, zorder=10)

    # Current NDVI position
    ax.axvline(v0_val, color=region_color, lw=1.2, ls=':', alpha=0.8)
    drift_at_v0 = ode_f(v0_val, **drivers_d)
    ax.plot(v0_val, drift_at_v0, marker='*', markersize=14,
            color=region_color, zorder=11, label=f'Current v₀={v0_val:.2f}')

    # Collapse threshold
    ax.axvline(THRESH, color='red', lw=1.0, ls='--', alpha=0.6, label='Collapse threshold')

    # Formatting
    ax.set_xlabel('NDVI', fontsize=9)
    ax.set_ylabel('Drift dNDVI/dt', fontsize=9)
    tier = getattr(ode_f, 'selection_tier', '?')
    ax.set_title(f'{region_label}  (λ={lambda_val:.4f}) [P{tier}]',
                 fontsize=10, y=1.01, fontweight='medium')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.2)


# ============================================================
# Main Summary Figure (V5 layout + Cell 13 EWS panel)
# ============================================================

def plot_main_figure(time_axis, sims_raj, sims_gobi_base,
                     sims_gobi_irr, sims_gobi_rest,
                     sims_raj_boost, sims_raj_drought,
                     pc_raj, pc_gb, pc_irr, pc_rst, pc_rb, pc_rd,
                     ode_raj, ode_gobi, drivers_raj, drivers_gobi,
                     v0_raj, v0_gobi, lambda_raj, lambda_gobi,
                     raj_boost_plan, raj_drought_plan,
                     gobi_irr_plan, gobi_rest_plan,
                     ews_gobi=None,
                     save_path='images/Main_Summary.png'):
    """
    Generate the main 5-panel summary figure.

    Layout:
      Row 0  [full width]: Phase portraits (Rajasthan + Gobi)
      Row 1  [left]:  Panel A — Baseline regional comparison
      Row 1  [right]: Panel B — Gobi policy scenarios
      Row 2  [left]:  Panel C — EWS signals (if provided)
      Row 2  [right]: Panel D — Rajasthan policy scenarios
    """
    plt.close('all')
    fig = plt.figure(figsize=(16, 18), dpi=100)
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(
        3, 2, hspace=0.35, wspace=0.25,
        left=0.06, right=0.97, top=0.95, bottom=0.04,
        height_ratios=[1.0, 0.9, 0.9],
    )

    # Row 0: Phase Portraits
    gs_pp = gs[0, :].subgridspec(1, 2, wspace=0.20)
    ax_pp1 = fig.add_subplot(gs_pp[0])
    ax_pp2 = fig.add_subplot(gs_pp[1])

    from .config import SIGN_CHANGE_RANGE_RAJ, SIGN_CHANGE_RANGE_GOBI

    plot_phase_portrait(
        ax_pp1, ode_raj, drivers_raj, v0_raj,
        lambda_val=lambda_raj,
        region_label='Rajasthan Canal', region_color='#2E7D32',
        intervention_plans=[
            (raj_boost_plan, ode_raj.K_default + 0.10, '#1565C0', 'Canal boost (K+0.10)'),
            (raj_drought_plan, ode_raj.K_default - 0.15, '#B71C1C', 'Drought (K-0.15)'),
        ],
        ndvi_range=(0.03, 0.90), sc_range=SIGN_CHANGE_RANGE_RAJ)

    plot_phase_portrait(
        ax_pp2, ode_gobi, drivers_gobi, v0_gobi,
        lambda_val=lambda_gobi,
        region_label='Gobi Green Wall', region_color='#E65100',
        intervention_plans=[
            (gobi_rest_plan, ode_gobi.K_default + 0.06, '#2E7D32', 'Full restoration (K+0.06)'),
            (gobi_irr_plan, ode_gobi.K_default, '#0277BD', 'Irrigation boost'),
        ],
        ndvi_range=(0.03, 0.50), sc_range=SIGN_CHANGE_RANGE_GOBI)

    # Row 1, Panel A: Baseline comparison
    ax_a = fig.add_subplot(gs[1, 0])
    ribbon(ax_a, time_axis, sims_raj, '#4CAF50', f'Rajasthan (collapse {pc_raj:.0%})')
    ribbon(ax_a, time_axis, sims_gobi_base, '#FF9800', f'Gobi baseline (collapse {pc_gb:.0%})')
    ax_a.axhline(THRESH, color='#F44336', lw=0.8, ls=':', alpha=0.7, label='Collapse threshold')
    ax_a.set_title('A — Baseline Regional Comparison', fontsize=11, pad=8)
    ax_a.set_xlabel('Years into future', fontsize=9)
    ax_a.set_ylabel('NDVI', fontsize=9)
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.15)

    # Row 1, Panel B: Gobi scenarios
    ax_b = fig.add_subplot(gs[1, 1])
    ribbon(ax_b, time_axis, sims_gobi_base, '#FF9800', f'Baseline ({pc_gb:.0%})')
    ribbon(ax_b, time_axis, sims_gobi_irr, '#4FC3F7', f'Irrigation boost ({pc_irr:.0%})', alpha=0.15)
    ribbon(ax_b, time_axis, sims_gobi_rest, '#81C784', f'Full restoration ({pc_rst:.0%})', alpha=0.15)
    ax_b.axvspan(0, 10, color='#4FC3F7', alpha=0.05)
    ax_b.axvspan(0, 50, color='#81C784', alpha=0.03)
    ax_b.axhline(THRESH, color='#F44336', lw=0.8, ls=':', alpha=0.7, label='Collapse threshold')
    ax_b.set_title('B — Gobi Policy Scenario Comparison', fontsize=11, pad=8)
    ax_b.set_xlabel('Years into future', fontsize=9)
    ax_b.set_ylabel('NDVI', fontsize=9)
    ax_b.legend(fontsize=8)
    ax_b.grid(alpha=0.15)

    # Row 2, Panel C: EWS signals (from Cell 13)
    ax_c = fig.add_subplot(gs[2, 0])
    if ews_gobi is not None:
        years_hist = np.arange(len(ews_gobi)) / 12.0
        ax_c_twin = ax_c.twinx()
        ax_c.plot(years_hist, ews_gobi['EWS_Variance'].bfill(),
                  color='#FF9800', lw=1.5, label='Rolling variance', alpha=0.85)
        ax_c.plot(years_hist, ews_gobi['EWS_AC1'].bfill(),
                  color='#F44336', lw=1.5, label='AC(1)', alpha=0.85, linestyle='--')
        ax_c_twin.plot(years_hist, ews_gobi['EWS_Composite'].fillna(0),
                       color='#CE93D8', lw=1.2, label='Composite EWS', alpha=0.7, linestyle=':')
        ax_c.axhline(0, color='gray', lw=0.4, alpha=0.3)
        ax_c.set_ylabel('EWS statistic', fontsize=9)
        ax_c_twin.set_ylabel('Composite EWS (z-score)', color='#CE93D8', fontsize=8)
        ax_c_twin.tick_params(colors='#CE93D8')
        lines1, labels1 = ax_c.get_legend_handles_labels()
        lines2, labels2 = ax_c_twin.get_legend_handles_labels()
        ax_c.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax_c.set_title('C — Early Warning System (Gobi, 2005–2023)', fontsize=11, pad=8)
    ax_c.set_xlabel('Years since 2005', fontsize=9)
    ax_c.grid(alpha=0.15)

    # Row 2, Panel D: Rajasthan scenarios
    ax_d = fig.add_subplot(gs[2, 1])
    ribbon(ax_d, time_axis, sims_raj, '#4CAF50', f'Baseline ({pc_raj:.0%})')
    ribbon(ax_d, time_axis, sims_raj_boost, '#1565C0', f'Canal boost ({pc_rb:.0%})', alpha=0.15)
    ribbon(ax_d, time_axis, sims_raj_drought, '#B71C1C', f'Canal failure ({pc_rd:.0%})', alpha=0.15)
    ax_d.axhline(THRESH, color='#F44336', lw=0.8, ls=':', alpha=0.7, label='Collapse threshold')
    ax_d.set_title('D — Rajasthan Policy Scenarios', fontsize=11, pad=8)
    ax_d.set_xlabel('Years into future', fontsize=9)
    ax_d.set_ylabel('NDVI', fontsize=9)
    ax_d.legend(fontsize=8)
    ax_d.grid(alpha=0.15)

    fig.suptitle(
        "SciML Desertification Reversal Analysis  |  Merged Pipeline  |  2005–2073",
        fontsize=13, y=0.98, fontweight='medium')

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close('all')
    print(f"  Main figure saved: {save_path}")


# ============================================================
# Diagnostic Figure
# ============================================================

def plot_diagnostics(ode_raj, ode_gobi, drivers_raj, drivers_gobi,
                     v0_raj, v0_gobi, lambda_raj, lambda_gobi,
                     df_rajasthan, df_gobi, feat_raj, feat_gobi,
                     save_path='images/Diagnostic_Plots.png'):
    """ODE landscape + R² validation figure."""
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    fig.patch.set_facecolor('white')
    fig.suptitle("Diagnostic Plots — ODE Landscapes & Model Validation",
                 fontsize=12, fontweight='medium')

    ndvi_r = np.linspace(0.03, 0.95, 200)

    for ax, ode_f, drv, lam, color, label, v0 in [
        (axes[0, 0], ode_raj, drivers_raj, lambda_raj, '#2E7D32', 'Rajasthan', v0_raj),
        (axes[0, 1], ode_gobi, drivers_gobi, lambda_gobi, '#E65100', 'Gobi', v0_gobi),
    ]:
        d = [ode_f(v, **drv) for v in ndvi_r]
        ax.plot(ndvi_r, d, color=color, lw=2)
        ax.axhline(0, color='black', lw=0.8, alpha=0.6)
        ax.axvline(v0, color=color, lw=0.8, ls=':', label=f'v0={v0:.2f}')
        ax.axvline(THRESH, color='red', lw=1.0, ls='--', alpha=0.6, label='Collapse threshold')
        ax.fill_between(ndvi_r, d, 0, where=[x > 0 for x in d],
                        alpha=0.12, color='green', label='Growing')
        ax.fill_between(ndvi_r, d, 0, where=[x <= 0 for x in d],
                        alpha=0.12, color='red', label='Shrinking')
        tier = getattr(ode_f, 'selection_tier', '?')
        ax.set_title(f'{label} ODE landscape (λ={lam:.4f}) [P{tier}]', fontsize=10)
        ax.set_xlabel('NDVI', fontsize=9)
        ax.set_ylabel('Drift dNDVI/dt', fontsize=9)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.2)

    # R² on deseasonalised target (FIX-AD)
    def predict_selected_equation(ode_f, dfS, feat_r):
        """
        Evaluate the selected tiered equation (not model default-best equation).

        This keeps the diagnostic panel consistent with ode_func.fit_r2 and
        with what is actually used downstream in simulations.
        """
        import sympy

        sym_vars = [sympy.Symbol(f) for f in feat_r]
        raw_fn = sympy.lambdify(sym_vars, ode_f.sympy_expr, modules=["numpy"])
        cols = [dfS[f].fillna(0).values for f in feat_r]

        yp = raw_fn(*cols)
        if np.isscalar(yp):
            yp = np.full(len(dfS), float(yp), dtype=float)
        yp = np.asarray(yp, dtype=float)

        if yp.shape[0] != len(dfS):
            yp = np.resize(yp, len(dfS))

        yp = np.nan_to_num(yp, nan=0.0, posinf=DRIFT_CLIP, neginf=-DRIFT_CLIP)
        yp = np.clip(yp, -DRIFT_CLIP, DRIFT_CLIP)

        if ode_f.alpha_inject != 0.0:
            ndvi_c = dfS['NDVI'].fillna(0).values
            yp += ode_f.alpha_inject * ndvi_c * (1.0 - ndvi_c / ode_f.K_default)

        return yp

    for ax, ode_f, df_r, feat_r, color, label in [
        (axes[1, 0], ode_raj, df_rajasthan, feat_raj, '#2E7D32', 'Rajasthan'),
        (axes[1, 1], ode_gobi, df_gobi, feat_gobi, '#E65100', 'Gobi'),
    ]:
        dfS  = df_r[feat_r].rolling(window=3, min_periods=1).mean()
        y_r  = dfS['NDVI'].diff().fillna(0)
        y_t  = (y_r - y_r.groupby(y_r.index.month).transform('mean')).values
        yp   = predict_selected_equation(ode_f, dfS, feat_r)
        r2   = 1 - np.var(y_t - yp) / (np.var(y_t) + 1e-12)
        fit_r2 = getattr(ode_f, 'fit_r2', r2)
        sp   = int(len(y_t) * 0.80)
        lim  = max(abs(y_t).max(), abs(yp).max()) * 1.08
        ax.scatter(y_t[:sp], yp[:sp], alpha=0.4, s=12, color=color, label='Train')
        ax.scatter(y_t[sp:], yp[sp:], alpha=0.7, s=16, color='navy', label='Test')
        ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, label='Perfect')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f'{label}: deseasonalised R²={fit_r2:.3f}', fontsize=10)
        ax.set_xlabel('Observed dNDVI/dt (deseasonalised)', fontsize=9)
        ax.set_ylabel('PySR predicted', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close('all')
    print(f"  Diagnostic plots saved: {save_path}")


# ============================================================
# Feature Time-Series
# ============================================================

def plot_feature_timeseries(df_rajasthan, df_gobi,
                            save_path='images/Feature_Diagnostics.png'):
    """Feature comparison figure: Rajasthan vs Gobi."""
    plt.close('all')
    plot_f = [f for f in FEATURES
              if f in df_rajasthan.columns and df_rajasthan[f].std() > 1e-6][:12]
    nr = (len(plot_f) + 1) // 2
    fig, axs = plt.subplots(nr, 2, figsize=(14, nr * 2.8), dpi=100)
    fig.patch.set_facecolor('white')
    fig.suptitle("Feature Time Series — Rajasthan (green) vs Gobi (orange)",
                 fontsize=12, fontweight='medium')

    for i, fn in enumerate(plot_f):
        a = axs[i // 2, i % 2]
        if fn in df_rajasthan.columns:
            df_rajasthan[fn].plot(ax=a, color='#2E7D32', alpha=0.8, lw=0.8, label='Rajasthan')
        if fn in df_gobi.columns:
            df_gobi[fn].plot(ax=a, color='#E65100', alpha=0.8, lw=0.8, label='Gobi')
        a.set_title(fn, fontsize=9)
        a.legend(fontsize=7, loc='upper left')
        a.grid(alpha=0.15)
        a.set_xlabel('')

    for j in range(len(plot_f), nr * 2):
        axs[j // 2, j % 2].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close('all')
    print(f"  Feature time-series saved: {save_path}")


# ============================================================
# Collapse Risk Bar Chart (from Cell 18)
# ============================================================

def plot_collapse_risk(pc_raj, pc_gb, pc_irr, pc_rst, pc_rb, pc_rd,
                       save_path='images/Collapse_Risk_Summary.png'):
    """Collapse probability summary bar chart."""
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    fig.patch.set_facecolor('white')

    scenarios = [
        'Raj Baseline', 'Raj Canal Boost', 'Raj Drought',
        'Gobi Baseline', 'Gobi Irrigation', 'Gobi Restoration',
    ]
    probs  = [pc_raj, pc_rb, pc_rd, pc_gb, pc_irr, pc_rst]
    colors = ['#4CAF50', '#1565C0', '#B71C1C', '#FF9800', '#0277BD', '#2E7D32']

    bars = ax.barh(scenarios, [p * 100 for p in probs], color=colors, alpha=0.85)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f'{p:.1%}', va='center', fontsize=9)

    ax.set_xlabel('Collapse Probability (%)', fontsize=10)
    ax.set_title('50-Year Collapse Risk Summary', fontsize=12, fontweight='medium')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close('all')
    print(f"  Collapse risk chart saved: {save_path}")
