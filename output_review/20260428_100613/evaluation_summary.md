# Run Evaluation: 20260428_100613

**Verdict:** PASS_WITH_CAVEAT

## Verdict Notes
- Fit stability check failed: rajasthan fit_r2 regression -0.0801 (< -0.015)
- Fit stability check failed: rajasthan fit_nmae_iqr increase +0.0249 (> +0.015)

## Region Diagnostics
| Region | Tier | R2 | nMAE(IQR) | Gate | tier_ok | fit_ok | clip_ok | sign_change_ok |
|---|---:|---:|---:|---|---|---|---|---|
| gobi | 1 | 0.1091 | 0.6589 | True | True | True | True | True |
| rajasthan | 2 | -0.0157 | 0.5678 | True | True | True | True | True |

## Collapse Metrics
### Rajasthan
- Ever collapse: {'baseline': 0.16666666666666666, 'major_interventions': 0.0, 'canal_boost': 0.0, 'drought': 0.9933333333333333}
- Persistent 2yr collapse: {'baseline': 0.12666666666666668, 'major_interventions': 0.0, 'canal_boost': 0.0, 'drought': 0.9866666666666667}
### Gobi
- Ever collapse: {'baseline': 0.9333333333333333, 'major_interventions': 0.11333333333333333, 'irrigation_boost': 0.8066666666666666, 'restoration': 0.11333333333333333}
- Persistent 2yr collapse: {'baseline': 0.7466666666666667, 'major_interventions': 0.006666666666666667, 'irrigation_boost': 0.54, 'restoration': 0.006666666666666667}

## Policy Direction Checks
- Rajasthan checks: {'boost_ok': True, 'stress_ok': True}
- Gobi checks: {'restoration_ok': True, 'irrigation_ok': True}

## Comparison With Previous
| Region | Metric | Previous | Current | Delta/Status |
|---|---|---:|---:|---|
| gobi | selection_tier | 1 | 1 | +0.0000 |
| gobi | fit_r2 | 0.07347052484044114 | 0.10910334481228923 | +0.0356 |
| gobi | fit_nmae_iqr | 0.6742350370430665 | 0.6589003694490728 | -0.0153 |
| gobi | reliability_pass | True | True | +0.0000 |
| rajasthan | selection_tier | 1 | 2 | +1.0000 |
| rajasthan | fit_r2 | 0.06431475168353618 | -0.015748981209874557 | -0.0801 |
| rajasthan | fit_nmae_iqr | 0.5429566472137326 | 0.5678222520646996 | +0.0249 |
| rajasthan | reliability_pass | True | True | +0.0000 |
