#!/usr/bin/env python3
"""
Evaluate a desertification run output and generate summary artifacts.

Usage examples:
  python scripts/evaluate_run.py desertification_outputs_20260418_154505.zip
  python scripts/evaluate_run.py \
      desertification_outputs_20260418_154505.zip \
      --previous output_review/20260418_142206/images/run_summary.json

Outputs are written inside the evaluated run folder:
  - evaluation_summary.md
  - evaluation_tables.tex
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Minimum fit-stability requirements when comparing with a previous run.
# Slightly relaxed to avoid over-penalizing small stochastic run-to-run shifts.
FIT_STABILITY_MAX_R2_DROP = 0.015
FIT_STABILITY_MAX_NMAE_INCREASE = 0.015


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _fmt_num(value: Any, decimals: int = 4) -> str:
    if not _is_number(value):
        return "NA"
    return f"{float(value):.{decimals}f}"


def _fmt_pct(value: Any, decimals: int = 1) -> str:
    if not _is_number(value):
        return "NA"
    return f"{100.0 * float(value):.{decimals}f}%"


def _delta(current: Any, previous: Any) -> str:
    if _is_number(current) and _is_number(previous):
        return f"{float(current) - float(previous):+.4f}"
    if current == previous:
        return "UNCHANGED"
    return "CHANGED"


def _run_id_from_name(name: str) -> str:
    match = re.search(r"(\d{8}_\d{6})", name)
    if match:
        return match.group(1)
    return Path(name).stem


def _extract_zip(zip_path: Path, output_root: Path) -> Path:
    run_id = _run_id_from_name(zip_path.name)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(run_dir)
    return run_dir


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_summary_path(input_path: Path, output_root: Path) -> Tuple[Path, Dict[str, Any], str]:
    if input_path.suffix.lower() == ".zip":
        run_dir = _extract_zip(input_path, output_root)
        summary_path = run_dir / "images" / "run_summary.json"
        summary = _load_json(summary_path)
        return run_dir, summary, _run_id_from_name(input_path.name)

    if input_path.is_dir():
        summary_path = input_path / "images" / "run_summary.json"
        summary = _load_json(summary_path)
        return input_path, summary, input_path.name

    if input_path.is_file() and input_path.name == "run_summary.json":
        summary = _load_json(input_path)
        run_dir = input_path.parent.parent
        return run_dir, summary, run_dir.name

    raise ValueError(f"Unsupported input path: {input_path}")


def _scenario_value(region_values: Dict[str, Any], preferred: List[str]) -> Any:
    for key in preferred:
        if key in region_values:
            return region_values[key]
    return None


def _policy_checks(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ever = summary.get("collapse_probability_ever", {})
    checks: Dict[str, Dict[str, Any]] = {}

    # Rajasthan checks
    raj = ever.get("rajasthan", {}) if isinstance(ever, dict) else {}
    raj_baseline = _scenario_value(raj, ["baseline"])
    raj_boost = _scenario_value(raj, ["canal_boost", "major_interventions"])
    raj_drought = _scenario_value(raj, ["drought"])
    raj_boost_ok = (
        _is_number(raj_boost)
        and _is_number(raj_baseline)
        and float(raj_boost) <= float(raj_baseline)
    )
    raj_drought_ok = (
        _is_number(raj_drought)
        and _is_number(raj_baseline)
        and float(raj_drought) >= float(raj_baseline)
    )
    checks["rajasthan"] = {
        "boost_ok": bool(raj_boost_ok),
        "stress_ok": bool(raj_drought_ok),
    }

    # Gobi checks
    gobi = ever.get("gobi", {}) if isinstance(ever, dict) else {}
    gobi_baseline = _scenario_value(gobi, ["baseline"])
    gobi_restoration = _scenario_value(gobi, ["restoration", "major_interventions"])
    gobi_irrigation = _scenario_value(gobi, ["irrigation_boost"])
    gobi_restoration_ok = (
        _is_number(gobi_restoration)
        and _is_number(gobi_baseline)
        and float(gobi_restoration) <= float(gobi_baseline)
    )
    gobi_irrigation_ok = (
        _is_number(gobi_irrigation)
        and _is_number(gobi_baseline)
        and float(gobi_irrigation) <= float(gobi_baseline)
    )
    checks["gobi"] = {
        "restoration_ok": bool(gobi_restoration_ok),
        "irrigation_ok": bool(gobi_irrigation_ok),
    }

    return checks


def _collect_regions(summary: Dict[str, Any]) -> List[str]:
    regions = set()
    for key in ["selection_tier", "fit_r2", "fit_nmae_iqr", "reliability_pass"]:
        value = summary.get(key, {})
        if isinstance(value, dict):
            regions.update(value.keys())
    return sorted(regions)


def _region_rows(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    checks_map = summary.get("reliability_checks", {})
    for region in _collect_regions(summary):
        checks = checks_map.get(region, {}) if isinstance(checks_map, dict) else {}
        rows.append(
            {
                "region": region,
                "tier": summary.get("selection_tier", {}).get(region),
                "fit_r2": summary.get("fit_r2", {}).get(region),
                "fit_nmae_iqr": summary.get("fit_nmae_iqr", {}).get(region),
                "gate": summary.get("reliability_pass", {}).get(region),
                "tier_ok": checks.get("tier_ok"),
                "fit_ok": checks.get("fit_ok"),
                "clip_ok": checks.get("clip_ok"),
                "sign_change_ok": checks.get("sign_change_ok"),
            }
        )
    return rows


def _fit_stability_issues(current: Dict[str, Any], previous: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    regions = sorted(set(_collect_regions(current)) & set(_collect_regions(previous)))

    for region in regions:
        cur_r2 = current.get("fit_r2", {}).get(region)
        prev_r2 = previous.get("fit_r2", {}).get(region)
        if _is_number(cur_r2) and _is_number(prev_r2):
            delta_r2 = float(cur_r2) - float(prev_r2)
            if delta_r2 < -FIT_STABILITY_MAX_R2_DROP:
                issues.append(
                    f"{region} fit_r2 regression {delta_r2:+.4f} "
                    f"(< -{FIT_STABILITY_MAX_R2_DROP:.3f})"
                )

        cur_nmae = current.get("fit_nmae_iqr", {}).get(region)
        prev_nmae = previous.get("fit_nmae_iqr", {}).get(region)
        if _is_number(cur_nmae) and _is_number(prev_nmae):
            delta_nmae = float(cur_nmae) - float(prev_nmae)
            if delta_nmae > FIT_STABILITY_MAX_NMAE_INCREASE:
                issues.append(
                    f"{region} fit_nmae_iqr increase {delta_nmae:+.4f} "
                    f"(> +{FIT_STABILITY_MAX_NMAE_INCREASE:.3f})"
                )

    return issues


def _overall_verdict(
    summary: Dict[str, Any],
    policy_checks: Dict[str, Dict[str, Any]],
    previous_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    rel = summary.get("reliability_pass", {})
    regions = _collect_regions(summary)

    all_reliable = bool(regions) and all(bool(rel.get(region)) for region in regions)
    if not all_reliable:
        reasons.append("One or more regions failed reliability gate")

    raj_policy = policy_checks.get("rajasthan", {})
    if not (raj_policy.get("boost_ok") and raj_policy.get("stress_ok")):
        reasons.append("Rajasthan policy direction check weak")

    gobi_policy = policy_checks.get("gobi", {})
    if not (gobi_policy.get("restoration_ok") and gobi_policy.get("irrigation_ok")):
        reasons.append("Gobi policy direction check weak")

    if previous_summary is not None:
        fit_issues = _fit_stability_issues(summary, previous_summary)
        for issue in fit_issues:
            reasons.append(f"Fit stability check failed: {issue}")

    if all_reliable and not reasons:
        return "PASS", ["Reliability and policy checks satisfied"]
    if all_reliable:
        return "PASS_WITH_CAVEAT", reasons
    return "HOLD", reasons


def _comparison_rows(current: Dict[str, Any], previous: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    regions = sorted(set(_collect_regions(current)) | set(_collect_regions(previous)))
    keys = [
        ("selection_tier", ["selection_tier"]),
        ("fit_r2", ["fit_r2"]),
        ("fit_nmae_iqr", ["fit_nmae_iqr"]),
        ("reliability_pass", ["reliability_pass"]),
    ]

    for region in regions:
        for metric_name, path in keys:
            cur_val = current.get(path[0], {}).get(region)
            prev_val = previous.get(path[0], {}).get(region)
            rows.append(
                {
                    "region": region,
                    "metric": metric_name,
                    "previous": prev_val,
                    "current": cur_val,
                    "delta": _delta(cur_val, prev_val),
                }
            )

    return rows


def _write_markdown(
    output_path: Path,
    run_id: str,
    summary: Dict[str, Any],
    verdict: str,
    verdict_reasons: List[str],
    policy_checks: Dict[str, Dict[str, Any]],
    comp_rows: Optional[List[Dict[str, Any]]],
) -> None:
    rows = _region_rows(summary)
    lines: List[str] = []
    lines.append(f"# Run Evaluation: {run_id}")
    lines.append("")
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")

    if verdict_reasons:
        lines.append("## Verdict Notes")
        for reason in verdict_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    lines.append("## Region Diagnostics")
    lines.append("| Region | Tier | R2 | nMAE(IQR) | Gate | tier_ok | fit_ok | clip_ok | sign_change_ok |")
    lines.append("|---|---:|---:|---:|---|---|---|---|---|")
    for row in rows:
        lines.append(
            "| {region} | {tier} | {r2} | {nmae} | {gate} | {tier_ok} | {fit_ok} | {clip_ok} | {sc_ok} |".format(
                region=row["region"],
                tier=row["tier"] if row["tier"] is not None else "NA",
                r2=_fmt_num(row["fit_r2"], 4),
                nmae=_fmt_num(row["fit_nmae_iqr"], 4),
                gate=row["gate"],
                tier_ok=row["tier_ok"],
                fit_ok=row["fit_ok"],
                clip_ok=row["clip_ok"],
                sc_ok=row["sign_change_ok"],
            )
        )
    lines.append("")

    ever = summary.get("collapse_probability_ever", {})
    p2 = summary.get("collapse_probability_persistent_2yr", {})
    lines.append("## Collapse Metrics")
    for region in ["rajasthan", "gobi"]:
        region_ever = ever.get(region, {}) if isinstance(ever, dict) else {}
        region_p2 = p2.get(region, {}) if isinstance(p2, dict) else {}
        lines.append(f"### {region.title()}")
        lines.append(f"- Ever collapse: {region_ever}")
        lines.append(f"- Persistent 2yr collapse: {region_p2}")
    lines.append("")

    lines.append("## Policy Direction Checks")
    lines.append(f"- Rajasthan checks: {policy_checks.get('rajasthan', {})}")
    lines.append(f"- Gobi checks: {policy_checks.get('gobi', {})}")
    lines.append("")

    if comp_rows is not None:
        lines.append("## Comparison With Previous")
        lines.append("| Region | Metric | Previous | Current | Delta/Status |")
        lines.append("|---|---|---:|---:|---|")
        for row in comp_rows:
            lines.append(
                f"| {row['region']} | {row['metric']} | {row['previous']} | {row['current']} | {row['delta']} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _bool_to_tf(value: Any) -> str:
    if value is True:
        return "T"
    if value is False:
        return "F"
    return "NA"


def _write_latex(output_path: Path, run_id: str, summary: Dict[str, Any], verdict: str) -> None:
    rows = _region_rows(summary)
    run_id_tex = run_id.replace("_", "\\_")
    lines: List[str] = []
    lines.append("% Auto-generated by scripts/evaluate_run.py")
    lines.append(f"% Run ID: {run_id}")
    lines.append("\\subsection{Automated Evaluation Snapshot}")
    lines.append(f"\\textbf{{Run ID:}} {run_id_tex}\\\\")
    lines.append(f"\\textbf{{Verdict:}} {verdict}")
    lines.append("")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Automated region diagnostics}")
    lines.append("\\begin{tabular}{lcccccccc}")
    lines.append("\\toprule")
    lines.append("Region & Tier & $R^2$ & nMAE(IQR) & Gate & tier\\_ok & fit\\_ok & clip\\_ok & sign\\_ok \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(
            "{region} & {tier} & {r2} & {nmae} & {gate} & {tier_ok} & {fit_ok} & {clip_ok} & {sc_ok} \\\\".format(
                region=row["region"].title(),
                tier=row["tier"] if row["tier"] is not None else "NA",
                r2=_fmt_num(row["fit_r2"], 4),
                nmae=_fmt_num(row["fit_nmae_iqr"], 4),
                gate=_bool_to_tf(row["gate"]),
                tier_ok=_bool_to_tf(row["tier_ok"]),
                fit_ok=_bool_to_tf(row["fit_ok"]),
                clip_ok=_bool_to_tf(row["clip_ok"]),
                sc_ok=_bool_to_tf(row["sign_change_ok"]),
            )
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _print_console_summary(run_id: str, summary: Dict[str, Any], verdict: str, reasons: List[str]) -> None:
    print(f"Run: {run_id}")
    print(f"Verdict: {verdict}")
    if reasons:
        print("Verdict notes:")
        for reason in reasons:
            print(f"  - {reason}")

    print("\nRegion diagnostics:")
    for row in _region_rows(summary):
        print(
            f"  {row['region']}: tier={row['tier']} R2={_fmt_num(row['fit_r2'])} "
            f"nMAE(IQR)={_fmt_num(row['fit_nmae_iqr'])} gate={row['gate']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a desertification run output")
    parser.add_argument("input", type=Path, help="Path to run zip, extracted run directory, or run_summary.json")
    parser.add_argument(
        "--previous",
        type=Path,
        default=None,
        help="Optional previous run zip, run directory, or run_summary.json for comparison",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output_review"),
        help="Directory used for extracting zip inputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir, summary, run_id = _resolve_summary_path(args.input, args.output_root)

    prev_summary = None
    if args.previous is not None:
        _, prev_summary, _ = _resolve_summary_path(args.previous, args.output_root)

    policy_checks = _policy_checks(summary)
    verdict, verdict_reasons = _overall_verdict(summary, policy_checks, prev_summary)
    comp_rows = _comparison_rows(summary, prev_summary) if prev_summary is not None else None

    md_path = run_dir / "evaluation_summary.md"
    tex_path = run_dir / "evaluation_tables.tex"

    _write_markdown(md_path, run_id, summary, verdict, verdict_reasons, policy_checks, comp_rows)
    _write_latex(tex_path, run_id, summary, verdict)
    _print_console_summary(run_id, summary, verdict, verdict_reasons)

    print(f"\nSaved: {md_path}")
    print(f"Saved: {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
