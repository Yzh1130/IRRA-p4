"""Summarize clean and ten degradation scenarios from a strict manifest."""

import argparse
import os
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.revision.score_io import (  # noqa: E402
    METRIC_KEYS,
    compute_retrieval_metrics,
    load_json_manifest,
    load_score_file,
    prepare_output_parents,
    report_paths,
    require_outputs_absent,
    require_score_consistency,
    resolve_manifest_path,
    write_json,
    write_markdown,
    write_tsv,
)


CLEAN_SCENARIO = "clean"
DEGRADED_SCENARIOS = (
    "rgb_missing",
    "ir_missing",
    "rgb_noise_005",
    "rgb_noise_010",
    "ir_noise_005",
    "ir_noise_010",
    "rgb_blur_k7",
    "ir_blur_k7",
    "both_noise_005",
    "both_blur_k7",
)
REQUIRED_SCENARIOS = (CLEAN_SCENARIO, *DEGRADED_SCENARIOS)


def validate_robustness_manifest(manifest):
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, dict):
        raise TypeError("Robustness manifest requires a 'scenarios' JSON object.")
    missing = [name for name in REQUIRED_SCENARIOS if name not in scenarios]
    if missing:
        raise ValueError(f"Robustness manifest is missing scenarios: {missing}")
    extra = sorted(set(scenarios) - set(REQUIRED_SCENARIOS))
    if extra:
        raise ValueError(f"Robustness manifest has unsupported scenarios: {extra}")
    for name in REQUIRED_SCENARIOS:
        record = scenarios[name]
        if not isinstance(record, dict) or not str(record.get("score_file", "")).strip():
            raise ValueError(f"Scenario {name!r} requires a non-empty score_file.")
        if name != CLEAN_SCENARIO and not isinstance(
            record.get("seen_in_training"), bool
        ):
            raise ValueError(
                f"Scenario {name!r} must explicitly set seen_in_training to true/false."
            )
    return scenarios


def summarize_metrics(model_name, scenario_metrics, seen_flags):
    clean = scenario_metrics[CLEAN_SCENARIO]
    degraded = [scenario_metrics[name] for name in DEGRADED_SCENARIOS]

    degraded_mean_r1 = statistics.fmean(row["R1"] for row in degraded)
    degraded_mean_map = statistics.fmean(row["mAP"] for row in degraded)
    delta_r1 = {
        name: abs(scenario_metrics[name]["R1"] - clean["R1"])
        for name in DEGRADED_SCENARIOS
    }
    delta_map = {
        name: abs(scenario_metrics[name]["mAP"] - clean["mAP"])
        for name in DEGRADED_SCENARIOS
    }
    without_rgb_missing = [
        value for name, value in delta_r1.items() if name != "rgb_missing"
    ]
    worst_name = min(DEGRADED_SCENARIOS, key=lambda name: scenario_metrics[name]["R1"])
    seen_names = [name for name in DEGRADED_SCENARIOS if seen_flags[name]]
    unseen_names = [name for name in DEGRADED_SCENARIOS if not seen_flags[name]]
    if not seen_names or not unseen_names:
        raise ValueError(
            "Manifest must label at least one degraded scenario seen_in_training=true "
            "and at least one false scenario."
        )

    per_scenario = {}
    for name in REQUIRED_SCENARIOS:
        row = dict(scenario_metrics[name])
        row["delta_R1_from_clean"] = row["R1"] - clean["R1"]
        row["delta_mAP_from_clean"] = row["mAP"] - clean["mAP"]
        row["abs_delta_R1"] = abs(row["delta_R1_from_clean"])
        row["abs_delta_mAP"] = abs(row["delta_mAP_from_clean"])
        row["seen_in_training"] = None if name == CLEAN_SCENARIO else seen_flags[name]
        per_scenario[name] = row

    return {
        "model_name": str(model_name),
        "scenario_count": len(REQUIRED_SCENARIOS),
        "degraded_scenario_count": len(DEGRADED_SCENARIOS),
        "clean_R1": clean["R1"],
        "clean_mAP": clean["mAP"],
        "degraded_mean_R1": degraded_mean_r1,
        "degraded_mean_mAP": degraded_mean_map,
        "mean_abs_delta_R1": statistics.fmean(delta_r1.values()),
        "mean_abs_delta_mAP": statistics.fmean(delta_map.values()),
        "mean_abs_delta_R1_excluding_rgb_missing": statistics.fmean(
            without_rgb_missing
        ),
        "worst_scenario": worst_name,
        "worst_scenario_R1": scenario_metrics[worst_name]["R1"],
        "worst_scenario_mAP": scenario_metrics[worst_name]["mAP"],
        "seen_degraded_scenarios": seen_names,
        "unseen_degraded_scenarios": unseen_names,
        "seen_degraded_mean_R1": statistics.fmean(
            scenario_metrics[name]["R1"] for name in seen_names
        ),
        "seen_degraded_mean_mAP": statistics.fmean(
            scenario_metrics[name]["mAP"] for name in seen_names
        ),
        "unseen_degraded_mean_R1": statistics.fmean(
            scenario_metrics[name]["R1"] for name in unseen_names
        ),
        "unseen_degraded_mean_mAP": statistics.fmean(
            scenario_metrics[name]["mAP"] for name in unseen_names
        ),
        "pareto": {"x_clean_R1": clean["R1"], "y_degraded_mean_R1": degraded_mean_r1},
        "per_scenario": per_scenario,
    }


def summarize_manifest(manifest, manifest_path, matrix_key="sim"):
    scenarios = validate_robustness_manifest(manifest)
    items = []
    metrics = {}
    seen_flags = {}
    input_files = {}
    for name in REQUIRED_SCENARIOS:
        record = scenarios[name]
        record_matrix_key = str(record.get("matrix_key", matrix_key))
        score_file = resolve_manifest_path(manifest_path, record["score_file"])
        item = load_score_file(score_file, matrix_key=record_matrix_key)
        items.append(item)
        input_files[name] = item["path"]
        metrics[name] = compute_retrieval_metrics(
            item["matrix"], item["txt_pids"], item["img_pids"]
        )
        if name != CLEAN_SCENARIO:
            seen_flags[name] = record["seen_in_training"]
    require_score_consistency(items)
    summary = summarize_metrics(manifest.get("model_name", ""), metrics, seen_flags)
    summary["input_files"] = input_files
    summary["manifest_file"] = manifest_path
    return summary


def _markdown(summary):
    lines = [
        f"# Robustness summary: {summary['model_name']}",
        "",
        "| Scenario | Seen in training | R1 | R5 | R10 | mAP | mINP | Abs ΔR1 | Abs ΔmAP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in REQUIRED_SCENARIOS:
        row = summary["per_scenario"][name]
        seen = "clean" if name == CLEAN_SCENARIO else str(row["seen_in_training"]).lower()
        lines.append(
            f"| {name} | {seen} | {row['R1']:.6f} | {row['R5']:.6f} | "
            f"{row['R10']:.6f} | {row['mAP']:.6f} | {row['mINP']:.6f} | "
            f"{row['abs_delta_R1']:.6f} | {row['abs_delta_mAP']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Clean R1: {summary['clean_R1']:.6f}",
            f"- Clean mAP: {summary['clean_mAP']:.6f}",
            f"- Ten-degradation mean R1: {summary['degraded_mean_R1']:.6f}",
            f"- Ten-degradation mean mAP: {summary['degraded_mean_mAP']:.6f}",
            f"- Mean |ΔR1|: {summary['mean_abs_delta_R1']:.6f}",
            f"- Mean |ΔmAP|: {summary['mean_abs_delta_mAP']:.6f}",
            f"- Worst scenario: `{summary['worst_scenario']}`",
        ]
    )
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize clean plus ten strictly named robustness scenarios."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--matrix-key", choices=["sim", "raw_sim"], default="sim")
    parser.add_argument("--output-prefix", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest, manifest_path = load_json_manifest(args.manifest)
    summary = summarize_manifest(manifest, manifest_path, matrix_key=args.matrix_key)
    paths = report_paths(args.output_prefix)
    require_outputs_absent(paths.values())
    prepare_output_parents(paths.values())
    write_json(paths["json"], summary)
    rows = [
        {"scenario": name, **summary["per_scenario"][name]}
        for name in REQUIRED_SCENARIOS
    ]
    fields = [
        "scenario",
        "seen_in_training",
        *METRIC_KEYS,
        "delta_R1_from_clean",
        "delta_mAP_from_clean",
        "abs_delta_R1",
        "abs_delta_mAP",
    ]
    write_tsv(paths["tsv"], rows, fields)
    write_markdown(paths["markdown"], _markdown(summary))
    print(f"Saved robustness summary: {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
