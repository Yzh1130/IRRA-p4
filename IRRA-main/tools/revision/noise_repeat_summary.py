"""Summarize repeated stochastic-degradation evaluations across three seeds."""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.revision.score_io import (  # noqa: E402
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


DEFAULT_EXPECTED_SEEDS = (2026, 2027, 2028)


def summarize_noise_groups(records, expected_seeds=DEFAULT_EXPECTED_SEEDS):
    expected_seeds = tuple(sorted(int(seed) for seed in expected_seeds))
    grouped = defaultdict(list)
    for record in records:
        grouped[(str(record["model"]), str(record["scenario"]))].append(record)
    summaries = []
    for (model, scenario), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["degrade_seed"]))
        seeds = [int(row["degrade_seed"]) for row in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate degradation seed for {model}/{scenario}: {seeds}")
        r1_values = [float(row["R1"]) for row in rows]
        map_values = [float(row["mAP"]) for row in rows]
        summaries.append(
            {
                "model": model,
                "scenario": scenario,
                "n": len(rows),
                "incomplete": tuple(seeds) != expected_seeds,
                "expected_seeds": list(expected_seeds),
                "observed_seeds": seeds,
                "R1_mean": statistics.fmean(r1_values),
                "R1_sample_std": (
                    statistics.stdev(r1_values) if len(r1_values) >= 2 else None
                ),
                "R1_min": min(r1_values),
                "R1_max": max(r1_values),
                "mAP_mean": statistics.fmean(map_values),
                "mAP_sample_std": (
                    statistics.stdev(map_values) if len(map_values) >= 2 else None
                ),
                "raw_values": [
                    {
                        "degrade_seed": int(row["degrade_seed"]),
                        "R1": float(row["R1"]),
                        "mAP": float(row["mAP"]),
                    }
                    for row in rows
                ],
            }
        )
    return summaries


def load_records(manifest, manifest_path, default_matrix_key="sim"):
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Noise-repeat manifest requires a non-empty 'records' list.")
    records = []
    score_items = []
    for raw in raw_records:
        for key in ("model", "scenario", "degrade_seed", "score_file"):
            if key not in raw:
                raise KeyError(f"Noise-repeat record is missing {key!r}: {raw}")
        item = load_score_file(
            resolve_manifest_path(manifest_path, raw["score_file"]),
            matrix_key=str(raw.get("matrix_key", default_matrix_key)),
        )
        score_items.append(item)
        metrics = compute_retrieval_metrics(
            item["matrix"], item["txt_pids"], item["img_pids"]
        )
        records.append(
            {
                "model": str(raw["model"]),
                "scenario": str(raw["scenario"]),
                "degrade_seed": int(raw["degrade_seed"]),
                "R1": metrics["R1"],
                "mAP": metrics["mAP"],
                "score_file": item["path"],
            }
        )
    if len(score_items) > 1:
        require_score_consistency(score_items)
    return records


def _markdown(summaries):
    lines = [
        "# Noise degradation repeat summary",
        "",
        "| Model | Scenario | n | Incomplete | R1 mean | R1 SD | R1 min | R1 max | mAP mean | mAP SD |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        r1_sd = "NA" if row["R1_sample_std"] is None else f"{row['R1_sample_std']:.6f}"
        map_sd = "NA" if row["mAP_sample_std"] is None else f"{row['mAP_sample_std']:.6f}"
        lines.append(
            f"| {row['model']} | {row['scenario']} | {row['n']} | "
            f"{str(row['incomplete']).lower()} | {row['R1_mean']:.6f} | {r1_sd} | "
            f"{row['R1_min']:.6f} | {row['R1_max']:.6f} | "
            f"{row['mAP_mean']:.6f} | {map_sd} |"
        )
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize stochastic degradation repeats for seeds 2026/2027/2028."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--matrix-key", choices=["sim", "raw_sim"], default="sim")
    parser.add_argument("--output-prefix", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest, manifest_path = load_json_manifest(args.manifest)
    expected_seeds = manifest.get("expected_seeds", list(DEFAULT_EXPECTED_SEEDS))
    records = load_records(manifest, manifest_path, default_matrix_key=args.matrix_key)
    summaries = summarize_noise_groups(records, expected_seeds=expected_seeds)
    result = {
        "manifest_file": manifest_path,
        "expected_seeds": [int(seed) for seed in expected_seeds],
        "groups": summaries,
    }
    paths = report_paths(args.output_prefix)
    require_outputs_absent(paths.values())
    prepare_output_parents(paths.values())
    write_json(paths["json"], result)
    fields = [
        "model",
        "scenario",
        "n",
        "incomplete",
        "observed_seeds",
        "R1_mean",
        "R1_sample_std",
        "R1_min",
        "R1_max",
        "mAP_mean",
        "mAP_sample_std",
        "raw_values",
    ]
    write_tsv(paths["tsv"], summaries, fields)
    write_markdown(paths["markdown"], _markdown(summaries))
    print(f"Saved noise-repeat summary: {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
