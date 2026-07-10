"""Paired Base/MDRC statistics across exactly five seeds."""

import argparse
import math
import os
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scipy import stats
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "paired_seed_statistics.py requires scipy; no approximate fallback is used."
    ) from exc

from tools.revision.score_io import (  # noqa: E402
    load_json_manifest,
    prepare_output_parents,
    report_paths,
    require_outputs_absent,
    resolve_manifest_path,
    write_json,
    write_markdown,
    write_tsv,
)


SUMMARY_KEYS = (
    "clean_R1",
    "degraded_mean_R1",
    "mean_abs_delta_R1",
    "clean_mAP",
    "degraded_mean_mAP",
)
PRIMARY_KEY = "mean_abs_delta_R1"


def _read_summary(path):
    payload, absolute_path = load_json_manifest(path)
    missing = [key for key in SUMMARY_KEYS if key not in payload]
    if missing:
        raise KeyError(f"Robustness summary {absolute_path} is missing: {missing}")
    return payload, absolute_path


def _descriptive(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "formatted_mean_plus_minus_std": (
            f"{statistics.fmean(values):.6f} ± {statistics.stdev(values):.6f}"
        ),
        "values": values,
    }


def _student_t_ci_95(values):
    values = [float(value) for value in values]
    n = len(values)
    mean_value = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    critical = float(stats.t.ppf(0.975, df=n - 1))
    half_width = critical * sample_std / math.sqrt(n)
    return {
        "mean": mean_value,
        "low": mean_value - half_width,
        "high": mean_value + half_width,
        "confidence_level": 0.95,
        "distribution": "Student t",
        "degrees_of_freedom": n - 1,
    }


def compute_paired_statistics(pairs):
    if len(pairs) != 5:
        raise ValueError(f"Exactly five paired seeds are required, got n={len(pairs)}.")
    seeds = [int(pair["seed"]) for pair in pairs]
    if len(set(seeds)) != 5:
        raise ValueError(f"Seed identifiers must be unique, got {seeds}.")

    per_seed = []
    base_values = {key: [] for key in SUMMARY_KEYS}
    mdrc_values = {key: [] for key in SUMMARY_KEYS}
    differences = {key: [] for key in SUMMARY_KEYS}
    for pair in sorted(pairs, key=lambda item: int(item["seed"])):
        base = pair["base"]
        mdrc = pair["mdrc"]
        row = {"seed": int(pair["seed"])}
        for key in SUMMARY_KEYS:
            base_value = float(base[key])
            mdrc_value = float(mdrc[key])
            difference = mdrc_value - base_value
            base_values[key].append(base_value)
            mdrc_values[key].append(mdrc_value)
            differences[key].append(difference)
            row[f"base_{key}"] = base_value
            row[f"mdrc_{key}"] = mdrc_value
            row[f"difference_mdrc_minus_base_{key}"] = difference
        per_seed.append(row)

    primary_base = base_values[PRIMARY_KEY]
    primary_mdrc = mdrc_values[PRIMARY_KEY]
    one_sided = stats.wilcoxon(
        primary_mdrc,
        primary_base,
        alternative="less",
        zero_method="wilcox",
        method="auto",
    )
    two_sided = stats.wilcoxon(
        primary_mdrc,
        primary_base,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return {
        "n": 5,
        "small_sample_limitation": (
            "n=5 provides limited power and discrete Wilcoxon p-values; report "
            "effect sizes and per-seed differences alongside inferential statistics."
        ),
        "difference_definition": "MDRC minus Base",
        "base": {key: _descriptive(base_values[key]) for key in SUMMARY_KEYS},
        "mdrc": {key: _descriptive(mdrc_values[key]) for key in SUMMARY_KEYS},
        "paired_differences": {
            key: {
                **_descriptive(differences[key]),
                "student_t_95_ci": _student_t_ci_95(differences[key]),
            }
            for key in SUMMARY_KEYS
        },
        "primary_hypothesis_metric": PRIMARY_KEY,
        "wilcoxon_one_sided": {
            "alternative": "MDRC mean |ΔR1| is lower than Base",
            "scipy_alternative": "less",
            "statistic": float(one_sided.statistic),
            "pvalue": float(one_sided.pvalue),
        },
        "wilcoxon_two_sided": {
            "alternative": "MDRC and Base mean |ΔR1| differ",
            "scipy_alternative": "two-sided",
            "statistic": float(two_sided.statistic),
            "pvalue": float(two_sided.pvalue),
        },
        "per_seed": per_seed,
    }


def load_pairs_from_manifest(manifest, manifest_path):
    raw_pairs = manifest.get("pairs")
    if not isinstance(raw_pairs, list):
        raise TypeError("Paired-seed manifest requires a 'pairs' list.")
    pairs = []
    for record in raw_pairs:
        if not isinstance(record, dict):
            raise TypeError("Each paired-seed record must be a JSON object.")
        for key in ("seed", "base_summary", "mdrc_summary"):
            if key not in record:
                raise KeyError(f"Paired-seed record is missing {key!r}: {record}")
        base_path = resolve_manifest_path(manifest_path, record["base_summary"])
        mdrc_path = resolve_manifest_path(manifest_path, record["mdrc_summary"])
        base, _ = _read_summary(base_path)
        mdrc, _ = _read_summary(mdrc_path)
        pairs.append({"seed": int(record["seed"]), "base": base, "mdrc": mdrc})
    return pairs


def _markdown(result):
    lines = [
        "# Five-seed paired statistics",
        "",
        f"- n: {result['n']}",
        f"- Limitation: {result['small_sample_limitation']}",
        "",
        "| Metric | Base mean ± SD | MDRC mean ± SD | Mean paired difference | 95% t CI |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for key in SUMMARY_KEYS:
        ci = result["paired_differences"][key]["student_t_95_ci"]
        lines.append(
            f"| {key} | {result['base'][key]['formatted_mean_plus_minus_std']} | "
            f"{result['mdrc'][key]['formatted_mean_plus_minus_std']} | "
            f"{result['paired_differences'][key]['mean']:.6f} | "
            f"[{ci['low']:.6f}, {ci['high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Wilcoxon on mean |ΔR1|",
            "",
            f"- One-sided p: {result['wilcoxon_one_sided']['pvalue']:.12g}",
            f"- Two-sided p: {result['wilcoxon_two_sided']['pvalue']:.12g}",
        ]
    )
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute paired Base/MDRC statistics for exactly five seeds."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest, manifest_path = load_json_manifest(args.manifest)
    result = compute_paired_statistics(load_pairs_from_manifest(manifest, manifest_path))
    result["manifest_file"] = manifest_path
    paths = report_paths(args.output_prefix)
    require_outputs_absent(paths.values())
    prepare_output_parents(paths.values())
    write_json(paths["json"], result)
    fields = ["seed"]
    for key in SUMMARY_KEYS:
        fields.extend(
            [
                f"base_{key}",
                f"mdrc_{key}",
                f"difference_mdrc_minus_base_{key}",
            ]
        )
    write_tsv(paths["tsv"], result["per_seed"], fields)
    write_markdown(paths["markdown"], _markdown(result))
    print(f"Saved paired-seed statistics: {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
