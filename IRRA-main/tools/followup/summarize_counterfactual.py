#!/usr/bin/env python3
"""Summarize seed-paired gate counterfactual results.

The input is the ``metrics_counterfactual.tsv`` emitted by
``gate_counterfactual_eval.py``.  All comparisons are within the same model,
seed, and cached degradation scenario: an alternative fusion mode minus the
learned gate.  This script never modifies the input file.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# These implementations are the repository's frozen statistical definitions.
from tools.stats.paired_stats import T975, holm, wilcoxon  # noqa: E402


CORE_SCENARIOS = (
    "clean",
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
NONCLEAN_SCENARIOS = tuple(name for name in CORE_SCENARIOS if name != "clean")
RGB_SCENARIOS = tuple(name for name in NONCLEAN_SCENARIOS if name.startswith("rgb_"))
IR_SCENARIOS = tuple(name for name in NONCLEAN_SCENARIOS if name.startswith("ir_"))
REQUIRED_MODES = ("learned", "equal", "calib", "rgb_only", "ir_only")
METRICS = ("R1", "R5", "R10", "mAP", "mINP")
COMPARISONS = (("equal-learned", "equal"), ("calib-learned", "calib"))
ENDPOINTS = ("P1", "P2", "clean", "ir_group", "rgb_group", "all10")
PRIMARY_ENDPOINTS = ("P1", "P2")
REQUIRED_COLUMNS = ("group", "seed", "scenario", "mode", *METRICS)

# Two-sided t(0.975, n-1), keyed by sample size.  Formal analyses use n=10;
# the additional entries make small synthetic checks and incomplete previews
# explicit without adding a scipy dependency.  The n=5 and n=10 values are
# overwritten by the values used in tools.stats.paired_stats.
_T975_BY_N = {
    2: 12.706205,
    3: 4.302653,
    4: 3.182446,
    5: 2.776445,
    6: 2.570582,
    7: 2.446912,
    8: 2.364624,
    9: 2.306004,
    10: 2.262157,
    11: 2.228139,
    12: 2.200985,
    13: 2.178813,
    14: 2.160369,
    15: 2.144787,
    16: 2.131450,
    17: 2.119905,
    18: 2.109816,
    19: 2.100922,
    20: 2.093024,
    21: 2.085963,
    22: 2.079614,
    23: 2.073873,
    24: 2.068658,
    25: 2.063899,
    26: 2.059539,
    27: 2.055529,
    28: 2.051831,
    29: 2.048407,
    30: 2.045230,
}
_T975_BY_N.update(T975)


class CounterfactualSummaryError(ValueError):
    """Raised when an input table cannot support the frozen analysis."""


def _numeric_seed(value: Any) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CounterfactualSummaryError(f"invalid seed: {value!r}") from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise CounterfactualSummaryError(f"seed must be an integer, got {value!r}")
    return int(parsed)


def _finite_metric(value: Any, *, name: str, key: tuple[str, int, str, str]) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CounterfactualSummaryError(
            f"{key}: metric {name} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise CounterfactualSummaryError(
            f"{key}: metric {name} must be finite, got {value!r}"
        )
    return parsed


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate a counterfactual metrics TSV without modifying it."""

    metrics_path = Path(path)
    if not metrics_path.is_file():
        raise CounterfactualSummaryError(f"metrics TSV does not exist: {metrics_path}")

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing_columns = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or ())]
        if missing_columns:
            raise CounterfactualSummaryError(
                f"{metrics_path}: missing columns {missing_columns}"
            )
        for line_number, raw in enumerate(reader, start=2):
            group = (raw.get("group") or "").strip()
            scenario = (raw.get("scenario") or "").strip()
            mode = (raw.get("mode") or "").strip()
            if not group:
                raise CounterfactualSummaryError(
                    f"{metrics_path}:{line_number}: group is empty"
                )
            seed = _numeric_seed(raw.get("seed"))
            key = (group, seed, scenario, mode)
            if key in seen:
                raise CounterfactualSummaryError(
                    f"{metrics_path}:{line_number}: duplicate row for {key}"
                )
            seen.add(key)
            row: dict[str, Any] = {
                "group": group,
                "seed": seed,
                "scenario": scenario,
                "mode": mode,
            }
            row.update(
                {
                    name: _finite_metric(raw.get(name), name=name, key=key)
                    for name in METRICS
                }
            )
            records.append(row)

    if not records:
        raise CounterfactualSummaryError(f"metrics TSV is empty: {metrics_path}")
    return records


def _scenario_r1_values(
    scenario_rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    """Normalize either a scenario mapping or in-memory metric rows."""

    if isinstance(scenario_rows, Mapping):
        values: dict[str, float] = {}
        for scenario, raw_value in scenario_rows.items():
            value = raw_value.get("R1") if isinstance(raw_value, Mapping) else raw_value
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise CounterfactualSummaryError(
                    f"scenario {scenario!r} has invalid R1: {value!r}"
                ) from exc
            if not math.isfinite(parsed):
                raise CounterfactualSummaryError(
                    f"scenario {scenario!r} has non-finite R1: {value!r}"
                )
            values[str(scenario)] = parsed
        return values

    values = {}
    for row in scenario_rows:
        if "scenario" not in row or "R1" not in row:
            raise CounterfactualSummaryError(
                "each in-memory row must contain scenario and R1"
            )
        scenario = str(row["scenario"])
        if scenario in values:
            raise CounterfactualSummaryError(
                f"duplicate scenario in one seed/mode endpoint calculation: {scenario}"
            )
        try:
            parsed = float(row["R1"])
        except (TypeError, ValueError) as exc:
            raise CounterfactualSummaryError(
                f"scenario {scenario!r} has invalid R1: {row['R1']!r}"
            ) from exc
        if not math.isfinite(parsed):
            raise CounterfactualSummaryError(
                f"scenario {scenario!r} has non-finite R1: {row['R1']!r}"
            )
        values[scenario] = parsed
    return values


def compute_p1_p2(
    scenario_rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    """Compute the two frozen R1 endpoints for exactly one seed and mode.

    P1 is the arithmetic mean of only ``rgb_noise_005`` and
    ``rgb_noise_010``.  P2 is exactly ``rgb_blur_k7``.  Missing inputs fail
    closed, preventing accidental averaging across seeds, modes, or scenarios.
    """

    values = _scenario_r1_values(scenario_rows)
    required = ("rgb_noise_005", "rgb_noise_010", "rgb_blur_k7")
    missing = [name for name in required if name not in values]
    if missing:
        raise CounterfactualSummaryError(
            f"cannot compute P1/P2; missing scenarios {missing}"
        )
    return {
        "P1": (values["rgb_noise_005"] + values["rgb_noise_010"]) / 2.0,
        "P2": values["rgb_blur_k7"],
    }


def compute_seed_endpoints(
    scenario_rows: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> dict[str, float]:
    """Compute P1/P2 and the frozen descriptive R1 aggregates."""

    values = _scenario_r1_values(scenario_rows)
    missing = [name for name in CORE_SCENARIOS if name not in values]
    if missing:
        raise CounterfactualSummaryError(
            f"cannot compute seed endpoints; missing scenarios {missing}"
        )
    endpoints = compute_p1_p2(values)
    endpoints.update(
        {
            "clean": values["clean"],
            "ir_group": statistics.fmean(values[name] for name in IR_SCENARIOS),
            "rgb_group": statistics.fmean(values[name] for name in RGB_SCENARIOS),
            "all10": statistics.fmean(values[name] for name in NONCLEAN_SCENARIOS),
        }
    )
    return endpoints


def _index_records(
    records: Sequence[Mapping[str, Any]],
    groups: Sequence[str] | None,
) -> tuple[dict[tuple[str, int, str, str], dict[str, Any]], list[str], dict[str, list[int]]]:
    requested = None if groups is None else {group.strip() for group in groups if group.strip()}
    if requested == set():
        raise CounterfactualSummaryError("--groups selected no groups")

    index: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    seeds_by_group: dict[str, set[int]] = {}
    for source in records:
        group = str(source["group"])
        if requested is not None and group not in requested:
            continue
        seed = _numeric_seed(source["seed"])
        scenario = str(source["scenario"])
        mode = str(source["mode"])
        key = (group, seed, scenario, mode)
        if key in index:
            raise CounterfactualSummaryError(f"duplicate row for {key}")
        row = {"group": group, "seed": seed, "scenario": scenario, "mode": mode}
        row.update({name: float(source[name]) for name in METRICS})
        index[key] = row
        seeds_by_group.setdefault(group, set()).add(seed)

    available_groups = sorted(seeds_by_group)
    if requested is not None:
        missing_groups = sorted(requested - set(available_groups))
        if missing_groups:
            raise CounterfactualSummaryError(
                f"requested groups are absent from metrics: {missing_groups}"
            )
    if not available_groups:
        raise CounterfactualSummaryError("no rows remain after group selection")

    normalized_seeds = {group: sorted(seeds_by_group[group]) for group in available_groups}
    for group in available_groups:
        for seed in normalized_seeds[group]:
            missing = [
                (scenario, mode)
                for scenario in CORE_SCENARIOS
                for mode in REQUIRED_MODES
                if (group, seed, scenario, mode) not in index
            ]
            if missing:
                raise CounterfactualSummaryError(
                    f"{group} seed {seed}: missing scenario/mode rows {missing}"
                )
    return index, available_groups, normalized_seeds


def _endpoints_for(
    index: Mapping[tuple[str, int, str, str], Mapping[str, Any]],
    group: str,
    seed: int,
    mode: str,
) -> dict[str, float]:
    rows = [index[(group, seed, scenario, mode)] for scenario in CORE_SCENARIOS]
    return compute_seed_endpoints(rows)


def _paired_statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise CounterfactualSummaryError("cannot summarize an empty paired vector")
    n = len(values)
    mean = statistics.fmean(values)
    improved = sum(value > 0 for value in values)
    raw_p = wilcoxon(list(values))
    if n == 1:
        return {
            "n": n,
            "mean": mean,
            "sd": None,
            "ci_low": None,
            "ci_high": None,
            "improved": improved,
            "p": raw_p,
        }
    if n not in _T975_BY_N:
        raise CounterfactualSummaryError(
            f"no frozen t(0.975, n-1) critical value for n={n}; supported n=1..30"
        )
    sd = statistics.stdev(values)
    half_width = _T975_BY_N[n] * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "improved": improved,
        "p": raw_p,
    }


def build_summary_tables(
    records: Sequence[Mapping[str, Any]],
    groups: Sequence[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build all output tables in memory for testing and CLI serialization."""

    index, selected_groups, seeds_by_group = _index_records(records, groups)
    difference_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    input_zero_rows: list[dict[str, Any]] = []

    for group in selected_groups:
        seeds = seeds_by_group[group]
        endpoint_cache = {
            (seed, mode): _endpoints_for(index, group, seed, mode)
            for seed in seeds
            for mode in ("learned", "equal", "calib")
        }
        group_summary_indices: dict[tuple[str, str], int] = {}
        for comparison, alternative_mode in COMPARISONS:
            for endpoint in ENDPOINTS:
                values = []
                for seed in seeds:
                    difference = (
                        endpoint_cache[(seed, alternative_mode)][endpoint]
                        - endpoint_cache[(seed, "learned")][endpoint]
                    )
                    values.append(difference)
                    difference_rows.append(
                        {
                            "group": group,
                            "comparison": comparison,
                            "endpoint": endpoint,
                            "seed": seed,
                            "difference": difference,
                        }
                    )

                stats = _paired_statistics(values)
                is_endpoint_test = endpoint in PRIMARY_ENDPOINTS
                is_primary = (
                    group == "D"
                    and comparison == "equal-learned"
                    and is_endpoint_test
                )
                row = {
                    "group": group,
                    "comparison": comparison,
                    "endpoint": endpoint,
                    "role": (
                        "primary"
                        if is_primary
                        else "secondary"
                        if is_endpoint_test
                        else "descriptive"
                    ),
                    "n": stats["n"],
                    "mean_difference": stats["mean"],
                    "sd": stats["sd"],
                    "mean_pm_sd": (
                        ""
                        if stats["sd"] is None
                        else f"{stats['mean']:+.6f}±{stats['sd']:.6f}"
                    ),
                    "ci95_low": stats["ci_low"],
                    "ci95_high": stats["ci_high"],
                    "improved": stats["improved"],
                    "wilcoxon_p": stats["p"] if is_endpoint_test else None,
                    "holm_p": None,
                }
                group_summary_indices[(comparison, endpoint)] = len(summary_rows)
                summary_rows.append(row)

        if group == "D":
            primary_indices = [
                group_summary_indices[("equal-learned", endpoint)]
                for endpoint in PRIMARY_ENDPOINTS
            ]
            adjusted = holm(
                [float(summary_rows[i]["wilcoxon_p"]) for i in primary_indices]
            )
            for index_number, adjusted_p in zip(primary_indices, adjusted):
                summary_rows[index_number]["holm_p"] = adjusted_p

        for seed in seeds:
            for scenario in CORE_SCENARIOS:
                row: dict[str, Any] = {
                    "group": group,
                    "seed": seed,
                    "scenario": scenario,
                }
                for mode in ("rgb_only", "ir_only"):
                    source = index[(group, seed, scenario, mode)]
                    for metric in METRICS:
                        row[f"{mode}_{metric}"] = source[metric]
                single_rows.append(row)

            for scenario, remaining_modality in (
                ("rgb_missing", "ir"),
                ("ir_missing", "rgb"),
            ):
                source = index[(group, seed, scenario, "learned")]
                input_zero_rows.append(
                    {
                        "group": group,
                        "seed": seed,
                        "scenario": scenario,
                        "remaining_modality": remaining_modality,
                        **{metric: source[metric] for metric in METRICS},
                    }
                )

    return {
        "paired_differences": difference_rows,
        "summary": summary_rows,
        "single_modality": single_rows,
        "input_zero": input_zero_rows,
    }


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _write_tsv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(fieldnames),
                delimiter="\t",
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({name: _format_cell(row.get(name)) for name in fieldnames})
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_summary_tables(
    tables: Mapping[str, Sequence[Mapping[str, Any]]], out_dir: str | Path
) -> list[Path]:
    """Atomically write the four analysis tables and return their paths."""

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    specifications = (
        (
            "counterfactual_paired_differences.tsv",
            ("group", "comparison", "endpoint", "seed", "difference"),
            "paired_differences",
        ),
        (
            "counterfactual_summary.tsv",
            (
                "group",
                "comparison",
                "endpoint",
                "role",
                "n",
                "mean_difference",
                "sd",
                "mean_pm_sd",
                "ci95_low",
                "ci95_high",
                "improved",
                "wilcoxon_p",
                "holm_p",
            ),
            "summary",
        ),
        (
            "single_modality_by_scenario.tsv",
            (
                "group",
                "seed",
                "scenario",
                *(f"rgb_only_{metric}" for metric in METRICS),
                *(f"ir_only_{metric}" for metric in METRICS),
            ),
            "single_modality",
        ),
        (
            "input_zero_learned.tsv",
            ("group", "seed", "scenario", "remaining_modality", *METRICS),
            "input_zero",
        ),
    )
    paths = []
    for filename, fieldnames, table_name in specifications:
        path = destination / filename
        _write_tsv(path, fieldnames, tables[table_name])
        paths.append(path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize fixed-model MRAF gate counterfactual metrics."
    )
    parser.add_argument(
        "--metrics",
        "--input",
        dest="metrics",
        required=True,
        help="metrics_counterfactual.tsv produced by gate_counterfactual_eval.py",
    )
    parser.add_argument(
        "--out",
        help="output directory (default: the metrics TSV directory)",
    )
    parser.add_argument(
        "--groups",
        help="optional comma-separated group subset; by default all groups are summarized",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics_path = Path(args.metrics).resolve()
    groups = None if not args.groups else [item.strip() for item in args.groups.split(",")]
    try:
        records = read_metrics(metrics_path)
        tables = build_summary_tables(records, groups=groups)
        output_paths = write_summary_tables(tables, args.out or metrics_path.parent)
    except (CounterfactualSummaryError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summarized_groups = sorted({row["group"] for row in tables["summary"]})
    print(
        "[COUNTERFACTUAL_SUMMARY] "
        f"input={metrics_path} groups={','.join(summarized_groups)}"
    )
    for path in output_paths:
        print(f"[OUTPUT] {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
