#!/usr/bin/env python3
"""Freeze Stage 2 MRAF hyperparameters from the locked development grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("devA", "devT010", "devT020", "devT030", "devT050")
SEEDS = (1, 2, 3)
METRIC_NAMES = ("R1", "R5", "R10", "mAP", "mINP")
METRICS_HEADER = (
    "group",
    "seed",
    "scenario",
    "mode",
    "strength",
    "kernel",
    "dseed",
    "split",
    *METRIC_NAMES,
)
FOUR_DECIMALS = Decimal("0.0001")
TIE_THRESHOLD = Decimal("0.3000")
E_CANDIDATE_TEMPERATURES = (
    Decimal("0.2"),
    Decimal("0.3"),
    Decimal("0.5"),
)


@dataclass(frozen=True)
class ScenarioSpec:
    mode: str
    strength: Decimal
    kernel: int
    degradation_seed: int


SCENARIO_SPECS: dict[str, ScenarioSpec] = {
    "clean": ScenarioSpec("none", Decimal("1.0"), 7, 2026),
    "rgb_missing": ScenarioSpec("rgb_missing", Decimal("1.0"), 7, 2026),
    "ir_missing": ScenarioSpec("ir_missing", Decimal("1.0"), 7, 2026),
    "rgb_noise_005": ScenarioSpec("rgb_noise", Decimal("0.05"), 7, 2026),
    "rgb_noise_010": ScenarioSpec("rgb_noise", Decimal("0.10"), 7, 2026),
    "ir_noise_005": ScenarioSpec("ir_noise", Decimal("0.05"), 7, 2026),
    "ir_noise_010": ScenarioSpec("ir_noise", Decimal("0.10"), 7, 2026),
    "rgb_blur_k7": ScenarioSpec("rgb_blur", Decimal("1.0"), 7, 2026),
    "ir_blur_k7": ScenarioSpec("ir_blur", Decimal("1.0"), 7, 2026),
    "both_noise_005": ScenarioSpec("both_noise", Decimal("0.05"), 7, 2026),
    "both_blur_k7": ScenarioSpec("both_blur", Decimal("1.0"), 7, 2026),
}
DEGRADED_SCENARIOS = tuple(
    scenario for scenario in SCENARIO_SPECS if scenario != "clean"
)
TEMPERATURE_GROUPS = {
    Decimal("0.1"): "devT010",
    Decimal("0.2"): "devT020",
    Decimal("0.3"): "devT030",
    Decimal("0.5"): "devT050",
}


@dataclass(frozen=True)
class MetricRow:
    group: str
    seed: int
    scenario: str
    mode: str
    strength: Decimal
    kernel: int
    degradation_seed: int
    split: str
    metrics: Mapping[str, Decimal]

    @property
    def r1(self) -> Decimal:
        return self.metrics["R1"]


class FreezeValidationError(ValueError):
    """Raised when development evidence violates the frozen protocol."""


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _read_bytes(path: Path, label: str) -> bytes:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FreezeValidationError(f"{label} does not exist: {resolved}")
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise FreezeValidationError(f"Failed to read {label} {resolved}: {exc}") from exc


def _decode_utf8(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FreezeValidationError(f"{label} must be UTF-8: {exc}") from exc


def _parse_decimal(value: Any, label: str) -> Decimal:
    text = str(value).strip()
    if not text:
        raise FreezeValidationError(f"{label} must not be empty.")
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise FreezeValidationError(f"{label} must be a decimal number, got {value!r}.") from exc
    if not parsed.is_finite():
        raise FreezeValidationError(f"{label} must be finite, got {value!r}.")
    return parsed


def _parse_integer(value: Any, label: str) -> int:
    text = str(value).strip()
    try:
        parsed = int(text)
    except ValueError as exc:
        raise FreezeValidationError(f"{label} must be an integer, got {value!r}.") from exc
    if str(parsed) != text:
        raise FreezeValidationError(f"{label} must be an integer, got {value!r}.")
    return parsed


def _mean(values: Sequence[Decimal], label: str) -> Decimal:
    if not values:
        raise FreezeValidationError(f"Cannot compute an empty mean for {label}.")
    return sum(values, Decimal("0")) / Decimal(len(values))


def _sample_std(values: Sequence[Decimal], label: str) -> Decimal:
    if len(values) < 2:
        raise FreezeValidationError(
            f"At least two values are required for sample SD of {label}."
        )
    center = _mean(values, label)
    variance = sum((value - center) ** 2 for value in values) / Decimal(
        len(values) - 1
    )
    return variance.sqrt()


def _round4(value: Decimal) -> Decimal:
    return value.quantize(FOUR_DECIMALS, rounding=ROUND_HALF_UP)


def _temperature_key(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _validate_metric_value(value: Any, label: str) -> Decimal:
    parsed = _parse_decimal(value, label)
    if not Decimal("0") <= parsed <= Decimal("100"):
        raise FreezeValidationError(f"{label} must be in [0, 100], got {parsed}.")
    return parsed


def load_dev_metrics(path: Path) -> tuple[dict[tuple[str, int, str], MetricRow], str]:
    """Load and strictly validate the complete 5 x 3 x 11 val-only grid."""
    raw = _read_bytes(path, "development metrics TSV")
    text = _decode_utf8(raw, "development metrics TSV")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if tuple(reader.fieldnames or ()) != METRICS_HEADER:
        raise FreezeValidationError(
            "metrics.tsv header mismatch: expected "
            f"{list(METRICS_HEADER)}, got {reader.fieldnames}."
        )

    rows: dict[tuple[str, int, str], MetricRow] = {}
    for line_number, raw_row in enumerate(reader, start=2):
        group = str(raw_row["group"]).strip()
        if group not in GROUPS:
            raise FreezeValidationError(
                f"metrics.tsv:{line_number} has unknown group {group!r}."
            )
        seed = _parse_integer(raw_row["seed"], f"metrics.tsv:{line_number} seed")
        if seed not in SEEDS:
            raise FreezeValidationError(
                f"metrics.tsv:{line_number} seed must be one of {list(SEEDS)}, got {seed}."
            )
        scenario = str(raw_row["scenario"]).strip()
        if scenario not in SCENARIO_SPECS:
            raise FreezeValidationError(
                f"metrics.tsv:{line_number} has unknown scenario {scenario!r}."
            )
        spec = SCENARIO_SPECS[scenario]
        mode = str(raw_row["mode"]).strip()
        strength = _parse_decimal(
            raw_row["strength"], f"metrics.tsv:{line_number} strength"
        )
        kernel = _parse_integer(
            raw_row["kernel"], f"metrics.tsv:{line_number} kernel"
        )
        degradation_seed = _parse_integer(
            raw_row["dseed"], f"metrics.tsv:{line_number} dseed"
        )
        split = str(raw_row["split"]).strip()
        if split != "val":
            raise FreezeValidationError(
                f"metrics.tsv:{line_number} split must be 'val', got {split!r}."
            )
        if (
            mode != spec.mode
            or strength != spec.strength
            or kernel != spec.kernel
            or degradation_seed != spec.degradation_seed
        ):
            raise FreezeValidationError(
                f"metrics.tsv:{line_number} parameters do not match scenario "
                f"{scenario!r}: expected mode={spec.mode}, strength={spec.strength}, "
                f"kernel={spec.kernel}, dseed={spec.degradation_seed}; got "
                f"mode={mode}, strength={strength}, kernel={kernel}, "
                f"dseed={degradation_seed}."
            )

        metrics = {
            metric: _validate_metric_value(
                raw_row[metric], f"metrics.tsv:{line_number} {metric}"
            )
            for metric in METRIC_NAMES
        }
        key = (group, seed, scenario)
        if key in rows:
            raise FreezeValidationError(
                "metrics.tsv contains a duplicate group/seed/scenario row: "
                f"{key}."
            )
        rows[key] = MetricRow(
            group=group,
            seed=seed,
            scenario=scenario,
            mode=mode,
            strength=strength,
            kernel=kernel,
            degradation_seed=degradation_seed,
            split=split,
            metrics=metrics,
        )

    expected_keys = {
        (group, seed, scenario)
        for group in GROUPS
        for seed in SEEDS
        for scenario in SCENARIO_SPECS
    }
    missing = sorted(expected_keys - set(rows))
    extra = sorted(set(rows) - expected_keys)
    if missing or extra or len(rows) != len(expected_keys):
        raise FreezeValidationError(
            "metrics.tsv must contain exactly 5 x 3 x 11 = 165 rows; "
            f"got {len(rows)}, missing={missing[:5]}, extra={extra[:5]}."
        )
    return rows, _md5_bytes(raw)


def select_temperature(
    candidate_means: Mapping[Decimal | float | str, Decimal | float | str],
) -> tuple[Decimal, dict[str, Any]]:
    """Apply the locked E rule after ROUND_HALF_UP quantization to four decimals."""
    normalized: dict[Decimal, Decimal] = {}
    for raw_temperature, raw_mean in candidate_means.items():
        temperature = _parse_decimal(raw_temperature, "candidate temperature")
        if temperature in normalized:
            raise FreezeValidationError(
                f"Duplicate candidate temperature after normalization: {temperature}."
            )
        normalized[temperature] = _round4(
            _parse_decimal(raw_mean, f"mean R1 for T_r={temperature}")
        )
    if set(normalized) != set(E_CANDIDATE_TEMPERATURES):
        raise FreezeValidationError(
            "E candidates must be exactly 0.2, 0.3, and 0.5; got "
            f"{sorted(normalized)}."
        )

    ranked = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
    top_temperature, top_mean = ranked[0]
    second_temperature, second_mean = ranked[1]
    difference = top_mean - second_mean
    use_smaller_temperature = difference < TIE_THRESHOLD
    selected = (
        min(top_temperature, second_temperature)
        if use_smaller_temperature
        else top_temperature
    )
    evidence = {
        "top_temperature": top_temperature,
        "top_mean_r1_rounded4": top_mean,
        "second_temperature": second_temperature,
        "second_mean_r1_rounded4": second_mean,
        "top_minus_second_rounded4": difference,
        "threshold": TIE_THRESHOLD,
        "strictly_below_threshold": use_smaller_temperature,
    }
    return selected, evidence


def _temperature_summary(
    rows: Mapping[tuple[str, int, str], MetricRow],
    temperature: Decimal,
) -> dict[str, Any]:
    group = TEMPERATURE_GROUPS[temperature]
    per_seed_raw = {
        seed: _mean(
            [rows[(group, seed, scenario)].r1 for scenario in DEGRADED_SCENARIOS],
            f"{group} seed {seed} degraded R1",
        )
        for seed in SEEDS
    }
    seed_values = [per_seed_raw[seed] for seed in SEEDS]
    clean_values = [rows[(group, seed, "clean")].r1 for seed in SEEDS]
    return {
        "group": group,
        "per_seed": {seed: _round4(value) for seed, value in per_seed_raw.items()},
        "mean": _round4(_mean(seed_values, f"{group} seed means")),
        "std": _round4(_sample_std(seed_values, f"{group} seed means")),
        "clean_mean": _round4(_mean(clean_values, f"{group} clean R1")),
        "clean_std": _round4(_sample_std(clean_values, f"{group} clean R1")),
        "n_seeds": len(SEEDS),
        "n_degraded_scenarios_per_seed": len(DEGRADED_SCENARIOS),
    }


def _estimate_utility(
    rows: Mapping[tuple[str, int, str], MetricRow],
) -> dict[str, Any]:
    rgb_per_seed = {
        seed: rows[("devA", seed, "ir_missing")].r1 for seed in SEEDS
    }
    ir_per_seed = {
        seed: rows[("devA", seed, "rgb_missing")].r1 for seed in SEEDS
    }
    raw_rgb = _mean(list(rgb_per_seed.values()), "devA RGB-only utility")
    raw_ir = _mean(list(ir_per_seed.values()), "devA IR-only utility")
    total = raw_rgb + raw_ir
    if raw_rgb <= 0 or raw_ir <= 0 or total <= 0:
        raise FreezeValidationError(
            "Utility estimates must both be positive because mraf_utility_prior "
            f"requires positive values; got rgb={raw_rgb}, ir={raw_ir}."
        )
    return {
        "raw_rgb": raw_rgb,
        "raw_ir": raw_ir,
        "rgb_per_seed": rgb_per_seed,
        "ir_per_seed": ir_per_seed,
        "normalized": (raw_rgb / total, raw_ir / total),
    }


def _load_dev_split(path: Path) -> tuple[Mapping[str, Any], str]:
    raw = _read_bytes(path, "development split JSON")
    text = _decode_utf8(raw, "development split JSON")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FreezeValidationError(f"Invalid development split JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise FreezeValidationError("Development split JSON root must be an object.")
    if payload.get("n_dev") != 190 or payload.get("n_train_reduced") != 1076:
        raise FreezeValidationError(
            "Development split must declare n_dev=190 and n_train_reduced=1076; "
            f"got n_dev={payload.get('n_dev')!r}, "
            f"n_train_reduced={payload.get('n_train_reduced')!r}."
        )
    return payload, _md5_bytes(raw)


def _summary_json(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "group": summary["group"],
        "mean_r1": float(summary["mean"]),
        "std_r1": float(summary["std"]),
        "per_seed_r1": {
            str(seed): float(value) for seed, value in summary["per_seed"].items()
        },
        "n_seeds": int(summary["n_seeds"]),
        "n_degraded_scenarios_per_seed": int(
            summary["n_degraded_scenarios_per_seed"]
        ),
    }


def _selection_json(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "top_temperature": float(evidence["top_temperature"]),
        "top_mean_r1_rounded4": float(evidence["top_mean_r1_rounded4"]),
        "second_temperature": float(evidence["second_temperature"]),
        "second_mean_r1_rounded4": float(evidence["second_mean_r1_rounded4"]),
        "top_minus_second_rounded4": float(
            evidence["top_minus_second_rounded4"]
        ),
        "threshold": float(evidence["threshold"]),
        "strictly_below_threshold": bool(evidence["strictly_below_threshold"]),
        "rounding": "ROUND_HALF_UP to 4 decimal places before comparison",
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def build_frozen_artifacts(
    dev_metrics_path: Path,
    dev_split_path: Path,
    *,
    frozen_at: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Return the frozen JSON payload and reviewer-facing Markdown table."""
    rows, metrics_md5 = load_dev_metrics(dev_metrics_path)
    _, split_md5 = _load_dev_split(dev_split_path)

    summaries = {
        temperature: _temperature_summary(rows, temperature)
        for temperature in TEMPERATURE_GROUPS
    }
    selected_temperature, selection = select_temperature(
        {
            temperature: summaries[temperature]["mean"]
            for temperature in E_CANDIDATE_TEMPERATURES
        }
    )
    utility = _estimate_utility(rows)
    utility_normalized = [float(value) for value in utility["normalized"]]
    frozen_timestamp = frozen_at or _utc_now()

    payload = {
        "E": {"mraf_target_temperature": float(selected_temperature)},
        "U": {"mraf_utility_prior": utility_normalized},
        "selection_evidence": {
            "tr_candidates": {
                _temperature_key(temperature): _summary_json(summaries[temperature])
                for temperature in E_CANDIDATE_TEMPERATURES
            },
            "tr_reference_0.1": _summary_json(summaries[Decimal("0.1")]),
            "tr_selection": _selection_json(selection),
            "utility_raw": {
                "rgb": float(utility["raw_rgb"]),
                "ir": float(utility["raw_ir"]),
                "rgb_per_seed": {
                    str(seed): float(value)
                    for seed, value in utility["rgb_per_seed"].items()
                },
                "ir_per_seed": {
                    str(seed): float(value)
                    for seed, value in utility["ir_per_seed"].items()
                },
            },
            "utility_normalized": utility_normalized,
            "dev_split_md5": split_md5,
            "dev_metrics_md5": metrics_md5,
            "n_seeds": len(SEEDS),
            "frozen_at": frozen_timestamp,
        },
    }
    markdown = _build_selection_markdown(
        summaries=summaries,
        selected_temperature=selected_temperature,
        selection=selection,
        utility=utility,
        split_md5=split_md5,
        metrics_md5=metrics_md5,
        frozen_at=frozen_timestamp,
    )
    return payload, markdown


def _format4(value: Decimal) -> str:
    return format(_round4(value), ".4f")


def _build_selection_markdown(
    *,
    summaries: Mapping[Decimal, Mapping[str, Any]],
    selected_temperature: Decimal,
    selection: Mapping[str, Any],
    utility: Mapping[str, Any],
    split_md5: str,
    metrics_md5: str,
    frozen_at: str,
) -> str:
    lines = [
        "# Development-set hyperparameter selection",
        "",
        "All results below use only the locked `val` split. R1 values are percentage points.",
        "",
        "## MRAF target temperature",
        "",
        "Each seed value is its mean R1 over the ten degraded scenarios (clean excluded).",
        "The degraded and clean summaries are the mean ± sample standard deviation "
        "across three seeds.",
        "",
        "| T_r | role | seed 1 | seed 2 | seed 3 | degraded R1 mean ± SD | "
        "clean R1 mean ± SD | selected for E |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for temperature in TEMPERATURE_GROUPS:
        summary = summaries[temperature]
        role = "reference" if temperature == Decimal("0.1") else "candidate"
        selected = "yes" if temperature == selected_temperature else "no"
        per_seed = summary["per_seed"]
        lines.append(
            f"| {_temperature_key(temperature)} | {role} | "
            f"{_format4(per_seed[1])} | {_format4(per_seed[2])} | "
            f"{_format4(per_seed[3])} | {_format4(summary['mean'])} ± "
            f"{_format4(summary['std'])} | {_format4(summary['clean_mean'])} ± "
            f"{_format4(summary['clean_std'])} | {selected} |"
        )
    lines.extend(
        [
            "",
            "Selection compares candidate means after `ROUND_HALF_UP` rounding to four "
            "decimal places. If the top-minus-second difference is strictly below "
            "0.3000 percentage points, the smaller of those two temperatures is used.",
            "",
            f"Observed top-minus-second difference: "
            f"{_format4(selection['top_minus_second_rounded4'])}; "
            f"tie rule applied: "
            f"{'yes' if selection['strictly_below_threshold'] else 'no'}.",
            "",
            "## Utility prior",
            "",
            "`ir_missing` is RGB-only and estimates RGB utility; `rgb_missing` is "
            "thermal-only and estimates IR utility.",
            "",
            "| modality | source scenario | seed 1 | seed 2 | seed 3 | raw mean | normalized |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            f"| RGB | `ir_missing` | "
            f"{_format4(utility['rgb_per_seed'][1])} | "
            f"{_format4(utility['rgb_per_seed'][2])} | "
            f"{_format4(utility['rgb_per_seed'][3])} | "
            f"{_format4(utility['raw_rgb'])} | "
            f"{float(utility['normalized'][0]):.8f} |",
            f"| IR | `rgb_missing` | "
            f"{_format4(utility['ir_per_seed'][1])} | "
            f"{_format4(utility['ir_per_seed'][2])} | "
            f"{_format4(utility['ir_per_seed'][3])} | "
            f"{_format4(utility['raw_ir'])} | "
            f"{float(utility['normalized'][1]):.8f} |",
            "",
            "## Audit provenance",
            "",
            f"- Development split MD5: `{split_md5}`",
            f"- Development metrics MD5: `{metrics_md5}`",
            f"- Seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
            f"- Frozen at: `{frozen_at}`",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_atomic(path: Path, text: str) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, resolved)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_frozen_artifacts(
    payload: Mapping[str, Any],
    markdown: str,
    out_path: Path,
    table_path: Path,
) -> None:
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    markdown_text = markdown if markdown.endswith("\n") else markdown + "\n"
    _write_text_atomic(out_path, json_text)
    _write_text_atomic(table_path, markdown_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze E target temperature and U modality utility from the complete "
            "val-only Stage 2 development grid."
        )
    )
    parser.add_argument(
        "--dev-metrics",
        type=Path,
        required=True,
        help="metrics.tsv containing exactly 5 groups x 3 seeds x 11 scenarios",
    )
    parser.add_argument(
        "--dev-split",
        type=Path,
        default=PROJECT_ROOT / "splits" / "dev_190.json",
        help="locked dev split JSON used for provenance MD5",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output FROZEN_HPARAMS.json path",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "protocol" / "DEV_SELECTION_TABLE.md",
        help="reviewer-facing Markdown selection table output path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload, markdown = build_frozen_artifacts(
            dev_metrics_path=args.dev_metrics,
            dev_split_path=args.dev_split,
        )
        write_frozen_artifacts(payload, markdown, args.out, args.table_out)
    except (FreezeValidationError, OSError) as exc:
        parser.exit(2, f"freeze_hparams.py: error: {exc}\n")

    print(markdown, end="")
    print(f"FROZEN_HPARAMS_JSON={args.out.expanduser().resolve()}")
    print(f"DEV_SELECTION_TABLE={args.table_out.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
