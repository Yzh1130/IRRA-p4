#!/usr/bin/env python3
"""Measure the locked external-dataset precondition from the dev2 A grid.

The input is deliberately narrower than a general experiment table: it must be
the complete 3-seed, 11-scenario, ``group=A``, ``split=val`` result produced by
Stage 5 phase 1.  Rejecting mixed or partial tables prevents test results or a
different arm from silently entering the preregistration evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUP = "A"
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
ONE_HUNDRED = Decimal("100")
RGBT_PEDES_TEST_UTILITY_RATIO = Decimal("3.80")
RGBT_PEDES_DEV_UTILITY_RATIO = Decimal("2.68")


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
NOISE_SCENARIOS = (
    "rgb_noise_005",
    "ir_noise_005",
    "rgb_noise_010",
    "ir_noise_010",
)


@dataclass(frozen=True)
class MetricRow:
    seed: int
    scenario: str
    metrics: Mapping[str, Decimal]

    @property
    def r1(self) -> Decimal:
        return self.metrics["R1"]


class PreconditionValidationError(ValueError):
    """Raised when phase-1 evidence violates the locked protocol."""


def _read_bytes(path: Path) -> bytes:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise PreconditionValidationError(
            f"External precondition metrics TSV does not exist: {resolved}"
        )
    try:
        return resolved.read_bytes()
    except OSError as exc:
        raise PreconditionValidationError(
            f"Failed to read external precondition metrics TSV {resolved}: {exc}"
        ) from exc


def _decode_utf8(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreconditionValidationError(
            f"External precondition metrics TSV must be UTF-8: {exc}"
        ) from exc


def _parse_integer(value: Any, label: str) -> int:
    text = str(value).strip()
    try:
        parsed = int(text)
    except ValueError as exc:
        raise PreconditionValidationError(
            f"{label} must be an integer, got {value!r}."
        ) from exc
    if str(parsed) != text:
        raise PreconditionValidationError(
            f"{label} must be an integer, got {value!r}."
        )
    return parsed


def _parse_decimal(value: Any, label: str) -> Decimal:
    text = str(value).strip()
    if not text:
        raise PreconditionValidationError(f"{label} must not be empty.")
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise PreconditionValidationError(
            f"{label} must be a decimal number, got {value!r}."
        ) from exc
    if not parsed.is_finite():
        raise PreconditionValidationError(
            f"{label} must be finite, got {value!r}."
        )
    return parsed


def _parse_metric(value: Any, label: str) -> Decimal:
    parsed = _parse_decimal(value, label)
    if not Decimal("0") <= parsed <= ONE_HUNDRED:
        raise PreconditionValidationError(
            f"{label} must be in [0, 100], got {parsed}."
        )
    return parsed


def _mean(values: Sequence[Decimal], label: str) -> Decimal:
    if not values:
        raise PreconditionValidationError(f"Cannot compute an empty mean for {label}.")
    return sum(values, Decimal("0")) / Decimal(len(values))


def _round4(value: Decimal) -> Decimal:
    return value.quantize(FOUR_DECIMALS, rounding=ROUND_HALF_UP)


def _format4(value: Decimal) -> str:
    return format(_round4(value), ".4f")


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def load_precondition_metrics(
    path: Path,
) -> tuple[dict[tuple[int, str], MetricRow], str]:
    """Load exactly A x seeds 1/2/3 x the 11 locked val scenarios."""
    raw = _read_bytes(path)
    text = _decode_utf8(raw)
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if tuple(reader.fieldnames or ()) != METRICS_HEADER:
        raise PreconditionValidationError(
            "metrics.tsv header mismatch: expected exactly "
            f"{list(METRICS_HEADER)}, got {reader.fieldnames}."
        )

    rows: dict[tuple[int, str], MetricRow] = {}
    for line_number, raw_row in enumerate(reader, start=2):
        label = f"metrics.tsv:{line_number}"
        if None in raw_row:
            raise PreconditionValidationError(
                f"{label} has columns beyond the required 13-column schema: "
                f"{raw_row[None]!r}."
            )
        missing_columns = [
            field for field in METRICS_HEADER if raw_row[field] is None
        ]
        if missing_columns:
            raise PreconditionValidationError(
                f"{label} is missing values for columns {missing_columns}."
            )
        group = str(raw_row["group"]).strip()
        if group != GROUP:
            raise PreconditionValidationError(
                f"{label} group must be {GROUP!r}, got {group!r}."
            )
        seed = _parse_integer(raw_row["seed"], f"{label} seed")
        if seed not in SEEDS:
            raise PreconditionValidationError(
                f"{label} seed must be one of {list(SEEDS)}, got {seed}."
            )
        scenario = str(raw_row["scenario"]).strip()
        if scenario not in SCENARIO_SPECS:
            raise PreconditionValidationError(
                f"{label} has unknown scenario {scenario!r}."
            )
        split = str(raw_row["split"]).strip()
        if split != "val":
            raise PreconditionValidationError(
                f"{label} split must be 'val', got {split!r}."
            )

        spec = SCENARIO_SPECS[scenario]
        mode = str(raw_row["mode"]).strip()
        strength = _parse_decimal(raw_row["strength"], f"{label} strength")
        kernel = _parse_integer(raw_row["kernel"], f"{label} kernel")
        degradation_seed = _parse_integer(raw_row["dseed"], f"{label} dseed")
        if (
            mode != spec.mode
            or strength != spec.strength
            or kernel != spec.kernel
            or degradation_seed != spec.degradation_seed
        ):
            raise PreconditionValidationError(
                f"{label} parameters do not match scenario {scenario!r}: expected "
                f"mode={spec.mode}, strength={spec.strength}, kernel={spec.kernel}, "
                f"dseed={spec.degradation_seed}; got mode={mode}, "
                f"strength={strength}, kernel={kernel}, dseed={degradation_seed}."
            )

        metrics = {
            metric: _parse_metric(raw_row[metric], f"{label} {metric}")
            for metric in METRIC_NAMES
        }
        key = (seed, scenario)
        if key in rows:
            raise PreconditionValidationError(
                "metrics.tsv contains a duplicate seed/scenario row: "
                f"seed={seed}, scenario={scenario!r}."
            )
        rows[key] = MetricRow(seed=seed, scenario=scenario, metrics=metrics)

    expected_keys = {
        (seed, scenario) for seed in SEEDS for scenario in SCENARIO_SPECS
    }
    missing = sorted(expected_keys - set(rows))
    extra = sorted(set(rows) - expected_keys)
    if len(rows) != len(expected_keys) or missing or extra:
        raise PreconditionValidationError(
            "metrics.tsv must contain exactly A x 3 seeds x 11 scenarios = 33 "
            f"rows; got {len(rows)}, missing={missing[:5]}, extra={extra[:5]}."
        )
    return rows, _md5_bytes(raw)


def _ordering(rgb_value: Decimal, ir_value: Decimal) -> str:
    if rgb_value > ir_value:
        return "RGB > IR"
    if rgb_value < ir_value:
        return "IR > RGB"
    return "RGB = IR"


def _more_sensitive_modality(rgb_drop: Decimal, ir_drop: Decimal) -> str:
    if rgb_drop > ir_drop:
        return "RGB"
    if rgb_drop < ir_drop:
        return "IR"
    return "equal"


def compute_precondition(
    rows: Mapping[tuple[int, str], MetricRow],
) -> dict[str, Any]:
    """Compute utilities and per-seed clean-relative noise sensitivity."""
    rgb_per_seed = {
        seed: rows[(seed, "ir_missing")].r1 for seed in SEEDS
    }
    ir_per_seed = {
        seed: rows[(seed, "rgb_missing")].r1 for seed in SEEDS
    }
    u_rgb = _mean(list(rgb_per_seed.values()), "RGB-only utility")
    u_ir = _mean(list(ir_per_seed.values()), "IR-only utility")
    if u_rgb <= 0 or u_ir <= 0:
        raise PreconditionValidationError(
            "Both utility estimates must be positive to compute their ratio; "
            f"got u_rgb={u_rgb}, u_ir={u_ir}."
        )

    noise_drops: dict[str, dict[str, Any]] = {}
    for scenario in NOISE_SCENARIOS:
        per_seed: dict[int, Decimal] = {}
        for seed in SEEDS:
            clean = rows[(seed, "clean")].r1
            if clean <= 0:
                raise PreconditionValidationError(
                    f"Clean R1 must be positive for seed {seed}, got {clean}."
                )
            noisy = rows[(seed, scenario)].r1
            per_seed[seed] = (clean - noisy) / clean * ONE_HUNDRED
        noise_drops[scenario] = {
            "per_seed": per_seed,
            "mean": _mean(
                list(per_seed.values()), f"{scenario} relative R1 drop"
            ),
        }

    rgb_sensitivity = _mean(
        [
            noise_drops[name]["mean"]
            for name in NOISE_SCENARIOS
            if name.startswith("rgb_")
        ],
        "RGB noise sensitivity",
    )
    ir_sensitivity = _mean(
        [
            noise_drops[name]["mean"]
            for name in NOISE_SCENARIOS
            if name.startswith("ir_")
        ],
        "IR noise sensitivity",
    )
    utility_order = _ordering(u_rgb, u_ir)
    sensitivity_order = _ordering(rgb_sensitivity, ir_sensitivity)
    ordering_is_opposite = (
        (u_rgb > u_ir and rgb_sensitivity < ir_sensitivity)
        or (u_rgb < u_ir and rgb_sensitivity > ir_sensitivity)
    )
    return {
        "rgb_per_seed": rgb_per_seed,
        "ir_per_seed": ir_per_seed,
        "u_rgb": u_rgb,
        "u_ir": u_ir,
        "utility_ratio": u_rgb / u_ir,
        "noise_drops": noise_drops,
        "rgb_sensitivity": rgb_sensitivity,
        "ir_sensitivity": ir_sensitivity,
        "utility_order": utility_order,
        "sensitivity_order": sensitivity_order,
        "ordering_is_opposite": ordering_is_opposite,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def build_precondition_markdown(
    result: Mapping[str, Any],
    *,
    metrics_path: Path,
    metrics_md5: str,
    generated_at: str | None = None,
) -> str:
    """Build the auditable Markdown report without writing it."""
    timestamp = generated_at or _utc_now()
    rgb_per_seed = result["rgb_per_seed"]
    ir_per_seed = result["ir_per_seed"]
    lines = [
        "# RGBNT201-PEDES external-validation precondition",
        "",
        "This report uses only configuration A on the locked `dev2` (`val`) split.",
        "R1 values are percentage points; relative drops are percentages of each "
        "seed's clean R1.",
        "",
        "## Modality utility",
        "",
        "`ir_missing` is RGB-only and estimates `û_rgb`; `rgb_missing` is IR-only "
        "and estimates `û_ir`.",
        "",
        "| estimate | source scenario | seed 1 R1 | seed 2 R1 | seed 3 R1 | mean R1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        f"| `û_rgb` | `ir_missing` | {_format4(rgb_per_seed[1])} | "
        f"{_format4(rgb_per_seed[2])} | {_format4(rgb_per_seed[3])} | "
        f"{_format4(result['u_rgb'])} |",
        f"| `û_ir` | `rgb_missing` | {_format4(ir_per_seed[1])} | "
        f"{_format4(ir_per_seed[2])} | {_format4(ir_per_seed[3])} | "
        f"{_format4(result['u_ir'])} |",
        "",
        f"**Measured utility ratio `û_rgb / û_ir`: "
        f"{_format4(result['utility_ratio'])}.**",
        "",
        "## Noise sensitivity relative to clean",
        "",
        "For every seed and scenario, relative R1 drop is "
        "`100 × (R1_clean - R1_noisy) / R1_clean`; a negative value means the "
        "noisy score was higher than clean.",
        "",
        "| scenario | perturbed modality | seed 1 drop (%) | seed 2 drop (%) | "
        "seed 3 drop (%) | mean drop (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario in NOISE_SCENARIOS:
        modality = "RGB" if scenario.startswith("rgb_") else "IR"
        drop = result["noise_drops"][scenario]
        per_seed = drop["per_seed"]
        lines.append(
            f"| `{scenario}` | {modality} | {_format4(per_seed[1])} | "
            f"{_format4(per_seed[2])} | {_format4(per_seed[3])} | "
            f"{_format4(drop['mean'])} |"
        )
    lines.extend(
        [
            "",
            "Matched-strength comparison:",
            "",
            "| noise strength | RGB mean relative drop (%) | "
            "IR mean relative drop (%) | more sensitive modality |",
            "| ---: | ---: | ---: | --- |",
        ]
    )
    for suffix, strength in (("005", "0.05"), ("010", "0.10")):
        rgb_drop = result["noise_drops"][f"rgb_noise_{suffix}"]["mean"]
        ir_drop = result["noise_drops"][f"ir_noise_{suffix}"]["mean"]
        lines.append(
            f"| {strength} | {_format4(rgb_drop)} | {_format4(ir_drop)} | "
            f"{_more_sensitive_modality(rgb_drop, ir_drop)} |"
        )
    lines.extend(
        [
            "",
            f"- Utility ordering: **{result['utility_order']}**.",
            f"- Mean noise-sensitivity ordering across 0.05 and 0.10: "
            f"**{result['sensitivity_order']}** "
            f"(RGB {_format4(result['rgb_sensitivity'])}%, "
            f"IR {_format4(result['ir_sensitivity'])}%).",
            f"- Sensitivity and utility orderings are opposite: "
            f"**{'yes' if result['ordering_is_opposite'] else 'no'}**.",
            "",
            "## Cross-dataset utility-ratio anchors",
            "",
            "| dataset | split | `û_rgb / û_ir` | source |",
            "| --- | --- | ---: | --- |",
            f"| RGBNT201-PEDES | dev2 (`val`) | "
            f"{_format4(result['utility_ratio'])} | measured above |",
            f"| RGBT-PEDES | test | {RGBT_PEDES_TEST_UTILITY_RATIO:.2f} | "
            "locked comparison anchor |",
            f"| RGBT-PEDES | development | {RGBT_PEDES_DEV_UTILITY_RATIO:.2f} | "
            "locked comparison anchor |",
            "",
            "## Audit provenance",
            "",
            f"- Metrics source: `{metrics_path.expanduser().resolve()}`",
            f"- Metrics MD5: `{metrics_md5}`",
            f"- Accepted group: `{GROUP}`",
            "- Accepted split: `val`",
            f"- Seeds: `{', '.join(str(seed) for seed in SEEDS)}`",
            f"- Scenarios: `{', '.join(SCENARIO_SPECS)}`",
            f"- Validated rows: `{len(SEEDS) * len(SCENARIO_SPECS)}`",
            f"- Generated at: `{timestamp}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_text_atomic(path: Path, text: str) -> None:
    """Atomically write UTF-8 text with LF line endings."""
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
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
            handle.write(normalized)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, resolved)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure RGBNT201-PEDES modality utility and clean-relative noise "
            "sensitivity from the complete phase-1 A/dev2 grid."
        )
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="phase-1 metrics.tsv containing exactly A x 3 seeds x 11 val scenarios",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "protocol" / "EXTERNAL_PRECONDITION.md",
        help="output Markdown report path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows, metrics_md5 = load_precondition_metrics(args.metrics)
        result = compute_precondition(rows)
        markdown = build_precondition_markdown(
            result,
            metrics_path=args.metrics,
            metrics_md5=metrics_md5,
        )
        write_text_atomic(args.out, markdown)
    except (PreconditionValidationError, OSError) as exc:
        parser.exit(2, f"measure_precondition.py: error: {exc}\n")

    print(markdown, end="")
    print(f"METRICS_MD5={metrics_md5}")
    print(f"EXTERNAL_PRECONDITION={args.out.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
