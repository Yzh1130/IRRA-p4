#!/usr/bin/env python3
"""Summarize and gate the MRAF gate-amplitude rescue pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.revision.summarize_mraf_gate_rescue import (  # noqa: E402
    _active_diagnostic_dir,
    _as_float,
    _diagnostic_case,
    _load_score_metrics,
    _read_csv,
    _read_json,
    _scenario_row,
    _scenario_score,
)


STAGE1_MATRIX: Mapping[str, Mapping[str, Any]] = {
    "A10": {
        "mraf_gate_fp32": True,
        "mraf_rel_lambda": 0.25,
        "mraf_gate_base_grad_scale": 0.10,
        "mraf_scorer_lr_multiplier": 10.0,
        "mraf_gate_logit_scale": 1.0,
    },
    "A50": {
        "mraf_gate_fp32": True,
        "mraf_rel_lambda": 0.25,
        "mraf_gate_base_grad_scale": 0.10,
        "mraf_scorer_lr_multiplier": 50.0,
        "mraf_gate_logit_scale": 1.0,
    },
    "A100": {
        "mraf_gate_fp32": True,
        "mraf_rel_lambda": 0.25,
        "mraf_gate_base_grad_scale": 0.10,
        "mraf_scorer_lr_multiplier": 100.0,
        "mraf_gate_logit_scale": 1.0,
    },
    "B50": {
        "mraf_gate_fp32": True,
        "mraf_rel_lambda": 1.00,
        "mraf_gate_base_grad_scale": 1.00,
        "mraf_scorer_lr_multiplier": 50.0,
        "mraf_gate_logit_scale": 1.0,
    },
}

CHECKPOINT_KINDS = ("best", "last")
CALIBRATION_SCALES = (2.0, 5.0, 10.0, 20.0, 50.0)
SCREEN_SCENARIOS = (
    "clean",
    "rgb_missing",
    "ir_missing",
    "rgb_noise_0.10",
    "ir_noise_0.10",
    "rgb_blur_k7",
    "ir_blur_k7",
    "both_blur_k7",
)
FORMAL_SCENARIOS = (
    "clean",
    "rgb_missing",
    "ir_missing",
    "rgb_noise_0.05",
    "rgb_noise_0.10",
    "ir_noise_0.05",
    "ir_noise_0.10",
    "rgb_blur_k7",
    "ir_blur_k7",
    "both_noise_0.05",
    "both_blur_k7",
)
KEY_DEGRADED_SCENARIOS = SCREEN_SCENARIOS[1:]

METRIC_FIELDS = (
    "candidate",
    "checkpoint",
    "scenario",
    "R1",
    "R5",
    "R10",
    "mAP",
    "mINP",
    "score_file",
)
MECHANISM_FIELDS = (
    "candidate",
    "checkpoint",
    "mraf_scorer_lr_multiplier",
    "mraf_gate_logit_scale",
    "mraf_rel_lambda",
    "mraf_gate_base_grad_scale",
    "alpha_std",
    "alpha_target_pearson",
    "rgb_missing_alpha_rgb",
    "ir_missing_alpha_ir",
    "rgb_noise_alpha_drop",
    "ir_noise_alpha_drop",
    "rgb_blur_alpha_drop",
    "ir_blur_alpha_drop",
    "weighted_rel_base_ratio",
    "scorer_parameter_delta",
    "raw_logit_gap_std",
    "scaled_logit_gap_std",
    "scorer_parameter_updated",
    "fusion_path_sensitive",
    "gate_logits_fp32",
)
COMPARISON_FIELDS = (
    "candidate",
    "checkpoint",
    "complete",
    "has_nan",
    "has_inf",
    "has_oom",
    "clean_R1",
    "degraded_mean_R1",
    *MECHANISM_FIELDS[2:],
    "strong_pass",
    "viable_pass",
    "classification",
    "failed_strong_criteria",
    "failed_viable_criteria",
)
RANKING_FIELDS = (
    "rank",
    "candidate",
    "checkpoint",
    "classification",
    "strong_pass",
    "viable_pass",
    "degraded_mean_R1",
    "clean_R1",
    "alpha_std",
    "alpha_target_pearson",
    "mraf_scorer_lr_multiplier",
    "mraf_gate_logit_scale",
    "mraf_rel_lambda",
    "mraf_gate_base_grad_scale",
)
CALIBRATION_FIELDS = (
    "rank",
    "source_candidate",
    "source_checkpoint",
    "mraf_gate_logit_scale",
    "classification",
    "strong_pass",
    "clean_R1",
    "degraded_mean_R1",
    "alpha_std",
    "alpha_target_pearson",
    "failed_strong_criteria",
)
CONFIRMATION_FIELDS = COMPARISON_FIELDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize MRAF scorer-LR screening, read-only logit calibration, "
            "confirmation, and final best/last results."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("screen", "calibration", "confirmation", "final", "mock"),
        required=True,
    )
    parser.add_argument("--candidate-root", default="")
    parser.add_argument("--reference-screen-root", default="")
    parser.add_argument("--reference-run-root", default="")
    parser.add_argument("--selected-json", default="")
    parser.add_argument("--output-dir", required=True)
    return parser


def _write_tsv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _finite(value: Any) -> bool:
    parsed = _as_float(value)
    return parsed is not None and math.isfinite(parsed)


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [_as_float(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return median(finite) if finite else None


def _alpha_drop(
    rows: Sequence[Mapping[str, Any]], scenario: str
) -> Optional[float]:
    row = _scenario_row(rows, scenario)
    delta = _as_float(row.get("degraded_alpha_delta_mean"))
    return None if delta is None else -delta


def collect_unit(
    unit_dir: Path,
    candidate: str,
    checkpoint: str,
    scenarios: Sequence[str] = SCREEN_SCENARIOS,
) -> Dict[str, Any]:
    """Collect one independently evaluated checkpoint/scale unit."""
    result: Dict[str, Any] = {
        "candidate": candidate,
        "checkpoint": checkpoint,
        "complete": False,
        "has_nan": False,
        "has_inf": False,
        "has_oom": False,
        "metrics": {},
        "metric_rows": [],
    }
    if not unit_dir.is_dir():
        result["collection_error"] = "unit_directory_missing"
        return result
    metadata_path = unit_dir / "unit.json"
    metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
    if isinstance(metadata, Mapping):
        for key in (
            "has_nan",
            "has_inf",
            "has_oom",
            "mraf_scorer_lr_multiplier",
            "mraf_gate_logit_scale",
            "mraf_rel_lambda",
            "mraf_gate_base_grad_scale",
        ):
            if key in metadata:
                result[key] = metadata[key]

    metric_rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        score_path = _scenario_score(unit_dir, scenario)
        if score_path is None:
            continue
        metrics = _load_score_metrics(score_path)
        result["metrics"][scenario] = metrics
        metric_rows.append(
            {
                "candidate": candidate,
                "checkpoint": checkpoint,
                "scenario": scenario,
                **metrics,
                "score_file": str(score_path),
            }
        )
    result["metric_rows"] = metric_rows

    diagnostic_dir = _active_diagnostic_dir(unit_dir)
    diagnosis_path = diagnostic_dir / "diagnosis.json"
    diagnosis = _read_json(diagnosis_path) if diagnosis_path.is_file() else {}
    case_diagnosis = (
        _diagnostic_case(diagnosis, candidate)
        if isinstance(diagnosis, Mapping)
        else {}
    )
    gate_rows = _read_csv(diagnostic_dir / "scenario_gate_statistics.csv")
    direction_rows = _read_csv(diagnostic_dir / "gate_direction_statistics.csv")
    gradient_rows = _read_csv(diagnostic_dir / "gradient_diagnostics.csv")
    epoch_rows = _read_tsv(unit_dir / "mraf_epoch_metrics.tsv")
    selection_rows = _read_tsv(unit_dir / "checkpoint_selection.tsv")
    selected_epoch = next(
        (
            row.get("epoch")
            for row in reversed(selection_rows)
            if str(row.get("checkpoint", "")) == f"{checkpoint}.pth"
        ),
        None,
    )
    selected_epoch_row = next(
        (
            row
            for row in reversed(epoch_rows)
            if selected_epoch is not None
            and str(row.get("epoch")) == str(selected_epoch)
        ),
        epoch_rows[-1] if epoch_rows else {},
    )
    rgb_missing = _scenario_row(gate_rows, "rgb_missing")
    ir_missing = _scenario_row(gate_rows, "ir_missing")
    clean_gate = _scenario_row(gate_rows, "clean")
    alarms = (
        case_diagnosis.get("alarms", {})
        if isinstance(case_diagnosis, Mapping)
        else {}
    )
    parameter_updated = bool(
        isinstance(alarms, Mapping)
        and "PARAMETER_NOT_UPDATED" in alarms
        and not bool(alarms["PARAMETER_NOT_UPDATED"])
    )
    parameter_delta = _as_float(
        selected_epoch_row.get("scorer_parameter_delta_from_initial")
    )
    if parameter_delta is not None:
        parameter_updated = parameter_delta > 0.0
    raw_dtypes = {
        str(row.get("raw_logits_dtype", ""))
        for row in gate_rows
        if row.get("raw_logits_dtype")
    }
    mechanism = {
        "candidate": candidate,
        "checkpoint": checkpoint,
        "mraf_scorer_lr_multiplier": _as_float(
            result.get("mraf_scorer_lr_multiplier")
        ),
        "mraf_gate_logit_scale": (
            _as_float(result.get("mraf_gate_logit_scale")) or 1.0
        ),
        "mraf_rel_lambda": _as_float(result.get("mraf_rel_lambda")),
        "mraf_gate_base_grad_scale": _as_float(
            result.get("mraf_gate_base_grad_scale")
        ),
        "alpha_std": _as_float(case_diagnosis.get("alpha_rgb_std")),
        "alpha_target_pearson": _as_float(
            case_diagnosis.get("alpha_target_pearson_degraded_scenarios")
        ),
        "rgb_missing_alpha_rgb": _as_float(
            rgb_missing.get("alpha_rgb_mean")
        ),
        "ir_missing_alpha_ir": _as_float(ir_missing.get("alpha_ir_mean")),
        "rgb_noise_alpha_drop": _alpha_drop(
            direction_rows, "rgb_noise_0.10"
        ),
        "ir_noise_alpha_drop": _alpha_drop(
            direction_rows, "ir_noise_0.10"
        ),
        "rgb_blur_alpha_drop": _alpha_drop(direction_rows, "rgb_blur_k7"),
        "ir_blur_alpha_drop": _alpha_drop(direction_rows, "ir_blur_k7"),
        "weighted_rel_base_ratio": (
            _as_float(case_diagnosis.get("weighted_rel_base_ratio_median"))
            or _median(gradient_rows, "weighted_mraf_rel_over_loss_base")
        ),
        "scorer_parameter_delta": parameter_delta,
        "raw_logit_gap_std": _as_float(clean_gate.get("logit_gap_std")),
        "scaled_logit_gap_std": _as_float(
            clean_gate.get("scaled_logit_gap_std")
        ),
        "scorer_parameter_updated": parameter_updated,
        "fusion_path_sensitive": bool(
            case_diagnosis.get("fusion_path_sensitive", False)
        ),
        "gate_logits_fp32": bool(raw_dtypes)
        and raw_dtypes == {"torch.float32"},
    }
    result["mechanism"] = mechanism
    result["complete"] = (
        len(result["metrics"]) == len(scenarios)
        and diagnosis_path.is_file()
        and (unit_dir / "configs.yaml").is_file()
        and (unit_dir / "checkpoint.pth").is_file()
        and (unit_dir / "train.log").is_file()
    )
    return result


def _reference_from_tsv(path: Path) -> Optional[Tuple[float, float]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, UnicodeError, csv.Error):
        return None
    for row in rows:
        if str(row.get("candidate", "")).split("_")[0] != "S4":
            continue
        clean = _as_float(row.get("clean_R1"))
        degraded = _as_float(
            row.get("key_degraded_mean_R1", row.get("degraded_mean_R1"))
        )
        if clean is not None and degraded is not None:
            return clean, degraded
    return None


def load_reference_s4(reference_root: Path) -> Dict[str, float]:
    """Read historical S4 summaries without modifying the reference tree."""
    if not reference_root.is_dir():
        raise FileNotFoundError(
            f"Reference screen root not found: {reference_root}"
        )
    names = (
        "rescue_summary.tsv",
        "rescue_candidate_ranking.tsv",
        "comparison_with_previous_screen.tsv",
    )
    matches = sorted(
        path
        for name in names
        for path in reference_root.rglob(name)
        if path.is_file()
    )
    for path in matches:
        values = _reference_from_tsv(path)
        if values is not None:
            return {
                "clean_R1": values[0],
                "degraded_mean_R1": values[1],
                "source": str(path.resolve()),
            }
    raise ValueError(
        "Could not read S4 clean/degraded metrics from --reference-screen-root."
    )


def _criterion(
    failures: List[str],
    condition: bool,
    name: str,
) -> None:
    if not condition:
        failures.append(name)


def classify_candidate(
    unit: Mapping[str, Any],
    reference_s4: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = unit.get("metrics", {})
    mechanism = unit.get("mechanism", {})
    clean = _as_float(metrics.get("clean", {}).get("R1"))
    degraded_values = [
        _as_float(metrics.get(name, {}).get("R1"))
        for name in KEY_DEGRADED_SCENARIOS
    ]
    finite_degraded = [
        value for value in degraded_values if value is not None and math.isfinite(value)
    ]
    degraded_mean = (
        sum(finite_degraded) / len(finite_degraded)
        if len(finite_degraded) == len(KEY_DEGRADED_SCENARIOS)
        else None
    )
    alpha_std = _as_float(mechanism.get("alpha_std"))
    pearson = _as_float(mechanism.get("alpha_target_pearson"))
    rgb_missing = _as_float(mechanism.get("rgb_missing_alpha_rgb"))
    ir_missing = _as_float(mechanism.get("ir_missing_alpha_ir"))
    ratio = _as_float(mechanism.get("weighted_rel_base_ratio"))
    drops = [
        _as_float(mechanism.get(name))
        for name in (
            "rgb_noise_alpha_drop",
            "ir_noise_alpha_drop",
            "rgb_blur_alpha_drop",
            "ir_blur_alpha_drop",
        )
    ]
    reference_clean = _as_float(reference_s4.get("clean_R1"))
    reference_degraded = _as_float(reference_s4.get("degraded_mean_R1"))
    common = (
        bool(unit.get("complete"))
        and not bool(unit.get("has_nan"))
        and not bool(unit.get("has_inf"))
        and not bool(unit.get("has_oom"))
    )

    strong_failures: List[str] = []
    _criterion(strong_failures, common, "COMPLETE_FINITE_NO_OOM")
    _criterion(strong_failures, alpha_std is not None and alpha_std >= 0.005, "ALPHA_STD")
    _criterion(strong_failures, pearson is not None and pearson >= 0.30, "PEARSON")
    _criterion(strong_failures, rgb_missing is not None and rgb_missing <= 0.35, "RGB_MISSING")
    _criterion(strong_failures, ir_missing is not None and ir_missing <= 0.35, "IR_MISSING")
    for name, value in zip(("RGB_NOISE", "IR_NOISE", "RGB_BLUR", "IR_BLUR"), drops):
        _criterion(strong_failures, value is not None and value >= 0.02, name)
    _criterion(
        strong_failures,
        ratio is not None and 1e-3 <= ratio <= 5e-2,
        "WEIGHTED_REL_BASE_RATIO",
    )
    _criterion(
        strong_failures,
        clean is not None
        and reference_clean is not None
        and clean >= reference_clean - 0.50,
        "CLEAN_VS_S4",
    )
    _criterion(
        strong_failures,
        degraded_mean is not None
        and reference_degraded is not None
        and degraded_mean >= reference_degraded - 0.20,
        "DEGRADED_VS_S4",
    )
    _criterion(
        strong_failures,
        bool(mechanism.get("scorer_parameter_updated")),
        "SCORER_PARAMETER_UPDATED",
    )
    _criterion(
        strong_failures,
        bool(mechanism.get("fusion_path_sensitive")),
        "FUSION_PATH_SENSITIVE",
    )

    viable_failures: List[str] = []
    _criterion(viable_failures, common, "COMPLETE_FINITE_NO_OOM")
    _criterion(viable_failures, alpha_std is not None and alpha_std >= 0.001, "ALPHA_STD")
    _criterion(viable_failures, pearson is not None and pearson >= 0.30, "PEARSON")
    _criterion(viable_failures, rgb_missing is not None and rgb_missing <= 0.45, "RGB_MISSING")
    _criterion(viable_failures, ir_missing is not None and ir_missing <= 0.45, "IR_MISSING")
    _criterion(
        viable_failures,
        sum(value is not None and value >= 0.002 for value in drops) >= 3,
        "THREE_DEGRADATION_DIRECTIONS",
    )
    _criterion(
        viable_failures,
        clean is not None
        and reference_clean is not None
        and clean >= reference_clean - 0.50,
        "CLEAN_VS_S4",
    )
    _criterion(
        viable_failures,
        degraded_mean is not None
        and reference_degraded is not None
        and degraded_mean >= reference_degraded - 0.20,
        "DEGRADED_VS_S4",
    )
    _criterion(
        viable_failures,
        bool(mechanism.get("scorer_parameter_updated")),
        "SCORER_PARAMETER_UPDATED",
    )
    _criterion(
        viable_failures,
        bool(mechanism.get("fusion_path_sensitive")),
        "FUSION_PATH_SENSITIVE",
    )
    strong = not strong_failures
    viable = strong or not viable_failures
    return {
        "candidate": unit.get("candidate"),
        "checkpoint": unit.get("checkpoint"),
        "complete": bool(unit.get("complete")),
        "has_nan": bool(unit.get("has_nan")),
        "has_inf": bool(unit.get("has_inf")),
        "has_oom": bool(unit.get("has_oom")),
        "clean_R1": clean,
        "degraded_mean_R1": degraded_mean,
        **mechanism,
        "strong_pass": strong,
        "viable_pass": viable,
        "classification": (
            "STRONG_PASS" if strong else "VIABLE_PASS" if viable else "FAIL"
        ),
        "failed_strong_criteria": ",".join(strong_failures),
        "failed_viable_criteria": ",".join(viable_failures),
    }


def rank_candidates(
    rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        level = 0 if row.get("strong_pass") else 1 if row.get("viable_pass") else 2
        return (
            level,
            -(_as_float(row.get("degraded_mean_R1")) or -1e30),
            -(_as_float(row.get("clean_R1")) or -1e30),
            -(_as_float(row.get("alpha_std")) or -1e30),
            -(_as_float(row.get("alpha_target_pearson")) or -1e30),
            str(row.get("candidate")),
            str(row.get("checkpoint")),
        )

    ordered = sorted(rows, key=key)
    return [
        {"rank": index, **dict(row)}
        for index, row in enumerate(ordered, start=1)
    ]


def _unit_dirs(root: Path) -> List[Tuple[str, str, Path]]:
    result = []
    for candidate in STAGE1_MATRIX:
        for checkpoint in CHECKPOINT_KINDS:
            label = f"{candidate}_{checkpoint}"
            result.append((candidate, checkpoint, root / label))
    return result


def summarize_screen(
    candidate_root: Path,
    reference_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    reference = load_reference_s4(reference_root)
    units = [
        collect_unit(path, candidate, checkpoint)
        for candidate, checkpoint, path in _unit_dirs(candidate_root)
    ]
    rows = [classify_candidate(unit, reference) for unit in units]
    ranking = rank_candidates(rows)
    metric_rows = [
        row for unit in units for row in unit.get("metric_rows", [])
    ]
    mechanism_rows = [
        unit["mechanism"] for unit in units if "mechanism" in unit
    ]
    _write_tsv(output_dir / "amplitude_screen_metrics.tsv", METRIC_FIELDS, metric_rows)
    _write_tsv(
        output_dir / "amplitude_screen_mechanism.tsv",
        MECHANISM_FIELDS,
        mechanism_rows,
    )
    _write_tsv(
        output_dir / "checkpoint_comparison.tsv",
        COMPARISON_FIELDS,
        rows,
    )
    _write_tsv(
        output_dir / "amplitude_candidate_ranking.tsv",
        RANKING_FIELDS,
        ranking,
    )
    selected = next(
        (row for row in ranking if bool(row.get("strong_pass"))),
        None,
    )
    viable = next(
        (row for row in ranking if bool(row.get("viable_pass"))),
        None,
    )
    payload = {
        "status": (
            "STRONG_SELECTED"
            if selected is not None
            else "VIABLE_SELECTED_FOR_CALIBRATION"
            if viable is not None
            else "AMPLITUDE_SCREEN_FAILED"
        ),
        "selected_candidate": (
            selected.get("candidate") if selected is not None else None
        ),
        "selected_checkpoint": (
            selected.get("checkpoint") if selected is not None else None
        ),
        "calibration_candidate": (
            viable.get("candidate") if viable is not None else None
        ),
        "calibration_checkpoint": (
            viable.get("checkpoint") if viable is not None else None
        ),
        "mraf_scorer_lr_multiplier": (
            _as_float((selected or viable or {}).get("mraf_scorer_lr_multiplier"))
        ),
        "mraf_gate_logit_scale": 1.0,
        "mraf_rel_lambda": (
            STAGE1_MATRIX.get((selected or viable or {}).get("candidate"), {}).get(
                "mraf_rel_lambda"
            )
        ),
        "mraf_gate_base_grad_scale": (
            STAGE1_MATRIX.get((selected or viable or {}).get("candidate"), {}).get(
                "mraf_gate_base_grad_scale"
            )
        ),
        "reference_s4": reference,
    }
    _write_json(output_dir / "selected_amplitude_candidate.json", payload)
    return payload


def summarize_calibration(
    candidate_root: Path,
    reference_root: Path,
    selected_json: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    reference = load_reference_s4(reference_root)
    selection = _read_json(selected_json)
    source_candidate = str(selection["calibration_candidate"])
    source_checkpoint = str(selection["calibration_checkpoint"])
    rows = []
    for scale in CALIBRATION_SCALES:
        label = f"scale_{format(scale, 'g')}"
        unit = collect_unit(
            candidate_root / label,
            source_candidate,
            source_checkpoint,
        )
        unit["mraf_gate_logit_scale"] = scale
        unit.setdefault("mechanism", {})["mraf_gate_logit_scale"] = scale
        row = classify_candidate(unit, reference)
        row["source_candidate"] = source_candidate
        row["source_checkpoint"] = source_checkpoint
        rows.append(row)
    ranking = rank_candidates(rows)
    _write_tsv(
        output_dir / "calibration_screen.tsv",
        CALIBRATION_FIELDS,
        ranking,
    )
    selected = next(
        (row for row in ranking if bool(row.get("strong_pass"))),
        None,
    )
    payload = dict(selection)
    if selected is None:
        payload.update(
            {
                "status": "CALIBRATION_FAILED",
                "selected_candidate": None,
                "selected_checkpoint": None,
            }
        )
    else:
        payload.update(
            {
                "status": "STRONG_SELECTED",
                "selected_candidate": source_candidate,
                "selected_checkpoint": source_checkpoint,
                "mraf_gate_logit_scale": _as_float(
                    selected.get("mraf_gate_logit_scale")
                ),
                "selection_source": "read_only_calibration",
            }
        )
    _write_json(output_dir / "selected_amplitude_candidate.json", payload)
    return payload


def summarize_confirmation(
    candidate_root: Path,
    reference_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    reference = load_reference_s4(reference_root)
    units = [
        collect_unit(candidate_root / kind, "confirmation", kind)
        for kind in CHECKPOINT_KINDS
    ]
    rows = rank_candidates(
        [classify_candidate(unit, reference) for unit in units]
    )
    _write_tsv(
        output_dir / "confirmation_summary.tsv",
        CONFIRMATION_FIELDS,
        rows,
    )
    selected = next(
        (row for row in rows if bool(row.get("strong_pass"))),
        None,
    )
    return {
        "status": "CONFIRMATION_PASSED" if selected else "CONFIRMATION_FAILED",
        "checkpoint": selected.get("checkpoint") if selected else None,
    }


COMPARISON_OUTPUT_FIELDS = (
    "comparison",
    "root",
    "seed",
    "selected_clean_R1",
    "selected_degraded_mean_R1",
    "reference_candidate",
    "reference_clean_R1",
    "reference_degraded_mean_R1",
    "delta_clean_R1",
    "delta_degraded_mean_R1",
    "source",
    "note",
)


def _discover_old_reference(root: Path) -> Dict[str, Any]:
    if not root.is_dir():
        return {}
    names = (
        "final_robustness_summary.tsv",
        "robustness_summary.tsv",
        "rescue_summary.tsv",
    )
    for name in names:
        for path in sorted(root.rglob(name)):
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle, delimiter="\t"))
            except (OSError, UnicodeError, csv.Error):
                continue
            for row in rows:
                clean = _as_float(row.get("clean_R1"))
                degraded = _as_float(
                    row.get(
                        "degraded_mean_R1",
                        row.get("key_degraded_mean_R1"),
                    )
                )
                if clean is not None and degraded is not None:
                    return {
                        "candidate": row.get(
                            "candidate", row.get("checkpoint", "old_seed1")
                        ),
                        "clean_R1": clean,
                        "degraded_mean_R1": degraded,
                        "source": str(path.resolve()),
                    }
    return {}


def _comparison_rows(
    label: str,
    selected: Mapping[str, Any],
    root: Path,
    reference: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    selected_clean = _as_float(selected.get("clean_R1"))
    selected_degraded = _as_float(selected.get("degraded_mean_R1"))
    reference_clean = _as_float(reference.get("clean_R1"))
    reference_degraded = _as_float(reference.get("degraded_mean_R1"))
    return [
        {
            "comparison": label,
            "root": str(root.resolve()) if root.exists() else str(root),
            "seed": 1,
            "selected_clean_R1": selected_clean,
            "selected_degraded_mean_R1": selected_degraded,
            "reference_candidate": reference.get("candidate", "S4"),
            "reference_clean_R1": reference_clean,
            "reference_degraded_mean_R1": reference_degraded,
            "delta_clean_R1": (
                selected_clean - reference_clean
                if selected_clean is not None and reference_clean is not None
                else None
            ),
            "delta_degraded_mean_R1": (
                selected_degraded - reference_degraded
                if selected_degraded is not None
                and reference_degraded is not None
                else None
            ),
            "source": reference.get("source"),
            "note": "read-only reference; unavailable values are left blank",
        }
    ]


def summarize_final(
    candidate_root: Path,
    reference_root: Path,
    reference_run_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    reference = load_reference_s4(reference_root)
    units = [
        collect_unit(
            candidate_root / kind,
            "formal",
            kind,
            scenarios=FORMAL_SCENARIOS,
        )
        for kind in CHECKPOINT_KINDS
    ]
    rows = rank_candidates(
        [classify_candidate(unit, reference) for unit in units]
    )
    selected = next(
        (row for row in rows if bool(row.get("strong_pass"))),
        None,
    )
    metric_rows = [
        row for unit in units for row in unit.get("metric_rows", [])
    ]
    mechanism_rows = [
        unit["mechanism"] for unit in units if "mechanism" in unit
    ]
    _write_tsv(output_dir / "final_metrics.tsv", METRIC_FIELDS, metric_rows)
    _write_tsv(
        output_dir / "final_robustness_summary.tsv",
        COMPARISON_FIELDS,
        rows,
    )
    _write_tsv(
        output_dir / "final_mechanism_summary.tsv",
        MECHANISM_FIELDS,
        mechanism_rows,
    )
    _write_tsv(
        output_dir / "comparison_with_previous_screen.tsv",
        COMPARISON_OUTPUT_FIELDS,
        _comparison_rows(
            "previous_S4_screen",
            selected or {},
            reference_root,
            {"candidate": "S4", **reference},
        ),
    )
    old_reference = _discover_old_reference(reference_run_root)
    _write_tsv(
        output_dir / "comparison_with_old_seed1.tsv",
        COMPARISON_OUTPUT_FIELDS,
        _comparison_rows(
            "old_seed1",
            selected or {},
            reference_run_root,
            old_reference,
        ),
    )
    status = "COMPLETED" if selected is not None else "FAILED"
    selection_path = output_dir / "selected_amplitude_candidate.json"
    selection = _read_json(selection_path) if selection_path.is_file() else {}
    selected_scale = _as_float(
        selection.get("mraf_gate_logit_scale")
        if isinstance(selection, Mapping)
        else None
    )
    screen_mechanism = _read_tsv(
        output_dir / "amplitude_screen_mechanism.tsv"
    )
    observed_amplitudes = [
        (
            _as_float(row.get("mraf_scorer_lr_multiplier")),
            _as_float(row.get("raw_logit_gap_std")),
        )
        for row in screen_mechanism
    ]
    observed_amplitudes = [
        pair
        for pair in observed_amplitudes
        if pair[0] is not None and pair[1] is not None
    ]
    lr_amplitude_summary = (
        "unavailable"
        if not observed_amplitudes
        else (
            f"min_lr_multiplier={min(observed_amplitudes)[0]}:"
            f"raw_gap_std={min(observed_amplitudes)[1]:.8g};"
            f"max_lr_multiplier={max(observed_amplitudes)[0]}:"
            f"raw_gap_std={max(observed_amplitudes)[1]:.8g}"
        )
    )
    report = [
        f"STATUS={status}",
        f"selected_checkpoint={(selected or {}).get('checkpoint', 'NONE')}",
        f"scorer_lr_amplitude_observation={lr_amplitude_summary}",
        (
            "best_vs_last="
            f"{(selected or {}).get('checkpoint', 'NONE')} selected by strict "
            "mechanism plus clean/degraded ranking"
        ),
        f"logit_scale_required={str((selected_scale or 1.0) != 1.0).lower()}",
        (
            "pearson_and_amplitude_simultaneously_passed="
            f"{str(selected is not None).lower()}"
        ),
        (
            "missing_modality_downweighted="
            f"{str(selected is not None).lower()}"
        ),
        (
            "clean_and_robustness_preserved_vs_S4="
            f"{str(selected is not None).lower()}"
        ),
        f"eligible_for_seed2_to_5={str(selected is not None).lower()}",
        "all_results_seed=1",
        "warning=short-cycle screening is not a final paper conclusion",
    ]
    (output_dir / "FINAL_REPORT.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    return {"status": status, "selected": selected}


def create_mock_outputs(output_dir: Path) -> Dict[str, Any]:
    """Create deterministic schema fixtures without importing torch or training."""
    output_dir.mkdir(parents=True, exist_ok=False)
    unit_rows = []
    metric_rows = []
    mechanism_rows = []
    for index, candidate in enumerate(STAGE1_MATRIX):
        for checkpoint in CHECKPOINT_KINDS:
            strong = candidate == "A50" and checkpoint == "last"
            row = {
                "candidate": candidate,
                "checkpoint": checkpoint,
                "complete": True,
                "has_nan": False,
                "has_inf": False,
                "has_oom": False,
                "clean_R1": 49.9,
                "degraded_mean_R1": 42.0,
                **STAGE1_MATRIX[candidate],
                "alpha_std": 0.006 if strong else 0.0015,
                "alpha_target_pearson": 0.5,
                "rgb_missing_alpha_rgb": 0.3 if strong else 0.4,
                "ir_missing_alpha_ir": 0.3 if strong else 0.4,
                "rgb_noise_alpha_drop": 0.03 if strong else 0.003,
                "ir_noise_alpha_drop": 0.03 if strong else 0.003,
                "rgb_blur_alpha_drop": 0.03 if strong else 0.003,
                "ir_blur_alpha_drop": 0.03 if strong else 0.003,
                "weighted_rel_base_ratio": 0.01,
                "scorer_parameter_delta": 0.1,
                "raw_logit_gap_std": 0.02,
                "scaled_logit_gap_std": 0.02,
                "scorer_parameter_updated": True,
                "fusion_path_sensitive": True,
                "gate_logits_fp32": True,
                "strong_pass": strong,
                "viable_pass": True,
                "classification": "STRONG_PASS" if strong else "VIABLE_PASS",
                "failed_strong_criteria": "" if strong else "ALPHA_STD",
                "failed_viable_criteria": "",
            }
            unit_rows.append(row)
            mechanism_rows.append(row)
            for scenario in SCREEN_SCENARIOS:
                metric_rows.append(
                    {
                        "candidate": candidate,
                        "checkpoint": checkpoint,
                        "scenario": scenario,
                        "R1": 49.9 if scenario == "clean" else 42.0,
                        "R5": 70.0,
                        "R10": 80.0,
                        "mAP": 45.0,
                        "mINP": 30.0,
                        "score_file": "MOCK",
                    }
                )
    ranking = rank_candidates(unit_rows)
    _write_tsv(output_dir / "amplitude_screen_metrics.tsv", METRIC_FIELDS, metric_rows)
    _write_tsv(output_dir / "amplitude_screen_mechanism.tsv", MECHANISM_FIELDS, mechanism_rows)
    _write_tsv(output_dir / "checkpoint_comparison.tsv", COMPARISON_FIELDS, unit_rows)
    _write_tsv(output_dir / "amplitude_candidate_ranking.tsv", RANKING_FIELDS, ranking)
    _write_tsv(
        output_dir / "calibration_screen.tsv",
        CALIBRATION_FIELDS,
        [
            {
                "rank": index,
                "source_candidate": "A50",
                "source_checkpoint": "last",
                "mraf_gate_logit_scale": scale,
                "classification": "STRONG_PASS" if scale == 5.0 else "FAIL",
                "strong_pass": scale == 5.0,
                "clean_R1": 49.9,
                "degraded_mean_R1": 42.0,
                "alpha_std": 0.006,
                "alpha_target_pearson": 0.5,
                "failed_strong_criteria": "" if scale == 5.0 else "MOCK",
            }
            for index, scale in enumerate(CALIBRATION_SCALES, start=1)
        ],
    )
    selection = {
        "status": "STRONG_SELECTED",
        "selected_candidate": "A50",
        "selected_checkpoint": "last",
        "mraf_scorer_lr_multiplier": 50.0,
        "mraf_gate_logit_scale": 5.0,
        "mraf_rel_lambda": 0.25,
        "mraf_gate_base_grad_scale": 0.10,
        "selection_source": "mock",
    }
    _write_json(output_dir / "selected_amplitude_candidate.json", selection)
    confirmation = [dict(unit_rows[0], candidate="confirmation")]
    confirmation[0].update(strong_pass=True, classification="STRONG_PASS")
    _write_tsv(output_dir / "confirmation_summary.tsv", CONFIRMATION_FIELDS, confirmation)
    formal_metrics = [
        dict(metric_rows[0], candidate="formal", checkpoint="best", scenario=scenario)
        for scenario in FORMAL_SCENARIOS
    ]
    _write_tsv(output_dir / "final_metrics.tsv", METRIC_FIELDS, formal_metrics)
    _write_tsv(output_dir / "final_robustness_summary.tsv", COMPARISON_FIELDS, confirmation)
    _write_tsv(output_dir / "final_mechanism_summary.tsv", MECHANISM_FIELDS, confirmation)
    _write_tsv(
        output_dir / "comparison_with_previous_screen.tsv",
        COMPARISON_OUTPUT_FIELDS,
        _comparison_rows(
            "mock_previous",
            confirmation[0],
            Path("MOCK"),
            {
                "candidate": "S4",
                "clean_R1": 49.88,
                "degraded_mean_R1": 41.796,
                "source": "MOCK",
            },
        ),
    )
    _write_tsv(
        output_dir / "comparison_with_old_seed1.tsv",
        COMPARISON_OUTPUT_FIELDS,
        _comparison_rows(
            "mock_old",
            confirmation[0],
            Path("MOCK"),
            {
                "candidate": "old_seed1",
                "clean_R1": 49.0,
                "degraded_mean_R1": 41.0,
                "source": "MOCK",
            },
        ),
    )
    (output_dir / "FINAL_REPORT.txt").write_text(
        "STATUS=MOCK_COMPLETED\nmock=true\nall_results_seed=1\n"
        "warning=short-cycle screening is not a final paper conclusion\n",
        encoding="utf-8",
    )
    return selection


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    if args.mode == "mock":
        create_mock_outputs(output_dir)
        print(f"mock_output_dir={output_dir.resolve()}")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_root = Path(args.candidate_root)
    reference_root = Path(args.reference_screen_root)
    if args.mode == "screen":
        result = summarize_screen(candidate_root, reference_root, output_dir)
    elif args.mode == "calibration":
        result = summarize_calibration(
            candidate_root,
            reference_root,
            Path(args.selected_json),
            output_dir,
        )
    elif args.mode == "confirmation":
        result = summarize_confirmation(
            candidate_root,
            reference_root,
            output_dir,
        )
    else:
        result = summarize_final(
            candidate_root,
            reference_root,
            Path(args.reference_run_root),
            output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
