#!/usr/bin/env python3
"""Summarize and gate the MRAF gate-amplitude rescue pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

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
from utils.config import load_config_mapping  # noqa: E402


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
EXPECTED_ORIGINAL_COMMIT = "c667946b2c51740bfa7892f5a2daee73d1279b44"
ALLOWED_HOTFIX_PATHS = frozenset(
    {
        "IRRA-main/tools/revision/summarize_mraf_gate_amplitude_rescue.py",
        "IRRA-main/processor/processor.py",
        "IRRA-main/scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh",
        "IRRA-main/tests/test_mraf_gate_amplitude_rescue.py",
        "IRRA-main/docs/revision/mraf_gate_amplitude_rescue_pipeline.md",
    }
)
LEGACY_MERGED_TSV_FIELDS: Mapping[str, Tuple[str, str]] = {
    "target_rgb_meantarget_rgb_std": (
        "target_rgb_mean",
        "target_rgb_std",
    ),
    "scorer_first_weight_normscorer_last_weight_norm": (
        "scorer_first_weight_norm",
        "scorer_last_weight_norm",
    ),
}
REFERENCE_CLEAN_ALIASES = (
    "clean_R1",
    "clean_r1",
    "clean_rank1",
    "R1_clean",
)
REFERENCE_DEGRADED_ALIASES = (
    "key_degraded_mean_R1",
    "degraded_mean_R1",
    "mean_degraded_R1",
    "key_degraded_R1_mean",
)
REFERENCE_RELATIVE_PATHS = (
    Path("screen/rescue_screen_summary.tsv"),
    Path("rescue_screen_summary.tsv"),
    Path("screen/rescue_screen_metrics.tsv"),
    Path("rescue_screen_metrics.tsv"),
    Path("screen/rescue_candidate_ranking.tsv"),
    Path("rescue_candidate_ranking.tsv"),
    Path("screen/comparison_with_previous_screen.tsv"),
    Path("comparison_with_previous_screen.tsv"),
)
RESUME_PROTOCOL: Mapping[str, Any] = {
    "batch_size": 64,
    "test_batch_size": 512,
    "loss_names": "sdm+mlm+id",
    "MLM": True,
    "warmup_epochs": 5,
    "eval_use_fused_mu": True,
    "degrade_aug_mode": "none",
    "mraf_enabled": True,
    "mraf_gate_fp32": True,
    "mdrc_enable": True,
    "mdrc_prob": 0.5,
    "mdrc_lambda": 0.05,
    "mdrc_loss_scale": 100,
    "mdrc_warmup_epoch": 3,
    "num_epoch": 10,
    "mraf_rel_warmup_epoch": 1,
}

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
    "legacy_header_repaired",
    "legacy_header_path",
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
        choices=(
            "screen",
            "calibration",
            "confirmation",
            "final",
            "mock",
            "resume-plan",
        ),
        required=True,
    )
    parser.add_argument("--candidate-root", default="")
    parser.add_argument("--reference-screen-root", default="")
    parser.add_argument("--reference-run-root", default="")
    parser.add_argument("--selected-json", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT.parent))
    parser.add_argument("--resume-code-commit", default="")
    parser.add_argument("--record-resume", action="store_true")
    return parser


def _validate_fields(fields: Sequence[str]) -> Tuple[str, ...]:
    normalized = tuple(str(field) for field in fields)
    if not normalized or any(not field.strip() for field in normalized):
        raise ValueError("TSV schema contains an empty field name")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"TSV schema contains duplicate fields: {normalized}")
    return normalized


def _atomic_replace_text(
    path: Path,
    write: Callable[[Any], None],
    validate: Optional[Callable[[Path], None]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if validate is not None:
            validate(Path(temporary_name))
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise


def _write_tsv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    schema = _validate_fields(fields)
    materialized = [dict(row) for row in rows]

    def write(handle: Any) -> None:
        writer = csv.DictWriter(
            handle,
            fieldnames=schema,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: row.get(field) for field in schema})

    def validate(temporary: Path) -> None:
        with temporary.open("r", encoding="utf-8", newline="") as handle:
            parsed = list(csv.reader(handle, delimiter="\t"))
        if not parsed or tuple(parsed[0]) != schema:
            raise ValueError(
                f"Atomic TSV validation failed for {path}: header mismatch"
            )
        if any(len(row) != len(schema) for row in parsed[1:]):
            raise ValueError(
                f"Atomic TSV validation failed for {path}: row width mismatch"
            )

    _atomic_replace_text(path, write, validate)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def write(handle: Any) -> None:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    def validate(temporary: Path) -> None:
        with temporary.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        if not isinstance(parsed, Mapping):
            raise ValueError(f"Atomic JSON validation failed for {path}")

    _atomic_replace_text(path, write, validate)


def _write_text(path: Path, text: str) -> None:
    _atomic_replace_text(path, lambda handle: handle.write(text))


@dataclass(frozen=True)
class TSVReadResult:
    rows: List[Dict[str, str]]
    fields: Tuple[str, ...]
    original_fields: Tuple[str, ...]
    legacy_header_repaired: bool
    source: str


def read_tsv_compatible(path: Path) -> TSVReadResult:
    """Read a standard TSV or split only the two known legacy merged headers."""
    if not path.is_file():
        return TSVReadResult([], (), (), False, str(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.reader(handle, delimiter="\t"))
    if not raw_rows:
        return TSVReadResult([], (), (), False, str(path.resolve()))
    original_fields = tuple(field.strip() for field in raw_rows[0])
    _validate_fields(original_fields)
    data_rows = raw_rows[1:]
    widths = {len(row) for row in data_rows}
    if len(widths) > 1:
        raise ValueError(
            f"Inconsistent TSV row widths in {path}: header={len(original_fields)}, "
            f"data_widths={sorted(widths)}"
        )
    data_width = next(iter(widths), len(original_fields))
    repaired_fields: List[str] = []
    repaired = False
    for field in original_fields:
        replacement = LEGACY_MERGED_TSV_FIELDS.get(field)
        if replacement is None:
            repaired_fields.append(field)
        else:
            repaired_fields.extend(replacement)
            repaired = True
    fields = _validate_fields(repaired_fields)
    if len(fields) != data_width:
        raise ValueError(
            f"Unsupported TSV schema mismatch in {path}: "
            f"original_header={list(original_fields)}, repaired_header={list(fields)}, "
            f"header_columns={len(fields)}, data_columns={data_width}; "
            "only whitelisted legacy merged headers may be repaired"
        )
    rows = [dict(zip(fields, row)) for row in data_rows]
    return TSVReadResult(
        rows=rows,
        fields=fields,
        original_fields=original_fields,
        legacy_header_repaired=repaired,
        source=str(path.resolve()),
    )


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    return read_tsv_compatible(path).rows


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
    epoch_result = read_tsv_compatible(unit_dir / "mraf_epoch_metrics.tsv")
    epoch_rows = epoch_result.rows
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
        "legacy_header_repaired": epoch_result.legacy_header_repaired,
        "legacy_header_path": (
            epoch_result.source if epoch_result.legacy_header_repaired else ""
        ),
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


def _parse_finite_reference_value(
    row: Mapping[str, str],
    aliases: Sequence[str],
    metric_name: str,
) -> Tuple[float, str]:
    for alias in aliases:
        if alias not in row:
            continue
        raw = row.get(alias)
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{metric_name} field {alias!r} cannot be converted from "
                f"{raw!r}: {error}"
            ) from error
        if not math.isfinite(value):
            raise ValueError(
                f"{metric_name} field {alias!r} must be finite, got {raw!r}"
            )
        return value, alias
    raise KeyError(
        f"{metric_name} field missing; accepted aliases={list(aliases)}"
    )


def _reference_from_tsv(path: Path) -> Tuple[float, float, Tuple[str, ...]]:
    fields: Tuple[str, ...] = ()
    s4_found = False
    s4_fields: Tuple[str, ...] = ()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError("TSV has no header")
            normalized_fields = tuple(field.strip() for field in reader.fieldnames)
            _validate_fields(normalized_fields)
            fields = normalized_fields
            rows = []
            for raw_row in reader:
                normalized_row = {
                    str(key).strip(): value
                    for key, value in raw_row.items()
                    if key is not None
                }
                rows.append(normalized_row)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        raise ValueError(
            f"path={path}; headers={list(fields)}; s4_found={s4_found}; "
            f"s4_fields={list(s4_fields)}; reason={type(error).__name__}: {error}"
        ) from error
    for row in rows:
        if str(row.get("candidate", "")).strip() != "S4":
            continue
        s4_found = True
        s4_fields = tuple(sorted(row))
        try:
            clean, _ = _parse_finite_reference_value(
                row, REFERENCE_CLEAN_ALIASES, "clean"
            )
            degraded, _ = _parse_finite_reference_value(
                row, REFERENCE_DEGRADED_ALIASES, "degraded"
            )
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"path={path}; headers={list(fields)}; s4_found=true; "
                f"s4_fields={list(s4_fields)}; reason={type(error).__name__}: {error}"
            ) from error
        return clean, degraded, fields
    raise ValueError(
        f"path={path}; headers={list(fields)}; s4_found=false; "
        f"s4_fields=[]; reason=exact candidate 'S4' was not found"
    )


def load_reference_s4(reference_root: Path) -> Dict[str, Any]:
    """Read historical S4 summaries without modifying the reference tree."""
    if not reference_root.is_dir():
        raise FileNotFoundError(
            f"Reference screen root not found: {reference_root}"
        )
    searched: List[Path] = []
    candidates: List[Path] = []
    seen = set()
    for relative in REFERENCE_RELATIVE_PATHS:
        path = reference_root / relative
        searched.append(path)
        if path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            candidates.append(path)
    for name in (
        "rescue_screen_summary.tsv",
        "rescue_screen_metrics.tsv",
        "rescue_summary.tsv",
        "rescue_candidate_ranking.tsv",
        "comparison_with_previous_screen.tsv",
    ):
        for path in sorted(reference_root.rglob(name)):
            if path not in searched:
                searched.append(path)
            if path.is_file() and path.resolve() not in seen:
                seen.add(path.resolve())
                candidates.append(path)

    errors: List[str] = []
    for path in candidates:
        try:
            clean, degraded, _ = _reference_from_tsv(path)
            return {
                "clean_R1": clean,
                "key_degraded_mean_R1": degraded,
                "degraded_mean_R1": degraded,
                "source": str(path.resolve()),
            }
        except ValueError as error:
            errors.append(str(error))
    raise ValueError(
        "Could not read finite S4 clean/degraded metrics. "
        f"reference_root={reference_root.resolve()}; "
        f"searched_paths={[str(path) for path in searched]}; "
        f"file_diagnostics={errors or ['no candidate TSV file exists']}"
    )


def _read_key_value_file(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required key-value file is missing: {path}")
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _numbers_equal(first: Any, second: Any) -> bool:
    left = _as_float(first)
    right = _as_float(second)
    return (
        left is not None
        and right is not None
        and math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
    )


def _git_is_ancestor(repo_root: Path, original: str, current: str) -> bool:
    command = _git_command_prefix(repo_root)
    completed = subprocess.run(
        [*command, "merge-base", "--is-ancestor", original, current],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.returncode == 0


def _git_changed_paths(
    repo_root: Path, original: str, current: str
) -> Sequence[str]:
    command = _git_command_prefix(repo_root)
    completed = subprocess.run(
        [*command, "diff", "--name-only", f"{original}..{current}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def _git_command_prefix(repo_root: Path) -> List[str]:
    """Support normal repos and Windows worktree pointers invoked through WSL."""
    pointer = repo_root / ".git"
    if not pointer.is_file():
        return ["git", "-C", str(repo_root)]
    first_line = pointer.read_text(encoding="utf-8-sig").splitlines()[0].strip()
    if not first_line.startswith("gitdir:"):
        return ["git", "-C", str(repo_root)]
    raw = first_line.split(":", 1)[1].strip()
    match = re.match(r"^([A-Za-z]):[/\\](.*)$", raw)
    if match and os.name != "nt":
        git_dir = Path(f"/mnt/{match.group(1).lower()}/{match.group(2)}")
    else:
        git_dir = Path(raw)
        if not git_dir.is_absolute():
            git_dir = (repo_root / git_dir).resolve()
    return [
        "git",
        f"--git-dir={git_dir}",
        f"--work-tree={repo_root}",
    ]


@dataclass(frozen=True)
class ResumePlan:
    valid: bool
    run_root: str
    original_commit: str
    resume_commit: str
    old_status: str
    old_failed_stage: str
    next_stage: str
    reasons: Tuple[str, ...]
    training_candidates_skipped: int = 0
    checkpoint_units_skipped: int = 0
    evaluations_skipped: int = 0
    diagnostics_skipped: int = 0

    def machine_lines(self) -> List[str]:
        lines = [
            f"RESUME_PLAN_VALID={str(self.valid).lower()}",
            f"ORIGINAL_RUN_COMMIT={self.original_commit or 'UNKNOWN'}",
            f"RESUME_CODE_COMMIT={self.resume_commit or 'UNKNOWN'}",
            (
                "STAGE1_TRAIN_CANDIDATES_SKIPPED="
                f"{self.training_candidates_skipped}"
            ),
            f"STAGE1_CHECKPOINT_UNITS_SKIPPED={self.checkpoint_units_skipped}",
            f"STAGE1_EVALUATIONS_SKIPPED={self.evaluations_skipped}",
            f"STAGE1_DIAGNOSTICS_SKIPPED={self.diagnostics_skipped}",
            f"NEXT_STAGE={self.next_stage or 'NONE'}",
            f"RUN_ROOT={self.run_root}",
        ]
        lines.extend(f"RESUME_PLAN_ERROR={reason}" for reason in self.reasons)
        return lines


def validate_resume_plan(
    run_root: Path,
    repo_root: Path,
    resume_commit: str,
    *,
    ancestry_checker: Optional[Callable[[Path, str, str], bool]] = None,
    changed_paths_reader: Optional[
        Callable[[Path, str, str], Sequence[str]]
    ] = None,
) -> ResumePlan:
    """Validate an aggregation-only Stage-1 resume without modifying artifacts."""
    reasons: List[str] = []
    try:
        manifest = _read_key_value_file(run_root / "experiment_manifest.txt")
    except (OSError, UnicodeError) as error:
        manifest = {}
        reasons.append(str(error))
    try:
        status = _read_key_value_file(run_root / "status.txt")
    except (OSError, UnicodeError) as error:
        status = {}
        reasons.append(str(error))

    original = manifest.get("git_head", "")
    old_status = status.get("STATUS", "")
    old_failed_stage = status.get("FAILED_STAGE", "")
    if manifest.get("PIPELINE_ID") != "mraf_gate_amplitude_rescue_v3":
        reasons.append("experiment_manifest PIPELINE_ID is not mraf_gate_amplitude_rescue_v3")
    if original != EXPECTED_ORIGINAL_COMMIT:
        reasons.append(
            f"original commit must be {EXPECTED_ORIGINAL_COMMIT}, got {original or 'missing'}"
        )
    if old_status != "FAILED" or old_failed_stage != "amplitude_screen":
        reasons.append(
            "resume requires STATUS=FAILED and FAILED_STAGE=amplitude_screen, "
            f"got STATUS={old_status or 'missing'} "
            f"FAILED_STAGE={old_failed_stage or 'missing'}"
        )
    if manifest.get("seed") != "1":
        reasons.append(f"resume seed must remain 1, got {manifest.get('seed', 'missing')}")
    for key in ("data_root", "base_config", "reference_screen_root", "reference_run_root"):
        if not manifest.get(key):
            reasons.append(f"experiment_manifest lacks non-empty {key}")
    reference_value = manifest.get("reference_screen_root", "")
    if reference_value:
        try:
            load_reference_s4(Path(reference_value))
        except (OSError, ValueError) as error:
            reasons.append(f"reference S4 validation failed: {error}")

    if original and resume_commit and resume_commit != original:
        ancestry = ancestry_checker or _git_is_ancestor
        changed_reader = changed_paths_reader or _git_changed_paths
        try:
            if not ancestry(repo_root, original, resume_commit):
                reasons.append(
                    f"resume commit {resume_commit} is not a descendant of {original}"
                )
            changed = set(changed_reader(repo_root, original, resume_commit))
            disallowed = sorted(changed - ALLOWED_HOTFIX_PATHS)
            if disallowed:
                reasons.append(
                    f"cross-commit resume contains non-hotfix paths: {disallowed}"
                )
        except (OSError, subprocess.SubprocessError) as error:
            reasons.append(f"Git compatibility validation failed: {error}")
    elif not resume_commit:
        reasons.append("resume code commit is empty")

    candidate_status_path = run_root / "candidate_status.tsv"
    try:
        status_rows = read_tsv_compatible(candidate_status_path).rows
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        status_rows = []
        reasons.append(f"candidate status cannot be read: {error}")
    latest_status: Dict[Tuple[str, str], str] = {}
    for row in status_rows:
        key = (
            str(row.get("candidate", "")).strip(),
            str(row.get("checkpoint", "")).strip(),
        )
        latest_status[key] = str(row.get("status", "")).strip()

    completed_units = 0
    completed_evaluations = 0
    completed_diagnostics = 0
    completed_train_candidates = 0
    train_root = run_root / "stage1" / "train"
    unit_root = run_root / "stage1" / "units"
    for candidate, expected in STAGE1_MATRIX.items():
        train_dir = train_root / candidate
        if train_dir.is_dir() and all(
            any(path.is_file() and path.stat().st_size > 0 for path in train_dir.rglob(name))
            for name in ("configs.yaml", "best.pth", "last.pth")
        ):
            completed_train_candidates += 1
        else:
            reasons.append(f"{candidate}: complete Stage-1 training artifacts are missing")
        for checkpoint in CHECKPOINT_KINDS:
            label = f"{candidate}_{checkpoint}"
            unit = unit_root / label
            if latest_status.get((candidate, checkpoint)) != "COMPLETED":
                reasons.append(
                    f"{label}: candidate_status is "
                    f"{latest_status.get((candidate, checkpoint), 'missing')}, expected COMPLETED"
                )
            required_files = (
                "checkpoint.pth",
                "configs.yaml",
                "train.log",
                "mraf_epoch_metrics.tsv",
                "checkpoint_selection.tsv",
                "unit.json",
            )
            unit_complete = True
            for name in required_files:
                path = unit / name
                if not path.is_file() or path.stat().st_size <= 0:
                    reasons.append(f"{label}: missing or empty {path}")
                    unit_complete = False
            metadata: Mapping[str, Any] = {}
            metadata_path = unit / "unit.json"
            if metadata_path.is_file():
                try:
                    loaded = _read_json(metadata_path)
                    metadata = loaded if isinstance(loaded, Mapping) else {}
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    reasons.append(f"{label}: invalid unit.json: {error}")
                    unit_complete = False
            if metadata:
                if str(metadata.get("candidate", "")).strip() != candidate:
                    reasons.append(f"{label}: unit candidate matrix label changed")
                    unit_complete = False
                if str(metadata.get("checkpoint", "")).strip() != checkpoint:
                    reasons.append(f"{label}: unit checkpoint label changed")
                    unit_complete = False
                for field in (
                    "mraf_rel_lambda",
                    "mraf_gate_base_grad_scale",
                    "mraf_scorer_lr_multiplier",
                    "mraf_gate_logit_scale",
                ):
                    if not _numbers_equal(metadata.get(field), expected[field]):
                        reasons.append(
                            f"{label}: {field} differs from fixed Stage-1 matrix"
                        )
                        unit_complete = False
                for flag in ("has_nan", "has_inf", "has_oom"):
                    if _is_true(metadata.get(flag, False)):
                        reasons.append(f"{label}: {flag}=true")
                        unit_complete = False
            config_path = unit / "configs.yaml"
            if config_path.is_file():
                try:
                    config = load_config_mapping(str(config_path))
                except (OSError, ValueError, TypeError) as error:
                    config = {}
                    reasons.append(f"{label}: configs.yaml cannot be loaded: {error}")
                    unit_complete = False
                for field, expected_value in RESUME_PROTOCOL.items():
                    actual = config.get(field)
                    matches = (
                        _is_true(actual) == expected_value
                        if isinstance(expected_value, bool)
                        else _numbers_equal(actual, expected_value)
                        if isinstance(expected_value, (int, float))
                        else str(actual).strip() == str(expected_value)
                    )
                    if not matches:
                        reasons.append(
                            f"{label}: protocol {field}={actual!r}, "
                            f"expected {expected_value!r}"
                        )
                        unit_complete = False
                for field in (
                    "mraf_rel_lambda",
                    "mraf_gate_base_grad_scale",
                    "mraf_scorer_lr_multiplier",
                    "mraf_gate_logit_scale",
                ):
                    if not _numbers_equal(config.get(field), expected[field]):
                        reasons.append(
                            f"{label}: config {field} differs from fixed Stage-1 matrix"
                        )
                        unit_complete = False
            evaluation_complete = True
            for scenario in SCREEN_SCENARIOS:
                score = _scenario_score(unit, scenario)
                if score is None or not score.is_file() or score.stat().st_size <= 0:
                    reasons.append(f"{label}: missing non-empty evaluation for {scenario}")
                    evaluation_complete = False
            diagnostic = _active_diagnostic_dir(unit)
            diagnostic_complete = True
            for name in (
                "diagnosis.json",
                "scenario_gate_statistics.csv",
                "gate_direction_statistics.csv",
                "gradient_diagnostics.csv",
                "fusion_path_check.json",
            ):
                path = diagnostic / name
                if not path.is_file() or path.stat().st_size <= 0:
                    reasons.append(f"{label}: missing or empty diagnostic {path}")
                    diagnostic_complete = False
            if unit_complete:
                completed_units += 1
            if evaluation_complete:
                completed_evaluations += 1
            if diagnostic_complete:
                completed_diagnostics += 1

    valid = not reasons
    return ResumePlan(
        valid=valid,
        run_root=str(run_root.resolve()),
        original_commit=original,
        resume_commit=resume_commit,
        old_status=old_status,
        old_failed_stage=old_failed_stage,
        next_stage="amplitude_screen" if valid else "",
        reasons=tuple(reasons),
        training_candidates_skipped=completed_train_candidates,
        checkpoint_units_skipped=completed_units,
        evaluations_skipped=completed_evaluations,
        diagnostics_skipped=completed_diagnostics,
    )


RESUME_HISTORY_FIELDS = (
    "timestamp",
    "original_commit",
    "resume_commit",
    "old_status",
    "old_failed_stage",
    "next_stage",
    "compatibility_reason",
)


def record_resume_history(run_root: Path, plan: ResumePlan) -> None:
    if not plan.valid:
        raise ValueError("Cannot record an invalid resume plan")
    path = run_root / "resume_history.tsv"
    rows = _read_tsv(path)
    rows.append(
        {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "original_commit": plan.original_commit,
            "resume_commit": plan.resume_commit,
            "old_status": plan.old_status,
            "old_failed_stage": plan.old_failed_stage,
            "next_stage": plan.next_stage,
            "compatibility_reason": (
                "strict aggregation-only hotfix descendant; "
                "completed Stage-1 artifacts verified"
            ),
        }
    )
    _write_tsv(path, RESUME_HISTORY_FIELDS, rows)


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
    manifest_path = output_dir / "experiment_manifest.txt"
    manifest = (
        _read_key_value_file(manifest_path) if manifest_path.is_file() else {}
    )
    history_rows = _read_tsv(output_dir / "resume_history.tsv")
    training_commit = manifest.get("git_head", "UNKNOWN")
    aggregation_commit = (
        history_rows[-1].get("resume_commit", training_commit)
        if history_rows
        else training_commit
    )
    report = [
        f"STATUS={status}",
        f"training_commit={training_commit}",
        f"resume_aggregation_commit={aggregation_commit}",
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
    _write_text(output_dir / "FINAL_REPORT.txt", "\n".join(report) + "\n")
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
    _write_text(
        output_dir / "FINAL_REPORT.txt",
        "STATUS=MOCK_COMPLETED\nmock=true\nall_results_seed=1\n"
        "warning=short-cycle screening is not a final paper conclusion\n"
    )
    return selection


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "resume-plan":
        if not args.run_root:
            raise SystemExit("--run-root is required for --mode resume-plan")
        resume_commit = args.resume_code_commit
        if not resume_commit:
            completed = subprocess.run(
                [*_git_command_prefix(Path(args.repo_root)), "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            resume_commit = completed.stdout.strip()
        plan = validate_resume_plan(
            Path(args.run_root),
            Path(args.repo_root),
            resume_commit,
        )
        for line in plan.machine_lines():
            print(line)
        if plan.valid and args.record_resume:
            record_resume_history(Path(args.run_root), plan)
        return 0 if plan.valid else 2
    if not args.output_dir:
        raise SystemExit("--output-dir is required unless --mode resume-plan")
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
