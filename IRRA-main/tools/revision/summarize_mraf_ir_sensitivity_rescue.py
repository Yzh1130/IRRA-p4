#!/usr/bin/env python3
"""Summarize the read-only MRAF IR-sensitivity rescue V4 pipeline.

This module deliberately reuses the V3 unit collector and the unchanged
STRONG/VIABLE classifier.  It never imports the training entrypoint, mutates a
checkpoint, or writes below the V3 source run-root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.revision import summarize_mraf_gate_amplitude_rescue as v3


PIPELINE_ID = "mraf_ir_sensitivity_rescue_v4"
V3_TRAINING_COMMIT = "c667946b2c51740bfa7892f5a2daee73d1279b44"
SOURCE_STATUS = "CONFIRMATION_FAILED"
SOURCE_FAILED_STAGE = "confirmation"
SOURCE_EXIT_CODE = "5"
CHECKPOINTS = ("best", "last")
READONLY_SCALES = (2.5, 3.0, 4.0, 5.0)
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
    "both_noise_0.10",
    "both_blur_k7",
)
FORMAL_SEEDS = (1, 2, 3, 4, 5)
TERMINAL_STATUSES = frozenset(
    {
        "COMPLETED",
        "IR_SENSITIVITY_RESCUE_FAILED",
        "IR_SENSITIVITY_CONFIRMATION_FAILED",
        "FORMAL_SEED1_FAILED",
        "MULTISEED_FAILED",
        "FAILED",
    }
)

LEGACY_MERGED_FIELDS: Mapping[str, Tuple[str, str]] = {
    **v3.LEGACY_MERGED_TSV_FIELDS,
    "mraf_rel_lambdamraf_gate_base_grad_scale": (
        "mraf_rel_lambda",
        "mraf_gate_base_grad_scale",
    ),
    "scorer_parameter_deltaraw_logit_gap_std": (
        "scorer_parameter_delta",
        "raw_logit_gap_std",
    ),
}

METRIC_FIELDS = v3.METRIC_FIELDS
MECHANISM_FIELDS = v3.MECHANISM_FIELDS
SUMMARY_FIELDS = v3.COMPARISON_FIELDS
RANKING_FIELDS = ("rank", *SUMMARY_FIELDS)
SCALE_RANKING_FIELDS = (
    "rank",
    "scale",
    *SUMMARY_FIELDS,
)
SELECTION_FIELDS = (
    "status",
    "selection_source",
    "selected_scale",
    "selected_checkpoint",
    "classification",
    "clean_R1",
    "degraded_mean_R1",
    "mraf_scorer_lr_multiplier",
    "mraf_rel_lambda",
    "mraf_gate_base_grad_scale",
)
STAGE_STATUS_FIELDS = (
    "timestamp",
    "stage",
    "status",
    "exit_code",
    "details",
)
SEED_STATUS_FIELDS = (
    "timestamp",
    "seed",
    "status",
    "exit_code",
    "selected_checkpoint",
    "classification",
)
RESUME_HISTORY_FIELDS = (
    "timestamp",
    "previous_status",
    "previous_stage",
    "resume_commit",
    "next_stage",
    "reason",
)
SEED_LEVEL_FIELDS = (
    "seed",
    "selected_checkpoint",
    "classification",
    "clean_R1",
    "clean_R5",
    "clean_R10",
    "clean_mAP",
    "clean_mINP",
    "degraded_mean_R1",
    "alpha_std",
    "alpha_target_pearson",
    "rgb_missing_alpha_rgb",
    "ir_missing_alpha_ir",
    "rgb_noise_alpha_drop",
    "ir_noise_alpha_drop",
    "rgb_blur_alpha_drop",
    "ir_blur_alpha_drop",
    "weighted_rel_base_ratio",
)
AGGREGATE_FIELDS = (
    "metric",
    "mean",
    "std",
    "min",
    "max",
    "n",
)
ROBUSTNESS_AGGREGATE_FIELDS = (
    "scenario",
    "metric",
    "mean",
    "std",
    "min",
    "max",
    "n",
)
CLASSIFICATION_COUNT_FIELDS = ("classification", "seed_count")
PAPER_MAIN_FIELDS = (
    "scope",
    "seed",
    "R1",
    "R5",
    "R10",
    "mAP",
    "mINP",
)
PAPER_ROBUSTNESS_FIELDS = (
    "seed",
    "scenario",
    "R1",
    "R5",
    "R10",
    "mAP",
    "mINP",
)
REFERENCE_COMPARISON_FIELDS = (
    "reference",
    "reference_source",
    "seed1_clean_R1",
    "seed1_degraded_mean_R1",
    "reference_clean_R1",
    "reference_degraded_mean_R1",
    "delta_clean_R1",
    "delta_degraded_mean_R1",
)


def _validate_schema(fields: Sequence[str]) -> Tuple[str, ...]:
    schema = tuple(str(field) for field in fields)
    if not schema or any(not field for field in schema):
        raise ValueError("TSV schema contains an empty field")
    if len(schema) != len(set(schema)):
        raise ValueError(f"TSV schema contains duplicate fields: {schema}")
    return schema


def _atomic_text(
    path: Path,
    writer: Callable[[Any], None],
    validator: Optional[Callable[[Path], None]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
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
            temporary = handle.name
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if validator is not None:
            validator(Path(temporary))
        os.replace(temporary, path)
    except BaseException:
        if temporary:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
        raise


def write_tsv_atomic(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write one explicit fixed-schema TSV and validate its shape before replace."""
    schema = _validate_schema(fields)
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
            writer.writerow(row)

    def validate(temporary: Path) -> None:
        with temporary.open("r", encoding="utf-8", newline="") as handle:
            parsed = list(csv.reader(handle, delimiter="\t"))
        if not parsed or tuple(parsed[0]) != schema:
            raise ValueError(f"Atomic TSV header validation failed: {path}")
        if any(len(row) != len(schema) for row in parsed[1:]):
            raise ValueError(f"Atomic TSV row-width validation failed: {path}")

    _atomic_text(path, write, validate)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    def write(handle: Any) -> None:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")

    def validate(temporary: Path) -> None:
        parsed = json.loads(temporary.read_text(encoding="utf-8"))
        if not isinstance(parsed, Mapping):
            raise ValueError(f"Atomic JSON validation failed: {path}")

    _atomic_text(path, write, validate)


def write_text_atomic(path: Path, text: str) -> None:
    _atomic_text(path, lambda handle: handle.write(text))


@dataclass(frozen=True)
class CompatibleTSV:
    rows: List[Dict[str, str]]
    fields: Tuple[str, ...]
    original_fields: Tuple[str, ...]
    legacy_header_repaired: bool
    source: str


def read_tsv_compatible(path: Path) -> CompatibleTSV:
    """Read fixed TSVs and repair only the four explicitly known merged headers."""
    if not path.is_file():
        return CompatibleTSV([], (), (), False, str(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.reader(handle, delimiter="\t"))
    if not raw:
        return CompatibleTSV([], (), (), False, str(path.resolve()))
    original = _validate_schema(tuple(field.strip() for field in raw[0]))
    widths = {len(row) for row in raw[1:]}
    if len(widths) > 1:
        raise ValueError(
            f"Inconsistent TSV row widths: path={path}, widths={sorted(widths)}"
        )
    data_width = next(iter(widths), len(original))
    repaired: List[str] = []
    changed = False
    for field in original:
        replacement = LEGACY_MERGED_FIELDS.get(field)
        if replacement is None:
            repaired.append(field)
        else:
            repaired.extend(replacement)
            changed = True
    fields = _validate_schema(repaired)
    if len(fields) != data_width:
        raise ValueError(
            "Unsupported TSV schema mismatch; only whitelisted merged headers "
            f"may be repaired: path={path}, original={list(original)}, "
            f"repaired={list(fields)}, data_width={data_width}"
        )
    return CompatibleTSV(
        rows=[dict(zip(fields, row)) for row in raw[1:]],
        fields=fields,
        original_fields=original,
        legacy_header_repaired=changed,
        source=str(path.resolve()),
    )


def _read_key_values(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    result: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourcePlan:
    valid: bool
    source_run_root: str
    confirmation_status: str
    best_found: bool
    last_found: bool
    reasons: Tuple[str, ...]
    best_checkpoint: str = ""
    last_checkpoint: str = ""
    config_path: str = ""
    legacy_header_repaired: bool = False

    def machine_lines(self) -> List[str]:
        lines = [
            f"PLAN_VALID={str(self.valid).lower()}",
            f"SOURCE_CONFIRMATION_STATUS={self.confirmation_status or 'UNKNOWN'}",
            f"SOURCE_BEST_FOUND={str(self.best_found).lower()}",
            f"SOURCE_LAST_FOUND={str(self.last_found).lower()}",
            f"SOURCE_RUN_ROOT={self.source_run_root}",
            f"SOURCE_LEGACY_HEADER_REPAIRED={str(self.legacy_header_repaired).lower()}",
        ]
        lines.extend(f"PLAN_ERROR={reason}" for reason in self.reasons)
        return lines


def validate_source_run(source_run_root: Path) -> SourcePlan:
    """Validate the exact V3 confirmation source without writing to it."""
    reasons: List[str] = []
    try:
        status = _read_key_values(source_run_root / "status.txt")
    except (OSError, UnicodeError) as error:
        status = {}
        reasons.append(str(error))
    if status.get("STATUS") != SOURCE_STATUS:
        reasons.append(
            f"source STATUS must be {SOURCE_STATUS}, got "
            f"{status.get('STATUS', 'missing')}"
        )
    if status.get("FAILED_STAGE") != SOURCE_FAILED_STAGE:
        reasons.append(
            f"source FAILED_STAGE must be {SOURCE_FAILED_STAGE}, got "
            f"{status.get('FAILED_STAGE', 'missing')}"
        )
    if status.get("EXIT_CODE") != SOURCE_EXIT_CODE:
        reasons.append(
            f"source EXIT_CODE must be {SOURCE_EXIT_CODE}, got "
            f"{status.get('EXIT_CODE', 'missing')}"
        )
    try:
        manifest = _read_key_values(source_run_root / "experiment_manifest.txt")
    except (OSError, UnicodeError) as error:
        manifest = {}
        reasons.append(str(error))
    if manifest.get("git_head") != V3_TRAINING_COMMIT:
        reasons.append(
            f"source git_head must be {V3_TRAINING_COMMIT}, got "
            f"{manifest.get('git_head', 'missing')}"
        )
    selection_path = source_run_root / "selected_amplitude_candidate.json"
    selection: Mapping[str, Any] = {}
    try:
        selection = _read_json(selection_path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        reasons.append(f"source selection cannot be read: {error}")
    expected_selection: Mapping[str, Any] = {
        "status": "STRONG_SELECTED",
        "selection_source": "read_only_calibration",
        "selected_candidate": "B50",
        "selected_checkpoint": "best",
        "mraf_scorer_lr_multiplier": 50.0,
        "mraf_gate_logit_scale": 2.0,
        "mraf_rel_lambda": 1.0,
        "mraf_gate_base_grad_scale": 1.0,
    }
    for field, expected in expected_selection.items():
        actual = selection.get(field)
        if isinstance(expected, str):
            matches = str(actual).strip() == expected
        else:
            try:
                matches = math.isclose(
                    float(actual), expected, rel_tol=1e-9, abs_tol=1e-12
                )
            except (TypeError, ValueError):
                matches = False
        if not matches:
            reasons.append(
                f"source selection {field}={actual!r}, expected {expected!r}"
            )

    paths: Dict[str, Path] = {}
    for checkpoint in CHECKPOINTS:
        unit = source_run_root / "confirmation" / "units" / checkpoint
        for name in (
            "checkpoint.pth",
            "configs.yaml",
            "train.log",
            "mraf_epoch_metrics.tsv",
            "checkpoint_selection.tsv",
        ):
            path = unit / name
            if not path.is_file() or path.stat().st_size <= 0:
                reasons.append(f"source {checkpoint} lacks non-empty {path}")
            paths[f"{checkpoint}:{name}"] = path
        config_path = unit / "configs.yaml"
        if config_path.is_file():
            try:
                config = v3.load_config_mapping(str(config_path))
            except (OSError, TypeError, ValueError) as error:
                config = {}
                reasons.append(
                    f"source {checkpoint} configs.yaml cannot be loaded: {error}"
                )
            expected_config: Mapping[str, Any] = {
                "mraf_gate_fp32": True,
                "mraf_scorer_lr_multiplier": 50.0,
                "mraf_gate_logit_scale": 2.0,
                "mraf_rel_lambda": 1.0,
                "mraf_gate_base_grad_scale": 1.0,
                "num_epoch": 12,
                "warmup_epochs": 5,
                "mraf_rel_warmup_epoch": 1,
            }
            for field, expected in expected_config.items():
                actual = config.get(field)
                if isinstance(expected, bool):
                    matches = v3._is_true(actual) == expected
                else:
                    matches = v3._numbers_equal(actual, expected)
                if not matches:
                    reasons.append(
                        f"source {checkpoint} config {field}={actual!r}, "
                        f"expected {expected!r}"
                    )
    best_found = (
        paths.get("best:checkpoint.pth", Path()).is_file()
        if "best:checkpoint.pth" in paths
        else False
    )
    last_found = (
        paths.get("last:checkpoint.pth", Path()).is_file()
        if "last:checkpoint.pth" in paths
        else False
    )
    repaired = False
    legacy_summary = source_run_root / "confirmation_summary.tsv"
    if legacy_summary.is_file():
        try:
            repaired = read_tsv_compatible(legacy_summary).legacy_header_repaired
        except (OSError, UnicodeError, csv.Error, ValueError) as error:
            reasons.append(f"legacy confirmation_summary.tsv rejected: {error}")
    return SourcePlan(
        valid=not reasons,
        source_run_root=str(source_run_root.resolve()),
        confirmation_status=status.get("STATUS", ""),
        best_found=best_found,
        last_found=last_found,
        reasons=tuple(reasons),
        best_checkpoint=str(paths.get("best:checkpoint.pth", "")),
        last_checkpoint=str(paths.get("last:checkpoint.pth", "")),
        config_path=str(paths.get("best:configs.yaml", "")),
        legacy_header_repaired=repaired,
    )


def _as_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _checkpoint_preference(value: Any) -> int:
    return 0 if str(value) == "best" else 1


def rank_checkpoint_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        classification = str(row.get("classification", "FAIL"))
        level = {"STRONG_PASS": 0, "VIABLE_PASS": 1}.get(classification, 2)
        degraded = _as_float(row.get("degraded_mean_R1"))
        clean = _as_float(row.get("clean_R1"))
        return (
            level,
            -(degraded if degraded is not None else -1e30),
            -(clean if clean is not None else -1e30),
            _checkpoint_preference(row.get("checkpoint")),
        )

    return [
        {"rank": index, **dict(row)}
        for index, row in enumerate(sorted(rows, key=key), start=1)
    ]


def select_strong_checkpoint(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    ranked = rank_checkpoint_rows(rows)
    return next(
        (dict(row) for row in ranked if row.get("classification") == "STRONG_PASS"),
        None,
    )


def rank_scale_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank with the immutable V4 selection rule."""
    def key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        classification = str(row.get("classification", "FAIL"))
        level = {"STRONG_PASS": 0, "VIABLE_PASS": 1}.get(classification, 2)
        scale = _as_float(row.get("scale"))
        degraded = _as_float(row.get("degraded_mean_R1"))
        clean = _as_float(row.get("clean_R1"))
        return (
            level,
            scale if scale is not None else math.inf,
            -(degraded if degraded is not None else -1e30),
            -(clean if clean is not None else -1e30),
            _checkpoint_preference(row.get("checkpoint")),
        )

    return [
        {"rank": index, **dict(row)}
        for index, row in enumerate(sorted(rows, key=key), start=1)
    ]


def select_readonly_scale(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Select only STRONG: minimum scale, then degraded, clean, then best."""
    strong = [
        row for row in rows if str(row.get("classification")) == "STRONG_PASS"
    ]
    if not strong:
        return None
    return dict(rank_scale_rows(strong)[0])


def _collect_units(
    unit_specs: Sequence[Tuple[str, str, Path]],
    scenarios: Sequence[str],
    reference_root: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    reference = v3.load_reference_s4(reference_root)
    units = [
        v3.collect_unit(path, candidate, checkpoint, scenarios=scenarios)
        for candidate, checkpoint, path in unit_specs
    ]
    rows = [
        v3.classify_candidate(unit, reference)
        for unit in units
    ]
    metrics = [
        dict(row)
        for unit in units
        for row in unit.get("metric_rows", [])
    ]
    mechanisms = [
        dict(unit["mechanism"]) for unit in units if "mechanism" in unit
    ]
    return rows, metrics, mechanisms


def _selection_payload(
    selected: Optional[Mapping[str, Any]],
    source: str,
    failed_status: str,
) -> Dict[str, Any]:
    if selected is None:
        return {
            "status": failed_status,
            "selection_source": source,
            "selected_scale": None,
            "selected_checkpoint": None,
            "classification": "FAIL",
            "clean_R1": None,
            "degraded_mean_R1": None,
            "mraf_scorer_lr_multiplier": 50.0,
            "mraf_rel_lambda": 1.0,
            "mraf_gate_base_grad_scale": 1.0,
        }
    return {
        "status": "STRONG_SELECTED",
        "selection_source": source,
        "selected_scale": _as_float(
            selected.get("scale", selected.get("mraf_gate_logit_scale", 2.0))
        ),
        "selected_checkpoint": selected.get("checkpoint"),
        "classification": selected.get("classification"),
        "clean_R1": _as_float(selected.get("clean_R1")),
        "degraded_mean_R1": _as_float(selected.get("degraded_mean_R1")),
        "mraf_scorer_lr_multiplier": 50.0,
        "mraf_rel_lambda": 1.0,
        "mraf_gate_base_grad_scale": 1.0,
    }


def summarize_full_rediagnosis(
    candidate_root: Path,
    reference_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    specs = [
        ("source_confirmation", checkpoint, candidate_root / checkpoint)
        for checkpoint in CHECKPOINTS
    ]
    rows, metrics, mechanisms = _collect_units(
        specs, SCREEN_SCENARIOS, reference_root
    )
    ranked = rank_checkpoint_rows(rows)
    selected = select_strong_checkpoint(rows)
    write_tsv_atomic(
        output_dir / "full_rediagnosis_metrics.tsv", METRIC_FIELDS, metrics
    )
    write_tsv_atomic(
        output_dir / "full_rediagnosis_mechanism.tsv",
        MECHANISM_FIELDS,
        mechanisms,
    )
    write_tsv_atomic(
        output_dir / "full_rediagnosis_ranking.tsv", RANKING_FIELDS, ranked
    )
    if selected is not None:
        payload = _selection_payload(
            {**selected, "scale": 2.0},
            "full_rediagnosis",
            "IR_SENSITIVITY_RESCUE_FAILED",
        )
        write_json_atomic(output_dir / "selected_ir_rescue_scale.json", payload)
    return {
        "status": (
            "STRONG_SELECTED" if selected is not None else "READONLY_SCALE_REQUIRED"
        ),
        "selected": selected,
    }


def summarize_readonly_scale(
    candidate_root: Path,
    reference_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    specs = [
        (f"scale_{scale:g}", checkpoint, candidate_root / f"scale_{scale:g}" / checkpoint)
        for scale in READONLY_SCALES
        for checkpoint in CHECKPOINTS
    ]
    rows, metrics, mechanisms = _collect_units(
        specs, SCREEN_SCENARIOS, reference_root
    )
    scale_rows: List[Dict[str, Any]] = []
    for row in rows:
        candidate = str(row.get("candidate", ""))
        scale = _as_float(candidate.removeprefix("scale_"))
        scale_rows.append({"scale": scale, **dict(row)})
    ranked = rank_scale_rows(scale_rows)
    selected = select_readonly_scale(scale_rows)
    write_tsv_atomic(
        output_dir / "readonly_scale_metrics.tsv", METRIC_FIELDS, metrics
    )
    write_tsv_atomic(
        output_dir / "readonly_scale_mechanism.tsv",
        MECHANISM_FIELDS,
        mechanisms,
    )
    write_tsv_atomic(
        output_dir / "readonly_scale_ranking.tsv", SCALE_RANKING_FIELDS, ranked
    )
    payload = _selection_payload(
        selected,
        "readonly_scale_screen",
        "IR_SENSITIVITY_RESCUE_FAILED",
    )
    write_json_atomic(output_dir / "selected_ir_rescue_scale.json", payload)
    return {"status": payload["status"], "selected": selected}


def summarize_confirmation(
    candidate_root: Path,
    reference_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    specs = [
        ("ir_rescue_confirmation", checkpoint, candidate_root / checkpoint)
        for checkpoint in CHECKPOINTS
    ]
    rows, metrics, mechanisms = _collect_units(
        specs, SCREEN_SCENARIOS, reference_root
    )
    ranked = rank_checkpoint_rows(rows)
    selected = select_strong_checkpoint(rows)
    write_tsv_atomic(
        output_dir / "ir_rescue_confirmation_metrics.tsv", METRIC_FIELDS, metrics
    )
    write_tsv_atomic(
        output_dir / "ir_rescue_confirmation_mechanism.tsv",
        MECHANISM_FIELDS,
        mechanisms,
    )
    write_tsv_atomic(
        output_dir / "ir_rescue_confirmation_summary.tsv",
        RANKING_FIELDS,
        ranked,
    )
    return {
        "status": (
            "CONFIRMATION_PASSED"
            if selected is not None
            else "IR_SENSITIVITY_CONFIRMATION_FAILED"
        ),
        "selected": selected,
    }


def summarize_formal_seed(
    candidate_root: Path,
    reference_root: Path,
    output_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    specs = [
        (f"formal_seed{seed}", checkpoint, candidate_root / checkpoint)
        for checkpoint in CHECKPOINTS
    ]
    rows, metrics, mechanisms = _collect_units(
        specs, FORMAL_SCENARIOS, reference_root
    )
    ranked = rank_checkpoint_rows(rows)
    selected = ranked[0] if ranked else None
    strong = select_strong_checkpoint(rows)
    write_tsv_atomic(
        output_dir / f"formal_seed{seed}_metrics.tsv", METRIC_FIELDS, metrics
    )
    write_tsv_atomic(
        output_dir / f"formal_seed{seed}_mechanism.tsv",
        MECHANISM_FIELDS,
        mechanisms,
    )
    write_tsv_atomic(
        output_dir / f"formal_seed{seed}_summary.tsv", RANKING_FIELDS, ranked
    )
    payload = _selection_payload(
        selected,
        f"formal_seed{seed}",
        "FORMAL_SEED1_FAILED" if seed == 1 else "FAILED",
    )
    payload["status"] = (
        "STRONG_SELECTED"
        if strong is not None and selected is not None
        else "FORMAL_SEED1_FAILED"
        if seed == 1
        else "FORMAL_SEED_COMPLETED"
    )
    payload["seed"] = seed
    write_json_atomic(
        output_dir / f"selected_formal_seed{seed}_checkpoint.json", payload
    )
    if seed == 1:
        write_json_atomic(
            output_dir / "selected_formal_seed1_checkpoint.json", payload
        )
    return {
        "status": payload["status"],
        "selected": selected,
        "strong_selected": strong,
    }


def _selected_seed_row(run_root: Path, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    selection = _read_json(
        run_root / f"selected_formal_seed{seed}_checkpoint.json"
    )
    checkpoint = str(selection.get("selected_checkpoint", ""))
    summary_rows = read_tsv_compatible(
        run_root / f"formal_seed{seed}_summary.tsv"
    ).rows
    selected = next(
        (dict(row) for row in summary_rows if row.get("checkpoint") == checkpoint),
        {},
    )
    metrics = read_tsv_compatible(
        run_root / f"formal_seed{seed}_metrics.tsv"
    ).rows
    return selected, metrics


def _aggregate(values: Sequence[float]) -> Dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _clean_metric(metrics: Sequence[Mapping[str, Any]], name: str) -> Optional[float]:
    row = next((row for row in metrics if row.get("scenario") == "clean"), {})
    return _as_float(row.get(name))


def summarize_multiseed(
    run_root: Path,
    output_dir: Path,
    reference_screen_root: Optional[Path] = None,
    reference_run_root: Optional[Path] = None,
) -> Dict[str, Any]:
    seed_rows: List[Dict[str, Any]] = []
    robustness_rows: List[Dict[str, Any]] = []
    selected_summaries: List[Dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        selected, metrics = _selected_seed_row(run_root, seed)
        if not selected or len(metrics) != len(FORMAL_SCENARIOS):
            raise ValueError(f"seed {seed} formal artifacts are incomplete")
        selected_summaries.append(selected)
        seed_rows.append(
            {
                "seed": seed,
                "selected_checkpoint": selected.get("checkpoint"),
                "classification": selected.get("classification"),
                "clean_R1": _clean_metric(metrics, "R1"),
                "clean_R5": _clean_metric(metrics, "R5"),
                "clean_R10": _clean_metric(metrics, "R10"),
                "clean_mAP": _clean_metric(metrics, "mAP"),
                "clean_mINP": _clean_metric(metrics, "mINP"),
                "degraded_mean_R1": _as_float(selected.get("degraded_mean_R1")),
                **{
                    field: _as_float(selected.get(field))
                    for field in (
                        "alpha_std",
                        "alpha_target_pearson",
                        "rgb_missing_alpha_rgb",
                        "ir_missing_alpha_ir",
                        "rgb_noise_alpha_drop",
                        "ir_noise_alpha_drop",
                        "rgb_blur_alpha_drop",
                        "ir_blur_alpha_drop",
                        "weighted_rel_base_ratio",
                    )
                },
            }
        )
        robustness_rows.extend({"seed": seed, **dict(row)} for row in metrics)

    aggregate_rows = []
    for field in SEED_LEVEL_FIELDS[3:]:
        values = [
            value
            for row in seed_rows
            if (value := _as_float(row.get(field))) is not None
        ]
        if values:
            aggregate_rows.append({"metric": field, **_aggregate(values)})
    counts = {
        name: sum(row.get("classification") == name for row in seed_rows)
        for name in ("STRONG_PASS", "VIABLE_PASS", "FAIL")
    }
    write_tsv_atomic(
        output_dir / "multiseed_retrieval_summary.tsv",
        AGGREGATE_FIELDS,
        [
            row
            for row in aggregate_rows
            if str(row["metric"]).startswith("clean_")
        ],
    )
    write_tsv_atomic(
        output_dir / "multiseed_robustness_summary.tsv",
        ROBUSTNESS_AGGREGATE_FIELDS,
        [
            {
                "scenario": scenario,
                "metric": metric,
                **_aggregate(
                    [
                        value
                        for row in robustness_rows
                        if row.get("scenario") == scenario
                        and (value := _as_float(row.get(metric))) is not None
                    ]
                ),
            }
            for scenario in FORMAL_SCENARIOS
            for metric in ("R1", "R5", "R10", "mAP", "mINP")
        ]
        + [
            {
                "scenario": "key_degraded_mean",
                "metric": "R1",
                **_aggregate(
                    [
                        float(row["degraded_mean_R1"])
                        for row in seed_rows
                        if row.get("degraded_mean_R1") is not None
                    ]
                ),
            }
        ],
    )
    write_tsv_atomic(
        output_dir / "multiseed_mechanism_summary.tsv",
        AGGREGATE_FIELDS,
        [
            row
            for row in aggregate_rows
            if row["metric"]
            not in {
                "clean_R1",
                "clean_R5",
                "clean_R10",
                "clean_mAP",
                "clean_mINP",
                "degraded_mean_R1",
            }
        ],
    )
    write_tsv_atomic(
        output_dir / "multiseed_classification_summary.tsv",
        CLASSIFICATION_COUNT_FIELDS,
        [
            {"classification": name, "seed_count": count}
            for name, count in counts.items()
        ],
    )
    paper = output_dir / "paper_tables"
    main_rows = [
        {
            "scope": "seed",
            "seed": row["seed"],
            "R1": row["clean_R1"],
            "R5": row["clean_R5"],
            "R10": row["clean_R10"],
            "mAP": row["clean_mAP"],
            "mINP": row["clean_mINP"],
        }
        for row in seed_rows
    ]
    main_rows.append(
        {
            "scope": "five_seed_mean",
            "seed": "1-5",
            **{
                metric: statistics.mean(
                    float(row[f"clean_{metric}"]) for row in seed_rows
                )
                for metric in ("R1", "R5", "R10", "mAP", "mINP")
            },
        }
    )
    main_rows.append(
        {
            "scope": "five_seed_std",
            "seed": "1-5",
            **{
                metric: statistics.stdev(
                    float(row[f"clean_{metric}"]) for row in seed_rows
                )
                for metric in ("R1", "R5", "R10", "mAP", "mINP")
            },
        }
    )
    write_tsv_atomic(paper / "main_retrieval_results.tsv", PAPER_MAIN_FIELDS, main_rows)
    write_tsv_atomic(
        paper / "robustness_results.tsv",
        PAPER_ROBUSTNESS_FIELDS,
        [
            {field: row.get(field, "") for field in PAPER_ROBUSTNESS_FIELDS}
            for row in robustness_rows
        ],
    )
    write_tsv_atomic(
        paper / "mechanism_results.tsv", SEED_LEVEL_FIELDS, seed_rows
    )
    scale_rows = read_tsv_compatible(
        run_root / "readonly_scale_ranking.tsv"
    ).rows
    if not scale_rows:
        scale_rows = [
            {"scale": 2.0, **row}
            for row in read_tsv_compatible(
                run_root / "full_rediagnosis_ranking.tsv"
            ).rows
        ]
    write_tsv_atomic(
        paper / "ablation_scale_results.tsv", SCALE_RANKING_FIELDS, scale_rows
    )
    write_tsv_atomic(
        paper / "seed_level_results.tsv", SEED_LEVEL_FIELDS, seed_rows
    )
    seed1 = seed_rows[0]
    comparisons: List[Dict[str, Any]] = []
    references: List[Tuple[str, Mapping[str, Any]]] = []
    if reference_screen_root is not None and reference_screen_root.is_dir():
        reference = v3.load_reference_s4(reference_screen_root)
        references.append(("S4", reference))
    if reference_run_root is not None and reference_run_root.is_dir():
        reference = v3._discover_old_reference(reference_run_root)
        if reference:
            references.append(("old_seed1", reference))
    for name, reference in references:
        reference_clean = _as_float(reference.get("clean_R1"))
        reference_degraded = _as_float(
            reference.get(
                "degraded_mean_R1",
                reference.get("key_degraded_mean_R1"),
            )
        )
        seed_clean = _as_float(seed1.get("clean_R1"))
        seed_degraded = _as_float(seed1.get("degraded_mean_R1"))
        comparisons.append(
            {
                "reference": name,
                "reference_source": reference.get("source", ""),
                "seed1_clean_R1": seed_clean,
                "seed1_degraded_mean_R1": seed_degraded,
                "reference_clean_R1": reference_clean,
                "reference_degraded_mean_R1": reference_degraded,
                "delta_clean_R1": (
                    seed_clean - reference_clean
                    if seed_clean is not None and reference_clean is not None
                    else None
                ),
                "delta_degraded_mean_R1": (
                    seed_degraded - reference_degraded
                    if seed_degraded is not None
                    and reference_degraded is not None
                    else None
                ),
            }
        )
    write_tsv_atomic(
        output_dir / "comparison_with_s4_and_old_seed1.tsv",
        REFERENCE_COMPARISON_FIELDS,
        comparisons,
    )
    _write_paper_ready_reports(output_dir, seed_rows, counts)
    return {"status": "COMPLETED", "seed_rows": seed_rows, "counts": counts}


def _write_paper_ready_reports(
    output_dir: Path,
    seed_rows: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
) -> None:
    selection_path = output_dir / "selected_ir_rescue_scale.json"
    selection = _read_json(selection_path) if selection_path.is_file() else {}
    selected_scale = _as_float(selection.get("selected_scale"))
    selection_source = str(selection.get("selection_source", "unavailable"))
    seed1 = seed_rows[0]

    def values(field: str) -> List[float]:
        return [
            value
            for row in seed_rows
            if (value := _as_float(row.get(field))) is not None
        ]

    clean = values("clean_R1")
    degraded = values("degraded_mean_R1")
    alpha = values("alpha_std")
    pearson = values("alpha_target_pearson")

    def mean_std(series: Sequence[float]) -> str:
        if not series:
            return "unavailable"
        std = statistics.stdev(series) if len(series) > 1 else 0.0
        return f"{statistics.mean(series):.6f} ± {std:.6f}"

    confirmation_note = (
        "scale=2在24-batch复诊中已达到STRONG，因此按预注册规则跳过新的"
        "12 epoch confirmation。"
        if selected_scale == 2.0
        else (
            f"只读搜索选出scale={selected_scale:g}后，从pretrained初始化重新运行"
            "12 epoch confirmation；其best/last原始结果见"
            "`ir_rescue_confirmation_summary.tsv`。"
            if selected_scale is not None
            else "未获得可报告的STRONG scale。"
        )
    )
    confirmation_note_en = (
        "Scale 2 passed the 24-batch STRONG gate, so the newly trained 12-epoch "
        "confirmation was skipped by the preregistered rule."
        if selected_scale == 2.0
        else (
            f"After read-only selection of scale {selected_scale:g}, a fresh "
            "12-epoch confirmation was trained from the pretrained initialization; "
            "raw best/last results are retained in "
            "`ir_rescue_confirmation_summary.tsv`."
            if selected_scale is not None
            else "No STRONG scale was available for reporting."
        )
    )
    zh_content = {
        "最终方法配置": (
            f"最终scale={selected_scale if selected_scale is not None else 'unavailable'}，"
            "scorer学习率倍率为50，reliability lambda为1，base-gradient scale为1，"
            "门控使用FP32。正式实验为60 epoch、warmup 3、degrade seed 2026，"
            "每个seed均从相同pretrained初始化。"
        ),
        "Stage 1筛选结论": (
            "V3固定选择B50/best（scorer LR倍率50、logit scale 2）；该选择只作为"
            "V4只读源，不作为正式模型的续训起点。"
        ),
        "原scale=2 confirmation失败原因": (
            "V3的12 epoch confirmation在best和last上均为VIABLE_PASS，"
            "原STRONG门禁失败项为IR_NOISE和IR_BLUR；门槛没有在V4中降低。"
        ),
        "24-batch复诊结论": (
            f"完整复诊使用24个batch；最终选择来源为`{selection_source}`。"
            "完整best/last指标见`full_rediagnosis_*`文件。"
        ),
        "scale搜索结果": (
            f"自动选择scale={selected_scale if selected_scale is not None else 'NONE'}。"
            "候选固定为2.5/3/4/5，只允许STRONG，排序为最小scale、degraded mean、"
            "clean R1、best。"
        ),
        "新12 epoch确认结果": confirmation_note,
        "正式seed1结果": (
            f"seed1选择{seed1.get('selected_checkpoint')}，分类为"
            f"{seed1.get('classification')}，clean R1={seed1.get('clean_R1')}，"
            f"degraded mean R1={seed1.get('degraded_mean_R1')}。"
        ),
        "五种子均值与标准差": (
            f"clean R1为{mean_std(clean)}，degraded mean R1为"
            f"{mean_std(degraded)}；STRONG/VIABLE/FAIL seed数量分别为"
            f"{counts.get('STRONG_PASS', 0)}/{counts.get('VIABLE_PASS', 0)}/"
            f"{counts.get('FAIL', 0)}。"
        ),
        "11场景结果表": (
            "五个seed的全部11场景原始结果见"
            "`paper_tables/robustness_results.tsv`，逐场景mean/std/min/max见"
            "`multiseed_robustness_summary.tsv`。"
        ),
        "机制指标表": (
            f"alpha_std为{mean_std(alpha)}，alpha-target Pearson为"
            f"{mean_std(pearson)}；其余缺失模态权重、四类方向下降和"
            "weighted-rel/base见`paper_tables/mechanism_results.tsv`。"
        ),
        "相对S4和旧seed1的变化": (
            "所有差值均由原始指标直接计算并保存在"
            "`comparison_with_s4_and_old_seed1.tsv`；缺少参考时保持为空，"
            "不进行推测。"
        ),
        "可写入论文的客观结论": (
            f"在固定阈值和五个正式seed下，观察到"
            f"{counts.get('STRONG_PASS', 0)}个STRONG、"
            f"{counts.get('VIABLE_PASS', 0)}个VIABLE和"
            f"{counts.get('FAIL', 0)}个FAIL；这些计数与归档均值可作为客观结果。"
        ),
        "不能写入论文的过度结论": (
            "不得将只读scale搜索、单seed结果或机制相关性表述为普适因果结论，"
            "也不得在没有外部方法统一复现时宣称SOTA。"
        ),
        "局限性": (
            "scale只在预注册的有限集合中搜索，退化集合是合成协议，五种子统计"
            "不能替代跨数据集或真实传感器故障验证。"
        ),
        "单seed与五seed结果边界": (
            "Stage 1、复诊、scale选择和confirmation均为seed1门禁；论文主结果"
            "只引用正式seed1–5完成后的均值、标准差和原始seed级表。"
        ),
        "论文段落：实验设置、消融实验、鲁棒性分析、机制分析": (
            f"实验设置：我们在固定训练协议下使用五个随机种子，正式训练60 epoch，"
            f"最终门控scale为{selected_scale if selected_scale is not None else '未选择'}。"
            "消融实验：scale仅通过冻结checkpoint的只读前向筛选，未改变模型权重。"
            f"鲁棒性分析：五种子clean R1为{mean_std(clean)}，degraded mean R1为"
            f"{mean_std(degraded)}。机制分析：alpha标准差为{mean_std(alpha)}，"
            f"alpha-target Pearson为{mean_std(pearson)}，全部原始seed和场景结果"
            "均已保留。"
        ),
    }
    en_content = {
        "Final Method Configuration": (
            f"The final gate scale is {selected_scale}; the scorer LR multiplier, "
            "reliability lambda, and base-gradient scale are 50, 1, and 1. "
            "Formal runs use 60 epochs, a three-epoch warmup, degradation seed "
            "2026, FP32 gating, and the same pretrained initialization."
        ),
        "Stage-1 Screening Conclusion": (
            "V3 selected B50/best at scorer-LR multiplier 50 and logit scale 2. "
            "It is used only as a read-only diagnostic source, never as a formal "
            "training initialization."
        ),
        "Why the Original Scale-2 Confirmation Failed": (
            "Both V3 confirmation checkpoints were VIABLE_PASS and missed the "
            "unchanged STRONG gate on IR_NOISE and IR_BLUR."
        ),
        "24-Batch Rediagnosis": (
            f"The full rediagnosis used 24 batches and selected from "
            f"`{selection_source}`; all best/last evidence is retained."
        ),
        "Read-Only Scale Search": (
            f"The selected scale is {selected_scale}. Only STRONG candidates were "
            "eligible under the fixed scale, degraded-R1, clean-R1, and best tie-break."
        ),
        "New 12-Epoch Confirmation": confirmation_note_en,
        "Formal Seed-1 Result": (
            f"Seed 1 selected {seed1.get('selected_checkpoint')} with "
            f"{seed1.get('classification')}, clean R1={seed1.get('clean_R1')}, "
            f"and degraded mean R1={seed1.get('degraded_mean_R1')}."
        ),
        "Five-Seed Mean and Standard Deviation": (
            f"Clean R1 is {mean_std(clean)} and degraded mean R1 is "
            f"{mean_std(degraded)}. STRONG/VIABLE/FAIL counts are "
            f"{counts.get('STRONG_PASS', 0)}/{counts.get('VIABLE_PASS', 0)}/"
            f"{counts.get('FAIL', 0)}."
        ),
        "Eleven-Scenario Results": (
            "All seed-level scenario results and per-scenario aggregates are stored "
            "in the robustness paper table and multiseed robustness summary."
        ),
        "Mechanism Metrics": (
            f"Alpha standard deviation is {mean_std(alpha)} and alpha-target "
            f"Pearson is {mean_std(pearson)}; all remaining mechanism values are "
            "reported at seed level."
        ),
        "Changes Relative to S4 and the Previous Seed-1 Run": (
            "Direct metric deltas are stored in "
            "`comparison_with_s4_and_old_seed1.tsv`; unavailable references remain blank."
        ),
        "Objective Claims Suitable for the Paper": (
            f"Under fixed gates, {counts.get('STRONG_PASS', 0)} of five seeds are "
            "STRONG; archived means, standard deviations, and counts are objective evidence."
        ),
        "Claims That Must Not Be Made": (
            "Read-only scale screening, a single seed, or correlation alone does not "
            "establish a universal causal mechanism or state-of-the-art performance."
        ),
        "Limitations": (
            "The search uses a finite preregistered scale set and synthetic degradations; "
            "it does not replace cross-dataset or real sensor-failure validation."
        ),
        "Single-Seed Versus Five-Seed Boundary": (
            "Screening and confirmation are seed-1 gates. Final claims use only the "
            "completed formal five-seed statistics and retain every seed-level result."
        ),
        "Paper-Ready Experimental, Ablation, Robustness, and Mechanism Text": (
            f"Experimental setup: five formal seeds were trained for 60 epochs at "
            f"gate scale {selected_scale}. Ablation: scale was selected by frozen, "
            "read-only checkpoint evaluation. Robustness: clean R1 and degraded mean "
            f"R1 are {mean_std(clean)} and {mean_std(degraded)}. Mechanism analysis: "
            f"alpha variation and alpha-target Pearson are {mean_std(alpha)} and "
            f"{mean_std(pearson)}, with all raw seed/scenario evidence retained."
        ),
    }
    zh = ["# MRAF IR Sensitivity Rescue V4 论文结果", ""]
    en = ["# MRAF IR Sensitivity Rescue V4 Paper-Ready Results", ""]
    for title, content in zh_content.items():
        zh.extend([f"## {title}", "", content, ""])
    for title, content in en_content.items():
        en.extend([f"## {title}", "", content, ""])
    write_text_atomic(output_dir / "PAPER_READY_RESULTS_ZH.md", "\n".join(zh))
    write_text_atomic(output_dir / "PAPER_READY_RESULTS_EN.md", "\n".join(en))
    report = [
        "STATUS=COMPLETED",
        f"formal_seed_count={len(seed_rows)}",
        f"strong_seed_count={counts.get('STRONG_PASS', 0)}",
        f"viable_seed_count={counts.get('VIABLE_PASS', 0)}",
        f"fail_seed_count={counts.get('FAIL', 0)}",
        "single_seed_and_five_seed_results_are_explicitly_separated=true",
        "thresholds_unchanged=true",
    ]
    write_text_atomic(output_dir / "FINAL_REPORT.txt", "\n".join(report) + "\n")


def create_mock_outputs(output_dir: Path) -> Dict[str, Any]:
    """Create complete deterministic V4 schemas without torch, GPU, or training."""
    output_dir.mkdir(parents=True, exist_ok=False)
    empty_metric = {
        "candidate": "mock",
        "checkpoint": "best",
        "scenario": "clean",
        "R1": 52.0,
        "R5": 70.0,
        "R10": 80.0,
        "mAP": 45.0,
        "mINP": 30.0,
        "score_file": "MOCK",
    }
    mechanism = {
        field: (
            "mock"
            if field == "candidate"
            else "best"
            if field == "checkpoint"
            else True
            if field in {
                "scorer_parameter_updated",
                "fusion_path_sensitive",
                "gate_logits_fp32",
            }
            else False
            if field == "legacy_header_repaired"
            else 0.1
        )
        for field in MECHANISM_FIELDS
    }
    summary = {
        **{field: "" for field in SUMMARY_FIELDS},
        "candidate": "mock",
        "checkpoint": "best",
        "complete": True,
        "clean_R1": 52.0,
        "degraded_mean_R1": 43.5,
        "classification": "STRONG_PASS",
        "strong_pass": True,
        "viable_pass": True,
    }
    for name in (
        "full_rediagnosis_metrics.tsv",
        "readonly_scale_metrics.tsv",
        "ir_rescue_confirmation_metrics.tsv",
    ):
        write_tsv_atomic(output_dir / name, METRIC_FIELDS, [empty_metric])
    for name in (
        "full_rediagnosis_mechanism.tsv",
        "readonly_scale_mechanism.tsv",
        "ir_rescue_confirmation_mechanism.tsv",
    ):
        write_tsv_atomic(output_dir / name, MECHANISM_FIELDS, [mechanism])
    write_tsv_atomic(
        output_dir / "full_rediagnosis_ranking.tsv",
        RANKING_FIELDS,
        [{"rank": 1, **summary}],
    )
    write_tsv_atomic(
        output_dir / "readonly_scale_ranking.tsv",
        SCALE_RANKING_FIELDS,
        [{"rank": 1, "scale": 2.5, **summary}],
    )
    write_tsv_atomic(
        output_dir / "ir_rescue_confirmation_summary.tsv",
        RANKING_FIELDS,
        [{"rank": 1, **summary}],
    )
    selection = _selection_payload(
        {**summary, "scale": 2.5}, "mock", "IR_SENSITIVITY_RESCUE_FAILED"
    )
    write_json_atomic(output_dir / "selected_ir_rescue_scale.json", selection)
    for seed in FORMAL_SEEDS:
        metrics = [
            {
                **empty_metric,
                "candidate": f"formal_seed{seed}",
                "scenario": scenario,
            }
            for scenario in FORMAL_SCENARIOS
        ]
        write_tsv_atomic(
            output_dir / f"formal_seed{seed}_metrics.tsv", METRIC_FIELDS, metrics
        )
        write_tsv_atomic(
            output_dir / f"formal_seed{seed}_mechanism.tsv",
            MECHANISM_FIELDS,
            [{**mechanism, "candidate": f"formal_seed{seed}"}],
        )
        write_tsv_atomic(
            output_dir / f"formal_seed{seed}_summary.tsv",
            RANKING_FIELDS,
            [{"rank": 1, **summary, "candidate": f"formal_seed{seed}"}],
        )
        payload = {
            **selection,
            "selection_source": f"formal_seed{seed}",
            "selected_scale": 2.5,
            "seed": seed,
        }
        write_json_atomic(
            output_dir / f"selected_formal_seed{seed}_checkpoint.json", payload
        )
    write_tsv_atomic(
        output_dir / "stage_status.tsv",
        STAGE_STATUS_FIELDS,
        [
            {
                "timestamp": "MOCK",
                "stage": "mock",
                "status": "COMPLETED",
                "exit_code": 0,
                "details": "no training",
            }
        ],
    )
    write_tsv_atomic(
        output_dir / "seed_status.tsv",
        SEED_STATUS_FIELDS,
        [
            {
                "timestamp": "MOCK",
                "seed": seed,
                "status": "COMPLETED",
                "exit_code": 0,
                "selected_checkpoint": "best",
                "classification": "STRONG_PASS",
            }
            for seed in FORMAL_SEEDS
        ],
    )
    write_tsv_atomic(output_dir / "resume_history.tsv", RESUME_HISTORY_FIELDS, [])
    summarize_multiseed(output_dir, output_dir)
    write_text_atomic(
        output_dir / "experiment_manifest.txt",
        f"PIPELINE_ID={PIPELINE_ID}\nmock=true\n",
    )
    write_text_atomic(
        output_dir / "status.txt",
        "STATUS=MOCK_COMPLETED\nFAILED_STAGE=NONE\nEXIT_CODE=0\n",
    )
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize MRAF IR-sensitivity V4 rediagnosis, scale selection, "
            "fresh confirmation, formal seeds, and paper-ready outputs."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "source-plan",
            "full-rediagnosis",
            "readonly-scale",
            "confirmation",
            "formal-seed",
            "multiseed",
            "mock",
        ),
    )
    parser.add_argument("--source-run-root", default="")
    parser.add_argument("--candidate-root", default="")
    parser.add_argument("--reference-screen-root", default="")
    parser.add_argument("--reference-run-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "source-plan":
        plan = validate_source_run(Path(args.source_run_root))
        print("\n".join(plan.machine_lines()))
        return 0 if plan.valid else 2
    if not args.output_dir:
        raise SystemExit("--output-dir is required")
    output = Path(args.output_dir)
    if args.mode == "mock":
        selection = create_mock_outputs(output)
        print(json.dumps({"status": "MOCK_COMPLETED", "selection": selection}))
        return 0
    if not args.reference_screen_root:
        raise SystemExit("--reference-screen-root is required")
    reference = Path(args.reference_screen_root)
    candidate = Path(args.candidate_root)
    if args.mode == "full-rediagnosis":
        result = summarize_full_rediagnosis(candidate, reference, output)
    elif args.mode == "readonly-scale":
        result = summarize_readonly_scale(candidate, reference, output)
    elif args.mode == "confirmation":
        result = summarize_confirmation(candidate, reference, output)
    elif args.mode == "formal-seed":
        if args.seed not in FORMAL_SEEDS:
            raise SystemExit("--seed must be one of 1,2,3,4,5")
        result = summarize_formal_seed(candidate, reference, output, args.seed)
    elif args.mode == "multiseed":
        if not args.run_root:
            raise SystemExit("--run-root is required for multiseed")
        result = summarize_multiseed(
            Path(args.run_root),
            output,
            Path(args.reference_screen_root)
            if args.reference_screen_root
            else None,
            Path(args.reference_run_root)
            if args.reference_run_root
            else None,
        )
    else:  # pragma: no cover
        raise AssertionError(args.mode)
    print(json.dumps(result, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
