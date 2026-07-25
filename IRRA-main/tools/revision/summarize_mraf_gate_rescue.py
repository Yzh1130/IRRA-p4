#!/usr/bin/env python3
"""Summarize, hard-filter, and rank the fixed MRAF gate-rescue screen."""

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

from utils.config import ConfigLoadError, load_config_mapping  # noqa: E402


CANDIDATE_MATRIX: Mapping[str, Mapping[str, Any]] = {
    "S0": {
        "mraf_gate_fp32": False,
        "mraf_rel_lambda": 0.01,
        "mraf_gate_base_grad_scale": 1.0,
    },
    "S1": {
        "mraf_gate_fp32": True,
        "mraf_rel_lambda": 0.01,
        "mraf_gate_base_grad_scale": 1.0,
    },
    "S2": {
        "mraf_gate_fp32": True,
        "mraf_rel_lambda": 0.25,
        "mraf_gate_base_grad_scale": 1.0,
    },
    "S3": {
        "mraf_gate_fp32": True,
        "mraf_rel_lambda": 1.00,
        "mraf_gate_base_grad_scale": 1.0,
    },
    "S4": {
        "mraf_gate_fp32": True,
        "mraf_rel_lambda": 0.25,
        "mraf_gate_base_grad_scale": 0.10,
    },
    "S5": {
        "mraf_gate_fp32": True,
        "mraf_rel_lambda": 0.25,
        "mraf_gate_base_grad_scale": 0.00,
    },
}

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
KEY_DEGRADED_SCENARIOS = SCREEN_SCENARIOS[1:]
FINAL_SCENARIOS = (
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

METRICS_FIELDS = (
    "candidate",
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
    "alpha_rgb_std",
    "alpha_target_pearson",
    "rgb_missing_alpha_rgb_mean",
    "ir_missing_alpha_ir_mean",
    "rgb_noise_alpha_drop",
    "ir_noise_alpha_drop",
    "rgb_blur_alpha_drop",
    "ir_blur_alpha_drop",
    "direction_correct_fraction",
    "weighted_rel_base_median",
    "scorer_rel_only_gradient",
    "scorer_parameter_updated",
    "fusion_path_sensitive",
    "gate_logits_fp32",
)
SUMMARY_FIELDS = (
    "candidate",
    "mraf_gate_fp32",
    "mraf_rel_lambda",
    "mraf_gate_base_grad_scale",
    "complete",
    "has_nan",
    "has_inf",
    "has_oom",
    "clean_R1",
    "clean_mAP",
    "clean_mINP",
    "key_degraded_mean_R1",
    "relative_clean_mean_drop_R1",
    *MECHANISM_FIELDS[1:],
    "hard_filter_pass",
    "failed_criteria",
)
RANKING_FIELDS = (
    "rank",
    "candidate",
    "hard_filter_pass",
    "key_degraded_mean_R1",
    "clean_R1",
    "alpha_target_pearson",
    "missing_modality_alpha_sum",
    "mraf_rel_lambda",
    "mraf_gate_base_grad_scale",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the fixed S0-S5 MRAF gate-rescue candidates, apply the "
            "documented hard filters, and select the best passing candidate."
        )
    )
    parser.add_argument("--candidate-root", default="")
    parser.add_argument("--final-run-dir", default="")
    parser.add_argument(
        "--old-run-root",
        default="",
        help="optional root containing old formal *_<case>_seed1 runs",
    )
    parser.add_argument(
        "--base-config",
        default="",
        help="historical formal configs.yaml recorded in the final report",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _finite(value: Any) -> bool:
    parsed = _as_float(value)
    return parsed is not None and math.isfinite(parsed)


def _find_unique(directory: Path, names: Iterable[str]) -> Optional[Path]:
    matches: List[Path] = []
    for name in names:
        matches.extend(path for path in directory.rglob(name) if path.is_file())
    unique = sorted(set(path.resolve() for path in matches))
    if not unique:
        return None
    if len(unique) > 1:
        raise ValueError(
            f"Expected at most one of {tuple(names)} under {directory}, "
            f"found: {[str(path) for path in unique]}"
        )
    return unique[0]


def _load_score_metrics(path: Path) -> Dict[str, float]:
    if path.suffix.lower() == ".json":
        payload = _read_json(path)
    else:
        import torch

        payload = torch.load(str(path), map_location=torch.device("cpu"))
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("metrics"), Mapping
    ):
        raise ValueError(f"Score file lacks a metrics mapping: {path}")
    metrics = payload["metrics"]
    result: Dict[str, float] = {}
    for key in ("R1", "R5", "R10", "mAP", "mINP"):
        value = _as_float(metrics.get(key))
        if value is None or not math.isfinite(value):
            raise ValueError(f"Score file {path} lacks finite metric {key}.")
        result[key] = value
    return result


def _scenario_score(candidate_dir: Path, scenario: str) -> Optional[Path]:
    candidates = (
        candidate_dir / "eval" / scenario / "scores.pt",
        candidate_dir / "eval" / scenario / "scores.json",
        candidate_dir / "eval" / f"{scenario}.pt",
        candidate_dir / "eval" / f"{scenario}.json",
    )
    existing = [path.resolve() for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(
            f"Multiple score files for {candidate_dir.name}/{scenario}: "
            f"{[str(path) for path in existing]}"
        )
    return existing[0] if existing else None


def _scenario_row(
    rows: Sequence[Mapping[str, Any]], scenario: str
) -> Mapping[str, Any]:
    return next((row for row in rows if row.get("scenario") == scenario), {})


def _resolve_active_child(root: Path, pointer_name: str, default_name: str) -> Path:
    pointer = root / pointer_name
    if not pointer.is_file():
        return root / default_name
    relative = pointer.read_text(encoding="utf-8").strip()
    if not relative:
        raise ValueError(f"Empty active-path pointer: {pointer}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Active-path pointer escapes its run directory: {pointer}"
        ) from error
    return resolved


def _active_candidate_dir(candidate_dir: Path) -> Path:
    return _resolve_active_child(candidate_dir, "active_attempt.txt", ".")


def _active_diagnostic_dir(run_dir: Path) -> Path:
    return _resolve_active_child(
        run_dir, "active_diagnostics.txt", "diagnostics"
    )


def _diagnostic_case(diagnosis: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    cases = diagnosis.get("cases", {})
    if not isinstance(cases, Mapping):
        return {}
    if label in cases and isinstance(cases[label], Mapping):
        return cases[label]
    if len(cases) == 1:
        only = next(iter(cases.values()))
        return only if isinstance(only, Mapping) else {}
    return {}


def _median_numeric(
    rows: Sequence[Mapping[str, Any]], key: str
) -> Optional[float]:
    values = [_as_float(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return median(finite) if finite else None


def collect_candidate(candidate_root: Path, label: str) -> Dict[str, Any]:
    config = dict(CANDIDATE_MATRIX[label])
    candidate_container = candidate_root / label
    candidate_dir = _active_candidate_dir(candidate_container)
    result: Dict[str, Any] = {
        "candidate": label,
        **config,
        "complete": False,
        "has_nan": False,
        "has_inf": False,
        "has_oom": False,
        "metrics": {},
    }
    if not candidate_container.is_dir() or not candidate_dir.is_dir():
        result["collection_error"] = "candidate_directory_missing"
        return result

    config_path = candidate_dir / "configs.yaml"
    saved_config = (
        load_config_mapping(config_path) if config_path.is_file() else None
    )
    if saved_config is not None:
        for key, expected in config.items():
            if saved_config.get(key) != expected:
                raise ValueError(
                    f"{label} saved config mismatch for {key}: "
                    f"{saved_config.get(key)!r} != {expected!r}; "
                    f"config={config_path}"
                )
        result["config_path"] = str(config_path.resolve())
        result["config_valid"] = True

    metadata_path = candidate_dir / "candidate.json"
    if metadata_path.is_file():
        metadata = _read_json(metadata_path)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"candidate.json must contain an object: {metadata_path}")
        for key, expected in config.items():
            if key in metadata and metadata[key] != expected:
                raise ValueError(
                    f"{label} matrix mismatch for {key}: "
                    f"{metadata[key]!r} != {expected!r}"
                )
        result["has_nan"] = bool(metadata.get("has_nan", False))
        result["has_inf"] = bool(metadata.get("has_inf", False))
        result["has_oom"] = bool(metadata.get("has_oom", False))

    metric_rows: List[Dict[str, Any]] = []
    for scenario in SCREEN_SCENARIOS:
        score_path = _scenario_score(candidate_dir, scenario)
        if score_path is None:
            continue
        metrics = _load_score_metrics(score_path)
        result["metrics"][scenario] = metrics
        metric_rows.append(
            {
                "candidate": label,
                "scenario": scenario,
                **metrics,
                "score_file": str(score_path),
            }
        )
    result["metric_rows"] = metric_rows

    diagnostic_dir = _active_diagnostic_dir(candidate_dir)
    diagnosis_path = diagnostic_dir / "diagnosis.json"
    diagnosis = _read_json(diagnosis_path) if diagnosis_path.is_file() else {}
    case_diagnosis = (
        _diagnostic_case(diagnosis, label)
        if isinstance(diagnosis, Mapping)
        else {}
    )
    gate_rows = _read_csv(diagnostic_dir / "scenario_gate_statistics.csv")
    direction_rows = _read_csv(
        diagnostic_dir / "gate_direction_statistics.csv"
    )
    gradient_rows = _read_csv(diagnostic_dir / "gradient_diagnostics.csv")

    rgb_missing = _scenario_row(gate_rows, "rgb_missing")
    ir_missing = _scenario_row(gate_rows, "ir_missing")
    direction_by_scenario = {
        str(row.get("scenario")): row for row in direction_rows
    }

    def alpha_drop(scenario: str) -> Optional[float]:
        row = direction_by_scenario.get(scenario, {})
        delta = _as_float(row.get("degraded_alpha_delta_mean"))
        return -delta if delta is not None else None

    direction_values = [
        _as_float(row.get("correct_direction_fraction"))
        for row in direction_rows
        if str(row.get("scenario")) in {
            "rgb_noise_0.10",
            "ir_noise_0.10",
            "rgb_blur_k7",
            "ir_blur_k7",
        }
    ]
    direction_values = [value for value in direction_values if value is not None]
    raw_dtypes = {
        str(row.get("raw_logits_dtype", "")) for row in gate_rows if row
    }
    alarms = case_diagnosis.get("alarms", {})
    parameter_updated = bool(
        isinstance(alarms, Mapping)
        and "PARAMETER_NOT_UPDATED" in alarms
        and not bool(alarms["PARAMETER_NOT_UPDATED"])
    )
    mechanism = {
        "candidate": label,
        "alpha_rgb_std": _as_float(case_diagnosis.get("alpha_rgb_std")),
        "alpha_target_pearson": _as_float(
            case_diagnosis.get("alpha_target_pearson_degraded_scenarios")
        ),
        "rgb_missing_alpha_rgb_mean": _as_float(
            rgb_missing.get("alpha_rgb_mean")
        ),
        "ir_missing_alpha_ir_mean": _as_float(
            ir_missing.get("alpha_ir_mean")
        ),
        "rgb_noise_alpha_drop": alpha_drop("rgb_noise_0.10"),
        "ir_noise_alpha_drop": alpha_drop("ir_noise_0.10"),
        "rgb_blur_alpha_drop": alpha_drop("rgb_blur_k7"),
        "ir_blur_alpha_drop": alpha_drop("ir_blur_k7"),
        "direction_correct_fraction": (
            sum(direction_values) / len(direction_values)
            if direction_values
            else None
        ),
        "weighted_rel_base_median": (
            _as_float(case_diagnosis.get("weighted_rel_base_ratio_median"))
            or _median_numeric(
                gradient_rows, "weighted_mraf_rel_over_loss_base"
            )
        ),
        "scorer_rel_only_gradient": (
            _as_float(case_diagnosis.get("rel_only_last_weight_grad_norm_median"))
            or _median_numeric(
                gradient_rows, "rel_only_scorer_last_weight_grad_norm"
            )
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
        len(result["metrics"]) == len(SCREEN_SCENARIOS)
        and diagnosis_path.is_file()
        and saved_config is not None
        and (candidate_dir / "best.pth").is_file()
        and (candidate_dir / "train.log").is_file()
    )
    return result


def apply_hard_filters(
    candidates: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    by_label = {row["candidate"]: row for row in candidates}
    baseline = by_label.get("S0", {})
    baseline_metrics = baseline.get("metrics", {})
    baseline_clean = _as_float(baseline_metrics.get("clean", {}).get("R1"))
    baseline_degraded_values = [
        _as_float(baseline_metrics.get(name, {}).get("R1"))
        for name in KEY_DEGRADED_SCENARIOS
    ]
    baseline_degraded = (
        sum(value for value in baseline_degraded_values if value is not None)
        / len(KEY_DEGRADED_SCENARIOS)
        if all(value is not None for value in baseline_degraded_values)
        else None
    )

    output: List[Dict[str, Any]] = []
    for candidate in candidates:
        metrics = candidate.get("metrics", {})
        mechanism = candidate.get("mechanism", {})
        clean = metrics.get("clean", {})
        clean_r1 = _as_float(clean.get("R1"))
        degraded = [
            _as_float(metrics.get(name, {}).get("R1"))
            for name in KEY_DEGRADED_SCENARIOS
        ]
        degraded_mean = (
            sum(value for value in degraded if value is not None)
            / len(KEY_DEGRADED_SCENARIOS)
            if all(value is not None for value in degraded)
            else None
        )
        failures: List[str] = []

        def require(condition: bool, name: str) -> None:
            if not condition:
                failures.append(name)

        require(bool(candidate.get("complete")), "INCOMPLETE")
        require(not candidate.get("has_nan"), "NAN")
        require(not candidate.get("has_inf"), "INF")
        require(not candidate.get("has_oom"), "OOM")
        require(
            _finite(mechanism.get("alpha_rgb_std"))
            and float(mechanism["alpha_rgb_std"]) >= 0.005,
            "ALPHA_STD",
        )
        require(
            _finite(mechanism.get("alpha_target_pearson"))
            and float(mechanism["alpha_target_pearson"]) >= 0.30,
            "PEARSON",
        )
        require(
            _finite(mechanism.get("rgb_missing_alpha_rgb_mean"))
            and float(mechanism["rgb_missing_alpha_rgb_mean"]) <= 0.35,
            "RGB_MISSING_WEIGHT",
        )
        require(
            _finite(mechanism.get("ir_missing_alpha_ir_mean"))
            and float(mechanism["ir_missing_alpha_ir_mean"]) <= 0.35,
            "IR_MISSING_WEIGHT",
        )
        for key, label in (
            ("rgb_noise_alpha_drop", "RGB_NOISE_DIRECTION"),
            ("ir_noise_alpha_drop", "IR_NOISE_DIRECTION"),
            ("rgb_blur_alpha_drop", "RGB_BLUR_DIRECTION"),
            ("ir_blur_alpha_drop", "IR_BLUR_DIRECTION"),
        ):
            require(
                _finite(mechanism.get(key)) and float(mechanism[key]) >= 0.02,
                label,
            )
        ratio = _as_float(mechanism.get("weighted_rel_base_median"))
        require(ratio is not None and 1e-3 <= ratio <= 5e-2, "LOSS_RATIO")
        require(
            clean_r1 is not None
            and baseline_clean is not None
            and clean_r1 >= baseline_clean - 0.50,
            "CLEAN_R1_REGRESSION",
        )
        require(
            degraded_mean is not None
            and baseline_degraded is not None
            and degraded_mean >= baseline_degraded - 0.20,
            "DEGRADED_R1_REGRESSION",
        )
        require(
            bool(mechanism.get("scorer_parameter_updated")),
            "PARAMETER_NOT_UPDATED",
        )
        require(
            bool(mechanism.get("fusion_path_sensitive")),
            "FUSION_PATH_NOT_SENSITIVE",
        )
        row = {
            "candidate": candidate["candidate"],
            **CANDIDATE_MATRIX[candidate["candidate"]],
            "complete": bool(candidate.get("complete")),
            "has_nan": bool(candidate.get("has_nan")),
            "has_inf": bool(candidate.get("has_inf")),
            "has_oom": bool(candidate.get("has_oom")),
            "clean_R1": clean_r1,
            "clean_mAP": _as_float(clean.get("mAP")),
            "clean_mINP": _as_float(clean.get("mINP")),
            "key_degraded_mean_R1": degraded_mean,
            "relative_clean_mean_drop_R1": (
                clean_r1 - degraded_mean
                if clean_r1 is not None and degraded_mean is not None
                else None
            ),
            **mechanism,
            "hard_filter_pass": not failures,
            "failed_criteria": ",".join(failures),
        }
        output.append(row)
    return output


def rank_candidates(summary_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    passing = [row for row in summary_rows if row.get("hard_filter_pass")]

    def key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        missing_sum = float(row["rgb_missing_alpha_rgb_mean"]) + float(
            row["ir_missing_alpha_ir_mean"]
        )
        return (
            -float(row["key_degraded_mean_R1"]),
            -float(row["clean_R1"]),
            -float(row["alpha_target_pearson"]),
            missing_sum,
            float(row["mraf_rel_lambda"]),
            -float(row["mraf_gate_base_grad_scale"]),
            str(row["candidate"]),
        )

    ranked: List[Dict[str, Any]] = []
    for index, row in enumerate(sorted(passing, key=key), start=1):
        ranked.append(
            {
                "rank": index,
                "candidate": row["candidate"],
                "hard_filter_pass": True,
                "key_degraded_mean_R1": row["key_degraded_mean_R1"],
                "clean_R1": row["clean_R1"],
                "alpha_target_pearson": row["alpha_target_pearson"],
                "missing_modality_alpha_sum": (
                    float(row["rgb_missing_alpha_rgb_mean"])
                    + float(row["ir_missing_alpha_ir_mean"])
                ),
                "mraf_rel_lambda": row["mraf_rel_lambda"],
                "mraf_gate_base_grad_scale": row[
                    "mraf_gate_base_grad_scale"
                ],
            }
        )
    return ranked


def _write_tsv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def summarize(candidate_root: Path, output_dir: Path) -> Dict[str, Path]:
    candidate_root = candidate_root.resolve()
    output_dir = output_dir.resolve()
    if not candidate_root.is_dir():
        raise FileNotFoundError(f"Candidate root not found: {candidate_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty summary directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        collect_candidate(candidate_root, label) for label in CANDIDATE_MATRIX
    ]
    summary_rows = apply_hard_filters(candidates)
    ranking_rows = rank_candidates(summary_rows)
    metric_rows = [
        row for candidate in candidates for row in candidate.get("metric_rows", [])
    ]
    mechanism_rows = [
        candidate["mechanism"]
        for candidate in candidates
        if "mechanism" in candidate
    ]
    selected = ranking_rows[0]["candidate"] if ranking_rows else None
    if selected is None:
        nearest = sorted(
            summary_rows,
            key=lambda row: (
                len([item for item in str(row["failed_criteria"]).split(",") if item]),
                str(row["candidate"]),
            ),
        )[0]
        selection = {
            "status": "SCREEN_FAILED",
            "selected_candidate": None,
            "nearest_candidate": nearest["candidate"],
            "failed_criteria": nearest["failed_criteria"].split(","),
        }
    else:
        selection = {
            "status": "SELECTED",
            "selected_candidate": selected,
            "configuration": dict(CANDIDATE_MATRIX[selected]),
        }

    paths = {
        "metrics": output_dir / "rescue_screen_metrics.tsv",
        "mechanism": output_dir / "rescue_screen_mechanism.tsv",
        "summary": output_dir / "rescue_screen_summary.tsv",
        "ranking": output_dir / "rescue_candidate_ranking.tsv",
        "selected": output_dir / "selected_candidate.json",
        "report": output_dir / "screen_report.txt",
    }
    _write_tsv(paths["metrics"], METRICS_FIELDS, metric_rows)
    _write_tsv(paths["mechanism"], MECHANISM_FIELDS, mechanism_rows)
    _write_tsv(paths["summary"], SUMMARY_FIELDS, summary_rows)
    _write_tsv(paths["ranking"], RANKING_FIELDS, ranking_rows)
    paths["selected"].write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "MRAF gate rescue screen",
        f"status={selection['status']}",
        f"selected_candidate={selected or 'NONE'}",
        "thresholds_are_screening_rules_not_paper_conclusions=true",
        "",
    ]
    for row in summary_rows:
        report_lines.append(
            f"{row['candidate']}: pass={row['hard_filter_pass']} "
            f"failed={row['failed_criteria'] or 'NONE'}"
        )
    paths["report"].write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return paths


def _final_metrics(run_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_scenario: Dict[str, Dict[str, float]] = {}
    for scenario in FINAL_SCENARIOS:
        score_path = _scenario_score(run_dir, scenario)
        if score_path is None:
            raise FileNotFoundError(
                f"Final run is missing score file for {scenario}: {run_dir}"
            )
        metrics = _load_score_metrics(score_path)
        by_scenario[scenario] = metrics
        rows.append(
            {
                "scenario": scenario,
                **metrics,
                "score_file": str(score_path),
            }
        )
    degraded = [by_scenario[name]["R1"] for name in FINAL_SCENARIOS[1:]]
    clean_r1 = by_scenario["clean"]["R1"]
    aggregate = {
        "clean_R1": clean_r1,
        "clean_mAP": by_scenario["clean"]["mAP"],
        "clean_mINP": by_scenario["clean"]["mINP"],
        "degraded_mean_R1": sum(degraded) / len(degraded),
        "relative_clean_mean_drop_R1": (
            clean_r1 - sum(degraded) / len(degraded)
        ),
        "worst_scenario": min(
            FINAL_SCENARIOS[1:],
            key=lambda name: by_scenario[name]["R1"],
        ),
        "worst_R1": min(degraded),
    }
    return rows, aggregate


def _old_comparison_row(
    old_run_root: Optional[Path], case: str
) -> Dict[str, Any]:
    row = {
        "model": f"old_{case}_seed1",
        "run_dir": None,
        "clean_R1": None,
        "clean_mAP": None,
        "clean_mINP": None,
        "degraded_mean_R1": None,
        "alpha_rgb_std": None,
        "alpha_target_pearson": None,
        "weighted_rel_base_median": None,
    }
    if old_run_root is None or not old_run_root.is_dir():
        return row
    matches = sorted(
        path
        for path in old_run_root.rglob(f"*_{case}_seed1")
        if path.is_dir() and "candidates" not in path.parts
    )
    if len(matches) != 1:
        return row
    run_dir = matches[0].resolve()
    row["run_dir"] = str(run_dir)
    try:
        config_path = _find_unique(run_dir, ("configs.yaml",))
        if config_path is not None:
            load_config_mapping(config_path)
        clean_path = _scenario_score(run_dir, "clean")
        if clean_path is not None:
            clean = _load_score_metrics(clean_path)
            row.update(
                {
                    "clean_R1": clean["R1"],
                    "clean_mAP": clean["mAP"],
                    "clean_mINP": clean["mINP"],
                }
            )
        degraded_values = []
        for scenario in FINAL_SCENARIOS[1:]:
            score_path = _scenario_score(run_dir, scenario)
            if score_path is not None:
                degraded_values.append(_load_score_metrics(score_path)["R1"])
        if len(degraded_values) == len(FINAL_SCENARIOS) - 1:
            row["degraded_mean_R1"] = sum(degraded_values) / len(
                degraded_values
            )
        diagnosis_path = _find_unique(run_dir, ("diagnosis.json",))
        if diagnosis_path is not None:
            diagnosis = _read_json(diagnosis_path)
            case_row = _diagnostic_case(diagnosis, case)
            row["alpha_rgb_std"] = case_row.get("alpha_rgb_std")
            row["alpha_target_pearson"] = case_row.get(
                "alpha_target_pearson_degraded_scenarios"
            )
            row["weighted_rel_base_median"] = case_row.get(
                "weighted_rel_base_ratio_median"
            )
    except ConfigLoadError:
        raise
    except (FileNotFoundError, ValueError):
        # Historical runs are optional provenance; unavailable fields remain NA.
        pass
    return row


def summarize_final(
    final_run_dir: Path,
    output_dir: Path,
    old_run_root: Optional[Path] = None,
    base_config: Optional[Path] = None,
) -> Dict[str, Path]:
    final_run_dir = final_run_dir.resolve()
    output_dir = output_dir.resolve()
    if not final_run_dir.is_dir():
        raise FileNotFoundError(f"Final run directory not found: {final_run_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty final summary directory: {output_dir}"
        )
    final_config_path = final_run_dir / "configs.yaml"
    final_config = load_config_mapping(final_config_path)
    base_config_path = base_config.resolve() if base_config is not None else None
    historical_config = (
        load_config_mapping(base_config_path)
        if base_config_path is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, aggregate = _final_metrics(final_run_dir)
    diagnosis_path = _active_diagnostic_dir(final_run_dir) / "diagnosis.json"
    if not diagnosis_path.is_file():
        raise FileNotFoundError(f"Final diagnosis missing: {diagnosis_path}")
    diagnosis = _read_json(diagnosis_path)
    cases = diagnosis.get("cases", {})
    if not isinstance(cases, Mapping) or len(cases) != 1:
        raise ValueError("Final diagnosis must contain exactly one case.")
    mechanism = dict(next(iter(cases.values())))
    mechanism_row = {
        "alpha_rgb_std": mechanism.get("alpha_rgb_std"),
        "target_rgb_std_degraded_scenarios": mechanism.get(
            "target_rgb_std_degraded_scenarios"
        ),
        "alpha_target_pearson": mechanism.get(
            "alpha_target_pearson_degraded_scenarios"
        ),
        "weighted_rel_base_median": mechanism.get(
            "weighted_rel_base_ratio_median"
        ),
        "scorer_rel_only_gradient": mechanism.get(
            "rel_only_last_weight_grad_norm_median"
        ),
        "fusion_path_sensitive": mechanism.get("fusion_path_sensitive"),
        "fp16_quantization_warning": (
            mechanism.get("alarms", {}).get("FP16_QUANTIZATION_WARNING")
            if isinstance(mechanism.get("alarms"), Mapping)
            else None
        ),
    }
    comparison = [
        _old_comparison_row(old_run_root, case)
        for case in ("mraf_only", "mdrc_mraf", "mdrc_mraf_norel")
    ]
    comparison.append(
        {
            "model": "new_selected_formal",
            "run_dir": str(final_run_dir),
            "clean_R1": aggregate["clean_R1"],
            "clean_mAP": aggregate["clean_mAP"],
            "clean_mINP": aggregate["clean_mINP"],
            "degraded_mean_R1": aggregate["degraded_mean_R1"],
            "alpha_rgb_std": mechanism_row["alpha_rgb_std"],
            "alpha_target_pearson": mechanism_row[
                "alpha_target_pearson"
            ],
            "weighted_rel_base_median": mechanism_row[
                "weighted_rel_base_median"
            ],
        }
    )
    paths = {
        "metrics": output_dir / "final_metrics.tsv",
        "robustness": output_dir / "final_robustness_summary.tsv",
        "mechanism": output_dir / "final_mechanism_summary.tsv",
        "comparison": output_dir / "comparison_with_old_seed1.tsv",
        "report": output_dir / "FINAL_REPORT.txt",
    }
    _write_tsv(
        paths["metrics"],
        ("scenario", "R1", "R5", "R10", "mAP", "mINP", "score_file"),
        metric_rows,
    )
    _write_tsv(paths["robustness"], tuple(aggregate), [aggregate])
    _write_tsv(paths["mechanism"], tuple(mechanism_row), [mechanism_row])
    _write_tsv(
        paths["comparison"],
        (
            "model",
            "run_dir",
            "clean_R1",
            "clean_mAP",
            "clean_mINP",
            "degraded_mean_R1",
            "alpha_rgb_std",
            "alpha_target_pearson",
            "weighted_rel_base_median",
        ),
        comparison,
    )
    paths["report"].write_text(
        "\n".join(
            [
                "MRAF gate rescue final report",
                f"final_run_dir={final_run_dir}",
                f"formal_config={final_config_path.resolve()}",
                f"formal_config_valid={bool(final_config)}",
                f"base_config={base_config_path or 'NONE'}",
                f"base_config_valid={historical_config is not None}",
                f"clean_R1={aggregate['clean_R1']}",
                f"clean_mAP={aggregate['clean_mAP']}",
                f"clean_mINP={aggregate['clean_mINP']}",
                f"degraded_mean_R1={aggregate['degraded_mean_R1']}",
                f"alpha_rgb_std={mechanism_row['alpha_rgb_std']}",
                "alpha_target_pearson="
                f"{mechanism_row['alpha_target_pearson']}",
                "weighted_rel_base_median="
                f"{mechanism_row['weighted_rel_base_median']}",
                "fp16_quantization_warning="
                f"{mechanism_row['fp16_quantization_warning']}",
                "multi_seed_continuation_ready="
                f"{bool(mechanism_row['fusion_path_sensitive'])}",
                "thresholds_are_diagnostic_only=true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    candidate_mode = bool(str(args.candidate_root).strip())
    final_mode = bool(str(args.final_run_dir).strip())
    if candidate_mode == final_mode:
        raise ValueError(
            "Choose exactly one of --candidate-root or --final-run-dir."
        )
    if candidate_mode:
        paths = summarize(Path(args.candidate_root), Path(args.output_dir))
    else:
        old_root = (
            Path(args.old_run_root)
            if str(args.old_run_root).strip()
            else None
        )
        paths = summarize_final(
            Path(args.final_run_dir),
            Path(args.output_dir),
            old_run_root=old_root,
            base_config=(
                Path(args.base_config)
                if str(args.base_config).strip()
                else None
            ),
        )
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
