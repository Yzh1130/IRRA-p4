#!/usr/bin/env python3
"""Read-only mechanism diagnostics for trained MRAF checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_dataloader
from model import build_model
from model.mraf import compute_mraf_reliability_targets
from processor.processor import (
    _add_mdrc_to_total_loss,
    _add_mraf_to_total_loss,
    _build_degraded_branch_context,
    _build_total_loss,
    _compute_mdrc_terms,
    _compute_mraf_reliability_terms,
    _mdrc_enabled,
    _mraf_enabled,
    _should_execute_degraded_branch,
    apply_img_local_filter_stage,
)
from tools.revision.analyze_mraf_weights import SCENARIOS
from utils.checkpoint import Checkpointer, strip_prefix_if_present
from utils.iotools import load_train_configs
from utils.metrics import Evaluator


SUPPORTED_CASES = ("mraf_only", "mdrc_mraf", "mdrc_mraf_norel")
STAT_SUFFIXES = ("mean", "std", "min", "p10", "p50", "p90", "max")
GATE_VALUE_NAMES = ("logit_rgb", "logit_ir", "logit_gap", "alpha_rgb", "alpha_ir")
GRADIENT_PARAMETER_ROLES = (
    "scorer_first_weight",
    "scorer_first_bias",
    "scorer_last_weight",
    "scorer_last_bias",
)

DIAGNOSTIC_THRESHOLDS = {
    "alpha_std": 0.005,
    "target_std": 0.02,
    "alpha_target_pearson": 0.3,
    "weighted_rel_base_ratio": 1e-4,
    "parameter_l2_near_zero": 1e-8,
    "gradient_norm_near_zero": 1e-8,
    "fusion_feature_l2_near_zero": 1e-6,
}

ALARM_NAMES = (
    "PARAMETER_NOT_UPDATED",
    "TARGET_COLLAPSED_TO_HALF",
    "ALPHA_COLLAPSED_TO_HALF",
    "ALPHA_TARGET_LOW_CORRELATION",
    "RELIABILITY_LOSS_TOO_SMALL",
    "GATE_GRADIENT_TOO_SMALL",
    "FUSION_PATH_NOT_SENSITIVE",
    "FP16_QUANTIZATION_WARNING",
)

CHECKPOINT_PARAMETER_FIELDS = (
    "case",
    "checkpoint",
    "parameter_name",
    "role",
    "dtype",
    "shape",
    "numel",
    "mean",
    "std",
    "abs_mean",
    "abs_max",
    "l2_norm",
    "zero_fraction",
    "near_zero",
    "tensor_sha256",
)

SCENARIO_GATE_FIELDS = (
    "case",
    "scenario",
    "num_samples",
    "statistics_dtype",
    "raw_logits_dtype",
    "raw_alpha_dtype",
) + tuple(
    f"{value_name}_{suffix}"
    for value_name in GATE_VALUE_NAMES
    for suffix in STAT_SUFFIXES
) + ("max_weight_sum_error",)

SCENARIO_TARGET_FIELDS = (
    "case",
    "scenario",
    "num_samples",
    "drift_rgb_mean",
    "drift_rgb_std",
    "drift_ir_mean",
    "drift_ir_std",
    "target_rgb_mean",
    "target_rgb_std",
    "target_rgb_p10",
    "target_rgb_p50",
    "target_rgb_p90",
    "target_ir_mean",
    "target_ir_std",
    "target_ir_p10",
    "target_ir_p50",
    "target_ir_p90",
    "target_close_to_half",
    "alpha_target_mae",
    "alpha_target_mse",
    "alpha_target_pearson",
    "alpha_target_spearman",
    "alpha_target_direction_agreement",
)

GATE_DIRECTION_FIELDS = (
    "case",
    "scenario",
    "degraded_modality",
    "num_samples",
    "alpha_rgb_delta_mean",
    "alpha_rgb_delta_median",
    "alpha_ir_delta_mean",
    "alpha_ir_delta_median",
    "degraded_alpha_delta_mean",
    "degraded_alpha_delta_median",
    "counterpart_alpha_delta_mean",
    "counterpart_alpha_delta_median",
    "degraded_alpha_mean_decreased",
    "counterpart_alpha_mean_increased",
    "correct_direction_fraction",
)

GRADIENT_FIELDS = (
    "case",
    "batch_index",
    "diagnostic_epoch",
    "original_batch_size",
    "effective_microbatch_size",
    "gradient_microbatch_size",
    "cuda_memory_allocated_before_mb",
    "cuda_memory_allocated_after_forward_mb",
    "cuda_peak_memory_allocated_mb",
    "cuda_peak_memory_reserved_mb",
    "loss_base",
    "loss_mraf_rel",
    "lambda_mraf_rel_cur",
    "weighted_mraf_rel",
    "weighted_mraf_rel_over_loss_base",
    "total_loss",
) + tuple(
    f"{source}_{role}_grad_norm"
    for source in ("total", "rel_only")
    for role in GRADIENT_PARAMETER_ROLES
) + ("parameters_unchanged", "optimizer_step_executed")


@dataclass(frozen=True)
class RunArtifacts:
    case: str
    run_dir: Path
    config_path: Path
    checkpoint_path: Path
    train_log_path: Path

    def serializable(self) -> Dict[str, str]:
        return {
            "case": self.case,
            "run_dir": str(self.run_dir),
            "config_path": str(self.config_path),
            "checkpoint_path": str(self.checkpoint_path),
            "train_log_path": str(self.train_log_path),
        }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why trained MRAF gates collapse toward 0.5/0.5 without "
            "updating model parameters or checkpoints."
        )
    )
    parser.add_argument(
        "--run-root",
        default="",
        help="experiment root containing formal/*_<case>_seed1 directories",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="TextReID dataset root passed to the saved training configuration",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=SUPPORTED_CASES,
        help="one or more trained MRAF cases",
    )
    parser.add_argument(
        "--config-file",
        default="",
        help="direct single-run mode: saved configs.yaml path",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="direct single-run mode: trained checkpoint path",
    )
    parser.add_argument(
        "--case-label",
        default="",
        help="direct single-run mode: stable label used in diagnostic outputs",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="torch device used for forward/backward diagnostics",
    )
    parser.add_argument(
        "--degrade-seed",
        type=int,
        default=2026,
        help="fixed seed used by the existing evaluation degradation implementation",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=24,
        help="positive maximum number of gallery batches per scenario",
    )
    parser.add_argument(
        "--gradient-batches",
        type=int,
        default=3,
        help="number of training batches used for gradient probes (minimum: 3)",
    )
    parser.add_argument(
        "--gradient-microbatch-size",
        type=_positive_int,
        default=4,
        help=(
            "positive microbatch size used only by gradient diagnostics "
            "(default: 4)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new, non-existing directory for diagnostic outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only validate/discover inputs; do not build models or write outputs",
    )
    return parser


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _find_unique_file(run_dir: Path, filename: str) -> Path:
    direct = run_dir / filename
    if direct.is_file():
        return direct.resolve()
    matches = sorted(path.resolve() for path in run_dir.rglob(filename) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {filename!r} under {run_dir}, found "
            f"{len(matches)}: {[str(path) for path in matches]}"
        )
    return matches[0]


def discover_run_artifacts(
    run_root: Path,
    cases: Sequence[str],
) -> List[RunArtifacts]:
    formal_root = run_root.resolve() / "formal"
    if not formal_root.is_dir():
        raise FileNotFoundError(f"Formal experiment directory not found: {formal_root}")
    artifacts: List[RunArtifacts] = []
    for case in cases:
        if case not in SUPPORTED_CASES:
            raise ValueError(f"Unsupported case: {case}")
        matches = sorted(
            path.resolve()
            for path in formal_root.glob(f"*_{case}_seed1")
            if path.is_dir()
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one '*_{case}_seed1' directory under "
                f"{formal_root}, found {len(matches)}: "
                f"{[str(path) for path in matches]}"
            )
        run_dir = matches[0]
        artifacts.append(
            RunArtifacts(
                case=case,
                run_dir=run_dir,
                config_path=_find_unique_file(run_dir, "configs.yaml"),
                checkpoint_path=_find_unique_file(run_dir, "best.pth"),
                train_log_path=_find_unique_file(run_dir, "train.log"),
            )
        )
    return artifacts


def discover_direct_artifact(
    config_path: Path,
    checkpoint_path: Path,
    case_label: str,
) -> RunArtifacts:
    """Resolve one candidate without changing the legacy formal-run layout."""
    # Keep the caller-visible parent before following candidate symlinks:
    # pipeline configs/best.pth can point into train_output while train.log
    # intentionally remains at the candidate root.
    config_path = config_path.expanduser().absolute()
    checkpoint_path = checkpoint_path.expanduser().absolute()
    case_label = str(case_label).strip()
    if not case_label:
        raise ValueError("--case-label must not be empty in direct mode.")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    common = Path(
        os.path.commonpath(
            [str(config_path.parent), str(checkpoint_path.parent)]
        )
    ).resolve()
    return RunArtifacts(
        case=case_label,
        run_dir=common,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        train_log_path=_find_unique_file(common, "train.log"),
    )


def unwrap_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping.")
    candidate: Optional[Mapping[str, Any]] = None
    for key in ("model", "state_dict", "model_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping) and any(
            torch.is_tensor(value) for value in nested.values()
        ):
            candidate = nested
            break
    if candidate is None and any(torch.is_tensor(value) for value in checkpoint.values()):
        candidate = checkpoint
    if candidate is None:
        raise ValueError("Checkpoint does not contain a tensor state_dict.")
    if not all(isinstance(key, str) for key in candidate):
        raise TypeError("State-dict keys must be strings.")
    return strip_prefix_if_present(candidate, "module.")


def filter_mraf_parameters(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    parameters = {
        key: value
        for key, value in state_dict.items()
        if key.startswith("mraf_fusion.") and torch.is_tensor(value)
    }
    if not parameters:
        raise ValueError("Checkpoint contains no mraf_fusion.* parameters.")
    return dict(sorted(parameters.items()))


def identify_mraf_parameter_role(parameter_name: str) -> str:
    suffix_to_role = {
        "mraf_fusion.scorer.0.weight": "scorer_first_weight",
        "mraf_fusion.scorer.0.bias": "scorer_first_bias",
        "mraf_fusion.scorer.3.weight": "scorer_last_weight",
        "mraf_fusion.scorer.3.bias": "scorer_last_bias",
    }
    return suffix_to_role.get(parameter_name, "mraf_other")


def checkpoint_parameter_rows(
    case: str,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    state_dict = unwrap_state_dict(checkpoint)
    parameters = filter_mraf_parameters(state_dict)
    rows: List[Dict[str, Any]] = []
    for name, tensor in parameters.items():
        value = tensor.detach().cpu().to(torch.float32)
        flat = value.reshape(-1)
        l2_norm = float(torch.linalg.vector_norm(flat).item())
        rows.append(
            {
                "case": case,
                "checkpoint": str(checkpoint_path),
                "parameter_name": name,
                "role": identify_mraf_parameter_role(name),
                "dtype": str(tensor.dtype),
                "shape": "x".join(str(dim) for dim in tensor.shape),
                "numel": int(tensor.numel()),
                "mean": float(flat.mean().item()),
                "std": float(flat.std(unbiased=False).item()),
                "abs_mean": float(flat.abs().mean().item()),
                "abs_max": float(flat.abs().max().item()),
                "l2_norm": l2_norm,
                "zero_fraction": float((flat == 0).float().mean().item()),
                "near_zero": bool(
                    l2_norm <= DIAGNOSTIC_THRESHOLDS["parameter_l2_near_zero"]
                ),
                "tensor_sha256": _tensor_sha256(tensor),
            }
        )
    return rows, parameters


def compare_checkpoint_parameters(
    parameters_by_case: Mapping[str, Mapping[str, torch.Tensor]],
) -> Dict[str, Any]:
    cases = list(parameters_by_case)
    comparisons: List[Dict[str, Any]] = []
    for first_index, first_case in enumerate(cases):
        for second_case in cases[first_index + 1 :]:
            first = parameters_by_case[first_case]
            second = parameters_by_case[second_case]
            common_names = sorted(set(first).intersection(second))
            missing = sorted(set(first).symmetric_difference(second))
            max_abs_diff = 0.0
            l2_sq = 0.0
            for name in common_names:
                first_value = first[name].detach().cpu().to(torch.float32)
                second_value = second[name].detach().cpu().to(torch.float32)
                if first_value.shape != second_value.shape:
                    missing.append(f"shape_mismatch:{name}")
                    continue
                delta = first_value - second_value
                max_abs_diff = max(max_abs_diff, float(delta.abs().max().item()))
                l2_sq += float(delta.square().sum().item())
            comparisons.append(
                {
                    "first_case": first_case,
                    "second_case": second_case,
                    "common_parameter_count": len(common_names),
                    "incompatible_parameters": missing,
                    "max_abs_diff": max_abs_diff,
                    "l2_diff": math.sqrt(l2_sq),
                    "identical": not missing and max_abs_diff == 0.0,
                }
            )
    return {
        "pairwise": comparisons,
        "all_mraf_parameters_identical": bool(comparisons)
        and all(row["identical"] for row in comparisons),
    }


def as_fp32_vector(values: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    return values.detach().cpu().to(torch.float32).reshape(-1)


def summarize_vector(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    vector = as_fp32_vector(values)
    if vector.numel() == 0:
        raise ValueError(f"Cannot summarize an empty tensor for {prefix}.")
    quantiles = torch.quantile(
        vector,
        vector.new_tensor([0.10, 0.50, 0.90]),
    )
    return {
        f"{prefix}_mean": float(vector.mean().item()),
        f"{prefix}_std": float(vector.std(unbiased=False).item()),
        f"{prefix}_min": float(vector.min().item()),
        f"{prefix}_p10": float(quantiles[0].item()),
        f"{prefix}_p50": float(quantiles[1].item()),
        f"{prefix}_p90": float(quantiles[2].item()),
        f"{prefix}_max": float(vector.max().item()),
    }


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    vector = as_fp32_vector(values).to(torch.float64)
    order = torch.argsort(vector, stable=True)
    sorted_values = vector[order]
    sorted_ranks = torch.empty_like(sorted_values)
    start = 0
    while start < sorted_values.numel():
        end = start + 1
        while end < sorted_values.numel() and bool(
            sorted_values[end] == sorted_values[start]
        ):
            end += 1
        average_rank = (float(start) + float(end - 1)) / 2.0
        sorted_ranks[start:end] = average_rank
        start = end
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def pearson_correlation(
    first: torch.Tensor,
    second: torch.Tensor,
    eps: float = 1e-12,
) -> Optional[float]:
    first_vector = as_fp32_vector(first).to(torch.float64)
    second_vector = as_fp32_vector(second).to(torch.float64)
    if first_vector.shape != second_vector.shape:
        raise ValueError("Correlation inputs must have matching shapes.")
    if first_vector.numel() < 2:
        return None
    first_centered = first_vector - first_vector.mean()
    second_centered = second_vector - second_vector.mean()
    denominator = torch.linalg.vector_norm(first_centered) * torch.linalg.vector_norm(
        second_centered
    )
    if not bool(torch.isfinite(denominator)) or float(denominator.item()) <= eps:
        return None
    correlation = (first_centered * second_centered).sum() / denominator
    return float(correlation.clamp(-1.0, 1.0).item())


def spearman_correlation(
    first: torch.Tensor,
    second: torch.Tensor,
) -> Optional[float]:
    first_vector = as_fp32_vector(first)
    second_vector = as_fp32_vector(second)
    if first_vector.shape != second_vector.shape:
        raise ValueError("Correlation inputs must have matching shapes.")
    return pearson_correlation(
        _average_ranks(first_vector),
        _average_ranks(second_vector),
    )


def alpha_target_metrics(
    alpha: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Optional[float]]:
    alpha_fp32 = alpha.detach().cpu().to(torch.float32)
    target_fp32 = target.detach().cpu().to(torch.float32)
    if alpha_fp32.shape != target_fp32.shape or alpha_fp32.dim() != 2:
        raise ValueError("alpha and target must be matching 2D tensors.")
    if alpha_fp32.shape[1] != 2:
        raise ValueError("alpha and target must have shape [N, 2].")
    delta = alpha_fp32 - target_fp32
    alpha_gap = alpha_fp32[:, 0] - alpha_fp32[:, 1]
    target_gap = target_fp32[:, 0] - target_fp32[:, 1]
    direction_agreement = (
        torch.sign(alpha_gap) == torch.sign(target_gap)
    ).to(torch.float32)
    return {
        "alpha_target_mae": float(delta.abs().mean().item()),
        "alpha_target_mse": float(delta.square().mean().item()),
        # RGB and IR weights are complementary. Report one independent channel
        # rather than flattening two duplicated/complementary observations.
        "alpha_target_pearson": pearson_correlation(
            alpha_fp32[:, 0],
            target_fp32[:, 0],
        ),
        "alpha_target_spearman": spearman_correlation(
            alpha_fp32[:, 0],
            target_fp32[:, 0],
        ),
        "alpha_target_direction_agreement": float(
            direction_agreement.mean().item()
        ),
    }


def gate_direction_row(
    case: str,
    scenario: str,
    clean_alpha: torch.Tensor,
    degraded_alpha: torch.Tensor,
) -> Dict[str, Any]:
    clean = clean_alpha.detach().cpu().to(torch.float32)
    degraded = degraded_alpha.detach().cpu().to(torch.float32)
    if clean.shape != degraded.shape or clean.dim() != 2 or clean.shape[1] != 2:
        raise ValueError("Clean/degraded alpha must both have shape [N, 2].")
    rgb_delta = degraded[:, 0] - clean[:, 0]
    ir_delta = degraded[:, 1] - clean[:, 1]
    if scenario.startswith("rgb_"):
        modality = "rgb"
        degraded_delta = rgb_delta
        counterpart_delta = ir_delta
        correct = (rgb_delta < 0.0) & (ir_delta > 0.0)
        correct_fraction: Optional[float] = float(correct.float().mean().item())
    elif scenario.startswith("ir_"):
        modality = "ir"
        degraded_delta = ir_delta
        counterpart_delta = rgb_delta
        correct = (ir_delta < 0.0) & (rgb_delta > 0.0)
        correct_fraction = float(correct.float().mean().item())
    elif scenario.startswith("both_"):
        modality = "both"
        degraded_delta = torch.full_like(rgb_delta, float("nan"))
        counterpart_delta = torch.full_like(rgb_delta, float("nan"))
        correct_fraction = None
    else:
        raise ValueError(f"Scenario has no degradation direction: {scenario}")
    return {
        "case": case,
        "scenario": scenario,
        "degraded_modality": modality,
        "num_samples": int(clean.shape[0]),
        "alpha_rgb_delta_mean": float(rgb_delta.mean().item()),
        "alpha_rgb_delta_median": float(torch.quantile(rgb_delta, 0.5).item()),
        "alpha_ir_delta_mean": float(ir_delta.mean().item()),
        "alpha_ir_delta_median": float(torch.quantile(ir_delta, 0.5).item()),
        "degraded_alpha_delta_mean": (
            None if modality == "both" else float(degraded_delta.mean().item())
        ),
        "degraded_alpha_delta_median": (
            None
            if modality == "both"
            else float(torch.quantile(degraded_delta, 0.5).item())
        ),
        "counterpart_alpha_delta_mean": (
            None if modality == "both" else float(counterpart_delta.mean().item())
        ),
        "counterpart_alpha_delta_median": (
            None
            if modality == "both"
            else float(torch.quantile(counterpart_delta, 0.5).item())
        ),
        "degraded_alpha_mean_decreased": (
            None if modality == "both" else bool(degraded_delta.mean().item() < 0.0)
        ),
        "counterpart_alpha_mean_increased": (
            None
            if modality == "both"
            else bool(counterpart_delta.mean().item() > 0.0)
        ),
        "correct_direction_fraction": correct_fraction,
    }


def _model_core(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


@contextlib.contextmanager
def force_mraf_gate(
    mraf_module: nn.Module,
    alpha_rgb: float,
    alpha_ir: float,
):
    alpha_rgb = float(alpha_rgb)
    alpha_ir = float(alpha_ir)
    if alpha_rgb < 0.0 or alpha_ir < 0.0:
        raise ValueError("Forced MRAF weights must be non-negative.")
    if not math.isclose(alpha_rgb + alpha_ir, 1.0, abs_tol=1e-12):
        raise ValueError("Forced MRAF weights must sum to 1.")
    had_instance_forward = "forward" in mraf_module.__dict__
    original_instance_forward = mraf_module.__dict__.get("forward")

    def _forced_forward(self, f_rgb, f_ir, return_aux=False):
        if not torch.is_tensor(f_rgb) or not torch.is_tensor(f_ir):
            raise TypeError("Forced MRAF inputs must be tensors.")
        if f_rgb.shape != f_ir.shape or f_rgb.dim() != 2:
            raise ValueError("Forced MRAF inputs must share shape [B, D].")
        alpha_fp32 = torch.tensor(
            [alpha_rgb, alpha_ir],
            device=f_rgb.device,
            dtype=torch.float32,
        ).view(1, 2).expand(f_rgb.shape[0], -1)
        alpha_for_fusion = alpha_fp32.to(dtype=f_rgb.dtype)
        fused = 2.0 * (
            alpha_for_fusion[:, 0:1] * f_rgb
            + alpha_for_fusion[:, 1:2] * f_ir
        )
        if not return_aux:
            return fused
        temperature = float(getattr(self, "gate_temperature", 1.0))
        logits = (
            torch.log(alpha_fp32.clamp_min(torch.finfo(torch.float32).tiny))
            * temperature
        )
        return fused, {
            "fused": fused,
            "alpha_rgb": alpha_for_fusion[:, 0:1],
            "alpha_ir": alpha_for_fusion[:, 1:2],
            "alpha": alpha_for_fusion,
            "logits": logits,
        }

    mraf_module.forward = MethodType(_forced_forward, mraf_module)
    try:
        yield mraf_module
    finally:
        if had_instance_forward:
            mraf_module.forward = original_instance_forward
        else:
            delattr(mraf_module, "forward")


def scorer_named_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    core = _model_core(model)
    if not bool(getattr(core, "mraf_enabled", False)) or not hasattr(
        core, "mraf_fusion"
    ):
        raise ValueError("Model does not have an enabled MRAF fusion module.")
    scorer = core.mraf_fusion.scorer
    linear_layers = [layer for layer in scorer if isinstance(layer, nn.Linear)]
    if len(linear_layers) < 2:
        raise ValueError("MRAF scorer must contain at least two Linear layers.")
    first = linear_layers[0]
    last = linear_layers[-1]
    named = [
        ("scorer_first_weight", first.weight),
        ("scorer_first_bias", first.bias),
        ("scorer_last_weight", last.weight),
        ("scorer_last_bias", last.bias),
    ]
    if any(parameter is None for _, parameter in named):
        raise ValueError("MRAF scorer Linear layers must expose weight and bias.")
    return named


def measure_loss_gradients(
    losses: Mapping[str, Optional[torch.Tensor]],
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> Dict[str, Dict[str, Optional[float]]]:
    active = [
        (name, loss)
        for name, loss in losses.items()
        if torch.is_tensor(loss) and bool(loss.requires_grad)
    ]
    results: Dict[str, Dict[str, Optional[float]]] = {}
    parameters = [parameter for _, parameter in named_parameters]
    active_index = 0
    for source_name, loss in losses.items():
        if not torch.is_tensor(loss) or not bool(loss.requires_grad):
            results[source_name] = {
                role: None for role, _ in named_parameters
            }
            continue
        retain_graph = active_index < len(active) - 1
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        results[source_name] = {
            role: (
                None
                if gradient is None
                else float(
                    torch.linalg.vector_norm(
                        gradient.detach().to(torch.float32)
                    ).item()
                )
            )
            for (role, _), gradient in zip(named_parameters, gradients)
        }
        active_index += 1
    return results


def _scorer_snapshot(
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_parameters
    }


def _snapshot_equal(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> bool:
    return first.keys() == second.keys() and all(
        torch.equal(first[name], second[name]) for name in first
    )


def infer_diagnostic_batch_size(batch: Mapping[str, Any]) -> int:
    if not isinstance(batch, Mapping):
        raise TypeError("Diagnostic batch must be a mapping.")
    preferred_keys = (
        "images",
        "thermal_images",
        "pids",
        "caption_ids",
        "mlm_labels",
    )
    ordered_keys = list(preferred_keys) + [
        key for key in batch if key not in preferred_keys
    ]
    for key in ordered_keys:
        if key not in batch:
            continue
        value = batch[key]
        if torch.is_tensor(value) and value.dim() > 0:
            size = int(value.shape[0])
            if size > 0:
                return size
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return len(value)
    raise ValueError("Could not infer a positive batch dimension.")


def slice_batch_for_diagnostics(
    batch: Mapping[str, Any],
    max_samples: int,
) -> Tuple[Dict[str, Any], int, int]:
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer.")
    original_batch_size = infer_diagnostic_batch_size(batch)
    effective_microbatch_size = min(original_batch_size, max_samples)
    sliced: Dict[str, Any] = {}
    for key, value in batch.items():
        if (
            torch.is_tensor(value)
            and value.dim() > 0
            and int(value.shape[0]) == original_batch_size
        ):
            sliced[key] = value[:effective_microbatch_size]
        elif isinstance(value, list) and len(value) == original_batch_size:
            sliced[key] = value[:effective_microbatch_size]
        elif isinstance(value, tuple) and len(value) == original_batch_size:
            sliced[key] = value[:effective_microbatch_size]
        else:
            sliced[key] = value
    return sliced, original_batch_size, effective_microbatch_size


def release_cuda_objects(*objects: Any) -> None:
    # Callers clear their own strong references before invoking this helper.
    # Dropping the temporary argument tuple here prevents the helper from
    # extending those lifetimes while cyclic garbage is collected.
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reset_cuda_peak_memory_stats(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    return float(torch.cuda.memory_allocated(device) / (1024.0**2))


def _cuda_memory_allocated_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_allocated(device) / (1024.0**2))


def _cuda_peak_memory_mb(device: torch.device) -> Tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        float(torch.cuda.max_memory_allocated(device) / (1024.0**2)),
        float(torch.cuda.max_memory_reserved(device) / (1024.0**2)),
    )


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _sample_ids(pid_batch: Any) -> torch.Tensor:
    if torch.is_tensor(pid_batch):
        return pid_batch.detach().cpu().to(torch.int64).reshape(-1)
    return torch.as_tensor(pid_batch, dtype=torch.int64).reshape(-1)


def _sample_order_hash(
    sample_batches: Sequence[torch.Tensor],
    feature_batches: Sequence[torch.Tensor],
) -> str:
    if len(sample_batches) != len(feature_batches):
        raise ValueError("Sample IDs and feature fingerprints must have equal batches.")
    digest = hashlib.sha256()
    sample_index = 0
    for sample_batch, feature_batch in zip(sample_batches, feature_batches):
        features = feature_batch.detach().cpu().to(torch.float32).contiguous()
        if features.shape[0] != sample_batch.numel():
            raise ValueError("Sample IDs and feature fingerprints must align.")
        for row_index, pid in enumerate(sample_batch.tolist()):
            digest.update(f"{sample_index}:{int(pid)}\n".encode("utf-8"))
            digest.update(features[row_index].view(torch.uint8).numpy().tobytes())
            sample_index += 1
    return digest.hexdigest()


def _require_eval_outputs(outputs: Mapping[str, torch.Tensor]) -> None:
    required = (
        "global_feat",
        "rgb_global_feat",
        "thermal_global_feat",
        "mraf_alpha",
        "mraf_logits",
    )
    missing = [key for key in required if key not in outputs]
    if missing:
        raise RuntimeError(f"MRAF eval outputs are missing keys: {missing}")


def _collect_clean_cache(
    model: nn.Module,
    img_loader: Iterable,
    device: torch.device,
    max_batches: int,
) -> Tuple[List[Dict[str, Any]], str]:
    cache: List[Dict[str, Any]] = []
    pid_batches: List[torch.Tensor] = []
    feature_batches: List[torch.Tensor] = []
    model.eval()
    for batch_index, data in enumerate(img_loader):
        if len(data) != 3:
            raise ValueError(
                "MRAF diagnostics require RGB-T gallery batches "
                "(pid, rgb_image, thermal_image)."
            )
        pids, rgb_image, thermal_image = data
        rgb_image = rgb_image.to(device)
        thermal_image = thermal_image.to(device)
        with torch.no_grad():
            outputs = model.encode_rgbt_eval(
                rgb_image,
                thermal_image,
                return_local_tokens=True,
            )
        _require_eval_outputs(outputs)
        pid_tensor = _sample_ids(pids)
        pid_batches.append(pid_tensor)
        feature_batches.append(
            torch.cat(
                [
                    outputs["rgb_global_feat"].detach().cpu().to(torch.float32),
                    outputs["thermal_global_feat"]
                    .detach()
                    .cpu()
                    .to(torch.float32),
                ],
                dim=1,
            )
        )
        cache.append(
            {
                "sample_ids": pid_tensor,
                "rgb_feature": outputs["rgb_global_feat"].detach().cpu(),
                "ir_feature": outputs["thermal_global_feat"].detach().cpu(),
                "alpha": outputs["mraf_alpha"].detach().cpu().to(torch.float32),
                "logits": outputs["mraf_logits"].detach().cpu().to(torch.float32),
                "raw_alpha_dtype": str(outputs["mraf_alpha"].dtype),
                "raw_logits_dtype": str(outputs["mraf_logits"].dtype),
            }
        )
        if batch_index + 1 >= max_batches:
            break
    if not cache:
        raise ValueError("No gallery batches were collected.")
    return cache, _sample_order_hash(pid_batches, feature_batches)


def _concat_records(records: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    tensor_keys = (
        "sample_ids",
        "logit_rgb",
        "logit_ir",
        "logit_gap",
        "alpha_rgb",
        "alpha_ir",
        "drift_rgb",
        "drift_ir",
        "target_rgb",
        "target_ir",
    )
    return {
        key: torch.cat([record[key] for record in records], dim=0)
        for key in tensor_keys
    }


def _record_from_outputs(
    sample_ids: torch.Tensor,
    clean_rgb: torch.Tensor,
    clean_ir: torch.Tensor,
    degraded_outputs: Mapping[str, torch.Tensor],
    target_temperature: float,
) -> Dict[str, torch.Tensor]:
    _require_eval_outputs(degraded_outputs)
    degraded_rgb = degraded_outputs["rgb_global_feat"]
    degraded_ir = degraded_outputs["thermal_global_feat"]
    clean_rgb_device = clean_rgb.to(
        device=degraded_rgb.device,
        dtype=degraded_rgb.dtype,
    )
    clean_ir_device = clean_ir.to(
        device=degraded_ir.device,
        dtype=degraded_ir.dtype,
    )
    target_outputs = compute_mraf_reliability_targets(
        clean_rgb=clean_rgb_device,
        clean_ir=clean_ir_device,
        degraded_rgb=degraded_rgb,
        degraded_ir=degraded_ir,
        target_temperature=target_temperature,
    )
    logits = degraded_outputs["mraf_logits"].detach().to(torch.float32).cpu()
    alpha = degraded_outputs["mraf_alpha"].detach().to(torch.float32).cpu()
    return {
        "sample_ids": sample_ids.detach().cpu(),
        "logit_rgb": logits[:, 0],
        "logit_ir": logits[:, 1],
        "logit_gap": logits[:, 0] - logits[:, 1],
        "alpha_rgb": alpha[:, 0],
        "alpha_ir": alpha[:, 1],
        "drift_rgb": target_outputs["drift_rgb"].detach().to(torch.float32).cpu(),
        "drift_ir": target_outputs["drift_ir"].detach().to(torch.float32).cpu(),
        "target_rgb": target_outputs["target_rgb"]
        .detach()
        .to(torch.float32)
        .cpu()
        .reshape(-1),
        "target_ir": target_outputs["target_ir"]
        .detach()
        .to(torch.float32)
        .cpu()
        .reshape(-1),
    }


def _clean_record(
    cache_item: Mapping[str, Any],
    target_temperature: float,
) -> Dict[str, torch.Tensor]:
    outputs = {
        "global_feat": cache_item["rgb_feature"] + cache_item["ir_feature"],
        "rgb_global_feat": cache_item["rgb_feature"],
        "thermal_global_feat": cache_item["ir_feature"],
        "mraf_alpha": cache_item["alpha"],
        "mraf_logits": cache_item["logits"],
    }
    return _record_from_outputs(
        sample_ids=cache_item["sample_ids"],
        clean_rgb=cache_item["rgb_feature"],
        clean_ir=cache_item["ir_feature"],
        degraded_outputs=outputs,
        target_temperature=target_temperature,
    )


def _collect_degraded_scenario(
    model: nn.Module,
    img_loader: Iterable,
    txt_loader: Iterable,
    clean_cache: Sequence[Mapping[str, Any]],
    scenario_name: str,
    degrade_mode: str,
    strength: float,
    blur_kernel: int,
    device: torch.device,
    degrade_seed: int,
    target_temperature: float,
) -> Tuple[Dict[str, torch.Tensor], str, str]:
    model.args.eval_degrade_mode = degrade_mode
    model.args.eval_degrade_strength = float(strength)
    model.args.eval_degrade_blur_kernel = int(blur_kernel)
    model.args.eval_degrade_seed = int(degrade_seed)
    evaluator = Evaluator(
        img_loader,
        txt_loader,
        source=f"mraf_mechanism_{scenario_name}",
    )
    records: List[Dict[str, torch.Tensor]] = []
    raw_logits_dtypes = set()
    raw_alpha_dtypes = set()
    for batch_index, data in enumerate(img_loader):
        if batch_index >= len(clean_cache):
            break
        if len(data) != 3:
            raise ValueError(
                "MRAF diagnostics require RGB-T gallery batches "
                "(pid, rgb_image, thermal_image)."
            )
        pids, rgb_image, thermal_image = data
        cached = clean_cache[batch_index]
        sample_ids = _sample_ids(pids)
        if not torch.equal(sample_ids, cached["sample_ids"]):
            raise RuntimeError(
                f"Sample order changed in scenario {scenario_name}, batch "
                f"{batch_index + 1}."
            )
        rgb_image = rgb_image.to(device)
        thermal_image = thermal_image.to(device)
        degraded_rgb, degraded_ir = evaluator._apply_eval_degradation(
            model,
            rgb_image,
            thermal_image,
        )
        with torch.no_grad():
            outputs = model.encode_rgbt_eval(
                degraded_rgb,
                degraded_ir,
                return_local_tokens=True,
            )
        raw_logits_dtypes.add(str(outputs["mraf_logits"].dtype))
        raw_alpha_dtypes.add(str(outputs["mraf_alpha"].dtype))
        records.append(
            _record_from_outputs(
                sample_ids=sample_ids,
                clean_rgb=cached["rgb_feature"],
                clean_ir=cached["ir_feature"],
                degraded_outputs=outputs,
                target_temperature=target_temperature,
            )
        )
    if len(records) != len(clean_cache):
        raise RuntimeError(
            f"Scenario {scenario_name} collected {len(records)} batches, "
            f"expected {len(clean_cache)}."
        )
    return (
        _concat_records(records),
        ",".join(sorted(raw_logits_dtypes)),
        ",".join(sorted(raw_alpha_dtypes)),
    )


def _gate_statistics_row(
    case: str,
    scenario: str,
    record: Mapping[str, torch.Tensor],
    raw_logits_dtype: str,
    raw_alpha_dtype: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "case": case,
        "scenario": scenario,
        "num_samples": int(record["alpha_rgb"].numel()),
        "statistics_dtype": str(torch.float32),
        "raw_logits_dtype": raw_logits_dtype,
        "raw_alpha_dtype": raw_alpha_dtype,
    }
    for value_name in GATE_VALUE_NAMES:
        row.update(summarize_vector(record[value_name], value_name))
    row["max_weight_sum_error"] = float(
        (
            record["alpha_rgb"].to(torch.float32)
            + record["alpha_ir"].to(torch.float32)
            - 1.0
        )
        .abs()
        .max()
        .item()
    )
    return row


def _target_statistics_row(
    case: str,
    scenario: str,
    record: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    target_rgb = as_fp32_vector(record["target_rgb"])
    target_ir = as_fp32_vector(record["target_ir"])
    alpha = torch.stack(
        [
            as_fp32_vector(record["alpha_rgb"]),
            as_fp32_vector(record["alpha_ir"]),
        ],
        dim=1,
    )
    target = torch.stack([target_rgb, target_ir], dim=1)
    metrics = alpha_target_metrics(alpha, target)
    target_close = (
        float(target_rgb.std(unbiased=False).item())
        < DIAGNOSTIC_THRESHOLDS["target_std"]
        and abs(float(target_rgb.mean().item()) - 0.5)
        < DIAGNOSTIC_THRESHOLDS["target_std"]
    )
    row: Dict[str, Any] = {
        "case": case,
        "scenario": scenario,
        "num_samples": int(target_rgb.numel()),
        "drift_rgb_mean": float(
            as_fp32_vector(record["drift_rgb"]).mean().item()
        ),
        "drift_rgb_std": float(
            as_fp32_vector(record["drift_rgb"]).std(unbiased=False).item()
        ),
        "drift_ir_mean": float(as_fp32_vector(record["drift_ir"]).mean().item()),
        "drift_ir_std": float(
            as_fp32_vector(record["drift_ir"]).std(unbiased=False).item()
        ),
        "target_rgb_mean": float(target_rgb.mean().item()),
        "target_rgb_std": float(target_rgb.std(unbiased=False).item()),
        "target_rgb_p10": float(torch.quantile(target_rgb, 0.10).item()),
        "target_rgb_p50": float(torch.quantile(target_rgb, 0.50).item()),
        "target_rgb_p90": float(torch.quantile(target_rgb, 0.90).item()),
        "target_ir_mean": float(target_ir.mean().item()),
        "target_ir_std": float(target_ir.std(unbiased=False).item()),
        "target_ir_p10": float(torch.quantile(target_ir, 0.10).item()),
        "target_ir_p50": float(torch.quantile(target_ir, 0.50).item()),
        "target_ir_p90": float(torch.quantile(target_ir, 0.90).item()),
        "target_close_to_half": bool(target_close),
    }
    row.update(metrics)
    return row


def collect_scenario_diagnostics(
    case: str,
    model: nn.Module,
    img_loader: Iterable,
    txt_loader: Iterable,
    device: torch.device,
    degrade_seed: int,
    max_batches: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Dict[str, torch.Tensor]],
    str,
]:
    clean_cache, order_hash = _collect_clean_cache(
        model=model,
        img_loader=img_loader,
        device=device,
        max_batches=max_batches,
    )
    target_temperature = float(
        getattr(model.args, "mraf_target_temperature", 0.1)
    )
    clean_records = [
        _clean_record(cache_item, target_temperature) for cache_item in clean_cache
    ]
    clean = _concat_records(clean_records)
    clean_logits_dtype = ",".join(
        sorted({item["raw_logits_dtype"] for item in clean_cache})
    )
    clean_alpha_dtype = ",".join(
        sorted({item["raw_alpha_dtype"] for item in clean_cache})
    )

    gate_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    direction_rows: List[Dict[str, Any]] = []
    sample_records: Dict[str, Dict[str, torch.Tensor]] = {}
    for scenario_name, mode, strength, blur_kernel in SCENARIOS:
        if scenario_name == "clean":
            record = clean
            raw_logits_dtype = clean_logits_dtype
            raw_alpha_dtype = clean_alpha_dtype
        else:
            record, raw_logits_dtype, raw_alpha_dtype = _collect_degraded_scenario(
                model=model,
                img_loader=img_loader,
                txt_loader=txt_loader,
                clean_cache=clean_cache,
                scenario_name=scenario_name,
                degrade_mode=mode,
                strength=strength,
                blur_kernel=blur_kernel,
                device=device,
                degrade_seed=degrade_seed,
                target_temperature=target_temperature,
            )
            clean_alpha = torch.stack(
                [clean["alpha_rgb"], clean["alpha_ir"]],
                dim=1,
            )
            degraded_alpha = torch.stack(
                [record["alpha_rgb"], record["alpha_ir"]],
                dim=1,
            )
            direction_rows.append(
                gate_direction_row(
                    case=case,
                    scenario=scenario_name,
                    clean_alpha=clean_alpha,
                    degraded_alpha=degraded_alpha,
                )
            )
        gate_rows.append(
            _gate_statistics_row(
                case=case,
                scenario=scenario_name,
                record=record,
                raw_logits_dtype=raw_logits_dtype,
                raw_alpha_dtype=raw_alpha_dtype,
            )
        )
        target_rows.append(
            _target_statistics_row(case, scenario_name, record)
        )
        sample_records[scenario_name] = {
            key: value.detach().cpu().to(
                torch.int64 if key == "sample_ids" else torch.float32
            )
            for key, value in record.items()
        }
    return gate_rows, target_rows, direction_rows, sample_records, order_hash


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _feature_sha256(feature: torch.Tensor) -> str:
    return _tensor_sha256(feature.detach().cpu().to(torch.float32))


def run_fusion_path_check(
    case: str,
    model: nn.Module,
    img_loader: Iterable,
    txt_loader: Iterable,
    device: torch.device,
) -> Dict[str, Any]:
    core = _model_core(model)
    if not hasattr(core, "mraf_fusion"):
        raise ValueError("Fusion-path check requires an MRAF-enabled model.")
    was_training = model.training
    model.eval()
    mraf_module = core.mraf_fusion
    state_before = _module_state_sha256(mraf_module)
    image_data = next(iter(img_loader))
    text_data = next(iter(txt_loader))
    if len(image_data) != 3 or len(text_data) != 2:
        raise ValueError("Unexpected RGB-T image/text evaluation batch format.")
    _, rgb_image, thermal_image = image_data
    _, caption_ids = text_data
    rgb_image = rgb_image.to(device)
    thermal_image = thermal_image.to(device)
    caption_ids = caption_ids.to(device)
    modes = (
        ("learned_gate", None),
        ("fixed_half", (0.5, 0.5)),
        ("rgb_only", (1.0, 0.0)),
        ("ir_only", (0.0, 1.0)),
    )
    mode_values: Dict[str, Dict[str, Any]] = {}
    try:
        with torch.no_grad():
            text_feature = core.encode_text_eval(caption_ids).to(torch.float32)
        for mode_name, forced_weights in modes:
            manager = (
                contextlib.nullcontext()
                if forced_weights is None
                else force_mraf_gate(
                    mraf_module,
                    forced_weights[0],
                    forced_weights[1],
                )
            )
            with manager, torch.no_grad():
                outputs = core.encode_rgbt_eval(
                    rgb_image,
                    thermal_image,
                    return_local_tokens=True,
                )
                feature = outputs["global_feat"].detach().to(torch.float32).cpu()
            image_norm = F.normalize(feature, p=2, dim=-1, eps=1e-12)
            text_norm = F.normalize(
                text_feature.detach().cpu(),
                p=2,
                dim=-1,
                eps=1e-12,
            )
            scores = text_norm @ image_norm.t()
            mode_values[mode_name] = {
                "feature": feature,
                "scores": scores,
                "top1": scores.argmax(dim=1),
                "feature_sha256": _feature_sha256(feature),
            }
    finally:
        model.train(was_training)
    state_after = _module_state_sha256(mraf_module)
    if state_before != state_after:
        raise RuntimeError(
            "MRAF module state changed during the read-only fusion-path check."
        )
    learned = mode_values["learned_gate"]
    comparisons: Dict[str, Dict[str, Any]] = {}
    for mode_name, values in mode_values.items():
        delta = values["feature"] - learned["feature"]
        feature_l2 = torch.linalg.vector_norm(delta, dim=1)
        cosine = F.cosine_similarity(
            values["feature"],
            learned["feature"],
            dim=1,
            eps=1e-12,
        )
        score_delta = values["scores"] - learned["scores"]
        comparisons[mode_name] = {
            "feature_sha256": values["feature_sha256"],
            "feature_l2_mean_vs_learned": float(feature_l2.mean().item()),
            "feature_l2_max_vs_learned": float(feature_l2.max().item()),
            "feature_cosine_mean_vs_learned": float(cosine.mean().item()),
            "score_abs_mean_change_vs_learned": float(
                score_delta.abs().mean().item()
            ),
            "score_l2_change_vs_learned": float(
                torch.linalg.vector_norm(score_delta).item()
            ),
            "top1_index_change_fraction_vs_learned": float(
                (values["top1"] != learned["top1"]).float().mean().item()
            ),
        }
    sensitive = max(
        comparisons["rgb_only"]["feature_l2_max_vs_learned"],
        comparisons["ir_only"]["feature_l2_max_vs_learned"],
    ) > DIAGNOSTIC_THRESHOLDS["fusion_feature_l2_near_zero"]
    return {
        "case": case,
        "eval_use_fused_mu": bool(getattr(core, "eval_use_fused_mu", False)),
        "num_images": int(rgb_image.shape[0]),
        "num_texts": int(caption_ids.shape[0]),
        "mraf_state_sha256_before": state_before,
        "mraf_state_sha256_after": state_after,
        "state_restored": state_before == state_after,
        "fusion_path_sensitive": bool(sensitive),
        "modes": comparisons,
    }


def _run_gradient_probe(
    source: str,
    model: nn.Module,
    batch: Mapping[str, Any],
    args: Any,
    device: torch.device,
    diagnostic_epoch: int,
    named_parameters: Sequence[Tuple[str, nn.Parameter]],
) -> Dict[str, Any]:
    if source not in {"total", "rel_only"}:
        raise ValueError(f"Unsupported gradient probe source: {source}")

    ret = None
    base_loss = None
    total_loss = None
    degraded_context = None
    mdrc_terms = None
    mraf_terms = None
    weighted_rel = None
    probe_loss = None
    model.zero_grad(set_to_none=True)
    release_cuda_objects()
    allocated_before_mb = _reset_cuda_peak_memory_stats(device)
    try:
        ret = model(batch)
        allocated_after_forward_mb = _cuda_memory_allocated_mb(device)

        if source == "total":
            base_loss = _build_total_loss(args, ret, epoch=diagnostic_epoch)
            total_loss = base_loss

        if _should_execute_degraded_branch(args, diagnostic_epoch):
            degraded_context = _build_degraded_branch_context(
                args,
                model,
                batch,
            )

        if source == "total" and _mdrc_enabled(args):
            mdrc_terms = _compute_mdrc_terms(
                args,
                model,
                batch,
                ret,
                diagnostic_epoch,
                degraded_context=degraded_context,
            )
            total_loss = _add_mdrc_to_total_loss(
                args,
                total_loss,
                mdrc_terms,
            )

        if _mraf_enabled(args):
            mraf_terms = _compute_mraf_reliability_terms(
                args,
                ret,
                degraded_context,
                diagnostic_epoch,
            )
            if mraf_terms:
                weighted_rel = mraf_terms["loss_mraf_rel_weighted"]
                if source == "total":
                    total_loss = _add_mraf_to_total_loss(
                        args,
                        total_loss,
                        mraf_terms,
                    )

        probe_loss = total_loss if source == "total" else weighted_rel
        gradient_values = measure_loss_gradients(
            {source: probe_loss},
            named_parameters,
        )[source]
        peak_allocated_mb, peak_reserved_mb = _cuda_peak_memory_mb(device)

        result: Dict[str, Any] = {
            "gradients": gradient_values,
            "rel_active": bool(mraf_terms),
            "cuda_memory_allocated_before_mb": allocated_before_mb,
            "cuda_memory_allocated_after_forward_mb": (
                allocated_after_forward_mb
            ),
            "cuda_peak_memory_allocated_mb": peak_allocated_mb,
            "cuda_peak_memory_reserved_mb": peak_reserved_mb,
        }
        if source == "total":
            if base_loss is None or total_loss is None:
                raise RuntimeError("Total-loss gradient probe did not build losses.")
            if mraf_terms:
                loss_mraf_rel = mraf_terms["loss_mraf_rel"]
                lambda_mraf = mraf_terms["lambda_mraf_rel_cur"]
                weighted_scalar = float(
                    weighted_rel.detach().to(torch.float32).item()
                )
            else:
                loss_mraf_rel = base_loss.detach().new_zeros(())
                lambda_mraf = base_loss.detach().new_zeros(())
                weighted_scalar = 0.0
            base_scalar = float(base_loss.detach().to(torch.float32).item())
            result["losses"] = {
                "loss_base": base_scalar,
                "loss_mraf_rel": float(
                    loss_mraf_rel.detach().to(torch.float32).item()
                ),
                "lambda_mraf_rel_cur": float(
                    lambda_mraf.detach().to(torch.float32).item()
                ),
                "weighted_mraf_rel": weighted_scalar,
                "weighted_mraf_rel_over_loss_base": (
                    weighted_scalar / max(abs(base_scalar), 1e-12)
                ),
                "total_loss": float(
                    total_loss.detach().to(torch.float32).item()
                ),
            }
        return result
    finally:
        model.zero_grad(set_to_none=True)
        del (
            ret,
            base_loss,
            total_loss,
            degraded_context,
            mdrc_terms,
            mraf_terms,
            weighted_rel,
            probe_loss,
        )
        release_cuda_objects()


def run_gradient_diagnostics(
    case: str,
    model: nn.Module,
    train_loader: Iterable,
    args: Any,
    device: torch.device,
    gradient_batches: int,
    gradient_microbatch_size: int,
    diagnostic_epoch: int,
) -> List[Dict[str, Any]]:
    if str(getattr(args, "degrade_aug_mode", "none")) != "none":
        raise ValueError(
            "Gradient diagnostics only support the formal degrade_aug_mode=none path."
        )
    if int(gradient_microbatch_size) <= 0:
        raise ValueError("gradient_microbatch_size must be a positive integer.")

    model.train()
    named_parameters = scorer_named_parameters(model)
    rows: List[Dict[str, Any]] = []
    for batch_index, original_batch in enumerate(train_loader):
        if batch_index >= gradient_batches:
            break
        (
            sliced_batch,
            original_batch_size,
            effective_microbatch_size,
        ) = slice_batch_for_diagnostics(
            original_batch,
            gradient_microbatch_size,
        )
        microbatch = _move_batch_to_device(sliced_batch, device)
        before = _scorer_snapshot(named_parameters)

        total_probe = _run_gradient_probe(
            source="total",
            model=model,
            batch=microbatch,
            args=args,
            device=device,
            diagnostic_epoch=diagnostic_epoch,
            named_parameters=named_parameters,
        )
        if total_probe["rel_active"]:
            rel_probe = _run_gradient_probe(
                source="rel_only",
                model=model,
                batch=microbatch,
                args=args,
                device=device,
                diagnostic_epoch=diagnostic_epoch,
                named_parameters=named_parameters,
            )
        else:
            rel_probe = {
                "gradients": {
                    role: None for role in GRADIENT_PARAMETER_ROLES
                },
                "cuda_memory_allocated_before_mb": 0.0,
                "cuda_memory_allocated_after_forward_mb": 0.0,
                "cuda_peak_memory_allocated_mb": 0.0,
                "cuda_peak_memory_reserved_mb": 0.0,
            }

        after = _scorer_snapshot(named_parameters)
        parameters_unchanged = _snapshot_equal(before, after)
        if not parameters_unchanged:
            raise RuntimeError(
                "MRAF scorer parameters changed during read-only gradient diagnostics."
            )

        memory_keys = (
            "cuda_memory_allocated_before_mb",
            "cuda_memory_allocated_after_forward_mb",
            "cuda_peak_memory_allocated_mb",
            "cuda_peak_memory_reserved_mb",
        )
        row: Dict[str, Any] = {
            "case": case,
            "batch_index": batch_index + 1,
            "diagnostic_epoch": int(diagnostic_epoch),
            "original_batch_size": original_batch_size,
            "effective_microbatch_size": effective_microbatch_size,
            "gradient_microbatch_size": int(gradient_microbatch_size),
            **{
                key: max(float(total_probe[key]), float(rel_probe[key]))
                for key in memory_keys
            },
            **total_probe["losses"],
            "parameters_unchanged": parameters_unchanged,
            "optimizer_step_executed": False,
        }
        for source, probe in (
            ("total", total_probe),
            ("rel_only", rel_probe),
        ):
            for role in GRADIENT_PARAMETER_ROLES:
                row[f"{source}_{role}_grad_norm"] = probe["gradients"][role]
        rows.append(row)

        del (
            before,
            after,
            total_probe,
            rel_probe,
            microbatch,
            sliced_batch,
            original_batch,
        )
        release_cuda_objects()

    model.zero_grad(set_to_none=True)
    if len(rows) != gradient_batches:
        raise RuntimeError(
            f"Gradient diagnostics collected {len(rows)} batches, expected "
            f"{gradient_batches}."
        )
    return rows


def _load_config_args(
    artifact: RunArtifacts,
    data_root: Path,
    training: bool,
) -> Any:
    args = copy.deepcopy(load_train_configs(str(artifact.config_path)))
    args.root_dir = str(data_root.resolve())
    args.training = bool(training)
    args.distributed = False
    if not bool(getattr(args, "mraf_enabled", False)):
        raise ValueError(f"Config does not enable MRAF: {artifact.config_path}")
    return args


def _checkpoint_epoch(checkpoint: Mapping[str, Any], args: Any) -> int:
    value = checkpoint.get("epoch", getattr(args, "num_epoch", 1))
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(getattr(args, "num_epoch", 1)))


def build_case_resources(
    artifact: RunArtifacts,
    data_root: Path,
    device: torch.device,
) -> Tuple[Any, nn.Module, Iterable, Iterable, Iterable, Mapping[str, Any], int]:
    train_args = _load_config_args(artifact, data_root, training=True)
    train_loader, _, _, train_num_classes = build_dataloader(train_args)
    eval_args = _load_config_args(artifact, data_root, training=False)
    img_loader, txt_loader, eval_num_classes = build_dataloader(eval_args)
    if int(train_num_classes) != int(eval_num_classes):
        raise RuntimeError(
            "Training/evaluation dataloaders disagree on num_classes: "
            f"{train_num_classes} vs {eval_num_classes}."
        )

    # Construct and load exactly one model per case. The same immutable
    # checkpoint instance is switched between eval() and train() for the
    # separate read-only diagnostic stages.
    model = build_model(train_args, num_classes=train_num_classes)
    checkpoint = Checkpointer(model).load(str(artifact.checkpoint_path))
    epoch = _checkpoint_epoch(checkpoint, train_args)
    apply_img_local_filter_stage(
        model,
        train_args,
        epoch,
        logger=None,
        prefix="MRAF_DIAGNOSTIC",
    )
    model.to(device).eval()
    return (
        train_args,
        model,
        train_loader,
        img_loader,
        txt_loader,
        checkpoint,
        epoch,
    )


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(torch.tensor(values, dtype=torch.float64).median().item())


def build_diagnosis(
    metadata: Mapping[str, Any],
    parameter_rows: Sequence[Mapping[str, Any]],
    checkpoint_comparison: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    fusion_checks: Mapping[str, Mapping[str, Any]],
    sample_records: Mapping[str, Mapping[str, Mapping[str, torch.Tensor]]],
) -> Dict[str, Any]:
    cases = metadata["cases"]
    case_results: Dict[str, Any] = {}
    for case in cases:
        case_parameter_rows = [row for row in parameter_rows if row["case"] == case]
        last_weight_rows = [
            row
            for row in case_parameter_rows
            if row["role"] == "scorer_last_weight"
        ]
        last_bias_rows = [
            row
            for row in case_parameter_rows
            if row["role"] == "scorer_last_bias"
        ]
        last_weight_l2 = (
            None if not last_weight_rows else float(last_weight_rows[0]["l2_norm"])
        )
        last_bias_l2 = (
            None if not last_bias_rows else float(last_bias_rows[0]["l2_norm"])
        )
        last_linear_l2 = (
            None
            if last_weight_l2 is None or last_bias_l2 is None
            else math.sqrt(last_weight_l2**2 + last_bias_l2**2)
        )
        records = sample_records[case]
        alpha_rgb = torch.cat(
            [records[name]["alpha_rgb"] for name, *_ in SCENARIOS],
            dim=0,
        ).to(torch.float32)
        degraded_names = [name for name, *_ in SCENARIOS if name != "clean"]
        target_rgb = torch.cat(
            [records[name]["target_rgb"] for name in degraded_names],
            dim=0,
        ).to(torch.float32)
        alpha_target_alpha = torch.cat(
            [
                torch.stack(
                    [
                        records[name]["alpha_rgb"],
                        records[name]["alpha_ir"],
                    ],
                    dim=1,
                )
                for name in degraded_names
            ],
            dim=0,
        )
        alpha_target_target = torch.cat(
            [
                torch.stack(
                    [
                        records[name]["target_rgb"],
                        records[name]["target_ir"],
                    ],
                    dim=1,
                )
                for name in degraded_names
            ],
            dim=0,
        )
        aggregate_metrics = alpha_target_metrics(
            alpha_target_alpha,
            alpha_target_target,
        )
        case_gradient_rows = [row for row in gradient_rows if row["case"] == case]
        ratios = [
            float(row["weighted_mraf_rel_over_loss_base"])
            for row in case_gradient_rows
        ]
        total_last_gradients = [
            float(row["total_scorer_last_weight_grad_norm"])
            for row in case_gradient_rows
            if row["total_scorer_last_weight_grad_norm"] is not None
        ]
        rel_last_gradients = [
            float(row["rel_only_scorer_last_weight_grad_norm"])
            for row in case_gradient_rows
            if row["rel_only_scorer_last_weight_grad_norm"] is not None
        ]
        gate_case_rows = [row for row in gate_rows if row["case"] == case]
        half_precision_seen = any(
            "float16" in str(row["raw_logits_dtype"])
            or "bfloat16" in str(row["raw_logits_dtype"])
            or "float16" in str(row["raw_alpha_dtype"])
            or "bfloat16" in str(row["raw_alpha_dtype"])
            for row in gate_case_rows
        )
        alpha_std = float(alpha_rgb.std(unbiased=False).item())
        target_std = float(target_rgb.std(unbiased=False).item())
        pearson = aggregate_metrics["alpha_target_pearson"]
        ratio_median = _median(ratios)
        total_gradient_median = _median(total_last_gradients)
        rel_gradient_median = _median(rel_last_gradients)
        fusion_sensitive = bool(fusion_checks[case]["fusion_path_sensitive"])
        alarms = {
            "PARAMETER_NOT_UPDATED": (
                last_linear_l2 is None
                or last_linear_l2
                <= DIAGNOSTIC_THRESHOLDS["parameter_l2_near_zero"]
                or bool(
                    checkpoint_comparison["all_mraf_parameters_identical"]
                )
            ),
            "TARGET_COLLAPSED_TO_HALF": (
                target_std < DIAGNOSTIC_THRESHOLDS["target_std"]
            ),
            "ALPHA_COLLAPSED_TO_HALF": (
                alpha_std < DIAGNOSTIC_THRESHOLDS["alpha_std"]
            ),
            "ALPHA_TARGET_LOW_CORRELATION": (
                pearson is None
                or pearson
                < DIAGNOSTIC_THRESHOLDS["alpha_target_pearson"]
            ),
            "RELIABILITY_LOSS_TOO_SMALL": (
                ratio_median is None
                or ratio_median
                < DIAGNOSTIC_THRESHOLDS["weighted_rel_base_ratio"]
            ),
            "GATE_GRADIENT_TOO_SMALL": (
                total_gradient_median is None
                or total_gradient_median
                <= DIAGNOSTIC_THRESHOLDS["gradient_norm_near_zero"]
            ),
            "FUSION_PATH_NOT_SENSITIVE": not fusion_sensitive,
            "FP16_QUANTIZATION_WARNING": half_precision_seen,
        }
        case_results[case] = {
            "alpha_rgb_std": alpha_std,
            "target_rgb_std_degraded_scenarios": target_std,
            "alpha_target_pearson_degraded_scenarios": pearson,
            "alpha_target_spearman_degraded_scenarios": aggregate_metrics[
                "alpha_target_spearman"
            ],
            "weighted_rel_base_ratio_median": ratio_median,
            "total_last_weight_grad_norm_median": total_gradient_median,
            "rel_only_last_weight_grad_norm_median": rel_gradient_median,
            "scorer_last_weight_l2_norm": last_weight_l2,
            "scorer_last_bias_l2_norm": last_bias_l2,
            "scorer_last_linear_l2_norm": last_linear_l2,
            "fusion_path_sensitive": fusion_sensitive,
            "statistics_converted_to_fp32": True,
            "alarms": alarms,
        }
    return {
        "metadata": metadata,
        "thresholds": DIAGNOSTIC_THRESHOLDS,
        "checkpoint_parameter_comparison": checkpoint_comparison,
        "cases": case_results,
    }


def write_summary(path: Path, diagnosis: Mapping[str, Any]) -> None:
    lines = [
        "MRAF mechanism diagnosis",
        f"git_head={diagnosis['metadata']['git_head']}",
        "gradient_microbatch_size="
        f"{diagnosis['metadata']['gradient_microbatch_size']}",
        "thresholds_are_diagnostic_only=true",
        "",
    ]
    comparison = diagnosis["checkpoint_parameter_comparison"]
    lines.append(
        "all_mraf_parameters_identical="
        f"{comparison['all_mraf_parameters_identical']}"
    )
    lines.append("")
    for case, result in diagnosis["cases"].items():
        lines.append(f"[{case}]")
        lines.append(f"alpha_rgb_std={result['alpha_rgb_std']}")
        lines.append(
            "target_rgb_std_degraded_scenarios="
            f"{result['target_rgb_std_degraded_scenarios']}"
        )
        lines.append(
            "alpha_target_pearson_degraded_scenarios="
            f"{result['alpha_target_pearson_degraded_scenarios']}"
        )
        lines.append(
            "weighted_rel_base_ratio_median="
            f"{result['weighted_rel_base_ratio_median']}"
        )
        lines.append(
            "total_last_weight_grad_norm_median="
            f"{result['total_last_weight_grad_norm_median']}"
        )
        lines.append(
            "rel_only_last_weight_grad_norm_median="
            f"{result['rel_only_last_weight_grad_norm_median']}"
        )
        lines.append(
            "scorer_last_linear_l2_norm="
            f"{result['scorer_last_linear_l2_norm']}"
        )
        lines.append(
            f"fusion_path_sensitive={result['fusion_path_sensitive']}"
        )
        for alarm_name in ALARM_NAMES:
            status = "ALERT" if result["alarms"][alarm_name] else "OK"
            lines.append(f"{alarm_name}={status}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _prepare_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(
            "Refusing to reuse an existing diagnostic output directory: "
            f"{resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _validate_cli(
    args: argparse.Namespace,
) -> Tuple[Optional[Path], Path, torch.device, List[RunArtifacts], str]:
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be a positive integer.")
    if args.gradient_batches <= 0:
        raise ValueError("--gradient-batches must be a positive integer.")
    if args.gradient_batches < 3 and not args.dry_run:
        raise ValueError(
            "--gradient-batches must be at least 3 for mechanism diagnostics."
        )
    if int(args.gradient_microbatch_size) <= 0:
        raise ValueError(
            "--gradient-microbatch-size must be a positive integer."
        )
    legacy_values = (bool(str(args.run_root).strip()), bool(args.cases))
    direct_values = (
        bool(str(args.config_file).strip()),
        bool(str(args.checkpoint).strip()),
        bool(str(args.case_label).strip()),
    )
    legacy_mode = all(legacy_values)
    direct_mode = all(direct_values)
    if any(legacy_values) and not legacy_mode:
        raise ValueError("Legacy mode requires both --run-root and --cases.")
    if any(direct_values) and not direct_mode:
        raise ValueError(
            "Direct mode requires --config-file, --checkpoint, and --case-label."
        )
    if legacy_mode == direct_mode:
        raise ValueError(
            "Choose exactly one input mode: --run-root/--cases or "
            "--config-file/--checkpoint/--case-label."
        )
    if args.cases and len(set(args.cases)) != len(args.cases):
        raise ValueError("--cases must not contain duplicate case names.")
    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if legacy_mode:
        run_root = Path(args.run_root).resolve()
        if not run_root.is_dir():
            raise FileNotFoundError(f"Run root not found: {run_root}")
        artifacts = discover_run_artifacts(run_root, args.cases)
        input_mode = "run_root"
    else:
        run_root = None
        artifacts = [
            discover_direct_artifact(
                Path(args.config_file),
                Path(args.checkpoint),
                args.case_label,
            )
        ]
        input_mode = "direct"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available() and not args.dry_run:
        raise RuntimeError("CUDA was requested but is not available.")
    return run_root, data_root, device, artifacts, input_mode


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_root, data_root, device, artifacts, input_mode = _validate_cli(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "git_head": _git_head(),
                    "data_root": str(data_root),
                    "input_mode": input_mode,
                    "device": str(device),
                    "degrade_seed": int(args.degrade_seed),
                    "max_batches": int(args.max_batches),
                    "gradient_batches": int(args.gradient_batches),
                    "gradient_microbatch_size": int(
                        args.gradient_microbatch_size
                    ),
                    "artifacts": [artifact.serializable() for artifact in artifacts],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    output_dir = _prepare_output_dir(Path(args.output_dir))
    parameter_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    direction_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    parameters_by_case: Dict[str, Dict[str, torch.Tensor]] = {}
    fusion_checks: Dict[str, Dict[str, Any]] = {}
    sample_records: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    metadata_cases: Dict[str, Dict[str, Any]] = {}

    for artifact in artifacts:
        raw_checkpoint = None
        case_model = None
        train_args = None
        train_loader = None
        img_loader = None
        txt_loader = None
        case_checkpoint = None
        try:
            _seed_everything(args.degrade_seed)
            raw_checkpoint = torch.load(
                str(artifact.checkpoint_path),
                map_location=torch.device("cpu"),
            )
            rows, mraf_parameters = checkpoint_parameter_rows(
                artifact.case,
                artifact.checkpoint_path,
                raw_checkpoint,
            )
            parameter_rows.extend(rows)
            parameters_by_case[artifact.case] = mraf_parameters
            raw_checkpoint = None
            release_cuda_objects()

            (
                train_args,
                case_model,
                train_loader,
                img_loader,
                txt_loader,
                case_checkpoint,
                diagnostic_epoch,
            ) = build_case_resources(
                artifact,
                data_root,
                device,
            )
            case_checkpoint = None

            case_model.eval()
            (
                case_gate_rows,
                case_target_rows,
                case_direction_rows,
                case_sample_records,
                order_hash,
            ) = collect_scenario_diagnostics(
                case=artifact.case,
                model=case_model,
                img_loader=img_loader,
                txt_loader=txt_loader,
                device=device,
                degrade_seed=args.degrade_seed,
                max_batches=args.max_batches,
            )
            gate_rows.extend(case_gate_rows)
            target_rows.extend(case_target_rows)
            direction_rows.extend(case_direction_rows)
            sample_records[artifact.case] = case_sample_records
            fusion_checks[artifact.case] = run_fusion_path_check(
                case=artifact.case,
                model=case_model,
                img_loader=img_loader,
                txt_loader=txt_loader,
                device=device,
            )

            _seed_everything(args.degrade_seed)
            gradient_rows.extend(
                run_gradient_diagnostics(
                    case=artifact.case,
                    model=case_model,
                    train_loader=train_loader,
                    args=train_args,
                    device=device,
                    gradient_batches=args.gradient_batches,
                    gradient_microbatch_size=(
                        args.gradient_microbatch_size
                    ),
                    diagnostic_epoch=diagnostic_epoch,
                )
            )

            metadata_cases[artifact.case] = {
                **artifact.serializable(),
                "checkpoint_sha256": _sha256_file(artifact.checkpoint_path),
                "checkpoint_epoch": int(diagnostic_epoch),
                "sample_order_sha256": order_hash,
                "gradient_microbatch_size": int(
                    args.gradient_microbatch_size
                ),
            }
        finally:
            if case_model is not None:
                case_model.zero_grad(set_to_none=True)
            del (
                raw_checkpoint,
                case_checkpoint,
                train_args,
                train_loader,
                img_loader,
                txt_loader,
                case_model,
            )
            release_cuda_objects()

    checkpoint_comparison = compare_checkpoint_parameters(parameters_by_case)
    metadata = {
        "git_head": _git_head(),
        "device": str(device),
        "degrade_seed": int(args.degrade_seed),
        "max_batches": int(args.max_batches),
        "gradient_batches": int(args.gradient_batches),
        "gradient_microbatch_size": int(args.gradient_microbatch_size),
        "run_root": str(run_root) if run_root is not None else None,
        "input_mode": input_mode,
        "data_root": str(data_root),
        "cases": metadata_cases,
        "statistics_dtype": str(torch.float32),
        "optimizer_step_executed": False,
        "checkpoint_modified": False,
    }
    diagnosis = build_diagnosis(
        metadata=metadata,
        parameter_rows=parameter_rows,
        checkpoint_comparison=checkpoint_comparison,
        gate_rows=gate_rows,
        target_rows=target_rows,
        gradient_rows=gradient_rows,
        fusion_checks=fusion_checks,
        sample_records=sample_records,
    )

    output_paths = {
        "checkpoint_parameter_stats": output_dir
        / "checkpoint_parameter_stats.csv",
        "scenario_gate_statistics": output_dir / "scenario_gate_statistics.csv",
        "scenario_target_statistics": output_dir
        / "scenario_target_statistics.csv",
        "sample_level_diagnostics": output_dir / "sample_level_diagnostics.pt",
        "gate_direction_statistics": output_dir
        / "gate_direction_statistics.csv",
        "gradient_diagnostics": output_dir / "gradient_diagnostics.csv",
        "fusion_path_check": output_dir / "fusion_path_check.json",
        "summary": output_dir / "summary.txt",
        "diagnosis": output_dir / "diagnosis.json",
    }
    _write_csv(
        output_paths["checkpoint_parameter_stats"],
        CHECKPOINT_PARAMETER_FIELDS,
        parameter_rows,
    )
    _write_csv(
        output_paths["scenario_gate_statistics"],
        SCENARIO_GATE_FIELDS,
        gate_rows,
    )
    _write_csv(
        output_paths["scenario_target_statistics"],
        SCENARIO_TARGET_FIELDS,
        target_rows,
    )
    _write_csv(
        output_paths["gate_direction_statistics"],
        GATE_DIRECTION_FIELDS,
        direction_rows,
    )
    _write_csv(
        output_paths["gradient_diagnostics"],
        GRADIENT_FIELDS,
        gradient_rows,
    )
    torch.save(
        {
            "metadata": metadata,
            "cases": sample_records,
        },
        output_paths["sample_level_diagnostics"],
    )
    _json_dump(
        output_paths["fusion_path_check"],
        {
            "metadata": metadata,
            "cases": fusion_checks,
        },
    )
    _json_dump(output_paths["diagnosis"], diagnosis)
    write_summary(output_paths["summary"], diagnosis)
    for name, path in output_paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
