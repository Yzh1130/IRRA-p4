"""Modality Reliability-Adaptive Fusion (MRAF)."""

from __future__ import annotations

import math
import logging
import os
from contextvars import ContextVar
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrTuple = Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]


_MRAF_MEM_DISABLED = object()
_MRAF_MEM_SCOPE: ContextVar[object] = ContextVar(
    "mraf_mem_scope", default=None
)


def mraf_mem_debug_start_iteration(epoch: int, iteration: int, enabled: bool) -> None:
    """Enable snapshots only for the first two requested CUDA iterations."""
    if (
        enabled
        and int(iteration) <= 2
        and os.environ.get("MRAF_MEM_DEBUG") == "1"
    ):
        _MRAF_MEM_SCOPE.set(
            {"epoch": int(epoch), "iteration": int(iteration), "call_count": 0}
        )
    else:
        # Distinguish a later training iteration from a standalone eval call.
        _MRAF_MEM_SCOPE.set(_MRAF_MEM_DISABLED)


def mraf_mem_debug_clear_iteration() -> None:
    _MRAF_MEM_SCOPE.set(None)


def mraf_mem_debug_active(device=None) -> bool:
    """Return whether the active iteration permits CUDA memory logging."""
    if os.environ.get("MRAF_MEM_DEBUG") != "1":
        return False
    if _MRAF_MEM_SCOPE.get() is _MRAF_MEM_DISABLED:
        return False
    if not torch.cuda.is_available():
        return False
    if device is None:
        return True
    return torch.device(device).type == "cuda"


def mraf_mem_debug_call_count() -> int:
    scope = _MRAF_MEM_SCOPE.get()
    return scope["call_count"] if isinstance(scope, dict) else 0


def _mraf_mem_debug_next_call(module: nn.Module) -> int:
    scope = _MRAF_MEM_SCOPE.get()
    if scope is None:
        call_count = getattr(module, "_mem_debug_standalone_calls", 0) + 1
        module._mem_debug_standalone_calls = call_count
        return call_count
    scope["call_count"] += 1
    return scope["call_count"]


def mraf_mem_debug_log(
    logger,
    stage: str,
    device,
    include_reserved: bool = False,
    include_peak: bool = True,
    **fields,
) -> None:
    """Log allocator counters without retaining tensors or changing the graph."""
    if not mraf_mem_debug_active(device):
        return
    scope = _MRAF_MEM_SCOPE.get()
    snapshot = {
        "stage": stage,
        "epoch": scope["epoch"] if isinstance(scope, dict) else "standalone",
        "iteration": scope["iteration"] if isinstance(scope, dict) else "-",
        **fields,
        "allocated_bytes": torch.cuda.memory_allocated(device),
    }
    if include_reserved:
        snapshot["reserved_bytes"] = torch.cuda.memory_reserved(device)
    if include_peak:
        snapshot["max_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    logger.info("[MRAF_MEM] %s", " ".join(f"{key}={value}" for key, value in snapshot.items()))


def _validate_feature_tensor(name: str, feature: torch.Tensor) -> None:
    if not torch.is_tensor(feature):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if feature.dim() != 2:
        raise ValueError(
            f"{name} must have shape [B, D], got {tuple(feature.shape)}."
        )
    if not feature.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {feature.dtype}.")
    if not bool(torch.isfinite(feature).all().item()):
        raise ValueError(f"{name} contains NaN or Inf values.")


def _validate_feature_pair(
    first_name: str,
    first: torch.Tensor,
    second_name: str,
    second: torch.Tensor,
) -> None:
    _validate_feature_tensor(first_name, first)
    _validate_feature_tensor(second_name, second)
    if first.shape[0] != second.shape[0]:
        raise ValueError(
            f"{first_name} and {second_name} must share batch size, got "
            f"{first.shape[0]} and {second.shape[0]}."
        )
    if first.shape[1] != second.shape[1]:
        raise ValueError(
            f"{first_name} and {second_name} must share feature dimension, got "
            f"{first.shape[1]} and {second.shape[1]}."
        )
    if first.device != second.device:
        raise ValueError(
            f"{first_name} and {second_name} must be on the same device, got "
            f"{first.device} and {second.device}."
        )
    if first.dtype != second.dtype:
        raise ValueError(
            f"{first_name} and {second_name} must share dtype, got "
            f"{first.dtype} and {second.dtype}."
        )


class ModalityReliabilityAdaptiveFusion(nn.Module):
    """Fuse two global modality features using a shared reliability scorer."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        gate_temperature: float = 1.0,
        gate_logit_scale: float = 1.0,
        gate_fp32: bool = False,
        gate_base_grad_scale: float = 1.0,
        equalize_modality_norm: bool = False,
        freeze_gate: bool = False,
        rel_isolate_input: bool = False,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0:
            raise ValueError(f"feature_dim must be > 0, got {feature_dim}.")
        if int(hidden_dim) <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if float(gate_temperature) <= 0.0:
            raise ValueError(
                f"gate_temperature must be > 0, got {gate_temperature}."
            )
        if (
            not math.isfinite(float(gate_logit_scale))
            or float(gate_logit_scale) <= 0.0
        ):
            raise ValueError(
                f"gate_logit_scale must be > 0, got {gate_logit_scale}."
            )
        if not 0.0 <= float(gate_base_grad_scale) <= 1.0:
            raise ValueError(
                "gate_base_grad_scale must be in [0, 1], got "
                f"{gate_base_grad_scale}."
            )

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.gate_temperature = float(gate_temperature)
        self.gate_logit_scale = float(gate_logit_scale)
        self.gate_fp32 = bool(gate_fp32)
        self.gate_base_grad_scale = float(gate_base_grad_scale)
        self.equalize_modality_norm = bool(equalize_modality_norm)
        self.freeze_gate = bool(freeze_gate)
        self.rel_isolate_input = bool(rel_isolate_input)
        self.scorer = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self._zero_initialize_output_layer()
        if self.gate_fp32:
            self.scorer.float()

    @property
    def output_layer(self) -> nn.Linear:
        """Return the zero-initialized final scorer layer."""
        return self.scorer[-1]

    def _zero_initialize_output_layer(self) -> None:
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    @staticmethod
    def _equalize_feature_norms(
        f_rgb: torch.Tensor,
        f_ir: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale both modalities to their per-sample arithmetic mean norm."""
        n_rgb = f_rgb.norm(p=2, dim=-1, keepdim=True).clamp_min(float(eps))
        n_ir = f_ir.norm(p=2, dim=-1, keepdim=True).clamp_min(float(eps))
        n_bar = 0.5 * (n_rgb + n_ir)
        return f_rgb * (n_bar / n_rgb), f_ir * (n_bar / n_ir)

    def _apply(self, fn):
        """Keep the optional scorer in FP32 across model.half()/to()/conversion."""
        super()._apply(fn)
        if self.gate_fp32:
            self.scorer.float()
        return self

    def _recompute_isolated_alpha(
        self,
        rgb_normalized: torch.Tensor,
        ir_normalized: torch.Tensor,
        dropout_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Reuse each main-branch Dropout mask without its encoder input graph."""
        if len(dropout_outputs) != 2:
            raise RuntimeError(
                "MRAF isolation expected separate RGB and IR Dropout outputs, "
                f"got {len(dropout_outputs)}."
            )

        def isolated_logit(
            normalized: torch.Tensor,
            dropped: torch.Tensor,
        ) -> torch.Tensor:
            hidden = self.scorer[1](self.scorer[0](normalized.detach()))
            if self.training and self.dropout > 0.0:
                mask = (dropped != 0).to(dtype=dropped.dtype)
                # nn.Dropout scales the mask first, then multiplies hidden;
                # preserve that float32 rounding order for bitwise equality.
                hidden = hidden * (mask / (1.0 - self.dropout))
            # In eval mode or with p=0, the effective mask is all ones.
            return self.scorer[3](hidden)

        raw_logits = torch.cat(
            [
                isolated_logit(rgb_normalized, dropout_outputs[0]),
                isolated_logit(ir_normalized, dropout_outputs[1]),
            ],
            dim=-1,
        )
        scaled_logits = raw_logits * self.gate_logit_scale
        return F.softmax(scaled_logits / self.gate_temperature, dim=-1)

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_ir: torch.Tensor,
        return_aux: bool = False,
        mem_debug_entry: str = "direct",
    ) -> TensorOrTuple:
        """Return baseline-preserving reliability-weighted RGB/IR fusion."""
        _validate_feature_pair("f_rgb", f_rgb, "f_ir", f_ir)
        if f_rgb.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, got "
                f"{f_rgb.shape[1]}."
            )

        scorer_device = self.output_layer.weight.device
        if f_rgb.device != scorer_device:
            raise ValueError(
                "MRAF inputs and scorer must be on the same device, got "
                f"{f_rgb.device} and {scorer_device}."
            )

        mem_debug_on = mraf_mem_debug_active(f_rgb.device)
        mem_call_seq = _mraf_mem_debug_next_call(self) if mem_debug_on else 0
        mem_logger = logging.getLogger("IRRA.train") if mem_debug_on else None
        scorer_dtype = self.output_layer.weight.dtype
        dropout_outputs: list[torch.Tensor] = []
        dropout_hook = None
        if self.rel_isolate_input and not self.freeze_gate:
            # The original scorer/nn.Dropout calls below remain untouched.
            # Capture their two separate modality masks without drawing RNG.
            if mem_debug_on:
                def capture_dropout(_module, _inputs, output):
                    dropout_outputs.append(output)
                    mraf_mem_debug_log(
                        mem_logger,
                        "dropout_hook",
                        f_rgb.device,
                        call_seq=mem_call_seq,
                        entry=mem_debug_entry,
                        trigger_count=len(dropout_outputs),
                        output_shape=tuple(output.shape),
                    )

                dropout_hook = self.scorer[2].register_forward_hook(capture_dropout)
            else:
                dropout_hook = self.scorer[2].register_forward_hook(
                    lambda _module, _inputs, output: dropout_outputs.append(output)
                )
        if mem_debug_on:
            mraf_mem_debug_log(
                mem_logger,
                "forward_entry",
                f_rgb.device,
                call_seq=mem_call_seq,
                entry=mem_debug_entry,
                f_rgb_shape=tuple(f_rgb.shape),
                f_ir_shape=tuple(f_ir.shape),
                f_rgb_dtype=f_rgb.dtype,
                f_ir_dtype=f_ir.dtype,
                training=self.training,
                hook_registered=dropout_hook is not None,
            )
        if self.gate_fp32:
            # Disable the surrounding AMP context for the complete gate and
            # fusion path.  Merely storing FP32 parameters is insufficient:
            # autocast would otherwise quantize Linear outputs on supported
            # devices before softmax.
            with torch.autocast(device_type=f_rgb.device.type, enabled=False):
                rgb_for_fusion = f_rgb.float()
                ir_for_fusion = f_ir.float()
                rgb_normalized = F.normalize(rgb_for_fusion, p=2, dim=-1)
                ir_normalized = F.normalize(ir_for_fusion, p=2, dim=-1)
                if self.freeze_gate:
                    raw_logits = rgb_normalized.new_zeros(
                        (f_rgb.shape[0], 2)
                    )
                    scaled_logits = raw_logits.clone()
                    alpha = rgb_normalized.new_full(
                        (f_rgb.shape[0], 2), 0.5
                    )
                else:
                    rgb_logit = self.scorer(rgb_normalized)
                    ir_logit = self.scorer(ir_normalized)
                    raw_logits = torch.cat([rgb_logit, ir_logit], dim=-1)
                    scaled_logits = raw_logits * self.gate_logit_scale
                    alpha = F.softmax(
                        scaled_logits / self.gate_temperature, dim=-1
                    )
                if self.gate_base_grad_scale == 1.0:
                    alpha_for_base = alpha
                else:
                    alpha_for_base = (
                        alpha.detach()
                        + self.gate_base_grad_scale * (alpha - alpha.detach())
                    )
                alpha_rgb = alpha_for_base[:, 0:1]
                alpha_ir = alpha_for_base[:, 1:2]
                alpha_aux = alpha
                if self.equalize_modality_norm:
                    rgb_for_fusion, ir_for_fusion = (
                        self._equalize_feature_norms(
                            rgb_for_fusion, ir_for_fusion
                        )
                    )
                fused = 2.0 * (
                    alpha_rgb * rgb_for_fusion
                    + alpha_ir * ir_for_fusion
                )
        else:
            rgb_normalized = F.normalize(f_rgb, p=2, dim=-1).to(
                dtype=scorer_dtype
            )
            ir_normalized = F.normalize(f_ir, p=2, dim=-1).to(
                dtype=scorer_dtype
            )
            if self.freeze_gate:
                raw_logits = rgb_normalized.new_zeros((f_rgb.shape[0], 2))
                scaled_logits = raw_logits.clone()
                alpha = rgb_normalized.new_full((f_rgb.shape[0], 2), 0.5)
            else:
                rgb_logit = self.scorer(rgb_normalized)
                ir_logit = self.scorer(ir_normalized)
                raw_logits = torch.cat([rgb_logit, ir_logit], dim=-1)
                scaled_logits = raw_logits * self.gate_logit_scale
                alpha = F.softmax(
                    scaled_logits / self.gate_temperature, dim=-1
                )
            if self.gate_base_grad_scale == 1.0:
                alpha_for_base = alpha
            else:
                alpha_for_base = alpha.detach() + self.gate_base_grad_scale * (
                    alpha - alpha.detach()
                )
            alpha_for_fusion = alpha_for_base.to(dtype=f_rgb.dtype)
            # Preserve the legacy auxiliary dtype while keeping reliability
            # supervision on the unscaled alpha graph.
            alpha_aux = alpha.to(dtype=f_rgb.dtype)
            alpha_rgb = alpha_for_fusion[:, 0:1]
            alpha_ir = alpha_for_fusion[:, 1:2]
            if self.equalize_modality_norm:
                rgb_for_fusion, ir_for_fusion = (
                    self._equalize_feature_norms(f_rgb, f_ir)
                )
                fused = 2.0 * (
                    alpha_rgb * rgb_for_fusion
                    + alpha_ir * ir_for_fusion
                )
            else:
                fused = 2.0 * (alpha_rgb * f_rgb + alpha_ir * f_ir)
        if dropout_hook is not None:
            dropout_hook.remove()
            alpha_main_aux = alpha_aux
            if mem_debug_on:
                mraf_mem_debug_log(
                    mem_logger, "isolated_alpha_before", f_rgb.device,
                    call_seq=mem_call_seq, entry=mem_debug_entry,
                )
            if self.gate_fp32:
                with torch.autocast(device_type=f_rgb.device.type, enabled=False):
                    alpha_aux = self._recompute_isolated_alpha(
                        rgb_normalized, ir_normalized, dropout_outputs
                    )
            else:
                alpha_aux = self._recompute_isolated_alpha(
                    rgb_normalized, ir_normalized, dropout_outputs
                ).to(dtype=f_rgb.dtype)
            if mem_debug_on:
                mraf_mem_debug_log(
                    mem_logger, "isolated_alpha_after", f_rgb.device,
                    call_seq=mem_call_seq, entry=mem_debug_entry,
                )
        else:
            alpha_main_aux = alpha_aux
        if not bool(torch.isfinite(alpha).all().item()):
            raise ValueError("MRAF reliability weights contain NaN or Inf values.")
        if not bool(torch.isfinite(fused).all().item()):
            raise ValueError("MRAF fused features contain NaN or Inf values.")

        if not return_aux:
            if mem_debug_on:
                mraf_mem_debug_log(
                    mem_logger, "forward_exit", f_rgb.device,
                    call_seq=mem_call_seq, entry=mem_debug_entry,
                    f_rgb_shape=tuple(f_rgb.shape), f_ir_shape=tuple(f_ir.shape),
                    f_rgb_dtype=f_rgb.dtype, f_ir_dtype=f_ir.dtype,
                    training=self.training, hook_registered=dropout_hook is not None,
                    hook_trigger_count=len(dropout_outputs), return_aux=False,
                )
            return fused
        aux = {
            "fused": fused,
            # The RGB/IR aliases also weight fused variance in model/build.py;
            # only the reliability-supervision "alpha" uses the isolated graph.
            "alpha_rgb": alpha_main_aux[:, 0:1],
            "alpha_ir": alpha_main_aux[:, 1:2],
            "alpha": alpha_aux,
            # Keep the legacy alias on raw logits; new code should use the
            # explicit raw/scaled names.
            "logits": raw_logits,
            "raw_logits": raw_logits,
            "scaled_logits": scaled_logits,
            "gate_logit_scale": self.gate_logit_scale,
        }
        if self.rel_isolate_input:
            aux["alpha_main"] = alpha_main_aux
            aux["alpha_aux"] = alpha_aux
        if mem_debug_on:
            mraf_mem_debug_log(
                mem_logger, "forward_exit", f_rgb.device,
                call_seq=mem_call_seq, entry=mem_debug_entry,
                f_rgb_shape=tuple(f_rgb.shape), f_ir_shape=tuple(f_ir.shape),
                f_rgb_dtype=f_rgb.dtype, f_ir_dtype=f_ir.dtype,
                training=self.training, hook_registered=dropout_hook is not None,
                hook_trigger_count=len(dropout_outputs), return_aux=True,
            )
        return fused, aux


def compute_mraf_reliability_targets(
    clean_rgb: torch.Tensor,
    clean_ir: torch.Tensor,
    degraded_rgb: torch.Tensor,
    degraded_ir: torch.Tensor,
    target_temperature: float = 0.1,
    eps: float = 1e-8,
    utility_prior: Optional[Sequence[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Build detached reliability targets from clean/degraded feature drift."""
    if float(target_temperature) <= 0.0:
        raise ValueError(
            f"target_temperature must be > 0, got {target_temperature}."
        )
    _validate_feature_pair("clean_rgb", clean_rgb, "clean_ir", clean_ir)
    _validate_feature_pair(
        "degraded_rgb", degraded_rgb, "degraded_ir", degraded_ir
    )
    _validate_feature_pair("clean_rgb", clean_rgb, "degraded_rgb", degraded_rgb)
    _validate_feature_pair("clean_ir", clean_ir, "degraded_ir", degraded_ir)

    drift_rgb = 1.0 - F.cosine_similarity(
        degraded_rgb.float(),
        clean_rgb.detach().float(),
        dim=-1,
        eps=float(eps),
    )
    drift_ir = 1.0 - F.cosine_similarity(
        degraded_ir.float(),
        clean_ir.detach().float(),
        dim=-1,
        eps=float(eps),
    )
    target_logits = torch.stack([-drift_rgb, -drift_ir], dim=-1)
    effective_utility_prior = target_logits.new_ones(2)
    if utility_prior is not None:
        try:
            effective_utility_prior = torch.as_tensor(utility_prior).detach().to(
                device=target_logits.device,
                dtype=target_logits.dtype,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "utility_prior must be a length-2 sequence of positive values."
            ) from exc
        if effective_utility_prior.dim() != 1 or effective_utility_prior.numel() != 2:
            raise ValueError(
                "utility_prior must be a length-2 sequence ordered as "
                "[u_rgb, u_ir]."
            )
        if not bool(
            (
                torch.isfinite(effective_utility_prior)
                & (effective_utility_prior > 0.0)
            ).all().item()
        ):
            raise ValueError("utility_prior values must be finite and > 0.")

    scaled_target_logits = target_logits / float(target_temperature)
    if not bool(
        torch.equal(
            effective_utility_prior,
            torch.ones_like(effective_utility_prior),
        )
    ):
        # Add log(u) after temperature division. Adding it to target_logits
        # before division would incorrectly produce (-d + log(u)) / T_r.
        scaled_target_logits = (
            scaled_target_logits + effective_utility_prior.log()
        )
    target = F.softmax(scaled_target_logits, dim=-1).detach()
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("MRAF reliability targets contain NaN or Inf values.")
    return {
        "target": target,
        "target_rgb": target[:, 0:1],
        "target_ir": target[:, 1:2],
        "drift_rgb": drift_rgb,
        "drift_ir": drift_ir,
        "utility_prior": effective_utility_prior,
    }


def compute_mraf_reliability_loss(
    alpha_degraded: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return MSE supervision for attached degraded-branch gate weights."""
    _validate_feature_pair(
        "alpha_degraded", alpha_degraded, "target", target
    )
    if alpha_degraded.shape[1] != 2:
        raise ValueError(
            "alpha_degraded and target must have shape [B, 2], got "
            f"{tuple(alpha_degraded.shape)}."
        )
    if target.requires_grad:
        raise ValueError("MRAF reliability target must be detached.")
    return F.mse_loss(alpha_degraded.float(), target.float())


def resolve_mraf_rel_lambda(
    base_lambda: float,
    warmup_epoch: int,
    zero_based_epoch: int,
) -> float:
    """Resolve the requested independent linear MRAF reliability warmup."""
    base_lambda = float(base_lambda)
    warmup_epoch = int(warmup_epoch)
    zero_based_epoch = max(0, int(zero_based_epoch))
    if base_lambda < 0.0:
        raise ValueError(f"base_lambda must be >= 0, got {base_lambda}.")
    if warmup_epoch < 0:
        raise ValueError(f"warmup_epoch must be >= 0, got {warmup_epoch}.")
    if base_lambda == 0.0:
        return 0.0
    if warmup_epoch > 0 and zero_based_epoch < warmup_epoch:
        return base_lambda * float(zero_based_epoch + 1) / float(warmup_epoch)
    return base_lambda
