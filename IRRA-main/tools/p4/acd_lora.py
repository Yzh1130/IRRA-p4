"""Explicitly routed FP32 q/v LoRA for the frozen IRRA visual transformer.

Disabled routing calls the original bound forward without tensor operations.
Enabled routing forms the effective q/v projection weight in FP32, then casts
that weight to the original attention dtype. The visual tower otherwise keeps
its original precision, so zero B reproduces the original FP16 computation.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import types

import torch
from torch import nn
from torch.nn import functional as F


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class QVLoRA(nn.Module):
    def __init__(self, width, rank, alpha, generator, device):
        super().__init__()
        self.rank, self.alpha = int(rank), float(alpha)
        for projection in ("q", "v"):
            a = torch.empty(rank, width, dtype=torch.float32)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
            self.register_parameter(projection + "_A", nn.Parameter(a.to(device)))
            self.register_parameter(projection + "_B", nn.Parameter(
                torch.zeros(width, rank, dtype=torch.float32, device=device)))

    def delta(self):
        scale = self.alpha / self.rank
        q = (self.q_B @ self.q_A) * scale
        v = (self.v_B @ self.v_A) * scale
        return torch.cat((q, torch.zeros_like(q), v), dim=0)


class LoRAController:
    def __init__(self, model, rank=8, alpha=16, init_seed=20261015,
                 expected_layers=12):
        if rank <= 0 or not math.isfinite(float(alpha)) or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        visual = model.base_model.visual
        blocks = visual.transformer.resblocks
        if len(blocks) != expected_layers:
            raise ValueError(f"Expected {expected_layers} visual blocks, got {len(blocks)}")
        if hasattr(visual, "_acd_controller"):
            raise ValueError("ACD LoRA has already been injected")
        self.visual = visual
        self.active = False
        self.layers = []
        self.config = dict(rank=int(rank), alpha=float(alpha),
                           init_seed=int(init_seed), layers=len(blocks),
                           projections=["q", "v"], dropout=0.0,
                           student_precision="native_visual_fp32_lora_weight_delta")
        generator = torch.Generator(device="cpu").manual_seed(init_seed)
        for block in blocks:
            attention = block.attn
            if not isinstance(attention, nn.MultiheadAttention) or not attention._qkv_same_embed_dim:
                raise ValueError("ACD requires ordinary same-width nn.MultiheadAttention")
            adapter = QVLoRA(attention.embed_dim, rank, alpha, generator,
                             attention.in_proj_weight.device)
            attention.add_module("acd_lora", adapter)
            original = attention.forward
            controller = self

            def routed(module, query, key, value, key_padding_mask=None,
                       need_weights=True, attn_mask=None,
                       average_attn_weights=True, is_causal=False,
                       _original=original):
                if not controller.active:
                    return _original(query, key, value,
                                     key_padding_mask=key_padding_mask,
                                     need_weights=need_weights, attn_mask=attn_mask,
                                     average_attn_weights=average_attn_weights,
                                     is_causal=is_causal)
                if module.batch_first and query.ndim == 3:
                    query, key, value = (x.transpose(0, 1) for x in (query, key, value))
                effective_weight = (module.in_proj_weight.float()
                                    + module.acd_lora.delta()).to(module.in_proj_weight.dtype)
                output, weights = F.multi_head_attention_forward(
                    query, key, value, module.embed_dim,
                    module.num_heads, effective_weight,
                    module.in_proj_bias, module.bias_k, module.bias_v,
                    module.add_zero_attn, module.dropout, module.out_proj.weight,
                    module.out_proj.bias, training=False,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, average_attn_weights=average_attn_weights,
                    is_causal=is_causal)
                if module.batch_first and query.ndim == 3:
                    output = output.transpose(0, 1)
                return output, weights

            attention.forward = types.MethodType(routed, attention)
            self.layers.append(adapter)
        # Do not register the controller as a module or serialize base weights.
        object.__setattr__(visual, "_acd_controller", self)

    @contextmanager
    def enabled(self, enabled=True):
        previous = self.active
        self.active = bool(enabled)
        try:
            yield self
        finally:
            self.active = previous

    def parameters(self):
        return (parameter for layer in self.layers for parameter in layer.parameters())

    def state_dict(self):
        return {f"blocks.{index}.{name}": value.detach().cpu().clone()
                for index, layer in enumerate(self.layers)
                for name, value in layer.state_dict().items()}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if set(state) != set(expected):
            raise ValueError("Adapter parameter keys do not match injected q/v LoRA")
        for name, value in state.items():
            if value.dtype != torch.float32 or value.shape != expected[name].shape:
                raise ValueError(f"Adapter shape/dtype mismatch: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite adapter parameter: {name}")
        for index, layer in enumerate(self.layers):
            prefix = f"blocks.{index}."
            layer.load_state_dict({name[len(prefix):]: value for name, value in state.items()
                                   if name.startswith(prefix)}, strict=True)


def inject_lora(model, rank=8, alpha=16, init_seed=20261015, expected_layers=12):
    return LoRAController(model, rank, alpha, init_seed, expected_layers)


def save_adapter(controller, path, base_sha256, extra_config=None):
    config = dict(controller.config, base_sha256=str(base_sha256))
    if extra_config:
        collision = set(config) & set(extra_config)
        if collision:
            raise ValueError(f"Cannot overwrite adapter structural configuration: {collision}")
        config.update(extra_config)
    torch.save({"schema_version": 1, "config": config,
                "lora_state_dict": controller.state_dict()}, path)


def read_adapter(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported ACD adapter format")
    if not isinstance(payload.get("config"), dict) or "base_sha256" not in payload["config"]:
        raise ValueError("ACD adapter lacks base checkpoint SHA256")
    if not isinstance(payload.get("lora_state_dict"), dict):
        raise ValueError("ACD adapter lacks LoRA parameter dictionary")
    return payload


def load_adapter(controller, path, expected_base_sha256):
    payload = read_adapter(path)
    config = payload["config"]
    if config["base_sha256"] != expected_base_sha256:
        raise ValueError("ACD adapter base checkpoint SHA256 mismatch")
    for key in ("rank", "alpha", "layers", "projections", "dropout", "student_precision"):
        if config.get(key) != controller.config[key]:
            raise ValueError(f"ACD adapter structural configuration mismatch: {key}")
    controller.load_state_dict(payload["lora_state_dict"])
    return payload
