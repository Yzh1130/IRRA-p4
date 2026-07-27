import inspect
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from model.mraf import ModalityReliabilityAdaptiveFusion
from processor.processor import (
    _mraf_initial_parameter_snapshot,
    _mraf_scorer_gradient_stats,
    _mraf_scorer_parameter_stats,
)
from utils.iotools import load_train_configs, save_train_configs
from utils.options import get_args, validate_mraf_options


def _inputs():
    torch.manual_seed(20260728)
    rgb = torch.randn(6, 12) * 1.45
    ir = torch.randn(6, 12)
    return rgb, ir


def _nonzero_scorer(module):
    torch.manual_seed(17)
    with torch.no_grad():
        module.output_layer.weight.normal_(mean=0.0, std=0.2)
        module.output_layer.bias.fill_(0.05)


def _legacy_forward(module, f_rgb, f_ir):
    """Exact pre-change forward arithmetic used for the default-path oracle."""
    scorer_dtype = module.output_layer.weight.dtype
    if module.gate_fp32:
        with torch.autocast(device_type=f_rgb.device.type, enabled=False):
            rgb_for_fusion = f_rgb.float()
            ir_for_fusion = f_ir.float()
            rgb_normalized = F.normalize(rgb_for_fusion, p=2, dim=-1)
            ir_normalized = F.normalize(ir_for_fusion, p=2, dim=-1)
            rgb_logit = module.scorer(rgb_normalized)
            ir_logit = module.scorer(ir_normalized)
            raw_logits = torch.cat([rgb_logit, ir_logit], dim=-1)
            scaled_logits = raw_logits * module.gate_logit_scale
            alpha = F.softmax(
                scaled_logits / module.gate_temperature, dim=-1
            )
            if module.gate_base_grad_scale == 1.0:
                alpha_for_base = alpha
            else:
                alpha_for_base = (
                    alpha.detach()
                    + module.gate_base_grad_scale
                    * (alpha - alpha.detach())
                )
            alpha_rgb = alpha_for_base[:, 0:1]
            alpha_ir = alpha_for_base[:, 1:2]
            alpha_aux = alpha
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
        rgb_logit = module.scorer(rgb_normalized)
        ir_logit = module.scorer(ir_normalized)
        raw_logits = torch.cat([rgb_logit, ir_logit], dim=-1)
        scaled_logits = raw_logits * module.gate_logit_scale
        alpha = F.softmax(
            scaled_logits / module.gate_temperature, dim=-1
        )
        if module.gate_base_grad_scale == 1.0:
            alpha_for_base = alpha
        else:
            alpha_for_base = (
                alpha.detach()
                + module.gate_base_grad_scale
                * (alpha - alpha.detach())
            )
        alpha_for_fusion = alpha_for_base.to(dtype=f_rgb.dtype)
        alpha_aux = alpha.to(dtype=f_rgb.dtype)
        alpha_rgb = alpha_for_fusion[:, 0:1]
        alpha_ir = alpha_for_fusion[:, 1:2]
        fused = 2.0 * (alpha_rgb * f_rgb + alpha_ir * f_ir)
    return fused, {
        "fused": fused,
        "alpha_rgb": alpha_aux[:, 0:1],
        "alpha_ir": alpha_aux[:, 1:2],
        "alpha": alpha_aux,
        "logits": raw_logits,
        "raw_logits": raw_logits,
        "scaled_logits": scaled_logits,
        "gate_logit_scale": module.gate_logit_scale,
    }


@pytest.mark.parametrize("gate_fp32", [False, True])
def test_default_forward_is_bit_exact_with_legacy_implementation(gate_fp32):
    rgb, ir = _inputs()
    module = ModalityReliabilityAdaptiveFusion(
        feature_dim=12,
        hidden_dim=7,
        dropout=0.0,
        gate_temperature=0.8,
        gate_logit_scale=2.5,
        gate_fp32=gate_fp32,
        gate_base_grad_scale=0.35,
        equalize_modality_norm=False,
        freeze_gate=False,
    ).eval()
    _nonzero_scorer(module)

    expected_fused, expected_aux = _legacy_forward(module, rgb, ir)
    actual_fused, actual_aux = module(rgb, ir, return_aux=True)

    torch.testing.assert_close(
        actual_fused, expected_fused, rtol=0.0, atol=0.0
    )
    assert actual_aux.keys() == expected_aux.keys()
    for key, expected in expected_aux.items():
        actual = actual_aux[key]
        if torch.is_tensor(expected):
            torch.testing.assert_close(
                actual, expected, rtol=0.0, atol=0.0
            )
        else:
            assert actual == expected


@pytest.mark.parametrize("gate_fp32", [False, True])
def test_equalized_modalities_have_the_per_sample_mean_norm(gate_fp32):
    rgb, ir = _inputs()
    working_rgb = rgb.float() if gate_fp32 else rgb
    working_ir = ir.float() if gate_fp32 else ir
    rgb_equalized, ir_equalized = (
        ModalityReliabilityAdaptiveFusion._equalize_feature_norms(
            working_rgb, working_ir
        )
    )
    expected = 0.5 * (
        working_rgb.norm(p=2, dim=-1)
        + working_ir.norm(p=2, dim=-1)
    )
    torch.testing.assert_close(
        rgb_equalized.norm(p=2, dim=-1), expected
    )
    torch.testing.assert_close(
        ir_equalized.norm(p=2, dim=-1), expected
    )


@pytest.mark.parametrize("gate_fp32", [False, True])
def test_equalized_half_gate_preserves_fused_feature_scale(gate_fp32):
    torch.manual_seed(31)
    direction = torch.randn(8, 16)
    rgb = 2.0 * direction
    ir = 0.5 * direction
    module = ModalityReliabilityAdaptiveFusion(
        feature_dim=16,
        hidden_dim=5,
        dropout=0.0,
        gate_fp32=gate_fp32,
        equalize_modality_norm=True,
        freeze_gate=True,
    ).eval()

    fused, aux = module(rgb, ir, return_aux=True)
    ratio = fused.norm(p=2, dim=-1) / (rgb + ir).norm(p=2, dim=-1)

    assert torch.all(aux["alpha"] == 0.5)
    assert bool(torch.all((ratio >= 0.7) & (ratio <= 1.4)).item())


@pytest.mark.parametrize("gate_fp32", [False, True])
def test_freeze_gate_is_half_and_scorer_receives_no_gradient(gate_fp32):
    rgb, ir = _inputs()
    rgb.requires_grad_(True)
    ir.requires_grad_(True)
    module = ModalityReliabilityAdaptiveFusion(
        feature_dim=12,
        hidden_dim=7,
        dropout=0.0,
        gate_fp32=gate_fp32,
        freeze_gate=True,
    )
    _nonzero_scorer(module)

    fused, aux = module(rgb, ir, return_aux=True)
    fused.square().mean().backward()

    assert torch.all(aux["alpha"] == 0.5)
    assert torch.count_nonzero(aux["raw_logits"]) == 0
    assert torch.count_nonzero(aux["scaled_logits"]) == 0
    assert rgb.grad is not None and bool(torch.isfinite(rgb.grad).all())
    assert ir.grad is not None and bool(torch.isfinite(ir.grad).all())
    assert all(
        parameter.grad is None
        or torch.count_nonzero(parameter.grad).item() == 0
        for parameter in module.scorer.parameters()
    )


def test_freeze_gate_return_aux_keys_match_dynamic_gate():
    rgb, ir = _inputs()
    dynamic = ModalityReliabilityAdaptiveFusion(
        feature_dim=12, hidden_dim=7, dropout=0.0
    ).eval()
    frozen = ModalityReliabilityAdaptiveFusion(
        feature_dim=12,
        hidden_dim=7,
        dropout=0.0,
        freeze_gate=True,
    ).eval()
    frozen.load_state_dict(dynamic.state_dict())

    _, dynamic_aux = dynamic(rgb, ir, return_aux=True)
    _, frozen_aux = frozen(rgb, ir, return_aux=True)

    assert frozen_aux.keys() == dynamic_aux.keys()


def test_processor_scorer_stats_tolerate_frozen_gate_grad_none():
    rgb, ir = _inputs()
    rgb.requires_grad_(True)
    ir.requires_grad_(True)
    module = ModalityReliabilityAdaptiveFusion(
        feature_dim=12,
        hidden_dim=7,
        dropout=0.0,
        freeze_gate=True,
    )
    wrapped = SimpleNamespace(mraf_fusion=module)
    initial_snapshot = _mraf_initial_parameter_snapshot(wrapped)

    fused = module(rgb, ir)
    fused.square().mean().backward()
    gradient_stats = _mraf_scorer_gradient_stats(wrapped)
    parameter_stats = _mraf_scorer_parameter_stats(
        wrapped, initial_snapshot
    )

    assert gradient_stats
    assert all(value == 0.0 for value in gradient_stats.values())
    assert parameter_stats["scorer_parameter_delta_from_initial"] == 0.0


@pytest.mark.parametrize("gate_fp32", [False, True])
def test_norm_equalization_and_frozen_gate_are_compatible(gate_fp32):
    rgb, ir = _inputs()
    module = ModalityReliabilityAdaptiveFusion(
        feature_dim=12,
        hidden_dim=7,
        dropout=0.0,
        gate_fp32=gate_fp32,
        equalize_modality_norm=True,
        freeze_gate=True,
    ).eval()

    fused, aux = module(rgb, ir, return_aux=True)
    working_rgb = rgb.float() if gate_fp32 else rgb
    working_ir = ir.float() if gate_fp32 else ir
    rgb_equalized, ir_equalized = module._equalize_feature_norms(
        working_rgb, working_ir
    )

    torch.testing.assert_close(fused, rgb_equalized + ir_equalized)
    assert torch.all(aux["alpha"] == 0.5)
    assert bool(torch.isfinite(fused).all())


@pytest.mark.parametrize("gate_fp32", [False, True])
def test_zero_alpha_std_full_forward_backward_is_finite(gate_fp32):
    rgb, ir = _inputs()
    rgb.requires_grad_(True)
    ir.requires_grad_(True)
    module = ModalityReliabilityAdaptiveFusion(
        feature_dim=12,
        hidden_dim=7,
        dropout=0.0,
        gate_fp32=gate_fp32,
        freeze_gate=True,
    )

    fused, aux = module(rgb, ir, return_aux=True)
    alpha_std = aux["alpha"].float().std(unbiased=False)
    loss = fused.float().square().mean() + alpha_std
    loss.backward()

    assert alpha_std.item() == 0.0
    assert bool(torch.isfinite(loss).item())
    assert bool(torch.isfinite(fused).all())
    assert bool(torch.isfinite(rgb.grad).all())
    assert bool(torch.isfinite(ir.grad).all())


def test_new_boolean_options_round_trip_through_configs_yaml(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mraf_enabled",
            "--mraf_equalize_modality_norm",
            "--mraf_freeze_gate",
        ],
    )
    training_args = get_args()
    save_train_configs(str(tmp_path), training_args)
    config_path = tmp_path / "configs.yaml"
    loaded = load_train_configs(str(config_path))

    assert loaded.mraf_equalize_modality_norm is True
    assert loaded.mraf_freeze_gate is True

    monkeypatch.setattr(
        sys, "argv", ["test.py", "--config_file", str(config_path), "--test"]
    )
    evaluation_args = get_args()
    assert evaluation_args.mraf_equalize_modality_norm is True
    assert evaluation_args.mraf_freeze_gate is True


def test_new_options_default_false_and_are_ignored_when_mraf_disabled(
    monkeypatch,
):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = get_args()
    assert args.mraf_equalize_modality_norm is False
    assert args.mraf_freeze_gate is False

    disabled = SimpleNamespace(
        mraf_enabled=False,
        mraf_equalize_modality_norm=True,
        mraf_freeze_gate=True,
        use_global_uncertainty=False,
        use_precision_fusion=False,
    )
    validate_mraf_options(disabled)


def test_model_builder_wires_both_new_flags():
    from model import build

    source = inspect.getsource(build.IRRA.__init__)
    assert 'getattr(args, "mraf_equalize_modality_norm", False)' in source
    assert 'getattr(args, "mraf_freeze_gate", False)' in source
