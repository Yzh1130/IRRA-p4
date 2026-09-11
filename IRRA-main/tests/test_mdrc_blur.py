import sys
from types import SimpleNamespace

import pytest
import torch

import processor.processor as processor
from tools.protocol.gen_configs import (
    build_configs,
    generate_supplementary_configs,
)
from utils.blur_ops import blur_tensor
from utils.metrics import Evaluator
import utils.mdrc as mdrc
import utils.options as options


DEFAULT_MODES = (
    "rgb_drop",
    "ir_drop",
    "rgb_noise",
    "ir_noise",
    "rgb_attenuate",
    "ir_attenuate",
)
DBLUR_MODES = [
    "rgb_drop",
    "ir_drop",
    "rgb_blur",
    "ir_blur",
    "rgb_attenuate",
    "ir_attenuate",
]
TENSOR_SHAPE = (4, 3, 32, 16)


def _random_tensor(dtype):
    generator = torch.Generator().manual_seed(20260911)
    if dtype == torch.int32:
        return torch.randint(
            0,
            256,
            TENSOR_SHAPE,
            generator=generator,
            dtype=dtype,
        )
    return torch.randn(
        TENSOR_SHAPE,
        generator=generator,
        dtype=torch.float32,
    ).to(dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
def test_apply_blur_preserves_mask_semantics_shape_and_dtype(dtype):
    tensor = _random_tensor(dtype)
    blurred = blur_tensor(tensor, 7)
    all_on = torch.ones(TENSOR_SHAPE[0], dtype=torch.bool)
    all_off = torch.zeros(TENSOR_SHAPE[0], dtype=torch.bool)
    mixed = torch.tensor([True, False, True, False])

    output_all = mdrc._apply_blur(tensor, all_on, kernel_size=7)
    output_none = mdrc._apply_blur(tensor, all_off, kernel_size=7)
    output_mixed = mdrc._apply_blur(tensor, mixed, kernel_size=7)

    assert output_all.shape == tensor.shape
    assert output_all.dtype == tensor.dtype
    assert torch.equal(output_all, blurred)
    assert torch.equal(output_none, tensor)
    assert torch.equal(output_mixed[mixed], blurred[mixed])
    assert torch.equal(output_mixed[~mixed], tensor[~mixed])


def test_training_blur_matches_evaluation_blur_bitwise_for_float32():
    tensor = _random_tensor(torch.float32)
    evaluation_blur = Evaluator._blur_eval_image(None, tensor, 7)
    assert torch.equal(blur_tensor(tensor, 7), evaluation_blur)
    assert torch.equal(
        mdrc._apply_blur(
            tensor,
            torch.ones(TENSOR_SHAPE[0], dtype=torch.bool),
            kernel_size=7,
        ),
        evaluation_blur,
    )


def test_apply_blur_preserves_unselected_large_integers_exactly():
    tensor = torch.tensor(
        [
            [[[1, 2, 3, 4]]],
            [[[16777217, 16777219, 2147483000, -16777217]]],
            [[[5, 6, 7, 8]]],
            [[[-2147483000, 16777217, 16777219, -16777217]]],
        ],
        dtype=torch.int32,
    )
    sample_mask = torch.tensor([True, False, True, False])

    output = mdrc._apply_blur(tensor, sample_mask, kernel_size=7)

    assert torch.equal(output[~sample_mask], tensor[~sample_mask])


@pytest.mark.parametrize(
    ("mode", "changed_modality"),
    [("rgb_blur", "rgb"), ("ir_blur", "ir")],
)
def test_degrade_rgbt_batch_applies_only_selected_blur_mode(
    mode,
    changed_modality,
):
    rgb = _random_tensor(torch.float32)
    thermal = _random_tensor(torch.float32).flip(-1)
    original_rgb = rgb.clone()
    original_thermal = thermal.clone()

    torch.manual_seed(20260911)
    degraded_rgb, degraded_thermal, info = mdrc.degrade_rgbt_batch(
        rgb,
        thermal,
        prob=1.0,
        modes=[mode],
        return_mode=True,
    )

    assert info["valid_modes"] == (mode,)
    assert bool(info["active_mask"].all().item())
    assert info["mode_counts"] == {mode: TENSOR_SHAPE[0], "none": 0}
    if changed_modality == "rgb":
        assert torch.equal(degraded_rgb, blur_tensor(rgb, 7))
        assert torch.equal(degraded_thermal, thermal)
    else:
        assert torch.equal(degraded_rgb, rgb)
        assert torch.equal(degraded_thermal, blur_tensor(thermal, 7))
    assert torch.equal(rgb, original_rgb)
    assert torch.equal(thermal, original_thermal)


def test_default_and_allowed_modes_are_shared_with_options(monkeypatch):
    assert mdrc.DEFAULT_DEGRADE_MODES == DEFAULT_MODES
    assert mdrc.ALLOWED_DEGRADE_MODES == DEFAULT_MODES + (
        "rgb_blur",
        "ir_blur",
    )
    assert mdrc._normalize_modes(None) == list(DEFAULT_MODES)
    assert options.DEFAULT_DEGRADE_MODES is mdrc.DEFAULT_DEGRADE_MODES
    assert options.ALLOWED_DEGRADE_MODES is mdrc.ALLOWED_DEGRADE_MODES

    monkeypatch.setattr(sys, "argv", ["test_mdrc_blur"])
    assert options.get_args().mdrc_degrade_modes == list(DEFAULT_MODES)

    monkeypatch.setattr(
        sys,
        "argv",
        ["test_mdrc_blur", "--mdrc_degrade_modes", *DBLUR_MODES],
    )
    assert options.get_args().mdrc_degrade_modes == DBLUR_MODES


def test_mdrc_style_augmentation_samples_only_original_six(monkeypatch):
    seen = {"choice_calls": []}

    def fake_choice(values):
        seen["choice_calls"].append(tuple(values))
        return values[0]

    def fake_degrade(rgb, thermal, prob=0.5, modes=None, return_mode=False):
        seen["prob"] = prob
        seen["modes"] = tuple(modes)
        return rgb.clone(), thermal.clone()

    monkeypatch.setattr(processor.random, "random", lambda: 0.0)
    monkeypatch.setattr(processor.random, "choice", fake_choice)
    monkeypatch.setattr(processor, "degrade_rgbt_batch", fake_degrade)
    batch = {
        "images": torch.randn(2, 3, 4, 4),
        "thermal_images": torch.randn(2, 3, 4, 4),
    }

    applied, mode = processor._apply_degrade_aug_to_batch(
        SimpleNamespace(degrade_aug_mode="mdrc_style"),
        batch,
    )

    assert (applied, mode) == (True, "rgb_drop")
    assert seen["choice_calls"] == [DEFAULT_MODES]
    assert seen["prob"] == 1.0
    assert seen["modes"] == ("rgb_drop",)
    assert processor.DEFAULT_DEGRADE_MODES is mdrc.DEFAULT_DEGRADE_MODES


def test_dblur_config_equals_d_except_for_degradation_modes():
    base_config = {"mdrc_degrade_modes": list(DEFAULT_MODES)}
    e_temperature = 0.2
    utility_prior = [0.7284522740054223, 0.2715477259945776]

    formal_d = build_configs(
        base_config,
        e_temperature,
        utility_prior,
    )["D"]
    supplementary = generate_supplementary_configs(
        base_config,
        e_temperature,
        utility_prior,
    )
    assert tuple(supplementary) == ("Dblur",)
    dblur = supplementary["Dblur"]
    assert formal_d["mdrc_degrade_modes"] == list(DEFAULT_MODES)
    assert dblur["mdrc_degrade_modes"] == DBLUR_MODES
    assert set(formal_d) == set(dblur)
    differing_keys = {
        key for key in formal_d if formal_d[key] != dblur[key]
    }
    assert differing_keys == {"mdrc_degrade_modes"}

    formal_without_modes = dict(formal_d)
    dblur_without_modes = dict(dblur)
    formal_without_modes.pop("mdrc_degrade_modes")
    dblur_without_modes.pop("mdrc_degrade_modes")
    assert dblur_without_modes == formal_without_modes
