import json
import os
from pathlib import Path
import subprocess
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
DNONOISE_MODES = [
    "rgb_drop",
    "ir_drop",
    "rgb_attenuate",
    "ir_attenuate",
]
TENSOR_SHAPE = (4, 3, 32, 16)
ROOT = Path(__file__).resolve().parents[1]


def _as_shell_path(path):
    resolved = str(Path(path).resolve())
    if os.name != "nt":
        return resolved
    drive, tail = os.path.splitdrive(resolved)
    tail = tail.lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive[0].lower()}/{tail}"


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
    assert tuple(supplementary) == ("Dblur", "DnoNoise", "D-iso")
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


def test_dno_noise_config_equals_d_except_for_degradation_modes():
    base_config = {"mdrc_degrade_modes": list(DEFAULT_MODES)}
    e_temperature = 0.2
    utility_prior = [0.7284522740054223, 0.2715477259945776]

    formal_d = build_configs(
        base_config,
        e_temperature,
        utility_prior,
    )["D"]
    dno_noise = generate_supplementary_configs(
        base_config,
        e_temperature,
        utility_prior,
    )["DnoNoise"]

    assert dno_noise["mdrc_degrade_modes"] == DNONOISE_MODES
    assert set(formal_d) == set(dno_noise)
    differing_keys = {
        key for key in formal_d if formal_d[key] != dno_noise[key]
    }
    assert differing_keys == {"mdrc_degrade_modes"}

    formal_without_modes = dict(formal_d)
    dno_noise_without_modes = dict(dno_noise)
    formal_without_modes.pop("mdrc_degrade_modes")
    dno_noise_without_modes.pop("mdrc_degrade_modes")
    assert dno_noise_without_modes == formal_without_modes


def test_dno_noise_runner_validate_only_plans_ten_runs(tmp_path):
    base_config = tmp_path / "base_config.yaml"
    frozen_hparams = tmp_path / "FROZEN_HPARAMS.json"
    output_dir = tmp_path / "dno_noise_validate"
    base_config.write_text(
        "dataset_name: RGBT-PEDES\n"
        "root_dir: datasets/TextReID\n"
        "training: true\n"
        "val_dataset: test\n"
        "dev_split_file: null\n"
        "MLM: true\n"
        "img_aug: true\n"
        "sampler: random\n"
        "num_instance: 4\n"
        "img_size: [384, 128]\n"
        "stride_size: 16\n"
        "text_length: 77\n"
        "pretrain_choice: ViT-B/16\n"
        "cmt_depth: 4\n"
        "temperature: 0.02\n"
        "save_full_eval_state_ckpt: false\n",
        encoding="utf-8",
    )
    frozen_hparams.write_text(
        json.dumps(
            {
                "E": {"mraf_target_temperature": 0.2},
                "U": {"mraf_utility_prior": [0.7, 0.3]},
            }
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["AUTO_SHUTDOWN"] = "0"

    missing_out = subprocess.run(
        [
            "bash",
            "tools/protocol/run_dblur_grid.sh",
            "--arms",
            "DnoNoise",
            "--validate-only",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert missing_out.returncode == 2
    assert "requires an explicit independent --out" in missing_out.stderr

    completed = subprocess.run(
        [
            "bash",
            "tools/protocol/run_dblur_grid.sh",
            "--base-config",
            _as_shell_path(base_config),
            "--frozen-hparams",
            _as_shell_path(frozen_hparams),
            "--out",
            _as_shell_path(output_dir),
            "--arms",
            "DnoNoise",
            "--validate-only",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    expected_plan = [
        f"[RUN_PLAN] DnoNoise_seed{seed}" for seed in range(101, 111)
    ]
    actual_plan = [
        line for line in completed.stdout.splitlines() if line.startswith("[RUN_PLAN]")
    ]
    assert actual_plan == expected_plan
    assert "selected_arms=DnoNoise planned_runs=10" in completed.stdout
    assert "[VALIDATE_ONLY_DONE]" in completed.stdout
    assert "[RUN_START]" not in completed.stdout
    assert "[TRAIN_START]" not in completed.stdout
    assert "[SHUTDOWN]" not in completed.stdout
    assert (output_dir / "configs" / "Dblur.yaml").is_file()
    assert (output_dir / "configs" / "DnoNoise.yaml").is_file()
    assert not any((output_dir / "runs").iterdir())
    assert not any((output_dir / "marks").iterdir())
