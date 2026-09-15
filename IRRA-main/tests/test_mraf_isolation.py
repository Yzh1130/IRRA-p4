"""Protocol-v2 checks for D-iso without training or real dataset access."""

import json
import os
import subprocess
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from model.mraf import (
    ModalityReliabilityAdaptiveFusion,
    compute_mraf_reliability_loss,
)
from tools.protocol.gen_configs import (
    ProtocolConfigError,
    build_configs,
    generate_supplementary_configs,
    validate_supplementary_configs,
)
from utils.mdrc import DEFAULT_DEGRADE_MODES


BASELINE_COMMIT = "50a2110e"
GIT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _module(isolated, gate_fp32=False):
    torch.manual_seed(20260915)
    module = ModalityReliabilityAdaptiveFusion(
        feature_dim=512,
        hidden_dim=128,
        dropout=0.1,
        gate_logit_scale=5.0,
        gate_fp32=gate_fp32,
        rel_isolate_input=isolated,
    )
    # The historical final layer is zero-initialized.  A nonzero fixed value
    # represents the protocol's already-supervised, effective-gradient state.
    with torch.no_grad():
        module.output_layer.weight.normal_(mean=0.0, std=0.02)
        module.output_layer.bias.fill_(0.03)
    return module


def _inputs():
    generator = torch.Generator().manual_seed(20260916)
    rgb = torch.randn(12, 512, generator=generator)
    ir = torch.randn(12, 512, generator=generator)
    return rgb, ir


def _baseline_class():
    result = subprocess.run(
        [
            "git",
            "-C",
            str(GIT_ROOT),
            "show",
            f"{BASELINE_COMMIT}:IRRA-main/model/mraf.py",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    module = types.ModuleType("baseline_mraf_before_d_iso")
    exec(compile(result.stdout, "baseline_mraf_before_d_iso.py", "exec"), module.__dict__)
    return module.ModalityReliabilityAdaptiveFusion


def _assert_zero_or_none(gradient):
    assert gradient is None or torch.equal(gradient, torch.zeros_like(gradient))


def _shell_path(path):
    resolved = str(Path(path).resolve())
    if os.name != "nt":
        return resolved
    drive, tail = os.path.splitdrive(resolved)
    tail = tail.lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive[0].lower()}/{tail}"


@pytest.mark.parametrize("gate_fp32", [False, True])
def test_training_alpha_aux_equals_main_alpha_with_separate_reused_masks(gate_fp32):
    module = _module(isolated=True, gate_fp32=gate_fp32).train()
    rgb, ir = _inputs()
    captured = []
    hook = module.scorer[2].register_forward_hook(
        lambda _layer, _inputs, output: captured.append(output.detach().clone())
    )
    torch.manual_seed(20260917)
    _, aux = module(rgb, ir, return_aux=True)
    hook.remove()

    assert len(captured) == 2
    rgb_mask = (captured[0] != 0).to(captured[0].dtype)
    ir_mask = (captured[1] != 0).to(captured[1].dtype)
    assert not torch.equal(rgb_mask, ir_mask)

    logits = []
    for feature, mask in ((rgb, rgb_mask), (ir, ir_mask)):
        normalized = F.normalize(feature, p=2, dim=-1).detach()
        hidden = module.scorer[1](module.scorer[0](normalized))
        logits.append(module.scorer[3](hidden * (mask / (1.0 - module.dropout))))
    expected = F.softmax(
        torch.cat(logits, dim=-1)
        * module.gate_logit_scale
        / module.gate_temperature,
        dim=-1,
    )

    assert torch.equal(aux["alpha_aux"], expected)
    assert torch.equal(aux["alpha_aux"], aux["alpha_main"])
    assert torch.equal(aux["alpha"], aux["alpha_aux"])
    assert torch.equal(aux["alpha_rgb"], aux["alpha_main"][:, :1])
    assert torch.equal(aux["alpha_ir"], aux["alpha_main"][:, 1:])


@pytest.mark.parametrize("gate_fp32", [False, True])
def test_large_batch_dropout_rounding_keeps_aux_alpha_bitwise_equal(gate_fp32):
    module = _module(isolated=True, gate_fp32=gate_fp32).train()
    generator = torch.Generator().manual_seed(0)
    rgb = torch.randn(100, 512, generator=generator)
    ir = torch.randn(100, 512, generator=generator)
    dropped = []
    hook = module.scorer[2].register_forward_hook(
        lambda _layer, _inputs, output: dropped.append(output.detach().clone())
    )
    torch.manual_seed(0)
    _, aux = module(rgb, ir, return_aux=True)
    hook.remove()

    assert len(dropped) == 2
    for features, dropout_output in ((rgb, dropped[0]), (ir, dropped[1])):
        normalized = F.normalize(features, p=2, dim=-1)
        hidden = module.scorer[1](module.scorer[0](normalized))
        mask = (dropout_output != 0).to(dropout_output.dtype)
        assert torch.equal(
            hidden * (mask / (1.0 - module.dropout)),
            dropout_output,
        )
    assert torch.equal(aux["alpha_aux"], aux["alpha_main"])


@pytest.mark.parametrize("gate_fp32", [False, True])
def test_reliability_loss_blocks_encoder_inputs_but_updates_both_scorer_layers(gate_fp32):
    module = _module(isolated=True, gate_fp32=gate_fp32).train()
    rgb, ir = (feature.detach().requires_grad_() for feature in _inputs())
    torch.manual_seed(20260918)
    _, aux = module(rgb, ir, return_aux=True)
    target = aux["alpha_aux"].new_tensor([0.75, 0.25]).expand_as(aux["alpha_aux"])
    loss = compute_mraf_reliability_loss(aux["alpha_aux"], target)
    loss.backward()

    _assert_zero_or_none(rgb.grad)
    _assert_zero_or_none(ir.grad)
    for layer in (module.scorer[0], module.scorer[3]):
        assert layer.weight.grad is not None
        assert bool(torch.isfinite(layer.weight.grad).all().item())
        assert bool((layer.weight.grad != 0).any().item())


@pytest.mark.parametrize("gate_fp32", [False, True])
def test_fusion_and_variance_gate_input_gradients_match_isolation_off(gate_fp32):
    isolated = _module(isolated=True, gate_fp32=gate_fp32).train()
    legacy = _module(isolated=False, gate_fp32=gate_fp32).train()
    legacy.load_state_dict(isolated.state_dict())
    base_rgb, base_ir = _inputs()
    results = []
    for module in (legacy, isolated):
        rgb = base_rgb.detach().clone().requires_grad_()
        ir = base_ir.detach().clone().requires_grad_()
        torch.manual_seed(20260919)
        fused, aux = module(rgb, ir, return_aux=True)
        rng_after_forward = torch.get_rng_state().clone()
        # The aliases also feed model/build.py's fused-variance calculation.
        gate_variance_proxy = (
            aux["alpha_rgb"].square() + 2.0 * aux["alpha_ir"].square()
        ).mean()
        loss = fused.square().mean() + gate_variance_proxy
        loss.backward()
        results.append((rgb.grad.clone(), ir.grad.clone(), rng_after_forward))

    for legacy_gradient, isolated_gradient in zip(results[0][:2], results[1][:2]):
        assert torch.allclose(
            legacy_gradient,
            isolated_gradient,
            rtol=1e-6,
            atol=1e-9,
        )
    assert torch.equal(results[0][2], results[1][2])


@pytest.mark.parametrize("gate_fp32", [False, True])
def test_isolation_off_matches_prechange_class_bitwise_and_rng(gate_fp32):
    baseline_type = _baseline_class()
    torch.manual_seed(20260920)
    baseline = baseline_type(
        feature_dim=512,
        hidden_dim=128,
        dropout=0.1,
        gate_logit_scale=5.0,
        gate_fp32=gate_fp32,
    ).train()
    torch.manual_seed(20260920)
    current = ModalityReliabilityAdaptiveFusion(
        feature_dim=512,
        hidden_dim=128,
        dropout=0.1,
        gate_logit_scale=5.0,
        gate_fp32=gate_fp32,
        rel_isolate_input=False,
    ).train()
    assert set(baseline.state_dict()) == set(current.state_dict())
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, current.state_dict()[name])
    with torch.no_grad():
        baseline.output_layer.weight.normal_(0.0, 0.02)
        baseline.output_layer.bias.fill_(0.03)
    current.load_state_dict(baseline.state_dict())

    rgb, ir = _inputs()
    torch.manual_seed(20260921)
    baseline_fused, baseline_aux = baseline(rgb, ir, return_aux=True)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(20260921)
    current_fused, current_aux = current(rgb, ir, return_aux=True)
    current_rng = torch.get_rng_state().clone()

    assert set(current_aux) == set(baseline_aux)
    for key, old_value in baseline_aux.items():
        if torch.is_tensor(old_value):
            assert torch.equal(old_value, current_aux[key]), key
        else:
            assert old_value == current_aux[key], key
    assert torch.equal(current_fused, baseline_fused)
    assert torch.equal(current_rng, baseline_rng)


def test_eval_alpha_aux_matches_main_without_dropout_mask_sampling():
    module = _module(isolated=True).eval()
    rgb, ir = _inputs()
    torch.manual_seed(20260922)
    before = torch.get_rng_state().clone()
    fused, aux = module(rgb, ir, return_aux=True)
    after = torch.get_rng_state().clone()

    assert torch.equal(aux["alpha_aux"], aux["alpha_main"])
    assert torch.equal(aux["alpha"], aux["alpha_aux"])
    assert torch.equal(before, after)
    assert bool(torch.isfinite(fused).all().item())


def test_d_iso_config_differs_from_formal_d_by_only_isolation_flag():
    base = {"mdrc_degrade_modes": list(DEFAULT_DEGRADE_MODES)}
    e_temperature = 0.2
    utility_prior = [0.7, 0.3]
    formal_d = build_configs(base, e_temperature, utility_prior)["D"]
    supplementary = generate_supplementary_configs(
        base, e_temperature, utility_prior
    )
    d_iso = supplementary["D-iso"]

    assert d_iso["mraf_rel_isolate_input"] is True
    assert d_iso["mdrc_degrade_modes"] == list(DEFAULT_DEGRADE_MODES)
    without_isolation = dict(d_iso)
    without_isolation.pop("mraf_rel_isolate_input")
    assert without_isolation == formal_d

    altered = {name: dict(config) for name, config in supplementary.items()}
    altered["D-iso"]["mdrc_degrade_modes"] = ["rgb_drop", "ir_drop"]
    with pytest.raises(ProtocolConfigError, match="D-iso must equal formal D"):
        validate_supplementary_configs(
            altered, base, e_temperature, utility_prior
        )


def test_d_iso_runner_validate_only_requires_out_and_never_trains(tmp_path):
    base_config = tmp_path / "base_config.yaml"
    frozen_hparams = tmp_path / "FROZEN_HPARAMS.json"
    output_dir = tmp_path / "d_iso_validate"
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
        "mdrc_degrade_modes: [rgb_drop, ir_drop, rgb_noise, ir_noise, "
        "rgb_attenuate, ir_attenuate]\n"
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
        ["bash", "tools/protocol/run_dblur_grid.sh", "--arms", "D-iso", "--validate-only"],
        cwd=PROJECT_ROOT,
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
            _shell_path(base_config),
            "--frozen-hparams",
            _shell_path(frozen_hparams),
            "--out",
            _shell_path(output_dir),
            "--arms",
            "D-iso",
            "--seeds",
            "101-102",
            "--validate-only",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    plans = [
        line for line in completed.stdout.splitlines() if line.startswith("[RUN_PLAN]")
    ]
    assert plans == ["[RUN_PLAN] D-iso_seed101", "[RUN_PLAN] D-iso_seed102"]
    assert "selected_arms=D-iso planned_runs=2" in completed.stdout
    for forbidden in ("[RUN_START]", "[TRAIN_START]", "[SHUTDOWN]"):
        assert forbidden not in completed.stdout
    assert (output_dir / "configs" / "D-iso.yaml").is_file()
    assert not any((output_dir / "runs").iterdir())
