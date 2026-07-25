import csv
import gc
import io
import inspect
import weakref
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from model.mraf import ModalityReliabilityAdaptiveFusion
from tools.revision import diagnose_mraf_mechanism as diagnostic
from tools.revision.diagnose_mraf_mechanism import (
    CHECKPOINT_PARAMETER_FIELDS,
    GATE_DIRECTION_FIELDS,
    GRADIENT_FIELDS,
    SCENARIO_GATE_FIELDS,
    SCENARIO_TARGET_FIELDS,
    _write_csv,
    alpha_target_metrics,
    as_fp32_vector,
    filter_mraf_parameters,
    force_mraf_gate,
    gate_direction_row,
    main,
    measure_loss_gradients,
    pearson_correlation,
    release_cuda_objects,
    run_gradient_diagnostics,
    slice_batch_for_diagnostics,
    spearman_correlation,
    summarize_vector,
    unwrap_state_dict,
)


class _TinyDiagnosticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mraf_enabled = True
        self.mraf_fusion = ModalityReliabilityAdaptiveFusion(
            feature_dim=4,
            hidden_dim=2,
            dropout=0.0,
        )
        self.seen_batch_sizes = []

    def forward(self, batch):
        self.seen_batch_sizes.append(int(batch["images"].shape[0]))
        fused = self.mraf_fusion(
            batch["images"],
            batch["thermal_images"],
        )
        return {"diagnostic_loss": fused.square().mean()}


def _patch_tiny_gradient_path(monkeypatch):
    monkeypatch.setattr(
        diagnostic,
        "_build_total_loss",
        lambda args, ret, epoch: ret["diagnostic_loss"],
    )
    monkeypatch.setattr(
        diagnostic,
        "_should_execute_degraded_branch",
        lambda args, epoch: False,
    )
    monkeypatch.setattr(diagnostic, "_mdrc_enabled", lambda args: False)
    monkeypatch.setattr(diagnostic, "_mraf_enabled", lambda args: False)


def _tiny_train_loader(num_batches=3, batch_size=5):
    return [
        {
            "images": torch.randn(batch_size, 4),
            "thermal_images": torch.randn(batch_size, 4),
            "pids": torch.arange(batch_size),
            "captions": [f"sample-{index}" for index in range(batch_size)],
        }
        for _ in range(num_batches)
    ]


def test_fp32_statistics_and_quantiles():
    values = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float16)
    converted = as_fp32_vector(values)
    stats = summarize_vector(values, "value")

    assert converted.dtype == torch.float32
    assert stats["value_mean"] == pytest.approx(1.5)
    assert stats["value_std"] == pytest.approx(
        torch.std(converted, unbiased=False).item()
    )
    assert stats["value_min"] == pytest.approx(0.0)
    assert stats["value_p10"] == pytest.approx(0.3)
    assert stats["value_p50"] == pytest.approx(1.5)
    assert stats["value_p90"] == pytest.approx(2.7)
    assert stats["value_max"] == pytest.approx(3.0)


def test_pearson_and_spearman_correlations():
    ascending = torch.tensor([1.0, 2.0, 3.0, 4.0])
    same_order = torch.tensor([10.0, 20.0, 30.0, 40.0])
    reverse_order = same_order.flip(0)

    assert pearson_correlation(ascending, same_order) == pytest.approx(1.0)
    assert spearman_correlation(ascending, same_order) == pytest.approx(1.0)
    assert spearman_correlation(ascending, reverse_order) == pytest.approx(-1.0)


def test_alpha_target_identical_has_zero_error_and_unit_correlation():
    alpha_rgb = torch.tensor([0.1, 0.3, 0.7, 0.9])
    alpha = torch.stack([alpha_rgb, 1.0 - alpha_rgb], dim=1)
    metrics = alpha_target_metrics(alpha, alpha)

    assert metrics["alpha_target_mae"] == pytest.approx(0.0)
    assert metrics["alpha_target_mse"] == pytest.approx(0.0)
    assert metrics["alpha_target_pearson"] == pytest.approx(1.0)
    assert metrics["alpha_target_spearman"] == pytest.approx(1.0)
    assert metrics["alpha_target_direction_agreement"] == pytest.approx(1.0)


def test_constant_alpha_reports_undefined_correlations():
    alpha = torch.full((4, 2), 0.5)
    target_rgb = torch.tensor([0.1, 0.4, 0.6, 0.9])
    target = torch.stack([target_rgb, 1.0 - target_rgb], dim=1)
    metrics = alpha_target_metrics(alpha, target)

    assert metrics["alpha_target_pearson"] is None
    assert metrics["alpha_target_spearman"] is None


def test_gate_direction_statistics_for_rgb_degradation():
    clean_rgb = torch.tensor([0.7, 0.6, 0.5, 0.4])
    degraded_rgb = torch.tensor([0.6, 0.5, 0.4, 0.3])
    clean_ir = 1.0 - clean_rgb
    degraded_ir = 1.0 - degraded_rgb

    row = gate_direction_row(
        "case",
        "rgb_noise_0.05",
        torch.stack([clean_rgb, clean_ir], dim=1),
        torch.stack([degraded_rgb, degraded_ir], dim=1),
    )

    assert row["degraded_modality"] == "rgb"
    assert row["degraded_alpha_delta_mean"] == pytest.approx(-0.1)
    assert row["counterpart_alpha_delta_mean"] == pytest.approx(0.1)
    assert row["degraded_alpha_mean_decreased"] is True
    assert row["counterpart_alpha_mean_increased"] is True
    assert row["correct_direction_fraction"] == pytest.approx(1.0)


def test_unwrap_state_dict_supports_common_checkpoint_layouts():
    tensor = torch.ones(2)
    direct = {"mraf_fusion.scorer.0.bias": tensor}

    assert unwrap_state_dict(direct) == direct
    assert unwrap_state_dict({"model": direct}) == direct
    assert unwrap_state_dict({"state_dict": direct}) == direct
    assert unwrap_state_dict({"model_state_dict": direct}) == direct
    assert unwrap_state_dict({"model": {"module.mraf_fusion.scorer.0.bias": tensor}}) == {
        "mraf_fusion.scorer.0.bias": tensor
    }


def test_filter_mraf_parameters_excludes_unrelated_tensors():
    state = {
        "mraf_fusion.scorer.0.weight": torch.ones(2, 2),
        "mraf_fusion.scorer.3.bias": torch.zeros(1),
        "base_model.visual.weight": torch.ones(1),
    }

    selected = filter_mraf_parameters(state)

    assert list(selected) == [
        "mraf_fusion.scorer.0.weight",
        "mraf_fusion.scorer.3.bias",
    ]


def test_force_mraf_gate_restores_original_forward_even_after_exception():
    module = ModalityReliabilityAdaptiveFusion(feature_dim=4, hidden_dim=2)
    rgb = torch.randn(3, 4)
    ir = torch.randn(3, 4)
    original = module(rgb, ir)

    with pytest.raises(RuntimeError, match="sentinel"):
        with force_mraf_gate(module, 1.0, 0.0):
            _, forced = module(rgb, ir, return_aux=True)
            assert torch.allclose(forced["alpha_rgb"], torch.ones(3, 1))
            assert torch.allclose(forced["alpha_ir"], torch.zeros(3, 1))
            raise RuntimeError("sentinel")

    restored = module(rgb, ir)
    assert torch.allclose(restored, original)


def test_measure_loss_gradients_does_not_update_parameters_or_populate_grad():
    layer = nn.Linear(3, 1)
    inputs = torch.randn(5, 3)
    output = layer(inputs)
    loss = output.square().mean()
    before = {name: value.detach().clone() for name, value in layer.named_parameters()}

    norms = measure_loss_gradients(
        {"loss": loss},
        [("weight", layer.weight), ("bias", layer.bias)],
    )

    assert norms["loss"]["weight"] is not None
    assert norms["loss"]["bias"] is not None
    for name, parameter in layer.named_parameters():
        assert torch.equal(parameter, before[name])
        assert parameter.grad is None

    source = inspect.getsource(measure_loss_gradients)
    assert "optimizer.step" not in source
    assert ".backward(" not in source


def test_gradient_microbatch_size_must_be_positive():
    parser = diagnostic.build_parser()
    common_args = [
        "--run-root",
        "runs",
        "--data-root",
        "data",
        "--cases",
        "mraf_only",
        "--output-dir",
        "output",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(
            common_args + ["--gradient-microbatch-size", "0"]
        )
    assert parser.parse_args(common_args).gradient_microbatch_size == 4


def test_direct_single_checkpoint_mode_discovers_artifacts(tmp_path):
    data_root = tmp_path / "data"
    run_dir = tmp_path / "candidate"
    data_root.mkdir()
    run_dir.mkdir()
    config_path = run_dir / "configs.yaml"
    checkpoint_path = run_dir / "best.pth"
    config_path.write_text(
        "dataset_name: RGBT-PEDES\n"
        "img_size: !!python/tuple\n"
        "  - 384\n"
        "  - 128\n",
        encoding="utf-8",
    )
    checkpoint_path.write_bytes(b"dry-run checkpoint")
    (run_dir / "train.log").write_text("fixture\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "--config-file",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--case-label",
                "S4",
                "--data-root",
                str(data_root),
                "--device",
                "cpu",
                "--max-batches",
                "1",
                "--gradient-batches",
                "1",
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
        )

    payload = stdout.getvalue()
    assert exit_code == 0
    assert '"input_mode": "direct"' in payload
    assert '"case": "S4"' in payload
    assert not output_dir.exists()


def test_cli_rejects_mixed_or_partial_input_modes(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    base = [
        "--data-root",
        str(data_root),
        "--output-dir",
        str(tmp_path / "output"),
        "--device",
        "cpu",
        "--dry-run",
    ]
    with pytest.raises(ValueError, match="requires"):
        main(base + ["--config-file", "configs.yaml"])
    with pytest.raises(ValueError, match="exactly one"):
        main(
            base
            + [
                "--run-root",
                str(tmp_path),
                "--cases",
                "mraf_only",
                "--config-file",
                "configs.yaml",
                "--checkpoint",
                "best.pth",
                "--case-label",
                "S0",
            ]
        )


def test_slice_batch_for_diagnostics_handles_batch_values_without_mutation():
    images = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    auxiliary = torch.arange(3)
    original = {
        "images": images,
        "pids": torch.arange(5),
        "captions": ["a", "b", "c", "d", "e"],
        "paths": ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"),
        "scalar": torch.tensor(7.0),
        "non_batch_tensor": auxiliary,
        "name": "fixture",
        "config": {"mode": "read-only"},
    }
    images_before = images.clone()
    captions_before = list(original["captions"])

    sliced, original_size, effective_size = slice_batch_for_diagnostics(
        original,
        max_samples=2,
    )

    assert original_size == 5
    assert effective_size == 2
    assert sliced is not original
    assert sliced["images"].shape == (2, 4)
    assert sliced["pids"].tolist() == [0, 1]
    assert sliced["captions"] == ["a", "b"]
    assert isinstance(sliced["captions"], list)
    assert sliced["paths"] == ("a.jpg", "b.jpg")
    assert isinstance(sliced["paths"], tuple)
    assert sliced["scalar"] is original["scalar"]
    assert sliced["non_batch_tensor"] is auxiliary
    assert sliced["name"] == "fixture"
    assert sliced["config"] is original["config"]
    assert torch.equal(original["images"], images_before)
    assert original["captions"] == captions_before


def test_release_cuda_objects_does_not_modify_model_parameters():
    model = nn.Linear(4, 2)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    release_cuda_objects(model)

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_cpu_microbatch_gradient_diagnostics_preserve_parameters(
    monkeypatch,
):
    _patch_tiny_gradient_path(monkeypatch)
    model = _TinyDiagnosticModel()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    rows = run_gradient_diagnostics(
        case="mraf_only",
        model=model,
        train_loader=_tiny_train_loader(),
        args=SimpleNamespace(degrade_aug_mode="none"),
        device=torch.device("cpu"),
        gradient_batches=3,
        gradient_microbatch_size=2,
        diagnostic_epoch=60,
    )

    assert len(rows) == 3
    assert model.seen_batch_sizes == [2, 2, 2]
    for row in rows:
        assert row["original_batch_size"] == 5
        assert row["effective_microbatch_size"] == 2
        assert row["gradient_microbatch_size"] == 2
        assert row["cuda_memory_allocated_before_mb"] == 0.0
        assert row["cuda_memory_allocated_after_forward_mb"] == 0.0
        assert row["cuda_peak_memory_allocated_mb"] == 0.0
        assert row["cuda_peak_memory_reserved_mb"] == 0.0
        assert row["parameters_unchanged"] is True
        assert row["optimizer_step_executed"] is False
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
        assert parameter.grad is None

    source = inspect.getsource(run_gradient_diagnostics)
    assert "build_optimizer" not in source
    assert ".step(" not in source


def test_repeated_cpu_diagnostics_do_not_retain_models_globally(monkeypatch):
    _patch_tiny_gradient_path(monkeypatch)

    def run_once():
        model = _TinyDiagnosticModel()
        run_gradient_diagnostics(
            case="mraf_only",
            model=model,
            train_loader=_tiny_train_loader(),
            args=SimpleNamespace(degrade_aug_mode="none"),
            device=torch.device("cpu"),
            gradient_batches=3,
            gradient_microbatch_size=2,
            diagnostic_epoch=60,
        )
        return weakref.ref(model)

    references = [run_once(), run_once()]
    gc.collect()

    assert all(reference() is None for reference in references)
    assert "build_eval_resources" not in inspect.getsource(diagnostic.main)
    assert "build_train_resources" not in inspect.getsource(diagnostic.main)
    assert inspect.getsource(diagnostic.main).count(
        "build_case_resources("
    ) == 1


def test_gradient_memory_schema_and_cli_help_are_stable():
    required_memory_fields = {
        "original_batch_size",
        "effective_microbatch_size",
        "gradient_microbatch_size",
        "cuda_memory_allocated_before_mb",
        "cuda_memory_allocated_after_forward_mb",
        "cuda_peak_memory_allocated_mb",
        "cuda_peak_memory_reserved_mb",
    }

    assert required_memory_fields.issubset(set(GRADIENT_FIELDS))
    assert "--gradient-microbatch-size" in diagnostic.build_parser().format_help()
    help_text = diagnostic.build_parser().format_help()
    assert "--config-file" in help_text
    assert "--checkpoint" in help_text
    assert "--case-label" in help_text


def test_existing_output_directory_is_rejected_after_failure(tmp_path):
    failed_output = tmp_path / "failed_diagnosis"
    failed_output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        diagnostic._prepare_output_dir(failed_output)

    assert not (failed_output / "summary.txt").exists()


@pytest.mark.parametrize(
    "fields",
    [
        CHECKPOINT_PARAMETER_FIELDS,
        SCENARIO_GATE_FIELDS,
        SCENARIO_TARGET_FIELDS,
        GATE_DIRECTION_FIELDS,
        GRADIENT_FIELDS,
    ],
)
def test_csv_field_order_is_stable(tmp_path, fields):
    output_path = tmp_path / "output.csv"
    row = {field: None for field in fields}
    _write_csv(output_path, fields, [row])

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == tuple(fields)
        assert len(list(reader)) == 1


def test_cpu_dry_run_discovers_artifacts_without_creating_output(tmp_path):
    run_root = tmp_path / "runs"
    data_root = tmp_path / "data"
    case_root = run_root / "formal" / "20260725_mraf_only_seed1"
    case_root.mkdir(parents=True)
    data_root.mkdir()
    (case_root / "configs.yaml").write_text(
        "dataset_name: RGBT-PEDES\n"
        "milestones: !!python/tuple\n"
        "  - 20\n"
        "  - 50\n",
        encoding="utf-8",
    )
    (case_root / "best.pth").write_bytes(b"not loaded during dry-run")
    (case_root / "train.log").write_text("dry-run fixture\n", encoding="utf-8")
    output_dir = tmp_path / "diagnostics"

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = main(
            [
                "--run-root",
                str(run_root),
                "--data-root",
                str(data_root),
                "--cases",
                "mraf_only",
                "--device",
                "cpu",
                "--degrade-seed",
                "2026",
                "--max-batches",
                "1",
                "--gradient-batches",
                "1",
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ]
        )

    assert exit_code == 0
    assert "mraf_only" in stdout.getvalue()
    assert "best.pth" in stdout.getvalue()
    assert '"gradient_microbatch_size": 4' in stdout.getvalue()
    assert not output_dir.exists()
