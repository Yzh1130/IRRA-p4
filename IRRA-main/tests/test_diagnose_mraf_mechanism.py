import csv
import io
import inspect
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch
from torch import nn

from model.mraf import ModalityReliabilityAdaptiveFusion
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
    spearman_correlation,
    summarize_vector,
    unwrap_state_dict,
)


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
    (case_root / "configs.yaml").write_text("dataset_name: RGBT-PEDES\n", encoding="utf-8")
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
    assert not output_dir.exists()
