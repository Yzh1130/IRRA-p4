from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from model.mraf import ModalityReliabilityAdaptiveFusion
import tools.followup.gate_counterfactual_eval as gate_eval
from tools.followup.summarize_counterfactual import (
    CORE_SCENARIOS,
    build_summary_tables,
    compute_p1_p2,
)


def test_equal_counterfactual_is_exact_scale_two_average():
    generator = torch.Generator().manual_seed(20260915)
    rgb = torch.randn(5, 7, generator=generator)
    ir = torch.randn(5, 7, generator=generator)
    fusion_shell = SimpleNamespace(equalize_modality_norm=False)

    fused, alpha = gate_eval.fuse_counterfactual_features(
        fusion_shell,
        rgb,
        ir,
        "equal",
    )

    expected = 2.0 * (0.5 * rgb + 0.5 * ir)
    assert torch.equal(fused, expected)
    assert torch.equal(alpha, torch.full((5, 2), 0.5))


def test_learned_counterfactual_reuses_mraf_alpha_bitwise():
    module = ModalityReliabilityAdaptiveFusion(
        7,
        hidden_dim=5,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        module.output_layer.weight.copy_(
            torch.tensor([[0.4, -0.2, 0.3, 0.1, -0.5]])
        )
        module.output_layer.bias.fill_(0.07)
    generator = torch.Generator().manual_seed(20260916)
    rgb = torch.randn(4, 7, generator=generator)
    ir = torch.randn(4, 7, generator=generator)

    with torch.inference_mode():
        expected_fused, expected_aux = module(rgb, ir, return_aux=True)
        actual_fused, actual_alpha = gate_eval.fuse_counterfactual_features(
            module,
            rgb,
            ir,
            "learned",
        )

    assert torch.equal(actual_alpha, expected_aux["alpha"])
    assert torch.equal(actual_fused, expected_fused)


def test_calibration_counts_one_image_pair_not_three_captions(monkeypatch):
    samples = [
        (7, 11, "rgb/a.jpg", "ir/a.jpg", caption)
        for caption in ("one", "two", "three")
    ]
    observed = {"pair_count": None, "encoded": 0}

    class FakeDataset:
        def __init__(self, pids, rgb_paths, t_paths, transform):
            observed["pair_count"] = len(pids)
            self.pids = list(pids)

        def __len__(self):
            return len(self.pids)

        def __getitem__(self, index):
            return (
                self.pids[index],
                torch.ones(3, 2, 2),
                torch.zeros(3, 2, 2),
            )

    class FakeModel:
        def encode_rgbt_eval(self, rgb, thermal, return_local_tokens=False):
            assert return_local_tokens is True
            observed["encoded"] += int(rgb.shape[0])
            return {
                "mraf_alpha": rgb.new_tensor([[0.25, 0.75]]).expand(
                    rgb.shape[0], -1
                )
            }

    monkeypatch.setattr(
        gate_eval,
        "RGBTPEDES",
        lambda **kwargs: SimpleNamespace(train=samples),
    )
    monkeypatch.setattr(gate_eval, "build_transforms", lambda **kwargs: object())
    monkeypatch.setattr(gate_eval, "ImageDatasetRGBT", FakeDataset)
    args = SimpleNamespace(
        root_dir="unused",
        img_size=(8, 4),
        test_batch_size=8,
        num_workers=0,
    )

    weights, count = gate_eval.compute_calibration_weights(
        FakeModel(),
        args,
        torch.device("cpu"),
    )

    assert observed == {"pair_count": 1, "encoded": 1}
    assert count == 1
    assert torch.equal(weights, torch.tensor([0.25, 0.75]))


def test_p1_is_mean_of_only_two_rgb_noise_r1_values():
    endpoints = compute_p1_p2(
        {
            "rgb_noise_005": {"R1": 10.0},
            "rgb_noise_010": {"R1": 30.0},
            "rgb_blur_k7": {"R1": 999.0},
            "clean": {"R1": -999.0},
        }
    )

    assert endpoints["P1"] == pytest.approx(20.0)
    assert endpoints["P2"] == pytest.approx(999.0)


def test_holm_primary_family_is_limited_to_d_equal_vs_learned():
    rows = []
    offsets = {
        "learned": 0.0,
        "equal": 1.0,
        "calib": 0.5,
        "rgb_only": 2.0,
        "ir_only": -2.0,
    }
    for group in ("D", "E"):
        for scenario_index, scenario in enumerate(CORE_SCENARIOS):
            for mode, offset in offsets.items():
                value = float(scenario_index) + offset
                rows.append(
                    {
                        "group": group,
                        "seed": 101,
                        "scenario": scenario,
                        "mode": mode,
                        "R1": value,
                        "R5": value,
                        "R10": value,
                        "mAP": value,
                        "mINP": value,
                    }
                )

    summary = build_summary_tables(rows)["summary"]
    keyed = {
        (row["group"], row["comparison"], row["endpoint"]): row
        for row in summary
    }

    for endpoint in ("P1", "P2"):
        primary = keyed[("D", "equal-learned", endpoint)]
        assert primary["role"] == "primary"
        assert primary["holm_p"] is not None
        secondary_group = keyed[("E", "equal-learned", endpoint)]
        assert secondary_group["role"] == "secondary"
        assert secondary_group["holm_p"] is None
        calibrated = keyed[("D", "calib-learned", endpoint)]
        assert calibrated["role"] == "secondary"
        assert calibrated["holm_p"] is None


def test_frozen_dseed_is_rejected_before_cuda_or_io(tmp_path):
    args = SimpleNamespace(
        calib_source="train_clean",
        dseed=2027,
        run_dir=[],
        out=str(tmp_path),
    )

    with pytest.raises(gate_eval.CounterfactualError, match="requires --dseed 2026"):
        gate_eval.run(args)


def test_output_must_not_overlap_main_grid_or_run_dir(tmp_path, monkeypatch):
    main_grid = tmp_path / "MAIN_GRID"
    run_dir = main_grid / "runs" / "D_seed101"
    independent = tmp_path / "FOLLOWUP_B"
    monkeypatch.setattr(gate_eval, "MAIN_METRICS", main_grid / "metrics.tsv")

    with pytest.raises(gate_eval.CounterfactualError, match="independent"):
        gate_eval._require_independent_output(main_grid / "followup", [run_dir])
    with pytest.raises(gate_eval.CounterfactualError, match="independent"):
        gate_eval._require_independent_output(run_dir, [run_dir])

    gate_eval._require_independent_output(independent, [run_dir])

    features = independent / "features"
    features.mkdir(parents=True)
    (features / "stale.pt").write_bytes(b"stale")
    with pytest.raises(FileExistsError, match="features"):
        gate_eval._require_fresh_outputs(independent)


def test_reference_rows_must_match_frozen_degradation_parameters(tmp_path):
    path = Path(tmp_path) / "metrics.tsv"
    header = (
        "group\tseed\tscenario\tmode\tstrength\tkernel\tdseed\tsplit\t"
        "R1\tR5\tR10\tmAP\tmINP\n"
    )
    rows = []
    for scenario, mode, strength, kernel in gate_eval.CORE_SCENARIOS:
        if scenario == "rgb_noise_005":
            mode = "ir_noise"
        rows.append(
            f"D\t101\t{scenario}\t{mode}\t{strength}\t{kernel}\t2026\ttest\t"
            "1\t2\t3\t4\t5\n"
        )
    path.write_text(header + "".join(rows), encoding="utf-8", newline="\n")

    with pytest.raises(
        gate_eval.CounterfactualError,
        match="scenario parameters do not match",
    ):
        gate_eval._load_reference_metrics(path, {("D", 101)}, 2026)
