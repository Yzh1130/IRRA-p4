import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from model.build import IRRA
from model import build as build_module
from model.mraf import (
    ModalityReliabilityAdaptiveFusion,
    compute_mraf_reliability_loss,
    compute_mraf_reliability_targets,
)
from processor.processor import (
    _compose_mdrc_mraf_loss,
    _need_degraded_branch,
)
from tools.revision.analyze_mraf_weights import summarize_alpha
from tools.revision.summarize_mraf_experiment import write_summary
from utils.checkpoint import Checkpointer, load_state_dict
from utils.mdrc import degrade_rgbt_batch
from utils.options import (
    MRAF_PRECISION_FUSION_ERROR,
    get_args,
    validate_mraf_options,
)


def make_args(**overrides):
    values = {
        "mraf_enabled": False,
        "mraf_hidden_dim": 8,
        "mraf_dropout": 0.0,
        "mraf_gate_temperature": 1.0,
        "mraf_gate_fp32": False,
        "mraf_gate_base_grad_scale": 1.0,
        "mraf_rel_lambda": 0.01,
        "mraf_target_temperature": 0.1,
        "mraf_rel_warmup_epoch": 3,
        "mraf_balance_lambda": 0.0,
        "mdrc_enable": False,
        "use_global_uncertainty": False,
        "use_precision_fusion": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ToyConditionalFusionModel(nn.Module):
    def __init__(
        self,
        enabled,
        feature_dim=4,
        gate_fp32=False,
        gate_base_grad_scale=1.0,
    ):
        super().__init__()
        self.projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.mraf_enabled = bool(enabled)
        if self.mraf_enabled:
            self.mraf_fusion = ModalityReliabilityAdaptiveFusion(
                feature_dim=feature_dim,
                hidden_dim=8,
                dropout=0.0,
                gate_temperature=1.0,
                gate_fp32=gate_fp32,
                gate_base_grad_scale=gate_base_grad_scale,
            )

    def forward(self, rgb, ir):
        rgb = self.projection(rgb)
        ir = self.projection(ir)
        if self.mraf_enabled:
            return self.mraf_fusion(rgb, ir)
        return rgb + ir


class TestMRAF(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)

    def test_zero_initialization_and_baseline_equivalence(self):
        module = ModalityReliabilityAdaptiveFusion(7, hidden_dim=5, dropout=0.4)
        module.train()
        rgb = torch.randn(6, 7)
        ir = torch.randn(6, 7)
        fused, aux = module(rgb, ir, return_aux=True)
        self.assertTrue(torch.equal(aux["alpha_rgb"], torch.full((6, 1), 0.5)))
        self.assertTrue(torch.equal(aux["alpha_ir"], torch.full((6, 1), 0.5)))
        self.assertTrue(torch.equal(fused, rgb + ir))
        self.assertTrue(torch.equal(module.output_layer.weight, torch.zeros_like(module.output_layer.weight)))
        self.assertTrue(torch.equal(module.output_layer.bias, torch.zeros_like(module.output_layer.bias)))

    def test_weight_normalization_and_range(self):
        module = ModalityReliabilityAdaptiveFusion(4, hidden_dim=6, dropout=0.0)
        with torch.no_grad():
            module.output_layer.weight.normal_()
            module.output_layer.bias.fill_(0.1)
        _, aux = module(torch.randn(9, 4), torch.randn(9, 4), return_aux=True)
        alpha = aux["alpha"]
        torch.testing.assert_close(alpha.sum(dim=1), torch.ones(9))
        self.assertTrue(bool((alpha >= 0.0).all()))
        self.assertTrue(bool((alpha <= 1.0).all()))

    def test_reliability_target_direction(self):
        clean_rgb = torch.tensor([[1.0, 0.0]])
        clean_ir = torch.tensor([[0.0, 1.0]])
        rgb_bad = torch.tensor([[-1.0, 0.0]])
        ir_good = clean_ir.clone()
        outputs = compute_mraf_reliability_targets(
            clean_rgb, clean_ir, rgb_bad, ir_good, target_temperature=0.1
        )
        self.assertGreater(outputs["target_ir"].item(), outputs["target_rgb"].item())

        outputs = compute_mraf_reliability_targets(
            clean_rgb,
            clean_ir,
            clean_rgb.clone(),
            torch.tensor([[0.0, -1.0]]),
            target_temperature=0.1,
        )
        self.assertGreater(outputs["target_rgb"].item(), outputs["target_ir"].item())

    def test_stop_gradient_and_scorer_gradient(self):
        module = ModalityReliabilityAdaptiveFusion(3, hidden_dim=5, dropout=0.0)
        clean_rgb = torch.randn(4, 3, requires_grad=True)
        clean_ir = torch.randn(4, 3, requires_grad=True)
        degraded_rgb = torch.randn(4, 3, requires_grad=True)
        degraded_ir = torch.randn(4, 3, requires_grad=True)
        targets = compute_mraf_reliability_targets(
            clean_rgb,
            clean_ir,
            degraded_rgb,
            degraded_ir,
            target_temperature=0.1,
        )
        self.assertFalse(targets["target"].requires_grad)
        _, aux = module(degraded_rgb, degraded_ir, return_aux=True)
        loss = compute_mraf_reliability_loss(aux["alpha"], targets["target"])
        loss.backward()
        scorer_grad = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in module.scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(scorer_grad, 0.0)
        self.assertIsNone(clean_rgb.grad)
        self.assertIsNone(clean_ir.grad)

    def test_mode_matrix_and_degraded_branch_contract(self):
        cases = (
            ("base", False, False, 0.01, False),
            ("mdrc", False, True, 0.01, True),
            ("mraf_only", True, False, 0.01, True),
            ("mdrc_mraf", True, True, 0.01, True),
            ("mraf_norel", True, False, 0.0, False),
            ("mdrc_mraf_norel", True, True, 0.0, True),
        )
        rgb = torch.randn(3, 4)
        ir = torch.randn(3, 4)
        for name, mraf, mdrc, rel_lambda, expected_branch in cases:
            with self.subTest(name=name):
                args = make_args(
                    mraf_enabled=mraf,
                    mdrc_enable=mdrc,
                    mraf_rel_lambda=rel_lambda,
                )
                self.assertEqual(_need_degraded_branch(args), expected_branch)
                model = ToyConditionalFusionModel(mraf)
                output = model(rgb, ir)
                self.assertEqual(tuple(output.shape), (3, 4))
                if not mraf:
                    expected = model.projection(rgb) + model.projection(ir)
                    torch.testing.assert_close(output, expected)

    def test_irra_global_fusion_switch_preserves_disabled_path(self):
        def build_shell(enabled):
            model = IRRA.__new__(IRRA)
            nn.Module.__init__(model)
            model.mraf_enabled = enabled
            model.use_rgbt_add_fusion = True
            model.use_global_uncertainty = False
            model.uncertainty_dist_type = "diag_gaussian"
            if enabled:
                model.mraf_fusion = ModalityReliabilityAdaptiveFusion(
                    4, hidden_dim=6, dropout=0.0
                )
            return model

        rgb = torch.randn(5, 4)
        ir = torch.randn(5, 4)
        logvar = torch.zeros_like(rgb)
        disabled_fused, _ = build_shell(False)._fuse_rgbt_global(
            rgb, logvar, ir, logvar, thermal_available=True
        )
        enabled_fused, _, aux = build_shell(True)._fuse_rgbt_global(
            rgb,
            logvar,
            ir,
            logvar,
            thermal_available=True,
            return_mraf_aux=True,
        )
        self.assertTrue(torch.equal(disabled_fused, rgb + ir))
        self.assertTrue(torch.equal(enabled_fused, rgb + ir))
        self.assertTrue(torch.equal(aux["alpha"], torch.full((5, 2), 0.5)))

    def test_uncertainty_precision_fusion_is_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, MRAF_PRECISION_FUSION_ERROR):
            validate_mraf_options(
                make_args(mraf_enabled=True, use_precision_fusion=True)
            )
        with self.assertRaisesRegex(ValueError, MRAF_PRECISION_FUSION_ERROR):
            validate_mraf_options(
                make_args(
                    mraf_enabled=True,
                    use_global_uncertainty=True,
                    use_precision_fusion=None,
                )
            )
        validate_mraf_options(
            make_args(
                mraf_enabled=True,
                use_global_uncertainty=True,
                use_precision_fusion=False,
            )
        )

    def test_conditional_state_dict_keys(self):
        disabled_keys = ToyConditionalFusionModel(False).state_dict().keys()
        enabled_keys = ToyConditionalFusionModel(True).state_dict().keys()
        self.assertFalse(any("mraf" in key for key in disabled_keys))
        self.assertTrue(any(key.startswith("mraf_fusion.") for key in enabled_keys))

    def test_new_checkpoint_roundtrip_is_strict_and_output_stable(self):
        source = ToyConditionalFusionModel(True).eval()
        with torch.no_grad():
            source.mraf_fusion.output_layer.weight.normal_()
            source.mraf_fusion.output_layer.bias.normal_()
        rgb = torch.randn(5, 4)
        ir = torch.randn(5, 4)
        expected = source(rgb, ir)
        with tempfile.TemporaryDirectory() as temp_dir:
            Checkpointer(
                source,
                save_dir=temp_dir,
                save_to_disk=True,
            ).save("mraf")
            target = ToyConditionalFusionModel(True).eval()
            Checkpointer(target).load(str(Path(temp_dir) / "mraf.pth"))
            torch.testing.assert_close(target(rgb, ir), expected)

    def test_old_checkpoint_compatibility_only_allows_mraf_missing_keys(self):
        old_model = ToyConditionalFusionModel(False)
        enabled_model = ToyConditionalFusionModel(True)
        with self.assertLogs("PersonSearch.checkpoint", level="WARNING") as logs:
            load_state_dict(enabled_model, copy.deepcopy(old_model.state_dict()))
        self.assertIn("Missing MRAF keys", "\n".join(logs.output))
        self.assertTrue(
            torch.equal(
                enabled_model.mraf_fusion.output_layer.weight,
                torch.zeros_like(enabled_model.mraf_fusion.output_layer.weight),
            )
        )

        unexpected = copy.deepcopy(old_model.state_dict())
        unexpected["unexpected.weight"] = torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "only permits missing MRAF"):
            load_state_dict(ToyConditionalFusionModel(True), unexpected)

        missing_common = copy.deepcopy(old_model.state_dict())
        missing_common.pop("projection.weight")
        with self.assertRaisesRegex(RuntimeError, "Non-MRAF missing keys"):
            load_state_dict(ToyConditionalFusionModel(True), missing_common)

    def test_mdrc_return_mode_preserves_outputs_and_rng_state(self):
        rgb = torch.randn(12, 3, 4, 4)
        ir = torch.randn(12, 3, 4, 4)
        modes = ["rgb_drop", "ir_drop", "rgb_noise", "ir_noise"]

        torch.manual_seed(77)
        np.random.seed(77)
        random.seed(77)
        old_rgb, old_ir = degrade_rgbt_batch(rgb, ir, prob=0.8, modes=modes)
        old_torch_next = torch.rand(5)
        old_numpy_next = np.random.rand(5)
        old_python_next = [random.random() for _ in range(5)]

        torch.manual_seed(77)
        np.random.seed(77)
        random.seed(77)
        new_rgb, new_ir, mode_info = degrade_rgbt_batch(
            rgb,
            ir,
            prob=0.8,
            modes=modes,
            return_mode=True,
        )
        new_torch_next = torch.rand(5)
        new_numpy_next = np.random.rand(5)
        new_python_next = [random.random() for _ in range(5)]

        self.assertTrue(torch.equal(old_rgb, new_rgb))
        self.assertTrue(torch.equal(old_ir, new_ir))
        self.assertEqual(mode_info["mode_indices"].shape[0], rgb.shape[0])
        self.assertEqual(sum(mode_info["mode_counts"].values()), rgb.shape[0])
        self.assertTrue(torch.equal(old_torch_next, new_torch_next))
        np.testing.assert_array_equal(old_numpy_next, new_numpy_next)
        self.assertEqual(old_python_next, new_python_next)

    def test_mdrc_scale_does_not_scale_mraf_loss(self):
        base = torch.tensor(1.0)
        mdrc = torch.tensor(2.0)
        mraf = torch.tensor(3.0)
        result = _compose_mdrc_mraf_loss(
            base,
            mdrc,
            lambda_mdrc=0.5,
            mdrc_loss_scale=10.0,
            mraf_rel_loss=mraf,
            lambda_mraf_rel=0.25,
        )
        torch.testing.assert_close(result, torch.tensor(11.75))
        result_larger_scale = _compose_mdrc_mraf_loss(
            base,
            mdrc,
            lambda_mdrc=0.5,
            mdrc_loss_scale=20.0,
            mraf_rel_loss=mraf,
            lambda_mraf_rel=0.25,
        )
        torch.testing.assert_close(result_larger_scale - result, torch.tensor(10.0))

    def test_device_dtype_and_input_validation(self):
        module = ModalityReliabilityAdaptiveFusion(4, hidden_dim=5, dropout=0.0)
        rgb = torch.randn(3, 4, dtype=torch.float64)
        ir = torch.randn(3, 4, dtype=torch.float64)
        output = module(rgb, ir)
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(output.device, rgb.device)
        with self.assertRaisesRegex(ValueError, "shape"):
            module(rgb.unsqueeze(0), ir.unsqueeze(0))
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            bad = rgb.clone()
            bad[0, 0] = float("nan")
            module(bad, ir)
        with self.assertRaisesRegex(ValueError, "gate_temperature"):
            ModalityReliabilityAdaptiveFusion(4, gate_temperature=0.0)
        for invalid_scale in (-0.01, 1.01):
            with self.assertRaisesRegex(ValueError, "gate_base_grad_scale"):
                ModalityReliabilityAdaptiveFusion(
                    4, gate_base_grad_scale=invalid_scale
                )
            with self.assertRaisesRegex(ValueError, "mraf_gate_base_grad_scale"):
                validate_mraf_options(
                    make_args(mraf_gate_base_grad_scale=invalid_scale)
                )

    def test_gate_fp32_survives_half_and_autocast(self):
        module = ModalityReliabilityAdaptiveFusion(
            4,
            hidden_dim=6,
            dropout=0.0,
            gate_fp32=True,
        ).half()
        with torch.no_grad():
            module.output_layer.weight.normal_(std=0.2)
            module.output_layer.bias.copy_(torch.tensor([0.03]))
        self.assertTrue(
            all(parameter.dtype == torch.float32 for parameter in module.scorer.parameters())
        )
        rgb = torch.randn(7, 4, dtype=torch.float16)
        ir = torch.randn(7, 4, dtype=torch.float16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            fused, aux = module(rgb, ir, return_aux=True)
        self.assertEqual(aux["logits"].dtype, torch.float32)
        self.assertEqual(aux["alpha"].dtype, torch.float32)
        self.assertEqual(fused.dtype, torch.float32)

        with torch.autocast(device_type="cpu", enabled=False):
            rgb32 = rgb.float()
            ir32 = ir.float()
            logits_ref = torch.cat(
                [
                    module.scorer(torch.nn.functional.normalize(rgb32, dim=-1)),
                    module.scorer(torch.nn.functional.normalize(ir32, dim=-1)),
                ],
                dim=-1,
            )
            alpha_ref = torch.softmax(
                logits_ref / module.gate_temperature, dim=-1
            )
            fused_ref = 2.0 * (
                alpha_ref[:, :1] * rgb32 + alpha_ref[:, 1:] * ir32
            )
        torch.testing.assert_close(aux["alpha"], alpha_ref, atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(fused, fused_ref, atol=1e-7, rtol=1e-7)

    def test_cli_defaults_and_build_conversion_order(self):
        with patch("sys.argv", ["train.py"]):
            args = get_args()
        self.assertFalse(args.mraf_gate_fp32)
        self.assertEqual(args.mraf_gate_base_grad_scale, 1.0)

        shell = nn.Module()
        shell.mraf_enabled = True
        shell.mraf_fusion = ModalityReliabilityAdaptiveFusion(
            4, hidden_dim=3, gate_fp32=True
        )

        def convert_to_half(model):
            model.mraf_fusion.scorer.half()

        with patch.object(build_module, "IRRA", return_value=shell), patch.object(
            build_module, "convert_weights", side_effect=convert_to_half
        ):
            built = build_module.build_model(SimpleNamespace(), num_classes=2)
        self.assertIs(built, shell)
        self.assertTrue(
            all(
                parameter.dtype == torch.float32
                for parameter in built.mraf_fusion.scorer.parameters()
            )
        )

    def test_gate_fp32_disabled_preserves_legacy_dtype_path(self):
        module = ModalityReliabilityAdaptiveFusion(
            4, hidden_dim=6, dropout=0.0, gate_fp32=False
        ).half()
        rgb = torch.randn(5, 4, dtype=torch.float16)
        ir = torch.randn(5, 4, dtype=torch.float16)
        fused, aux = module(rgb, ir, return_aux=True)
        self.assertEqual(aux["logits"].dtype, torch.float16)
        self.assertEqual(aux["alpha"].dtype, torch.float16)
        self.assertEqual(fused.dtype, torch.float16)
        self.assertTrue(torch.equal(fused, rgb + ir))

    def test_base_gate_gradient_scaling_and_forward_invariance(self):
        template = ModalityReliabilityAdaptiveFusion(
            4, hidden_dim=6, dropout=0.0, gate_fp32=True
        )
        with torch.no_grad():
            template.output_layer.weight.normal_(std=0.2)
            template.output_layer.bias.normal_(std=0.1)
        state = copy.deepcopy(template.state_dict())
        rgb_seed = torch.randn(8, 4)
        ir_seed = torch.randn(8, 4)
        outputs = {}
        scorer_grad_norms = {}
        input_grad_norms = {}
        for scale in (0.0, 0.1, 1.0):
            module = ModalityReliabilityAdaptiveFusion(
                4,
                hidden_dim=6,
                dropout=0.0,
                gate_fp32=True,
                gate_base_grad_scale=scale,
            )
            module.load_state_dict(state)
            rgb = rgb_seed.clone().requires_grad_(True)
            ir = ir_seed.clone().requires_grad_(True)
            fused = module(rgb, ir)
            loss = fused.square().mean()
            loss.backward()
            outputs[scale] = fused.detach()
            scorer_grad_norms[scale] = sum(
                float(parameter.grad.norm().item())
                for parameter in module.scorer.parameters()
                if parameter.grad is not None
            )
            input_grad_norms[scale] = float(rgb.grad.norm().item())
        torch.testing.assert_close(outputs[0.0], outputs[0.1], atol=0, rtol=0)
        torch.testing.assert_close(outputs[0.0], outputs[1.0], atol=0, rtol=0)
        self.assertEqual(scorer_grad_norms[0.0], 0.0)
        self.assertGreater(scorer_grad_norms[1.0], 0.0)
        self.assertAlmostEqual(
            scorer_grad_norms[0.1] / scorer_grad_norms[1.0],
            0.1,
            places=5,
        )
        self.assertGreater(input_grad_norms[0.0], 0.0)

    def test_reliability_gradient_ignores_base_gradient_scale(self):
        module = ModalityReliabilityAdaptiveFusion(
            3,
            hidden_dim=5,
            dropout=0.0,
            gate_fp32=True,
            gate_base_grad_scale=0.0,
        )
        with torch.no_grad():
            module.output_layer.weight.normal_(std=0.1)
        rgb = torch.randn(6, 3)
        ir = torch.randn(6, 3)
        _, aux = module(rgb, ir, return_aux=True)
        target = torch.tensor([[0.8, 0.2]], dtype=torch.float32).repeat(6, 1)
        compute_mraf_reliability_loss(aux["alpha"], target).backward()
        grad_norm = sum(
            float(parameter.grad.norm().item())
            for parameter in module.scorer.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_fp32_checkpoint_roundtrip_and_eval_scale_invariance(self):
        source = ToyConditionalFusionModel(
            True, gate_fp32=True, gate_base_grad_scale=0.0
        ).eval()
        with torch.no_grad():
            source.mraf_fusion.output_layer.weight.normal_(std=0.1)
        rgb = torch.randn(4, 4)
        ir = torch.randn(4, 4)
        expected = source(rgb, ir)
        target = ToyConditionalFusionModel(
            True, gate_fp32=True, gate_base_grad_scale=1.0
        ).eval()
        target.load_state_dict(source.state_dict())
        actual = target(rgb, ir)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertTrue(
            all(
                parameter.dtype == torch.float32
                for parameter in target.mraf_fusion.scorer.parameters()
            )
        )

    def test_analysis_and_summary_helpers_do_not_invent_values(self):
        row = summarize_alpha(
            "clean",
            torch.tensor([[0.4], [0.6]]),
            torch.tensor([[0.6], [0.4]]),
        )
        self.assertEqual(row["num_samples"], 2)
        self.assertAlmostEqual(row["max_weight_sum_error"], 0.0)
        run = {
            "method": "mraf_only",
            "train_seed": 1,
            "checkpoint": None,
            "best_epoch": None,
            "clean_r1": None,
            "clean_r5": None,
            "clean_r10": None,
            "clean_map": None,
            "clean_minp": None,
            "mean_abs_delta_r1": None,
            "mean_degraded_r1": None,
            "rgb_missing_r1": None,
            "ir_missing_r1": None,
            "rgb_noise_010_r1": None,
            "ir_noise_010_r1": None,
            "both_blur_k7_r1": None,
            "parameter_count": None,
            "inference_time": None,
            "test_elapsed_time": None,
            "alpha_statistics_file": None,
            "git_head": None,
            "config": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, csv_path, markdown_path = write_summary(
                [run], Path(temp_dir)
            )
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertIn("NA", markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
