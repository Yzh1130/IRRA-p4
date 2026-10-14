"""Synthetic CPU gradient diagnostics; never read real data or weights."""

import csv
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from model import objectives
from tools.p4 import grad_conflict
from tools.p4.deg_retrieval import DegRetrievalSampler
from tools.p4.grad_conflict import (
    CONDITIONS, MATRIX_NODES, diagnose_batches, encoder_parameter_groups,
    full_state_fingerprint, gradient_gram, gradient_pair_metrics,
    homogeneous_degradation, preserve_global_rng, readonly_model, scalar_summary,
)
from utils.mdrc import _apply_mode


def training_args():
    return SimpleNamespace(loss_names="sdm+mlm+id", use_rgbt_add_fusion=True,
                           use_mum=False, use_global_uncertainty=False, mraf_enabled=False,
                           mdrc_enable=False, use_synth_loss_weighting=False, batch_size=3,
                           num_workers=0, root_dir="synthetic", dev_split_file="synthetic",
                           train_seed=101, deg_rng_seed=None)


def batch():
    return {"pids": torch.tensor([0, 0, 1]), "image_ids": torch.tensor([10, 10, 20]),
            "images": torch.tensor([[1., 2., 2., 1.], [2., 1., 1., 2.], [1., 3., 1., 1.]]).reshape(3, 1, 2, 2),
            "thermal_images": torch.tensor([[2., 3., 1., 1.], [1., 2., 2., 3.], [2., 1., 3., 1.]]).reshape(3, 1, 2, 2),
            "caption_ids": torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]]),
            "mlm_ids": torch.tensor([[0, 2, 3], [2, 0, 4], [3, 4, 0]]),
            "mlm_labels": torch.tensor([[1, 0, 0], [0, 3, 0], [0, 0, 5]])}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.visual = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Dropout(.8))
        self.base_model.transformer = torch.nn.Linear(3, 3)
        self.base_model.token_embedding = torch.nn.Embedding(8, 3)
        self.base_model.positional_embedding = torch.nn.Parameter(torch.ones(1, 3) * .1)
        self.base_model.ln_final = torch.nn.LayerNorm(3)
        self.base_model.text_projection = torch.nn.Parameter(torch.eye(3))
        self.classifier = torch.nn.Linear(3, 2)
        self.logit_scale = torch.tensor(2.)
        self.register_buffer("persistent", torch.tensor([7.]))
        self.register_buffer("nonpersistent", torch.tensor([8.]), persistent=False)
        self.calls = []

    def text(self, captions):
        base = self.base_model
        tokens = base.token_embedding(captions).mean(1) + base.positional_embedding
        return base.ln_final(base.transformer(tokens)) @ base.text_projection

    def forward(self, source, mdrc_feature_only=False, p4_clean_features=None, p4_mode_meta=None):
        # Deliberately exercise every global RNG. The diagnostic must restore all
        # of them, even though the actual eval-mode CLIP dropout does not draw.
        random.random()
        np.random.random()
        torch.rand(1)
        self.calls.append({"feature_only": mdrc_feature_only, "training": self.training,
                           "grad_attached": p4_clean_features is not None and p4_clean_features["v_rgb_c"].requires_grad})
        visual = self.base_model.visual
        if p4_clean_features is None:
            rgb = visual(source["images"].flatten(1))
            ir = visual(source["thermal_images"].flatten(1))
        else:
            rgb, ir = p4_clean_features["v_rgb_c"], p4_clean_features["v_t_c"]
            for modality, key in (("rgb", "images"), ("ir", "thermal_images")):
                indices = p4_mode_meta[modality + "_mask"].nonzero().flatten()
                if indices.numel():
                    changed = visual(source[key].index_select(0, indices).flatten(1))
                    if modality == "rgb":
                        rgb = rgb.index_copy(0, indices, changed)
                    else:
                        ir = ir.index_copy(0, indices, changed)
        ret = {"v_rgb_c": rgb, "v_t_c": ir, "fused_det_feat": rgb + ir}
        if mdrc_feature_only:
            return ret
        text = self.text(source["caption_ids"])
        ret["text_det_feat"] = text
        ret["sdm_loss"] = objectives.compute_sdm(rgb + ir, text, source["pids"], self.logit_scale)
        ret["loss_sdm_prob"] = ret["sdm_loss"]
        ret["id_loss"] = F.cross_entropy(self.classifier(rgb + ir), source["pids"]) + F.cross_entropy(self.classifier(text), source["pids"])
        raw_mlm = (self.text(source["mlm_ids"]) - 1.).square().mean()
        ret["mlm_loss"], ret["loss_mlm"] = .7 * raw_mlm, raw_mlm
        return ret


def rng_snapshot():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state().clone(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "cudnn": (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)}


class GradConflictTest(unittest.TestCase):
    def assert_rng_same(self, left):
        right = rng_snapshot()
        self.assertEqual(left["python"], right["python"])
        self.assertEqual(left["numpy"][0], right["numpy"][0])
        self.assertTrue(np.array_equal(left["numpy"][1], right["numpy"][1]))
        self.assertEqual(left["numpy"][2:], right["numpy"][2:])
        self.assertTrue(torch.equal(left["torch"], right["torch"]))
        self.assertEqual(len(left["cuda"]), len(right["cuda"]))
        for a, b in zip(left["cuda"], right["cuda"]):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(left["cudnn"], right["cudnn"])

    def test_known_concatenated_cosine_norm_and_zero_undefined(self):
        left = [torch.tensor([3., 4.]), None]
        right = [torch.tensor([-4., 3.]), torch.tensor([12.])]
        values = gradient_pair_metrics(left, right)
        self.assertEqual(values["cosine"], 0.)
        self.assertEqual(values["clean_gradient_l2"], 5.)
        self.assertEqual(values["degraded_gradient_l2"], 13.)
        self.assertEqual(values["norm_ratio"], 2.6)
        self.assertEqual(gradient_pair_metrics(left, left)["cosine"], 1.)
        self.assertEqual(gradient_pair_metrics(left, [None, None])["norm_ratio"], 0.)
        self.assertIsNone(gradient_pair_metrics(left, [None, None])["cosine"])
        self.assertIsNone(gradient_pair_metrics([None, None], left)["norm_ratio"])
        torch.testing.assert_close(gradient_gram([left, right]), torch.tensor([[25., 0.], [0., 169.]], dtype=torch.float64))

    def test_homogeneous_matches_training_and_mixed_sampler_rng(self):
        source = batch()
        global_before = rng_snapshot()
        for state in CONDITIONS[:-1]:
            left, meta = homogeneous_degradation(source, state, torch.Generator().manual_seed(23))
            right, other = homogeneous_degradation(source, state, torch.Generator().manual_seed(23))
            self.assertTrue(torch.equal(left["images"], right["images"]))
            self.assertTrue(torch.equal(left["thermal_images"], right["thermal_images"]))
            self.assertEqual(meta["strengths"][0], meta["strengths"][1])
            target_key = "images" if state.startswith("rgb_") else "thermal_images"
            other_key = "thermal_images" if target_key == "images" else "images"
            self.assertIs(left[other_key], source[other_key])
            expected = _apply_mode(source[target_key], torch.ones(3, dtype=torch.bool), state,
                                   generator=torch.Generator().manual_seed(23), factors=meta["strengths"])
            self.assertTrue(torch.equal(left[target_key], expected))
            if state.endswith("_attenuate"):
                self.assertTrue(bool(((meta["strengths"] >= .2) & (meta["strengths"] <= .6)).all()))
            if state.endswith("_drop"):
                self.assertTrue(torch.equal(left[target_key], torch.zeros_like(source[target_key])))
        first = DegRetrievalSampler(seed=108919, branch_prob=1.)
        second = DegRetrievalSampler(seed=108919, branch_prob=1.)
        for a, b in zip(first.sample(source["images"], source["thermal_images"], source["image_ids"])[:2],
                        second.sample(source["images"], source["thermal_images"], source["image_ids"])[:2]):
            self.assertTrue(torch.equal(a, b))
        self.assert_rng_same(global_before)

    def test_end_to_end_groups_matrices_and_all_readonly_state(self):
        model = TinyModel()
        model.train()
        model.base_model.ln_final.eval()
        next(model.parameters()).requires_grad_(False)
        for index, parameter in enumerate(model.parameters()):
            if index % 2 == 0:
                parameter.grad = torch.full_like(parameter, index + 1.)
        saved_grads = [(p, p.grad, p.grad.clone() if p.grad is not None else None, p.requires_grad) for p in model.parameters()]
        modes = [module.training for module in model.modules()]
        buffer_versions = [buffer._version for buffer in model.buffers()]
        grad_versions = [grad._version if grad is not None else None for _, grad, _, _ in saved_grads]
        before, rng_before = full_state_fingerprint(model), rng_snapshot()
        with patch("sys.stdout", new=io.StringIO()):
            result = diagnose_batches(model, [batch(), batch()], training_args(), num_batches=2,
                                      bootstrap_samples=30)
        self.assert_rng_same(rng_before)
        self.assertEqual(full_state_fingerprint(model), before)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertEqual([buffer._version for buffer in model.buffers()], buffer_versions)
        self.assertEqual([p.grad._version if p.grad is not None else None for p, _, _, _ in saved_grads], grad_versions)
        for parameter, grad, values, required in saved_grads:
            self.assertIs(parameter.grad, grad)
            self.assertEqual(parameter.requires_grad, required)
            if values is not None:
                self.assertTrue(torch.equal(parameter.grad, values))
        self.assertTrue(all(not call["training"] for call in model.calls))
        self.assertTrue(all(call["grad_attached"] for call in model.calls if call["feature_only"]))
        self.assertEqual(len(result["batches"]), 2 * 7 * 2 * 2 * 2)
        self.assertEqual(len(result["pairwise_batch_cosines"]), 2 * 2 * 2 * 8 * 8)
        self.assertEqual(result["num_batches_observed"], 2)
        for loss_name in ("sdm", "ce"):
            for encoder in ("image_encoder", "text_encoder"):
                matrix = result["cosine_matrices"][loss_name][encoder]
                self.assertEqual(matrix["nodes"], list(MATRIX_NODES))
                for i in range(8):
                    self.assertAlmostEqual(matrix["statistics"][i][i]["mean"], 1., places=10)
                    for j in range(8):
                        self.assertEqual(matrix["statistics"][i][j], matrix["statistics"][j][i])
        self.assertFalse(any("classifier" in name for names in result["encoder_parameter_names"].values() for name in names))
        expected_text = {name for name, _ in model.named_parameters() if name.startswith("base_model.") and not name.startswith("base_model.visual.")}
        self.assertEqual(set(result["encoder_parameter_names"]["text_encoder"]), expected_text)
        self.assertIn("loss_mlm", result["clean_model_loss_keys"])
        self.assertIn("mlm_loss", result["clean_model_loss_keys"])
        self.assertEqual(result["model_state_sha256_before"], result["model_state_sha256_after"])
        json.dumps(result, allow_nan=False)

    def test_same_batches_degradation_across_different_checkpoint_weights(self):
        first, second = TinyModel(), TinyModel()
        with torch.no_grad():
            for parameter in second.parameters():
                parameter.mul_(1.2)
        outputs = []
        for model in (first, second):
            with patch("sys.stdout", new=io.StringIO()):
                outputs.append(diagnose_batches(model, [batch()], training_args(), num_batches=1, bootstrap_samples=5))
        for result in outputs:
            for item in result["batch_metadata"]:
                del item["seconds"], item["peak_gpu_memory_bytes"]
        self.assertEqual(outputs[0]["batch_metadata"], outputs[1]["batch_metadata"])
        # Both matrices keep the actual training clean SDM objective; their
        # loss names select only the seven degraded node objectives.
        rows = outputs[0]["batches"]
        pairs = outputs[0]["pairwise_batch_cosines"]
        for row in rows:
            if row["baseline"] == "clean_sdm":
                corresponding = next(x for x in pairs if x["loss_name"] == row["degraded_loss_name"] and x["encoder"] == row["encoder"] and x["left"] == "clean" and x["right"] == row["condition"])
                self.assertEqual(row["cosine"], corresponding["cosine"])

    def test_rng_flags_buffers_and_grad_values_restored_on_exception(self):
        model = TinyModel()
        model.train()
        parameter = next(model.parameters())
        parameter.grad = torch.ones_like(parameter)
        grad_ref, grad_value = parameter.grad, parameter.grad.clone()
        original_buffers = [buffer.clone() for buffer in model.buffers()]
        rng_before = rng_snapshot()
        with self.assertRaisesRegex(RuntimeError, "changed model"), preserve_global_rng():
            with readonly_model(model, encoder_parameter_groups(model)):
                model.nonpersistent.add_(1.)
                parameter.grad.add_(3.)
                random.random(), np.random.random(), torch.rand(1)
                if torch.cuda.is_available():
                    torch.rand(1, device="cuda")
                raise ValueError("synthetic failure")
        self.assertIs(parameter.grad, grad_ref)
        self.assertTrue(torch.equal(parameter.grad, grad_value))
        self.assertTrue(model.training)
        for original, buffer in zip(original_buffers, model.buffers()):
            self.assertTrue(torch.equal(original, buffer))
        self.assert_rng_same(rng_before)

    def test_diagnostic_forward_exception_restores_grad_flags_and_modes(self):
        model = TinyModel()
        model.train()
        first = next(model.parameters())
        first.requires_grad_(False)
        first.grad = torch.ones_like(first)
        grad_ref = first.grad
        before, rng_before = full_state_fingerprint(model), rng_snapshot()
        original = model.forward
        def fail_on_degraded(*args, **kwargs):
            if kwargs.get("mdrc_feature_only"):
                random.random(), np.random.random(), torch.rand(1)
                if torch.cuda.is_available():
                    torch.rand(1, device="cuda")
                raise ValueError("synthetic degraded failure")
            return original(*args, **kwargs)
        with patch.object(model, "forward", side_effect=fail_on_degraded), \
             self.assertRaisesRegex(ValueError, "synthetic degraded"):
            diagnose_batches(model, [batch()], training_args(), num_batches=1, bootstrap_samples=5)
        self.assertEqual(full_state_fingerprint(model), before)
        self.assert_rng_same(rng_before)
        self.assertIs(first.grad, grad_ref)
        self.assertTrue(torch.equal(first.grad, torch.ones_like(first)))
        self.assertFalse(first.requires_grad)
        self.assertTrue(model.training)

    def test_summary_bootstrap_is_local_and_excludes_null(self):
        before = rng_snapshot()
        left = scalar_summary([1., 2., 3., None], bootstrap_samples=100, seed=9)
        right = scalar_summary([1., 2., 3., None], bootstrap_samples=100, seed=9)
        self.assertEqual(left, right)
        self.assertEqual(left["count"], 3)
        self.assertEqual(left["excluded"], 1)
        self.assertEqual(left["mean"], 2.)
        self.assertAlmostEqual(left["std"], np.sqrt(2. / 3.))
        self.assertEqual(scalar_summary([None], 5)["ci95"], None)
        self.assert_rng_same(before)

    def test_cli_loader_checkpoint_seed_output_and_whole_main_rng(self):
        source, target = TinyModel(), TinyModel()
        original = batch()
        produced = []
        def batches():
            for index in range(3):
                produced.append(index)
                random.random(), np.random.random(), torch.rand(1)
                yield original
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint, config = directory / "last.pth", directory / "configs.yaml"
            torch.save({"model": source.state_dict(), "epoch": 3}, checkpoint)
            config.write_text("batch_size: 3\n", encoding="utf-8")
            before = rng_snapshot()
            with patch("datasets.build_dataloader", return_value=(batches(), None, None, 2)), \
                 patch("model.build_model", return_value=target), \
                 patch.object(grad_conflict, "_load_training_args", return_value=training_args()), \
                 patch.dict("os.environ", {"TRAIN_SEED": "101"}), \
                 patch("sys.stdout", new=io.StringIO()) as output:
                result = grad_conflict.main(["--checkpoint", str(checkpoint), "--config_file", str(config),
                                             "--output_dir", str(directory / "out"), "--num_batches", "2",
                                             "--bootstrap_samples", "10", "--device", "cpu"])
            self.assert_rng_same(before)
            self.assertEqual(produced, [0, 1])
            self.assertEqual(result["train_seed"], 101)
            self.assertEqual(result["deg_rng_seed"], 108919)
            self.assertEqual(result["checkpoint_epoch"], 3)
            self.assertEqual(result["peak_gpu_memory_bytes"], 0)
            self.assertGreater(result["elapsed_seconds"], 0)
            self.assertIn("peak_gpu_memory_bytes=0", output.getvalue())
            self.assertTrue((directory / "out" / "grad_conflict.json").is_file())
            with (directory / "out" / "batches.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 112)
            with (directory / "out" / "pairwise_cosines.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 512)
            with self.assertRaisesRegex(FileExistsError, "new or empty"):
                grad_conflict.main(["--checkpoint", str(checkpoint), "--config_file", str(config),
                                    "--output_dir", str(directory / "out"), "--device", "cpu"])
            self.assert_rng_same(before)


if __name__ == "__main__":
    unittest.main()
