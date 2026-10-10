"""CPU tests for P4 C0. The legacy oracle is read from Git, never written.

Run: python -B -m unittest tests.p4.test_deg_retrieval -v
"""
import ast
import copy
import math
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from model import build as build_module, objectives
from model.clip_model import convert_weights
from processor.processor import _build_total_loss
from tools.p4.deg_retrieval import (
    DegRetrievalSampler, EpochCounters, batch_label_statistics, compute_deg_loss,
    pid_targets, resolve_deg_rng_seed,
)
from utils.options import get_args


LEGACY_COMMIT = "2889f40c5008d00539cc65df78013ecd62f794d6"
REPO = Path(__file__).resolve().parents[3]


class TinyCLIP(nn.Module):
    """Synthetic backbone; actual IRRA SDM/ID/MLM heads still execute.

    Dropout deliberately advances the global RNG, making T-none sensitive to
    an extra degradation draw. Dimensions: 4x4 RGB images, four text tokens,
    64 embedding channels and a 16-word vocabulary.
    """
    def __init__(self):
        super().__init__()
        self.image_projection = nn.Linear(3 * 4 * 4, 64)
        self.token_embedding = nn.Embedding(16, 64)
        self.text_projection_layer = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.2)
        self.image_calls = []
        self.text_calls = 0

    def encode_image(self, image, modality="rgb", return_adapter_aux=False):
        self.image_calls.append((modality, int(image.shape[0])))
        pooled = self.image_projection(image.flatten(1).to(self.image_projection.weight.dtype))
        tokens = self.dropout(torch.stack((pooled, pooled * 0.5, pooled * 0.25), dim=1))
        return (tokens, None) if return_adapter_aux else tokens

    def encode_text(self, token_ids):
        self.text_calls += 1
        embeddings = self.token_embedding(token_ids).to(self.text_projection_layer.weight.dtype)
        return self.dropout(self.text_projection_layer(embeddings))


def tiny_builder(*args, **kwargs):
    return TinyCLIP(), {"embed_dim": 64}


def tiny_args(**changes):
    with patch.object(sys, "argv", ["train.py"]):
        args = get_args()
    for key, value in {
        "pretrain_choice": "synthetic", "img_size": (4, 4), "stride_size": 2,
        "text_length": 4, "vocab_size": 16, "cmt_depth": 1,
        "temperature": 0.5, "loss_names": "sdm+id+mlm", "MLM": True,
        "deg_retrieval_loss": "none", "use_mum": False,
        "use_global_uncertainty": False, "use_precision_fusion": False,
        "use_rgbt_add_fusion": True, "mraf_enabled": False,
        "mraf_rel_lambda": 0.0, "mdrc_enable": False,
    }.items():
        setattr(args, key, value)
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def tiny_batch():
    generator = torch.Generator(device="cpu").manual_seed(991)
    return {
        "images": torch.randn(3, 3, 4, 4, generator=generator),
        "thermal_images": torch.randn(3, 3, 4, 4, generator=generator),
        "pids": torch.tensor([0, 0, 1]), "image_ids": torch.tensor([7, 7, 8]),
        "caption_ids": torch.tensor([[14, 2, 3, 15], [14, 4, 5, 15], [14, 6, 7, 15]]),
        "mlm_ids": torch.tensor([[14, 13, 3, 15], [14, 4, 13, 15], [14, 13, 7, 15]]),
        "mlm_labels": torch.tensor([[0, 2, 0, 0], [0, 0, 5, 0], [0, 6, 0, 0]]),
    }


def _legacy_oracle():
    """Load the complete legacy IRRA class and original total-loss function.

    git show is read-only. Relative model imports resolve from package=model;
    no worktree, legacy source file or bytecode is created.
    """
    model_source = subprocess.check_output(
        ["git", "show", f"{LEGACY_COMMIT}:IRRA-main/model/build.py"], cwd=REPO,
        encoding="utf-8")
    processor_source = subprocess.check_output(
        ["git", "show", f"{LEGACY_COMMIT}:IRRA-main/processor/processor.py"], cwd=REPO,
        encoding="utf-8")
    module = types.ModuleType("model._p4_memory_legacy")
    module.__package__ = "model"
    exec(compile(model_source, "<git-2889f40c-model-build>", "exec"), module.__dict__)
    module.build_CLIP_from_openai_pretrained = tiny_builder
    fn = next(node for node in ast.parse(processor_source).body
              if isinstance(node, ast.FunctionDef) and node.name == "_build_total_loss")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<git-2889f40c-total-loss>", "exec"), namespace)
    return module.IRRA, namespace["_build_total_loss"]


class TestP4DegradedRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(1234)

    def test_none_two_steps_bitwise_legacy_loss_parameters_rng(self):
        legacy_class, legacy_total = _legacy_oracle()
        args = tiny_args()
        torch.manual_seed(100)
        legacy = legacy_class(copy.deepcopy(args), num_classes=2).train()
        torch.manual_seed(100)
        with patch.object(build_module, "build_CLIP_from_openai_pretrained", tiny_builder):
            current = build_module.IRRA(copy.deepcopy(args), num_classes=2).train()
        self.assertEqual(set(legacy.state_dict()), set(current.state_dict()))
        for key in legacy.state_dict():
            self.assertTrue(torch.equal(legacy.state_dict()[key], current.state_dict()[key]), key)
        batch = tiny_batch()

        def run(model, total_fn, use_current):
            torch.manual_seed(712)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
            sampler = DegRetrievalSampler("none", seed=2026)
            trace = []
            for _ in range(2):
                if use_current:
                    model._p4_encoder_call_counts = {"rgb": 0, "ir": 0}
                ret = model(batch)
                if use_current:
                    model._p4_encoder_call_counts = None
                    before = torch.get_rng_state().clone()
                    rgb, ir, meta = sampler.sample(batch["images"], batch["thermal_images"], batch["image_ids"])
                    self.assertIs(rgb, batch["images"])
                    self.assertIs(ir, batch["thermal_images"])
                    self.assertFalse(meta["enabled"])
                    self.assertIsNone(sampler.generator)
                    self.assertTrue(torch.equal(before, torch.get_rng_state()))
                loss = total_fn(args, ret, epoch=1)
                # Preserve the A alias behavior explicitly: raw MLM counts twice.
                self.assertTrue(torch.equal(loss, ret["sdm_loss"] + ret["id_loss"]
                                            + ret["mlm_loss"] + ret["loss_mlm"]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                trace.append((loss.detach().clone(),
                              {key: value.detach().clone() for key, value in model.state_dict().items()},
                              torch.get_rng_state().clone()))
            return trace

        old_trace = run(legacy, legacy_total, False)
        new_trace = run(current, _build_total_loss, True)
        for step, (old, new) in enumerate(zip(old_trace, new_trace), 1):
            self.assertTrue(torch.equal(old[0], new[0]), f"step {step} loss")
            self.assertTrue(torch.equal(old[2], new[2]), f"step {step} RNG")
            for key in old[1]:
                self.assertTrue(torch.equal(old[1][key], new[1][key]), f"step {step} {key}")

    def test_ce_hand_computed_three_by_three_and_single_positive(self):
        images = torch.eye(3, dtype=torch.float64)
        texts = torch.tensor([[1., 0., 0.], [0., 1., 0.],
                              [math.sqrt(0.5), 0., math.sqrt(0.5)]], dtype=torch.float64)
        pids = torch.tensor([1, 1, 2])
        # Explicit t2i logits (scale=1), target q and hand-calculated row losses.
        logits = torch.tensor([[1., 0., 0.], [0., 1., 0.],
                               [math.sqrt(0.5), 0., math.sqrt(0.5)]], dtype=torch.float64)
        q = torch.tensor([[0.5, 0.5, 0.], [0.5, 0.5, 0.], [0., 0., 1.]], dtype=torch.float64)
        t2i_rows = torch.tensor([1.0514447139320509, 1.0514447139320509,
                                 0.9135143578703041], dtype=torch.float64)
        i2t_rows = torch.tensor([1.2485730166666686, 1.0514447139320509,
                                 0.6861917388550699], dtype=torch.float64)
        t2i_mean, i2t_mean, total = 1.0054679285781354, 0.9954031564845964, 2.000871085062732
        torch.testing.assert_close(-(q * F.log_softmax(logits, dim=1)).sum(1), t2i_rows, rtol=0, atol=1e-14)
        torch.testing.assert_close(-(q * F.log_softmax(logits.t(), dim=1)).sum(1), i2t_rows, rtol=0, atol=1e-14)
        self.assertAlmostEqual(t2i_rows.mean().item(), t2i_mean, places=14)
        self.assertAlmostEqual(i2t_rows.mean().item(), i2t_mean, places=14)
        self.assertAlmostEqual(objectives.compute_multipos_ce(images, texts, pids, 1., 1e-8).item(), total, places=13)
        ordinary = F.cross_entropy(logits, torch.arange(3)) + F.cross_entropy(logits.t(), torch.arange(3))
        torch.testing.assert_close(objectives.compute_multipos_ce(images, texts, torch.arange(3), 1.),
                                   ordinary, rtol=0, atol=1e-14)

    def test_sdm_dispatch_is_existing_function_and_same_output(self):
        generator = torch.Generator().manual_seed(41)
        images = torch.randn(3, 7, generator=generator)
        texts = torch.randn(3, 7, generator=generator)
        pids = torch.tensor([1, 1, 2])
        expected = objectives.compute_sdm(images, texts, pids, 50.)
        with patch.object(objectives, "compute_sdm", wraps=objectives.compute_sdm) as spy:
            actual = compute_deg_loss(images, texts, pids, 50., "sdm")
        self.assertEqual(spy.call_count, 1)
        self.assertNotIn("image_id", spy.call_args.kwargs)
        self.assertNotIn("factor", spy.call_args.kwargs)
        self.assertTrue(torch.equal(actual, expected))

    def test_share_image_states_independent_noise_private_rng_and_seeds(self):
        rgb = torch.ones(256, 3, 4, 4)
        ir = torch.ones_like(rgb)
        ids = torch.arange(128).repeat_interleave(2)
        before = torch.get_rng_state().clone()
        first_sampler = DegRetrievalSampler("sdm", seed=2026)
        first = first_sampler.sample(rgb, ir, ids)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        a, b, meta = first
        self.assertTrue(torch.equal(meta["mode_indices"][::2], meta["mode_indices"][1::2]))
        self.assertTrue(torch.equal(meta["strengths"][::2], meta["strengths"][1::2]))
        noise_pairs = 0
        for row in range(0, len(ids), 2):
            mode = int(meta["mode_indices"][row])
            if mode in (2, 3):
                noise_pairs += 1
                changed = a if mode == 2 else b
                self.assertFalse(torch.equal(changed[row], changed[row + 1]))
        self.assertGreater(noise_pairs, 0)
        self.assertTrue(torch.equal(rgb, torch.ones_like(rgb)))
        same = DegRetrievalSampler("sdm", seed=2026).sample(rgb, ir, ids)
        for index in (0, 1):
            self.assertTrue(torch.equal(first[index], same[index]))
        self.assertTrue(torch.equal(meta["mode_indices"], same[2]["mode_indices"]))
        different = DegRetrievalSampler("sdm", seed=2027).sample(rgb, ir, ids)
        self.assertFalse(torch.equal(meta["mode_indices"], different[2]["mode_indices"]))
        generator_identity = id(first_sampler.generator)
        second = first_sampler.sample(rgb, ir, ids)
        self.assertEqual(id(first_sampler.generator), generator_identity)
        self.assertFalse(torch.equal(meta["mode_indices"], second[2]["mode_indices"]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_labels_and_counters_match_sdm_for_repeated_images_and_texts(self):
        examples = [([1, 1, 2], [7, 7, 8]), ([1, 1, 1, 2], [7, 8, 8, 9]),
                    ([1, 2, 3], [7, 8, 9])]
        real_log = torch.log
        for pid_values, image_values in examples:
            with self.subTest(pids=pid_values, images=image_values):
                pids = torch.tensor(pid_values)
                ids = torch.tensor(image_values)
                q = pid_targets(pids)
                observed = []

                def capture(value):
                    observed.append(value.detach().clone())
                    return real_log(value)

                with patch.object(objectives.torch, "log", side_effect=capture):
                    objectives.compute_sdm(torch.eye(len(pids)), torch.eye(len(pids)), pids, 1.)
                self.assertEqual(len(observed), 2)
                for direction_q in observed:
                    self.assertTrue(torch.equal(direction_q, q + 1e-8))
                stats = batch_label_statistics(pids, ids)
                self.assertEqual(stats["t2i"], stats["i2t"])
                counts = (q > 0).sum(1).float()
                self.assertEqual(stats["t2i"]["positive_columns"]["mean"], counts.sum().item() / len(pids))
                self.assertEqual(stats["t2i"]["negative_columns"]["mean"], (len(pids) - counts).sum().item() / len(pids))
                epoch = EpochCounters()
                epoch.add_batch(pids, ids)
                summary = epoch.summary(1, 0.1, 0)
                self.assertEqual(summary["text_samples"], len(pids))
                self.assertEqual(summary["unique_images"], len(set(image_values)))
                self.assertEqual(summary["label_statistics"], stats)

    def test_finite_manual_fp16_encoder_float_features_without_autocast(self):
        for loss_name in ("sdm", "ce"):
            with self.subTest(loss=loss_name):
                args = tiny_args(deg_retrieval_loss=loss_name)
                with patch.object(build_module, "build_CLIP_from_openai_pretrained", tiny_builder):
                    model = build_module.IRRA(args, num_classes=2).train()
                # This is the repository's A precision contract: convert_weights
                # halves Conv/Linear/attention; pooled global features call float.
                # No autocast/GradScaler exists in A, and none is introduced here.
                convert_weights(model)
                self.assertEqual(model.base_model.image_projection.weight.dtype, torch.float16)
                batch = tiny_batch()
                clean = model(batch)
                self.assertEqual(clean["fused_det_feat"].dtype, torch.float32)
                self.assertEqual(clean["text_det_feat"].dtype, torch.float32)
                rgb, ir, meta = DegRetrievalSampler(loss_name, seed=2026).sample(
                    batch["images"], batch["thermal_images"], batch["image_ids"])
                deg = model(dict(batch, images=rgb, thermal_images=ir), mdrc_feature_only=True,
                            p4_clean_features=clean, p4_mode_meta=meta)
                loss = _build_total_loss(args, clean) + compute_deg_loss(
                    deg["fused_det_feat"], clean["text_det_feat"], batch["pids"],
                    model.logit_scale, loss_name)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                gradients = [param.grad for param in model.parameters() if param.grad is not None]
                self.assertGreater(len(gradients), 0)
                self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))

    def test_selective_encoding_reuses_clean_features_with_gradient(self):
        args = tiny_args()
        with patch.object(build_module, "build_CLIP_from_openai_pretrained", tiny_builder):
            model = build_module.IRRA(args, num_classes=2).train()
        batch = tiny_batch()
        clean = model(batch)
        clean["v_rgb_c"].retain_grad()
        clean["v_t_c"].retain_grad()
        model.base_model.image_calls.clear()
        text_calls = model.base_model.text_calls
        rgb_mask = torch.tensor([True, False, True])
        ir_mask = ~rgb_mask
        meta = {"enabled": True, "rgb_mask": rgb_mask, "ir_mask": ir_mask}
        deg = model(batch, mdrc_feature_only=True, p4_clean_features=clean, p4_mode_meta=meta)
        self.assertEqual(model.base_model.image_calls, [("rgb", 2), ("thermal", 1)])
        self.assertEqual(model.base_model.text_calls, text_calls)
        self.assertEqual(deg["p4_encoder_calls"], {"rgb": 1, "ir": 1})
        self.assertFalse(any("loss" in key for key in deg))
        self.assertTrue(torch.equal(deg["v_t_c"][rgb_mask], clean["v_t_c"][rgb_mask]))
        self.assertTrue(torch.equal(deg["v_rgb_c"][ir_mask], clean["v_rgb_c"][ir_mask]))
        deg["fused_det_feat"].sum().backward()
        self.assertTrue(torch.equal(clean["v_t_c"].grad[rgb_mask], torch.ones(2, 64)))
        self.assertTrue(torch.equal(clean["v_rgb_c"].grad[ir_mask], torch.ones(1, 64)))
        self.assertTrue(torch.equal(clean["v_t_c"].grad[ir_mask], torch.zeros(1, 64)))
        self.assertTrue(torch.equal(clean["v_rgb_c"].grad[rgb_mask], torch.zeros(2, 64)))

    def test_seed_formula_and_disabled_probability(self):
        self.assertEqual(resolve_deg_rng_seed(101), 108919)
        self.assertEqual(resolve_deg_rng_seed(101, 2026), 2026)
        batch = tiny_batch()
        before = torch.get_rng_state().clone()
        sampler = DegRetrievalSampler("sdm", seed=2026, branch_prob=0.)
        rgb, ir, meta = sampler.sample(batch["images"], batch["thermal_images"], batch["image_ids"])
        self.assertFalse(meta["enabled"])
        self.assertIs(rgb, batch["images"])
        self.assertIs(ir, batch["thermal_images"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
