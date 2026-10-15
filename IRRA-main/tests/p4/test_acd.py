"""ACD algebra/routing regression and opt-in real-checkpoint CUDA checks."""
from __future__ import annotations

from argparse import Namespace
from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.p4.acd_lora import inject_lora, load_adapter, read_adapter, save_adapter
from tools.p4.acd_losses import acd_loss, joint_t2i, positive_targets, uniform_ce
from tools.p4.train_acd import (base_state_sha256, batch_hashes, clone_training_loader,
                                encode_training_batch, learning_rate_factor,
                                prepare_training, train_epochs)


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, 2)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        y = self.norm(x)
        return x + self.attn(y, y, y, need_weights=False)[0]


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 2, stride=2)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([Block(8), Block(8)])
        self.proj = nn.Parameter(torch.randn(8, 8) * 0.1)

    def forward(self, image, modality=None, return_adapter_aux=False):
        x = self.conv1(image).flatten(2).permute(2, 0, 1)
        for block in self.transformer.resblocks:
            x = block(x)
        x = x.permute(1, 0, 2) @ self.proj
        return (x, None) if return_adapter_aux else x


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.visual = TinyVisual()
        self.text = nn.Embedding(20, 8)

    def encode_image(self, image, modality="rgb"):
        visual = self.base_model.visual
        return visual(image.to(visual.conv1.weight.dtype), modality=modality)[:, 0].float()

    def encode_text_deterministic(self, captions):
        return self.text(captions[:, 0]).float()

    def encode_rgbt_global(self, rgb, ir):
        f_rgb = self.encode_image(rgb)
        f_ir = self.encode_image(ir, "thermal")
        return f_rgb, f_ir, f_rgb + f_ir, None, None, None


class AugmentedDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 24

    def __getitem__(self, index):
        # All worker random sources are exercised; dropout/model consumes none.
        value = torch.rand(3, 4, 4) + random.random() + float(np.random.rand())
        return {"images": value, "thermal_images": value.flip(-1) + torch.rand_like(value),
                "pids": torch.tensor(index % 4), "image_ids": torch.tensor(index),
                "caption_ids": torch.tensor([index % 20])}


def tiny_setup(seed=101):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = TinyModel().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    controller = inject_lora(model, rank=2, alpha=4, expected_layers=2)
    return model, controller


def synthetic_loader(seed=101, workers=0):
    loader = torch.utils.data.DataLoader(AugmentedDataset(), batch_size=4,
                                       shuffle=True, num_workers=workers)
    return clone_training_loader(loader, seed)


def smoke_cli(variant="A0", steps=5):
    return Namespace(variant=variant, lr=1e-3, epochs=4,
                     a1_impl="joint", smoke_steps=steps)


@contextmanager
def original_attention_forwards(model):
    """Bypass every injected wrapper without copying the frozen visual tower."""
    with ExitStack() as stack:
        for block in model.base_model.visual.transformer.resblocks:
            attention = block.attn
            original = nn.MultiheadAttention.forward.__get__(attention, nn.MultiheadAttention)
            stack.enter_context(patch.object(attention, "forward", original))
        yield


class ACDLossTests(unittest.TestCase):
    def test_joint_decomposition_and_logit_gradient_float64(self):
        generator = torch.Generator().manual_seed(23)
        z_f = torch.randn(8, 8, dtype=torch.float64, generator=generator)
        z_i = torch.randn(8, 8, dtype=torch.float64, generator=generator, requires_grad=True)
        pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        joint, _ = joint_t2i(z_f, z_i, pids, "joint")
        decomposed, _ = joint_t2i(z_f, z_i, pids, "decomposed")
        self.assertLess(float((joint - decomposed).abs()), 1e-10)
        gradient = torch.autograd.grad(joint, z_i)[0]
        pi_i = torch.sigmoid(torch.logsumexp(z_i, 1) - torch.logsumexp(z_f, 1))
        expected = (pi_i[:, None] * z_i.softmax(1) -
                    0.5 * positive_targets(pids, z_i.dtype)) / len(pids)
        torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)
        positive = pids[:, None].eq(pids[None, :])
        negative = ~positive
        inferred = gradient * len(pids) + 0.5 * positive_targets(pids, z_i.dtype)
        torch.testing.assert_close((inferred / z_i.softmax(1))[negative],
                                   pi_i[:, None].expand_as(z_i)[negative])
        print("ACD CPU joint/decomposed error=", float((joint - decomposed).abs()),
              "; coefficients c_pos=0.5, c_neg=pi_I verified", flush=True)

    def test_variants_and_uniform_ce_match_existing_definition(self):
        from tools.p4.deg_retrieval import compute_deg_loss
        generator = torch.Generator().manual_seed(5)
        text = F.normalize(torch.randn(6, 8, generator=generator), dim=1)
        image = F.normalize(torch.randn(6, 8, generator=generator), dim=1)
        pids = torch.tensor([0, 0, 1, 1, 2, 2])
        logits = 50 * text @ image.t()
        a0, _ = acd_loss(logits.detach(), logits, pids, "A0")
        expected = compute_deg_loss(image, text, pids, torch.tensor(50.0), "ce")
        torch.testing.assert_close(a0, expected)
        balanced, _ = acd_loss(logits, logits, pids, "A0bal")
        torch.testing.assert_close(balanced, 0.5 * uniform_ce(logits, pids)
                                   + uniform_ce(logits.t(), pids))

    def test_teacher_has_no_gradient(self):
        f = torch.randn(4, 4, requires_grad=True)
        i = torch.randn(4, 4, requires_grad=True)
        loss, _ = acd_loss(f, i, torch.arange(4), "A1")
        loss.backward()
        self.assertIsNone(f.grad)
        self.assertIsNotNone(i.grad)


class ACDRoutingTests(unittest.TestCase):
    def test_off_rgb_complete_ir_text_equal_original_attention_modules(self):
        model, controller = tiny_setup()
        batch = next(iter(synthetic_loader()))
        with torch.no_grad(), controller.enabled(False):
            with original_attention_forwards(model):
                rgb_original, ir_original, *_ = model.encode_rgbt_global(
                    batch["images"], batch["thermal_images"])
                text_original = model.encode_text_deterministic(batch["caption_ids"])
            rgb_off, ir_off, *_ = model.encode_rgbt_global(
                batch["images"], batch["thermal_images"])
            text_off = model.encode_text_deterministic(batch["caption_ids"])
        self.assertTrue(torch.equal(rgb_original, rgb_off))
        self.assertTrue(torch.equal(ir_original, ir_off))
        self.assertTrue(torch.equal(text_original, text_off))

    def test_zero_init_off_identity_independent_rng_and_nested_context(self):
        torch.manual_seed(5)
        model = TinyModel().eval()
        image = torch.randn(4, 3, 4, 4)
        original = model.encode_image(image)
        rng = torch.random.get_rng_state().clone()
        controller = inject_lora(model, rank=2, alpha=4, expected_layers=2)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(original, model.encode_image(image)))
        with controller.enabled():
            enabled = model.encode_image(image, "thermal")
            with controller.enabled(False):
                self.assertTrue(torch.equal(original, model.encode_image(image)))
            self.assertTrue(controller.active)
        self.assertFalse(controller.active)
        absolute = float((original - enabled).abs().max())
        relative = float(((original - enabled).abs() / original.abs().clamp_min(1e-8)).max())
        print(f"ACD CPU zero-init max_abs={absolute:.9g} max_relative={relative:.9g}", flush=True)
        self.assertLessEqual(relative, 1e-3)
        self.assertTrue(all(p.dtype == torch.float32 for p in controller.parameters()))

    def test_five_steps_only_adapter_changes_and_has_grad(self):
        model, controller = tiny_setup()
        before = base_state_sha256(model)
        original_adapter = controller.state_dict()
        batch = next(iter(synthetic_loader()))
        original = model.encode_image(batch["thermal_images"], "thermal").detach()
        with controller.enabled():
            initial_student = model.encode_image(batch["thermal_images"], "thermal").detach()
        with tempfile.TemporaryDirectory() as directory:
            record = train_epochs(model, controller, synthetic_loader(), smoke_cli(),
                                  Path(directory), {"base_checkpoint_sha256": "a" * 64})
            self.assertEqual(record["steps"], 5)
            self.assertEqual(record["encodings_total"], dict.fromkeys(
                ("text_teacher", "rgb_teacher", "ir_teacher", "ir_student"), 5))
            self.assertEqual(record["base_state_sha256_after"], before)
        self.assertTrue(any(not torch.equal(v, controller.state_dict()[k])
                            for k, v in original_adapter.items()))
        for name, parameter in model.named_parameters():
            if ".acd_lora." in name:
                self.assertIsNotNone(parameter.grad)
            else:
                self.assertIsNone(parameter.grad)
        with controller.enabled():
            changed = model.encode_image(batch["thermal_images"], "thermal").detach()
        self.assertGreater(float((original - changed).abs().max()), 0)
        self.assertGreater(float((initial_student - changed).abs().max()), 0)
        self.assertTrue(torch.equal(original, model.encode_image(batch["thermal_images"], "thermal")))

    def test_a1_parameter_gradients_match_fp32(self):
        model, controller = tiny_setup()
        z_f, z_i = encode_training_batch(model, controller, next(iter(synthetic_loader())))
        pids = torch.tensor([0, 1, 2, 3])
        joint, _ = acd_loss(z_f, z_i, pids, "A1", "joint")
        decomposed, _ = acd_loss(z_f, z_i, pids, "A1", "decomposed")
        self.assertLess(float((joint - decomposed).abs() / joint.abs()), 1e-6)
        parameters = list(controller.parameters())
        g_joint = torch.cat([x.reshape(-1) for x in torch.autograd.grad(joint, parameters, retain_graph=True)])
        g_decomposed = torch.cat([x.reshape(-1) for x in torch.autograd.grad(decomposed, parameters)])
        self.assertGreaterEqual(float(F.cosine_similarity(g_joint, g_decomposed, dim=0)), 0.99999)

    def test_twenty_step_hashes_equal_across_variants(self):
        hashes = []
        for variant in ("A0", "A0bal", "A1"):
            model, controller = tiny_setup()
            with tempfile.TemporaryDirectory() as directory:
                train_epochs(model, controller, synthetic_loader(), smoke_cli(variant, 20),
                             Path(directory), {"base_checkpoint_sha256": "a" * 64})
                hashes.append((Path(directory) / "step_hashes.txt").read_bytes())
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[1], hashes[2])
        self.assertEqual(len(hashes[0].splitlines()), 20)

    def test_adapter_roundtrip_and_base_sha_rejection(self):
        model, controller = tiny_setup()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_adapter(controller, path, "a" * 64)
            payload = read_adapter(path)
            self.assertEqual(set(payload), {"schema_version", "config", "lora_state_dict"})
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_adapter(controller, path, "b" * 64)
            load_adapter(controller, path, "a" * 64)

    def test_nonfinite_lora_alpha_rejected(self):
        for alpha in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                inject_lora(TinyModel(), rank=2, alpha=alpha, expected_layers=2)

    def test_checkpoint_split_membership_checked_before_loader_or_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "last.pth").write_bytes(b"not-loaded-in-preflight-test")
            (root / "configs.yaml").write_text("mocked: true", encoding="utf-8")
            requested = root / "requested.json"
            saved = root / "saved.json"
            cli = Namespace(base_run=str(root), root_dir=str(root / "data"),
                            dev_split_file=str(requested), data_seed=101)
            def original_args():
                return Namespace(dataset_name="RGBT-PEDES", batch_size=64,
                                 sampler="random", use_rgbt_add_fusion=True,
                                 deg_retrieval_loss="none", temperature=0.02,
                                 dev_split_file="recorded-base-split.json")
            split = {"n_train": 1076, "n_dev": 190,
                     "split_identity_sha256": "requested-identity-membership"}
            args = original_args()
            with patch("tools.p4.train_batch_stats._load_training_args", return_value=args), \
                    patch("tools.p4.export_dev_records._resolve_annotation", return_value=root / "annotation.json"), \
                    patch("tools.p4.export_dev_records._resolve_saved_split", return_value=(saved, args.dev_split_file)) as resolver, \
                    patch("tools.p4.export_dev_records.validate_dev_split", side_effect=[
                        split, dict(split, split_identity_sha256="other-1076-190-membership")]), \
                    patch("datasets.build_dataloader") as build, \
                    self.assertRaisesRegex(ValueError, "differs from base checkpoint"):
                prepare_training(cli)
            build.assert_not_called()
            self.assertEqual(args.dev_split_file, "recorded-base-split.json")
            self.assertEqual(resolver.call_args.args[0].dev_split_file,
                             "recorded-base-split.json")
            args = original_args()
            with patch("tools.p4.train_batch_stats._load_training_args", return_value=args), \
                    patch("tools.p4.export_dev_records._resolve_annotation", return_value=root / "annotation.json"), \
                    patch("tools.p4.export_dev_records._resolve_saved_split", return_value=(saved, args.dev_split_file)), \
                    patch("tools.p4.export_dev_records.validate_dev_split", side_effect=[split, dict(split)]), \
                    patch("datasets.build_dataloader", side_effect=RuntimeError("matching-preflight-reached-loader")), \
                    patch("train.set_seed"), \
                    self.assertRaisesRegex(RuntimeError, "matching-preflight-reached-loader"):
                prepare_training(cli)
            self.assertEqual(args.dev_split_file, str(requested.resolve()))
            self.assertFalse(args.MLM)

    def test_worker_seed_and_augmentation_flow_repeat(self):
        hashes = []
        for _ in range(2):
            torch.manual_seed(101)
            random.seed(101)
            np.random.seed(101)
            hashes.append([batch_hashes(batch, index) for index, batch in
                           enumerate(synthetic_loader(workers=1))])
        self.assertEqual(hashes[0], hashes[1])

    def test_warmup_then_cosine_schedule(self):
        values = [learning_rate_factor(i, 12, 3) for i in range(12)]
        self.assertEqual(values[:3], [1 / 3, 2 / 3, 1])
        self.assertEqual(values[3], 1)
        self.assertEqual(values[-1], 0)

    def _half_zero_and_five_steps(self, device):
        model, controller = tiny_setup()
        # Real injection occurs after the base has reached its final dtype and
        # device; moving a complete model afterward would also cast adapters.
        torch.manual_seed(101)
        model = TinyModel().to(device=device, dtype=torch.float16).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        controller = inject_lora(model, rank=2, alpha=4, expected_layers=2)
        loader = synthetic_loader()
        cpu_batch = next(iter(loader))
        batch = {name: value.to(device) if torch.is_tensor(value) else value
                 for name, value in cpu_batch.items()}
        with torch.no_grad():
            original = model.encode_image(batch["thermal_images"], "thermal")
            with controller.enabled():
                initial = model.encode_image(batch["thermal_images"], "thermal")
        max_abs = float((original - initial).abs().max())
        max_relative = float(((original - initial).abs() / original.abs().clamp_min(1e-8)).max())
        self.assertLessEqual(max_relative, 1e-3)
        self.assertTrue(torch.equal(original, initial))
        with tempfile.TemporaryDirectory() as directory:
            cli = smoke_cli()
            cli.lr = 1e-4
            train_epochs(model, controller, synthetic_loader(), cli, Path(directory),
                         {"base_checkpoint_sha256": "a" * 64})
        with torch.no_grad():
            off = model.encode_image(batch["thermal_images"], "thermal")
            with controller.enabled():
                trained = model.encode_image(batch["thermal_images"], "thermal")
        change = float((trained - initial).abs().max())
        self.assertGreater(change, 0)
        self.assertTrue(torch.equal(off, original))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in controller.parameters()))
        print(f"ACD synthetic {device} native-half zero max_abs={max_abs:.9g}, "
              f"max_relative={max_relative:.9g}; five_steps_lr1e-4 change={change:.9g}", flush=True)

    def test_cpu_half_zero_exact_and_five_steps_survive(self):
        self._half_zero_and_five_steps("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "Synthetic FP16 CUDA test requires CUDA")
    def test_cuda_half_zero_exact_and_five_steps_survive(self):
        self._half_zero_and_five_steps("cuda")


def real_skip_reason():
    reasons = []
    if not torch.cuda.is_available():
        reasons.append("CUDA unavailable")
    for name in ("BASE_RUN", "DATA_ROOT", "DEV_SPLIT"):
        value = os.environ.get(name)
        if not value:
            reasons.append(f"environment variable {name} is missing")
        elif not Path(value).exists():
            reasons.append(f"{name} path does not exist: {value}")
    if os.environ.get("BASE_RUN"):
        for filename in ("last.pth", "configs.yaml"):
            if not (Path(os.environ["BASE_RUN"]) / filename).is_file():
                reasons.append(f"BASE_RUN/{filename} is missing")
    return "; ".join(reasons)


REAL_SKIP = real_skip_reason()


@unittest.skipIf(bool(REAL_SKIP), "Real-checkpoint ACD test skipped: " + REAL_SKIP)
class ACDRealCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = Namespace(base_run=os.environ["BASE_RUN"], root_dir=os.environ["DATA_ROOT"],
                            dev_split_file=os.environ["DEV_SPLIT"], lora_rank=8,
                            lora_alpha=16, lora_init_seed=20261015, data_seed=101,
                            variant="A0", lr=1e-4, epochs=20, a1_impl="joint", smoke_steps=5)
        cls.model, cls.controller, cls.loader, cls.record = prepare_training(cls.cli)
        cls.initial = cls.controller.state_dict()

    def setUp(self):
        from train import set_seed
        set_seed(101)
        self.controller.load_state_dict(self.initial)
        for parameter in self.model.parameters():
            parameter.grad = None
        self.current_loader = clone_training_loader(self.loader, 101)
        self.batch = {key: value.cuda() if torch.is_tensor(value) else value
                      for key, value in next(iter(self.current_loader)).items()}

    def run_five(self):
        with tempfile.TemporaryDirectory() as directory:
            return train_epochs(self.model, self.controller, self.current_loader,
                                self.cli, Path(directory), dict(self.record))

    def test_1_zero_initialization_error(self):
        with torch.no_grad():
            original = self.model.encode_image(self.batch["thermal_images"], "thermal")
            with self.controller.enabled():
                adapted = self.model.encode_image(self.batch["thermal_images"], "thermal")
        absolute = float((original - adapted).abs().max())
        relative = float(((original - adapted).abs() / original.abs().clamp_min(1e-8)).max())
        print(f"ACD real zero-init max_abs={absolute:.9g} max_relative={relative:.9g}", flush=True)
        self.assertLessEqual(relative, 1e-3)

    def test_2_disabled_routes_bitwise_unchanged(self):
        with torch.no_grad(), self.controller.enabled(False):
            # Restore all twelve actual nn.MultiheadAttention.forward methods
            # temporarily, rather than compare two already wrapped OFF calls.
            with original_attention_forwards(self.model):
                rgb, ir, *_ = self.model.encode_rgbt_global(
                    self.batch["images"], self.batch["thermal_images"])
                text = self.model.encode_text_deterministic(self.batch["caption_ids"])
            rgb2, ir2, *_ = self.model.encode_rgbt_global(
                self.batch["images"], self.batch["thermal_images"])
            text2 = self.model.encode_text_deterministic(self.batch["caption_ids"])
        self.assertTrue(torch.equal(rgb, rgb2))
        self.assertTrue(torch.equal(ir, ir2))
        self.assertTrue(torch.equal(text, text2))

    def test_3_five_steps_base_sha_grad_and_parameters(self):
        before = base_state_sha256(self.model)
        self.run_five()
        self.assertEqual(before, base_state_sha256(self.model))
        self.assertTrue(any(not torch.equal(value, self.controller.state_dict()[name])
                            for name, value in self.initial.items()))
        for name, parameter in self.model.named_parameters():
            self.assertEqual(parameter.grad is not None, ".acd_lora." in name)

    def test_4_a1_losses_and_lora_gradients_fp32(self):
        z_f, z_i = encode_training_batch(self.model, self.controller, self.batch)
        joint, _ = acd_loss(z_f, z_i, self.batch["pids"], "A1", "joint")
        decomposed, _ = acd_loss(z_f, z_i, self.batch["pids"], "A1", "decomposed")
        self.assertLessEqual(float((joint - decomposed).abs() / joint.abs()), 1e-6)
        parameters = list(self.controller.parameters())
        left = torch.cat([g.reshape(-1) for g in torch.autograd.grad(joint, parameters, retain_graph=True)])
        right = torch.cat([g.reshape(-1) for g in torch.autograd.grad(decomposed, parameters)])
        cosine = float(F.cosine_similarity(left, right, dim=0))
        print(f"ACD real A1 joint/decomposed gradient_cosine={cosine:.9g}", flush=True)
        self.assertGreaterEqual(cosine, 0.99999)

    def test_5_twenty_steps_data_hashes_across_variants(self):
        from train import set_seed
        all_hashes = []
        for variant in ("A0", "A0bal", "A1"):
            set_seed(101)
            self.controller.load_state_dict(self.initial)
            cli = Namespace(**vars(self.cli))
            cli.variant, cli.smoke_steps = variant, 20
            with tempfile.TemporaryDirectory() as directory:
                train_epochs(self.model, self.controller, clone_training_loader(self.loader, 101),
                             cli, Path(directory), dict(self.record))
                all_hashes.append((Path(directory) / "step_hashes.txt").read_bytes())
        self.assertEqual(all_hashes[0], all_hashes[1])
        self.assertEqual(all_hashes[1], all_hashes[2])

    def test_6_fp16_adapter_change_survives(self):
        with torch.no_grad(), self.controller.enabled():
            initial = self.model.encode_image(self.batch["thermal_images"], "thermal")
        self.run_five()
        with torch.no_grad():
            off = self.model.encode_image(self.batch["thermal_images"], "thermal")
            with self.controller.enabled():
                on = self.model.encode_image(self.batch["thermal_images"], "thermal")
        difference = float((on - off).abs().max())
        trained_difference = float((on - initial).abs().max())
        print(f"ACD real FP16-trap max_feature_difference={difference:.9g}, "
              f"trained_vs_zero_adapter={trained_difference:.9g}", flush=True)
        self.assertGreater(difference, 0)
        self.assertGreater(trained_difference, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
