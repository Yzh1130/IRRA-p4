"""R10 mechanism gradients, legacy byte parity, private RNG, and opt-in CUDA tests."""
from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
for location in (CODE_ROOT, Path(__file__).resolve().parent):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from test_acd import REAL_SKIP, clone_training_loader, synthetic_loader, tiny_setup, smoke_cli
from tools.p4 import acd_losses, train_acd, teacher_ref_stats
from tools.p4.acd_losses import (acd_loss, draw_teacher_permutation, joint_t2i,
                                mechanism_t2i, positive_targets,
                                repulsion_diagnostics, uniform_ce)

LEGACY_COMMIT = "ee70023798e68a082f571775a5f23a4f25684d97"
VARIANTS = ("A0", "A0bal", "A1", "Pull", "A1const", "A1shuf", "DCLbal")


def legacy_module(filename):
    """Execute the user-requested trusted Git baseline, never an external file."""
    source = subprocess.check_output(
        ["git", "show", f"{LEGACY_COMMIT}:IRRA-main/tools/p4/{filename}.py"],
        cwd=CODE_ROOT, text=True, encoding="utf-8")
    module = types.ModuleType(f"_r10_legacy_{filename}")
    module.__file__ = str(CODE_ROOT / "tools" / "p4" / f"{filename}.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def rng_snapshot():
    return (torch.random.get_rng_state().clone(), np.random.get_state(),
            random.getstate(), [x.clone() for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else [])


def assert_rng_equal(test, before, after):
    test.assertTrue(torch.equal(before[0], after[0]))
    test.assertEqual(before[1][0], after[1][0])
    test.assertTrue(np.array_equal(before[1][1], after[1][1]))
    test.assertEqual(before[1][2:], after[1][2:])
    test.assertEqual(before[2], after[2])
    test.assertEqual(len(before[3]), len(after[3]))
    for left, right in zip(before[3], after[3]):
        test.assertTrue(torch.equal(left, right))


def logits_fixture():
    generator = torch.Generator().manual_seed(1234)
    teacher = torch.randn(12, 12, generator=generator, dtype=torch.float64,
                          requires_grad=True)
    student = torch.randn(12, 12, generator=generator, dtype=torch.float64,
                          requires_grad=True)
    # Unequal positive counts exercise all same-PID columns, including repeats.
    pids = torch.tensor([8, 8, 8, 2, 2, 19, 19, 19, 19, 4, 5, 5])
    return teacher, student, pids


def t2i(teacher, student, pids, variant, **options):
    if variant == "A0bal":
        return 0.5 * uniform_ce(student, pids)
    if variant == "A1":
        return joint_t2i(teacher, student, pids)[0]
    return mechanism_t2i(teacher, student, pids, variant, **options)[0]


def write_reference(directory, value=8.125):
    path = Path(directory) / "teacher_ref.json"
    path.write_text(json.dumps({"ref_mean_b64": value,
                                "complete_epoch": True,
                                "valid_as_full_epoch_reference": True}), encoding="utf-8")
    return path


class ACDMechanismGradientTests(unittest.TestCase):
    def test_fp64_t2i_closed_gradients_and_frozen_teacher(self):
        teacher, student, pids = logits_fixture()
        q = positive_targets(pids, student.dtype)
        a = torch.logsumexp(teacher.detach(), 1)
        b = torch.logsumexp(student.detach(), 1)
        permutation = torch.tensor([4, 0, 11, 7, 8, 9, 1, 10, 3, 5, 6, 2])
        ref_mean = 2.75
        for variant in VARIANTS[1:]:
            with self.subTest(variant=variant):
                options = {"ref_mean": ref_mean, "permutation": permutation}
                loss = t2i(teacher, student, pids, variant, **options)
                gradient, teacher_gradient = torch.autograd.grad(
                    loss, (student, teacher), allow_unused=True)
                if variant == "DCLbal":
                    negatives = ~pids[:, None].eq(pids[None, :])
                    repulsion = 0.5 * student.detach().masked_fill(
                        ~negatives, -torch.inf).softmax(1)
                else:
                    coefficient = {"A0bal": torch.full_like(b, 0.5),
                                   "A1": torch.sigmoid(b - a),
                                   "A1const": torch.sigmoid(b - ref_mean),
                                   "A1shuf": torch.sigmoid(b - a[permutation]),
                                   "Pull": torch.zeros_like(b)}[variant]
                    repulsion = coefficient[:, None] * student.detach().softmax(1)
                expected = (repulsion - 0.5 * q) / len(pids)
                self.assertLessEqual(float((gradient - expected).abs().max()), 1e-12)
                self.assertIsNone(teacher_gradient)

    def test_i2t_component_gradient_is_bitwise_identical_for_every_variant(self):
        teacher, student, pids = logits_fixture()
        reference = torch.autograd.grad(uniform_ce(student.t(), pids), student)[0]
        original_ce = acd_losses.uniform_ce
        for variant in VARIANTS:
            calls = []
            def captured_ce(logits, labels):
                value = original_ce(logits, labels)
                if (logits.data_ptr() == student.data_ptr()
                        and logits.stride() == student.t().stride()):
                    calls.append(value)
                return value
            with self.subTest(variant=variant), patch.object(
                    acd_losses, "uniform_ce", side_effect=captured_ce):
                acd_loss(teacher, student, pids, variant,
                         ref_mean=2.75, permutation=torch.arange(12).roll(1))
                self.assertEqual(len(calls), 1)
                actual = torch.autograd.grad(calls[0], student)[0]
                self.assertTrue(torch.equal(reference, actual))

    def test_const_and_shuf_closed_gradient_above_softplus_threshold(self):
        teacher, student, pids = logits_fixture()
        b = student.detach().logsumexp(1)
        q = positive_targets(pids, student.dtype)
        # F.softplus's default threshold=20 truncates the derivative to one;
        # the requested sigmoid derivative must remain accurate in FP64 here.
        reference = float(b.min()-21)
        teacher = (b-21-torch.log(torch.tensor(12.0, dtype=student.dtype)))[:, None].expand_as(teacher)
        for variant in ("A1const", "A1shuf"):
            with self.subTest(variant=variant):
                loss = mechanism_t2i(teacher, student, pids, variant,
                                     ref_mean=reference, permutation=torch.arange(12))[0]
                actual = torch.autograd.grad(loss, student)[0]
                a = reference if variant == "A1const" else teacher.logsumexp(1)
                expected = (torch.sigmoid(b-a)[:, None]*student.detach().softmax(1)-0.5*q)/12
                self.assertLessEqual(float((actual-expected).abs().max()), 1e-12)

    def test_a1_teacher_candidate_permutation_invariant_but_row_offset_matters(self):
        teacher, student, pids = logits_fixture()
        original = torch.autograd.grad(t2i(teacher, student, pids, "A1"), student)[0]
        generator = torch.Generator().manual_seed(91)
        indices = torch.stack([torch.randperm(12, generator=generator) for _ in range(12)])
        permuted = teacher.detach().gather(1, indices)
        reordered = torch.autograd.grad(t2i(permuted, student, pids, "A1"), student)[0]
        self.assertLessEqual(float((original - reordered).abs().max()), 1e-12)
        shifted = teacher.detach().clone()
        shifted[3] += 0.9
        torch.testing.assert_close(teacher[3].softmax(0), shifted[3].softmax(0),
                                   atol=1e-15, rtol=1e-15)
        changed = torch.autograd.grad(t2i(shifted, student, pids, "A1"), student)[0]
        self.assertGreater(float((original[3] - changed[3]).abs().max()), 1e-4)

    def test_shuffle_reproducible_private_cpu_generator_and_global_rng_unchanged(self):
        before = rng_snapshot()
        left = torch.Generator(device="cpu").manual_seed(20261017)
        right = torch.Generator(device="cpu").manual_seed(20261017)
        permutations = []
        for _ in range(20):
            one = draw_teacher_permutation(12, left)
            two = draw_teacher_permutation(12, right)
            self.assertEqual(one.device.type, "cpu")
            self.assertEqual(one.dtype, torch.int64)
            self.assertTrue(torch.equal(one, two))
            self.assertTrue(torch.equal(one.sort().values, torch.arange(12)))
            permutations.append(one.tolist())
        self.assertGreater(len({tuple(x) for x in permutations}), 1)
        assert_rng_equal(self, before, rng_snapshot())

    def test_dcl_excludes_every_positive_and_counts_all_positive_rows(self):
        teacher, student, pids = logits_fixture()
        positives = pids[:, None].eq(pids[None, :])
        large_positives = student.detach().masked_fill(positives, 1000).requires_grad_()
        loss, diagnostics = mechanism_t2i(teacher, large_positives, pids, "DCLbal")
        q = positive_targets(pids, student.dtype)
        expected = 0.5 * (-(q * large_positives).sum(1) +
                          large_positives.masked_fill(positives, -torch.inf).logsumexp(1)).mean()
        self.assertTrue(torch.equal(loss, expected))
        gradient = torch.autograd.grad(loss, large_positives)[0]
        self.assertTrue(torch.equal(gradient[positives], (-0.5 * q / 12)[positives]))
        self.assertEqual(int(diagnostics["t2i_no_negative_rows"]), 0)
        all_one = torch.zeros(12, dtype=torch.long)
        pull_only, diagnostics = mechanism_t2i(teacher, student, all_one, "DCLbal")
        expected = -0.5 * student.mean(1).mean()
        torch.testing.assert_close(pull_only, expected, atol=1e-15, rtol=1e-15)
        actual = torch.autograd.grad(pull_only, student)[0]
        self.assertTrue(torch.equal(actual, torch.full_like(student, -0.5 / 144)))
        self.assertEqual(int(diagnostics["t2i_no_negative_rows"]), 12)

    def test_repulsion_coefficients_for_all_seven_variants(self):
        teacher, student, pids = logits_fixture()
        a, b = teacher.detach().logsumexp(1), student.detach().logsumexp(1)
        permutation = torch.arange(12).roll(4)
        expected = {"A0": 1.0, "A0bal": 0.5, "A1": float(torch.sigmoid(b-a).mean()),
                    "Pull": 0.0, "A1const": float(torch.sigmoid(b-2.75).mean()),
                    "A1shuf": float(torch.sigmoid(b-a[permutation]).mean()), "DCLbal": 0.5}
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                diagnostics = repulsion_diagnostics(teacher, student, pids, variant,
                                                     ref_mean=2.75, permutation=permutation)
                self.assertLessEqual(abs(float(diagnostics["t2i_repulsion_coeff_mean"])
                                         - expected[variant]), 1e-15)

    def test_legacy_three_fp64_losses_diagnostics_and_gradients_bitwise(self):
        legacy = legacy_module("acd_losses")
        teacher, student, pids = logits_fixture()
        for variant in VARIANTS[:3]:
            for implementation in ("joint", "decomposed"):
                with self.subTest(variant=variant, implementation=implementation):
                    old_loss, old_diag = legacy.acd_loss(teacher, student, pids, variant, implementation)
                    new_loss, new_diag = acd_loss(teacher, student, pids, variant, implementation)
                    self.assertTrue(torch.equal(old_loss, new_loss))
                    self.assertEqual(old_diag.keys(), new_diag.keys())
                    for key in old_diag:
                        self.assertTrue(torch.equal(old_diag[key], new_diag[key]), key)
                    old_grad = torch.autograd.grad(old_loss, student)[0]
                    new_grad = torch.autograd.grad(new_loss, student)[0]
                    self.assertTrue(torch.equal(old_grad, new_grad))


class ACDMechanismTrainingTests(unittest.TestCase):
    def test_legacy_default_argument_record_matches_baseline_and_new_options_are_recorded(self):
        legacy = legacy_module("train_acd")
        arguments = ["--base_run", "base", "--root_dir", "data", "--dev_split_file", "split",
                     "--variant", "A1", "--output_dir", "out"]
        original = vars(legacy.build_parser().parse_args(arguments))
        current = train_acd.build_parser().parse_args(arguments)
        self.assertEqual(train_acd._run_record_arguments(current), original)
        for variant in VARIANTS[:3]:
            current.variant = variant
            expected = dict(original, variant=variant)
            self.assertEqual(train_acd._run_record_arguments(current), expected)
        current.variant = "A1shuf"
        self.assertEqual(train_acd._run_record_arguments(current), vars(current))
        self.assertEqual(train_acd._run_record_arguments(current)["shuffle_seed"], 20261017)
        current.variant, current.mechanism_diagnostics = "A1", True
        self.assertTrue(train_acd._run_record_arguments(current)["mechanism_diagnostics"])

    def test_all_seven_variants_twenty_step_hashes_identical_and_shuffle_recorded(self):
        hashes = []
        for variant in VARIANTS:
            model, controller = tiny_setup()
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                cli = smoke_cli(variant, 20)
                cli.ref_stats_json = str(write_reference(directory))
                cli.shuffle_seed = 20261017
                record = train_acd.train_epochs(model, controller, synthetic_loader(), cli,
                                               Path(directory), {"base_checkpoint_sha256": "a"*64})
                hashes.append((Path(directory) / "step_hashes.txt").read_bytes())
                if variant == "A1shuf":
                    generator = torch.Generator(device="cpu").manual_seed(cli.shuffle_seed)
                    expected = [{"step": index+1, "sha1": hashlib.sha1(
                        torch.randperm(4, generator=generator).numpy().tobytes()).hexdigest()}
                                for index in range(20)]
                    self.assertEqual(record["teacher_permutation_sha1"], expected)
                    self.assertEqual(record["seeds"]["shuffle_seed"], cli.shuffle_seed)
                if variant == "A1const":
                    self.assertEqual(record["ref_mean_b64"], 8.125)
                    self.assertEqual(record["ref_stats_json"]["sha256"], hashlib.sha256(
                        Path(cli.ref_stats_json).read_bytes()).hexdigest())
        self.assertTrue(all(value == hashes[0] for value in hashes))
        self.assertEqual(len(hashes[0].splitlines()), 20)

    def test_legacy_training_logs_records_and_adapter_bitwise_against_git_baseline(self):
        legacy = legacy_module("train_acd")
        legacy.acd_loss = legacy_module("acd_losses").acd_loss
        for variant in VARIANTS[:3]:
            outputs = []
            for trainer in (legacy.train_epochs, train_acd.train_epochs):
                model, controller = tiny_setup()
                with tempfile.TemporaryDirectory() as directory, \
                        patch("time.perf_counter", return_value=0.0), redirect_stdout(io.StringIO()):
                    result = trainer(model, controller, synthetic_loader(), smoke_cli(variant, 5),
                                     Path(directory), {"base_checkpoint_sha256": "a"*64})
                    outputs.append((copy.deepcopy(result), {
                        filename: (Path(directory)/filename).read_bytes() for filename in (
                            "step_log.jsonl", "step_hashes.txt", "train_log.txt", "adapter.pt")},
                        {key: value.clone() for key, value in controller.state_dict().items()}))
            with self.subTest(variant=variant):
                self.assertEqual(outputs[0][0], outputs[1][0])
                self.assertEqual(outputs[0][1], outputs[1][1])
                for key in outputs[0][2]:
                    self.assertTrue(torch.equal(outputs[0][2][key], outputs[1][2][key]), key)

    def test_legacy_mechanism_diagnostics_opt_in_changes_only_diagnostics(self):
        outputs = []
        for enabled in (False, True):
            model, controller = tiny_setup()
            cli = smoke_cli("A1", 5)
            cli.mechanism_diagnostics = enabled
            with tempfile.TemporaryDirectory() as directory, \
                    patch("time.perf_counter", return_value=0.0), redirect_stdout(io.StringIO()):
                train_acd.train_epochs(model, controller, synthetic_loader(), cli, Path(directory),
                                       {"base_checkpoint_sha256": "a"*64})
                rows = [json.loads(line) for line in (Path(directory)/"step_log.jsonl").read_text().splitlines()]
                outputs.append((rows, {key: value.clone() for key, value in controller.state_dict().items()}))
        for original, extended in zip(outputs[0][0], outputs[1][0]):
            self.assertNotIn("t2i_repulsion_coeff_mean", original)
            self.assertIn("t2i_repulsion_coeff_mean", extended)
            self.assertTrue(0 <= extended.pop("t2i_repulsion_coeff_mean") <= 1)
            self.assertEqual(original, extended)
        for key in outputs[0][1]:
            self.assertTrue(torch.equal(outputs[0][1][key], outputs[1][1][key]))

    def test_a1const_requires_finite_reference_and_records_loaded_coefficient(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "ref_stats_json"):
                train_acd.load_ref_stats(None)
            for value in (None, True, "8.0", float("nan"), float("inf")):
                path = write_reference(directory, value)
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                    train_acd.load_ref_stats(path)
            path = write_reference(directory, 8.125)
            loaded = train_acd.load_ref_stats(path)
            for field in ("complete_epoch", "valid_as_full_epoch_reference"):
                partial = {"ref_mean_b64": 8.125, field: False}
                path.write_text(json.dumps(partial), encoding="utf-8")
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "full|complete"):
                    train_acd.load_ref_stats(path)
            path.write_text(json.dumps({"ref_mean_b64": 8.125}), encoding="utf-8")
            self.assertEqual(train_acd.load_ref_stats(path)["ref_mean_b64"], 8.125)
            teacher, student, pids = logits_fixture()
            _, diagnostics = acd_loss(teacher, student, pids, "A1const", ref_mean=loaded["ref_mean_b64"])
            expected = torch.sigmoid(student.detach().logsumexp(1)-8.125).mean()
            self.assertTrue(torch.equal(diagnostics["t2i_repulsion_coeff_mean"], expected))


def teacher_batches():
    generator = torch.Generator().manual_seed(73)
    return [{"images": torch.randn(size, 3, 4, 4, generator=generator),
             "thermal_images": torch.randn(size, 3, 4, 4, generator=generator),
             "caption_ids": torch.arange(size).remainder(20)[:, None],
             "pids": torch.arange(size).remainder(5)} for size in (64, 64, 7, 64)]


class TeacherReferenceTests(unittest.TestCase):
    def test_full_epoch_b64_and_tail_statistics_teacher_only_and_rng_unchanged(self):
        model, controller = tiny_setup()
        batches = teacher_batches()
        expected, positives = [], []
        with torch.no_grad(), controller.enabled(False):
            for batch in batches:
                text = F.normalize(model.encode_text_deterministic(batch["caption_ids"]).float(), dim=1)
                rgb, ir, *_ = model.encode_rgbt_global(batch["images"], batch["thermal_images"])
                z_f = 50.0*(text @ F.normalize(rgb.float()+ir.float(), dim=1).t())
                expected.append(z_f.logsumexp(1).double())
                positives.append((positive_targets(batch["pids"], z_f.dtype)*z_f).sum(1).double())
        original = {key: value.clone() for key, value in model.state_dict().items()}
        before = rng_snapshot()
        original_encode_image = model.encode_image
        def teacher_image_only(*args, **kwargs):
            self.assertFalse(controller.active, "Student route was encoded")
            return original_encode_image(*args, **kwargs)
        with patch.object(model, "encode_rgbt_global", wraps=model.encode_rgbt_global) as image_calls, \
                patch.object(model, "encode_text_deterministic", wraps=model.encode_text_deterministic) as text_calls, \
                patch.object(model, "encode_image", side_effect=teacher_image_only):
            result = teacher_ref_stats.collect_teacher_stats(model, controller, batches)
        self.assertEqual(image_calls.call_count, 4)
        self.assertEqual(text_calls.call_count, 4)
        assert_rng_equal(self, before, rng_snapshot())
        self.assertFalse(controller.active)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]), key)
        all_rows = torch.cat(expected)
        full_rows = torch.cat([row for row in expected if len(row) == 64])
        self.assertEqual(result["ref_mean_all"], float(all_rows.mean()))
        self.assertEqual(result["ref_mean_b64"], float(full_rows.mean()))
        self.assertEqual(result["ref_std_all"], float(all_rows.std(unbiased=False)))
        self.assertEqual(result["ref_median_all"], float(all_rows.quantile(0.5)))
        for key, quantile in (("p05", .05), ("p25", .25), ("p75", .75), ("p95", .95)):
            self.assertEqual(result["ref_quantiles_all"][key], float(all_rows.quantile(quantile)))
        self.assertEqual(result["teacher_positive_logit_mean"], float(torch.cat(positives).mean()))
        self.assertEqual((result["row_count"], result["batch_count"]), (199, 4))
        self.assertEqual((result["full_batch64_row_count"], result["full_batch64_count"]), (192, 3))
        self.assertTrue(result["complete_epoch"])

    def test_three_batch_cli_smoke_metadata_no_overwrite_and_global_rng(self):
        model, controller = tiny_setup()
        batches = teacher_batches()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"stats.json"
            argv = ["--base_run", directory, "--root_dir", directory, "--dev_split_file", "split.json",
                    "--data_seed", "101", "--output_json", str(output), "--num_batches", "3"]
            before = rng_snapshot()
            with patch.object(teacher_ref_stats, "prepare_training", return_value=(
                    model, controller, batches, {"base_checkpoint_sha256": "a"*64})), \
                    redirect_stdout(io.StringIO()):
                result = teacher_ref_stats.main(argv)
            assert_rng_equal(self, before, rng_snapshot())
            self.assertEqual(result["batch_count"], 3)
            self.assertEqual(result["row_count"], 135)
            self.assertFalse(result["complete_epoch"])
            self.assertFalse(result["valid_as_full_epoch_reference"])
            self.assertEqual(result["status"], "smoke_completed")
            self.assertEqual(result["student_encodings"], 0)
            self.assertFalse(result["teacher_adapter_enabled"])
            self.assertEqual(result["seeds"]["data_seed"], 101)
            self.assertEqual(len(result["git_commit"]), 40)
            self.assertIn("elapsed_seconds", result)
            self.assertIn("peak_cuda_memory_allocated_bytes", result)
            before_bytes = output.read_bytes()
            with self.assertRaises(FileExistsError):
                teacher_ref_stats.main(argv)
            self.assertEqual(before_bytes, output.read_bytes())

    def test_no_full_batch_reference_is_explicit_and_training_rejects_it(self):
        model, controller = tiny_setup()
        result = teacher_ref_stats.collect_teacher_stats(model, controller, teacher_batches()[2:3])
        self.assertIsNone(result["ref_mean_b64"])
        self.assertEqual(result["full_batch64_count"], 0)
        self.assertFalse(result["valid_as_full_epoch_reference"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"stats.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite"):
                train_acd.load_ref_stats(path)


@unittest.skipIf(bool(REAL_SKIP), "Real-checkpoint R10 test skipped: " + REAL_SKIP)
class ACDMechanismRealCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls.cli = Namespace(base_run=os.environ["BASE_RUN"], root_dir=os.environ["DATA_ROOT"],
                            dev_split_file=os.environ["DEV_SPLIT"], lora_rank=8, lora_alpha=16,
                            lora_init_seed=20261015, data_seed=101, variant="A1", lr=1e-4,
                            epochs=20, a1_impl="joint", smoke_steps=20, shuffle_seed=20261017)
        cls.model, cls.controller, cls.loader, cls.record = train_acd.prepare_training(cls.cli)
        cls.initial = cls.controller.state_dict()

    def setUp(self):
        from train import set_seed
        set_seed(101)
        self.controller.load_state_dict(self.initial)
        for parameter in self.model.parameters():
            parameter.grad = None

    def test_real_all_variants_twenty_step_data_hashes_identical(self):
        from train import set_seed
        hashes = []
        for variant in VARIANTS:
            set_seed(101)
            self.controller.load_state_dict(self.initial)
            cli = Namespace(**vars(self.cli))
            cli.variant = variant
            with tempfile.TemporaryDirectory() as directory:
                cli.ref_stats_json = str(write_reference(directory))
                train_acd.train_epochs(self.model, self.controller,
                                       clone_training_loader(self.loader, 101), cli,
                                       Path(directory), copy.deepcopy(self.record))
                hashes.append((Path(directory)/"step_hashes.txt").read_bytes())
        self.assertTrue(all(value == hashes[0] for value in hashes))
        self.assertEqual(len(hashes[0].splitlines()), 20)

    def test_real_a1const_file_reference_coefficient_matches_sigmoid(self):
        batch = next(iter(clone_training_loader(self.loader, 101)))
        batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
        teacher, student = train_acd.encode_training_batch(self.model, self.controller, batch)
        with tempfile.TemporaryDirectory() as directory:
            path = write_reference(directory)
            loaded = train_acd.load_ref_stats(path)
            _, diagnostics = acd_loss(teacher.float(), student.float(), batch["pids"],
                                      "A1const", ref_mean=loaded["ref_mean_b64"])
        expected = torch.sigmoid(student.float().logsumexp(1)-8.125).mean()
        self.assertTrue(torch.equal(diagnostics["t2i_repulsion_coeff_mean"], expected))

    def test_real_teacher_ref_stats_three_batch_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            result = teacher_ref_stats.main([
                "--base_run", self.cli.base_run, "--root_dir", self.cli.root_dir,
                "--dev_split_file", self.cli.dev_split_file, "--data_seed", "101",
                "--output_json", str(Path(directory)/"stats.json"), "--num_batches", "3"])
        self.assertEqual(result["batch_count"], 3)
        self.assertEqual(result["row_count"], 192)
        self.assertIsNotNone(result["ref_mean_b64"])
        self.assertFalse(result["complete_epoch"])
        self.assertEqual(result["student_encodings"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
