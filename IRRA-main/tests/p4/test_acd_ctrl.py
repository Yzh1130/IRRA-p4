"""Pull controls: closed gradients, private FIFO, data parity, and Git regression."""
from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
for location in (CODE_ROOT, Path(__file__).resolve().parent):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

# Import helpers rather than TestCase classes: discovery must not rerun old tests.
from test_acd import REAL_SKIP, clone_training_loader, synthetic_loader, tiny_setup, smoke_cli
from test_acd_mech import assert_rng_equal, logits_fixture, rng_snapshot, write_reference
from tools.p4 import acd_losses, train_acd
from tools.p4.acd_losses import acd_loss, control_t2i, positive_targets, uniform_ce
from tools.p4.acd_queue import NegativeFeatureQueue

BASELINE = "7e796485b5e0e2e57d1e4002908bf42014f2dacf"
OLD_VARIANTS = ("A0", "A0bal", "A1", "Pull", "A1const", "A1shuf", "DCLbal")
CONTROLS = ("TempCE", "MarginCE", "QueueCE")
ALL_VARIANTS = OLD_VARIANTS + CONTROLS


def baseline_module(filename):
    """Execute only the explicitly requested, trusted repository commit."""
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:IRRA-main/tools/p4/{filename}.py"],
        cwd=CODE_ROOT, text=True, encoding="utf-8")
    module = types.ModuleType(f"_r10_ctrl_baseline_{filename}")
    module.__file__ = str(CODE_ROOT / "tools" / "p4" / f"{filename}.py")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def control_options(variant, pids, *, requires_grad=False):
    generator = torch.Generator(device="cpu").manual_seed(284)
    queued = torch.randn(len(pids), 20, generator=generator, dtype=torch.float64)
    queued.requires_grad_(requires_grad)
    queue_pids = torch.tensor([8, 2, 19, 4, 5, 8, 40, 41, 42, 19,
                               43, 44, 45, 46, 47, 48, 49, 50, 51, 52])
    options = {"t2i_tau": .07, "t2i_margin": .2}
    if variant == "QueueCE":
        options.update(queue_logits=queued, queue_pids=queue_pids)
    return options, queued, queue_pids


def own_probabilities(student, pids, variant, options):
    positive = pids[:, None].eq(pids[None, :])
    if variant == "TempCE":
        logits = student.detach() * (.02 / options["t2i_tau"])
    elif variant == "MarginCE":
        logits = student.detach() - 50.0 * options["t2i_margin"] * positive
    else:
        queue = options["queue_logits"].detach().masked_fill(
            pids[:, None].eq(options["queue_pids"][None, :]), -torch.inf)
        logits = torch.cat((student.detach(), queue), 1)
    return logits.softmax(1)


class ACDControlGradientTests(unittest.TestCase):
    def test_fp64_closed_gradients_teacher_and_queue_are_frozen(self):
        teacher, student, pids = logits_fixture()
        q = positive_targets(pids, student.dtype)
        for variant in CONTROLS:
            with self.subTest(variant=variant):
                options, queued, _ = control_options(variant, pids, requires_grad=True)
                loss, _ = control_t2i(student, pids, variant, **options)
                gradient, queue_gradient = torch.autograd.grad(
                    loss, (student, queued), allow_unused=True)
                probabilities = own_probabilities(student, pids, variant, options)
                expected = .5 * (probabilities[:, :len(pids)] - q) / len(pids)
                self.assertLessEqual(float((gradient - expected).abs().max()), 1e-12)
                self.assertIsNone(queue_gradient)
                total, _ = acd_loss(teacher, student, pids, variant, **options)
                _, frozen_teacher, frozen_queue = torch.autograd.grad(
                    total, (student, teacher, queued), allow_unused=True)
                self.assertIsNone(frozen_teacher)
                self.assertIsNone(frozen_queue)

    def test_degenerate_controls_match_a0bal_loss_and_gradient(self):
        _, student, pids = logits_fixture()
        expected = .5 * uniform_ce(student, pids)
        expected_gradient = torch.autograd.grad(expected, student)[0]
        cases = (("TempCE", {"t2i_tau": .02}),
                 ("MarginCE", {"t2i_margin": 0}),
                 ("QueueCE", {"queue_logits": student.new_empty((12, 0)),
                               "queue_pids": pids.new_empty((0,))}))
        for variant, options in cases:
            with self.subTest(variant=variant):
                loss, _ = control_t2i(student, pids, variant, **options)
                gradient = torch.autograd.grad(loss, student)[0]
                self.assertLessEqual(float((loss - expected).abs()), 1e-12)
                self.assertLessEqual(float((gradient - expected_gradient).abs().max()), 1e-12)

    def test_queue_excludes_all_same_pid_items_even_with_huge_logits(self):
        _, student, pids = logits_fixture()
        options, queued, queue_pids = control_options("QueueCE", pids)
        forbidden = pids[:, None].eq(queue_pids[None, :])
        changed = queued.masked_fill(forbidden, 1e20)
        original, _ = control_t2i(student, pids, "QueueCE", **options)
        original_gradient = torch.autograd.grad(original, student)[0]
        options["queue_logits"] = changed
        altered, _ = control_t2i(student, pids, "QueueCE", **options)
        altered_gradient = torch.autograd.grad(altered, student)[0]
        self.assertTrue(torch.equal(original, altered))
        self.assertTrue(torch.equal(original_gradient, altered_gradient))

    def test_own_softmax_diagnostics_and_unmodified_50_scale_ce(self):
        teacher, student, pids = logits_fixture()
        positive = pids[:, None].eq(pids[None, :])
        for variant in CONTROLS:
            with self.subTest(variant=variant):
                options, _, _ = control_options(variant, pids)
                _, diagnostics = acd_loss(teacher, student, pids, variant, **options)
                probabilities = own_probabilities(student, pids, variant, options)
                batch_probabilities = probabilities[:, :len(pids)]
                expected_positive = (batch_probabilities * positive).sum(1).mean()
                expected_negative = (batch_probabilities * ~positive).sum(1).mean()
                torch.testing.assert_close(diagnostics["t2i_pos_prob_mean"], expected_positive,
                                           atol=1e-15, rtol=1e-15)
                torch.testing.assert_close(diagnostics["t2i_batch_neg_mass_mean"], expected_negative,
                                           atol=1e-15, rtol=1e-15)
                self.assertTrue(torch.equal(diagnostics["l_I_t2i"], uniform_ce(student, pids)))
                self.assertTrue(torch.equal(diagnostics["l_I_i2t"], uniform_ce(student.t(), pids)))
                if variant == "QueueCE":
                    expected_mass = probabilities[:, len(pids):].sum(1).mean()
                    torch.testing.assert_close(diagnostics["t2i_queue_mass_mean"], expected_mass,
                                               atol=1e-15, rtol=1e-15)
                    self.assertEqual(int(diagnostics["queue_len"]), 20)
                    mass = (diagnostics["t2i_pos_prob_mean"]
                            + diagnostics["t2i_batch_neg_mass_mean"]
                            + diagnostics["t2i_queue_mass_mean"])
                else:
                    mass = (diagnostics["t2i_pos_prob_mean"]
                            + diagnostics["t2i_batch_neg_mass_mean"])
                self.assertLessEqual(abs(float(mass) - 1), 1e-14)

    def test_all_ten_variants_i2t_gradient_is_bitwise_identical(self):
        teacher, student, pids = logits_fixture()
        expected = torch.autograd.grad(uniform_ce(student.t(), pids), student)[0]
        original_ce = acd_losses.uniform_ce
        for variant in ALL_VARIANTS:
            captured = []
            def capture(logits, labels):
                value = original_ce(logits, labels)
                if logits.data_ptr() == student.data_ptr() and logits.stride() == student.t().stride():
                    captured.append(value)
                return value
            options, _, _ = control_options(variant, pids)
            options.update(ref_mean=2.75, permutation=torch.arange(12).roll(1))
            with self.subTest(variant=variant), patch.object(acd_losses, "uniform_ce", side_effect=capture):
                acd_loss(teacher, student, pids, variant, **options)
                self.assertEqual(len(captured), 1)
                actual = torch.autograd.grad(captured[0], student)[0]
                self.assertTrue(torch.equal(expected, actual))

    def test_old_seven_fp64_losses_gradients_and_all_diagnostics_match_git(self):
        baseline = baseline_module("acd_losses")
        teacher, student, pids = logits_fixture()
        options = {"ref_mean": 2.75, "permutation": torch.arange(12).roll(3)}
        for variant in OLD_VARIANTS:
            for implementation in ("joint", "decomposed"):
                with self.subTest(variant=variant, implementation=implementation):
                    old_loss, old_diagnostics = baseline.acd_loss(
                        teacher, student, pids, variant, implementation, **options)
                    new_loss, new_diagnostics = acd_loss(
                        teacher, student, pids, variant, implementation, **options)
                    self.assertTrue(torch.equal(old_loss, new_loss))
                    self.assertEqual(old_diagnostics.keys(), new_diagnostics.keys())
                    for key in old_diagnostics:
                        self.assertTrue(torch.equal(old_diagnostics[key], new_diagnostics[key]), key)
                    old_gradient = torch.autograd.grad(old_loss, student)[0]
                    new_gradient = torch.autograd.grad(new_loss, student)[0]
                    self.assertTrue(torch.equal(old_gradient, new_gradient))


class NegativeFeatureQueueTests(unittest.TestCase):
    def test_fifo_capacity_detach_fp32_normalization_and_no_input_alias(self):
        queue = NegativeFeatureQueue(5)
        generator = torch.Generator().manual_seed(77)
        # The encoder supplies L2-normalized features; the queue preserves their
        # FP32 values instead of normalizing a second time and rounding again.
        first = F.normalize(torch.randn(3, 8, generator=generator), dim=1).double().requires_grad_()
        second = F.normalize(torch.randn(4, 8, generator=generator), dim=1).double().requires_grad_()
        first_expected = first.detach().float().clone()
        second_expected = second.detach().float().clone()
        queue.enqueue(first, torch.tensor([10, 11, 12]))
        self.assertEqual(len(queue), 3)
        self.assertEqual(queue.features.dtype, torch.float32)
        self.assertFalse(queue.features.requires_grad)
        with torch.no_grad():
            first.zero_()
        self.assertTrue(torch.equal(queue.features, first_expected))
        queue.enqueue(second, torch.tensor([20, 21, 22, 23]))
        self.assertEqual(len(queue), 5)
        self.assertEqual(queue.pids.tolist(), [12, 20, 21, 22, 23])
        self.assertTrue(torch.equal(queue.features, torch.cat((first_expected, second_expected))[-5:]))
        self.assertFalse(queue.features.requires_grad)
        torch.testing.assert_close(queue.features.norm(dim=1), torch.ones(5), atol=2e-7, rtol=0)

    def test_queue_logits_mask_same_pid_detach_text_and_empty_queue(self):
        queue = NegativeFeatureQueue(20)
        generator = torch.Generator().manual_seed(78)
        text = F.normalize(torch.randn(4, 8, generator=generator), dim=1).requires_grad_()
        query_pids = torch.tensor([1, 2, 3, 1])
        self.assertEqual(tuple(queue.logits(text, query_pids).shape), (4, 0))
        images = torch.randn(6, 8, generator=generator, requires_grad=True)
        queue_pids = torch.tensor([1, 1, 2, 3, 7, 1])
        queue.enqueue(images, queue_pids)
        logits = queue.logits(text, query_pids)
        positive = query_pids[:, None].eq(queue_pids[None, :])
        expected = (50.0 * (text.detach() @ queue.features.t())).masked_fill(positive, -torch.inf)
        self.assertTrue(torch.equal(logits, expected))
        self.assertFalse(logits.requires_grad)
        self.assertTrue(torch.isneginf(logits[positive]).all())

    def test_control_losses_and_fifo_do_not_read_or_write_global_rng(self):
        teacher, student, pids = logits_fixture()
        fixtures = {variant: control_options(variant, pids)[0] for variant in CONTROLS}
        generator = torch.Generator(device="cpu").manual_seed(93)
        features = torch.randn(12, 8, generator=generator)
        text = F.normalize(torch.randn(12, 8, generator=generator), dim=1)
        before = rng_snapshot()
        queue = NegativeFeatureQueue(20)
        for _ in range(3):
            queue.logits(text, pids)
            queue.enqueue(features, pids)
            for variant in CONTROLS:
                acd_loss(teacher, student, pids, variant, **fixtures[variant])
        assert_rng_equal(self, before, rng_snapshot())


class ACDControlTrainingTests(unittest.TestCase):
    def test_feature_encoding_matches_original_logits_bitwise(self):
        model, controller = tiny_setup()
        batch = next(iter(synthetic_loader()))
        before = rng_snapshot()
        expected_teacher, expected_student = train_acd.encode_training_batch(model, controller, batch)
        teacher, student, text, features = train_acd.encode_training_batch_with_features(
            model, controller, batch)
        self.assertTrue(torch.equal(teacher, expected_teacher))
        self.assertTrue(torch.equal(student, expected_student))
        self.assertTrue(torch.equal(student, 50.0 * (text @ features.t())))
        self.assertFalse(teacher.requires_grad)
        self.assertFalse(text.requires_grad)
        self.assertTrue(features.requires_grad)
        self.assertFalse(controller.active)
        torch.testing.assert_close(text.norm(dim=1), torch.ones(4), atol=2e-7, rtol=0)
        torch.testing.assert_close(features.norm(dim=1), torch.ones(4), atol=2e-7, rtol=0)
        assert_rng_equal(self, before, rng_snapshot())

    def test_all_ten_twenty_step_data_hashes_identical_and_queue_trajectory(self):
        hashes = []
        for variant in ALL_VARIANTS:
            model, controller = tiny_setup()
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                cli = smoke_cli(variant, 20)
                cli.ref_stats_json = str(write_reference(directory))
                cli.shuffle_seed, cli.t2i_tau, cli.t2i_margin, cli.queue_size = 20261017, .07, .2, 11
                record = train_acd.train_epochs(model, controller, synthetic_loader(), cli,
                                               Path(directory), {"base_checkpoint_sha256": "a" * 64})
                hashes.append((Path(directory) / "step_hashes.txt").read_bytes())
                if variant in CONTROLS:
                    self.assertEqual(record["t2i_tau"], .07)
                    self.assertEqual(record["t2i_margin"], .2)
                    self.assertEqual(record["t2i_margin_logit"], 10.)
                    self.assertEqual(record["queue_size"], 11)
                if variant == "QueueCE":
                    rows = [json.loads(line) for line in (Path(directory) / "step_log.jsonl").read_text().splitlines()]
                    self.assertEqual([row["queue_len"] for row in rows],
                                     [min(11, 4 * step) for step in range(20)])
                    self.assertTrue(all(isinstance(row["queue_len"], int) for row in rows))
                    self.assertEqual(record["queue_len_at_loss"], [
                        {"step": step, "queue_len": min(11, 4 * (step - 1))}
                        for step in (1, 2, 10, 20)])
        self.assertTrue(all(value == hashes[0] for value in hashes))
        self.assertEqual(len(hashes[0].splitlines()), 20)

    def test_queue_enqueue_is_after_optimizer_and_uses_only_previous_steps(self):
        model, controller = tiny_setup()
        events, loss_lengths = [], []
        real_loss, real_step, real_enqueue = acd_loss, torch.optim.AdamW.step, NegativeFeatureQueue.enqueue
        def captured_loss(*args, **kwargs):
            events.append("loss")
            queued = kwargs["queue_logits"]
            loss_lengths.append(0 if queued is None else queued.shape[1])
            return real_loss(*args, **kwargs)
        def captured_step(optimizer, *args, **kwargs):
            events.append("optimizer")
            return real_step(optimizer, *args, **kwargs)
        def captured_enqueue(queue, *args, **kwargs):
            events.append("enqueue")
            return real_enqueue(queue, *args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(train_acd, "acd_loss", side_effect=captured_loss), \
                patch.object(torch.optim.AdamW, "step", captured_step), \
                patch.object(NegativeFeatureQueue, "enqueue", captured_enqueue):
            cli = smoke_cli("QueueCE", 5)
            cli.queue_size = 9
            train_acd.train_epochs(model, controller, synthetic_loader(), cli, Path(directory),
                                   {"base_checkpoint_sha256": "a" * 64})
        self.assertEqual(events, ["loss", "optimizer", "enqueue"] * 5)
        self.assertEqual(loss_lengths, [0, 4, 8, 9, 9])

    def test_queue_records_all_requested_steps_and_last_without_duplicate(self):
        model, controller = tiny_setup()
        cli = smoke_cli("QueueCE", 75)
        cli.epochs, cli.queue_size = 13, 256
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            record = train_acd.train_epochs(model, controller, synthetic_loader(), cli,
                                           Path(directory), {"base_checkpoint_sha256": "a" * 64})
        self.assertEqual(record["queue_len_at_loss"], [
            {"step": step, "queue_len": min(256, 4 * (step - 1))}
            for step in (1, 2, 10, 64, 65, 74, 75)])

    def test_old_seven_default_argument_records_match_git_baseline(self):
        baseline = baseline_module("train_acd")
        for variant in OLD_VARIANTS:
            arguments = ["--base_run", "base", "--root_dir", "data", "--dev_split_file", "split",
                         "--variant", variant, "--output_dir", "out"]
            old_cli = baseline.build_parser().parse_args(arguments)
            new_cli = train_acd.build_parser().parse_args(arguments)
            with self.subTest(variant=variant):
                self.assertEqual(baseline._run_record_arguments(old_cli),
                                 train_acd._run_record_arguments(new_cli))
        for variant in CONTROLS:
            cli = train_acd.build_parser().parse_args([
                "--base_run", "base", "--root_dir", "data", "--dev_split_file", "split",
                "--variant", variant, "--output_dir", "out"])
            recorded = train_acd._run_record_arguments(cli)
            self.assertEqual(recorded["t2i_tau"], .07)
            self.assertEqual(recorded["t2i_margin"], .2)
            self.assertEqual(recorded["queue_size"], 4096)

    def test_invalid_control_options_fail_before_output_directory_creation(self):
        cases = (("--t2i_tau", "0"), ("--t2i_tau", "-0.1"), ("--t2i_tau", "nan"),
                 ("--t2i_tau", "inf"), ("--t2i_margin", "-0.1"),
                 ("--t2i_margin", "nan"), ("--t2i_margin", "inf"),
                 ("--queue_size", "0"), ("--queue_size", "-2"))
        with tempfile.TemporaryDirectory() as directory:
            for index, (option, value) in enumerate(cases):
                output = Path(directory) / f"not-created-{index}"
                argv = ["--base_run", "missing", "--root_dir", "missing", "--dev_split_file", "missing",
                        "--variant", "Pull", "--output_dir", str(output), option, value]
                with self.subTest(option=option, value=value), \
                        self.assertRaises((ValueError, SystemExit)):
                    train_acd.main(argv)
                self.assertFalse(output.exists())

    def test_old_seven_logs_records_adapter_bytes_match_fixed_clock_git_baseline(self):
        baseline = baseline_module("train_acd")
        baseline.acd_loss = baseline_module("acd_losses").acd_loss
        with tempfile.TemporaryDirectory() as reference_directory:
            reference = str(write_reference(reference_directory))
            for variant in OLD_VARIANTS:
                outputs = []
                for trainer in (baseline.train_epochs, train_acd.train_epochs):
                    model, controller = tiny_setup()
                    cli = smoke_cli(variant, 5)
                    cli.ref_stats_json, cli.shuffle_seed = reference, 20261017
                    # Defaults must not add fields, logs, or extra tensor operations.
                    cli.t2i_tau, cli.t2i_margin, cli.queue_size = .07, .2, 4096
                    with tempfile.TemporaryDirectory() as directory, \
                            patch("time.perf_counter", return_value=0.0), redirect_stdout(io.StringIO()):
                        record = trainer(model, controller, synthetic_loader(), cli, Path(directory),
                                         {"base_checkpoint_sha256": "a" * 64})
                        outputs.append((copy.deepcopy(record), {
                            filename: (Path(directory) / filename).read_bytes() for filename in (
                                "step_log.jsonl", "step_hashes.txt", "train_log.txt", "adapter.pt")},
                            {key: value.clone() for key, value in controller.state_dict().items()}))
                with self.subTest(variant=variant):
                    self.assertEqual(outputs[0][0], outputs[1][0])
                    self.assertEqual(outputs[0][1], outputs[1][1])
                    for key in outputs[0][2]:
                        self.assertTrue(torch.equal(outputs[0][2][key], outputs[1][2][key]), key)


@unittest.skipIf(bool(REAL_SKIP), "Real-checkpoint R10 control test skipped: " + REAL_SKIP)
class ACDControlRealCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls.cli = Namespace(base_run=os.environ["BASE_RUN"], root_dir=os.environ["DATA_ROOT"],
                            dev_split_file=os.environ["DEV_SPLIT"], lora_rank=8, lora_alpha=16,
                            lora_init_seed=20261015, data_seed=101, variant="Pull", lr=1e-4,
                            epochs=20, a1_impl="joint", smoke_steps=20, shuffle_seed=20261017,
                            t2i_tau=.07, t2i_margin=.2, queue_size=4096)
        cls.model, cls.controller, cls.loader, cls.record = train_acd.prepare_training(cls.cli)
        cls.initial = cls.controller.state_dict()

    def test_real_pull_and_three_controls_twenty_step_data_hashes_identical(self):
        from train import set_seed
        hashes = []
        for variant in ("Pull",) + CONTROLS:
            set_seed(101)
            self.controller.load_state_dict(self.initial)
            cli = Namespace(**vars(self.cli))
            cli.variant = variant
            with tempfile.TemporaryDirectory() as directory:
                train_acd.train_epochs(self.model, self.controller,
                                       clone_training_loader(self.loader, 101), cli,
                                       Path(directory), copy.deepcopy(self.record))
                hashes.append((Path(directory) / "step_hashes.txt").read_bytes())
        self.assertTrue(all(value == hashes[0] for value in hashes))
        self.assertEqual(len(hashes[0].splitlines()), 20)

    def test_real_queue_five_steps_finite_and_loss_time_lengths(self):
        from train import set_seed
        set_seed(101)
        self.controller.load_state_dict(self.initial)
        cli = Namespace(**vars(self.cli))
        cli.variant, cli.smoke_steps = "QueueCE", 5
        with tempfile.TemporaryDirectory() as directory:
            train_acd.train_epochs(self.model, self.controller,
                                   clone_training_loader(self.loader, 101), cli,
                                   Path(directory), copy.deepcopy(self.record))
            rows = [json.loads(line) for line in (Path(directory) / "step_log.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertEqual([row["queue_len"] for row in rows], [64 * k for k in range(5)])
        self.assertTrue(all(row["loss"] is not None and torch.isfinite(torch.tensor(row["loss"]))
                            for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
