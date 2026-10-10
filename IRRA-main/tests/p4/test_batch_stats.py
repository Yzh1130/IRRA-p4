"""CPU-only diagnostics tests; synthetic tensors, no training or data access."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from model import objectives
from tools.p4.deg_retrieval import batch_label_statistics
from tools.p4.train_batch_stats import (
    TRAIN_STATES, empirical_ks, gradient_measurements, load_kappa,
    loss_from_logits, merge_label_statistics, pair_distributions,
    positive_targets, state_fingerprint, summarize_batches, weight_variants,
)
from tools.p4 import train_batch_stats


class BatchStatsTest(unittest.TestCase):
    def kappas(self, value=0.8):
        return {state: {"raw": value, "clipped": value} for state in TRAIN_STATES}

    def test_uniform_matches_existing_sdm_and_ce(self):
        image = torch.tensor([[1., 2., 1.], [2., -1., 2.], [-1., 1., 2.]])
        text = torch.tensor([[2., 1., 1.], [1., -2., 1.], [1., 2., -1.]])
        pids = torch.tensor([1, 1, 2])
        scale = torch.tensor(3.)
        logits = scale * ((text / text.norm(dim=1, keepdim=True))
                          @ (image / image.norm(dim=1, keepdim=True)).t())
        torch.testing.assert_close(
            loss_from_logits(logits, pids, "sdm"),
            objectives.compute_sdm(image, text, pids, scale), rtol=1e-6, atol=1e-6,
        )
        torch.testing.assert_close(
            loss_from_logits(logits, pids, "ce"),
            objectives.compute_multipos_ce(image, text, pids, scale),
        )

    def test_weighted_targets_keep_only_pid_positives_each_direction(self):
        pids = torch.tensor([1, 1, 2])
        a = torch.tensor([[.1, .9, .4], [.3, .7, .2], [.6, .8, .5]])
        positive, q_text, q_image = positive_targets(pids, a)
        expected_text = torch.tensor([[.1, .9, 0.], [.3, .7, 0.], [0., 0., 1.]])
        expected_image = torch.tensor([[.25, .75, 0.], [.5625, .4375, 0.], [0., 0., 1.]])
        torch.testing.assert_close(q_text, expected_text)
        torch.testing.assert_close(q_image, expected_image)
        self.assertTrue(torch.equal(q_text > 0, positive))
        self.assertTrue(torch.equal(q_image > 0, positive))
        torch.testing.assert_close(q_text.sum(1), torch.ones(3))
        torch.testing.assert_close(q_image.sum(1), torch.ones(3))

    def test_E_uses_actual_positive_pairs_not_all_cross_pid_pairs(self):
        pids = torch.tensor([1, 1, 2])
        states = ["rgb_drop", "ir_drop", "rgb_drop"]
        # Positive rgb_drop weights: .2, 1.0, .6 -> mean .6.
        # Including the three negative pairs would give a different mean.
        variants, details = weight_variants(
            pids, [1., 0., .5], states, self.kappas(), .05,
            torch.Generator().manual_seed(7),
        )
        self.assertAlmostEqual(details["state_positive_means"]["rgb_drop"], .6, places=6)
        torch.testing.assert_close(variants["E"][:, [0, 2]], torch.full((3, 2), .6))
        self.assertAlmostEqual(details["state_positive_means"]["ir_drop"], .6, places=6)

    def test_shuffle_stream_is_repeatable_and_preserves_global_rng(self):
        pids = torch.tensor([1, 1, 2, 3, 3, 4])
        states = list(TRAIN_STATES)
        global_before = torch.random.get_rng_state().clone()
        arguments = (pids, [0., .1, .2, .5, .7, 1.], states, self.kappas(), .05)
        left, left_detail = weight_variants(*arguments, torch.Generator().manual_seed(11))
        right, right_detail = weight_variants(*arguments, torch.Generator().manual_seed(11))
        self.assertEqual(left_detail["permutation"], right_detail["permutation"])
        self.assertTrue(torch.equal(left["D-shuf"], right["D-shuf"]))
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))
        self.assertEqual(sorted(left_detail["permutation"]), list(range(6)))

    def test_effective_weights_and_KS_use_actual_positive_pair_distribution(self):
        pids = torch.tensor([1, 1, 2])
        states = ["rgb_drop", "rgb_drop", "ir_drop"]
        variants, details = weight_variants(
            pids, [1., 0., .5], states, self.kappas(), .05,
            torch.Generator().manual_seed(7),
        )
        pairs = pair_distributions(pids, states, variants, details)
        self.assertEqual(len(pairs["rgb_drop"]["D"]["a"]), 4)
        self.assertEqual(len(pairs["ir_drop"]["D"]["a"]), 1)
        torch.testing.assert_close(torch.tensor(pairs["rgb_drop"]["D"]["e"]),
                                   torch.tensor([.1, .1, .5, .5]))
        self.assertAlmostEqual(pairs["ir_drop"]["D"]["e"][0], .6, places=6)
        self.assertEqual(empirical_ks([1, 1, 2], [1, 1, 2]), 0.)
        self.assertEqual(empirical_ks([1, 1], [2, 2]), 1.)
        self.assertAlmostEqual(empirical_ks([1, 2, 2, 3], [1, 2, 3, 3]), .25)
        self.assertIsNone(empirical_ks([], [1]))

    def test_gradients_are_distinct_finite_and_do_not_update_parameters(self):
        encoder = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            encoder.weight.copy_(torch.tensor([[1., .1, .2], [.2, 1., .1], [.1, .2, 1.]]))
        encoder.eval()
        before = state_fingerprint(encoder)
        # One shared encoder supplies both clean reused and re-encoded features.
        clean = encoder(torch.tensor([[1., 2., 1.], [2., 1., 2.], [1., 1., 2.]]))
        degraded = encoder(torch.tensor([[.2, .4, .2], [2., .2, 1.], [.1, .1, .2]]))
        image = clean + degraded
        text = torch.tensor([[2., 1., 1.], [1., 2., 1.], [-1., 1., 2.]])
        logits = 3. * ((text / text.norm(dim=1, keepdim=True))
                       @ (image / image.norm(dim=1, keepdim=True)).t())
        pids = torch.tensor([1, 1, 2])
        states = ["rgb_drop", "ir_noise", "ir_attenuate"]
        variants, details = weight_variants(
            pids, [.1, .5, 1.], states, self.kappas(), .05,
            torch.Generator().manual_seed(3),
        )
        measurements = gradient_measurements(logits, pids, variants, encoder.parameters())
        for by_variant in measurements.values():
            for record in by_variant.values():
                self.assertTrue(record["finite"])
                self.assertGreater(record["score_gradient_frobenius"], 0)
                self.assertGreater(record["encoder_parameter_gradient_l2"], 0)
                self.assertEqual(record["encoder_parameters_with_gradient"], 1)
                self.assertNotEqual(record["score_gradient_frobenius"],
                                    record["encoder_parameter_gradient_l2"])
            self.assertEqual(by_variant["uniform"]["relative_to_uniform"]["loss"], 0.)
        self.assertEqual(before, state_fingerprint(encoder))
        self.assertIsNone(encoder.weight.grad)
        pairs = pair_distributions(pids, states, variants, details)
        aggregate = summarize_batches([{"measurements": measurements}], pairs)
        self.assertIn("D_vs_D_shuf_effective_weight_KS_by_state", aggregate)
        # All fields must remain JSON-safe, including missing-state KS values.
        json.dumps(aggregate, allow_nan=False)

    def test_kappa_raw_clip_and_missing_state_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kappa.json"
            values = {state: {"R1": 10.} for state in TRAIN_STATES}
            values["rgb_drop"] = {"raw": 1.4, "clipped": 1.}
            values["ir_drop"] = {"raw": -.2, "clipped": 0.}
            path.write_text(json.dumps({"R1_clean": 20., "states": values}), encoding="utf-8")
            result = load_kappa(path)
            self.assertEqual(result["rgb_drop"]["raw"], 1.4)
            self.assertEqual(result["rgb_drop"]["clipped"], 1.)
            self.assertEqual(result["ir_drop"]["raw"], -.2)
            self.assertEqual(result["ir_drop"]["clipped"], 0.)
            self.assertEqual(result["rgb_noise"]["raw"], .5)
            del values["rgb_attenuate"]
            path.write_text(json.dumps({"R1_clean": 20., "states": values}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing measured kappa"):
                load_kappa(path)

    def test_label_histogram_aggregate_has_correct_row_counts(self):
        first = batch_label_statistics(torch.tensor([1, 1, 2]), torch.tensor([10, 10, 20]))
        second = batch_label_statistics(torch.tensor([3, 4]), torch.tensor([30, 40]))
        aggregate = merge_label_statistics([first, second], [3, 2])
        for direction in ("t2i", "i2t"):
            positives = aggregate[direction]["positive_columns"]
            self.assertEqual(positives["histogram"], {"1": 3, "2": 2})
            self.assertAlmostEqual(positives["mean"], 1.4)
            self.assertEqual(aggregate[direction]["duplicate_positive_columns"]["histogram"],
                             {"0": 3, "1": 2})

    def test_cli_synthetic_loader_strict_checkpoint_and_no_extra_batch(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_model = torch.nn.Module()
                self.base_model.visual = torch.nn.Linear(4, 3, bias=False)
                self.logit_scale = torch.tensor(2.)

            def forward(self, batch, mdrc_feature_only=False,
                        p4_clean_features=None, p4_mode_meta=None):
                visual = self.base_model.visual
                rgb = visual(batch["images"].flatten(1))
                ir = visual(batch["thermal_images"].flatten(1))
                if p4_clean_features is not None:
                    # Mimic reuse of the clean unmodified modality with gradient.
                    rgb = torch.where(p4_mode_meta["rgb_mask"][:, None], rgb,
                                      p4_clean_features["v_rgb_c"])
                    ir = torch.where(p4_mode_meta["ir_mask"][:, None], ir,
                                     p4_clean_features["v_t_c"])
                return {
                    "fused_det_feat": rgb + ir, "v_rgb_c": rgb, "v_t_c": ir,
                    "text_det_feat": batch["text_features"],
                    "p4_encoder_calls": {"rgb": 1, "ir": 1},
                }

        batch = {
            "pids": torch.tensor([1, 1, 2]), "image_ids": torch.tensor([10, 10, 20]),
            "images": torch.tensor([[[[1., 2.], [2., 1.]]],
                                     [[[2., 1.], [1., 2.]]],
                                     [[[1., 3.], [1., 1.]]]]),
            "thermal_images": torch.tensor([[[[2., 3.], [1., 1.]]],
                                             [[[1., 2.], [2., 3.]]],
                                             [[[2., 1.], [3., 1.]]]]),
            "text_features": torch.tensor([[2., 1., 1.], [1., 2., 1.], [-1., 1., 2.]]),
            "raw_captions": ["a red shirt", "black shoes", "green coat"],
        }
        produced = []

        def batches():
            for i in range(3):
                produced.append(i)
                yield batch

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "checkpoint.pth"
            source = ToyModel()
            torch.save({"model": source.state_dict(), "epoch": 3}, checkpoint)
            config = directory / "config.yaml"
            config.write_text("batch_size: 3\n", encoding="utf-8")
            kappa = directory / "kappa.json"
            kappa.write_text(json.dumps({state: .8 for state in TRAIN_STATES}), encoding="utf-8")
            args = SimpleNamespace(batch_size=3, num_workers=0, root_dir="synthetic",
                                   dev_split_file="synthetic-split", train_seed=101,
                                   deg_rng_seed=321)
            target = ToyModel()
            with patch("datasets.build_dataloader", return_value=(batches(), None, None, 2)), \
                 patch("model.build_model", return_value=target), \
                 patch("train.set_seed", wraps=__import__("train").set_seed) as seed_training, \
                 patch.object(train_batch_stats, "_load_training_args", return_value=args):
                result = train_batch_stats.main([
                    "--checkpoint", str(checkpoint), "--config_file", str(config),
                    "--kappa_json", str(kappa), "--output_dir", str(directory / "output"),
                    "--num_batches", "2", "--device", "cpu",
                ])
                seed_training.assert_called_once_with(101)
            self.assertEqual(produced, [0, 1])
            self.assertEqual(result["num_batches_observed"], 2)
            self.assertEqual(result["deg_rng_seed"], 321)
            self.assertEqual(result["shuffle_seed"], 322)
            self.assertTrue(result["parameters_unchanged"])
            self.assertEqual(result["model_state_sha256_before"],
                             result["model_state_sha256_after"])
            self.assertEqual(result["checkpoint_epoch"], 3)
            self.assertTrue(all(parameter.requires_grad for parameter in target.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))
            self.assertTrue((directory / "output" / "stats.json").is_file())
            json.dumps(result, allow_nan=False)

    def test_cli_rejects_invalid_batch_workers_or_nonfinite_floor(self):
        required = ["--checkpoint", "unused", "--config_file", "unused",
                    "--kappa_json", "unused", "--output_dir", "unused"]
        for option, value, message in (
            ("--batch_size", "0", "batch_size must be positive"),
            ("--num_workers", "-1", "num_workers must be nonnegative"),
            ("--floor", "nan", "floor in"),
            ("--floor", "inf", "floor in"),
        ):
            with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, message):
                train_batch_stats.main(required + [option, value])


if __name__ == "__main__":
    unittest.main()
