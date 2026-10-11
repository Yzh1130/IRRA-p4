"""Training-defined gallery scenarios reuse Evaluator without changing P3 modes."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from tools.p4.export_dev_records import (
    CORE_SCENARIOS, SCENARIO_NAMES, TRAIN_STATE_SCENARIOS,
    TRAIN_STATE_SCENARIO_NAMES, build_parser, build_summary, export,
    make_train_state_evaluator,
)
from utils.mdrc import _apply_mode
from utils.metrics import Evaluator


class _BatchList(list):
    pass


def _same_tensor_bits(first, second):
    return (first.dtype == second.dtype and first.shape == second.shape and
            torch.equal(first.contiguous().view(torch.uint8), second.contiguous().view(torch.uint8)))


class _SyntheticModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.args = args
        self.gallery_inputs = []

    def encode_text(self, captions):
        return captions.float()

    def encode_rgbt_eval(self, rgb, ir):
        self.gallery_inputs.append((rgb.detach().clone(), ir.detach().clone()))
        return rgb.flatten(1) + 0.7 * ir.flatten(1)


def _synthetic_loaders(batch_size=4):
    values = torch.arange(120, dtype=torch.float32).reshape(10, 3, 2, 2)
    rgb, ir = (values / 11).sin(), (values / 17 + 0.3).cos()
    pids = torch.arange(10).div(5, rounding_mode="floor")
    captions = rgb.flatten(1) + 0.7 * ir.flatten(1)
    images = _BatchList((pids[i:i + batch_size], rgb[i:i + batch_size], ir[i:i + batch_size])
                        for i in range(0, 10, batch_size))
    texts = _BatchList((pids[i:i + batch_size], captions[i:i + batch_size])
                       for i in range(0, 10, batch_size))
    texts.dataset = SimpleNamespace(captions=[f"red shirt {i}" for i in range(10)])
    return images, texts, rgb, ir


class TrainStateExportTests(unittest.TestCase):
    def _run_train_state(self, mode, batch_size=4, seed=2026):
        image_loader, text_loader, rgb, ir = _synthetic_loaders(batch_size)
        model = _SyntheticModel(SimpleNamespace(eval_degrade_mode=mode, eval_degrade_seed=seed))
        evaluator = make_train_state_evaluator(image_loader, text_loader, mode, seed,
                                               expected_gallery_size=10)
        global_state = torch.get_rng_state().clone()
        evaluator.eval(model)
        self.assertTrue(torch.equal(global_state, torch.get_rng_state()))
        degraded = tuple(torch.cat([batch[which] for batch in model.gallery_inputs])
                         for which in (0, 1))
        return evaluator, degraded, rgb, ir

    def test_all_six_run_and_match_training_function(self):
        for scenario, mode in TRAIN_STATE_SCENARIOS:
            with self.subTest(scenario=scenario):
                evaluator, degraded, rgb, ir = self._run_train_state(mode)
                target = rgb if mode.startswith("rgb_") else ir
                reference_generator = torch.Generator(device="cpu").manual_seed(2026)
                expected_rows, factors = [], []
                for row in target.split(1):
                    factor = None
                    if mode.endswith("_attenuate"):
                        factor = torch.empty(1).uniform_(0.2, 0.6, generator=reference_generator)
                        factors.append(float(factor.item()))
                    expected_rows.append(_apply_mode(row, torch.ones(1, dtype=torch.bool), mode,
                                                    generator=reference_generator, factors=factor))
                affected = 0 if mode.startswith("rgb_") else 1
                self.assertTrue(_same_tensor_bits(degraded[affected], torch.cat(expected_rows)))
                self.assertTrue(_same_tensor_bits(degraded[1 - affected], ir if affected == 0 else rgb))
                self.assertEqual(evaluator.training_state_record()["gallery_images_degraded_once"], 10)
                self.assertEqual(evaluator.training_state_record()["attenuation_factors_gallery_order"], factors)
                self.assertEqual(set(evaluator.last_eval_result), {"R1", "R5", "R10", "mAP", "mINP"})
                if mode.endswith("_attenuate"):
                    self.assertTrue(all(0.2 <= value <= 0.6 for value in factors))

    def test_same_seed_bitwise_batch_size_independent_and_seed_changes_noise(self):
        for _, mode in TRAIN_STATE_SCENARIOS:
            with self.subTest(mode=mode):
                first, tensors, _, _ = self._run_train_state(mode, batch_size=4)
                repeat, repeated, _, _ = self._run_train_state(mode, batch_size=4)
                resized, regrouped, _, _ = self._run_train_state(mode, batch_size=3)
                self.assertEqual(first.last_eval_result, repeat.last_eval_result)
                self.assertEqual(first.training_state_record(), resized.training_state_record())
                for a, b, c in zip(tensors, repeated, regrouped):
                    self.assertTrue(_same_tensor_bits(a, b))
                    self.assertTrue(_same_tensor_bits(a, c))
                if not mode.endswith("_drop"):
                    _, other, _, _ = self._run_train_state(mode, seed=2027)
                    affected = 0 if mode.startswith("rgb_") else 1
                    self.assertFalse(_same_tensor_bits(tensors[affected], other[affected]))

    def test_original_eleven_are_unchanged_when_training_scenarios_run(self):
        def run_core():
            results = []
            for _, mode, strength, kernel in CORE_SCENARIOS:
                images, texts, _, _ = _synthetic_loaders()
                args = SimpleNamespace(eval_degrade_mode=mode, eval_degrade_strength=strength,
                                       eval_degrade_blur_kernel=kernel, eval_degrade_seed=2026)
                model = _SyntheticModel(args)
                evaluator = Evaluator(images, texts)
                evaluator.eval(model)
                results.append((dict(evaluator.last_eval_result), model.gallery_inputs))
            return results

        before = run_core()
        for _, mode in TRAIN_STATE_SCENARIOS:
            self._run_train_state(mode)
        after = run_core()
        for (before_metrics, before_tensors), (after_metrics, after_tensors) in zip(before, after):
            self.assertEqual(before_metrics, after_metrics)
            for before_batch, after_batch in zip(before_tensors, after_tensors):
                for a, b in zip(before_batch, after_batch):
                    self.assertTrue(_same_tensor_bits(a, b))

    def test_gallery_single_pass_guard_and_invalid_mode(self):
        images, texts, rgb, ir = _synthetic_loaders()
        evaluator = make_train_state_evaluator(images, texts, "rgb_drop", expected_gallery_size=10)
        with self.assertRaisesRegex(ValueError, "every gallery image"):
            evaluator.training_state_record()
        evaluator._apply_eval_degradation(None, rgb, ir)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            evaluator._apply_eval_degradation(None, rgb[:1], ir[:1])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            make_train_state_evaluator(images, texts, "rgb_missing")

    def test_summary_keeps_original_ten_scenario_mean(self):
        core = [{"scenario": name, "R1": index, "mAP": 20 + index}
                for index, name in enumerate(SCENARIO_NAMES)]
        extra = [{"scenario": name, "R1": 100, "R5": 100, "R10": 100,
                  "mAP": 100, "mINP": 100} for name in TRAIN_STATE_SCENARIO_NAMES]
        old, combined = build_summary(core), build_summary(core + extra)
        for field in ("clean_R1", "clean_mAP", "degraded_R1_mean", "degraded_scenarios"):
            self.assertEqual(old[field], combined[field])
        self.assertEqual(list(combined["train_state_metrics"]), list(TRAIN_STATE_SCENARIO_NAMES))
        with self.assertRaises(ValueError):
            build_summary(core + extra[:-1])

    def test_export_writes_seventeen_scenarios_and_preserves_original_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data" / "RGBT-PEDES" / "RGBT-PEDES-Final"
            (data_root / "RGB").mkdir(parents=True)
            (data_root / "T").mkdir()
            annos = [{"id": 1 + i // 5, "split": "train", "file_path": f"{i}.jpg",
                      "captions": [f"red shirt {i}"]} for i in range(10)]
            annos += [{"id": 3, "split": "train", "file_path": "train.jpg", "captions": ["black coat"]},
                      {"id": 4, "split": "test", "file_path": "test.jpg", "captions": ["blue shirt"]}]
            annotation = data_root / "Caption.json"
            annotation.write_text(json.dumps(annos), encoding="utf-8")
            split = root / "dev.json"
            split.write_text(json.dumps({"dev_ids": [1, 2], "train_reduced_ids": [3],
                                         "n_dev": 2, "n_train_reduced": 1,
                                         "anno_md5": hashlib.md5(annotation.read_bytes()).hexdigest()}),
                             encoding="utf-8")
            config = root / "configs.yaml"
            config.write_text(yaml.safe_dump({"dataset_name": "RGBT-PEDES", "root_dir": str(root / "data"),
                                              "dev_split_file": str(split)}), encoding="utf-8")
            checkpoint = root / "synthetic.pth"
            torch.save({"model": _SyntheticModel(SimpleNamespace()).state_dict()}, checkpoint)
            images, texts, _, _ = _synthetic_loaders()
            cli = build_parser().parse_args(["--checkpoint", str(checkpoint), "--config_file", str(config),
                                             "--dev_split_file", str(split), "--output_dir", str(root / "core"),
                                             "--device", "cpu"])
            with patch("datasets.build_dataloader", return_value=(images, texts, 1)), \
                    patch("model.build_model", side_effect=lambda args, num_classes: _SyntheticModel(args)), \
                    patch("processor.processor.apply_img_local_filter_stage"), \
                    patch("utils.logger.setup_logger"):
                core_summary = export(cli)
                cli.output_dir = str(root / "combined")
                cli.train_state_scenarios = True
                combined_summary = export(cli)
                cli.output_dir = str(root / "repeat")
                repeat_summary = export(cli)
                cli.output_dir = str(root / "other_seed")
                cli.eval_degrade_seed = 2027
                other_summary = export(cli)
                cli.train_state_scenarios = False
                with self.assertRaisesRegex(ValueError, "fixes --eval_degrade_seed=2026"):
                    export(cli)
            with (root / "core" / "metrics.tsv").open(encoding="utf-8", newline="") as handle:
                core_rows = list(csv.DictReader(handle, delimiter="\t"))
            with (root / "combined" / "metrics.tsv").open(encoding="utf-8", newline="") as handle:
                combined_rows = list(csv.DictReader(handle, delimiter="\t"))
            with (root / "combined" / "per_query.csv").open(encoding="utf-8", newline="") as handle:
                query_rows = list(csv.DictReader(handle))
            with (root / "core" / "per_query.csv").open(encoding="utf-8", newline="") as handle:
                core_query_rows = list(csv.DictReader(handle))
            with (root / "repeat" / "metrics.tsv").open(encoding="utf-8", newline="") as handle:
                repeat_rows = list(csv.DictReader(handle, delimiter="\t"))
            with (root / "repeat" / "per_query.csv").open(encoding="utf-8", newline="") as handle:
                repeat_query_rows = list(csv.DictReader(handle))
            self.assertEqual(len(combined_rows), 17)
            self.assertEqual(combined_rows[:11], core_rows)
            self.assertEqual([row["scenario"] for row in combined_rows[11:]], list(TRAIN_STATE_SCENARIO_NAMES))
            self.assertEqual(len(query_rows), 17 * 10)
            self.assertEqual(query_rows[:11 * 10], core_query_rows)
            self.assertEqual(combined_rows, repeat_rows)
            self.assertEqual(query_rows, repeat_query_rows)
            self.assertEqual(set(row["scenario"] for row in query_rows),
                             set(SCENARIO_NAMES) | set(TRAIN_STATE_SCENARIO_NAMES))
            self.assertEqual(core_summary["degraded_R1_mean"], combined_summary["degraded_R1_mean"])
            self.assertEqual(len(combined_summary["provenance"]["training_state_degradation"]), 6)
            self.assertEqual(combined_summary["train_state_metrics"], repeat_summary["train_state_metrics"])
            self.assertEqual(other_summary["provenance"]["eval_degrade_seed"], 2026)
            self.assertEqual(other_summary["provenance"]["train_state_eval_degrade_seed"], 2027)
            self.assertTrue(all(record["eval_degrade_seed"] == 2027 for record in
                                other_summary["provenance"]["training_state_degradation"].values()))
            for name in SCENARIO_NAMES + TRAIN_STATE_SCENARIO_NAMES:
                combined_payload = torch.load(root / "combined" / "scores" / f"{name}.pt",
                                              map_location="cpu", weights_only=False)
                repeat_payload = torch.load(root / "repeat" / "scores" / f"{name}.pt",
                                            map_location="cpu", weights_only=False)
                other_payload = torch.load(root / "other_seed" / "scores" / f"{name}.pt",
                                           map_location="cpu", weights_only=False)
                for key in ("sim", "txt_pids", "img_pids", "txt_indices", "img_indices"):
                    self.assertTrue(_same_tensor_bits(combined_payload[key], repeat_payload[key]))
                    if name in SCENARIO_NAMES:
                        core_payload = torch.load(root / "core" / "scores" / f"{name}.pt",
                                                  map_location="cpu", weights_only=False)
                        self.assertTrue(_same_tensor_bits(core_payload[key], combined_payload[key]))
                        self.assertTrue(_same_tensor_bits(core_payload[key], other_payload[key]))
                if "noise_train" in name:
                    self.assertFalse(_same_tensor_bits(combined_payload["sim"], other_payload["sim"]))
            self.assertEqual(json.loads((root / "combined" / "summary.json").read_text(encoding="utf-8")),
                             combined_summary)


if __name__ == "__main__":
    unittest.main()
