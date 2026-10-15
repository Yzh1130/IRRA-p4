"""CPU synthetic verification of ACD-only export routing and invariance reports."""

from contextlib import contextmanager
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from tools.p4 import export_dev_records as exporter
from test_modality_features import _AdditiveModel, _loaders, _output_bytes, _run_export, _setup_export


class _Controller:
    config = {"rank": 2, "alpha": 4., "init_seed": 20261015, "layers": 1}

    def __init__(self):
        self.active = False

    @contextmanager
    def enabled(self):
        previous = self.active
        self.active = True
        try:
            yield
        finally:
            self.active = previous


class _TokenModel(_AdditiveModel):
    def __init__(self, args, controller):
        super().__init__(args)
        self.controller = controller
        self.routes = []

    def _encode_visual_tokens(self, image, modality="rgb", return_adapter_aux=False):
        self.routes.append((self.args.eval_degrade_mode, modality, self.controller.active))
        values = image.flatten(1).float()
        if self.controller.active:
            # Nonuniform update ensures cosine scores can change, not only norms.
            values = values + torch.arange(values.shape[1], device=values.device) * .031
        return values

    def encode_rgbt_global(self, rgb, ir):
        self.gallery_calls += 1
        rgb = self._encode_visual_tokens(rgb, modality="rgb")
        ir = self._encode_visual_tokens(ir, modality="thermal")
        return rgb, ir, rgb + ir, None, None, None


def _run_token_export(cli):
    images, texts = _loaders()
    controller, models = _Controller(), []

    def build(args, num_classes):
        model = _TokenModel(args, controller)
        models.append(model)
        return model

    with patch("datasets.build_dataloader", return_value=(images, texts, 1)), \
            patch("model.build_model", side_effect=build), \
            patch("processor.processor.apply_img_local_filter_stage"), \
            patch("utils.logger.setup_logger"), \
            patch.object(exporter, "load_acd_for_export", return_value=controller):
        exporter.export(cli)
    return models[0], controller


def _read_metrics(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["scenario"]: row for row in csv.DictReader(handle, delimiter="\t")}


class ACDExportTests(unittest.TestCase):
    def test_adapter_base_hash_checked_before_injection_and_saved_structure_used(self):
        from tools.p4.acd_lora import file_sha256

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, adapter = root / "last.pth", root / "adapter.pt"
            checkpoint.write_bytes(b"strictly-loaded-base-weights")
            digest = file_sha256(checkpoint)
            config = {"rank": 3, "alpha": 9., "init_seed": 71, "layers": 2,
                      "base_sha256": digest}
            torch.save({"schema_version": 1, "config": config, "lora_state_dict": {}}, adapter)
            model, controller = object(), _Controller()
            with patch("tools.p4.acd_lora.inject_lora", return_value=controller) as inject, \
                    patch("tools.p4.acd_lora.load_adapter") as load:
                self.assertIs(exporter.load_acd_for_export(model, adapter, checkpoint), controller)
                inject.assert_called_once_with(model, rank=3, alpha=9., init_seed=71, expected_layers=2)
                load.assert_called_once_with(controller, adapter, expected_base_sha256=digest)
            config["base_sha256"] = "0" * 64
            torch.save({"schema_version": 1, "config": config, "lora_state_dict": {}}, adapter)
            with patch("tools.p4.acd_lora.inject_lora") as inject, \
                    self.assertRaisesRegex(ValueError, "does not match --checkpoint"):
                exporter.load_acd_for_export(model, adapter, checkpoint)
            inject.assert_not_called()

    def test_dropblank_available_modality_scores_and_unaffected_other_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = _setup_export(root)
            _run_export(cli)
            baseline = _output_bytes(cli.output_dir)
            self.assertFalse((Path(cli.output_dir) / "metrics_dropblank.tsv").exists())
            self.assertFalse((Path(cli.output_dir) / "export_run_record.json").exists())
            cli.save_modality_features = True
            _run_export(cli)
            self.assertEqual(baseline, _output_bytes(cli.output_dir))
            original = _read_metrics(Path(cli.output_dir) / "metrics.tsv")
            extra = _read_metrics(Path(cli.output_dir) / "metrics_dropblank.tsv")
            self.assertEqual(set(original), set(extra))
            for scenario in original:
                if scenario not in ("rgb_missing", "ir_missing"):
                    self.assertEqual(original[scenario], extra[scenario])
                    continue
                feature = torch.load(Path(cli.output_dir) / "features" / f"{scenario}.pt", weights_only=False)
                key = "f_ir" if scenario == "rgb_missing" else "f_rgb"
                scores = F.normalize(feature["text_features"], dim=1) @ F.normalize(feature[key], dim=1).T
                from utils.metrics import rank
                cmc, mean_ap, mean_inp, _ = rank(scores, feature["txt_pids"], feature["img_pids"])
                self.assertEqual([float(extra[scenario][name]) for name in exporter.METRIC_FIELDS[1:]],
                                 [float(cmc[0]), float(cmc[4]), float(cmc[9]), float(mean_ap), float(mean_inp)])
            record = json.loads((Path(cli.output_dir) / "export_run_record.json").read_text(encoding="utf-8"))
            self.assertTrue(record["git_commit"])
            self.assertEqual(record["peak_gpu_memory_bytes"], 0)
            self.assertGreater(record["elapsed_seconds"], 0)
            self.assertIn("arguments", record)
            self.assertEqual(record["seeds"]["P3_eval_degrade_seed"], 2026)

    def test_route_only_missing_rgb_thermal_and_restore_after_exception(self):
        controller = _Controller()
        model = _TokenModel(SimpleNamespace(eval_degrade_mode="none"), controller).eval()
        tensor = torch.ones(1, 3, 2, 2)
        original = model._encode_visual_tokens(tensor)
        for scenario in exporter.SCENARIO_NAMES + exporter.TRAIN_STATE_SCENARIO_NAMES:
            with exporter.acd_missing_ir_route(model, controller, scenario):
                rgb = model._encode_visual_tokens(tensor, modality="rgb")
                ir = model._encode_visual_tokens(tensor, modality="thermal")
                self.assertTrue(torch.equal(original, rgb))
                self.assertEqual(torch.equal(original, ir), scenario != "rgb_missing")
            self.assertNotIn("_encode_visual_tokens", model.__dict__)
            self.assertFalse(controller.active)
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with exporter.acd_missing_ir_route(model, controller, "rgb_missing"):
                raise RuntimeError("expected")
        self.assertNotIn("_encode_visual_tokens", model.__dict__)
        self.assertFalse(controller.active)

    def test_adapter_export_comparison_only_allows_missing_rgb_derivatives_11_and_17(self):
        for include_train in (False, True):
            with self.subTest(include_train=include_train), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cli = _setup_export(root)
                cli.output_dir = str(root / "reference")
                cli.train_state_scenarios = include_train
                cli.save_modality_features = True
                _run_token_export(cli)
                reference = Path(cli.output_dir)
                adapter = root / "adapter.pt"
                adapter.write_bytes(b"synthetic-controller-is-injected-by-test")
                cli.output_dir = str(root / "adapted")
                cli.acd_adapter = str(adapter)
                cli.compare_export = str(reference)
                cli.save_modality_features = False  # Adapter/compare enable capture implicitly.
                model, controller = _run_token_export(cli)
                self.assertFalse(controller.active)
                self.assertNotIn("_encode_visual_tokens", model.__dict__)
                self.assertEqual(sum(active for _, _, active in model.routes), 3)
                self.assertTrue(all(mode == "rgb_missing" and modality == "thermal"
                                    for mode, modality, active in model.routes if active))
                report = json.loads((Path(cli.output_dir) / "invariance.json").read_text(encoding="utf-8"))
                self.assertTrue(report["overall_pass"])
                self.assertEqual(report["unexpected_differences"], [])
                self.assertIn("features/rgb_missing.pt:f_ir", report["expected_differences"])
                self.assertIn("features/rgb_missing.pt:f_fused", report["expected_differences"])
                self.assertTrue(report["features"]["rgb_missing"]["keys"]["f_rgb"]["identical"])
                self.assertTrue(report["features"]["rgb_missing"]["keys"]["text_features"]["identical"])
                self.assertGreater(report["features"]["rgb_missing"]["keys"]["f_ir"]["max_abs_diff"], 0)
                for scenario, comparison in report["features"].items():
                    if scenario != "rgb_missing":
                        self.assertTrue(comparison["all_keys_identical"], scenario)

    def test_recursive_comparison_is_bitwise_and_checks_nested_numpy_strings(self):
        reference = {"tensor": torch.tensor([0., 1.]), "metadata": {"name": "same", "ids": [1, 2]},
                     "array": np.array([0., 2.], dtype=np.float32)}
        candidate = copy.deepcopy(reference)
        candidate["tensor"][0] = -0.
        candidate["array"][1] += .25
        candidate["metadata"]["name"] = "different"
        compared = exporter.compare_feature_values(reference, candidate)
        self.assertFalse(compared["tensor"]["identical"])
        self.assertEqual(compared["tensor"]["max_abs_diff"], 0.)
        self.assertEqual(compared["array"]["max_abs_diff"], .25)
        self.assertFalse(compared["metadata.name"]["identical"])
        self.assertFalse(compared["metadata"]["identical"])
        self.assertTrue(compared["metadata.ids[0]"]["identical"])
        self.assertTrue(compared["metadata.ids"]["identical"])

    def test_unexpected_feature_metadata_or_metric_changes_fail_with_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = _setup_export(root)
            cli.save_modality_features = True
            cli.output_dir = str(root / "reference")
            _run_export(cli)
            reference = Path(cli.output_dir)
            for kind in ("metadata", "metric", "missing_allowed_key", "dtype"):
                with self.subTest(kind=kind):
                    import shutil
                    target = root / kind
                    shutil.copytree(reference, target)
                    if kind == "metric":
                        path = target / "metrics.tsv"
                        rows = path.read_text(encoding="utf-8").splitlines()
                        values = rows[1].split("\t")
                        values[1] = str(float(values[1]) + 1)
                        rows[1] = "\t".join(values)
                        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
                    else:
                        scene = "rgb_missing" if kind != "metadata" else "clean"
                        path = target / "features" / f"{scene}.pt"
                        payload = torch.load(path, weights_only=False)
                        if kind == "metadata":
                            payload["metadata"]["normalization"] += " bad"
                        elif kind == "dtype":
                            payload["f_ir"] = payload["f_ir"].double()
                        else:
                            del payload["f_ir"]
                        torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, "Unexpected export changes"):
                        exporter.compare_exports(reference, target, allow_acd_changes=True)
                    report = json.loads((target / "invariance.json").read_text(encoding="utf-8"))
                    self.assertFalse(report["overall_pass"])
                    self.assertTrue(report["unexpected_differences"])

    def test_compare_reference_preflight_and_no_unrequested_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = _setup_export(root)
            cli.compare_export = str(root / "nonexistent")
            with patch("model.build_model") as build, self.assertRaisesRegex(ValueError, "all 11"):
                exporter.export(cli)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
