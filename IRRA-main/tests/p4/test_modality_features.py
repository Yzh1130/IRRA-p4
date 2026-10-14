"""Synthetic tests of original-path feature capture and exact P-feat analysis."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import yaml

from model.build import IRRA
from tools.p4 import export_dev_records as exporter
from tools.p4.analysis.modality_decomposition import analyze_export_dirs, decompose_features, retrieval_statistics
from utils.metrics import Evaluator


class _BatchList(list):
    pass


class _AdditiveModel(torch.nn.Module):
    """Use actual IRRA eval/fusion methods with synthetic visual/text tensors."""
    encode_rgbt_deterministic = IRRA.encode_rgbt_deterministic
    encode_rgbt_eval = IRRA.encode_rgbt_eval
    _encode_visual_global_gaussian = IRRA._encode_visual_global_gaussian
    _fuse_rgbt_global = IRRA._fuse_rgbt_global
    _maybe_eval_sampling = IRRA._maybe_eval_sampling

    def __init__(self, args):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.args = args
        self.use_rgbt_add_fusion = True
        self.use_global_uncertainty = self.use_mum = self.mraf_enabled = False
        self.eval_use_mu_only = False
        self.eval_use_fused_mu = True
        self.use_prob_sampling = True  # The actual eval method must still avoid sampling.
        self.use_modality_text_prompt = False
        self.gallery_calls = self.text_calls = 0

    def _use_diag_gaussian_global(self):
        return False

    def _log_mraf_mem_callsite(self, *args):
        pass

    def encode_rgbt_global(self, rgb, ir):
        self.gallery_calls += 1
        rgb, ir = rgb.flatten(1).float(), ir.flatten(1).float()
        return rgb, ir, rgb + ir, None, None, None

    def encode_text(self, captions):
        self.text_calls += 1
        return captions.float()


def _loaders():
    values = torch.arange(120, dtype=torch.float32).reshape(10, 3, 2, 2)
    rgb, ir = (values / 11).sin(), (values / 17 + .3).cos()
    pids = torch.arange(10).div(5, rounding_mode="floor")
    captions = rgb.flatten(1) + ir.flatten(1)
    images = _BatchList((pids[i:i + 4], rgb[i:i + 4], ir[i:i + 4]) for i in range(0, 10, 4))
    texts = _BatchList((pids[i:i + 4], captions[i:i + 4]) for i in range(0, 10, 4))
    texts.dataset = SimpleNamespace(captions=[f"red shirt {i}" for i in range(10)])
    return images, texts


def _setup_export(root):
    data_root = root / "data" / "RGBT-PEDES" / "RGBT-PEDES-Final"
    (data_root / "RGB").mkdir(parents=True)
    (data_root / "T").mkdir()
    annos = [{"id": 3, "split": "train", "file_path": "training.jpg", "captions": ["black coat"]}]
    annos += [{"id": 1 + i // 5, "split": "train", "file_path": f"{i}.jpg",
               "captions": [f"red shirt {i}"]} for i in range(10)]
    annos += [{"id": 4, "split": "test", "file_path": "test.jpg", "captions": ["blue shirt"]}]
    annotation = data_root / "Caption.json"
    annotation.write_text(json.dumps(annos), encoding="utf-8")
    split = root / "dev.json"
    split.write_text(json.dumps({"dev_ids": [1, 2], "train_reduced_ids": [3],
                                 "n_dev": 2, "n_train_reduced": 1,
                                 "anno_md5": hashlib.md5(annotation.read_bytes()).hexdigest()}), encoding="utf-8")
    config = root / "configs.yaml"
    config.write_text(yaml.safe_dump({"dataset_name": "RGBT-PEDES", "root_dir": str(root / "data"),
                                     "dev_split_file": str(split), "use_mum": False,
                                     "use_rgbt_add_fusion": True, "use_global_uncertainty": False,
                                     "eval_use_fused_mu": True, "use_batch_uncertainty": True,
                                     "use_id_uncertainty": True}), encoding="utf-8")
    checkpoint = root / "synthetic.pth"
    torch.save({"model": _AdditiveModel(SimpleNamespace()).state_dict()}, checkpoint)
    return exporter.build_parser().parse_args([
        "--checkpoint", str(checkpoint), "--config_file", str(config),
        "--dev_split_file", str(split), "--output_dir", str(root / "export"), "--device", "cpu",
    ])


def _run_export(cli, implementation=exporter.export):
    images, texts = _loaders()
    models = []

    def build(args, num_classes):
        model = _AdditiveModel(args)
        models.append(model)
        return model

    with patch("datasets.build_dataloader", return_value=(images, texts, 1)), \
            patch("model.build_model", side_effect=build), \
            patch("processor.processor.apply_img_local_filter_stage"), \
            patch("utils.logger.setup_logger"):
        result = implementation(cli)
    return result, models[0]


def _output_bytes(directory):
    directory = Path(directory)
    paths = [directory / "metrics.tsv", directory / "per_query.csv", directory / "summary.json"]
    paths += sorted((directory / "scores").glob("*.pt"))
    return {str(path.relative_to(directory)): path.read_bytes() for path in paths}


def _legacy_exporter():
    """Execute the task's trusted, fixed baseline source without writing a file."""
    source = subprocess.check_output(
        ["git", "show", "f90fede3:IRRA-main/tools/p4/export_dev_records.py"],
        cwd=exporter.CODE_ROOT.parent, text=True, encoding="utf-8",
    )
    legacy = ModuleType("p4_f90_export_baseline")
    legacy.__file__ = str(Path(exporter.__file__).resolve())
    exec(compile(source, legacy.__file__, "exec"), legacy.__dict__)
    return legacy.export


def _small_features():
    generator = torch.Generator().manual_seed(319)
    rgb, ir = torch.randn(10, 6, generator=generator), torch.randn(10, 6, generator=generator)
    text = torch.randn(4, 6, generator=generator)
    return {"f_rgb": rgb, "f_ir": ir, "f_fused": rgb + ir, "text_features": text,
            "img_pids": torch.tensor([0] * 5 + [1] * 5), "txt_pids": torch.tensor([0, 1, 0, 1])}


class ModalityFeatureTests(unittest.TestCase):
    def test_exact_decomposition_and_direct_single_feature_probe(self):
        features = _small_features()
        result = decompose_features(features)
        self.assertLess(result["max_decomposition_error"], 1e-5)
        self.assertTrue(torch.allclose(result["rgb_contribution_scores"] + result["ir_contribution_scores"],
                                       result["fused_scores"], atol=1e-5, rtol=0))
        for modality in ("rgb", "ir"):
            direct = F.normalize(features["text_features"], dim=1) @ F.normalize(features[f"f_{modality}"], dim=1).T
            self.assertTrue(torch.equal(direct, result[f"P_feat_{modality}_scores"]))
            expected = retrieval_statistics(direct, features["txt_pids"], features["img_pids"])
            actual = result["retrieval"][f"P-feat-{modality}"]
            self.assertTrue(torch.equal(expected["best_pos_rank"], actual["best_pos_rank"]))
            self.assertTrue(torch.equal(expected["AP"], actual["AP"]))
        for row in result["query_rows"]:
            self.assertAlmostEqual(row["best_positive_rgb_contribution"] + row["best_positive_ir_contribution"],
                                   row["best_positive_score"], places=5)
            self.assertAlmostEqual(row["margin_rgb_contribution"] + row["margin_ir_contribution"], row["margin"], places=5)
        for stats in result["geometry"].values():
            self.assertTrue({"mean", "q05", "q25", "q50", "q75", "q95"} <= set(stats))

    def test_nonpositive_score_shares_undefined_and_negative_parts_reported(self):
        features = {"f_rgb": torch.tensor([[1., 0.], [-2., 0.]]),
                    "f_ir": torch.tensor([[-2., 0.], [3., 0.]]),
                    "f_fused": torch.tensor([[-1., 0.], [1., 0.]]),
                    "text_features": torch.tensor([[1., 0.], [1., 0.]]),
                    "img_pids": torch.tensor([0, 1]), "txt_pids": torch.tensor([0, 1])}
        result = decompose_features(features)
        first, second = result["query_rows"]
        self.assertIsNone(first["best_positive_rgb_share"])
        self.assertIsNone(first["best_positive_ir_share"])
        self.assertEqual(second["best_positive_rgb_share"], -2.)
        self.assertEqual(second["best_positive_ir_share"], 3.)
        self.assertEqual(result["negative_contribution_fractions"], {"best_positive_rgb": .5, "best_positive_ir": .5})

    def test_saved_score_ranking_and_ties_anchor_fusion(self):
        features = {"f_rgb": torch.tensor([[1., 0.]] * 4), "f_ir": torch.zeros(4, 2),
                    "f_fused": torch.tensor([[1., 0.]] * 4), "text_features": torch.tensor([[1., 0.]]),
                    "img_pids": torch.tensor([0, 0, 1, 1]), "txt_pids": torch.tensor([0])}
        stored = torch.tensor([[1. - 4e-6, 1. - 2e-6, 1., 1. - 6e-6]])
        payload = {"sim": stored, "txt_pids": features["txt_pids"], "img_pids": features["img_pids"]}
        result = decompose_features(features, payload)
        self.assertTrue(torch.equal(result["fused_scores"], stored))
        self.assertEqual(result["query_rows"][0]["best_positive_gallery_index"], 1)
        self.assertEqual(result["query_rows"][0]["strongest_negative_gallery_index"], 2)
        self.assertEqual(int(result["retrieval"]["fused"]["best_pos_rank"][0]), 2)
        payload["sim"] = torch.ones_like(stored)
        tied = decompose_features(features, payload)
        expected_order = torch.argsort(payload["sim"], descending=True, dim=1)[0]
        expected_positive = next(int(i) for i in expected_order if int(features["img_pids"][i]) == 0)
        expected_negative = next(int(i) for i in expected_order if int(features["img_pids"][i]) != 0)
        self.assertEqual(tied["query_rows"][0]["best_positive_gallery_index"], expected_positive)
        self.assertEqual(tied["query_rows"][0]["strongest_negative_gallery_index"], expected_negative)

    def test_bad_fusion_or_score_payload_rejected(self):
        features = _small_features()
        bad = copy.deepcopy(features)
        bad["f_fused"][0, 0] += .1
        with self.assertRaisesRegex(ValueError, "not exactly"):
            decompose_features(bad)
        reference = decompose_features(features)["fused_scores"]
        payload = {"sim": reference + .01, "txt_pids": features["txt_pids"], "img_pids": features["img_pids"]}
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            decompose_features(features, payload)

    def test_original_f90_file_bytes_identical_with_flag_off_and_on_for_11_and_17(self):
        legacy = _legacy_exporter()
        for include_train in (False, True):
            with self.subTest(include_train=include_train), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cli = _setup_export(root)
                cli.train_state_scenarios = include_train
                legacy_summary, legacy_model = _run_export(cli, legacy)
                baseline = _output_bytes(cli.output_dir)
                off_summary, off_model = _run_export(cli)
                self.assertEqual(baseline, _output_bytes(cli.output_dir))
                self.assertEqual(legacy_summary, off_summary)
                self.assertFalse((Path(cli.output_dir) / "features").exists())
                cli.save_modality_features = True
                on_summary, on_model = _run_export(cli)
                self.assertEqual(baseline, _output_bytes(cli.output_dir))
                self.assertEqual(legacy_summary, on_summary)
                expected_count = 17 if include_train else 11
                feature_paths = sorted((Path(cli.output_dir) / "features").glob("*.pt"))
                self.assertEqual(len(feature_paths), expected_count)
                self.assertEqual((legacy_model.gallery_calls, legacy_model.text_calls),
                                 (on_model.gallery_calls, on_model.text_calls))
                self.assertEqual((off_model.gallery_calls, off_model.text_calls),
                                 (on_model.gallery_calls, on_model.text_calls))
                for path in feature_paths:
                    feature = torch.load(path, map_location="cpu", weights_only=False)
                    self.assertTrue(torch.equal(feature["f_fused"], feature["f_rgb"] + feature["f_ir"]))
                    self.assertEqual(feature["image_ids"].tolist(), list(range(1, 11)))
                    self.assertEqual(feature["img_indices"].tolist(), list(range(10)))
                    self.assertNotEqual(feature["image_ids"].tolist(), feature["img_indices"].tolist())
                    self.assertEqual(feature["gallery_annotation_indices"], list(range(1, 11)))
                    self.assertNotIn("_encode_visual_global_gaussian", on_model.__dict__)
                result = analyze_export_dirs([cli.output_dir], root / "analysis", n_bootstrap=5)
                self.assertEqual(len(result["checkpoints"][0]["scenarios"]), expected_count)
                with self.assertRaises(FileExistsError):
                    analyze_export_dirs([cli.output_dir], root / "analysis", n_bootstrap=5)
                with self.assertRaisesRegex(FileExistsError, "overwrite modality"):
                    _run_export(cli)

    def test_probe_feat_runs_saved_feature_analysis_and_zero_stays_unsupported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = _setup_export(root)
            cli.probe = "feat"
            _run_export(cli)
            self.assertTrue((Path(cli.output_dir) / "features" / "clean.pt").is_file())
            probe_summary = json.loads((Path(cli.output_dir) / "probes" / "feat" / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("P-feat-rgb", probe_summary["checkpoints"][0]["scenarios"]["clean"]["metrics"])
            cli.probe = "zero"
            with self.assertRaisesRegex(NotImplementedError, "P-zero"):
                exporter.export(cli)

    def test_multiple_checkpoints_common_11_of_17_order_validation_and_no_wrong_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = _setup_export(root)
            cli.save_modality_features = True
            cli.output_dir = str(root / "base")
            _run_export(cli)
            cli.output_dir = str(root / "candidate")
            cli.train_state_scenarios = True
            _run_export(cli)
            result = analyze_export_dirs([root / "base", root / "candidate"], root / "paired", n_bootstrap=5)
            comparison = result["paired_checkpoint_comparisons"][0]
            self.assertEqual(len(comparison["comparison_scenarios"]), 11)
            self.assertEqual(len(comparison["per_scenario"]), 33)
            self.assertNotIn("mean_all", comparison)
            self.assertNotIn("mean_degraded", comparison)
            self.assertTrue(all(value["R1"]["difference"] == 0 for value in comparison["per_scenario"].values()))
            for feature_path in (root / "candidate" / "features").glob("*.pt"):
                feature = torch.load(feature_path, map_location="cpu", weights_only=False)
                feature["query_ids"][0], feature["query_ids"][1] = feature["query_ids"][1], feature["query_ids"][0]
                torch.save(feature, feature_path)
            with self.assertRaisesRegex(ValueError, "ordering differs"):
                analyze_export_dirs([root / "base", root / "candidate"], root / "bad_pair", n_bootstrap=5)
            self.assertFalse((root / "bad_pair").exists())

    def test_incomplete_scenes_and_reranked_scores_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cli = _setup_export(root)
            cli.save_modality_features = True
            _run_export(cli)
            score_path = Path(cli.output_dir) / "scores" / "clean.pt"
            payload = torch.load(score_path, map_location="cpu", weights_only=False)
            payload["metadata"]["use_qalr"] = True
            torch.save(payload, score_path)
            with self.assertRaisesRegex(ValueError, "postprocessing"):
                analyze_export_dirs([cli.output_dir], root / "bad_scores", n_bootstrap=5)
            (Path(cli.output_dir) / "features" / "clean.pt").unlink()
            with self.assertRaisesRegex(ValueError, "all 11"):
                analyze_export_dirs([cli.output_dir], root / "incomplete", n_bootstrap=5)

    def test_recorder_restores_methods_on_failure_and_rejects_active_auxiliary_only(self):
        images, texts = _loaders()
        args = SimpleNamespace(use_batch_uncertainty=True, use_id_uncertainty=True)
        model = _AdditiveModel(args).eval()
        evaluator = Evaluator(images, texts)
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with exporter.ModalityFeatureRecorder(evaluator, model):
                with patch.object(model, "encode_rgbt_global", side_effect=RuntimeError("synthetic failure")):
                    evaluator._encode_gallery_feature(model, images[0][1], images[0][2])
        self.assertNotIn("_encode_visual_global_gaussian", model.__dict__)
        self.assertNotIn("_encode_gallery_feature", evaluator.__dict__)
        self.assertNotIn("_encode_text_feature", evaluator.__dict__)
        for flag in ("use_mum", "mraf_enabled", "use_global_uncertainty", "eval_use_mu_only"):
            setattr(model, flag, True)
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "additive"):
                exporter.ModalityFeatureRecorder(evaluator, model)
            setattr(model, flag, False)


if __name__ == "__main__":
    unittest.main()
