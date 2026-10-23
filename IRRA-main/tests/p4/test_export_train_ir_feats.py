"""Synthetic CPU checks for the R13 1b exporter (no checkpoint/model inference)."""
from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.p4 import export_train_ir_feats as exporter


def synthetic_files(directory, *, train_count=1076):
    """One row per image, deliberately unsorted raw IDs and repeated captions."""
    root = Path(directory)
    data = root / "data" / "RGBT-PEDES" / "RGBT-PEDES-Final"
    (data / "RGB").mkdir(parents=True)
    (data / "T").mkdir()
    train_ids = list(range(900000, 900000 + train_count))
    train_order = list(reversed(train_ids))
    annos = [
        {"id": pid, "split": "train", "file_path": f"source/{pid}_{index}.jpg",
         "captions": [f"caption {index} {j}" for j in range(3 if index == 0 else 2)]}
        for index, pid in enumerate(train_order)
    ]
    # A second image of the first identity is a separate gallery item.
    annos.insert(2, {"id": train_order[0], "split": "train", "file_path": "source/again.jpg",
                     "captions": ["one", "two", "three", "four"]})
    annos += [
        {"id": 21, "split": "train", "file_path": "source/dev21.jpg", "captions": ["dev a", "dev b"]},
        {"id": 7, "split": "train", "file_path": "source/dev7.jpg", "captions": ["dev c"]},
        {"id": 50, "split": "test", "file_path": "source/test.jpg", "captions": ["official test"]},
    ]
    annotation = data / "Caption.json"
    annotation.write_text(json.dumps(annos), encoding="utf-8")
    split_data = {
        "train_reduced_ids": train_ids, "dev_ids": [7, 21],
        "n_train_reduced": len(train_ids), "n_dev": 2,
        "anno_md5": hashlib.md5(annotation.read_bytes()).hexdigest(),
    }
    split = root / "dev.json"
    split.write_text(json.dumps(split_data), encoding="utf-8")
    checkpoint = root / "base.pth"
    checkpoint.write_bytes(b"synthetic-only checkpoint identity")
    config = root / "configs.yaml"
    config.write_text("dataset_name: RGBT-PEDES\n", encoding="utf-8")
    return {
        "root": root, "root_dir": data.parents[1], "data": data,
        "annotation": annotation, "annos": annos, "split": split,
        "split_data": split_data, "checkpoint": checkpoint, "config": config,
        "train_annos": [a for a in annos if a["split"] == "train" and a["id"] in set(train_ids)],
    }


def synthetic_cli(files, **overrides):
    values = dict(
        checkpoint=str(files["checkpoint"]), config_file=str(files["config"]),
        root_dir=str(files["root_dir"]), dev_split_file=str(files["split"]),
        output=str(files["root"] / "train.npz"), batch_size=4,
        num_workers=0, device="cpu", verify_dev_npz=str(files["root"] / "base.npz"),
        limit=None,
    )
    values.update(overrides)
    return Namespace(**values)


def synthetic_config(files):
    return Namespace(
        dataset_name="RGBT-PEDES", root_dir=str(files["root_dir"]),
        dev_split_file=str(files["split"]), img_size=(384, 128),
        test_batch_size=4, num_workers=0,
    )


class ExportTrainMetadataTests(unittest.TestCase):
    def test_train_is_one_image_per_annotation_in_source_order_with_raw_pids(self):
        with tempfile.TemporaryDirectory() as directory:
            files = synthetic_files(directory)
            metadata = exporter.prepare_metadata(synthetic_cli(files), synthetic_config(files))
            annos = files["train_annos"]
            gallery = metadata["train_gallery"]
            raw_pids = [a["id"] for a in annos]
            self.assertEqual(metadata["train_raw_pids"], raw_pids)
            self.assertEqual(metadata["train_file_paths"], [a["file_path"] for a in annos])
            self.assertEqual(len(raw_pids), 1077)
            self.assertEqual(len(gallery["image_pids"]), len(annos))
            self.assertLess(len(annos), sum(len(a["captions"]) for a in annos))
            # A repeat image PID is retained; identities are remapped only in loader labels.
            self.assertEqual(raw_pids[:3], [901075, 901074, 901075])
            self.assertEqual(gallery["image_pids"][:3], [1075, 1074, 1075])
            for anno, rgb, ir in zip(annos, gallery["img_paths"], gallery["img_paths_t"]):
                self.assertEqual(Path(rgb), files["data"] / "RGB" / Path(anno["file_path"]).name)
                self.assertEqual(Path(ir), files["data"] / "T" / Path(anno["file_path"]).name)
            self.assertEqual(metadata["dev_data"]["gallery_pids"], [1, 0])
            self.assertEqual(metadata["dev_data"]["gallery_raw_pids"], [21, 7])
            self.assertEqual(metadata["dev_data"]["n_train"], 1076)
            self.assertFalse(set(raw_pids) & set(metadata["dev_data"]["gallery_raw_pids"]))

    def test_limit_selects_image_prefix_and_does_not_expand_captions(self):
        with tempfile.TemporaryDirectory() as directory:
            files = synthetic_files(directory)
            metadata = exporter.prepare_metadata(synthetic_cli(files, limit=3), synthetic_config(files))
            self.assertEqual(metadata["train_raw_pids"], [901075, 901074, 901075])
            self.assertEqual(metadata["train_file_paths"], [a["file_path"] for a in files["train_annos"][:3]])
            self.assertTrue(all(len(metadata["train_gallery"][key]) == 3
                                for key in ("image_pids", "img_paths", "img_paths_t")))
            # Limiting training export must not limit the dev verification gallery.
            self.assertEqual(metadata["dev_data"]["gallery_pids"], [1, 0])
            metadata = exporter.prepare_metadata(synthetic_cli(files, limit=100000), synthetic_config(files))
            self.assertEqual(len(metadata["train_raw_pids"]), 1077)

    def test_requires_exactly_1076_train_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            files = synthetic_files(directory, train_count=1075)
            with self.assertRaisesRegex(ValueError, "1076"):
                exporter.prepare_metadata(synthetic_cli(files), synthetic_config(files))

    def test_loader_is_sequential_per_image_with_deterministic_eval_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            files = synthetic_files(directory)
            cli, config = synthetic_cli(files, limit=3), synthetic_config(files)
            metadata = exporter.prepare_metadata(cli, config)
            loader = exporter.build_gallery_loader(metadata["train_gallery"], config, cli)
            self.assertEqual(loader.batch_size, 4)
            self.assertEqual(loader.num_workers, 0)
            self.assertFalse(loader.drop_last)
            self.assertEqual(type(loader.sampler).__name__, "SequentialSampler")
            self.assertEqual(type(loader.dataset).__name__, "ImageDatasetRGBT")
            self.assertEqual(list(loader.dataset.image_pids), [1075, 1074, 1075])
            self.assertEqual(list(loader.dataset.img_paths_rgb), metadata["train_gallery"]["img_paths"])
            self.assertEqual(list(loader.dataset.img_paths_t), metadata["train_gallery"]["img_paths_t"])
            self.assertEqual([type(t).__name__ for t in loader.dataset.transform.transforms],
                             ["Resize", "ToTensor", "Normalize"])

    def test_identity_overlap_is_rejected_by_shared_split_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            files = synthetic_files(directory)
            split = files["split_data"]
            split["dev_ids"] = [7, split["train_reduced_ids"][0]]
            files["split"].write_text(json.dumps(split), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                exporter.prepare_metadata(synthetic_cli(files), synthetic_config(files))

    def test_saved_split_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            files = synthetic_files(directory)
            saved_split = dict(files["split_data"])
            saved_split["train_reduced_ids"] = list(saved_split["train_reduced_ids"])
            # Keep each split valid, disjoint, and exactly 1076 identities; change the partition.
            old_train = saved_split["train_reduced_ids"][0]
            saved_split["train_reduced_ids"][0] = 7
            saved_split["dev_ids"] = [old_train, 21]
            saved_path = files["root"] / "other_training_split.json"
            saved_path.write_text(json.dumps(saved_split), encoding="utf-8")
            config = synthetic_config(files)
            config.dev_split_file = str(saved_path)
            with self.assertRaisesRegex(ValueError, "match|differ"):
                exporter.prepare_metadata(synthetic_cli(files), config)

    def test_missing_saved_training_split_is_not_treated_as_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            files = synthetic_files(directory)
            config = synthetic_config(files)
            config.dev_split_file = ""
            with self.assertRaisesRegex(ValueError, "no dev_split_file"):
                exporter.prepare_metadata(synthetic_cli(files), config)

    def test_inaccessible_saved_split_can_be_explicitly_relocated(self):
        with tempfile.TemporaryDirectory() as directory:
            files = synthetic_files(directory)
            config = synthetic_config(files)
            config.dev_split_file = "/nonexistent/autodl/relocated_dev_split.json"
            metadata = exporter.prepare_metadata(synthetic_cli(files), config)
            self.assertEqual(Path(metadata["training_split"]), files["split"].resolve())
            self.assertEqual(metadata["saved_training_split"], config.dev_split_file)

    def test_provenance_records_actual_file_and_identity_hashes_and_git_state(self):
        with tempfile.TemporaryDirectory() as directory:
            files = synthetic_files(directory)
            cli = synthetic_cli(files, limit=3)
            metadata = exporter.prepare_metadata(cli, synthetic_config(files))
            verification = {"path": cli.verify_dev_npz, "max_absolute_difference": 0.}
            with patch.object(exporter.subprocess, "check_output", side_effect=[
                "a" * 40 + "\n", "?? tools/p4/new_export.py\n",
            ]) as git:
                provenance = exporter.make_provenance(cli, metadata, verification)
            self.assertEqual(provenance["checkpoint_sha256"], hashlib.sha256(files["checkpoint"].read_bytes()).hexdigest())
            self.assertEqual(provenance["config_sha256"], hashlib.sha256(files["config"].read_bytes()).hexdigest())
            self.assertEqual(provenance["split_identity_sha256"], metadata["dev_data"]["split_identity_sha256"])
            self.assertEqual(provenance["git_head"], "a" * 40)
            self.assertIs(provenance["git_dirty"], True)
            self.assertEqual(provenance["git_status_porcelain"], ["?? tools/p4/new_export.py"])
            self.assertEqual(provenance["arguments"], vars(cli))
            self.assertEqual(provenance["verify_dev"], verification)
            self.assertEqual(provenance["n_images"], 3)
            self.assertEqual(provenance["n_train_reduced_identities"], 1076)
            self.assertEqual(git.call_args_list[0].args[0], ["git", "rev-parse", "HEAD"])
            self.assertEqual(git.call_args_list[1].args[0], ["git", "status", "--porcelain"])


class ExportTrainArrayTests(unittest.TestCase):
    def test_verify_dev_accepts_same_order_and_absolute_error_at_most_1em5(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "base.npz"
            original = np.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
            pids = np.asarray([1, 0], dtype=np.int64)
            np.savez(reference, **{"clean/f_ir": original, "clean/img_pids": pids})
            self.assertEqual(exporter.verify_dev_ir(original.copy(), pids.copy(), reference), 0.)
            # A representable error safely below the absolute 1e-5 threshold is accepted.
            perturbed = original.copy()
            perturbed[0, 1] += 8e-6
            error = exporter.verify_dev_ir(perturbed, pids, reference)
            self.assertGreater(error, 0.)
            self.assertLessEqual(error, 1e-5)

    def test_verify_dev_threshold_is_inclusive_and_absolute(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "base.npz"
            pids = np.asarray([0], dtype=np.int64)
            np.savez(reference, **{"clean/f_ir": np.zeros((1, 2), dtype=np.float64), "clean/img_pids": pids})
            threshold = np.asarray([[1e-5, 0.]], dtype=np.float64)
            self.assertEqual(exporter.verify_dev_ir(threshold, pids, reference), 1e-5)
            threshold[0, 0] = np.nextafter(np.float64(1e-5), np.float64(np.inf))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                exporter.verify_dev_ir(threshold, pids, reference)

    def test_verify_dev_rejects_feature_error_pid_reorder_shapes_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "base.npz"
            original = np.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
            pids = np.asarray([1, 0], dtype=np.int64)
            np.savez(reference, **{"clean/f_ir": original, "clean/img_pids": pids})
            wrong = original.copy()
            wrong[0, 1] += 2e-5
            nan = original.copy()
            nan[0, 0] = np.nan
            for features, labels in ((wrong, pids), (original, pids[::-1]),
                                      (original[:1], pids), (nan, pids)):
                with self.subTest(features=features, labels=labels), self.assertRaises(ValueError):
                    exporter.verify_dev_ir(features, labels, reference)
            np.savez(reference, **{"clean/f_rgb": original, "clean/img_pids": pids})
            with self.assertRaises((ValueError, KeyError)):
                exporter.verify_dev_ir(original, pids, reference)

    def test_write_npz_uses_float32_integer_labels_unicode_paths_and_json_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "train.npz"
            rgb = np.asarray([[1., 2.], [3., 4.]], dtype=np.float64)
            ir = np.asarray([[5., 6.], [7., 8.]], dtype=np.float64)
            provenance = {"checkpoint_sha256": "a" * 64, "split_identity_sha256": "b" * 64,
                          "git_head": "c" * 40, "git_dirty": True}
            exporter.write_npz(output, rgb, ir, [900002, 900001], ["source/a.jpg", "source/b.jpg"], provenance)
            with np.load(output, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {"f_ir", "f_rgb", "raw_pids", "file_paths", "provenance"})
                self.assertEqual(saved["f_ir"].dtype, np.dtype("float32"))
                self.assertEqual(saved["f_rgb"].dtype, np.dtype("float32"))
                self.assertIn(saved["raw_pids"].dtype.kind, "iu")
                self.assertEqual(saved["file_paths"].dtype.kind, "U")
                self.assertEqual(saved["provenance"].dtype.kind, "U")
                np.testing.assert_array_equal(saved["f_rgb"], rgb.astype(np.float32))
                np.testing.assert_array_equal(saved["f_ir"], ir.astype(np.float32))
                np.testing.assert_array_equal(saved["raw_pids"], [900002, 900001])
                self.assertEqual(saved["file_paths"].tolist(), ["source/a.jpg", "source/b.jpg"])
                self.assertEqual(json.loads(saved["provenance"].item()), provenance)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                exporter.write_npz(output, rgb * 0, ir * 0, [1, 2], ["a", "b"], {})
            self.assertEqual(output.read_bytes(), before)

    def test_existing_output_is_rejected_before_config_or_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            files = synthetic_files(directory)
            cli = synthetic_cli(files)
            output = Path(cli.output)
            output.write_bytes(b"keep existing output byte-for-byte")
            with patch.object(exporter, "_load_config", side_effect=AssertionError("must reject before loading")), \
                    patch.object(exporter, "build_inference_model", side_effect=AssertionError("no model loading")):
                with self.assertRaises(FileExistsError):
                    exporter.export(cli)
            self.assertEqual(output.read_bytes(), b"keep existing output byte-for-byte")


class ExportTrainPipelineTests(unittest.TestCase):
    def _run_mocked(self, files, cli, *, dev_error=0.):
        config = synthetic_config(files)
        encoded = []

        def encode(_model, loader, expected_pids):
            labels = list(expected_pids)
            encoded.append(labels)
            features = np.arange(len(labels) * 3, dtype=np.float32).reshape(-1, 3)
            if labels == [1, 0]:
                return features + 10., features + dev_error
            return features + 10., features

        with patch.object(exporter, "_load_config", return_value=config), \
                patch.object(exporter, "build_inference_model", return_value=object()) as build, \
                patch.object(exporter, "build_gallery_loader", side_effect=lambda gallery, _args, _cli: gallery), \
                patch.object(exporter, "encode_gallery_features", side_effect=encode):
            result = exporter.export(cli)
        build.assert_called_once()
        return result, encoded

    def test_mocked_pipeline_verifies_dev_then_exports_train_image_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            files = synthetic_files(directory)
            cli = synthetic_cli(files, limit=3)
            np.savez(cli.verify_dev_npz, **{
                "clean/f_ir": np.arange(6, dtype=np.float32).reshape(2, 3),
                "clean/img_pids": np.asarray([1, 0], dtype=np.int64),
            })
            result, encoded = self._run_mocked(files, cli)
            self.assertEqual(encoded, [[1, 0], [1075, 1074, 1075]])
            self.assertEqual(result["n_images"], 3)
            self.assertEqual(Path(result["output"]), Path(cli.output).resolve())
            with np.load(cli.output, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["raw_pids"], [901075, 901074, 901075])
                np.testing.assert_array_equal(saved["f_ir"], np.arange(9, dtype=np.float32).reshape(3, 3))
                np.testing.assert_array_equal(saved["f_rgb"], np.arange(9, dtype=np.float32).reshape(3, 3) + 10.)
                provenance = json.loads(saved["provenance"].item())
            self.assertEqual(provenance, result["provenance"])

    def test_failed_dev_verification_never_encodes_train_or_writes_output(self):
        for failure in ("feature", "pid"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                files = synthetic_files(directory)
                cli = synthetic_cli(files, limit=3)
                pids = [0, 1] if failure == "pid" else [1, 0]
                np.savez(cli.verify_dev_npz, **{
                    "clean/f_ir": np.arange(6, dtype=np.float32).reshape(2, 3),
                    "clean/img_pids": np.asarray(pids, dtype=np.int64),
                })
                with patch.object(exporter, "write_npz") as writer, \
                        patch.object(exporter, "encode_gallery_features", return_value=(
                            np.zeros((2, 3), dtype=np.float32),
                            np.arange(6, dtype=np.float32).reshape(2, 3) + (2e-5 if failure == "feature" else 0.),
                        )) as encoder, \
                        patch.object(exporter, "build_gallery_loader", return_value=object()), \
                        patch.object(exporter, "build_inference_model", return_value=object()), \
                        patch.object(exporter, "_load_config", return_value=synthetic_config(files)):
                    with self.assertRaises(ValueError):
                        exporter.export(cli)
                writer.assert_not_called()
                encoder.assert_called_once()
                self.assertFalse(Path(cli.output).exists())

    def test_invalid_numeric_arguments_fail_before_model_load_and_output_creation(self):
        cases = ({"limit": 0}, {"limit": -1}, {"batch_size": 0}, {"num_workers": -1})
        for overrides in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                files = synthetic_files(directory)
                cli = synthetic_cli(files, **overrides)
                with patch.object(exporter, "_load_config", side_effect=AssertionError("validate first")), \
                        self.assertRaises(ValueError):
                    exporter.export(cli)
                self.assertFalse(Path(cli.output).exists())

    def test_missing_or_inaccessible_dev_reference_is_rejected_before_loading(self):
        for reference in (None, "does_not_exist.npz"):
            with self.subTest(reference=reference), tempfile.TemporaryDirectory() as directory:
                files = synthetic_files(directory)
                path = None if reference is None else str(files["root"] / reference)
                cli = synthetic_cli(files, verify_dev_npz=path)
                with patch.object(exporter, "_load_config", side_effect=AssertionError("reject before config")), \
                        patch.object(exporter, "build_inference_model", side_effect=AssertionError("reject before model")):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        exporter.export(cli)
                self.assertFalse(Path(cli.output).exists())


if __name__ == "__main__":
    unittest.main()
