"""Subprocess checks for the immutable round-nine calibration tools."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


HAS_PANDAS = importlib.util.find_spec("pandas") is not None
CODE_ROOT = Path(__file__).resolve().parents[2]
CALIB = CODE_ROOT / "tools" / "p4" / "analysis" / "calib_protocol.py"
PACK = CODE_ROOT / "tools" / "p4" / "pack_feats.py"
SCENARIOS = (
    "clean", "rgb_missing", "ir_missing", "rgb_noise_010",
    "ir_noise_010", "rgb_blur_k7", "ir_blur_k7",
)


@unittest.skipUnless(HAS_PANDAS, "pandas is not installed; calibration protocol tests require pandas")
class CalibrationProtocolTests(unittest.TestCase):
    def run_python(self, *args):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Anaconda NumPy and PyTorch can ship separate Intel OpenMP runtimes.
        # Keep immutable tools intact and select MKL's supported sequential
        # backend only in the isolated child; never allow duplicate runtimes.
        env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
        result = subprocess.run(
            [sys.executable, "-B", *map(str, args)],
            cwd=CODE_ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        return result

    def run_script(self, script, *args):
        return self.run_python(script, *args)

    def test_synthetic_run(self):
        rng = np.random.default_rng(20261015)
        gp = np.repeat(np.arange(16, dtype=np.int64), 3)
        tp = np.repeat(np.arange(16, dtype=np.int64), 4)
        prototypes = rng.normal(size=(16, 32)).astype(np.float32)
        arrays = {
            "clean/text_features": (prototypes[tp] + 0.35 * rng.normal(size=(64, 32))).astype(np.float32),
            "clean/txt_pids": tp,
            "clean/img_pids": gp,
            "clean/query_raw_pids": tp + 1000,
        }
        for scenario in SCENARIOS:
            for key in ("f_rgb", "f_ir"):
                arrays[f"{scenario}/{key}"] = (
                    prototypes[gp] + 0.45 * rng.normal(size=(48, 32))
                ).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix="p4_calib_synth_") as temp:
            root = Path(temp)
            np.savez_compressed(root / "synth.npz", **arrays)
            assignments = root / "assignments.csv"
            with assignments.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["mixture", "seed", "gallery_index", "state"])
                writer.writeheader()
                for seed in (2026, 2027, 2028):
                    assign_rng = np.random.default_rng(seed)
                    choices = {
                        "M1": assign_rng.choice(
                            ["clean", "rgb_noise_010", "ir_noise_010", "rgb_missing", "ir_missing", "rgb_blur_k7"], 48
                        ),
                        "M2": assign_rng.choice(["clean", "rgb_noise_010"], 48),
                        "M3": assign_rng.choice(["clean", "rgb_missing"], 48, p=[0.8, 0.2]),
                    }
                    for mixture, states in choices.items():
                        for index, state in enumerate(states):
                            writer.writerow({"mixture": mixture, "seed": seed, "gallery_index": index, "state": state})
            output = root / "not_created_yet"
            self.run_script(
                CALIB, "--feats_dir", root, "--models", "synth",
                "--assignments", assignments, "--out_dir", output,
                "--steps", 20, "--split_seeds", 2026, 2027, "--n_boot", 50,
            )
            expected = {
                "splits.json", "params.jsonl", "assignments_M4.csv", "perq_synth.npz",
                "summary.csv", "comparisons.csv", "branch_mass.csv", "meta.json",
            }
            self.assertTrue(expected.issubset({path.name for path in output.iterdir()}))
            import pandas as pd
            summary = pd.read_csv(output / "summary.csv")
            self.assertFalse(summary[["R1", "mAP"]].isna().any().any())
            rows = summary[summary["group"] == "all"]
            for (protocol, mixture), group in rows.groupby(["protocol", "mixture"]):
                expected_count = 384 if mixture in {"M1", "M2", "M3", "M4"} else 128
                self.assertEqual(set(group["n_query_evals"]), {expected_count}, (protocol, mixture))
                expected_methods = {"raw_add", "drop", "bstar"}
                if protocol == "P-ind":
                    expected_methods |= {"rk_affine", "rk_affine_hub", "rk_pwl", "rk_item"}
                self.assertEqual(set(group["method"]), expected_methods, (protocol, mixture))
            splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
            self.assertEqual(set(splits), {"2026", "2027"})
            for folds in splits.values():
                fold0, fold1 = set(folds["fold0"]), set(folds["fold1"])
                self.assertFalse(fold0 & fold1)
                self.assertEqual(fold0 | fold1, set(range(1000, 1016)))

    def test_pack_feats_roundtrip(self):
        generator = torch.Generator().manual_seed(20261015)
        rgb = torch.randn(48, 32, generator=generator)
        ir = torch.randn(48, 32, generator=generator)
        text = torch.randn(64, 32, generator=generator)
        img_pids = torch.arange(16).repeat_interleave(3)
        txt_pids = torch.arange(16).repeat_interleave(4)
        with tempfile.TemporaryDirectory(prefix="p4_pack_feats_") as temp:
            root = Path(temp)
            export = root / "export"
            features = export / "features"
            features.mkdir(parents=True)
            for scenario in SCENARIOS:
                torch.save({
                    "scenario": scenario,
                    "f_rgb": rgb.clone(),
                    "f_ir": ir + 0.25 if scenario == "rgb_missing" else ir.clone(),
                    "f_fused": rgb + ir,
                    "text_features": text.clone(),
                    "img_pids": img_pids.clone(),
                    "txt_pids": txt_pids.clone(),
                    "query_raw_pids": txt_pids + 1000,
                    "metadata": {"purpose": "synthetic roundtrip", "seed": 20261015},
                }, features / f"{scenario}.pt")
            output = root / "packed"
            self.run_script(PACK, "--export_dirs", export, "--names", "synth", "--out_dir", output)
            path = output / "synth.npz"
            with np.load(path) as archive:
                self.assertIn("rgb_missing/f_ir", archive.files)
                np.testing.assert_array_equal(archive["rgb_missing/f_ir"], (ir + 0.25).numpy())
                for scenario in SCENARIOS[1:]:
                    self.assertNotIn(f"{scenario}/f_rgb", archive.files)
                    if scenario != "rgb_missing":
                        self.assertNotIn(f"{scenario}/f_ir", archive.files)
            manifest = json.loads((output / "synth_manifest.json").read_text(encoding="utf-8"))
            for scenario in SCENARIOS:
                self.assertEqual(manifest[f"{scenario}/f_fused"], "skipped_fused")
            for scenario in SCENARIOS[1:]:
                self.assertEqual(manifest[f"{scenario}/f_rgb"], "same_as_clean")
                self.assertEqual(manifest[f"{scenario}/text_features"], "same_as_clean")
                if scenario != "rgb_missing":
                    self.assertEqual(manifest[f"{scenario}/f_ir"], "same_as_clean")
            # Feats intentionally performs NumPy matrix products. Keep this
            # import and BLAS use in the same isolated child as the CLI checks.
            recovered_path = root / "recovered.npz"
            self.run_python("-c", "\n".join([
                "import importlib.util, numpy as np, sys",
                "spec = importlib.util.spec_from_file_location('p4_immutable_calib_protocol_test', sys.argv[1])",
                "module = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(module)",
                "features = module.Feats(sys.argv[2])",
                "arrays = {f'{s}/f_rgb': v for s, v in features.fr.items()}",
                "arrays.update({f'{s}/f_ir': v for s, v in features.fi.items()})",
                "np.savez(sys.argv[3], **arrays)",
            ]), CALIB, path, recovered_path)
            with np.load(recovered_path) as archive:
                recovered = {key: archive[key].copy() for key in archive.files}
            for scenario in SCENARIOS:
                np.testing.assert_array_equal(recovered[f"{scenario}/f_rgb"], rgb.numpy())
                expected = (ir + 0.25).numpy() if scenario == "rgb_missing" else ir.numpy()
                np.testing.assert_array_equal(recovered[f"{scenario}/f_ir"], expected)


if __name__ == "__main__":
    unittest.main()
