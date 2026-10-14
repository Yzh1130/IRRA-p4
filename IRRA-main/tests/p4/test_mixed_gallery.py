"""Synthetic CPU coverage for verified mixed-gallery diagnostics."""

import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

from tools.p4.export_dev_records import SCENARIO_NAMES, TRAIN_STATE_SCENARIO_NAMES, per_query_records, validate_dev_split
from tools.p4.analysis.mixed_gallery_from_scores import (
    analyze_verified_export, assign_gallery_states, clustered_difference, clustered_differences, load_verified_export,
    positive_negative_mix, positive_pair_ranks, ranking_arrays, run_analysis,
    select_state_columns, top1_error_state_rates, verify_homogeneous_records,
)


class MixedGalleryTests(unittest.TestCase):
    def make_export(self, root, name="base", train_states=False):
        annotation = root / "annotation.json"
        split = root / "split.json"
        annos = [
            {"id": 10, "split": "train", "captions": ["red coat", "blue bag"]},
            {"id": 10, "split": "train", "captions": ["black shoes"]},
            {"id": 20, "split": "train", "captions": ["green coat"]},
            {"id": 20, "split": "train", "captions": ["white bag"]},
            {"id": 30, "split": "train", "captions": ["gray coat"]},
            {"id": 40, "split": "test", "captions": ["brown shirt"]},
        ]
        annotation.write_text(json.dumps(annos), encoding="utf-8")
        split.write_text(json.dumps({"dev_ids": [10, 20], "train_reduced_ids": [30],
                                     "n_dev": 2, "n_train_reduced": 1,
                                     "anno_md5": hashlib.md5(annotation.read_bytes()).hexdigest()}), encoding="utf-8")
        data = validate_dev_split(annotation, split)
        export_dir = root / name
        (export_dir / "scores").mkdir(parents=True)
        generator = torch.Generator().manual_seed(55)
        base = torch.rand((5, 4), generator=generator)
        payloads, rows = {}, []
        names = (*SCENARIO_NAMES, *TRAIN_STATE_SCENARIO_NAMES) if train_states else SCENARIO_NAMES
        for index, scenario in enumerate(names):
            sim = torch.roll(base, shifts=index % 4, dims=1)
            payload = {
                "sim": sim, "txt_pids": torch.tensor([q["eval_pid"] for q in data["queries"]]),
                "img_pids": torch.tensor(data["gallery_pids"]), "txt_indices": torch.arange(5),
                "img_indices": torch.arange(4), "metadata": {"score_semantics": "global_cosine_similarity",
                "score_shape": [5, 4], "test_ckpt_file": f"/{name}/last.pth", "model_name": name},
            }
            payloads[scenario] = payload
            torch.save(payload, export_dir / "scores" / f"{scenario}.pt")
            rows.extend(per_query_records(payload, data, scenario))
        self.write_rows(export_dir / "per_query.csv", rows)
        return export_dir, annotation, split, data, payloads, rows

    @staticmethod
    def write_rows(path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_column_selection_and_positive_negative_selection(self):
        a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        b = a + 10
        mixed = select_state_columns({"a": a, "b": b}, ["b", "a", "b"])
        torch.testing.assert_close(mixed, torch.tensor([[11., 2., 13.], [14., 5., 16.]]), rtol=0, atol=0)
        result = positive_negative_mix(a, b, torch.tensor([0, 1]), torch.tensor([0, 0, 1]))
        torch.testing.assert_close(result, torch.tensor([[1., 2., 13.], [14., 15., 6.]]), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            select_state_columns({"a": a}, ["a"])

    def test_eleven_and_seventeen_homogeneous_selfchecks_csv_precision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for train_states in (False, True):
                export, annotation, split, data, payloads, rows = self.make_export(
                    root, "all17" if train_states else "core11", train_states)
                actual = load_verified_export(export, split, annotation)
                self.assertEqual(len(actual["scores"]), 17 if train_states else 11)
                verified = verify_homogeneous_records(payloads, data, rows)
                self.assertEqual(len(verified), len(payloads))
                for row in verified["clean"]:
                    expected = next(r for r in rows if r["scenario"] == "clean" and r["query_id"] == row["query_id"])
                    self.assertEqual(row["best_pos_rank"], expected["best_pos_rank"])
                    self.assertEqual(row["AP"], expected["AP"])
                changed = [{**rows[0], "AP": float(np.nextafter(rows[0]["AP"], float("inf")))}, *rows[1:]]
                with self.assertRaisesRegex(ValueError, "self-check.*AP"):
                    verify_homogeneous_records(payloads, data, changed)
                if train_states:
                    changed_train = [*rows[:-1], {**rows[-1], "best_pos_rank": rows[-1]["best_pos_rank"] + 1}]
                    with self.assertRaisesRegex(ValueError, TRAIN_STATE_SCENARIO_NAMES[-1]):
                        verify_homogeneous_records(payloads, data, changed_train)

    def test_historical_safe_config_and_inactive_uncertainty_switches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export, annotation, split, _, payloads, _ = self.make_export(root)
            fixture = Path(__file__).with_name("fixtures") / "devA_hist.yaml"
            for scenario, payload in payloads.items():
                payload["metadata"]["config_file"] = str(fixture)
                torch.save(payload, export / "scores" / f"{scenario}.pt")
            verified = load_verified_export(export, split, annotation)
            self.assertEqual(verified["source_configs_checked"], [str(fixture.resolve())])

    def test_failed_ap_rank_source_and_order_checks_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export, annotation, split, _, payloads, rows = self.make_export(root)
            original = export / "per_query.csv"
            for changed in ({"AP": rows[0]["AP"] - .01 if rows[0]["AP"] >= .01 else rows[0]["AP"] + .01},
                            {"best_pos_rank": rows[0]["best_pos_rank"] + 1}):
                self.write_rows(original, [{**rows[0], **changed}, *rows[1:]])
                with self.assertRaisesRegex(ValueError, "self-check"):
                    run_analysis([export], split, root / "failed", annotation_file=annotation, n_resamples=10)
                self.assertFalse((root / "failed").exists())
            self.write_rows(original, rows)
            bad = {**payloads["clean"], "img_indices": torch.tensor([1, 0, 2, 3])}
            torch.save(bad, export / "scores" / "clean.pt")
            with self.assertRaisesRegex(ValueError, "Noncanonical"):
                load_verified_export(export, split, annotation)
            bad = {**payloads["clean"], "metadata": {**payloads["clean"]["metadata"], "test_ckpt_file": "/other.pth"}}
            torch.save(bad, export / "scores" / "clean.pt")
            with self.assertRaisesRegex(ValueError, "mix checkpoint"):
                load_verified_export(export, split, annotation)
            bad["metadata"]["score_semantics"] = "qalr_rank_score"
            torch.save(bad, export / "scores" / "clean.pt")
            with self.assertRaisesRegex(ValueError, "global_cosine"):
                load_verified_export(export, split, annotation)

    def test_positive_pair_rank_counts_only_strictly_higher_negatives(self):
        qids, gids = torch.tensor([0]), torch.tensor([0, 0, 1, 2])
        # Other positive at 9 does not count; tied negative at 3 does not count.
        mixed = torch.tensor([[3., 9., 3., 4.]])
        homogeneous = torch.tensor([[3., 9., 2., 3.]])
        rows = positive_pair_ranks(mixed, {"clean": homogeneous}, qids, gids,
                                   ["clean"] * 4, [{"query_id": "0:0", "pid": 10}])
        self.assertEqual([(r["mixed_positive_pair_rank"], r["homogeneous_positive_pair_rank"], r["rank_difference"])
                          for r in rows], [(2, 1, 1), (1, 1, 0)])

    def test_query_negative_state_denominator_hand_calculation(self):
        qids, gids = torch.tensor([0, 1]), torch.tensor([0, 0, 1, 2])
        # Query0 negatives are clean/clean -> clean ratio 1; query1 negatives
        # are rgb/rgb/clean -> rgb ratio 1/(2/3)=1.5. Average each across 2 errors.
        sim = torch.tensor([[1., 2., 4., 3.], [4., 3., 1., 2.]])
        ranked = ranking_arrays(sim, qids, gids)
        summary, details = top1_error_state_rates(ranked, qids, gids,
                                                  ["rgb", "rgb", "clean", "clean"], ["rgb", "clean", "absent"])
        by_state = {r["state"]: r for r in summary}
        self.assertAlmostEqual(by_state["rgb"]["query_normalized_error_ratio"], .75)
        self.assertAlmostEqual(by_state["clean"]["query_normalized_error_ratio"], .5)
        self.assertAlmostEqual(by_state["rgb"]["error_share_over_mean_negative_share"], 1.5)
        self.assertAlmostEqual(by_state["clean"]["error_share_over_mean_negative_share"], .75)
        self.assertEqual(by_state["absent"]["query_normalized_error_ratio"], 0)
        self.assertIsNone(by_state["absent"]["error_share_over_mean_negative_share"])
        self.assertEqual(len(details), 6)
        self.assertEqual(by_state["rgb"]["available_error_queries"], 1)

    def test_assignments_reproducible_model_independent_and_global_rng_unchanged(self):
        torch_state, numpy_state, python_state = torch.random.get_rng_state(), np.random.get_state(), random.getstate()
        a = assign_gallery_states(100, "M1", 2026)
        self.assertEqual(a, assign_gallery_states(100, "M1", 2026))
        self.assertNotEqual(a, assign_gallery_states(100, "M1", 2027))
        clustered_difference([1, 2, 3], [0, 1, 1], n_resamples=20)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        np.testing.assert_equal(numpy_state, np.random.get_state())
        self.assertEqual(python_state, random.getstate())

    def test_multiple_exports_identical_assignments_paired_ci_and_seed_average(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_a, annotation, split, _, _, _ = self.make_export(root, "base", True)
            export_b, _, _, _, _, _ = self.make_export(root, "c0", False)
            output = root / "analysis"
            output.mkdir()  # Existing empty output is allowed.
            result = run_analysis([export_a, export_b], split, output, annotation_file=annotation,
                                  n_resamples=25, names=["base_clean", "c0_ce"])
            self.assertEqual(result["exports"][0]["assignments"], result["exports"][1]["assignments"])
            self.assertEqual([e["n_scenarios_checked"] for e in result["exports"]], [17, 11])
            self.assertEqual(len(result["exports"][0]["matrix"]), 55)
            for cell in result["exports"][0]["matrix"]:
                if cell["positive_state"] == cell["negative_state"]:
                    for metric in ("R1", "mAP", "best_pos_rank_mean"):
                        self.assertEqual(cell[f"delta_{metric}"], 0)
                        self.assertEqual((cell[f"delta_{metric}_ci95_low"], cell[f"delta_{metric}_ci95_high"]), (0, 0))
            for row in result["checkpoint_comparisons"]:
                self.assertEqual(row["difference_B_minus_A"], 0)
                self.assertEqual((row["ci95_low"], row["ci95_high"]), (0, 0))
            by_mixture = result["exports"][0]["random_metrics"]
            for mixture in ("M1", "M2", "M3"):
                seed_rows = [row for row in by_mixture if row["mixture"] == mixture and row["seed"] != "mean"]
                average = next(row for row in by_mixture if row["mixture"] == mixture and row["seed"] == "mean")
                self.assertAlmostEqual(average["R1"], np.mean([row["R1"] for row in seed_rows]))
                self.assertAlmostEqual(average["mAP"], np.mean([row["mAP"] for row in seed_rows]))
                self.assertIn("delta_R1_ci95_low", average)
            verified = load_verified_export(export_a, split, annotation)
            detail = analyze_verified_export(verified, n_resamples=25)
            clean = ranking_arrays(verified["scores"]["clean"],
                                   torch.tensor([q["eval_pid"] for q in verified["dev_data"]["queries"]]),
                                   torch.tensor(verified["dev_data"]["gallery_pids"]))
            clean_vectors = np.column_stack(((clean["best_pos_rank"].numpy() == 1) * 100.,
                                             clean["AP"].numpy().astype(np.float64) * 100.,
                                             clean["best_pos_rank"].numpy().astype(np.float64)))
            seed_average = np.mean([detail["_random_vectors"][("M1", seed)] for seed in (2026, 2027, 2028)], axis=0)
            expected_ci = clustered_differences(seed_average - clean_vectors,
                                                [q["pid"] for q in verified["dev_data"]["queries"]], n_resamples=25)
            actual_average = next(row for row in by_mixture if row["mixture"] == "M1" and row["seed"] == "mean")
            for metric, expected in zip(("R1", "mAP", "best_pos_rank_mean"), expected_ci):
                self.assertEqual(actual_average[f"delta_{metric}"], expected["difference"])
                self.assertEqual([actual_average[f"delta_{metric}_ci95_low"], actual_average[f"delta_{metric}_ci95_high"]],
                                 expected["ci95"])
            self.assertTrue((output / "checkpoint_comparisons.csv").is_file())
            with self.assertRaises(FileExistsError):
                run_analysis([export_a], split, output, annotation_file=annotation, n_resamples=10)

    def test_cross_directory_ordering_mismatch_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_a, annotation, split, _, _, _ = self.make_export(root, "base")
            export_b, _, _, _, _, _ = self.make_export(root, "c0")
            payload = torch.load(export_b / "scores" / "clean.pt", weights_only=True)
            payload["txt_pids"] = payload["txt_pids"].flip(0)
            torch.save(payload, export_b / "scores" / "clean.pt")
            with self.assertRaisesRegex(ValueError, "PID order"):
                run_analysis([export_a, export_b], split, root / "bad", annotation_file=annotation, n_resamples=10)
            self.assertFalse((root / "bad").exists())
            # Also exercise the explicit cross-directory order equality guard.
            verified_a = load_verified_export(export_a, split, annotation)
            verified_b = {**verified_a, "dev_data": {**verified_a["dev_data"],
                                                   "gallery_annotation_indices": [1, 0, 2, 3]}}
            with patch("tools.p4.analysis.mixed_gallery_from_scores.load_verified_export", side_effect=[verified_a, verified_b]):
                with self.assertRaisesRegex(ValueError, "Multiple exports disagree"):
                    run_analysis([export_a, export_b], split, root / "bad", annotation_file=annotation,
                                 n_resamples=10, names=["base", "c0"])

    def test_cli_cpu_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export, annotation, split, _, _, _ = self.make_export(root)
            script = Path(__file__).resolve().parents[2] / "tools" / "p4" / "analysis" / "mixed_gallery_from_scores.py"
            completed = subprocess.run([sys.executable, "-B", str(script), "--export_dirs", str(export), "--names", "base",
                                        "--split_file", str(split), "--annotation_file", str(annotation),
                                        "--output_dir", str(root / "cli"), "--n_resamples", "10", "--seeds", "2026"],
                                       check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(completed.stdout)["n_exports"], 1)
            result = json.loads((root / "cli" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["bootstrap"]["seed"], 2026)


if __name__ == "__main__":
    unittest.main()
