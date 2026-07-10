import tempfile
import unittest
from pathlib import Path

import torch

from tools.revision.color_group_evaluation import (
    compile_color_patterns,
    match_color_terms,
)
from tools.revision.compare_score_files import compare_score_files
from tools.revision.ensemble_scores import build_ensemble_payload
from tools.revision.feature_norm_analysis import compute_feature_diagnostics
from tools.revision.intensity_curve_summary import sort_intensity_records
from tools.revision.noise_repeat_summary import summarize_noise_groups
from tools.revision.paired_seed_statistics import compute_paired_statistics
from tools.revision.score_io import (
    ScoreConsistencyError,
    compute_retrieval_metrics,
    fuse_score_items,
    load_score_file,
    require_score_consistency,
    row_zscore,
)
from tools.revision.summarize_robustness import validate_robustness_manifest


class RevisionAnalysisToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.txt_pids = torch.tensor([0, 1, 2], dtype=torch.long)
        self.img_pids = torch.tensor(
            [0, 1, 2, 0, 1, 2, 3, 4, 5, 6], dtype=torch.long
        )
        self.base_matrix = torch.tensor(
            [
                [0.9, 0.1, 0.2, 0.8, 0.0, -0.1, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.9, 0.2, 0.0, 0.8, -0.1, 0.3, 0.4, 0.5, 0.6],
                [0.1, 0.2, 0.9, 0.0, -0.1, 0.8, 0.3, 0.4, 0.5, 0.6],
            ],
            dtype=torch.float32,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _payload(
        self,
        matrix=None,
        txt_pids=None,
        img_pids=None,
        txt_indices=None,
        img_indices=None,
        order_hash="order-1",
    ):
        matrix = self.base_matrix.clone() if matrix is None else matrix.clone()
        return {
            "sim": matrix,
            "raw_sim": matrix.clone(),
            "txt_pids": self.txt_pids.clone() if txt_pids is None else txt_pids.clone(),
            "img_pids": self.img_pids.clone() if img_pids is None else img_pids.clone(),
            "txt_indices": (
                torch.arange(matrix.shape[0], dtype=torch.long)
                if txt_indices is None
                else txt_indices.clone()
            ),
            "img_indices": (
                torch.arange(matrix.shape[1], dtype=torch.long)
                if img_indices is None
                else img_indices.clone()
            ),
            "metrics": {"R1": 100.0, "mAP": 100.0},
            "metadata": {"source": "unit-test"},
            "order_hash": order_hash,
        }

    def _save(self, name, **kwargs):
        path = self.root / name
        torch.save(self._payload(**kwargs), path)
        return path

    def test_score_order_mismatch_refuses_fusion(self):
        first = load_score_file(self._save("first.pt"))
        second = load_score_file(
            self._save("second.pt", order_hash="different-order")
        )
        with self.assertRaises(ScoreConsistencyError):
            require_score_consistency([first, second])

    def test_pid_match_but_indices_mismatch_refuses_fusion(self):
        first = load_score_file(self._save("first.pt"))
        second = load_score_file(
            self._save("second.pt", txt_indices=torch.tensor([1, 0, 2]))
        )
        self.assertTrue(torch.equal(first["txt_pids"], second["txt_pids"]))
        with self.assertRaisesRegex(ScoreConsistencyError, "txt_indices"):
            require_score_consistency([first, second])

    def test_nan_and_inf_are_rejected(self):
        for name, invalid_value in (("nan.pt", float("nan")), ("inf.pt", float("inf"))):
            matrix = self.base_matrix.clone()
            matrix[0, 0] = invalid_value
            path = self._save(name, matrix=matrix)
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    load_score_file(path)

    def test_row_zscore_has_near_zero_row_means(self):
        normalized = row_zscore(self.base_matrix, epsilon=1e-6)
        self.assertTrue(
            torch.allclose(
                normalized.mean(dim=1),
                torch.zeros(self.base_matrix.shape[0]),
                atol=1e-6,
                rtol=0.0,
            )
        )

    def test_equal_fusion_preserves_shape(self):
        first = load_score_file(self._save("first.pt"))
        second = load_score_file(
            self._save("second.pt", matrix=self.base_matrix + 0.05)
        )
        fused, weights = fuse_score_items(
            [first, second], method="row_zscore_equal", epsilon=1e-6
        )
        self.assertEqual(fused.shape, self.base_matrix.shape)
        self.assertEqual(weights, [0.5, 0.5])

    def test_retrieval_metrics_match_expected_perfect_ranking(self):
        metrics = compute_retrieval_metrics(
            self.base_matrix, self.txt_pids, self.img_pids
        )
        for key in ("R1", "R5", "R10", "mAP", "mINP"):
            self.assertEqual(metrics[key], 100.0)

    def test_compare_score_files_identifies_same_and_different(self):
        first = self._save("first.pt")
        same = self._save("same.pt")
        different = self._save("different.pt", matrix=self.base_matrix + 0.01)
        same_report = compare_score_files(first, same)
        different_report = compare_score_files(first, different)
        self.assertTrue(same_report["exact_equal"])
        self.assertTrue(same_report["matched"])
        self.assertFalse(different_report["exact_equal"])
        self.assertFalse(different_report["matched"])
        self.assertGreater(different_report["max_abs_diff"], 0.0)

    def test_oracle_is_disabled_by_default(self):
        score_file = self._save("single.pt")
        payload = build_ensemble_payload([score_file])
        self.assertFalse(payload["metadata"]["oracle_only"])
        self.assertTrue(payload["metadata"]["equal_weights"])
        with self.assertRaisesRegex(ValueError, "exactly 7"):
            build_ensemble_payload([score_file], oracle_weights=[1.0])

    def test_color_term_uses_word_boundaries(self):
        patterns = compile_color_patterns(["red", "light", "multi-colored"])
        self.assertEqual(match_color_terms("A red shirt.", patterns)["red"], 1)
        self.assertNotIn("red", match_color_terms("A colored shirt.", patterns))
        self.assertNotIn("light", match_color_terms("A lightweight jacket.", patterns))
        self.assertIn(
            "multi-colored",
            match_color_terms("A MULTI-COLORED top.", patterns),
        )

    def test_wilcoxon_result_is_computed_from_inputs(self):
        def summary(primary):
            return {
                "clean_R1": 60.0,
                "degraded_mean_R1": 45.0,
                "mean_abs_delta_R1": float(primary),
                "clean_mAP": 55.0,
                "degraded_mean_mAP": 40.0,
            }

        all_improve = [
            {"seed": seed, "base": summary(10.0), "mdrc": summary(9.0)}
            for seed in range(1, 6)
        ]
        mixed_differences = [1.0, -1.0, 2.0, -2.0, 3.0]
        mixed = [
            {
                "seed": seed,
                "base": summary(10.0),
                "mdrc": summary(10.0 + mixed_differences[seed - 1]),
            }
            for seed in range(1, 6)
        ]
        result_a = compute_paired_statistics(all_improve)
        result_b = compute_paired_statistics(mixed)
        self.assertNotEqual(
            result_a["wilcoxon_one_sided"]["pvalue"],
            result_b["wilcoxon_one_sided"]["pvalue"],
        )

    def test_missing_robustness_scenario_raises(self):
        with self.assertRaisesRegex(ValueError, "missing scenarios"):
            validate_robustness_manifest(
                {"scenarios": {"clean": {"score_file": "clean.pt"}}}
            )

    def test_noise_repeat_with_fewer_than_three_is_incomplete(self):
        summaries = summarize_noise_groups(
            [
                {
                    "model": "Base",
                    "scenario": "rgb_noise_005",
                    "degrade_seed": 2026,
                    "R1": 50.0,
                    "mAP": 45.0,
                },
                {
                    "model": "Base",
                    "scenario": "rgb_noise_005",
                    "degrade_seed": 2027,
                    "R1": 51.0,
                    "mAP": 46.0,
                },
            ]
        )
        self.assertEqual(summaries[0]["n"], 2)
        self.assertTrue(summaries[0]["incomplete"])

    def test_intensity_records_sort_numerically(self):
        rows = [
            {"series": "rgb_noise", "intensity": 0.2},
            {"series": "rgb_noise", "intensity": 0.02},
            {"series": "rgb_noise", "intensity": 0.1},
            {"series": "rgb_noise", "intensity": 0.05},
        ]
        ordered = sort_intensity_records(rows)
        self.assertEqual(
            [row["intensity"] for row in ordered], [0.02, 0.05, 0.1, 0.2]
        )

    def test_feature_norm_diagnostics_use_existing_three_features(self):
        rgb = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        thermal = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
        fused = rgb + thermal
        per_sample, summary = compute_feature_diagnostics(rgb, thermal, fused)
        self.assertEqual(per_sample["rgb_l2_norm"][0], 5.0)
        self.assertEqual(per_sample["thermal_l2_norm"][0], 2.0)
        self.assertEqual(len(per_sample["rgb_to_fused_cosine"]), 2)
        self.assertEqual(summary["rgb_l2_norm"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
