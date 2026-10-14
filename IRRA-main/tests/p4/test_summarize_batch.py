"""Synthetic-only CPU tests for saved-record batch summaries."""

import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import torch

from tools.p4.analysis.summarize_batch import (
    GROUPS, available_probes, distribution, failure_labels, fit_regressions,
    load_export_records, load_training_run, measured_kappas, parse_validation_log, run_summary,
)
from tools.p4.export_dev_records import SCENARIO_NAMES, TRAIN_STATE_SCENARIO_NAMES
from tools.p4.train_batch_stats import TRAIN_STATES


class BatchSummaryTests(unittest.TestCase):
    def fixture(self, root, name="base", train_states=False, changed=False):
        export_dir, run_dir = root / name, root / f"run_{name}"
        export_dir.mkdir()
        run_dir.mkdir()
        tokens, colors = [1, 4, 5, 2, 10, 6, 10, 5], [0, 1, 1, 1, 1, 3, 3, 2]
        clean = [1, 2, 6, 12, 1, 3, 9, 11]
        if changed:
            clean = [2, 1, 6, 12, 1, 1, 9, 11]
        rows, metric_rows = [], []
        scenes = (*SCENARIO_NAMES, *TRAIN_STATE_SCENARIO_NAMES) if train_states else SCENARIO_NAMES
        for scenario in scenes:
            ranks = clean
            if scenario == "rgb_missing":
                ranks = [2, 2, 1, 12, 2, 1, 9, 11]
            elif scenario == "ir_missing":
                ranks = [1, 1, 12, 12, 1, 3, 11, 11]
            elif scenario.endswith("_train"):
                ranks = [1] * 8
            scenario_rows = [{"query_id": f"{index}:0", "pid": index // 2,
                              "caption": f"caption {index}", "scenario": scenario,
                              "best_pos_rank": rank, "AP": 1 / rank, "n_pos": 1,
                              "n_tokens": tokens[index], "n_color": colors[index],
                              "rho": colors[index] / tokens[index]}
                             for index, rank in enumerate(ranks)]
            rows.extend(scenario_rows)
            metric_rows.append({"scenario": scenario,
                                "R1": sum(rank == 1 for rank in ranks) / 8 * 100,
                                "R5": sum(rank <= 5 for rank in ranks) / 8 * 100,
                                "R10": sum(rank <= 10 for rank in ranks) / 8 * 100,
                                "mAP": sum(1 / rank for rank in ranks) / 8 * 100,
                                "mINP": sum(1 / rank for rank in ranks) / 8 * 100})
        self.write_csv(export_dir / "per_query.csv", rows)
        self.write_csv(export_dir / "metrics.tsv", metric_rows, "\t")
        (export_dir / "summary.json").write_text(json.dumps({"clean_R1": metric_rows[0]["R1"],
                                                              "provenance": {"n_queries": 8,
                                                              "split_identity_sha256": "synthetic_split"}}), encoding="utf-8")
        epochs = [{"epoch": epoch, "partial": False, "text_samples": 16, "unique_images": 4,
                   "degraded_copies": 16 if changed else 0,
                   "image_encoder_calls": {"rgb_clean": 2, "ir_clean": 2,
                                            "rgb_degraded": epoch if changed else 0,
                                            "ir_degraded": 2 if changed else 0},
                   "degradation_mode_counts": {"rgb_drop": 8, "ir_noise": 8} if changed else {},
                   "seconds": 1.0, "peak_memory_bytes": 0} for epoch in (1, 2)]
        (run_dir / "run_record.json").write_text(json.dumps({"epochs": epochs, "steps": [{"step": i} for i in range(4)],
                                                             "git_commit": "synthetic", "train_seed": 101}), encoding="utf-8")
        (run_dir / "train_log.txt").write_text(
            "unrelated table | t2i | 99 | 99 | 99 | 99 | 99 |\n"
            f"Validation Results - Epoch: 1\n| t2i | 10 | 20 | 30 | 5 | 4 |\n"
            "best R1: 99 at epoch 1\n"
            f"Validation Results - Epoch: 2\n| t2i | {metric_rows[0]['R1']:.3f} | 50 | 60 | 40 | 30 |\n"
            "best R1: 100\n| t2i | 100 | 100 | 100 | 100 | 100 |\n", encoding="utf-8")
        return export_dir, run_dir, rows, metric_rows

    @staticmethod
    def write_csv(path, rows, delimiter=","):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)

    def test_eleven_seventeen_groups_unique_queries_and_distribution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export, run, _, _ = self.fixture(root, train_states=True)
            result = run_summary([export], [run], root / "output", kappa=["rgb_drop=.6667"], n_resamples=25)
            self.assertEqual(result["n_unique_queries"], 8)
            item = result["exports"][0]
            self.assertEqual(len(item["metrics"]), 17)
            self.assertEqual(item["rho_distribution"]["n"], 8)
            for group in item["group_means"]:
                names = GROUPS[group["group"]]
                expected = sum(next(row["R1"] for row in item["metrics"] if row["scenario"] == scene) for scene in names) / len(names)
                self.assertEqual(group["R1"], expected)
                self.assertFalse(any(scene.endswith("_train") for scene in names))
            clean_bins = next(row for row in item["rank_bins"] if row["scenario"] == "clean")
            self.assertEqual([clean_bins[key] for key in ("rank_1", "rank_2_5", "rank_6_10", "rank_gt10")], [2, 2, 2, 2])
            rho = [row["rho"] for row in load_export_records(export)["queries"]]
            expected = distribution([max(.05, 1 - value * .6667) for value in rho])
            self.assertEqual(item["k2"][0]["mean"], expected["mean"])
            self.assertEqual(item["k2"][0]["std"], expected["std"])

    def test_failure_probe_definitions_and_no_vacuous_empty_probe_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            export, _, _, _ = self.fixture(Path(temporary), train_states=True)
            labels, rows = failure_labels(load_export_records(export)["by_scene"])
            clean = next(row for row in labels if row["scenario"] == "clean")
            self.assertEqual(clean["fusion_failures"], 6)
            self.assertEqual((clean["label_A_count"], clean["label_B_count"], clean["oracle_R1"]), (3, 1, 62.5))
            self.assertEqual((clean["label_A_percent"], clean["label_B_percent"]), (37.5, 12.5))
            for row in labels:
                if row["scenario"] in ("rgb_missing", "both_noise_005", "rgb_drop_train"):
                    self.assertIsNone(row["label_A_count"])
                    self.assertIsNone(row["label_B_count"])
                    self.assertIsNone(row["oracle_R1"])
            rgb = next(row for row in rows if row["scenario"] == "rgb_noise_010")
            self.assertIsNone(rgb["probe_RGB_only_rank"])
            self.assertIsNotNone(rgb["probe_IR_only_rank"])
            self.assertEqual(available_probes("ir_blur_k7"), (("P-zero_RGB-only", "ir_missing"),))

    def test_paired_differences_changes_and_common_train_scene_handling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, run_a, _, _ = self.fixture(root, "base", train_states=True)
            b, run_b, _, _ = self.fixture(root, "ce", changed=True)
            result = run_summary([a, b], [run_a, run_b], root / "output", names=["base_clean", "c0_ce"], n_resamples=50)
            clean = next(row for row in result["pairwise_comparisons"] if row["condition"] == "clean" and row["metric"] == "R1")
            self.assertEqual(clean["difference_B_minus_A"], 12.5)
            self.assertEqual((clean["changed_correct"], clean["changed_wrong"]), (2, 1))
            self.assertLessEqual(clean["ci95_low"], clean["difference_B_minus_A"])
            self.assertGreaterEqual(clean["ci95_high"], clean["difference_B_minus_A"])
            self.assertFalse(any(row["condition"].endswith("_train") for row in result["pairwise_comparisons"]))
            self.assertTrue(any(row["metric"] == "R5" for row in result["pairwise_comparisons"]))
            self.assertTrue(any(row["metric"] == "R10" for row in result["pairwise_comparisons"]))
            self.assertTrue(all(row["ci95"] is None for row in result["point_only_differences"]))

    def test_regression_known_coefficients_pid_bootstrap_and_rng_unchanged(self):
        rho = torch.tensor([0., .25, .2, .5, .1, .5, .3, .4], dtype=torch.float64)
        tokens = torch.tensor([1., 4, 5, 2, 10, 6, 10, 5], dtype=torch.float64)
        z_rho, z_tokens = (rho - rho.mean()) / rho.std(unbiased=False), (tokens - tokens.mean()) / tokens.std(unbiased=False)
        y = 2 + 3 * z_rho - 4 * z_tokens
        torch_state, python_state = torch.random.get_rng_state(), random.getstate()
        result = fit_regressions(rho, tokens, y, [0, 0, 1, 1, 2, 2, 3, 3], n_resamples=100)
        torch.testing.assert_close(torch.tensor(result["coefficients"]).view(-1), torch.tensor([2., 3., -4.]), atol=1e-6, rtol=0)
        self.assertGreater(result["bootstrap_valid"], 0)
        self.assertEqual(result["n_queries"], 8)
        self.assertEqual(result["n_pids"], 4)
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertEqual(result, fit_regressions(rho, tokens, y, [0, 0, 1, 1, 2, 2, 3, 3], n_resamples=100))
        constant = fit_regressions(torch.ones(8), tokens, y, [0, 0, 1, 1, 2, 2, 3, 3], n_resamples=10)
        self.assertIsNone(constant["coefficients"])

    def test_kappa_complete_schema_manual_clip_sources_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "kappa.json"
            path.write_text(json.dumps({"R1_clean": 80., "states": {state: {"raw": .5, "clipped": .5,
                                               "source_file": "synthetic_metrics.tsv"} for state in TRAIN_STATES}}), encoding="utf-8")
            self.assertEqual(len(measured_kappas(path)), 6)
            manual = measured_kappas(manual=["rgb_drop=2", "ir_drop=-1"])
            self.assertEqual((manual["rgb_drop"]["raw"], manual["rgb_drop"]["clipped"]), (2, 1))
            self.assertEqual(manual["ir_drop"]["clipped"], 0)
            self.assertEqual(manual["rgb_drop"]["source"]["kind"], "manual")
            for item in (["rgb_drop=nan"], ["unknown=.5"], ["rgb_drop=.5", "rgb_drop=.6"]):
                with self.assertRaises(ValueError):
                    measured_kappas(manual=item)
            with self.assertRaises(ValueError):
                measured_kappas(path, ["rgb_drop=.5"])
            export, run, rows, _ = self.fixture(Path(temporary))
            self.write_csv(export / "per_query.csv", [{**row, "rho": 1., "n_color": 1}
                                                     if row["query_id"] == "0:0" else row for row in rows])
            result = run_summary([export], [run], Path(temporary) / "clipped_output",
                                 kappa=["rgb_drop=2"], n_resamples=10)
            self.assertEqual(result["exports"][0]["k2"][0]["min"], .05)
            self.assertEqual(result["exports"][0]["k2"][0]["raw_kappa"], 2.)

    def test_validation_parser_does_not_take_best_r1_or_unmarked_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, run, _, _ = self.fixture(Path(temporary), changed=True)
            loaded = load_training_run(run)
            self.assertEqual([row["R1"] for row in loaded["validation_metrics"]], [10., 37.5])
            self.assertEqual(loaded["totals"]["recorded_step_rows"], 4)
            self.assertEqual(loaded["totals"]["text_samples"], 32)
            self.assertEqual(loaded["totals"]["epoch_unique_image_exposures"], 8)
            self.assertIsNone(loaded["totals"]["unique_images_union"])
            self.assertEqual(loaded["totals"]["image_encoder_calls"]["rgb_degraded"], 3)
            bad = Path(temporary) / "bad_log.txt"
            bad.write_text("Validation Results - Epoch: 1\nValidation Results - Epoch: 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no t2i"):
                parse_validation_log(bad)
            record_path = run / "run_record.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            del record["epochs"][0]["degraded_copies"]
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing counters"):
                load_training_run(run)

    def test_export_mismatch_checks_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export, run, rows, metrics = self.fixture(root)
            changed = [dict(row) for row in rows]
            changed[8], changed[9] = changed[9], changed[8]
            self.write_csv(export / "per_query.csv", changed)
            with self.assertRaisesRegex(ValueError, "order differs"):
                run_summary([export], [run], root / "bad", n_resamples=10)
            self.assertFalse((root / "bad").exists())
            self.write_csv(export / "per_query.csv", rows)
            self.write_csv(export / "metrics.tsv", [{**metrics[0], "R1": 99}, *metrics[1:]], "\t")
            with self.assertRaisesRegex(ValueError, "Query/metrics disagreement"):
                load_export_records(export)
            self.write_csv(export / "metrics.tsv", metrics, "\t")
            self.write_csv(export / "per_query.csv", [*rows, rows[0]])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_export_records(export)

    def test_cross_model_query_and_recorded_split_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, run_a, _, _ = self.fixture(root, "base")
            b, run_b, rows, _ = self.fixture(root, "ce")
            self.write_csv(b / "per_query.csv", [{**row, "caption": "different"} if row["query_id"] == "0:0" else row for row in rows])
            with self.assertRaisesRegex(ValueError, "Cross-model query"):
                run_summary([a, b], [run_a, run_b], root / "bad", n_resamples=10)
            self.write_csv(b / "per_query.csv", rows)
            (b / "summary.json").write_text(json.dumps({"provenance": {"split_identity_sha256": "different"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split hashes"):
                run_summary([a, b], [run_a, run_b], root / "bad", n_resamples=10)
            self.assertFalse((root / "bad").exists())

    def test_empty_output_allowed_nonempty_refused_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export, run, _, _ = self.fixture(root)
            output = root / "output"
            output.mkdir()
            result = run_summary([export], [run], output, n_resamples=10)
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(len(result["exports"][0]["sources"]["per_query"]["sha256"]), 64)
            self.assertTrue((output / "00_base" / "validation_metrics.csv").is_file())
            with self.assertRaises(FileExistsError):
                run_summary([export], [run], output, n_resamples=10)

    def test_cli_cpu_synthetic_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export, run, _, _ = self.fixture(root)
            script = Path(__file__).resolve().parents[2] / "tools" / "p4" / "analysis" / "summarize_batch.py"
            completed = subprocess.run([sys.executable, "-B", str(script), "--export_dirs", str(export),
                                        "--run_dirs", str(run), "--names", "base_clean", "--output_dir", str(root / "cli"),
                                        "--kappa", "rgb_drop=.6667", "--n_resamples", "10"],
                                       check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(completed.stdout)["n_unique_queries"], 8)


if __name__ == "__main__":
    unittest.main()
