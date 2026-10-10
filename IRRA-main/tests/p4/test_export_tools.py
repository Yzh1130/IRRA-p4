"""CPU-only synthetic tests for export provenance, per-query metrics and pairing."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from tools.p4.export_dev_records import (
    CORE_SCENARIOS, SCENARIO_NAMES, build_summary, caption_token_counts,
    load_strict_checkpoint, per_query_records, validate_dev_split,
)
from tools.p4.paired_bootstrap import align_records, load_records, paired_bootstrap


class ExportToolsTests(unittest.TestCase):
    def test_core_scenario_contract(self):
        self.assertEqual(len(CORE_SCENARIOS), 11)
        self.assertEqual(SCENARIO_NAMES[0], "clean")
        self.assertEqual(CORE_SCENARIOS[3], ("rgb_noise_005", "rgb_noise", 0.05, 7))
        self.assertEqual(CORE_SCENARIOS[-1], ("both_blur_k7", "both_blur", 1.0, 7))

    def test_token_counts_are_occurrences_after_stopwords(self):
        counts = caption_token_counts("A RED, red shirt with the Blue bag.")
        self.assertEqual(counts, {"n_tokens": 5, "n_color": 3, "rho": 0.6})
        self.assertEqual(caption_token_counts("the and a"), {"n_tokens": 0, "n_color": 0, "rho": 0.0})

    def _write_split(self, root):
        annos = [
            {"id": 1, "split": "train", "file_path": "1.jpg", "captions": ["red shirt", "blue bag"]},
            {"id": 2, "split": "train", "file_path": "2.jpg", "captions": ["black shoes"]},
            {"id": 3, "split": "test", "file_path": "3.jpg", "captions": ["white coat"]},
        ]
        annotation_path = root / "Caption.json"
        annotation_path.write_text(json.dumps(annos), encoding="utf-8")
        split = {
            "dev_ids": [1], "train_reduced_ids": [2], "n_dev": 1, "n_train_reduced": 1,
            "dev_records": 1, "dev_captions": 2, "train_reduced_records": 1,
            "train_reduced_captions": 1,
            "anno_md5": hashlib.md5(annotation_path.read_bytes()).hexdigest(),
        }
        split_path = root / "dev.json"
        split_path.write_text(json.dumps(split), encoding="utf-8")
        return annotation_path, split_path, split

    def test_split_hash_identities_counts_and_query_order(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation, split, _ = self._write_split(Path(directory))
            result = validate_dev_split(annotation, split)
        self.assertEqual(result["n_train"], 1)
        self.assertEqual(result["gallery_pids"], [0])
        self.assertEqual(result["gallery_raw_pids"], [1])
        self.assertEqual([q["query_id"] for q in result["queries"]], ["0:0", "0:1"])
        self.assertEqual([q["caption"] for q in result["queries"]], ["red shirt", "blue bag"])

    def test_split_rejects_overlap_wrong_hash_and_wrong_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation, split_path, valid = self._write_split(Path(directory))
            cases = (
                {**valid, "train_reduced_ids": [1]},
                {**valid, "dev_ids": [3]},
                {**valid, "anno_md5": "0" * 32},
                {**valid, "dev_captions": 3},
                {**valid, "dev_ids": [1, 1]},
            )
            for split in cases:
                with self.subTest(split=split):
                    split_path.write_text(json.dumps(split), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        validate_dev_split(annotation, split_path)

    def test_strict_checkpoint_keys_and_shapes(self):
        reference = torch.nn.Linear(2, 3)
        target = torch.nn.Linear(2, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pth"
            state = reference.state_dict()
            torch.save({"model": {"module." + key: value for key, value in state.items()}, "epoch": 2}, path)
            checkpoint = load_strict_checkpoint(target, path)
            self.assertEqual(checkpoint["epoch"], 2)
            for key, value in state.items():
                self.assertTrue(torch.equal(value, target.state_dict()[key]))
            for bad in ({"weight": state["weight"]}, {**state, "unexpected": torch.ones(1)},
                        {**state, "weight": torch.ones(2, 2)}):
                torch.save({"model": bad}, path)
                with self.assertRaises(RuntimeError):
                    load_strict_checkpoint(target, path)

    def _scores(self):
        return {
            "sim": torch.tensor([[3.0, 2.0, 1.0], [2.0, 1.0, 3.0], [3.0, 2.0, 1.0]]),
            "txt_pids": torch.tensor([0, 0, 1]), "img_pids": torch.tensor([0, 0, 1]),
            "txt_indices": torch.arange(3), "img_indices": torch.arange(3),
        }, {
            "queries": [
                {"query_id": "0:0", "pid": 10, "eval_pid": 0, "caption": "red coat"},
                {"query_id": "0:1", "pid": 10, "eval_pid": 0, "caption": "blue bag"},
                {"query_id": "1:0", "pid": 20, "eval_pid": 1, "caption": "black shoes"},
            ],
            "gallery_pids": [0, 0, 1],
        }

    def test_per_query_ap_matches_existing_rank(self):
        from utils.metrics import rank

        payload, metadata = self._scores()
        rows = per_query_records(payload, metadata, "clean")
        self.assertEqual([row["best_pos_rank"] for row in rows], [1, 2, 3])
        self.assertEqual([row["n_pos"] for row in rows], [2, 2, 1])
        expected = [1.0, (1 / 2 + 2 / 3) / 2, 1 / 3]
        for row, ap in zip(rows, expected):
            self.assertAlmostEqual(row["AP"], ap, places=6)
        cmc, mAP, _, _ = rank(payload["sim"], payload["txt_pids"], payload["img_pids"], max_rank=3)
        self.assertAlmostEqual(sum(row["AP"] for row in rows) / 3 * 100, float(mAP), places=4)
        self.assertAlmostEqual(sum(row["best_pos_rank"] == 1 for row in rows) / 3 * 100, float(cmc[0]), places=4)

    def test_per_query_rejects_order_nonfinite_and_no_positive(self):
        payload, metadata = self._scores()
        bad_order = {**payload, "txt_indices": torch.tensor([1, 0, 2])}
        bad_pid = {**payload, "txt_pids": torch.tensor([1, 0, 0])}
        bad_nan = {**payload, "sim": payload["sim"].clone()}
        bad_nan["sim"][0, 0] = float("nan")
        for bad in (bad_order, bad_pid, bad_nan):
            with self.assertRaises(ValueError):
                per_query_records(bad, metadata, "clean")
        missing = {**metadata, "gallery_pids": [0, 0, 0]}
        bad = {**payload, "img_pids": torch.tensor([0, 0, 0])}
        with self.assertRaisesRegex(ValueError, "no positive"):
            per_query_records(bad, missing, "clean")

    def test_summary_excludes_clean_from_degraded_mean(self):
        rows = [{"scenario": name, "R1": index, "mAP": 20 + index}
                for index, name in enumerate(SCENARIO_NAMES)]
        rows[0]["R1"] = 100
        summary = build_summary(rows)
        self.assertEqual(summary["degraded_R1_mean"], 5.5)
        self.assertEqual(summary["clean_R1"], 100)
        self.assertEqual(summary["clean_mAP"], 20)
        with self.assertRaises(ValueError):
            build_summary(rows[:-1])


class PairedBootstrapTests(unittest.TestCase):
    def _records(self):
        a, b = [], []
        for scenario in ("clean", "rgb_missing"):
            # PID 1 has one query, PID 2 has three. Only PID 1 improves.
            for query_id, pid in (("a", 1), ("b", 2), ("c", 2), ("d", 2)):
                base = {"query_id": query_id, "pid": pid, "caption": query_id,
                        "scenario": scenario, "n_pos": 1, "best_pos_rank": 2, "AP": 0.5}
                a.append(base)
                b.append({**base, "best_pos_rank": 1 if pid == 1 else 2,
                          "AP": 1.0 if pid == 1 else 0.5})
        return a, b

    def test_pid_clusters_query_weighting_reproducibility_and_units(self):
        a, b = self._records()
        result = paired_bootstrap(a, b, n_resamples=1000, seed=2026)
        self.assertEqual(result, paired_bootstrap(a, list(reversed(b)), n_resamples=1000, seed=2026))
        self.assertEqual(result["mean_degraded"]["R1"]["difference"], 25.0)
        self.assertEqual(result["mean_degraded"]["mAP"]["difference"], 12.5)
        self.assertEqual(result["per_scenario"]["clean"]["R1"]["ci95"], [0.0, 100.0])
        self.assertEqual(result["n_pids"], 2)
        self.assertEqual(result["n_queries"], 4)

    def test_identical_files_have_zero_ci_and_pairing_errors_fail(self):
        a, b = self._records()
        result = paired_bootstrap(a, a, n_resamples=100)
        self.assertEqual(result["mean_all"]["R1"], {"difference": 0.0, "ci95": [0.0, 0.0]})
        self.assertEqual(result["mean_all"]["mAP"], {"difference": 0.0, "ci95": [0.0, 0.0]})
        with self.assertRaises(ValueError):
            align_records(a, b[:-1])
        with self.assertRaises(ValueError):
            align_records(a, b + [b[0]])
        mismatch = [dict(row) for row in b]
        mismatch[0]["caption"] = "wrong"
        with self.assertRaises(ValueError):
            align_records(a, mismatch)
        with self.assertRaises(ValueError):
            paired_bootstrap(a, b, scenarios=["unknown"])

    def test_csv_loader_checks_ap_fraction_and_columns(self):
        a, _ = self._records()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(a[0]))
                writer.writeheader()
                writer.writerows(a)
            self.assertEqual(load_records(path), a)
            invalid = [dict(row) for row in a]
            invalid[0]["AP"] = 50.0
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(a[0]))
                writer.writeheader()
                writer.writerows(invalid)
            with self.assertRaises(ValueError):
                load_records(path)


if __name__ == "__main__":
    unittest.main()
