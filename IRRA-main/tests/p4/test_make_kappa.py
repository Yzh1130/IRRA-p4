"""CPU tests for measured kappa generation and diagnostics interoperability."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.p4.make_kappa import STATE_SCENARIO_PAIRS, build_kappa, main
from tools.p4.train_batch_stats import load_kappa


class MakeKappaTests(unittest.TestCase):
    def _rows(self):
        values = (40.0, 80.0, 0.0, 100.0, 20.0, 60.0)
        return [{"scenario": "clean", "R1": 80.0}] + [
            {"scenario": scenario, "R1": value}
            for (_, scenario), value in zip(STATE_SCENARIO_PAIRS, values)
        ]

    def _write_metrics(self, root, rows):
        path = root / "metrics.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("scenario", "R1"), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_complete_metrics_raw_clipped_provenance_and_load_kappa(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_metrics(root, self._rows() + [
                {"scenario": "rgb_missing", "R1": 17.0},
                {"scenario": "rgb_noise_010", "R1": 27.0},
            ])
            result = build_kappa(source)
            self.assertEqual(result["R1_clean"], 80.0)
            self.assertEqual(result["source_metrics"], str(source.resolve()))
            self.assertEqual(result["source_metrics_sha256"],
                             hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertFalse(result["checkpoint_provenance_verified_by_this_tool"])
            expected_raw = (0.5, 0.0, 1.0, -0.25, 0.75, 0.25)
            expected_clipped = (0.5, 0.0, 1.0, 0.0, 0.75, 0.25)
            for (state, scenario), raw, clipped in zip(
                STATE_SCENARIO_PAIRS, expected_raw, expected_clipped
            ):
                self.assertEqual(result["states"][state]["raw"], raw)
                self.assertEqual(result["states"][state]["clipped"], clipped)
                self.assertEqual(result["states"][state]["scenario"], scenario)
                self.assertEqual(result["states"][state]["source_file"], str(source.resolve()))
            output = root / "kappa.json"
            output.write_text(json.dumps(result), encoding="utf-8")
            loaded = load_kappa(output)
            for state, raw, clipped in zip(
                (state for state, _ in STATE_SCENARIO_PAIRS), expected_raw, expected_clipped
            ):
                self.assertEqual(loaded[state]["raw"], raw)
                self.assertEqual(loaded[state]["clipped"], clipped)

    def test_missing_any_required_scenario_is_error_without_p3_substitution(self):
        rows = self._rows()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for missing in rows:
                with self.subTest(missing=missing["scenario"]):
                    source = self._write_metrics(root, [row for row in rows if row is not missing] + [
                        {"scenario": "rgb_missing", "R1": 40.0},
                        {"scenario": "ir_missing", "R1": 40.0},
                        {"scenario": "rgb_noise_010", "R1": 40.0},
                        {"scenario": "ir_noise_010", "R1": 40.0},
                    ])
                    with self.assertRaisesRegex(ValueError, "Missing required measured scenarios"):
                        build_kappa(source)

    def test_duplicate_nonfinite_out_of_range_and_zero_clean_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self._rows()
            for duplicate in rows:
                with self.subTest(duplicate=duplicate["scenario"]):
                    source = self._write_metrics(root, rows + [duplicate])
                    with self.assertRaisesRegex(ValueError, "Duplicate measured scenario"):
                        build_kappa(source)
            for value in ("nan", "inf", "-inf", "garbage", "", -1.0, 100.01):
                with self.subTest(invalid=value):
                    invalid = [dict(row) for row in rows]
                    invalid[1]["R1"] = value
                    source = self._write_metrics(root, invalid)
                    with self.assertRaises(ValueError):
                        build_kappa(source)
            zero_clean = [dict(row) for row in rows]
            zero_clean[0]["R1"] = 0.0
            source = self._write_metrics(root, zero_clean)
            with self.assertRaisesRegex(ValueError, "R1_clean must be positive"):
                build_kappa(source)

    def test_actual_cli_and_overwrite_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._write_metrics(root, self._rows())
            output = root / "new_directory" / "kappa.json"
            code_root = Path(__file__).resolve().parents[2]
            command = [sys.executable, "-B", str(code_root / "tools/p4/make_kappa.py"),
                       "--metrics_tsv", str(source), "--output", str(output)]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("Wrote six measured training-state kappas", completed.stdout)
            self.assertEqual(len(load_kappa(output)), 6)
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                main(["--metrics", str(source), "--output", str(output)])
            self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
