"""The recording wrapper leaves immutable scripts and child outputs untouched."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.p4.run_recorded import effective_parameters

ROOT = Path(__file__).resolve().parents[2]


class RecordedProtocolTests(unittest.TestCase):
    def test_supplied_script_bytes_are_immutable(self):
        expected = {
            "analysis/calib_protocol.py": "445891472b3a9d587bc0f92dee9188746670ee8c144e312b7d8cbc6ea90226ff",
            "pack_feats.py": "15abd8312bafa628cf2192ff1c64b5bc33e167e1f7d8bf5fd0dcbd545b7a8cff",
            "analysis/verify_joint_loss.py": "7546cb5c525a810ac0e422b8e08726772a38107af3d51ec22d6b36841db80eb7",
        }
        for filename, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / "tools/p4" / filename).read_bytes()).hexdigest(), digest)

    def test_defaults_and_explicit_calibration_arguments_without_import(self):
        script = ROOT / "tools/p4/analysis/calib_protocol.py"
        parameters = effective_parameters(script, ["--feats_dir", "F", "--models", "base", "--assignments", "A",
                                                    "--out_dir", "O", "--split_seeds", "2026", "--skip_rank"])
        self.assertEqual(parameters["steps"], 400)
        self.assertEqual(parameters["split_seeds"], [2026])
        self.assertEqual(parameters["assign_seeds"], [2026, 2027, 2028])
        self.assertTrue(parameters["skip_rank"])

    def test_cli_records_help_and_rejects_existing_record(self):
        script = ROOT / "tools/p4/pack_feats.py"
        before = hashlib.sha256(script.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            record = Path(temporary) / "child_not_precreated" / "run_record.json"
            command = [sys.executable, "-B", str(ROOT / "tools/p4/run_recorded.py"),
                       "--script", "pack_feats", "--record_file", str(record), "--", "--help"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(record.read_text(encoding="utf-8"))
            self.assertEqual(data["script_sha256"], before)
            self.assertTrue(data["parameters"]["help_only"])
            self.assertEqual(data["peak_cuda_memory_bytes"], 0)
            self.assertGreater(data["elapsed_seconds"], 0)
            self.assertTrue(data["git_commit"])
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(json.loads(record.read_text(encoding="utf-8")), data)
        self.assertEqual(hashlib.sha256(script.read_bytes()).hexdigest(), before)
