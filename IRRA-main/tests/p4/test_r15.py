"""CPU acceptance of the preserved round-15 protocol (commands.txt, items 1-6).

R13_INPUTS is required. R15_PKG enables reference-package numeric comparisons;
R15_QUICK=1 selects the documented server smoke. Optional R15_TEST_OUTPUTS keeps
all generated, synthetic/development-only outputs and subprocess logs for review.
No model is loaded and no official-test annotation or feature file is read here.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[2]
TRANSFER = CODE_ROOT / "tools" / "p4" / "transfer"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
QUICK = os.environ.get("R15_QUICK") == "1"
PACKAGE = Path(os.environ["R15_PKG"]).resolve() if os.environ.get("R15_PKG") else None


def required_r15_inputs():
    value = os.environ.get("R13_INPUTS")
    if not value:
        raise RuntimeError("R13_INPUTS must name the round-13 inputs directory; skipping is not allowed")
    root = Path(value).expanduser().resolve()
    for name in ("base_ref_s102.npz", "assignments_s102.csv"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    return root


INPUTS = required_r15_inputs()
ANCHORS = {
    "r15_test.py": "670a89bed5c5a7a367382bc3d9484c1f38455f828826e6f41711db4674fdb82c",
    "judge_test.py": "1375f6add366f72e2ff9677d16b6a84fab950a2ece4d99a94494dd5c123e126a",
    "r14_app.py": "78465684f8880e0f093df5471ccc9c019fc2e2685dad87a0ff5dbbd877cb0443",
    "make_synth.py": "8b61574ed9486fddb699e03ff2c9e9dae64923e96831e58230ada90304ed73b7",
    "build_pseudo.py": "8ccf85042786439884fa19685008cb94e92a1751f7756ee0cb3915e1c85ba778",
}
METHODS = "B* CAET A-SRC-2f A-2f-EM A-2f-ctxn A-2f-ctxn-g GP-same KR RVN QE-DBA KRir RVNir QEir FeatImp FeatImp-g KR+M RVN+M QE+M KRir+M RVNir+M QEir+M GP+M QE+CAET QE+SRC2f QE+EM QE+Orc DB Raw Oracle-A".split()
QUICK_METHODS = ["B*", "A-2f-ctxn", "QE-DBA", "QE+M"]
PSEUDO_METHODS = "B* CAET A-SRC-2f A-2f-EM A-2f-ctxn A-2f-ctxn-g GP-same FeatImp FeatImp-g KR KRir QE-DBA QEir QE+M QEir+M Oracle-A".split()
COMPARE_FILES = ("summary.csv", "cells.csv", "diag_cells.csv", "selection.json",
                 "ctxlog.csv", "nnlog.csv", "neardup.csv", "perq.pkl")


def cpu_environment():
    result = os.environ.copy()
    result.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  MKL_THREADING_LAYER="SEQUENTIAL", CUDA_VISIBLE_DEVICES="-1",
                  PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                  PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def protocol_output_file(directory, suffix):
    """Use the run's unique prefix; *_cells.csv also matches diag_cells.csv."""
    directory = Path(directory)
    selections = list(directory.glob("r12_*_selection.json"))
    if len(selections) != 1:
        raise AssertionError(f"Missing or ambiguous protocol run prefix: {directory}")
    prefix = selections[0].name.removesuffix("selection.json")
    return directory / (prefix + suffix)


def chosen_numbers(value):
    """Parse numeric configuration repr without eval or importing package code."""
    if isinstance(value, str):
        text = re.sub(r"(?:np\.)?(?:float32|float64|int32|int64)\(([^()]*)\)", r"\1", value)
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value
    return value


class R15ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("R15_TEST_OUTPUTS"):
            cls.root = Path(os.environ["R15_TEST_OUTPUTS"]).expanduser().resolve()
            cls.root.mkdir(parents=True, exist_ok=True)
        else:
            cls.temp = tempfile.TemporaryDirectory(prefix="r15_cpu_")
            cls.root = Path(cls.temp.name)
            cls.addClassCleanup(cls.temp.cleanup)
        cls.timings = {}
        cls.artifacts = {}
        cls.array_checks = {}
        cls.process_status = {}

    def run_script(self, name, script, arguments):
        argv = [sys.executable, "-B", str(TRANSFER / script), *map(str, arguments)]
        started = time.perf_counter()
        process = subprocess.run(argv, cwd=CODE_ROOT, env=cpu_environment(),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", timeout=3600)
        elapsed = time.perf_counter() - started
        self.timings[name] = elapsed
        self.process_status[name] = {"returncode": process.returncode, "elapsed_seconds": elapsed}
        (self.root / "subprocess_status.json").write_text(json.dumps(self.process_status, indent=2), encoding="utf-8")
        (self.root / f"{name}.log").write_text(process.stdout, encoding="utf-8")
        (self.root / f"{name}.command.json").write_text(json.dumps(argv, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.root / "timings.json").write_text(json.dumps(self.timings, indent=2), encoding="utf-8")
        self.assertEqual(process.returncode, 0, f"{name}: exit {process.returncode}\n{process.stdout[-20000:]}")
        return process.stdout

    def new_output(self, name):
        output = self.root / name
        output.mkdir()  # Existing outputs are never overwritten, including test artifacts.
        return output

    def assert_identical(self, first, second, tag):
        for suffix in COMPARE_FILES:
            with self.subTest(tag=tag, file=suffix):
                left, right = first / f"r12_{tag}_{suffix}", second / f"r12_{tag}_{suffix}"
                self.assertTrue(left.is_file(), str(left))
                self.assertTrue(right.is_file(), str(right))
                self.assertEqual(left.read_bytes(), right.read_bytes(), suffix)

    def common_options(self, output, tag):
        return ["--b12", ".", "--feats", INPUTS, "--assign", INPUTS / "assignments_s102.csv",
                "--out", output, "--base", "base_ref_s102", "--adapter", "base_ref_s102", "--tag", tag]

    def check_derived(self, directory, names, *, strict):
        if PACKAGE is None:
            return
        expected = json.loads((PACKAGE / "ref" / "derived_inputs_arrays.json").read_text(encoding="utf-8"))
        for name in names:
            path = directory / name
            record = expected[name]
            if name.endswith(".csv"):
                actual = {"file_sha256": sha256(path)}
            else:
                with np.load(path, allow_pickle=False) as data:
                    actual = {key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype),
                                    "sha256": hashlib.sha256(data[key].tobytes()).hexdigest()}
                              for key in data.files}
            self.array_checks[name] = {"matches_reference": actual == record, "actual": actual}
            if strict:
                self.assertEqual(actual, record, f"Pseudo input differs: {name}")
        (self.root / "derived_inputs_check.json").write_text(json.dumps(self.array_checks, indent=2), encoding="utf-8")

    def test_01_reference_script_bytes(self):
        for name, expected in ANCHORS.items():
            with self.subTest(script=name):
                self.assertEqual(sha256(TRANSFER / name), expected)
                if PACKAGE is not None:
                    self.assertEqual((TRANSFER / name).read_bytes(), (PACKAGE / "scripts" / name).read_bytes())

    def test_02_development_same_machine_bytes(self):
        started = time.perf_counter()
        outputs = {}
        for script in ("r14_app.py", "r15_test.py"):
            name = "reg_" + script.removesuffix(".py")
            output = self.new_output(name)
            opts = self.common_options(output, "reg")
            opts += ["--split_seeds", "2026", "--assign_seeds", *(["2026"] if QUICK else ["2026", "2027", "2028"]),
                     "--mixes", *(["M4", "M5f"] if QUICK else "M0 M4 M6c M1g M4c60 M9 M5e M5f M4id M10".split()),
                     "--skip_joint", "--methods", *(QUICK_METHODS if QUICK else METHODS),
                     "--grid_preset", "v2", "--select_rule", "constrained", "--r_max", "0.999999", "--tie_break", "bstar"]
            self.run_script(name, script, opts)
            outputs[script] = output
        self.assert_identical(outputs["r14_app.py"], outputs["r15_test.py"], "reg")
        self.artifacts["reg"] = outputs["r15_test.py"]
        self.timings["item_02"] = time.perf_counter() - started
        self.persist()

    @unittest.skipIf(QUICK, "R15_QUICK=1: item 3 omitted by the documented quick mode")
    def test_03_pseudo_test_same_machine_bytes(self):
        started = time.perf_counter()
        pseudo = self.new_output("pseudo_inputs")
        self.run_script("build_pseudo", "build_pseudo.py", [TRANSFER / "r15_test.py", INPUTS, "base_ref_s102", INPUTS / "assignments_s102.csv", pseudo])
        self.check_derived(pseudo, ("pseudo_dev.npz", "pseudo_test.npz", "pseudo_assign.csv"), strict=True)
        outputs = {}
        for script in ("r14_app.py", "r15_test.py"):
            name = "pseudo_" + script.removesuffix(".py")
            output = self.new_output(name)
            opts = ["--b12", ".", "--feats", pseudo, "--assign", pseudo / "pseudo_assign.csv", "--out", output,
                    "--base", "pseudo_dev", "--adapter", "pseudo_dev", "--tag", "pseudo", "--assign_seeds", "2026",
                    "--mixes", "D_M4", "D_M5f", "D_M1g", "D_M4id", "--skip_joint", "--methods", *PSEUDO_METHODS,
                    "--r_max", "0.999999", "--tie_break", "bstar", "--test_npz", pseudo / "pseudo_test.npz",
                    "--test_seed", "2026", "--allow_crossfit_test", "--grid_preset", "v2", "--select_rule", "constrained"]
            self.run_script(name, script, opts)
            outputs[script] = output
        self.assert_identical(outputs["r14_app.py"], outputs["r15_test.py"], "pseudo")
        self.artifacts["pseudo"] = outputs["r15_test.py"]
        self.timings["item_03"] = time.perf_counter() - started
        self.persist()

    def test_04_synthetic_test_scale_and_judge(self):
        started = time.perf_counter()
        synth = self.new_output("synth_inputs")
        self.run_script("make_synth", "make_synth.py", [INPUTS / "base_ref_s102.npz", synth / "synth_test.npz", synth / "synth_train_ir.npz"])
        self.check_derived(synth, ("synth_test.npz", "synth_train_ir.npz"), strict=False)
        outputs = {}
        specifications = [("syn2", ["7001", "7002"])] if QUICK else [("syn2", ["7001", "7002"]), ("syn1", ["7001"]), ("syn1b", ["7002"])]
        for tag, seeds in specifications:
            output = self.new_output(tag)
            opts = ["--b12", ".", "--feats", INPUTS, "--assign", "none", "--base", "base_ref_s102", "--adapter", "base_ref_s102",
                    "--assign_seeds", "7031", "--mixes", *(["M4", "M5f"] if QUICK else ["M0", "M4", "M5f", "M10"]),
                    "--skip_joint", "--methods", *(QUICK_METHODS if QUICK else METHODS), "--grid_preset", "dev", "--select_rule", "constrained",
                    "--r_max", "0.999999", "--tie_break", "bstar", "--test_npz", synth / "synth_test.npz", "--link_map", "train_lda",
                    "--train_ir_npz", synth / "synth_train_ir.npz", "--test_parts", "3", "--out", output, "--tag", tag, "--test_seed", *seeds]
            self.run_script(tag, "r15_test.py", opts)
            outputs[tag] = output
        _, summary = csv_rows(outputs["syn2"] / "r12_syn2_summary.csv")
        evalsets = {row["evalset"] for row in summary}
        self.assertIn("test", evalsets)
        self.assertTrue(any(value == "p3" or value.startswith("p3_") for value in evalsets), evalsets)
        if not QUICK:
            fields, pooled = csv_rows(outputs["syn2"] / "r12_syn2_cells.csv")
            for seed, tag in (("7001", "syn1"), ("7002", "syn1b")):
                single_fields, single = csv_rows(outputs[tag] / f"r12_{tag}_cells.csv")
                self.assertEqual(fields, single_fields)
                self.assertEqual([row for row in pooled if row["split"] == seed], single, f"fit {seed}: original CSV field strings differ")
            self.artifacts["synth"] = outputs["syn1"]
        qraw = f"syn={INPUTS / 'base_ref_s102.npz'},{synth / 'synth_test.npz'}"
        judge = self.run_script("judge_test", "judge_test.py", ["--runs", f"syn={outputs['syn2']}", "--qraw", qraw, "--evalset", "test", "--boot", "200"])
        self.assertIn("[claims]", judge)
        self.assertIn("[table 4]", judge)
        if not QUICK:
            judge_p3 = self.run_script("judge_p3", "judge_test.py", ["--runs", f"syn={outputs['syn2']}", "--qraw", qraw, "--evalset", "p3", "--boot", "200", "--no_claims"])
            self.assertIn("[table 4]", judge_p3)
            merged = self.run_script("judge_merged", "judge_test.py", ["--runs", f"syn={outputs['syn1']},{outputs['syn1b']}", "--qraw", qraw, "--evalset", "test", "--boot", "200"])
            def tables(value):
                return value[value.index("[table 1]"):value.index("\n[claims]")]
            self.assertEqual(tables(judge), tables(merged), "Pooled judge tables 1-4 differ from split runs")
        self.timings["item_04"] = time.perf_counter() - started
        self.persist()

    def persist(self):
        (self.root / "timings.json").write_text(json.dumps(self.timings, indent=2), encoding="utf-8")
        (self.root / "artifacts.json").write_text(json.dumps({key: str(value) for key, value in self.artifacts.items()}, indent=2), encoding="utf-8")

    @unittest.skipIf(PACKAGE is None or QUICK, "Cross-platform comparison requires R15_PKG and full mode")
    def test_06_reference_numeric_comparison(self):
        comparisons = {}
        for tag, directory in (("reg", "reg_r15"), ("pseudo", "pseudo_r15"), ("synth", "synth_r15")):
            self.assertIn(tag, self.artifacts, "Run the ordered full acceptance suite before cross-platform comparison")
            actual_root = self.artifacts[tag]
            reference_root = PACKAGE / "ref" / directory
            for suffix in ("summary.csv", "cells.csv", "diag_cells.csv", "selection.json"):
                reference = protocol_output_file(reference_root, suffix)
                if not reference.is_file() and tag == "synth" and suffix == "summary.csv":
                    # This is the only absent numeric reference documented by commands.txt.
                    continue
                self.assertTrue(reference.is_file(), f"Missing reference: {tag}/{suffix}")
                actual = protocol_output_file(actual_root, suffix)
                self.assertTrue(actual.is_file(), f"Missing actual output: {tag}/{suffix}")
                key = f"{tag}/{suffix}"
                if suffix.endswith(".json"):
                    maximum, count = self.compare_selection(actual, reference)
                else:
                    maximum, count = self.compare_csv_numbers(actual, reference)
                label = "pass" if maximum <= 1e-6 else "platform_numeric_difference"
                comparisons[key] = {"max_abs_difference": maximum, "numeric_values": count, "classification": label}
                (self.root / "cross_platform.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
                self.assertLessEqual(maximum, .01, f"{key}: {maximum} > 0.01; stop and investigate without altering protocol")

    def compare_csv_numbers(self, actual_path, reference_path):
        actual_fields, actual = csv_rows(actual_path)
        ref_fields, reference = csv_rows(reference_path)
        self.assertEqual(set(actual_fields), set(ref_fields), "CSV columns differ")
        keys = [key for key in ("split", "fold", "evalset", "mixture", "aseed", "method") if key in ref_fields]
        self.assertTrue(keys)
        def indexed(rows):
            result = {tuple(row[key] for key in keys): row for row in rows}
            self.assertEqual(len(result), len(rows), "Duplicate reference alignment keys")
            return result
        left, right = indexed(actual), indexed(reference)
        self.assertEqual(set(left), set(right), "Row alignment differs")
        differences = []
        for key in right:
            for name in ref_fields:
                if name in keys:
                    continue
                try:
                    # These three protocol CSVs contain only numeric values
                    # outside their alignment keys. Pandas writes NaN as blank.
                    b = float(right[key][name]) if right[key][name].strip() else math.nan
                except ValueError:
                    self.fail(f"Invalid reference numeric value: {key}/{name}={right[key][name]!r}")
                try:
                    a = float(left[key][name]) if left[key][name].strip() else math.nan
                except ValueError:
                    self.fail(f"Invalid actual numeric value: {key}/{name}={left[key][name]!r}")
                differences.append(self.numeric_difference(a, b, (key, name)))
        self.assertTrue(differences, "No numeric reference values were compared")
        return max(differences, default=0.0), len(differences)

    def numeric_difference(self, actual, reference, context):
        if math.isnan(actual) or math.isnan(reference):
            self.assertTrue(math.isnan(actual) and math.isnan(reference), f"NaN mismatch: {context}")
            return 0.0
        if math.isinf(actual) or math.isinf(reference):
            self.assertEqual(actual, reference, f"Infinity mismatch: {context}")
            return 0.0
        return abs(actual - reference)

    def compare_selection(self, actual_path, reference_path):
        actual = json.loads(actual_path.read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        differences = []
        def compare(a, b, context):
            a, b = chosen_numbers(a), chosen_numbers(b)
            if isinstance(a, bool) or isinstance(b, bool):
                self.assertEqual(a, b, context)
            elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
                differences.append(self.numeric_difference(float(a), float(b), context))
            elif isinstance(a, dict) and isinstance(b, dict):
                self.assertEqual(set(a), set(b), context)
                for key in b:
                    compare(a[key], b[key], f"{context}/{key}")
            elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                self.assertEqual(len(a), len(b), context)
                for index, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, f"{context}/{index}")
            else:
                self.assertEqual(a, b, context)
        sort_key = lambda row: (row["split"], row["fold"], row["method"])
        compare(sorted(actual, key=sort_key), sorted(reference, key=sort_key), "selection")
        return max(differences, default=0.0), len(differences)


class R15NumericComparisonUnitTests(unittest.TestCase):
    def test_protocol_paths_distinguish_cells_and_diag(self):
        with tempfile.TemporaryDirectory(prefix="r15_prefix_") as folder:
            root = Path(folder)
            for suffix in ("selection.json", "cells.csv", "diag_cells.csv"):
                (root / ("r12_syn1_" + suffix)).write_text("", encoding="utf-8")
            self.assertEqual(protocol_output_file(root, "cells.csv"), root / "r12_syn1_cells.csv")
            self.assertEqual(protocol_output_file(root, "diag_cells.csv"), root / "r12_syn1_diag_cells.csv")
            (root / "r12_extra_selection.json").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "ambiguous"):
                protocol_output_file(root, "cells.csv")

    def test_numeric_comparison_rejects_corrupt_or_missing_values(self):
        checker = R15ProtocolTests("test_01_reference_script_bytes")
        with tempfile.TemporaryDirectory(prefix="r15_numeric_") as folder:
            actual, reference = Path(folder) / "actual.csv", Path(folder) / "reference.csv"
            reference.write_text("method,R1,mAP\nB*,78.123,60\n", encoding="utf-8")
            for content in ("method,mAP\nB*,60\n",
                            "method,R1,mAP\nB*,CORRUPT,60\n",
                            "method,R1,mAP\nB*,,60\n"):
                with self.subTest(content=content):
                    actual.write_text(content, encoding="utf-8")
                    with self.assertRaises(AssertionError):
                        checker.compare_csv_numbers(actual, reference)
            reference.write_text("method\nB*\n", encoding="utf-8")
            actual.write_bytes(reference.read_bytes())
            with self.assertRaisesRegex(AssertionError, "No numeric"):
                checker.compare_csv_numbers(actual, reference)

    def test_numeric_comparison_reports_difference_and_matching_nan(self):
        checker = R15ProtocolTests("test_01_reference_script_bytes")
        with tempfile.TemporaryDirectory(prefix="r15_numeric_") as folder:
            actual, reference = Path(folder) / "actual.csv", Path(folder) / "reference.csv"
            reference.write_text("method,R1,mAP\nB*,78.123,nan\n", encoding="utf-8")
            actual.write_text("method,R1,mAP\nB*,78.124,\n", encoding="utf-8")
            maximum, count = checker.compare_csv_numbers(actual, reference)
            self.assertAlmostEqual(maximum, .001, places=12)
            self.assertEqual(count, 2)


# The exporter-only synthetic tests are appended below; they do not load models.

# Section ⑤: synthetic-only split exporter tests. No official data or model is read.
import copy as r15_copy
import contextlib as r15_contextlib
import hashlib as r15_hashlib
import io as r15_io
import json as r15_json
from pathlib import Path as r15_Path
import shutil as r15_shutil
import tempfile as r15_tempfile
from types import SimpleNamespace as r15_SimpleNamespace
import unittest as r15_unittest
from unittest.mock import MagicMock as r15_MagicMock, patch as r15_patch

import torch as r15_torch

from tools.p4 import export_dev_records as r15_old_exporter
from tools.p4 import export_eval_split as r15_split_exporter


def _r15_export_fixture(root):
    """Nonmonotone test IDs, interleaved rows, repeated IDs and multi-captions."""
    data = root / "data" / "RGBT-PEDES" / "RGBT-PEDES-Final"
    (data / "RGB").mkdir(parents=True)
    (data / "T").mkdir()
    annos = [
        {"id": 30, "split": "train", "file_path": "RGBT/train.jpg", "captions": ["training coat"]},
        {"id": 90, "split": "test", "file_path": "RGBT/t90a.jpg", "captions": ["red coat", "black shoes"]},
        {"id": 10, "split": "train", "file_path": "RGBT/dev10.jpg", "captions": ["blue shirt", "white shoes"]},
        {"id": 70, "split": "test", "file_path": "RGBT/t70.jpg", "captions": ["green shirt"]},
        {"id": 90, "split": "test", "file_path": "RGBT/t90b.jpg", "captions": ["red trousers"]},
        {"id": 20, "split": "train", "file_path": "RGBT/dev20.jpg", "captions": ["yellow hat"]},
        {"id": 80, "split": "test", "file_path": "RGBT/t80.jpg", "captions": ["grey coat"]},
    ]
    annotation = data / "Caption.json"
    annotation.write_text(r15_json.dumps(annos), encoding="utf-8")
    split = root / "dev.json"
    split_data = {"dev_ids": [20, 10], "train_reduced_ids": [30], "n_dev": 2,
                  "n_train_reduced": 1,
                  "anno_md5": r15_hashlib.md5(annotation.read_bytes()).hexdigest()}
    split.write_text(r15_json.dumps(split_data), encoding="utf-8")
    checkpoint = root / "synthetic.pth"
    checkpoint.write_bytes(b"synthetic checkpoint; never loaded by a real model")
    config = root / "configs.yaml"
    config.write_text("synthetic_config: true\n", encoding="utf-8")
    return r15_SimpleNamespace(root=root, data=data, annotation=annotation,
                               split=split, annos=annos, split_data=split_data,
                               checkpoint=checkpoint, config=config)


def _r15_rewrite_annotation(fixture, annos):
    fixture.annotation.write_text(r15_json.dumps(annos), encoding="utf-8")
    split = r15_copy.deepcopy(fixture.split_data)
    split["anno_md5"] = r15_hashlib.md5(fixture.annotation.read_bytes()).hexdigest()
    fixture.split.write_text(r15_json.dumps(split), encoding="utf-8")


def _r15_export_cli(fixture, split="dev", extra=()):
    return r15_split_exporter.build_parser().parse_args([
        "--checkpoint", str(fixture.checkpoint), "--config_file", str(fixture.config),
        "--dev_split_file", str(fixture.split), "--output_dir", str(fixture.root / "export"),
        "--root_dir", str(fixture.root / "data"), "--device", "cpu", "--split", split,
        *extra,
    ])


def _r15_mock_export(fixture, split="dev", *, wrong_classes=False, wrong_captions=False,
                     saved_split=None):
    """Exercise real export control flow with no encoder and a mocked Evaluator."""
    cli = _r15_export_cli(fixture, split, ["--save_modality_features"])
    args = r15_SimpleNamespace(dataset_name="RGBT-PEDES", root_dir=str(fixture.root / "data"),
                               dev_split_file=str(saved_split or fixture.split))
    captured = {"loader_eval_dataset": [], "saved_data": [], "evaluators": []}
    selected = (r15_old_exporter.validate_dev_split(fixture.annotation, fixture.split)
                if split == "dev" else
                r15_split_exporter.build_test_data(fixture.annotation, fixture.split))
    model = r15_MagicMock(name="synthetic_model")
    model.to.return_value = model
    model.eval.return_value = model

    def loader(loader_args):
        captured["loader_eval_dataset"].append(loader_args.eval_dataset)
        captions = [row["caption"] for row in selected["queries"]]
        if wrong_captions:
            captions = list(reversed(captions))
        txt = r15_SimpleNamespace(dataset=r15_SimpleNamespace(captions=captions))
        return r15_SimpleNamespace(), txt, selected["n_train"] + int(wrong_classes)

    class FakeEvaluator:
        def __init__(self, image_loader, text_loader, source):
            self.last_eval_result = {key: 100.0 for key in r15_old_exporter.METRIC_FIELDS[1:]}
            captured["evaluators"].append(self)

    class FakeRecorder:
        def save(self, path, payload, eval_data, scenario):
            captured["saved_data"].append(r15_copy.deepcopy(eval_data))
            path = r15_Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            r15_torch.save({"scenario": scenario}, path)
            return {}

    def evaluate(evaluator, ignored_model, enabled):
        gids = r15_torch.tensor(selected["gallery_pids"], dtype=r15_torch.long)
        qids = r15_torch.tensor([q["eval_pid"] for q in selected["queries"]], dtype=r15_torch.long)
        sim = qids[:, None].eq(gids[None, :]).float()
        path = r15_Path(args.eval_scores_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        r15_torch.save({"sim": sim, "img_pids": gids, "txt_pids": qids,
                        "img_indices": r15_torch.arange(len(gids)),
                        "txt_indices": r15_torch.arange(len(qids))}, path)
        return FakeRecorder() if enabled else None

    with r15_patch("utils.iotools.load_train_configs", return_value=args), \
            r15_patch("datasets.build_dataloader", side_effect=loader), \
            r15_patch("model.build_model", return_value=model), \
            r15_patch("utils.metrics.Evaluator", FakeEvaluator), \
            r15_patch("processor.processor.apply_img_local_filter_stage"), \
            r15_patch("utils.logger.setup_logger"), \
            r15_patch.object(r15_split_exporter, "load_strict_checkpoint", return_value={"epoch": 1}), \
            r15_patch.object(r15_split_exporter, "_eval_with_optional_features", side_effect=evaluate), \
            r15_patch.object(r15_split_exporter, "dropblank_metrics", side_effect=lambda f, m, s: {"scenario": s, **m}):
        summary = r15_split_exporter.export_split(cli)
    return cli, captured, selected, summary


class R15ExportUnitTests(r15_unittest.TestCase):
    """Requirement ⑤, entirely synthetic even when called by the full suite."""

    def test_05a_test_order_and_contiguous_pids_match_actual_dataset_processor(self):
        from datasets.rgbtpedes import RGBTPEDES

        with r15_tempfile.TemporaryDirectory() as temporary:
            fixture = _r15_export_fixture(r15_Path(temporary))
            actual = r15_split_exporter.build_test_data(fixture.annotation, fixture.split)
            dataset = RGBTPEDES(root=str(fixture.root / "data"), verbose=False,
                                dev_split_file=str(fixture.split))
            expected, labels = dataset._process_anno(dataset.test_annos)
            self.assertEqual(actual["gallery_pids"], expected["image_pids"])
            self.assertEqual(actual["gallery_raw_pids"], [90, 70, 90, 80])
            self.assertEqual(actual["gallery_pids"], [2, 0, 2, 1])
            self.assertEqual([q["caption"] for q in actual["queries"]], expected["captions"])
            self.assertEqual([q["eval_pid"] for q in actual["queries"]], expected["caption_pids"])
            self.assertEqual([q["query_id"] for q in actual["queries"]], ["1:0", "1:1", "3:0", "4:0", "6:0"])
            self.assertEqual([q["pid"] for q in actual["queries"]], [90, 90, 70, 90, 80])
            self.assertEqual(actual["n_test_ids"], len(labels))
            self.assertEqual(actual["test_records"], 4)
            self.assertEqual(actual["test_captions"], 5)
            dev = r15_old_exporter.validate_dev_split(fixture.annotation, fixture.split)
            self.assertEqual(actual["n_train"], dev["n_train"])
            for key in ("annotation", "split", "split_identity_sha256"):
                self.assertEqual(actual[key], dev[key])

    def test_05b_test_identity_overlap_with_dev_or_train_rejected(self):
        for raw_pid in (10, 30):
            with self.subTest(overlap_pid=raw_pid), r15_tempfile.TemporaryDirectory() as temporary:
                fixture = _r15_export_fixture(r15_Path(temporary))
                annos = r15_copy.deepcopy(fixture.annos)
                annos[1]["id"] = raw_pid
                _r15_rewrite_annotation(fixture, annos)
                with self.assertRaisesRegex(ValueError, "overlap"):
                    r15_split_exporter.build_test_data(fixture.annotation, fixture.split)

    def test_05c_absent_test_rows_and_zero_caption_identity_rejected(self):
        for case in ("no_test", "all_empty", "one_empty_identity"):
            with self.subTest(case=case), r15_tempfile.TemporaryDirectory() as temporary:
                fixture = _r15_export_fixture(r15_Path(temporary))
                annos = r15_copy.deepcopy(fixture.annos)
                if case == "no_test":
                    annos = [a for a in annos if a["split"] != "test"]
                else:
                    for row in annos:
                        if row["split"] == "test" and (case == "all_empty" or row["id"] == 70):
                            row["captions"] = []
                _r15_rewrite_annotation(fixture, annos)
                with self.assertRaises(ValueError):
                    r15_split_exporter.build_test_data(fixture.annotation, fixture.split)

    def test_05d_split_is_required_and_old_parser_arguments_are_preserved(self):
        with r15_contextlib.redirect_stderr(r15_io.StringIO()), self.assertRaises(SystemExit):
            r15_split_exporter.build_parser().parse_args(["--checkpoint", "x", "--output_dir", "y"])
        old = r15_old_exporter.build_parser().parse_args(["--checkpoint", "x", "--output_dir", "y"])
        new = r15_split_exporter.build_parser().parse_args(["--checkpoint", "x", "--output_dir", "y", "--split", "dev"])
        self.assertEqual({k: v for k, v in vars(new).items() if k != "split"}, vars(old))

    def test_05e_existing_output_is_refused_before_building_a_model(self):
        for has_file in (False, True):
            with self.subTest(nonempty=has_file), r15_tempfile.TemporaryDirectory() as temporary:
                fixture = _r15_export_fixture(r15_Path(temporary))
                cli = _r15_export_cli(fixture)
                output = r15_Path(cli.output_dir)
                output.mkdir()
                if has_file:
                    (output / "metrics.tsv").write_bytes(b"keep existing bytes\n")
                before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
                with r15_patch("model.build_model") as builder, self.assertRaises(FileExistsError):
                    r15_split_exporter.export_split(cli)
                builder.assert_not_called()
                self.assertEqual(before, {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()})

    def test_05f_mocked_export_routes_dev_and_test_and_passes_exact_saved_data(self):
        for split in ("dev", "test"):
            with self.subTest(split=split), r15_tempfile.TemporaryDirectory() as temporary:
                fixture = _r15_export_fixture(r15_Path(temporary))
                cli, captured, selected, summary = _r15_mock_export(fixture, split)
                self.assertEqual(captured["loader_eval_dataset"], ["val" if split == "dev" else "test"])
                self.assertEqual(len(captured["evaluators"]), 11)
                self.assertEqual(len({id(e) for e in captured["evaluators"]}), 11)
                selected_rows = [(i, row) for i, row in enumerate(fixture.annos)
                                 if (row["split"] == "test" if split == "test" else
                                     row["split"] == "train" and row["id"] in {10, 20})]
                expected = r15_copy.deepcopy(selected)
                expected["gallery_annotation_indices"] = [i for i, row in selected_rows]
                expected["gallery_file_paths"] = [row["file_path"] for i, row in selected_rows]
                self.assertEqual(captured["saved_data"], [expected] * 11)
                self.assertEqual(len(list((r15_Path(cli.output_dir) / "scores").glob("*.pt"))), 11)
                if split == "test":
                    record = r15_json.loads((r15_Path(cli.output_dir) / "export_run_record.json").read_text(encoding="utf-8"))
                    for key, expected_count in (("split", "test"), ("n_test_ids", 3), ("test_records", 4), ("test_captions", 5)):
                        self.assertEqual(summary["provenance"][key], expected_count)
                        self.assertEqual(record[key], expected_count)

    def test_05g_training_split_class_count_and_caption_order_guards_are_kept(self):
        for case in ("class_count", "caption_order", "saved_split"):
            with self.subTest(case=case), r15_tempfile.TemporaryDirectory() as temporary:
                fixture = _r15_export_fixture(r15_Path(temporary))
                saved_split = None
                if case == "saved_split":
                    saved_split = fixture.root / "other_dev.json"
                    data = r15_copy.deepcopy(fixture.split_data)
                    data["dev_ids"], data["train_reduced_ids"] = [20, 30], [10]
                    saved_split.write_text(r15_json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "verified reduced|Caption loader order|training split does not match"):
                    _r15_mock_export(fixture, "test", wrong_classes=case == "class_count",
                                     wrong_captions=case == "caption_order", saved_split=saved_split)
                self.assertFalse((fixture.root / "export").exists())

    def test_05h_dev_metrics_scores_and_features_are_byte_identical_to_existing_export(self):
        # Reuse the established synthetic actual-Evaluator fixture, never a CLIP
        # checkpoint or an official split. Only fixtures are removed/recreated.
        import sys as r15_sys

        test_directory = str(r15_old_exporter.CODE_ROOT / "tests" / "p4")
        if test_directory not in r15_sys.path:
            r15_sys.path.insert(0, test_directory)
        import test_modality_features as fixtures

        with r15_tempfile.TemporaryDirectory() as temporary:
            cli = fixtures._setup_export(r15_Path(temporary))
            cli.save_modality_features = True
            cli.split = "dev"  # The existing entry point tolerates this metadata.
            output = r15_Path(cli.output_dir)
            with r15_contextlib.redirect_stdout(r15_io.StringIO()), \
                    r15_patch.object(r15_old_exporter.time, "perf_counter", return_value=0.0), \
                    r15_patch.object(r15_old_exporter, "datetime") as clock:
                clock.now.return_value.isoformat.return_value = "2000-01-01T00:00:00+00:00"
                old_summary, _ = fixtures._run_export(cli, r15_old_exporter.export)
            ignored = {"test_log.txt"}
            before = {str(p.relative_to(output)): p.read_bytes()
                      for p in output.rglob("*") if p.is_file() and p.name not in ignored}
            r15_shutil.rmtree(output)
            with r15_contextlib.redirect_stdout(r15_io.StringIO()), \
                    r15_patch.object(r15_old_exporter.time, "perf_counter", return_value=0.0), \
                    r15_patch.object(r15_split_exporter, "datetime") as clock:
                clock.now.return_value.isoformat.return_value = "2000-01-01T00:00:00+00:00"
                new_summary, _ = fixtures._run_export(cli, r15_split_exporter.export_split)
            after = {str(p.relative_to(output)): p.read_bytes()
                     for p in output.rglob("*") if p.is_file() and p.name not in ignored}
            self.assertEqual(old_summary, new_summary)
            self.assertEqual(set(before), set(after))
            self.assertIn("metrics.tsv", before)
            self.assertIn("metrics_dropblank.tsv", before)
            self.assertIn("summary.json", before)
            self.assertIn("per_query.csv", before)
            self.assertIn("export_run_record.json", before)
            self.assertEqual(len([name for name in before if name.startswith("scores/") or name.startswith("scores\\")]), 11)
            self.assertEqual(len([name for name in before if name.startswith("features/") or name.startswith("features\\")]), 11)
            for name in before:
                with self.subTest(output=name):
                    self.assertEqual(before[name], after[name])


class R15TieAwareCheckTests(r15_unittest.TestCase):
    """Round-15 fix: exact score ties may move the CPU per-query mAP only within their bounds."""

    @staticmethod
    def _payload(rows, txt_pids, img_pids):
        return {"sim": r15_torch.tensor(rows, dtype=r15_torch.float32),
                "txt_pids": r15_torch.tensor(txt_pids, dtype=r15_torch.long),
                "img_pids": r15_torch.tensor(img_pids, dtype=r15_torch.long)}

    def test_05i_tie_bounds_hand_examples(self):
        tied = self._payload([[0.5, 0.5, 0.1]], [0], [0, 1, 2])
        self.assertEqual(r15_split_exporter.ap_tie_bounds(tied), (50.0, 100.0, 1))
        untied = self._payload([[0.9, 0.5, 0.1]], [0], [0, 1, 2])
        self.assertEqual(r15_split_exporter.ap_tie_bounds(untied), (100.0, 100.0, 0))
        two = self._payload([[0.7, 0.7, 0.7, 0.1]], [0], [0, 1, 0, 2])   # positives at ties: best ranks 1,2; worst 2,3
        worst, best, n = r15_split_exporter.ap_tie_bounds(two)
        self.assertAlmostEqual(best, 100.0, places=12)
        self.assertAlmostEqual(worst, 100.0 * (1 / 2 + 2 / 3) / 2, places=12)
        self.assertEqual(n, 1)

    def test_05j_check_accepts_ties_within_bounds_and_rejects_others(self):
        payload = self._payload([[0.5, 0.5, 0.1], [0.1, 0.9, 0.2]], [0, 1], [0, 1, 2])   # query 0 tied; bounds [75, 100]
        pessimistic = [{"AP": 0.5}, {"AP": 1.0}]
        for split in ("test", "dev"):
            with self.subTest(split=split):
                logger, ties = r15_MagicMock(), {}
                r15_split_exporter.check_per_query_ap(pessimistic, payload, {"mAP": 100.0}, "rgb_blur_k7", split, logger, ties)
                self.assertTrue(logger.info.called)
                if split == "test":
                    self.assertEqual(ties["rgb_blur_k7"], {"tie_dependent_queries": 1, "per_query_mean_ap": 75.0,
                                                           "ap_bounds": [75.0, 100.0], "evaluator_map": 100.0})
                else:
                    self.assertEqual(ties, {})
                with self.assertRaisesRegex(ValueError, "Per-query AP disagrees.*tie bounds"):
                    r15_split_exporter.check_per_query_ap(pessimistic, payload, {"mAP": 100.5}, "x", split, logger, {})
        untied = self._payload([[0.9, 0.5, 0.1]], [0], [0, 1, 2])
        for split in ("test", "dev"):
            with self.subTest(split=split, case="untied mismatch"), self.assertRaisesRegex(ValueError, "Per-query AP disagrees"):
                r15_split_exporter.check_per_query_ap([{"AP": 1.0}], untied, {"mAP": 99.0}, "x", split, r15_MagicMock(), {})

    def test_05k_dev_strict_pass_adds_nothing_and_test_records_zero_ties(self):
        payload = self._payload([[0.9, 0.5, 0.1]], [0], [0, 1, 2])
        logger, ties = r15_MagicMock(), {}
        with r15_patch.object(r15_split_exporter, "ap_tie_bounds") as bounds:
            r15_split_exporter.check_per_query_ap([{"AP": 1.0}], payload, {"mAP": 100.0}, "clean", "dev", logger, ties)
            bounds.assert_not_called()
        self.assertEqual(ties, {})
        self.assertFalse(logger.info.called)
        r15_split_exporter.check_per_query_ap([{"AP": 1.0}], payload, {"mAP": 100.0}, "clean", "test", logger, ties)
        self.assertEqual(ties["clean"]["tie_dependent_queries"], 0)
        self.assertFalse(logger.info.called)

    def test_05l_mocked_exports_record_tie_check_only_for_test(self):
        for split in ("dev", "test"):
            with self.subTest(split=split), r15_tempfile.TemporaryDirectory() as temporary:
                fixture = _r15_export_fixture(r15_Path(temporary))
                cli, _, _, summary = _r15_mock_export(fixture, split)
                record = r15_json.loads((r15_Path(cli.output_dir) / "export_run_record.json").read_text(encoding="utf-8"))
                if split == "dev":
                    self.assertNotIn("ap_tie_check", record)
                    self.assertNotIn("ap_tie_check", summary["provenance"])
                else:
                    self.assertEqual(len(record["ap_tie_check"]), 11)
                    self.assertTrue(all(v["tie_dependent_queries"] == 0 for v in record["ap_tie_check"].values()))
                    self.assertEqual(summary["provenance"]["ap_tie_check"], record["ap_tie_check"])


if __name__ == "__main__":
    unittest.main()
