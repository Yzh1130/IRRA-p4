"""CPU checks for the byte-preserved round-13 reference protocol.

Set R13_INPUTS to the checked round-13 inputs directory.  Missing inputs are an
error, rather than a reason to silently skip the protocol acceptance checks.
The only stage-1c operation allowed here is assignment-table verification;
all retrieval subprocesses keep the reference script's stage-1c guard enabled.
"""
from __future__ import annotations

import ast
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[2]
TRANSFER = CODE_ROOT / "tools" / "p4" / "transfer"
PROTOCOL = TRANSFER / "r13_protocol.py"


def required_inputs():
    value = os.environ.get("R13_INPUTS")
    if not value:
        raise RuntimeError("R13_INPUTS must name the round-13 inputs directory; skipping is not allowed")
    location = Path(value).expanduser().resolve()
    for name in ("base_ref_s101.npz", "base_ref_s102.npz", "assignments_s101.csv",
                 "assignments_s102.csv", "stage1c_assignments_s101.csv", "stage1c_assignments_s102.csv"):
        if not (location / name).is_file():
            raise FileNotFoundError(f"Required round-13 input is missing: {location / name}")
    return location


INPUTS = required_inputs()


def reference_functions():
    """Execute only the requested AST function definitions, never the module."""
    names = {"l2n", "post_update", "gal_ctx", "gal_ctx2", "gal_ctx_rows",
             "gal_ctx2_rows", "lda_map", "order_rows"}
    source = PROTOCOL.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PROTOCOL))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != names:
        raise AssertionError("The preserved reference does not define the eight required functions")
    namespace = {"np": np, "args": SimpleNamespace(r_max=1.0, tie_break="index")}
    # l2n is defined before the other functions and is consequently available to
    # their bodies in this namespace, together with np and the protocol options.
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, str(PROTOCOL), "exec"), namespace)
    return namespace


def cpu_environment():
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       MKL_THREADING_LAYER="SEQUENTIAL", CUDA_VISIBLE_DEVICES="",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    return environment


def run_python(script, arguments, *, timeout=1200):
    arguments = [str(value) for value in arguments]
    if "--stage1c_final" in arguments:
        raise AssertionError("The round-13 tests must never enable stage-1c retrieval")
    # -I ignores PYTHONIOENCODING; set UTF-8 explicitly for Windows subprocess
    # diagnostics containing the non-ASCII repository/input paths.
    return subprocess.run([sys.executable, "-I", "-X", "utf8", "-B", str(script), *arguments],
                          cwd=CODE_ROOT, env=cpu_environment(), text=True, encoding="utf-8",
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
                          check=False)


def protocol_arguments(output, *extra):
    return ["--b12", ".", "--feats", INPUTS,
            "--assign", INPUTS / "assignments_s102.csv", "--out", output,
            "--base", "base_ref_s102", "--adapter", "base_ref_s102", "--tag", "test_train_lda",
            "--split_seeds", "2026", "--assign_seeds", "2026", "--mixes", "M4", "M4id",
            "--skip_joint", "--methods", "B*", "CAET", "A-SRC-2f", *extra]


class R13SyntheticFunctionTests(unittest.TestCase):
    def setUp(self):
        self.functions = reference_functions()
        self.random = np.random.default_rng(132026)

    def test_zero_transfer_is_bitwise_identity(self):
        scores = self.random.uniform(.01, .99, (9, 13))
        targets = np.array([1, 4, 7, 12])
        sources = np.array([0, 2, 3, 5])
        result = self.functions["post_update"](scores, targets, sources, np.zeros((4, 4)))
        self.assertEqual(result.tobytes(), scores.tobytes())

    def test_only_ir_only_columns_are_changed(self):
        scores = self.random.uniform(.01, .99, (8, 15))
        availability = np.array([0, 1, 0, 2, 1, 1, 0, 2, 0, 1, 2, 0, 1, 0, 2])
        targets, sources = np.flatnonzero(availability == 1), np.flatnonzero(availability == 0)
        weights = self.random.random((len(targets), len(sources)))
        weights *= self.random.uniform(0, .999, (len(targets), 1)) / weights.sum(1, keepdims=True)
        result = self.functions["post_update"](scores, targets, sources, weights)
        self.assertEqual(result[:, availability != 1].tobytes(), scores[:, availability != 1].tobytes())
        self.assertGreater(float(np.max(np.abs(result[:, targets] - scores[:, targets]))), 0)

    def test_non_ir_only_original_top1_is_preserved_in_1000_random_cases(self):
        for case in range(1000):
            scores = self.random.uniform(.01, .99, (1, 16))
            original = int(np.argmax(scores[0]))
            availability = self.random.integers(0, 3, len(scores[0]))
            availability[original] = case % 2 * 2  # Alternate complete and RGB-only top-1 items.
            availability[(original + 1) % 16] = 0
            availability[(original + 2) % 16] = 1
            targets, sources = np.flatnonzero(availability == 1), np.flatnonzero(availability == 0)
            weights = self.random.random((len(targets), len(sources)))
            weights *= self.random.uniform(0, .999, (len(targets), 1)) / weights.sum(1, keepdims=True)
            updated = self.functions["post_update"](scores, targets, sources, weights)
            self.assertEqual(int(np.argmax(updated[0])), original, f"Failed random case {case}")

    def test_bstar_tie_break_retains_original_top1_at_unit_transfer(self):
        scores = np.array([[.3, .9, .6, .1]])
        updated = self.functions["post_update"](scores, np.array([0]), np.array([1]), np.ones((1, 1)))
        self.assertEqual(updated[0, 0], updated[0, 1])
        self.functions["args"].tie_break = "bstar"
        self.assertEqual(int(self.functions["order_rows"](updated, scores)[0, 0]), 1)

    def test_gallery_context_rows_none_is_bitwise_identical(self):
        features = self.functions["l2n"](self.random.standard_normal((43, 11)))
        links = features @ features.T
        availability = self.random.integers(0, 3, len(features))
        first = self.functions["gal_ctx"](links, availability)
        second = self.functions["gal_ctx_rows"](links, availability, rows=None)
        self.assertEqual(np.float64(first).tobytes(), np.float64(second).tobytes())
        first2 = self.functions["gal_ctx2"](links, availability)
        second2 = self.functions["gal_ctx2_rows"](links, availability, rows=None)
        self.assertEqual(np.asarray(first2).tobytes(), np.asarray(second2).tobytes())

    def test_random_availability_has_zero_mean_gallery_homophily(self):
        features = self.functions["l2n"](self.random.standard_normal((80, 11)))
        links = features @ features.T
        availability = np.repeat([0, 1, 2], [38, 32, 10])
        estimates = [self.functions["gal_ctx"](links, self.random.permutation(availability)) for _ in range(5000)]
        self.assertLess(abs(float(np.mean(estimates))), .01)

    def test_lda_map_cosine_matches_centered_inverse_scatter_metric(self):
        training = self.random.standard_normal((90, 7))
        identities = np.repeat(np.arange(30), 3)
        query = self.random.standard_normal((15, 7))
        gallery = self.random.standard_normal((19, 7))
        shrink = .5
        center = training.mean(0)
        scatter = np.zeros((7, 7))
        for identity in np.unique(identities):
            residual = training[identities == identity] - training[identities == identity].mean(0)
            scatter += residual.T @ residual
        scatter /= len(training)
        scatter = (1 - shrink) * scatter + shrink * np.trace(scatter) / 7 * np.eye(7)
        inverse = np.linalg.inv(scatter)
        x, y = query - center, gallery - center
        expected = (x @ inverse @ y.T) / np.sqrt(
            np.einsum("ni,ij,nj->n", x, inverse, x)[:, None]
            * np.einsum("ni,ij,nj->n", y, inverse, y)[None, :])
        transform = self.functions["lda_map"](training, identities, shrink=shrink)
        actual = transform(query) @ transform(gallery).T
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)


class R13ProtocolSubprocessTests(unittest.TestCase):
    def test_frozen_stage1c_assignment_tables_only_are_identical(self):
        result = run_python(TRANSFER / "verify_1c_assignments.py",
                            ["--script", PROTOCOL, "--inputs_dir", INPUTS])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("ALL IDENTICAL", result.stdout)
        self.assertIn("base s102: checked 27", result.stdout)
        self.assertIn("base s101: checked 27", result.stdout)

    def test_stage1c_mixture_and_seeds_are_refused_before_evaluation(self):
        for option, forbidden in (("--mixes", "M10"), ("--assign_seeds", "5031"), ("--split_seeds", "5031")):
            with self.subTest(option=option), tempfile.TemporaryDirectory(prefix="r13_refusal_") as directory:
                arguments = protocol_arguments(Path(directory), option, forbidden)
                result = run_python(PROTOCOL, arguments)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("refusing", result.stdout.lower())
                self.assertEqual(list(Path(directory).iterdir()), [])

    def train_feature_file(self, directory, *, overlap=False):
        with np.load(INPUTS / "base_ref_s102.npz") as development:
            dimension = int(development["clean/f_ir"].shape[1])
            development_identity = int(development["clean/query_raw_pids"][0])
        random = np.random.default_rng(139001)
        training = random.standard_normal((240, dimension)).astype(np.float32)
        identities = np.repeat(np.arange(900000, 900080, dtype=np.int64), 3)
        if overlap:
            identities[0] = development_identity
        target = Path(directory) / "synthetic_train_ir.npz"
        np.savez(target, f_ir=training, raw_pids=identities)
        return target

    def test_train_lda_rejects_training_development_identity_overlap(self):
        with tempfile.TemporaryDirectory(prefix="r13_overlap_") as directory:
            target = self.train_feature_file(directory, overlap=True)
            output = Path(directory) / "out"
            output.mkdir()
            result = run_python(PROTOCOL, protocol_arguments(output, "--link_map", "train_lda", "--train_ir_npz", target))
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("training identities overlap the development identities", result.stdout)
            self.assertEqual(list(output.iterdir()), [])

    def check_train_lda_mode(self, extra_options, expected_evaluation_sets, expected_map_rows):
        with tempfile.TemporaryDirectory(prefix="r13_train_lda_") as directory:
            target = self.train_feature_file(directory)
            output = Path(directory) / "out"
            output.mkdir()  # The preserved script intentionally does not create --out.
            result = run_python(PROTOCOL, protocol_arguments(
                output, "--link_map", "train_lda", "--train_ir_npz", target, *extra_options))
            self.assertEqual(result.returncode, 0, result.stdout)
            prefix = output / "r12_test_train_lda"
            with prefix.with_name(prefix.name + "_cells.csv").open(encoding="utf-8", newline="") as stream:
                cells = list(csv.DictReader(stream))
            self.assertEqual({row["evalset"] for row in cells}, expected_evaluation_sets)
            self.assertEqual({row["mixture"] for row in cells}, {"M4", "M4id"})
            self.assertEqual({row["split"] for row in cells}, {"2026"})
            self.assertEqual({row["aseed"] for row in cells}, {"2026"})
            with prefix.with_name(prefix.name + "_run_record.json").open(encoding="utf-8") as stream:
                record = json.load(stream)
            self.assertEqual(record["wccn_diag"]["n_train_ids"], 80)
            self.assertEqual(record["wccn_diag"]["n_train_items"], 240)
            self.assertEqual(record["args"]["link_map"], "train_lda")
            self.assertFalse(record["args"]["stage1c_final"])
            maplog = prefix.with_name(prefix.name + "_maplog.csv")
            if expected_map_rows:
                with maplog.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), expected_map_rows)
                self.assertEqual({row["evalset"] for row in rows}, expected_evaluation_sets)
            else:
                self.assertFalse(maplog.exists(), "full-gallery stress tests intentionally do not produce maplog")

    def test_train_lda_default_two_folds_write_maplog(self):
        self.check_train_lda_mode([], {"half"}, 2)

    def test_train_lda_three_folds_write_maplog(self):
        self.check_train_lda_mode(["--folds", "3"], {"small0", "small1", "small2", "large"}, 9)

    def test_train_lda_full_gallery_does_not_write_maplog(self):
        self.check_train_lda_mode(["--eval_gallery", "full"], {"full"}, 0)


if __name__ == "__main__":
    unittest.main()
