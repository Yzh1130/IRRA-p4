"""Synthetic R19 MLM-once and runtime-recipe checks; no dataset is loaded."""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from tools.p4.r19.training import (COMMON_ALLOWED, RGBNT_ALLOWED, config_differences,
                                   derive_config, runtime_health_base, validate_derivation, write_derived_config)
from utils.config import load_config_mapping


def extract_functions(path, names, namespace=None, source=None):
    namespace = {} if namespace is None else namespace
    tree = ast.parse(source if source is not None else path.read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError("Missing named definitions")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def loss_builder(source=None):
    return extract_functions(CODE / "processor/processor.py", {"_build_total_loss"}, source=source)["_build_total_loss"]


def parser_scope():
    modes = ("rgb_drop", "ir_drop", "rgb_noise", "ir_noise", "rgb_attenuate", "ir_attenuate")
    return extract_functions(CODE / "utils/options.py", {"get_args", "validate_mraf_options"},
        dict(argparse=argparse, math=math, load_config_mapping=load_config_mapping,
             DEFAULT_DEGRADE_MODES=modes, ALLOWED_DEGRADE_MODES=modes + ("rgb_blur", "ir_blur"),
             MRAF_PRECISION_FUSION_ERROR="incompatible fusion"))


def fake_recipe():
    return dict(dataset_name="RGBT-PEDES", name="base_clean", output_dir="original_runs",
                root_dir="data", dev_split_file="splits/dev.json", train_seed=1,
                deg_rng_seed=None, num_epoch=60, lr=1e-5, batch_size=64,
                loss_names="sdm+mlm+id", use_mum=False, mlm_loss_weight=1.0,
                img_size=[384, 128], val_dataset="val", untouched={"x": [1, 2]})


class MLMCountTests(unittest.TestCase):
    def test_default_and_explicit_false_preserve_original_order_bitwise(self):
        original = subprocess.check_output(["git", "show", "82221d47:IRRA-main/processor/processor.py"],
                                           cwd=CODE.parent).decode("utf-8")
        before, after = loss_builder(original), loss_builder()
        # Adversarial magnitudes make a change in summation order observable.
        ret = {"sdm_loss": 1e16, "loss_sdm_prob": 1e16, "mlm_loss": -1e16,
               "loss_mlm": 3.0, "id_loss": 2.0, "ordinary_metric": 99.0}
        snapshot = ret.copy()
        for args in (SimpleNamespace(use_mum=False),
                     SimpleNamespace(use_mum=False, mlm_single_count=False)):
            expected, actual = before(args, ret), after(args, ret)
            self.assertEqual(actual.hex(), expected.hex())
        self.assertEqual(ret, snapshot)

    def test_true_omits_only_unweighted_alias_and_keeps_weighted_value(self):
        ret = {"sdm_loss": 4.0, "loss_sdm_prob": 4.0, "mlm_loss": 6.0,
               "loss_mlm": 3.0, "id_loss": 2.0}
        self.assertEqual(loss_builder()(SimpleNamespace(mlm_single_count=False), ret), 15.0)
        self.assertEqual(loss_builder()(SimpleNamespace(mlm_single_count=True), ret), 12.0)
        self.assertEqual(set(ret), {"sdm_loss", "loss_sdm_prob", "mlm_loss", "loss_mlm", "id_loss"})

    def test_single_existing_alias_is_never_dropped(self):
        for key in ("mlm_loss", "loss_mlm"):
            with self.subTest(key=key):
                self.assertEqual(loss_builder()(SimpleNamespace(mlm_single_count=True), {key: 3.0}), 3.0)

    def test_probabilistic_branch_is_unchanged(self):
        args = SimpleNamespace(use_mum=True, mlm_single_count=True, lambda_det=0.5, lambda_mlm=2.0)
        ret = dict(loss_sdm_prob=4.0, loss_sdm_det=2.0, loss_mlm=3.0, mlm_loss=3.0)
        self.assertEqual(loss_builder()(args, ret), 11.0)

    def test_default_parser_false_cli_and_yaml_opt_in(self):
        scope = parser_scope()
        with mock.patch.object(sys, "argv", ["train.py"]):
            self.assertIs(scope["get_args"]().mlm_single_count, False)
        with mock.patch.object(sys, "argv", ["train.py", "--mlm_single_count"]):
            self.assertIs(scope["get_args"]().mlm_single_count, True)
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / "config.yaml"
            p.write_text("mlm_single_count: true\n", encoding="utf-8")
            with mock.patch.object(sys, "argv", ["train.py", "--config_file", str(p)]):
                self.assertIs(scope["get_args"]().mlm_single_count, True)

    def test_run_record_explicitly_records_flag(self):
        runrecord = CODE / "tools/p4/run_record.py"
        fixed = mock.Mock()
        fixed.check_output.side_effect = ["commit\n", "", "commit\n", ""]
        scope = extract_functions(runrecord, {"RunRecord"},
            dict(Path=Path, subprocess=fixed, __file__=str(runrecord), json=json,
                 _now=lambda: "fixed", resolve_deg_rng_seed=lambda seed, explicit: seed * 1000 + 7919))
        with tempfile.TemporaryDirectory() as temporary:
            for flag in (False, True):
                args = SimpleNamespace(output_dir=str(Path(temporary) / str(flag)), mlm_single_count=flag)
                record = scope["RunRecord"](args)
                self.assertIs(record.data["mlm_single_count"], flag)
                self.assertIs(json.loads(record.path.read_text(encoding="utf-8"))["mlm_single_count"], flag)


class RuntimeRecipeTests(unittest.TestCase):
    def test_rgbt_only_authorized_keys_change_and_input_is_untouched(self):
        base = fake_recipe(); snapshot = copy.deepcopy(base)
        derived = derive_config(base, "RGBT-PEDES", 102, "new_runs")
        self.assertEqual(base, snapshot)
        self.assertEqual(set(config_differences(base, derived)), set(COMMON_ALLOWED))
        self.assertEqual(derived["num_epoch"], 60)
        self.assertEqual(derived["name"], "base_mlm1_s102")
        self.assertEqual(derived["deg_rng_seed"], 109919)
        derived["untouched"]["x"].append(3)
        self.assertEqual(base["untouched"], {"x": [1, 2]})

    def test_rgbnt_only_adds_the_three_l1_dataset_overrides(self):
        base = fake_recipe()
        derived = derive_config(base, "RGBNT201-PEDES", 101, "rgbnt_runs", "rgbnt_data", "rgbnt_dev")
        self.assertEqual(set(config_differences(base, derived)), set(COMMON_ALLOWED | RGBNT_ALLOWED))
        self.assertEqual(derived["dataset_name"], "RGBNT201-PEDES")
        self.assertEqual(derived["root_dir"], "rgbnt_data")
        self.assertEqual(derived["dev_split_file"], "rgbnt_dev")

    def test_smoke_is_separate_and_is_the_only_epoch_override(self):
        base = fake_recipe()
        formal = derive_config(base, "RGBT-PEDES", 101, "formal")
        smoke = derive_config(base, "RGBT-PEDES", 101, "smoke", smoke=True)
        self.assertEqual(formal["num_epoch"], 60)
        self.assertEqual(smoke["num_epoch"], 2)
        self.assertTrue(smoke["name"].endswith("_smoke"))
        self.assertNotEqual(formal["output_dir"], smoke["output_dir"])
        with self.assertRaisesRegex(ValueError, "whitelist"):
            validate_derivation(base, smoke, "RGBT-PEDES", smoke=False)

    def test_forbidden_changes_paths_and_invalid_arguments_fail(self):
        base = fake_recipe()
        derived = derive_config(base, "RGBT-PEDES", 101, "new")
        for key, value in (("lr", 0.2), ("batch_size", 2), ("num_epoch", 2), ("loss_names", "sdm")):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "whitelist"):
                    validate_derivation(base, dict(derived, **{key: value}), "RGBT-PEDES")
        for key in ("root_dir", "dev_split_file"):
            with self.assertRaisesRegex(ValueError, "inherited"):
                derive_config(base, "RGBT-PEDES", 101, "new", **{key: "different"})
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError):derive_config(base, "RGBT-PEDES", bad, "new")
        with self.assertRaises(ValueError):derive_config(base, "other", 101, "new")
        with self.assertRaises(ValueError):derive_config(base, "RGBT-PEDES", 101, "")

    def test_parser_normalized_health_base_covers_every_saved_namespace_field(self):
        import yaml
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "base.yaml"
            derived = root / "derived.yaml"; expected = root / "expected.yaml"
            source.write_text(yaml.safe_dump(fake_recipe()), encoding="utf-8")
            write_derived_config(source, derived, "RGBT-PEDES", 101, "runs")
            before = derived.read_bytes(); original_argv = sys.argv
            record = runtime_health_base(derived, expected, parse_args=parser_scope()["get_args"])
            self.assertIs(sys.argv, original_argv)
            self.assertEqual(derived.read_bytes(), before)
            expected_map = load_config_mapping(expected)
            self.assertEqual(expected_map["config_file"], str(derived))
            self.assertIs(expected_map["distributed"], False)
            self.assertEqual(expected_map["train_seed"], 101)
            self.assertIn("mraf_rel_isolate_input", expected_map)
            self.assertIs(expected_map["mlm_single_count"], True)
            self.assertFalse(record["training_ran"])
            self.assertEqual(record["fields"], len(expected_map))
            # Model the namespace saved by train.py after parsing and its
            # documented seed/distributed/output-directory metadata assignments.
            with mock.patch.object(sys, "argv", ["train.py", "--config_file", str(derived)]):
                actual = vars(parser_scope()["get_args"]()).copy()
            actual.update(train_seed=101, distributed=False, output_dir="runs/RGBT-PEDES/fixed_base_mlm1_s101")
            actual_path = root / "saved.yaml"
            actual_path.write_text(yaml.dump(actual), encoding="utf-8")
            scope = extract_functions(CODE / "tools/p4/r18/workflow.py", {"config_compare"},
                dict(ALLOWED_CONFIG={"name", "output_dir", "train_seed", "deg_rng_seed", "dataset_name", "root_dir", "dev_split_file"}))
            comparison = scope["config_compare"](expected, actual_path)
            self.assertTrue(comparison["ok"], comparison)
            self.assertEqual(set(comparison["differences"]), {"output_dir"})
            self.assertEqual(set(expected_map), set(actual))
            with self.assertRaises(FileExistsError):
                runtime_health_base(derived, expected, parse_args=parser_scope()["get_args"])

    def test_historical_tuples_remain_tuples_after_runtime_yaml_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); base = root / "base.yaml"; output = root / "derived.yaml"
            base.write_text("dataset_name: RGBT-PEDES\nnum_epoch: 60\n"
                            "img_size: !!python/tuple [384, 128]\n", encoding="utf-8")
            record = write_derived_config(base, output, "RGBT-PEDES", 101, "new")
            self.assertEqual(load_config_mapping(output)["img_size"], (384, 128))
            self.assertNotIn("img_size", record["validation"]["differences"])

    def test_override_is_path_free_and_yaml_copy_has_sha_and_differences(self):
        override = CODE / "configs/p4/r19_mlm1_override.yaml"
        self.assertNotIn("/root/", override.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); base = root / "base.yaml"; output = root / "derived.yaml"
            import yaml
            base.write_text(yaml.safe_dump(fake_recipe()), encoding="utf-8")
            record = write_derived_config(base, output, "RGBT-PEDES", 101, "new")
            self.assertEqual(record["base_sha256"], hashlib.sha256(base.read_bytes()).hexdigest())
            self.assertEqual(record["generated_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(set(record["validation"]["differences"]), set(COMMON_ALLOWED))
            self.assertEqual(record["parameters"]["smoke"], False)
            self.assertIs(load_config_mapping(output)["mlm_single_count"], True)
            self.assertNotIn(b"\r\n", output.read_bytes())
            self.assertTrue(output.with_suffix(".derivation.json").is_file())
            with self.assertRaises(FileExistsError):
                write_derived_config(base, output, "RGBT-PEDES", 101, "new")


if __name__ == "__main__":
    unittest.main(verbosity=2)
