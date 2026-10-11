"""Validate A provenance requirements and the three single-switch variants."""
import copy
from pathlib import Path
import unittest

from tools.p4.prepare_configs import derive_configs
from utils.config import load_config_mapping


def a_config():
    # Synthetic explicit config, never a claim about historical run defaults.
    return dict(sampler="random", num_instance=4, lrscheduler="cosine",
                warmup_epochs=5, img_size=[384, 128], temperature=0.02,
                img_aug=True, MLM=True, loss_names="sdm+mlm+id", batch_size=64,
                num_epoch=60, lr=1e-5, optimizer="Adam", dataset_name="RGBT-PEDES",
                use_rgbt_add_fusion=True, use_mum=False, mlm_loss_weight=1.0,
                mdrc_enable=False, mraf_enabled=False, mraf_rel_lambda=0.0,
                degrade_aug_mode="none")


class ConfigTests(unittest.TestCase):
    def test_real_historical_devA_passes_and_preserves_uncertainty_flags(self):
        fixture = Path(__file__).with_name("fixtures") / "devA_hist.yaml"
        original = load_config_mapping(fixture)
        snapshot = copy.deepcopy(original)
        for flag in ("use_batch_uncertainty", "use_id_uncertainty"):
            self.assertIs(original[flag], True)
        variants = derive_configs(original)
        self.assertEqual(original, snapshot)
        reference = variants["base_clean"]
        for name, loss in (("base_clean", "none"), ("c0_sdm", "sdm"), ("c0_ce", "ce")):
            config = variants[name]
            self.assertEqual(config["deg_retrieval_loss"], loss)
            self.assertEqual(set(config), set(reference))
            differences = {key for key in reference if config[key] != reference[key]}
            self.assertEqual(differences, set() if name == "base_clean" else {"deg_retrieval_loss"})
            for flag in ("use_batch_uncertainty", "use_id_uncertainty"):
                self.assertIs(config[flag], original[flag])
            self.assertEqual(config["warmup_epochs"], original["warmup_epochs"])
            self.assertIsNone(config["use_precision_fusion"])

    def test_uncertainty_whitelist_preserves_true_and_false_source_values(self):
        for batch_flag in (False, True):
            for id_flag in (False, True):
                with self.subTest(batch=batch_flag, identity=id_flag):
                    original = dict(a_config(), use_batch_uncertainty=batch_flag,
                                    use_id_uncertainty=id_flag)
                    for config in derive_configs(original).values():
                        self.assertIs(config["use_batch_uncertainty"], batch_flag)
                        self.assertIs(config["use_id_uncertainty"], id_flag)

    def test_three_variants_only_retrieval_loss_differs(self):
        original = a_config()
        variants = derive_configs(original)
        self.assertNotIn("dev_split_file", original)
        expected = {"base_clean": "none", "c0_sdm": "sdm", "c0_ce": "ce"}
        common = None
        for name, config in variants.items():
            self.assertEqual(config["deg_retrieval_loss"], expected[name])
            remainder = {k: v for k, v in config.items() if k != "deg_retrieval_loss"}
            if common is None:
                common = remainder
            self.assertEqual(remainder, common)
            self.assertEqual(config["mlm_loss_weight"], original["mlm_loss_weight"])
            self.assertEqual(config["warmup_epochs"], original["warmup_epochs"])
            self.assertEqual(config["dev_split_file"], "splits/dev_190.json")

    def test_historical_missing_fusion_or_scheduler_is_rejected(self):
        for key in ("sampler", "lrscheduler", "warmup_epochs", "use_rgbt_add_fusion"):
            with self.subTest(key=key):
                config = a_config()
                del config[key]
                with self.assertRaisesRegex(ValueError, key):
                    derive_configs(config)

    def test_auxiliary_or_identity_source_is_rejected(self):
        for flag in ("use_image_local_filter", "use_attr_aux", "use_synth_caption_bank",
                     "use_hard_negative_loss", "use_mum", "use_prob_sampling",
                     "use_future_auxiliary", "mdrc_enable", "mraf_enabled"):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, flag):
                    derive_configs(dict(a_config(), **{flag: True}))
        with self.assertRaisesRegex(ValueError, "identity"):
            derive_configs(dict(a_config(), sampler="identity"))

    def test_nonfinite_weight_or_invalid_probability_is_rejected(self):
        for weight in (-1.0, float("nan"), float("inf")):
            with self.subTest(weight=weight):
                with self.assertRaises(ValueError):
                    derive_configs(a_config(), deg_lambda=weight)
        for probability in (-0.1, 1.1, float("nan")):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError):
                    derive_configs(a_config(), deg_branch_prob=probability)


if __name__ == "__main__":
    unittest.main()
