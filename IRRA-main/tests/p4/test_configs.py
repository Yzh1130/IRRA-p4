"""Validate A provenance requirements and the three single-switch variants."""
import unittest

from tools.p4.prepare_configs import derive_configs


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
                     "use_hard_negative_loss", "use_mum", "mdrc_enable", "mraf_enabled"):
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
