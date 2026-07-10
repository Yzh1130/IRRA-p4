import unittest
from pathlib import Path

import torch

from utils.eval_degradation import (
    NEW_EVAL_DEGRADATION_MODES,
    apply_brightness_degradation,
    apply_jpeg_degradation,
    apply_lowres_degradation,
    apply_new_eval_degradation,
    clip_denormalize,
    clip_normalize,
)


class EvaluationDegradationTest(unittest.TestCase):
    def setUp(self):
        pixel_count = 2 * 3 * 8 * 6
        rgb_pixels = torch.linspace(0.05, 0.95, pixel_count).reshape(2, 3, 8, 6)
        thermal_pixels = torch.flip(rgb_pixels, dims=(-1,))
        self.rgb = clip_normalize(rgb_pixels)
        self.thermal = clip_normalize(thermal_pixels)

    def test_clip_normalize_and_denormalize_are_approximately_inverse(self):
        pixels = clip_denormalize(self.rgb)
        restored = clip_normalize(pixels)
        self.assertTrue(torch.allclose(restored, self.rgb, atol=1e-6, rtol=1e-6))

    def test_brightness_preserves_shape(self):
        output = apply_brightness_degradation(self.rgb, 0.3, mode="rgb_brightness")
        self.assertEqual(output.shape, self.rgb.shape)

    def test_brightness_preserves_device(self):
        output = apply_brightness_degradation(self.rgb, 0.3, mode="rgb_brightness")
        self.assertEqual(output.device, self.rgb.device)

    def test_brightness_preserves_compatible_dtype(self):
        input_tensor = self.rgb.to(torch.float64)
        output = apply_brightness_degradation(
            input_tensor, 0.3, mode="rgb_brightness"
        )
        self.assertEqual(output.dtype, input_tensor.dtype)

    def test_rgb_brightness_does_not_modify_ir(self):
        _, thermal_output = apply_new_eval_degradation(
            "rgb_brightness", self.rgb, self.thermal, 0.3
        )
        self.assertTrue(torch.equal(thermal_output, self.thermal))

    def test_ir_brightness_does_not_modify_rgb(self):
        rgb_output, _ = apply_new_eval_degradation(
            "ir_brightness", self.rgb, self.thermal, 0.3
        )
        self.assertTrue(torch.equal(rgb_output, self.rgb))

    def test_brightness_reduces_mean_pixel_value(self):
        output = apply_brightness_degradation(self.rgb, 0.3, mode="rgb_brightness")
        input_mean = clip_denormalize(self.rgb).clamp(0.0, 1.0).mean()
        output_mean = clip_denormalize(output).clamp(0.0, 1.0).mean()
        self.assertLess(float(output_mean), float(input_mean))

    def test_brightness_factor_one_is_approximately_identity(self):
        output = apply_brightness_degradation(self.rgb, 1.0, mode="rgb_brightness")
        self.assertTrue(torch.allclose(output, self.rgb, atol=1e-6, rtol=1e-6))

    def test_brightness_rejects_invalid_factor(self):
        for factor in (0.0, -0.1, 1.1):
            with self.subTest(factor=factor):
                with self.assertRaisesRegex(ValueError, "rgb_brightness"):
                    apply_brightness_degradation(
                        self.rgb, factor, mode="rgb_brightness"
                    )

    def test_lowres_preserves_shape(self):
        output = apply_lowres_degradation(self.rgb, 0.25, mode="rgb_lowres")
        self.assertEqual(output.shape, self.rgb.shape)

    def test_lowres_scale_one_is_approximately_identity(self):
        output = apply_lowres_degradation(self.rgb, 1.0, mode="rgb_lowres")
        self.assertTrue(torch.allclose(output, self.rgb, atol=1e-6, rtol=1e-6))

    def test_lowres_handles_small_height_and_width(self):
        pixels = torch.tensor(
            [[[[0.2, 0.8]], [[0.4, 0.6]], [[0.1, 0.9]]]], dtype=torch.float32
        )
        input_tensor = clip_normalize(pixels)
        output = apply_lowres_degradation(input_tensor, 0.01, mode="rgb_lowres")
        self.assertEqual(output.shape, input_tensor.shape)

    def test_rgb_lowres_does_not_modify_ir(self):
        _, thermal_output = apply_new_eval_degradation(
            "rgb_lowres", self.rgb, self.thermal, 0.25
        )
        self.assertTrue(torch.equal(thermal_output, self.thermal))

    def test_ir_lowres_does_not_modify_rgb(self):
        rgb_output, _ = apply_new_eval_degradation(
            "ir_lowres", self.rgb, self.thermal, 0.25
        )
        self.assertTrue(torch.equal(rgb_output, self.rgb))

    def test_lowres_rejects_invalid_scale(self):
        for scale in (0.0, -0.1, 1.1):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "rgb_lowres"):
                    apply_lowres_degradation(self.rgb, scale, mode="rgb_lowres")

    def test_jpeg_preserves_shape(self):
        output = apply_jpeg_degradation(self.rgb, 20, mode="rgb_jpeg")
        self.assertEqual(output.shape, self.rgb.shape)

    def test_jpeg_decoded_pixels_stay_in_valid_range(self):
        output = apply_jpeg_degradation(self.rgb, 20, mode="rgb_jpeg")
        pixels = clip_denormalize(output)
        self.assertGreaterEqual(float(pixels.min()), 0.0)
        self.assertLessEqual(float(pixels.max()), 1.0)

    def test_jpeg_quality_100_runs(self):
        output = apply_jpeg_degradation(self.rgb, 100, mode="rgb_jpeg")
        self.assertEqual(output.shape, self.rgb.shape)

    def test_jpeg_quality_1_runs(self):
        output = apply_jpeg_degradation(self.rgb, 1, mode="rgb_jpeg")
        self.assertEqual(output.shape, self.rgb.shape)

    def test_jpeg_integral_float_quality_runs(self):
        output = apply_jpeg_degradation(self.rgb, 20.0, mode="rgb_jpeg")
        self.assertEqual(output.shape, self.rgb.shape)

    def test_jpeg_rejects_non_integral_quality(self):
        with self.assertRaisesRegex(ValueError, r"rgb_jpeg.*20\.5"):
            apply_jpeg_degradation(self.rgb, 20.5, mode="rgb_jpeg")

    def test_rgb_jpeg_does_not_modify_ir(self):
        _, thermal_output = apply_new_eval_degradation(
            "rgb_jpeg", self.rgb, self.thermal, 20
        )
        self.assertTrue(torch.equal(thermal_output, self.thermal))

    def test_ir_jpeg_does_not_modify_rgb(self):
        rgb_output, _ = apply_new_eval_degradation(
            "ir_jpeg", self.rgb, self.thermal, 20
        )
        self.assertTrue(torch.equal(rgb_output, self.rgb))

    def test_jpeg_is_deterministic_in_same_environment(self):
        first = apply_jpeg_degradation(self.rgb, 20, mode="rgb_jpeg")
        second = apply_jpeg_degradation(self.rgb, 20, mode="rgb_jpeg")
        self.assertTrue(torch.equal(first, second))

    def test_all_new_modes_run_on_cpu(self):
        strengths = {
            "rgb_brightness": 0.3,
            "ir_brightness": 0.3,
            "rgb_lowres": 0.25,
            "ir_lowres": 0.25,
            "rgb_jpeg": 20,
            "ir_jpeg": 20,
        }
        for mode in sorted(NEW_EVAL_DEGRADATION_MODES):
            with self.subTest(mode=mode):
                rgb_output, thermal_output = apply_new_eval_degradation(
                    mode, self.rgb.cpu(), self.thermal.cpu(), strengths[mode]
                )
                self.assertEqual(rgb_output.device.type, "cpu")
                self.assertEqual(thermal_output.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_all_new_modes_run_on_cuda_when_available(self):
        strengths = {
            "rgb_brightness": 0.3,
            "ir_brightness": 0.3,
            "rgb_lowres": 0.25,
            "ir_lowres": 0.25,
            "rgb_jpeg": 20,
            "ir_jpeg": 20,
        }
        rgb = self.rgb.cuda()
        thermal = self.thermal.cuda()
        for mode in sorted(NEW_EVAL_DEGRADATION_MODES):
            with self.subTest(mode=mode):
                rgb_output, thermal_output = apply_new_eval_degradation(
                    mode, rgb, thermal, strengths[mode]
                )
                self.assertEqual(rgb_output.device.type, "cuda")
                self.assertEqual(thermal_output.device.type, "cuda")

    def test_none_mode_does_not_change_inputs(self):
        rgb_output, thermal_output = apply_new_eval_degradation(
            "none", self.rgb, self.thermal, 1.0
        )
        self.assertTrue(torch.equal(rgb_output, self.rgb))
        self.assertTrue(torch.equal(thermal_output, self.thermal))

    def test_all_new_modes_preserve_shape_and_do_not_modify_inputs_in_place(self):
        strengths = {
            "rgb_brightness": 0.3,
            "ir_brightness": 0.3,
            "rgb_lowres": 0.25,
            "ir_lowres": 0.25,
            "rgb_jpeg": 20,
            "ir_jpeg": 20,
        }
        for mode in sorted(NEW_EVAL_DEGRADATION_MODES):
            with self.subTest(mode=mode):
                rgb = self.rgb.clone()
                thermal = self.thermal.clone()
                rgb_before = rgb.clone()
                thermal_before = thermal.clone()
                rgb_output, thermal_output = apply_new_eval_degradation(
                    mode, rgb, thermal, strengths[mode]
                )
                self.assertEqual(rgb_output.shape, rgb.shape)
                self.assertEqual(thermal_output.shape, thermal.shape)
                self.assertTrue(torch.equal(rgb, rgb_before))
                self.assertTrue(torch.equal(thermal, thermal_before))
                if mode.startswith("rgb_"):
                    self.assertTrue(torch.equal(thermal_output, thermal))
                else:
                    self.assertTrue(torch.equal(rgb_output, rgb))

    def test_legacy_metrics_degradation_source_block_is_intact(self):
        metrics_path = Path(__file__).resolve().parents[1] / "utils" / "metrics.py"
        source = metrics_path.read_text(encoding="utf-8")
        expected_legacy_block = '''        if mode == "none":
            return img, thermal_img

        strength = float(getattr(model_args, "eval_degrade_strength", 1.0))
        blur_kernel = int(getattr(model_args, "eval_degrade_blur_kernel", 7))
        seed = int(getattr(model_args, "eval_degrade_seed", 2026))

        if mode == "rgb_missing":
            img = torch.zeros_like(img)
        elif mode == "ir_missing":
            if thermal_img is not None:
                thermal_img = torch.zeros_like(thermal_img)
        elif mode == "rgb_noise":
            img = self._add_eval_noise(img, strength, seed)
        elif mode == "ir_noise":
            thermal_img = self._add_eval_noise(thermal_img, strength, seed)
        elif mode == "both_noise":
            img = self._add_eval_noise(img, strength, seed)
            thermal_img = self._add_eval_noise(thermal_img, strength, seed)
        elif mode == "rgb_blur":
            img = self._blur_eval_image(img, blur_kernel)
        elif mode == "ir_blur":
            thermal_img = self._blur_eval_image(thermal_img, blur_kernel)
        elif mode == "both_blur":
            img = self._blur_eval_image(img, blur_kernel)
            thermal_img = self._blur_eval_image(thermal_img, blur_kernel)'''
        self.assertEqual(source.count(expected_legacy_block), 1)


if __name__ == "__main__":
    unittest.main()
