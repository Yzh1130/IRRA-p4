#!/usr/bin/env python3
"""Prove exact degradation parity between IRRA and the DCAlign kit."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch


COPIED_MODE_STRENGTHS = (
    ("none", 1.0),
    ("rgb_brightness", 0.5),
    ("ir_brightness", 0.5),
    ("rgb_lowres", 0.5),
    ("ir_lowres", 0.5),
    ("rgb_jpeg", 70.0),
    ("ir_jpeg", 70.0),
)
CORE_MODES = (
    "none",
    "rgb_missing",
    "ir_missing",
    "rgb_noise",
    "ir_noise",
    "both_noise",
    "rgb_blur",
    "ir_blur",
    "both_blur",
)
NOISE_MODES = ("rgb_noise", "ir_noise", "both_noise")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_pair_equal(
    label: str,
    expected: tuple[torch.Tensor, torch.Tensor | None],
    actual: tuple[torch.Tensor, torch.Tensor | None],
) -> None:
    expected_rgb, expected_thermal = expected
    actual_rgb, actual_thermal = actual
    rgb_equal = torch.equal(expected_rgb, actual_rgb)
    thermal_equal = (
        expected_thermal is None
        and actual_thermal is None
        or torch.is_tensor(expected_thermal)
        and torch.is_tensor(actual_thermal)
        and torch.equal(expected_thermal, actual_thermal)
    )
    exact = bool(rgb_equal and thermal_equal)
    print(
        f"[DEGRADATION_PARITY] mode={label} "
        f"rgb_equal={str(rgb_equal).lower()} "
        f"thermal_equal={str(thermal_equal).lower()} "
        f"exact={str(exact).lower()}"
    )
    if not exact:
        rgb_delta = float((expected_rgb - actual_rgb).abs().max().item())
        if torch.is_tensor(expected_thermal) and torch.is_tensor(actual_thermal):
            thermal_delta = float(
                (expected_thermal - actual_thermal).abs().max().item()
            )
        else:
            thermal_delta = float("inf")
        raise AssertionError(
            f"degradation parity failed for {label}: "
            f"rgb_max_abs={rgb_delta}, thermal_max_abs={thermal_delta}"
        )


def _make_inputs(
    source_module: ModuleType, device: torch.device, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rgb_pixels = torch.rand((2, 3, 24, 16), generator=generator)
    thermal_pixels = torch.rand((2, 3, 24, 16), generator=generator)
    rgb = source_module.clip_normalize(rgb_pixels).to(device)
    thermal = source_module.clip_normalize(thermal_pixels).to(device)
    return rgb, thermal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assert byte-level source-copy provenance and exact output parity for "
            "every evaluation degradation used by the DCAlign suite."
        )
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    project_root = Path(__file__).resolve().parents[2]
    eval_script = _load_module(
        "_dcalign_eval_script_for_parity",
        Path(__file__).resolve().with_name("eval_dcalign_degraded.py"),
    )
    provenance = eval_script.assert_degradation_copy_provenance(project_root)
    print(
        "[DEGRADATION_SHA256] "
        + json.dumps(provenance, ensure_ascii=False, sort_keys=True)
    )

    sys.path.insert(0, str(project_root))
    try:
        source_module = importlib.import_module("utils.eval_degradation")
        metrics_module = importlib.import_module("utils.metrics")
    finally:
        if sys.path[0] == str(project_root):
            sys.path.pop(0)
    copied_module = _load_module(
        "_dcalign_eval_degradation_copy_for_parity",
        Path(__file__).resolve().with_name("eval_degradation.py"),
    )

    rgb, thermal = _make_inputs(source_module, device, args.seed)
    copied_results = 0
    for mode, strength in COPIED_MODE_STRENGTHS:
        expected = source_module.apply_new_eval_degradation(
            mode=mode,
            rgb_img=rgb.clone(),
            thermal_img=thermal.clone(),
            strength=strength,
        )
        actual = copied_module.apply_new_eval_degradation(
            mode=mode,
            rgb_img=rgb.clone(),
            thermal_img=thermal.clone(),
            strength=strength,
        )
        _assert_pair_equal(f"copy:{mode}", expected, actual)
        copied_results += 1

    core_results = 0
    for mode in CORE_MODES:
        model = SimpleNamespace(
            args=SimpleNamespace(
                eval_degrade_mode=mode,
                eval_degrade_strength=0.05,
                eval_degrade_blur_kernel=7,
                eval_degrade_seed=args.seed,
            )
        )
        reference = metrics_module.Evaluator(None, None)
        candidate = eval_script.EvalDegradationApplier(copied_module)
        expected = reference._apply_eval_degradation(
            model, rgb.clone(), thermal.clone()
        )
        actual = candidate.apply(
            mode=mode,
            rgb_img=rgb.clone(),
            thermal_img=thermal.clone(),
            strength=0.05,
            blur_kernel=7,
            seed=args.seed,
        )
        _assert_pair_equal(f"core:{mode}", expected, actual)
        core_results += 1

    # A single-batch comparison does not prove generator lifetime. Exercise two
    # consecutive batches with one Evaluator/applier per noise mode.
    rng_results = 0
    rgb_second, thermal_second = _make_inputs(source_module, device, args.seed + 1)
    for mode in NOISE_MODES:
        model = SimpleNamespace(
            args=SimpleNamespace(
                eval_degrade_mode=mode,
                eval_degrade_strength=0.05,
                eval_degrade_blur_kernel=7,
                eval_degrade_seed=args.seed,
            )
        )
        reference = metrics_module.Evaluator(None, None)
        candidate = eval_script.EvalDegradationApplier(copied_module)
        reference._apply_eval_degradation(model, rgb.clone(), thermal.clone())
        candidate.apply(
            mode,
            rgb.clone(),
            thermal.clone(),
            strength=0.05,
            blur_kernel=7,
            seed=args.seed,
        )
        expected_second = reference._apply_eval_degradation(
            model, rgb_second.clone(), thermal_second.clone()
        )
        actual_second = candidate.apply(
            mode,
            rgb_second.clone(),
            thermal_second.clone(),
            strength=0.05,
            blur_kernel=7,
            seed=args.seed,
        )
        _assert_pair_equal(f"rng_stream:{mode}:batch2", expected_second, actual_second)
        rng_results += 1

    print(
        "[DEGRADATION_PARITY_SUMMARY] "
        f"copy_cases={copied_results} core_cases={core_results} "
        f"rng_stream_cases={rng_results} total={copied_results + core_results + rng_results} "
        "status=PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
