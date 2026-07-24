#!/usr/bin/env python3
"""Analyze MRAF gate weights under standard RGB-T degradation scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_dataloader
from model import build_model
from utils.checkpoint import Checkpointer, strip_prefix_if_present
from utils.iotools import load_train_configs
from utils.metrics import Evaluator


SCENARIOS: Tuple[Tuple[str, str, float, int], ...] = (
    ("clean", "none", 1.0, 7),
    ("rgb_missing", "rgb_missing", 1.0, 7),
    ("ir_missing", "ir_missing", 1.0, 7),
    ("rgb_noise_0.05", "rgb_noise", 0.05, 7),
    ("rgb_noise_0.10", "rgb_noise", 0.10, 7),
    ("ir_noise_0.05", "ir_noise", 0.05, 7),
    ("ir_noise_0.10", "ir_noise", 0.10, 7),
    ("rgb_blur_k7", "rgb_blur", 1.0, 7),
    ("ir_blur_k7", "ir_blur", 1.0, 7),
    ("both_noise_0.05", "both_noise", 0.05, 7),
    ("both_blur_k7", "both_blur", 1.0, 7),
)

CSV_FIELDS = (
    "scenario",
    "num_samples",
    "alpha_rgb_mean",
    "alpha_rgb_std",
    "alpha_rgb_p10",
    "alpha_rgb_p50",
    "alpha_rgb_p90",
    "alpha_ir_mean",
    "alpha_ir_std",
    "alpha_ir_p10",
    "alpha_ir_p50",
    "alpha_ir_p90",
    "max_weight_sum_error",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure MRAF RGB/IR gate weights under fixed degradations."
    )
    parser.add_argument("--config-file", required=True, help="training configs.yaml")
    parser.add_argument("--checkpoint", required=True, help="MRAF checkpoint")
    parser.add_argument(
        "--output-prefix",
        default="mraf_weight_statistics",
        help="output path prefix; .csv and .json are appended",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device, e.g. cuda, cuda:0, or cpu",
    )
    parser.add_argument(
        "--degrade-seed",
        type=int,
        default=2026,
        help="fixed evaluation degradation seed",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="optional debug cap per scenario; <=0 evaluates the full gallery",
    )
    return parser


def _checkpoint_has_mraf(checkpoint: Dict) -> bool:
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        return False
    state_dict = strip_prefix_if_present(state_dict, "module.")
    return any(key.startswith("mraf_fusion.") for key in state_dict)


def _percentile(values: torch.Tensor, quantile: float) -> float:
    return float(torch.quantile(values, float(quantile)).item())


def summarize_alpha(
    scenario: str,
    alpha_rgb: torch.Tensor,
    alpha_ir: torch.Tensor,
) -> Dict[str, float]:
    if alpha_rgb.numel() == 0 or alpha_ir.numel() == 0:
        raise ValueError(f"No MRAF weights collected for scenario {scenario}.")
    alpha_rgb = alpha_rgb.detach().cpu().float().reshape(-1)
    alpha_ir = alpha_ir.detach().cpu().float().reshape(-1)
    if alpha_rgb.shape != alpha_ir.shape:
        raise ValueError("RGB and IR gate statistics must have matching shapes.")
    return {
        "scenario": scenario,
        "num_samples": int(alpha_rgb.numel()),
        "alpha_rgb_mean": float(alpha_rgb.mean().item()),
        "alpha_rgb_std": float(alpha_rgb.std(unbiased=False).item()),
        "alpha_rgb_p10": _percentile(alpha_rgb, 0.10),
        "alpha_rgb_p50": _percentile(alpha_rgb, 0.50),
        "alpha_rgb_p90": _percentile(alpha_rgb, 0.90),
        "alpha_ir_mean": float(alpha_ir.mean().item()),
        "alpha_ir_std": float(alpha_ir.std(unbiased=False).item()),
        "alpha_ir_p10": _percentile(alpha_ir, 0.10),
        "alpha_ir_p50": _percentile(alpha_ir, 0.50),
        "alpha_ir_p90": _percentile(alpha_ir, 0.90),
        "max_weight_sum_error": float(
            (alpha_rgb + alpha_ir - 1.0).abs().max().item()
        ),
    }


def _collect_scenario(
    model,
    img_loader: Iterable,
    txt_loader: Iterable,
    scenario_name: str,
    degrade_mode: str,
    strength: float,
    blur_kernel: int,
    device: torch.device,
    degrade_seed: int,
    max_batches: int,
) -> Dict[str, float]:
    model.args.eval_degrade_mode = degrade_mode
    model.args.eval_degrade_strength = float(strength)
    model.args.eval_degrade_blur_kernel = int(blur_kernel)
    model.args.eval_degrade_seed = int(degrade_seed)
    evaluator = Evaluator(img_loader, txt_loader, source="mraf_weight_analysis")
    rgb_weights: List[torch.Tensor] = []
    ir_weights: List[torch.Tensor] = []

    for batch_index, data in enumerate(img_loader):
        if len(data) != 3:
            raise ValueError(
                "MRAF weight analysis requires RGB-T gallery batches "
                "(pid, rgb_image, thermal_image)."
            )
        _, rgb_image, thermal_image = data
        rgb_image = rgb_image.to(device)
        thermal_image = thermal_image.to(device)
        rgb_image, thermal_image = evaluator._apply_eval_degradation(
            model,
            rgb_image,
            thermal_image,
        )
        with torch.no_grad():
            outputs = model.encode_rgbt_eval(
                rgb_image,
                thermal_image,
                return_local_tokens=True,
            )
        if "mraf_alpha_rgb" not in outputs or "mraf_alpha_ir" not in outputs:
            raise RuntimeError(
                "The model did not return MRAF gate weights during evaluation."
            )
        rgb_weights.append(outputs["mraf_alpha_rgb"].detach().cpu())
        ir_weights.append(outputs["mraf_alpha_ir"].detach().cpu())
        if int(max_batches) > 0 and batch_index + 1 >= int(max_batches):
            break

    return summarize_alpha(
        scenario_name,
        torch.cat(rgb_weights, dim=0),
        torch.cat(ir_weights, dim=0),
    )


def write_outputs(
    rows: List[Dict[str, float]],
    output_prefix: Path,
    metadata: Dict,
) -> Tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": metadata, "scenarios": rows},
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    return csv_path, json_path


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config_file).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_args = load_train_configs(str(config_path))
    if not bool(getattr(model_args, "mraf_enabled", False)):
        raise ValueError("The supplied config does not enable MRAF.")
    model_args.training = False
    model_args.distributed = False

    raw_checkpoint = torch.load(
        str(checkpoint_path),
        map_location=torch.device("cpu"),
    )
    if not _checkpoint_has_mraf(raw_checkpoint):
        raise ValueError(
            "The supplied checkpoint does not contain MRAF scorer parameters."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    img_loader, txt_loader, num_classes = build_dataloader(model_args)
    model = build_model(model_args, num_classes=num_classes)
    Checkpointer(model).load(str(checkpoint_path))
    model.to(device).eval()

    rows = [
        _collect_scenario(
            model=model,
            img_loader=img_loader,
            txt_loader=txt_loader,
            scenario_name=scenario_name,
            degrade_mode=degrade_mode,
            strength=strength,
            blur_kernel=blur_kernel,
            device=device,
            degrade_seed=args.degrade_seed,
            max_batches=args.max_batches,
        )
        for scenario_name, degrade_mode, strength, blur_kernel in SCENARIOS
    ]
    csv_path, json_path = write_outputs(
        rows,
        Path(args.output_prefix).resolve(),
        {
            "config_file": str(config_path),
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "degrade_seed": int(args.degrade_seed),
            "max_batches": int(args.max_batches),
        },
    )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
