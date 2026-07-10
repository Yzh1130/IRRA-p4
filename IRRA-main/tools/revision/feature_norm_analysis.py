"""Read-only feature-norm diagnostics using the existing public eval interface."""

import argparse
import csv
import logging
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_dataloader  # noqa: E402
from model import build_model  # noqa: E402
from processor.processor import (  # noqa: E402
    apply_img_local_filter_stage,
    is_full_eval_state_checkpoint,
    load_full_eval_state_checkpoint,
    restore_eval_extra_state,
)
from tools.revision.score_io import (  # noqa: E402
    prepare_output_parents,
    require_outputs_absent,
    write_json,
    write_markdown,
)
from utils.checkpoint import Checkpointer  # noqa: E402
from utils.iotools import load_train_configs  # noqa: E402


REQUIRED_FEATURE_KEYS = (
    "rgb_global_feat",
    "thermal_global_feat",
    "global_feat",
)


def summarize_values(values):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("Cannot summarize an empty feature statistic.")
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def compute_feature_diagnostics(rgb_features, thermal_features, fused_features, eps=1e-12):
    if not all(torch.is_tensor(value) for value in (rgb_features, thermal_features, fused_features)):
        raise TypeError("RGB, thermal and fused features must be torch tensors.")
    if rgb_features.dim() != 2:
        raise ValueError(
            f"Feature tensors must have shape [N,C], got {tuple(rgb_features.shape)}."
        )
    if rgb_features.shape != thermal_features.shape or rgb_features.shape != fused_features.shape:
        raise ValueError(
            "RGB, thermal and fused feature shapes must match exactly: "
            f"{tuple(rgb_features.shape)}, {tuple(thermal_features.shape)}, "
            f"{tuple(fused_features.shape)}."
        )
    rgb = rgb_features.detach().cpu().float()
    thermal = thermal_features.detach().cpu().float()
    fused = fused_features.detach().cpu().float()
    rgb_norm = rgb.norm(p=2, dim=1)
    thermal_norm = thermal.norm(p=2, dim=1)
    ratio = rgb_norm / thermal_norm.clamp_min(float(eps))
    rgb_fused_cosine = F.cosine_similarity(rgb, fused, dim=1, eps=float(eps))
    thermal_fused_cosine = F.cosine_similarity(
        thermal, fused, dim=1, eps=float(eps)
    )
    per_sample = {
        "rgb_l2_norm": rgb_norm.tolist(),
        "thermal_l2_norm": thermal_norm.tolist(),
        "rgb_to_thermal_norm_ratio": ratio.tolist(),
        "rgb_to_fused_cosine": rgb_fused_cosine.tolist(),
        "thermal_to_fused_cosine": thermal_fused_cosine.tolist(),
    }
    summary = {key: summarize_values(values) for key, values in per_sample.items()}
    return per_sample, summary


def _torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_and_gallery(config_file, checkpoint_file, root_dir, split, batch_size, device):
    config_file = os.path.abspath(config_file)
    checkpoint_file = os.path.abspath(checkpoint_file)
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file does not exist: {config_file}")
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    args = load_train_configs(config_file)
    args.training = False
    args.distributed = False
    args.eval_dataset = str(split)
    if root_dir:
        args.root_dir = str(root_dir)
    if batch_size is not None:
        args.test_batch_size = int(batch_size)

    img_loader, _, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    if not hasattr(model, "encode_rgbt_eval"):
        raise AttributeError(
            "Current model does not expose encode_rgbt_eval; feature analysis cannot "
            "obtain pre-fusion RGB/thermal features without a model change."
        )
    raw_checkpoint = _torch_load_checkpoint(checkpoint_file)
    full_state = is_full_eval_state_checkpoint(raw_checkpoint)
    if full_state:
        checkpoint = load_full_eval_state_checkpoint(
            model, checkpoint_file, logger=None, checkpoint=raw_checkpoint
        )
    else:
        checkpoint = Checkpointer(model).load(f=checkpoint_file)
    epoch = int(checkpoint.get("epoch", 1)) if isinstance(checkpoint, dict) else 1
    apply_img_local_filter_stage(
        model, args, epoch, logger=None, prefix="FEATURE_NORM_ANALYSIS"
    )
    if full_state:
        restore_eval_extra_state(
            model, checkpoint.get("extra_eval_state", {}), logger=None
        )
    model.to(device)
    model.eval()
    return model, img_loader, args, epoch, full_state


def collect_features(model, img_loader, device):
    rgb_features = []
    thermal_features = []
    fused_features = []
    pids = []
    with torch.inference_mode():
        for data in img_loader:
            if len(data) != 3:
                raise ValueError(
                    "feature_norm_analysis requires RGBT batches (pid, RGB, thermal)."
                )
            pid, rgb_img, thermal_img = data
            outputs = model.encode_rgbt_eval(
                rgb_img.to(device),
                thermal_img.to(device),
                return_local_tokens=True,
            )
            if not isinstance(outputs, dict):
                raise TypeError(
                    "encode_rgbt_eval(..., return_local_tokens=True) must return a dict."
                )
            missing = [key for key in REQUIRED_FEATURE_KEYS if key not in outputs]
            if missing:
                raise KeyError(
                    "Current model eval interface is missing required feature keys: "
                    f"{missing}. No features were fabricated."
                )
            rgb_features.append(outputs["rgb_global_feat"].detach().cpu().float())
            thermal_features.append(
                outputs["thermal_global_feat"].detach().cpu().float()
            )
            fused_features.append(outputs["global_feat"].detach().cpu().float())
            pids.append(pid.detach().cpu().long().view(-1))
    return (
        torch.cat(rgb_features, dim=0),
        torch.cat(thermal_features, dim=0),
        torch.cat(fused_features, dim=0),
        torch.cat(pids, dim=0),
    )


def _write_csv(path, rows):
    fieldnames = [
        "image_index",
        "pid",
        "rgb_path",
        "thermal_path",
        "rgb_l2_norm",
        "thermal_l2_norm",
        "rgb_to_thermal_norm_ratio",
        "rgb_to_fused_cosine",
        "thermal_to_fused_cosine",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(result):
    lines = [
        "# RGBT feature norm analysis",
        "",
        "Definitions: `rgb_global_feat` and `thermal_global_feat` are the existing "
        "model's modality features immediately before final fused selection; "
        "`global_feat` is the existing final evaluation feature. No contribution "
        "percentage is defined or reported.",
        "",
        f"- Samples: {result['sample_count']}",
        f"- Checkpoint epoch: {result['checkpoint_epoch']}",
        f"- Full eval-state checkpoint: {str(result['full_eval_state_checkpoint']).lower()}",
        "",
        "| Statistic | Mean | Sample SD | Median | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in result["summary"].items():
        sample_std = "NA" if values["sample_std"] is None else f"{values['sample_std']:.6f}"
        lines.append(
            f"| {name} | {values['mean']:.6f} | {sample_std} | "
            f"{values['median']:.6f} | {values['min']:.6f} | {values['max']:.6f} |"
        )
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze RGB, thermal and final fused feature norms through the existing "
            "read-only model eval interface."
        )
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root-dir", default="")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-prefix", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    prefix = os.path.abspath(args.output_prefix)
    paths = {
        "csv": prefix + ".csv",
        "json": prefix + ".json",
        "markdown": prefix + ".md",
    }
    require_outputs_absent(paths.values())
    prepare_output_parents(paths.values())

    logging.basicConfig(level=logging.INFO)
    model, img_loader, model_args, epoch, full_state = load_model_and_gallery(
        args.config_file,
        args.checkpoint,
        args.root_dir,
        args.split,
        args.batch_size,
        device,
    )
    if model.training:
        raise RuntimeError("Model unexpectedly remained in training mode.")
    rgb, thermal, fused, pids = collect_features(model, img_loader, device)
    per_sample, summary = compute_feature_diagnostics(rgb, thermal, fused)
    dataset = img_loader.dataset
    rgb_paths = list(getattr(dataset, "img_paths_rgb", [""] * pids.numel()))
    thermal_paths = list(getattr(dataset, "img_paths_t", [""] * pids.numel()))
    rows = []
    for index in range(pids.numel()):
        rows.append(
            {
                "image_index": index,
                "pid": int(pids[index]),
                "rgb_path": rgb_paths[index],
                "thermal_path": thermal_paths[index],
                **{key: values[index] for key, values in per_sample.items()},
            }
        )
    result = {
        "sample_count": pids.numel(),
        "feature_dim": rgb.shape[1],
        "summary": summary,
        "config_file": os.path.abspath(args.config_file),
        "checkpoint_file": os.path.abspath(args.checkpoint),
        "checkpoint_epoch": epoch,
        "full_eval_state_checkpoint": full_state,
        "dataset_name": str(getattr(model_args, "dataset_name", "")),
        "split": args.split,
        "device": str(device),
        "model_interface": "encode_rgbt_eval(return_local_tokens=True)",
        "feature_keys": list(REQUIRED_FEATURE_KEYS),
        "no_rgb_contribution_percentage_reported": True,
    }
    _write_csv(paths["csv"], rows)
    write_json(paths["json"], result)
    write_markdown(paths["markdown"], _markdown(result))
    print(f"Saved feature norm analysis: {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
