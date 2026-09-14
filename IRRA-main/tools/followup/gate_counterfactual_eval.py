#!/usr/bin/env python3
"""Evaluate fixed-model MRAF gate counterfactuals from cached features.

This tool is inference-only.  It reuses the repository's test-time model,
degradation and ``Evaluator`` paths and never updates model parameters.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_dataloader  # noqa: E402
from datasets.bases import ImageDatasetRGBT  # noqa: E402
from datasets.build import build_transforms  # noqa: E402
from datasets.rgbtpedes import RGBTPEDES  # noqa: E402
from model import build_model  # noqa: E402
from processor.processor import (  # noqa: E402
    apply_img_local_filter_stage,
    is_full_eval_state_checkpoint,
    load_full_eval_state_checkpoint,
    restore_eval_extra_state,
)
from utils.checkpoint import Checkpointer  # noqa: E402
from utils.iotools import load_train_configs  # noqa: E402
from utils.metrics import Evaluator  # noqa: E402


DEFAULT_OUT = Path("/root/autodl-tmp/FOLLOWUP_B")
MAIN_METRICS = Path("/root/autodl-tmp/MAIN_GRID/metrics.tsv")
ALLOWED_GROUPS = frozenset({"D", "E", "F"})
ALLOWED_MODES = ("learned", "equal", "calib", "rgb_only", "ir_only")
METRIC_NAMES = ("R1", "R5", "R10", "mAP", "mINP")
REPRODUCTION_TOLERANCE = 0.01
CORE_SCENARIOS = (
    ("clean", "none", 1.0, 7),
    ("rgb_missing", "rgb_missing", 1.0, 7),
    ("ir_missing", "ir_missing", 1.0, 7),
    ("rgb_noise_005", "rgb_noise", 0.05, 7),
    ("rgb_noise_010", "rgb_noise", 0.10, 7),
    ("ir_noise_005", "ir_noise", 0.05, 7),
    ("ir_noise_010", "ir_noise", 0.10, 7),
    ("rgb_blur_k7", "rgb_blur", 1.0, 7),
    ("ir_blur_k7", "ir_blur", 1.0, 7),
    ("both_noise_005", "both_noise", 0.05, 7),
    ("both_blur_k7", "both_blur", 1.0, 7),
)
UNSUPPORTED_EVAL_FLAGS = (
    "use_qalr",
    "use_qalr_v2",
    "qalr_v3_candidate_diag",
    "use_qalr_v3_gate",
    "use_qalr_v3_two_stage_gate",
    "use_transport_rerank",
    "use_uspc_slot_rerank",
    "use_modality_text_prompt",
)


class CounterfactualError(RuntimeError):
    """Raised when an input violates the frozen follow-up protocol."""


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_modes(value):
    modes = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not modes:
        raise argparse.ArgumentTypeError("--modes must contain at least one mode")
    if len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError(f"--modes contains duplicates: {modes!r}")
    unknown = [mode for mode in modes if mode not in ALLOWED_MODES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unsupported counterfactual modes {unknown}; expected {list(ALLOWED_MODES)}"
        )
    if "learned" not in modes:
        raise argparse.ArgumentTypeError(
            "--modes must include learned for the mandatory reproduction check"
        )
    return modes


def parse_run_identity(config_args):
    name = str(getattr(config_args, "name", "")).strip()
    match = re.fullmatch(r"(?P<group>.+)_seed(?P<seed>\d+)", name)
    if match is None:
        raise CounterfactualError(
            f"configs.yaml name must match '<group>_seed<seed>', got {name!r}"
        )
    group = match.group("group")
    seed = int(match.group("seed"))
    if group not in ALLOWED_GROUPS:
        raise CounterfactualError(
            f"group must be one of {sorted(ALLOWED_GROUPS)}, got {group!r}"
        )
    if seed not in range(101, 111):
        raise CounterfactualError(f"seed must be in 101..110, got {seed}")
    return group, seed


def _core_model(model):
    return model.module if hasattr(model, "module") else model


def _require_supported_model(model, args):
    core = _core_model(model)
    if not bool(getattr(core, "mraf_enabled", False)):
        raise CounterfactualError("counterfactual evaluation requires mraf_enabled=True")
    if not hasattr(core, "mraf_fusion"):
        raise CounterfactualError("model does not expose mraf_fusion")
    if bool(getattr(core.mraf_fusion, "equalize_modality_norm", False)):
        raise CounterfactualError(
            "frozen D/E/F protocol requires mraf_equalize_modality_norm=False"
        )
    enabled = [name for name in UNSUPPORTED_EVAL_FLAGS if bool(getattr(args, name, False))]
    if enabled:
        raise CounterfactualError(
            "cached-feature counterfactual evaluation does not support enabled "
            f"rerank/MTP flags: {enabled}"
        )


def load_run_model(run_dir, device):
    """Reuse the same config/dataloader/model/checkpoint sequence as test.py."""
    run_dir = Path(run_dir).expanduser().resolve()
    config_path = run_dir / "configs.yaml"
    checkpoint_path = run_dir / "last.pth"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing configs.yaml: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing last.pth: {checkpoint_path}")

    config_sha256 = sha256_file(config_path)
    args = load_train_configs(str(config_path))
    if sha256_file(config_path) != config_sha256:
        raise CounterfactualError(
            f"configs.yaml changed while it was being loaded: {config_path}"
        )
    group, seed = parse_run_identity(args)
    if str(getattr(args, "dataset_name", "")) != "RGBT-PEDES":
        raise CounterfactualError(
            "follow-up B is frozen for dataset_name='RGBT-PEDES', got "
            f"{getattr(args, 'dataset_name', None)!r}"
        )
    args.training = False
    args.distributed = False
    args.eval_dataset = "test"
    args.config_file = str(config_path)
    args.test_ckpt_file = str(checkpoint_path)
    args.save_eval_scores = False
    args.eval_scores_file = ""
    args.eval_scores_save_raw = False

    image_loader, text_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    raw_checkpoint = _torch_load(checkpoint_path)
    full_state = is_full_eval_state_checkpoint(raw_checkpoint)
    if full_state:
        checkpoint = load_full_eval_state_checkpoint(
            model,
            str(checkpoint_path),
            logger=None,
            checkpoint=raw_checkpoint,
        )
    else:
        checkpoint = Checkpointer(model).load(f=str(checkpoint_path))
    epoch = int(checkpoint.get("epoch", 1)) if isinstance(checkpoint, dict) else 1
    apply_img_local_filter_stage(
        model,
        args,
        epoch,
        logger=None,
        prefix="FOLLOWUP_B",
    )
    if full_state:
        extra_state = (
            checkpoint.get("extra_eval_state", {})
            if isinstance(checkpoint, dict)
            else {}
        )
        restore_eval_extra_state(model, extra_state, logger=None)
    model.to(device)
    model.eval()
    _require_supported_model(model, args)
    return {
        "run_dir": run_dir,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "checkpoint_path": checkpoint_path,
        "args": args,
        "group": group,
        "seed": seed,
        "model": model,
        "image_loader": image_loader,
        "text_loader": text_loader,
        "checkpoint_epoch": epoch,
        "full_state_checkpoint": bool(full_state),
    }


def unique_train_image_pairs(train_samples):
    """Return one ordered record per image pair from caption-expanded samples."""
    by_image_id = {}
    pair_to_image_id = {}
    ordered = []
    for sample in train_samples:
        if not isinstance(sample, (tuple, list)) or len(sample) < 5:
            raise CounterfactualError(
                "RGBT train samples must be 5-tuples "
                "(pid, image_id, rgb_path, t_path, caption)"
            )
        pid, image_id, rgb_path, t_path = sample[:4]
        image_id = int(image_id)
        record = (int(pid), image_id, str(rgb_path), str(t_path))
        pair_key = (str(Path(rgb_path)), str(Path(t_path)))
        previous = by_image_id.get(image_id)
        if previous is not None and previous != record:
            raise CounterfactualError(
                f"image_id {image_id} maps to inconsistent RGB/T records"
            )
        previous_id = pair_to_image_id.get(pair_key)
        if previous_id is not None and previous_id != image_id:
            raise CounterfactualError(
                f"RGB/T pair {pair_key!r} maps to multiple image_ids"
            )
        if previous is None:
            by_image_id[image_id] = record
            pair_to_image_id[pair_key] = image_id
            ordered.append(record)
    if not ordered:
        raise CounterfactualError("training split contains no RGB/T image pairs")
    return ordered


def mean_calibration_alpha(alpha_batches):
    total = torch.zeros(2, dtype=torch.float64)
    count = 0
    for alpha in alpha_batches:
        if not torch.is_tensor(alpha) or alpha.ndim != 2 or alpha.shape[1] != 2:
            raise CounterfactualError(
                "calibration alpha batches must be tensors with shape [N, 2]"
            )
        if not bool(torch.isfinite(alpha).all().item()):
            raise CounterfactualError("calibration alpha contains NaN or Inf")
        total += alpha.detach().cpu().double().sum(dim=0)
        count += int(alpha.shape[0])
    if count == 0:
        raise CounterfactualError("calibration received zero image pairs")
    weights = (total / count).float()
    if not torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6, rtol=0.0):
        raise CounterfactualError(
            f"calibration weights must sum to one, got {weights.tolist()}"
        )
    return weights, count


def compute_calibration_weights(model, args, device):
    """Compute clean-train mean alpha once per unique RGB/T image pair."""
    dataset = RGBTPEDES(root=args.root_dir, verbose=False, dev_split_file=None)
    pairs = unique_train_image_pairs(dataset.train)
    transform = build_transforms(img_size=args.img_size, is_train=False)
    image_set = ImageDatasetRGBT(
        [item[0] for item in pairs],
        [item[2] for item in pairs],
        [item[3] for item in pairs],
        transform,
    )
    loader = DataLoader(
        image_set,
        batch_size=int(args.test_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )
    alpha_batches = []
    with torch.inference_mode():
        for _, rgb_image, thermal_image in loader:
            outputs = model.encode_rgbt_eval(
                rgb_image.to(device),
                thermal_image.to(device),
                return_local_tokens=True,
            )
            if not isinstance(outputs, dict) or "mraf_alpha" not in outputs:
                raise CounterfactualError(
                    "model eval interface did not return mraf_alpha for calibration"
                )
            alpha_batches.append(outputs["mraf_alpha"])
    weights, count = mean_calibration_alpha(alpha_batches)
    if count != len(pairs):
        raise CounterfactualError(
            f"calibration counted {count} samples for {len(pairs)} image pairs"
        )
    return weights, count


def fuse_counterfactual_features(
    mraf_fusion,
    f_rgb,
    f_ir,
    mode,
    calib_weights=None,
):
    """Return a gate counterfactual and the alpha used, if applicable."""
    if f_rgb.shape != f_ir.shape or f_rgb.ndim != 2:
        raise ValueError(
            "f_rgb and f_ir must have the same [N, C] shape, got "
            f"{tuple(f_rgb.shape)} and {tuple(f_ir.shape)}"
        )
    if bool(getattr(mraf_fusion, "equalize_modality_norm", False)):
        raise CounterfactualError(
            "D/E/F gate replacement requires equalize_modality_norm=False"
        )
    if mode == "learned":
        fused, aux = mraf_fusion(f_rgb, f_ir, return_aux=True)
        return fused, aux["alpha"]
    if mode == "rgb_only":
        return f_rgb, None
    if mode == "ir_only":
        return f_ir, None
    if mode == "equal":
        weights = f_rgb.new_tensor([0.5, 0.5])
    elif mode == "calib":
        if calib_weights is None:
            raise ValueError("calib mode requires calib_weights")
        weights = torch.as_tensor(calib_weights, device=f_rgb.device, dtype=f_rgb.dtype)
        if weights.shape != (2,):
            raise ValueError(f"calib_weights must have shape [2], got {tuple(weights.shape)}")
        if not bool(torch.isfinite(weights).all().item()) or not torch.isclose(
            weights.sum(), weights.new_tensor(1.0), atol=1e-6, rtol=0.0
        ):
            raise ValueError("calib_weights must be finite and sum to one")
    else:
        raise ValueError(f"unsupported counterfactual mode: {mode!r}")
    alpha = weights.view(1, 2).expand(f_rgb.shape[0], -1)
    fused = 2.0 * (
        alpha[:, 0:1] * f_rgb + alpha[:, 1:2] * f_ir
    )
    return fused, alpha


def extract_text_features(evaluator, model, text_loader, device):
    qids = []
    qfeats = []
    with torch.inference_mode():
        for pid, caption in text_loader:
            outputs = evaluator._encode_text_feature(model, caption.to(device))
            if isinstance(outputs, dict):
                outputs = outputs["global_feat"]
            qids.append(pid.detach().cpu().long().view(-1))
            qfeats.append(outputs.detach().cpu().float())
    if not qids:
        raise CounterfactualError("text loader produced no queries")
    return torch.cat(qfeats, dim=0), torch.cat(qids, dim=0)


def extract_image_features(
    evaluator,
    model,
    image_loader,
    device,
):
    core = _core_model(model)
    gids = []
    rgb_features = []
    ir_features = []
    learned_features = []
    learned_alphas = []
    with torch.inference_mode():
        for data in image_loader:
            if len(data) != 3:
                raise CounterfactualError(
                    "follow-up B requires gallery batches (pid, RGB, thermal)"
                )
            pid, rgb_image, thermal_image = data
            rgb_image, thermal_image = evaluator._apply_eval_degradation(
                model,
                rgb_image.to(device),
                thermal_image.to(device),
            )
            outputs = model.encode_rgbt_eval(
                rgb_image,
                thermal_image,
                return_local_tokens=True,
            )
            required = {"global_feat", "rgb_global_feat", "thermal_global_feat"}
            if not isinstance(outputs, dict) or not required.issubset(outputs):
                missing = sorted(required - set(outputs if isinstance(outputs, dict) else ()))
                raise CounterfactualError(
                    f"model eval interface is missing pre-fusion feature keys: {missing}"
                )
            f_rgb = outputs["rgb_global_feat"]
            f_ir = outputs["thermal_global_feat"]
            learned, alpha = fuse_counterfactual_features(
                core.mraf_fusion,
                f_rgb,
                f_ir,
                "learned",
            )
            if not torch.equal(learned, outputs["global_feat"]):
                max_delta = float(
                    (learned.float() - outputs["global_feat"].float()).abs().max().item()
                )
                raise CounterfactualError(
                    "cached features do not reproduce the model's native learned fusion "
                    f"bitwise; max_delta={max_delta:.9g}"
                )
            if "mraf_alpha" in outputs and not torch.equal(alpha, outputs["mraf_alpha"]):
                raise CounterfactualError(
                    "direct MRAF alpha differs from encode_rgbt_eval mraf_alpha"
                )
            gids.append(pid.detach().cpu().long().view(-1))
            rgb_features.append(f_rgb.detach().cpu().float())
            ir_features.append(f_ir.detach().cpu().float())
            learned_features.append(learned.detach().cpu().float())
            learned_alphas.append(alpha.detach().cpu().float())
    if not gids:
        raise CounterfactualError("image loader produced no gallery images")
    return (
        torch.cat(rgb_features, dim=0),
        torch.cat(ir_features, dim=0),
        torch.cat(learned_features, dim=0),
        torch.cat(learned_alphas, dim=0),
        torch.cat(gids, dim=0),
    )


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale temporary file: {temporary}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class CachedFeatureEvaluator(Evaluator):
    """Reuse Evaluator.eval while replacing only its feature-extraction source."""

    def __init__(self, qfeats, gfeats, qids, gids):
        super().__init__(img_loader=(), txt_loader=(), source="followup_gate_counterfactual")
        self._cached = (qfeats, gfeats, qids, gids)

    def _compute_embedding(
        self,
        model,
        return_local_tokens=False,
        use_mtp_eval_fusion=False,
    ):
        if return_local_tokens or use_mtp_eval_fusion:
            raise CounterfactualError(
                "cached feature evaluation cannot provide local/MTP embeddings"
            )
        return self._cached


class _MetricAdapter(nn.Module):
    def __init__(self, device):
        super().__init__()
        self._device_anchor = nn.Parameter(
            torch.zeros((), device=device), requires_grad=False
        )
        self.args = SimpleNamespace(
            use_qalr=False,
            use_qalr_v2=False,
            qalr_v3_candidate_diag=False,
            use_qalr_v3_gate=False,
            use_qalr_v3_two_stage_gate=False,
            use_transport_rerank=False,
            use_uspc_slot_rerank=False,
            use_modality_text_prompt=False,
            save_eval_scores=False,
            eval_scores_save_raw=False,
            eval_degrade_mode="none",
            eval_degrade_strength=1.0,
            eval_degrade_blur_kernel=7,
            eval_degrade_seed=2026,
            training=False,
        )


def evaluate_cached_features(qfeats, gfeats, qids, gids, device):
    """Use Evaluator for L2 normalization, similarity, rank and all metrics."""
    evaluator = CachedFeatureEvaluator(
        qfeats.to(device),
        gfeats.to(device),
        qids.detach().cpu().long(),
        gids.detach().cpu().long(),
    )
    adapter = _MetricAdapter(device)
    with torch.inference_mode():
        evaluator.eval(adapter.eval())
    if evaluator.last_eval_result is None:
        raise CounterfactualError("Evaluator did not produce last_eval_result")
    return dict(evaluator.last_eval_result)


def _load_reference_metrics(path, requested_runs, dseed):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing MAIN_GRID metrics.tsv: {path}")
    requested = {
        (group, int(seed), scenario)
        for group, seed in requested_runs
        for scenario, _, _, _ in CORE_SCENARIOS
    }
    references = {}
    scenario_specs = {
        scenario: (degrade_mode, float(strength), int(kernel))
        for scenario, degrade_mode, strength, kernel in CORE_SCENARIOS
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "group",
            "seed",
            "scenario",
            "mode",
            "strength",
            "kernel",
            "dseed",
            "split",
            *METRIC_NAMES,
        }
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise CounterfactualError(
                f"MAIN_GRID metrics.tsv is missing columns: {missing_columns}"
            )
        for row in reader:
            key = (row["group"], int(row["seed"]), row["scenario"])
            if (
                key not in requested
                or row["split"] != "test"
                or int(float(row["dseed"])) != int(dseed)
            ):
                continue
            if key in references:
                raise CounterfactualError(
                    f"duplicate MAIN_GRID reference row for {key} at dseed={dseed}"
                )
            expected_mode, expected_strength, expected_kernel = scenario_specs[key[2]]
            actual_mode = row["mode"]
            actual_strength = float(row["strength"])
            actual_kernel = int(float(row["kernel"]))
            if (
                actual_mode != expected_mode
                or abs(actual_strength - expected_strength) > 1e-12
                or actual_kernel != expected_kernel
            ):
                raise CounterfactualError(
                    "MAIN_GRID scenario parameters do not match the frozen core "
                    f"specification for {key}: expected "
                    f"mode={expected_mode!r}, strength={expected_strength}, "
                    f"kernel={expected_kernel}; got mode={actual_mode!r}, "
                    f"strength={actual_strength}, kernel={actual_kernel}"
                )
            references[key] = {name: float(row[name]) for name in METRIC_NAMES}
    missing = sorted(requested - set(references))
    if missing:
        raise CounterfactualError(
            f"MAIN_GRID metrics.tsv is missing requested core rows: {missing}"
        )
    return references


def build_reproduction_rows(metric_rows, references, tolerance=REPRODUCTION_TOLERANCE):
    learned = {
        (row["group"], int(row["seed"]), row["scenario"]): row
        for row in metric_rows
        if row["mode"] == "learned"
    }
    output = []
    failures = []
    for key in sorted(references, key=lambda item: (item[0], item[1], item[2])):
        if key not in learned:
            raise CounterfactualError(f"missing reproduced learned row for {key}")
        row = learned[key]
        for metric in METRIC_NAMES:
            original = float(references[key][metric])
            reproduced = float(row[metric])
            delta = reproduced - original
            passed = abs(delta) <= float(tolerance)
            item = {
                "group": key[0],
                "seed": key[1],
                "scenario": key[2],
                "metric": metric,
                "original": original,
                "reproduced": reproduced,
                "delta": delta,
                "tolerance": float(tolerance),
                "passed": str(bool(passed)).lower(),
            }
            output.append(item)
            if not passed:
                failures.append(item)
    return output, failures


def _write_tsv_atomic(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale temporary file: {temporary}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale temporary file: {temporary}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _git_commit():
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_fresh_outputs(out_dir):
    names = (
        "metrics_counterfactual.tsv",
        "calib_weights.tsv",
        "reproduction_check.tsv",
        "run_record.json",
    )
    existing = [str(out_dir / name) for name in names if (out_dir / name).exists()]
    features_root = out_dir / "features"
    if features_root.exists() and any(features_root.iterdir()):
        existing.append(str(features_root))
    if existing:
        raise FileExistsError(
            "refusing to mix a new invocation with existing formal outputs: "
            f"{existing}"
        )


def _paths_overlap(left, right):
    left = Path(left).expanduser().resolve()
    right = Path(right).expanduser().resolve()
    return left == right or left in right.parents or right in left.parents


def _require_independent_output(out_dir, run_dirs):
    protected_paths = (MAIN_METRICS.parent.resolve(), *run_dirs)
    conflicts = [
        str(path)
        for path in protected_paths
        if _paths_overlap(out_dir, path)
    ]
    if conflicts:
        raise CounterfactualError(
            "--out must be independent of MAIN_GRID and every input run-dir; "
            f"out={out_dir}, overlapping_paths={conflicts}"
        )


def _run_identity_from_config(run_dir):
    config_path = Path(run_dir) / "configs.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing configs.yaml: {config_path}")
    return parse_run_identity(load_train_configs(str(config_path)))


def _report_reproduction_failures(failures):
    for item in failures:
        print(
            "[REPRODUCTION_MISMATCH] "
            f"group={item['group']} seed={item['seed']} "
            f"scenario={item['scenario']} metric={item['metric']} "
            f"original={item['original']:.9g} "
            f"reproduced={item['reproduced']:.9g} "
            f"delta={item['delta']:+.9g}",
            file=sys.stderr,
        )


def run(args):
    if args.calib_source != "train_clean":
        raise CounterfactualError(
            "the frozen protocol only permits --calib-source train_clean"
        )
    if int(args.dseed) != 2026:
        raise CounterfactualError(
            f"the frozen protocol requires --dseed 2026, got {args.dseed}"
        )
    run_dirs = []
    for group in args.run_dir:
        run_dirs.extend(group)
    run_dirs = [Path(path).expanduser().resolve() for path in run_dirs]
    if len(set(run_dirs)) != len(run_dirs):
        raise CounterfactualError("--run-dir contains duplicate paths")
    group_order = {"D": 0, "E": 1, "F": 2}
    input_identities = {
        run_dir: _run_identity_from_config(run_dir)
        for run_dir in run_dirs
    }
    if len(set(input_identities.values())) != len(input_identities):
        raise CounterfactualError(
            "--run-dir contains duplicate group/seed identities: "
            f"{list(input_identities.values())}"
        )
    run_dirs.sort(
        key=lambda path: (
            group_order[input_identities[path][0]],
            input_identities[path][1],
        )
    )

    out_dir = Path(args.out).expanduser().resolve()
    _require_independent_output(out_dir, run_dirs)
    _require_fresh_outputs(out_dir)
    reference_metrics = _load_reference_metrics(
        MAIN_METRICS,
        set(input_identities.values()),
        args.dseed,
    )
    if not torch.cuda.is_available():
        raise CounterfactualError("CUDA is required to reproduce MAIN_GRID evaluation")
    device = torch.device("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    features_root = out_dir / "features"
    features_root.mkdir(parents=True, exist_ok=True)

    code_commit = _git_commit()
    metric_rows = []
    calibration_rows = []
    reproduction_rows = []
    run_records = []
    identities = set()
    overall_start = time.monotonic()

    for run_dir in run_dirs:
        run_start = time.monotonic()
        metric_start = len(metric_rows)
        loaded = load_run_model(run_dir, device)
        identity = (loaded["group"], loaded["seed"])
        if identity != input_identities[run_dir]:
            raise CounterfactualError(
                "configs.yaml identity changed after preflight validation: "
                f"expected={input_identities[run_dir]}, loaded={identity}, "
                f"path={loaded['config_path']}"
            )
        if identity in identities:
            raise CounterfactualError(f"duplicate group/seed input: {identity}")
        identities.add(identity)
        model = loaded["model"]
        core = _core_model(model)

        if "calib" in args.modes:
            calib_weights, calibration_count = compute_calibration_weights(
                model,
                loaded["args"],
                device,
            )
        else:
            calib_weights = None
            calibration_count = 0
        if calib_weights is not None:
            calibration_rows.append(
                {
                    "group": loaded["group"],
                    "seed": loaded["seed"],
                    "source": args.calib_source,
                    "alpha_rgb": float(calib_weights[0].item()),
                    "alpha_ir": float(calib_weights[1].item()),
                    "n_image_pairs": calibration_count,
                }
            )

        text_evaluator = Evaluator(
            loaded["image_loader"],
            loaded["text_loader"],
            source="followup_gate_counterfactual_text",
        )
        text_features, query_pids = extract_text_features(
            text_evaluator,
            model,
            loaded["text_loader"],
            device,
        )
        cache_dir = features_root / f"{loaded['group']}_seed{loaded['seed']}"
        if cache_dir.exists() and any(cache_dir.iterdir()):
            raise FileExistsError(f"feature cache directory is not empty: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_bytes = 0

        for scenario, degrade_mode, strength, kernel in CORE_SCENARIOS:
            loaded["args"].eval_degrade_mode = degrade_mode
            loaded["args"].eval_degrade_strength = float(strength)
            loaded["args"].eval_degrade_blur_kernel = int(kernel)
            loaded["args"].eval_degrade_seed = int(args.dseed)
            scenario_evaluator = Evaluator(
                loaded["image_loader"],
                loaded["text_loader"],
                source=f"followup_gate_counterfactual:{scenario}",
            )
            f_rgb, f_ir, learned_fused, learned_alpha, gallery_pids = extract_image_features(
                scenario_evaluator,
                model,
                loaded["image_loader"],
                device,
            )
            cache_path = cache_dir / f"{scenario}.pt"
            if cache_path.exists():
                raise FileExistsError(f"refusing to overwrite feature cache: {cache_path}")
            _atomic_torch_save(
                {
                    "group": loaded["group"],
                    "seed": loaded["seed"],
                    "scenario": scenario,
                    "dseed": int(args.dseed),
                    "f_rgb": f_rgb.float(),
                    "f_ir": f_ir.float(),
                    "learned_fused": learned_fused.float(),
                    "learned_alpha": learned_alpha.float(),
                    "text_features": text_features.float(),
                    "gallery_pids": gallery_pids.long(),
                    "query_pids": query_pids.long(),
                },
                cache_path,
            )
            cache_bytes += cache_path.stat().st_size
            cached = _torch_load(cache_path)
            rgb_device = cached["f_rgb"].to(device)
            ir_device = cached["f_ir"].to(device)
            for mode in args.modes:
                if mode == "learned":
                    image_features = cached["learned_fused"].to(device)
                else:
                    image_features, counterfactual_alpha = fuse_counterfactual_features(
                        core.mraf_fusion,
                        rgb_device,
                        ir_device,
                        mode,
                        calib_weights=calib_weights,
                    )
                metrics = evaluate_cached_features(
                    cached["text_features"],
                    image_features,
                    cached["query_pids"],
                    cached["gallery_pids"],
                    device,
                )
                metric_rows.append(
                    {
                        "group": loaded["group"],
                        "seed": loaded["seed"],
                        "scenario": scenario,
                        "mode": mode,
                        **{name: metrics[name] for name in METRIC_NAMES},
                    }
                )
                if mode != "learned":
                    del counterfactual_alpha

        references = {
            key: value
            for key, value in reference_metrics.items()
            if key[:2] == identity
        }
        if len(references) != len(CORE_SCENARIOS):
            raise CounterfactualError(
                f"reference preflight produced {len(references)} core rows for "
                f"{identity}, expected {len(CORE_SCENARIOS)}"
            )
        run_reproduction_rows, failures = build_reproduction_rows(
            metric_rows[metric_start:],
            references,
        )
        reproduction_rows.extend(run_reproduction_rows)
        _write_tsv_atomic(
            out_dir / "reproduction_check.tsv",
            (
                "group",
                "seed",
                "scenario",
                "metric",
                "original",
                "reproduced",
                "delta",
                "tolerance",
                "passed",
            ),
            reproduction_rows,
        )
        if failures:
            _report_reproduction_failures(failures)
            raise CounterfactualError(
                f"{len(failures)} reproduced metrics exceed "
                f"±{REPRODUCTION_TOLERANCE}; metrics_counterfactual.tsv was not written"
            )

        run_records.append(
            {
                "group": loaded["group"],
                "seed": loaded["seed"],
                "run_dir": str(loaded["run_dir"]),
                "checkpoint_path": str(loaded["checkpoint_path"]),
                "configs_path": str(loaded["config_path"]),
                "configs_sha256": loaded["config_sha256"],
                "checkpoint_epoch_for_runtime_stage": loaded["checkpoint_epoch"],
                "full_state_checkpoint": loaded["full_state_checkpoint"],
                "elapsed_seconds": time.monotonic() - run_start,
                "cache_bytes": cache_bytes,
            }
        )
        loaded["model"] = None
        loaded["image_loader"] = None
        loaded["text_loader"] = None
        del (
            image_features,
            rgb_device,
            ir_device,
            cached,
            f_rgb,
            f_ir,
            learned_fused,
            learned_alpha,
            gallery_pids,
            scenario_evaluator,
            text_evaluator,
            text_features,
            query_pids,
            core,
            model,
            loaded,
        )
        torch.cuda.empty_cache()

    _write_tsv_atomic(
        out_dir / "metrics_counterfactual.tsv",
        ("group", "seed", "scenario", "mode", *METRIC_NAMES),
        metric_rows,
    )
    _write_tsv_atomic(
        out_dir / "calib_weights.tsv",
        ("group", "seed", "source", "alpha_rgb", "alpha_ir", "n_image_pairs"),
        calibration_rows,
    )
    _write_json_atomic(
        out_dir / "run_record.json",
        {
            "git_commit": code_commit,
            "dseed": int(args.dseed),
            "scenarios": [
                {
                    "scenario": scenario,
                    "mode": degrade_mode,
                    "strength": float(strength),
                    "kernel": int(kernel),
                }
                for scenario, degrade_mode, strength, kernel in CORE_SCENARIOS
            ],
            "modes": list(args.modes),
            "calib_source": args.calib_source,
            "main_metrics_path": str(MAIN_METRICS),
            "total_elapsed_seconds": time.monotonic() - overall_start,
            "total_cache_bytes": sum(item["cache_bytes"] for item in run_records),
            "runs": run_records,
        },
    )
    print(
        f"[FOLLOWUP_B_DONE] runs={len(run_records)} rows={len(metric_rows)} "
        f"out={out_dir}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fixed-model MRAF gate counterfactual evaluation (inference only)."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        nargs="+",
        required=True,
        help="run directory containing configs.yaml and last.pth; repeatable",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dseed", type=int, default=2026)
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=parse_modes(",".join(ALLOWED_MODES)),
    )
    parser.add_argument("--calib-source", default="train_clean")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (CounterfactualError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
