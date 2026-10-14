"""Read-only counterfactual retrieval gradients at a fixed checkpoint.

The image/text encoders use eval mode (including the MLM branch). Loss formulas,
training batches, degradation, and reuse of the undamaged clean modality match
P4 training. This measures objectives at fixed parameters, not optimizer steps.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.p4.deg_retrieval import DegRetrievalSampler, compute_deg_loss
from tools.p4.train_batch_stats import TRAIN_STATES, _load_training_args
from utils.mdrc import _apply_mode

BASELINES = ("clean_sdm", "clean_total")
CONDITIONS = TRAIN_STATES + ("mixed_train",)
MATRIX_NODES = ("clean",) + CONDITIONS


@contextmanager
def preserve_global_rng():
    """Restore Python, NumPy, CPU/CUDA torch RNG and training seed flags."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    cudnn_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags


def full_state_fingerprint(model):
    """Include every parameter and buffer, including nonpersistent buffers."""
    digest = hashlib.sha256()
    entries = list(model.named_parameters()) + list(model.named_buffers())
    for name, tensor in sorted(entries):
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def encoder_parameter_groups(model):
    """Use CLIP's complete shared visual and complete text tower parameters."""
    base = getattr(model, "base_model", None)
    if base is None or getattr(base, "visual", None) is None:
        raise ValueError("Cannot locate shared image encoder base_model.visual")
    components = ("transformer", "token_embedding", "positional_embedding",
                  "ln_final", "text_projection")
    for component in components:
        if not hasattr(base, component):
            raise ValueError(f"Cannot locate complete CLIP text encoder: {component}")
    image_ids = {id(p) for p in base.visual.parameters()}
    text_ids = set()
    for component in components:
        value = getattr(base, component)
        if isinstance(value, torch.nn.Parameter):
            text_ids.add(id(value))
        elif isinstance(value, torch.nn.Module):
            text_ids.update(id(p) for p in value.parameters())
        else:
            raise ValueError(f"Unexpected CLIP text component: {component}")
    if image_ids & text_ids:
        raise ValueError("Image and text parameter groups overlap")
    base_ids = {id(p) for p in base.parameters()}
    if base_ids != image_ids | text_ids:
        raise ValueError("Unclassified base_model parameters; tower definition is uncertain")
    named = list(model.named_parameters())
    groups = {"image_encoder": [(n, p) for n, p in named if id(p) in image_ids],
              "text_encoder": [(n, p) for n, p in named if id(p) in text_ids]}
    if any(not values for values in groups.values()):
        raise ValueError("Both encoder parameter groups must be nonempty")
    return groups


def _same_tensor_bits(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                       right.detach().contiguous().reshape(-1).view(torch.uint8))


@contextmanager
def readonly_model(model, groups):
    """Temporarily enable tower differentiation without changing .grad/state."""
    flags = [(p, p.requires_grad, p.grad,
              p.grad.detach().clone() if p.grad is not None else None)
             for p in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    before = full_state_fingerprint(model)
    selected = {id(p) for values in groups.values() for _, p in values}
    for parameter, _, _, _ in flags:
        parameter.requires_grad_(id(parameter) in selected)
    model.eval()
    try:
        yield before
    finally:
        changed = full_state_fingerprint(model) != before
        with torch.no_grad():
            for buffer, original in buffers:
                if not _same_tensor_bits(buffer, original):
                    buffer.copy_(original)
        for parameter, required, original_grad, grad_values in flags:
            parameter.requires_grad_(required)
            grad_changed = parameter.grad is not original_grad or (
                original_grad is not None and not _same_tensor_bits(original_grad, grad_values))
            if grad_changed:
                parameter.grad = original_grad
                if original_grad is not None:
                    with torch.no_grad():
                        original_grad.copy_(grad_values)
                changed = True
        for module, training in modes:
            module.training = training
        if changed:
            raise RuntimeError("Read-only diagnostic changed model parameters, buffers, or .grad")


def _gradient_values(loss, parameters, retain_graph=True):
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("Expected finite scalar loss")
    grads = torch.autograd.grad(loss, parameters, allow_unused=True,
                                retain_graph=retain_graph) if loss.requires_grad else [None] * len(parameters)
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):
        raise FloatingPointError("Non-finite encoder gradient")
    # Keep separate tensors, in their native precision, rather than concatenate
    # a very large CLIP gradient vector. None represents exactly zero gradient.
    return [g.detach().cpu() if g is not None else None for g in grads]


def gradient_pair_metrics(clean, degraded):
    """Cosine/norm of concatenated gradients, with unused parameters as zeros."""
    if len(clean) != len(degraded):
        raise ValueError("Gradient parameter lists must align")
    clean_terms, degraded_terms, dot_terms = [], [], []
    chunk_size = 1_048_576
    for left, right in zip(clean, degraded):
        if left is not None and right is not None and left.shape != right.shape:
            raise ValueError("Gradient tensor shapes must align")
        count = left.numel() if left is not None else right.numel() if right is not None else 0
        for start in range(0, count, chunk_size):
            a = left.reshape(-1)[start:start + chunk_size].double() if left is not None else None
            b = right.reshape(-1)[start:start + chunk_size].double() if right is not None else None
            if a is not None:
                clean_terms.append(float(a.square().sum()))
            if b is not None:
                degraded_terms.append(float(b.square().sum()))
            if a is not None and b is not None:
                dot_terms.append(float((a * b).sum()))
    clean_norm = math.sqrt(math.fsum(clean_terms))
    degraded_norm = math.sqrt(math.fsum(degraded_terms))
    cosine = math.fsum(dot_terms) / (clean_norm * degraded_norm) if clean_norm and degraded_norm else None
    if cosine is not None:
        cosine = max(-1.0, min(1.0, cosine))
    return {"cosine": cosine, "norm_ratio": degraded_norm / clean_norm if clean_norm else None,
            "clean_gradient_l2": clean_norm, "degraded_gradient_l2": degraded_norm,
            "clean_parameters_with_gradient": sum(g is not None for g in clean),
            "degraded_parameters_with_gradient": sum(g is not None for g in degraded)}


def gradient_gram(gradient_sets):
    """Chunked CPU float64 Gram matrix, without concatenating CLIP vectors."""
    sets = list(gradient_sets)
    if not sets or any(len(values) != len(sets[0]) for values in sets):
        raise ValueError("Aligned nonempty gradient sets required")
    gram = torch.zeros((len(sets), len(sets)), dtype=torch.float64)
    for components in zip(*sets):
        available = [g for g in components if g is not None]
        if not available:
            continue
        if any(g.shape != available[0].shape for g in available):
            raise ValueError("Gradient tensor shapes must align")
        for start in range(0, available[0].numel(), 262_144):
            end = min(start + 262_144, available[0].numel())
            chunk = torch.stack([
                g.reshape(-1)[start:end].double() if g is not None
                else torch.zeros(end - start, dtype=torch.float64)
                for g in components])
            gram.add_(chunk @ chunk.t())
    if not bool(torch.isfinite(gram).all()):
        raise FloatingPointError("Non-finite gradient Gram matrix")
    return gram


def _metrics_from_gram(gram, left, right, clean, degraded):
    norm_left = math.sqrt(max(0., float(gram[left, left])))
    norm_right = math.sqrt(max(0., float(gram[right, right])))
    cosine = float(gram[left, right]) / (norm_left * norm_right) if norm_left and norm_right else None
    return {"cosine": max(-1., min(1., cosine)) if cosine is not None else None,
            "norm_ratio": norm_right / norm_left if norm_left else None,
            "clean_gradient_l2": norm_left, "degraded_gradient_l2": norm_right,
            "clean_parameters_with_gradient": sum(g is not None for g in clean),
            "degraded_parameters_with_gradient": sum(g is not None for g in degraded)}


def tensor_mapping_fingerprint(values):
    """Fingerprint actual batch inputs to audit alignment across checkpoints."""
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        if torch.is_tensor(value):
            digest.update(key.encode("utf-8"))
            digest.update(str((tuple(value.shape), value.dtype)).encode("ascii"))
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def scalar_summary(values, bootstrap_samples=2000, seed=2026):
    """Batch means and iid batch bootstrap; null zero-norm cases are excluded."""
    valid = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if not np.isfinite(valid).all():
        raise FloatingPointError("Non-finite summary input")
    result = {"count": len(valid), "excluded": len(values) - len(valid),
              "mean": None, "std": None, "ci95": None}
    if not len(valid):
        return result
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(seed)
    means = np.asarray([valid[rng.integers(len(valid), size=len(valid))].mean()
                        for _ in range(bootstrap_samples)])
    result.update(mean=float(valid.mean()), std=float(valid.std(ddof=0)),
                  ci95=np.quantile(means, [.025, .975]).tolist())
    return result


def homogeneous_degradation(batch, state, generator):
    """Whole batch same state; attenuation is shared by repeated image IDs."""
    if state not in TRAIN_STATES:
        raise ValueError(f"Unsupported train state: {state}")
    rgb, ir = batch["images"], batch["thermal_images"]
    image_ids = batch["image_ids"].reshape(-1)
    if rgb.shape[0] != image_ids.numel() or ir.shape != rgb.shape:
        raise ValueError("Paired modalities and image_ids must align")
    unique, inverse = torch.unique(image_ids, sorted=True, return_inverse=True)
    value = 0.0 if state.endswith("_drop") else .1
    strengths = torch.full((unique.numel(),), value, device=rgb.device)
    if state.endswith("_attenuate"):
        strengths.uniform_(.2, .6, generator=generator)
    row_strengths = strengths[inverse]
    mask = torch.ones(rgb.shape[0], device=rgb.device, dtype=torch.bool)
    target = rgb if state.startswith("rgb_") else ir
    changed = _apply_mode(target, mask, state, generator=generator, factors=row_strengths)
    rgb_mask = mask if state.startswith("rgb_") else ~mask
    metadata = {"enabled": True, "valid_modes": TRAIN_STATES,
                "mode_indices": torch.full((rgb.shape[0],), TRAIN_STATES.index(state),
                                           device=rgb.device, dtype=torch.long),
                "strengths": row_strengths, "rgb_mask": rgb_mask, "ir_mask": ~rgb_mask,
                "unique_image_ids": unique}
    return dict(batch, images=changed if state.startswith("rgb_") else rgb,
                thermal_images=changed if state.startswith("ir_") else ir), metadata


def _validate_A(args):
    if set(str(args.loss_names).split("+")) != {"sdm", "id", "mlm"}:
        raise ValueError("Requires historical A loss_names=sdm+mlm+id")
    if not bool(getattr(args, "use_rgbt_add_fusion", False)):
        raise ValueError("Requires historical A additive RGB-T fusion")
    for flag in ("use_mum", "use_global_uncertainty", "mraf_enabled", "mdrc_enable",
                 "use_synth_loss_weighting"):
        if bool(getattr(args, flag, False)):
            raise ValueError(f"Requires historical A with {flag}=false")


def diagnose_batches(model, batches, args, num_batches=50, deg_seed=108919,
                     bootstrap_samples=2000, bootstrap_seed=2026):
    """Differentiate real clean and P4 degraded forward paths without backward."""
    if num_batches <= 0 or bootstrap_samples <= 0:
        raise ValueError("num_batches and bootstrap_samples must be positive")
    from processor.processor import _build_total_loss
    _validate_A(args)
    groups = encoder_parameter_groups(model)
    names = {key: [name for name, _ in values] for key, values in groups.items()}
    flat = [parameter for values in groups.values() for _, parameter in values]
    sizes = {key: len(values) for key, values in groups.items()}
    device = next(model.parameters()).device
    state_seeds = {state: int(deg_seed) + i + 1 for i, state in enumerate(TRAIN_STATES)}
    generators = {state: torch.Generator(device=device).manual_seed(seed)
                  for state, seed in state_seeds.items()}
    mixed = DegRetrievalSampler(loss_name="sdm", seed=deg_seed, branch_prob=1.)
    records, matrix_records, batch_metadata = [], [], []
    with preserve_global_rng(), readonly_model(model, groups) as before:
        for index, source in enumerate(islice(batches, num_batches)):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_started = time.perf_counter()
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in source.items()}
            if any(key not in batch for key in ("pids", "image_ids", "images", "thermal_images")):
                raise ValueError("Training batches need pids/image_ids and paired images")
            metadata_record = {"batch_index": index, "input_sha256": tensor_mapping_fingerprint(batch),
                               "pids": batch["pids"].detach().cpu().tolist(),
                               "image_ids": batch["image_ids"].detach().cpu().tolist(), "conditions": {}}
            clean = model(batch)
            clean_loss = clean["sdm_loss"]
            clean_total = _build_total_loss(args, clean)
            baseline_grads = {"clean_sdm": _gradient_values(clean_loss, flat),
                              "clean_total": _gradient_values(clean_total, flat)}
            clean_losses = {"clean_sdm": float(clean_loss.detach()),
                            "clean_total": float(clean_total.detach())}
            loss_keys = [key for key in clean if "loss" in key]
            features = {key: clean[key] for key in ("v_rgb_c", "v_t_c", "fused_det_feat", "text_det_feat")}
            matrix_grads = {"sdm": {"clean": baseline_grads["clean_sdm"]},
                            "ce": {"clean": baseline_grads["clean_sdm"]}}
            node_losses = {"sdm": {"clean": clean_losses["clean_sdm"]},
                           "ce": {"clean": clean_losses["clean_sdm"]}}
            del clean, clean_loss, clean_total
            for condition in CONDITIONS:
                if condition == "mixed_train":
                    rgb, ir, metadata = mixed.sample(batch["images"], batch["thermal_images"], batch["image_ids"])
                    degraded_batch = dict(batch, images=rgb, thermal_images=ir)
                else:
                    degraded_batch, metadata = homogeneous_degradation(batch, condition, generators[condition])
                metadata_record["conditions"][condition] = {
                    "input_sha256": tensor_mapping_fingerprint(degraded_batch),
                    "mode_indices": metadata["mode_indices"].detach().cpu().tolist(),
                    "strengths": metadata["strengths"].detach().cpu().tolist(),
                    "rgb_mask": metadata["rgb_mask"].detach().cpu().tolist(),
                    "ir_mask": metadata["ir_mask"].detach().cpu().tolist(),
                }
                degraded = model(degraded_batch, mdrc_feature_only=True,
                                 p4_clean_features=features, p4_mode_meta=metadata)
                for loss_name in ("sdm", "ce"):
                    loss = compute_deg_loss(degraded["fused_det_feat"], features["text_det_feat"],
                                            batch["pids"], model.logit_scale, loss_name)
                    last = condition == CONDITIONS[-1] and loss_name == "ce"
                    matrix_grads[loss_name][condition] = _gradient_values(loss, flat, retain_graph=not last)
                    node_losses[loss_name][condition] = float(loss.detach())
                    del loss
                del degraded, degraded_batch, metadata
            del features
            offset = 0
            record_start, matrix_start = len(records), len(matrix_records)
            for group, size in sizes.items():
                for loss_name in ("sdm", "ce"):
                    node_grads = [matrix_grads[loss_name][node][offset:offset + size] for node in MATRIX_NODES]
                    all_grads = node_grads + [baseline_grads[baseline][offset:offset + size] for baseline in BASELINES]
                    gram = gradient_gram(all_grads)
                    for left, left_name in enumerate(MATRIX_NODES):
                        for right, right_name in enumerate(MATRIX_NODES):
                            metrics = _metrics_from_gram(gram, left, right, node_grads[left], node_grads[right])
                            matrix_records.append({"batch_index": index, "loss_name": loss_name,
                                                   "encoder": group, "left": left_name, "right": right_name,
                                                   "cosine": metrics["cosine"]})
                    for node_index, condition in enumerate(MATRIX_NODES[1:], start=1):
                        for baseline_index, baseline in enumerate(BASELINES, start=len(MATRIX_NODES)):
                            metrics = _metrics_from_gram(gram, baseline_index, node_index,
                                                         all_grads[baseline_index], all_grads[node_index])
                            records.append({"batch_index": index, "batch_size": int(batch["pids"].numel()),
                                            "condition": condition, "degraded_loss_name": loss_name,
                                            "baseline": baseline, "encoder": group,
                                            "clean_loss": clean_losses[baseline],
                                            "degraded_loss": node_losses[loss_name][condition], **metrics})
                    del gram, all_grads, node_grads
                offset += size
            del baseline_grads, matrix_grads
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - batch_started
            metadata_record["seconds"] = elapsed
            metadata_record["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            batch_metadata.append(metadata_record)
            for record in records[record_start:] + matrix_records[matrix_start:]:
                record["batch_seconds"] = elapsed
            print(f"batch={index + 1}/{num_batches} seconds={elapsed:.3f} peak_gpu_memory_bytes={metadata_record['peak_gpu_memory_bytes']}")
    if not records:
        raise ValueError("Training loader produced no batches")
    after = full_state_fingerprint(model)
    if before != after:
        raise RuntimeError("Read-only model fingerprint mismatch")
    summaries = []
    for condition in CONDITIONS:
        for loss_name in ("sdm", "ce"):
            for baseline in BASELINES:
                for encoder in groups:
                    selected = [r for r in records if (r["condition"], r["degraded_loss_name"], r["baseline"], r["encoder"])
                                == (condition, loss_name, baseline, encoder)]
                    summaries.append({"condition": condition, "degraded_loss_name": loss_name,
                                      "baseline": baseline, "encoder": encoder,
                                      **{field: scalar_summary([r[field] for r in selected], bootstrap_samples, bootstrap_seed)
                                         for field in ("cosine", "norm_ratio")}})
    matrices = {}
    for loss_name in ("sdm", "ce"):
        matrices[loss_name] = {}
        for group in groups:
            values = [[scalar_summary([r["cosine"] for r in matrix_records
                                      if (r["loss_name"], r["encoder"], r["left"], r["right"])
                                      == (loss_name, group, left, right)], bootstrap_samples, bootstrap_seed)
                       for right in MATRIX_NODES] for left in MATRIX_NODES]
            matrices[loss_name][group] = {"nodes": list(MATRIX_NODES), "statistics": values}
    return {"schema_version": 1, "batches": records, "summaries": summaries,
            "pairwise_batch_cosines": matrix_records, "cosine_matrices": matrices,
            "batch_metadata": batch_metadata,
            "num_batches_observed": max(r["batch_index"] for r in records) + 1,
            "encoder_parameter_names": names,
            "encoder_parameter_counts": {key: sum(p.numel() for _, p in values) for key, values in groups.items()},
            "clean_model_loss_keys": loss_keys, "state_generator_seeds": state_seeds,
            "parameters_and_buffers_unchanged": True, "global_rng_restored": True,
            "model_state_sha256_before": before, "model_state_sha256_after": after}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--train_seed", type=int, default=None)
    parser.add_argument("--deg_rng_seed", type=int, default=None)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--dev_split_file", default=None)
    return parser


def _main(cli):
    started = time.perf_counter()
    if cli.num_batches <= 0 or cli.bootstrap_samples <= 0:
        raise ValueError("num_batches and bootstrap_samples must be positive")
    if cli.batch_size is not None and cli.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cli.num_workers is not None and cli.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    for path in (cli.checkpoint, cli.config_file):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    output_dir = Path(cli.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    from datasets import build_dataloader
    from model import build_model
    from train import set_seed
    from tools.p4.export_dev_records import load_strict_checkpoint
    args = _load_training_args(cli.config_file)
    _validate_A(args)
    args.training, args.distributed = True, False
    for name in ("batch_size", "num_workers", "root_dir", "dev_split_file"):
        if getattr(cli, name) is not None:
            setattr(args, name, getattr(cli, name))
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Training config needs batch_size > 0 and num_workers >= 0")
    train_seed = cli.train_seed if cli.train_seed is not None else int(os.environ.get("TRAIN_SEED", getattr(args, "train_seed", 101)))
    saved_deg_seed = getattr(args, "deg_rng_seed", None)
    deg_seed = cli.deg_rng_seed if cli.deg_rng_seed is not None else int(saved_deg_seed if saved_deg_seed is not None else train_seed * 1000 + 7919)
    device = "cuda" if cli.device == "auto" and torch.cuda.is_available() else "cpu" if cli.device == "auto" else cli.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_seed(train_seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_loader, _, _, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    checkpoint = load_strict_checkpoint(model, cli.checkpoint)
    if device == "cpu":
        model.float()
    model.to(device)
    result = diagnose_batches(model, train_loader, args, cli.num_batches, deg_seed,
                              cli.bootstrap_samples, cli.bootstrap_seed)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    result.update(checkpoint=str(Path(cli.checkpoint).resolve()), config_file=str(Path(cli.config_file).resolve()),
                  checkpoint_sha256=hashlib.sha256(Path(cli.checkpoint).read_bytes()).hexdigest(),
                  checkpoint_epoch=checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
                  config_sha256=hashlib.sha256(Path(cli.config_file).read_bytes()).hexdigest(),
                  git_commit=commit, training_args=vars(args), train_seed=train_seed, deg_rng_seed=deg_seed,
                  num_batches_requested=cli.num_batches, device=device, bootstrap_seed=cli.bootstrap_seed,
                  precision="repository_manual_fp16" if device == "cuda" else "cpu_fp32",
                  bootstrap_samples=cli.bootstrap_samples,
                  elapsed_seconds=time.perf_counter() - started,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                  definitions={
                      "mode": "fixed checkpoint eval mode; original training data augmentation/MLM masks; no optimizer, backward, or parameter update",
                      "dropout": "disabled by model.eval, matching train_batch_stats.py; these are fixed-checkpoint objective gradients, not stochastic model.train gradients",
                      "clean_sdm": "actual model forward sdm_loss: objectives.compute_sdm with original PID-only targets",
                      "clean_total": "processor._build_total_loss of clean forward only; historical A legacy sum includes both weighted mlm_loss and unweighted loss_mlm, excluding duplicate loss_sdm_prob",
                      "degraded": "whole B-image/B-text independent SDM or uniform multipos CE; attached clean text and undamaged clean modality reused, exactly P4 feature-only path",
                      "mixed_train": "DegRetrievalSampler with branch_prob=1 and original independent deg_rng_seed; modes/factors shared per unique image_id, noise per row",
                      "homogeneous": "all rows same training state via utils.mdrc._apply_mode; attenuation U(0.2,0.6) per unique image_id, sigma=0.1 noise without clipping",
                      "parameters": "complete base_model.visual; complete CLIP transformer/token_embedding/positional_embedding/ln_final/text_projection; excludes ID, MLM, auxiliary heads",
                      "aggregation": "unweighted batch mean and population std; percentile 95% iid batch bootstrap, not PID/query bootstrap",
                      "pairwise_matrices": "8x8 clean/six homogeneous/mixed objectives per tower; clean node always uses actual clean SDM for both matrices, while sdm/ce names select the seven degraded-node objectives; clean-total baseline is reported separately",
                      "zero_norm": "undefined cosine or ratio is null and excluded with counts; unused parameters are exact zero components",
                      "rng": "Python/NumPy/torch CPU/CUDA global RNG and cuDNN seed flags restored for entire main; state and mixed generators are independent",
                      "batch_timing": "batch_seconds includes forward, autograd and CPU Gram reduction, excludes loader fetch; elapsed_seconds also includes pipeline/model/checkpoint construction and fetching",
                  })
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "grad_conflict.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output_dir / "batches.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["batches"][0]))
        writer.writeheader()
        writer.writerows(result["batches"])
    with (output_dir / "pairwise_cosines.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["pairwise_batch_cosines"][0]))
        writer.writeheader()
        writer.writerows(result["pairwise_batch_cosines"])
    rows = []
    for summary in result["summaries"]:
        for metric in ("cosine", "norm_ratio"):
            values = summary[metric]
            rows.append({**{key: summary[key] for key in ("condition", "degraded_loss_name", "baseline", "encoder")},
                         "metric": metric, "count": values["count"], "excluded": values["excluded"],
                         "mean": values["mean"], "std": values["std"],
                         "ci95_low": values["ci95"][0] if values["ci95"] else None,
                         "ci95_high": values["ci95"][1] if values["ci95"] else None})
    with (output_dir / "summary.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{output_dir / 'grad_conflict.json'}: batches={result['num_batches_observed']}, parameters_and_buffers_unchanged=True, global_rng_restored=True, seconds={result['elapsed_seconds']:.3f}, peak_gpu_memory_bytes={result['peak_gpu_memory_bytes']}")
    return result


def main(argv=None):
    with preserve_global_rng():
        return _main(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
