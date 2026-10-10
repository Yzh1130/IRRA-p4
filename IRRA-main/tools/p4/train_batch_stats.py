"""Read-only training-batch diagnostics; no optimizer or checkpoint writes.

S is the scaled text-to-image cosine-logit matrix. Pair weights modify only
positive target mass: Q_t2i = (PID_mask * A) / row_sum, Q_i2t normalizes its
transpose independently. SDM uses KL(P || Q), CE uses -Q * log(P), and both
sum their two directional row means. This explicitly defines the experimental
weighted targets; the uniform target is exactly the existing PID-only target.

For distribution diagnostics e_ij = a_ij / m_i, where m_i counts PID-positive
image columns for text query i. e is deliberately not renormalized by sum(a).
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

TRAIN_STATES = (
    "rgb_drop", "ir_drop", "rgb_noise", "ir_noise",
    "rgb_attenuate", "ir_attenuate",
)
WEIGHT_VARIANTS = ("uniform", "E", "D", "D-shuf")
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def load_kappa(path):
    """Read six explicit train-state kappas, retaining raw and clipped values.

Accepted JSON: {"states": {state: raw_number_or_record}, "R1_clean": number}
or a direct {state: raw_number_or_record} mapping. Records contain ``raw``
(or ``kappa_raw``), or ``R1`` with R1_clean. Supplied clipped values are checked
against clipping the raw value, rather than silently trusted. Attenuation must
have its own measurement; no missing-state value is inferred from other modes.
"""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("kappa JSON must be an object")
    states = payload.get("states", payload)
    clean_r1 = payload.get("R1_clean", payload.get("clean_R1"))
    result = {}
    for state in TRAIN_STATES:
        if state not in states:
            raise ValueError(f"Missing measured kappa for train state: {state}")
        item = states[state]
        record = dict(item) if isinstance(item, dict) else {"raw": item}
        if "raw" in record or "kappa_raw" in record:
            raw = float(record.get("raw", record.get("kappa_raw")))
        elif "R1" in record and clean_r1 is not None:
            clean = float(clean_r1)
            if not math.isfinite(clean) or clean <= 0:
                raise ValueError("R1_clean must be finite and positive")
            raw = 1.0 - float(record["R1"]) / clean
        else:
            raise ValueError(f"{state} needs raw/kappa_raw or R1 with R1_clean")
        if not math.isfinite(raw):
            raise ValueError(f"Non-finite raw kappa for {state}")
        clipped = min(1.0, max(0.0, raw))
        supplied = record.get("clipped", record.get("kappa_clipped"))
        if supplied is not None and not math.isclose(
            float(supplied), clipped, abs_tol=1e-8, rel_tol=1e-6
        ):
            raise ValueError(f"Inconsistent raw/clipped kappa for {state}")
        result[state] = {"raw": raw, "clipped": clipped, "source": record}
    return result


def positive_targets(pids, pair_weights=None):
    """Return PID mask and independent t2i/i2t normalized positive targets."""
    pids = pids.reshape(-1)
    positive = pids[:, None].eq(pids[None, :])
    mass = positive.to(dtype=torch.float32)
    if pair_weights is not None:
        if pair_weights.shape != mass.shape:
            raise ValueError("pair_weights must have shape [batch, batch]")
        if not torch.isfinite(pair_weights).all() or (pair_weights <= 0).any():
            raise ValueError("pair_weights must be finite and strictly positive")
        mass = mass * pair_weights.to(device=pids.device, dtype=mass.dtype)
    q_t2i = mass / mass.sum(dim=1, keepdim=True)
    reverse = mass.t()
    q_i2t = reverse / reverse.sum(dim=1, keepdim=True)
    return positive, q_t2i, q_i2t


def loss_from_logits(logits, pids, loss_name, pair_weights=None, eps=1e-8):
    """Two-direction retrieval loss with explicit positive-target weighting."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("logits must be square [batch, batch]")
    if pids.numel() != logits.shape[0]:
        raise ValueError("PID count must match logits")
    _, q_t2i, q_i2t = positive_targets(pids, pair_weights)
    directional = []
    for scores, q in ((logits, q_t2i), (logits.t(), q_i2t)):
        log_p = F.log_softmax(scores, dim=1)
        if loss_name == "sdm":
            rows = (log_p.exp() * (log_p - (q + eps).log())).sum(dim=1)
        elif loss_name == "ce":
            rows = -(q * log_p).sum(dim=1)
        else:
            raise ValueError("loss_name must be sdm or ce")
        directional.append(rows.mean())
    return directional[0] + directional[1]


def weight_variants(pids, rho, states, kappas, floor, shuffle_generator):
    """Build uniform/E/D/D-shuf, with E fitted only on real positive pairs."""
    if not math.isfinite(float(floor)) or not 0 < floor <= 1:
        raise ValueError("floor must be in (0, 1]")
    pids = pids.reshape(-1)
    rho = torch.as_tensor(rho, device=pids.device, dtype=torch.float32).reshape(-1)
    if len(states) != pids.numel() or rho.numel() != pids.numel():
        raise ValueError("states/rho must each contain one entry per batch row")
    if not torch.isfinite(rho).all() or (rho < 0).any() or (rho > 1).any():
        raise ValueError("rho must be finite and in [0, 1]")
    kappa = rho.new_tensor([kappas[s]["clipped"] for s in states])
    positive = pids[:, None].eq(pids[None, :])
    raw_d = 1.0 - rho[:, None] * kappa[None, :]
    d = raw_d.clamp_min(floor)
    # CPU-only shuffle stream is independent of the sampler's device generator.
    permutation = torch.randperm(rho.numel(), generator=shuffle_generator)
    shuffled_rho = rho.index_select(0, permutation.to(device=rho.device))
    raw_shuf = 1.0 - shuffled_rho[:, None] * kappa[None, :]
    d_shuf = raw_shuf.clamp_min(floor)
    e = torch.ones_like(d)
    state_means = {}
    for state in sorted(set(states)):
        columns = torch.tensor([s == state for s in states], device=rho.device)
        selected = positive & columns[None, :]
        mean = d[selected].mean()
        e[:, columns] = mean
        state_means[state] = float(mean)
    return {
        "uniform": torch.ones_like(d), "E": e, "D": d, "D-shuf": d_shuf,
    }, {
        "permutation": permutation.tolist(), "state_positive_means": state_means,
        "floor_hit_D": raw_d <= floor, "floor_hit_D-shuf": raw_shuf <= floor,
    }


def distribution_summary(values, floor_flags=None):
    values = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if not values.numel():
        return {"count": 0, "mean": None, "std": None, "quantiles": {},
                "floor_rate": None}
    result = {
        "count": values.numel(), "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "quantiles": {str(int(q * 100)): float(torch.quantile(values, q))
                      for q in QUANTILES},
    }
    result["floor_rate"] = (
        float(torch.as_tensor(floor_flags, dtype=torch.float64).mean())
        if floor_flags is not None else None
    )
    return result


def empirical_ks(left, right):
    """Exact two-sample empirical CDF distance, including tied observations."""
    left = torch.sort(torch.as_tensor(left, dtype=torch.float64).reshape(-1)).values
    right = torch.sort(torch.as_tensor(right, dtype=torch.float64).reshape(-1)).values
    if not left.numel() or not right.numel():
        return None
    support = torch.unique(torch.cat((left, right)), sorted=True)
    cdf_left = torch.searchsorted(left, support, right=True).double() / left.numel()
    cdf_right = torch.searchsorted(right, support, right=True).double() / right.numel()
    return float((cdf_left - cdf_right).abs().max())


def pair_distributions(pids, states, variants, details):
    """Collect actual positive pairs by image-column degradation state."""
    positive = pids.reshape(-1, 1).eq(pids.reshape(1, -1))
    m_i = positive.sum(dim=1, keepdim=True).to(dtype=torch.float32)
    result = {}
    for state in sorted(set(states)):
        columns = torch.tensor([s == state for s in states], device=pids.device)
        selected = positive & columns[None, :]
        result[state] = {}
        for name, weights in variants.items():
            a = weights[selected].detach().cpu()
            effective = (weights / m_i)[selected].detach().cpu()
            flags = details.get(f"floor_hit_{name}")
            result[state][name] = {
                "a": a.tolist(), "e": effective.tolist(),
                "floor_flags": flags[selected].detach().cpu().tolist()
                if flags is not None else None,
            }
    return result


def gradient_measurements(logits, pids, variants, encoder_parameters):
    """Measure dL/dS separately from concatenated image-encoder parameter dL/dW."""
    params = list(encoder_parameters)
    if not params or any(not p.requires_grad for p in params):
        raise ValueError("encoder parameters must be nonempty and require grad")
    if not logits.requires_grad:
        raise ValueError("logits need an encoder-connected autograd graph")
    records = {}
    combinations = [(loss, name) for loss in ("sdm", "ce")
                    for name in WEIGHT_VARIANTS]
    for index, (loss_name, name) in enumerate(combinations):
        loss = loss_from_logits(logits, pids, loss_name, variants[name])
        score_grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        parameter_grads = torch.autograd.grad(
            loss, params, retain_graph=index < len(combinations) - 1,
            allow_unused=True,
        )
        norm_squared = sum(float(g.detach().double().square().sum())
                           for g in parameter_grads if g is not None)
        score_norm = float(score_grad.detach().double().square().sum().sqrt())
        finite = (torch.isfinite(loss).item() and torch.isfinite(score_grad).all().item()
                  and all(torch.isfinite(g).all().item() for g in parameter_grads
                          if g is not None))
        records.setdefault(loss_name, {})[name] = {
            "loss": float(loss.detach()), "score_gradient_frobenius": score_norm,
            "encoder_parameter_gradient_l2": math.sqrt(norm_squared),
            "encoder_parameters_with_gradient": sum(g is not None for g in parameter_grads),
            "finite": bool(finite),
        }
    for by_variant in records.values():
        baseline = by_variant["uniform"]
        for record in by_variant.values():
            record["relative_to_uniform"] = {}
            for field in ("loss", "score_gradient_frobenius", "encoder_parameter_gradient_l2"):
                denominator = baseline[field]
                record["relative_to_uniform"][field] = (
                    (record[field] - denominator) / abs(denominator)
                    if denominator != 0 else None
                )
    return records


def state_fingerprint(model):
    """Hash parameters and persistent buffers; does not retain a second model."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        byte_tensor = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(byte_tensor.numpy().tobytes())
    return digest.hexdigest()


def merge_label_statistics(statistics, batch_sizes):
    result = {}
    total = sum(batch_sizes)
    for direction in ("t2i", "i2t"):
        fields = statistics[0][direction]
        result[direction] = {}
        for field in fields:
            histogram = {}
            for item in statistics:
                for value, count in item[direction][field]["histogram"].items():
                    histogram[str(value)] = histogram.get(str(value), 0) + count
            result[direction][field] = {
                "mean": sum(item[direction][field]["mean"] * size
                            for item, size in zip(statistics, batch_sizes)) / total,
                "histogram": histogram,
            }
    return result


def summarize_batches(batch_records, pair_records):
    """Aggregate per-batch scalar diagnostics and positive-pair distributions."""
    weights, ks = {}, {}
    for state, by_variant in pair_records.items():
        weights[state] = {}
        for name, samples in by_variant.items():
            weights[state][name] = {
                "a": distribution_summary(samples["a"], samples["floor_flags"]),
                "effective_e": distribution_summary(samples["e"]),
            }
        ks[state] = empirical_ks(by_variant["D"]["e"], by_variant["D-shuf"]["e"])
    means = {}
    for loss_name in ("sdm", "ce"):
        means[loss_name] = {}
        for name in WEIGHT_VARIANTS:
            entries = [batch["measurements"][loss_name][name] for batch in batch_records]
            means[loss_name][name] = {
                field: sum(entry[field] for entry in entries) / len(entries)
                for field in ("loss", "score_gradient_frobenius", "encoder_parameter_gradient_l2")
            }
            means[loss_name][name]["all_finite"] = all(e["finite"] for e in entries)
        baseline = means[loss_name]["uniform"]
        for values in means[loss_name].values():
            values["relative_to_uniform"] = {
                field: (values[field] - baseline[field]) / abs(baseline[field])
                if baseline[field] else None
                for field in ("loss", "score_gradient_frobenius", "encoder_parameter_gradient_l2")
            }
    return {"positive_pair_weights_by_state": weights,
            "D_vs_D_shuf_effective_weight_KS_by_state": ks,
            "batch_mean_measurements": means}


def _load_training_args(config_file):
    from utils.options import get_args
    previous = sys.argv
    try:
        sys.argv = ["train_batch_stats", "--config_file", str(config_file)]
        return get_args()
    finally:
        sys.argv = previous


def _state_names(meta, batch_size):
    modes = list(meta["valid_modes"])
    indices = torch.as_tensor(meta["mode_indices"]).reshape(-1).cpu().tolist()
    if len(indices) != batch_size or any(i < 0 or i >= len(modes) for i in indices):
        raise ValueError("Expected a degradation state for every image column")
    names = [modes[i] for i in indices]
    if any(s not in TRAIN_STATES for s in names):
        raise ValueError("Stats require the six P4 training degradation states")
    return names


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--kappa_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--deg_rng_seed", type=int, default=None)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--train_seed", type=int, default=None)
    parser.add_argument("--floor", type=float, default=0.05)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--dev_split_file", default=None)
    return parser


def main(argv=None):
    cli = build_parser().parse_args(argv)
    if cli.num_batches <= 0 or not math.isfinite(cli.floor) or not 0 < cli.floor <= 1:
        raise ValueError("num_batches > 0 and floor in (0, 1] are required")
    if cli.batch_size is not None and cli.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cli.num_workers is not None and cli.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    for path in (cli.checkpoint, cli.config_file, cli.kappa_json):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    # Heavy repository imports stay here so pure diagnostics run on CPU tensors.
    from datasets import build_dataloader
    from model import build_model
    from train import set_seed
    from tools.p4.deg_retrieval import DegRetrievalSampler, batch_label_statistics
    from tools.p4.export_dev_records import (
        caption_token_counts, load_strict_checkpoint, load_word_list,
    )

    args = _load_training_args(cli.config_file)
    args.training, args.distributed = True, False
    for name in ("batch_size", "num_workers", "root_dir", "dev_split_file"):
        value = getattr(cli, name)
        if value is not None:
            setattr(args, name, value)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Training config needs batch_size > 0 and num_workers >= 0")
    train_seed = cli.train_seed
    if train_seed is None:
        train_seed = int(os.environ.get("TRAIN_SEED", getattr(args, "train_seed", 101)))
    deg_seed = cli.deg_rng_seed
    if deg_seed is None:
        saved_deg_seed = getattr(args, "deg_rng_seed", None)
        deg_seed = int(saved_deg_seed) if saved_deg_seed is not None else train_seed * 1000 + 7919
    shuffle_seed = cli.shuffle_seed if cli.shuffle_seed is not None else deg_seed + 1
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cli.device == "auto" else cli.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    kappas = load_kappa(cli.kappa_json)
    color_words = load_word_list(Path(__file__).with_name("color_lexicon.txt"))
    stopwords = load_word_list(Path(__file__).with_name("stopwords.txt"))
    # This seeds the ordinary training pipeline before dataset/model construction,
    # exactly as train.py does. The two diagnostic generators stay independent.
    set_seed(train_seed)
    train_loader, _, _, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    checkpoint = load_strict_checkpoint(model, cli.checkpoint)
    if device == "cpu":
        # The repository's manual FP16 CUDA path is preserved on CUDA. CPU
        # diagnostics explicitly use FP32 because FP16 CLIP kernels vary by CPU.
        model.float()
    model.to(device).eval()
    visual = getattr(getattr(model, "base_model", None), "visual", None)
    if visual is None:
        raise ValueError("Cannot locate shared image encoder model.base_model.visual")
    params = list(visual.parameters())
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for parameter, _ in flags:
        parameter.requires_grad_(False)
    for parameter in params:
        parameter.requires_grad_(True)
    before = state_fingerprint(model)
    sampler = DegRetrievalSampler(loss_name="sdm", seed=deg_seed, branch_prob=1.0)
    shuffle = torch.Generator(device="cpu").manual_seed(shuffle_seed)
    batch_records, label_records, batch_sizes = [], [], []
    accumulated_pairs = {
        state: {
            name: {"a": [], "e": [],
                   "floor_flags": [] if name in ("D", "D-shuf") else None}
            for name in WEIGHT_VARIANTS
        } for state in TRAIN_STATES
    }
    try:
        for index, batch in enumerate(islice(train_loader, cli.num_batches)):
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            if "image_ids" not in batch or "thermal_images" not in batch:
                raise ValueError("RGB-T train batch needs image_ids and thermal_images")
            size = batch["pids"].numel()
            captions = batch.get("raw_captions")
            if captions is None or len(captions) != size:
                raise ValueError("raw_captions is required for measured caption rho")
            caption_counts = [caption_token_counts(caption, color_words, stopwords)
                              for caption in captions]
            rho = [record["rho"] for record in caption_counts]
            label_stat = batch_label_statistics(batch["pids"], batch["image_ids"])
            label_records.append(label_stat)
            batch_sizes.append(size)
            clean = model(batch)
            rgb_deg, ir_deg, meta = sampler.sample(
                batch["images"], batch["thermal_images"], batch["image_ids"]
            )
            degraded_batch = dict(batch, images=rgb_deg, thermal_images=ir_deg)
            degraded = model(degraded_batch, mdrc_feature_only=True,
                             p4_clean_features=clean, p4_mode_meta=meta)
            image_feats = degraded["fused_det_feat"].float()
            text_feats = clean["text_det_feat"].float()
            # Match existing SDM feature normalization without adding an epsilon.
            image_norm = image_feats / image_feats.norm(dim=1, keepdim=True)
            text_norm = text_feats / text_feats.norm(dim=1, keepdim=True)
            logits = model.logit_scale.to(device) * (text_norm @ image_norm.t())
            states = _state_names(meta, size)
            variants, details = weight_variants(batch["pids"], rho, states,
                                                kappas, cli.floor, shuffle)
            pairs = pair_distributions(batch["pids"], states, variants, details)
            for state, by_variant in pairs.items():
                destination = accumulated_pairs.setdefault(state, {})
                for name, samples in by_variant.items():
                    target = destination.setdefault(name, {"a": [], "e": [],
                                                          "floor_flags": None})
                    target["a"].extend(samples["a"])
                    target["e"].extend(samples["e"])
                    if samples["floor_flags"] is not None:
                        if target["floor_flags"] is None:
                            target["floor_flags"] = []
                        target["floor_flags"].extend(samples["floor_flags"])
            measurements = gradient_measurements(logits, batch["pids"], variants, params)
            if not all(r["finite"] for v in measurements.values() for r in v.values()):
                raise FloatingPointError(f"Non-finite loss/gradient in batch {index}")
            batch_records.append({
                "batch_index": index, "num_texts": size,
                "unique_image_ids": batch["image_ids"].unique().numel(),
                "states": states, "caption_counts": caption_counts,
                "degradation_strengths": meta["strengths"].detach().cpu().tolist(),
                "degraded_encoder_calls": degraded.get("p4_encoder_calls"),
                "shuffle_permutation": details["permutation"],
                "state_positive_means": details["state_positive_means"],
                "label_statistics": label_stat, "measurements": measurements,
            })
            # Graphs and feature dictionaries must not accumulate across batches.
            del clean, degraded, logits, image_feats, text_feats, image_norm, text_norm
    finally:
        for parameter, required in flags:
            parameter.requires_grad_(required)
    if not batch_records:
        raise ValueError("Training loader produced no batches")
    after = state_fingerprint(model)
    if before != after:
        raise RuntimeError("Read-only diagnostic changed model parameters or buffers")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    output = {
        "schema_version": 1, "checkpoint": str(Path(cli.checkpoint).resolve()),
        "config_file": str(Path(cli.config_file).resolve()), "git_commit": commit,
        "kappa_json": str(Path(cli.kappa_json).resolve()),
        "kappa_json_sha256": hashlib.sha256(Path(cli.kappa_json).read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None,
        "training_args": vars(args), "num_batches_requested": cli.num_batches,
        "num_batches_observed": len(batch_records), "train_seed": train_seed,
        "deg_rng_seed": deg_seed, "shuffle_seed": shuffle_seed,
        "device": device, "precision": "repository_manual_fp16" if device == "cuda" else "cpu_fp32",
        "floor": cli.floor, "kappa": kappas,
        "definitions": {
            "S": "scaled t2i cosine logits; reverse direction uses S.transpose",
            "score_gradient_frobenius": "Frobenius norm of dL/dS for summed bidirectional loss",
            "encoder_parameter_gradient_l2": "L2 norm of concatenated gradients of shared base_model.visual parameters; excludes text and auxiliary heads",
            "weighted_target": "PID_mask*a, independently row-normalized for t2i and transposed i2t; no negative pair gains positive label",
            "E": "state's mean D weight over actual positive pairs within the same batch",
            "D": "max(floor, 1-rho_text*kappa(image_state))",
            "D-shuf": "same D formula after independent batch-local permutation of text rho",
            "effective_e": "a_ij / m_i; m_i is PID-positive image-column count for text row i",
            "floor_rate": "fraction of actual positive pairs where unfloored 1-rho*kappa <= floor",
            "aggregation": "scalar losses/norms: unweighted batch means; a/e/KS: pooled actual positive pairs by state",
            "rng": "train.set_seed(train_seed) initializes the original training loader/augmentation and model global RNGs; degradation and rho shuffle use separate seeded torch.Generators and never read/write the global RNG",
            "freeze": "model.eval, no optimizer and no update; image encoder requires_grad temporarily for autograd.grad, flags restored afterward",
        },
        "parameters_unchanged": True, "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "row_label_statistics": merge_label_statistics(label_records, batch_sizes),
        "batches": batch_records,
    }
    output.update(summarize_batches(batch_records, accumulated_pairs))
    output_dir = Path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "stats.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")
    print(f"{destination}: batches={len(batch_records)}, parameters_unchanged=True")
    return output


if __name__ == "__main__":
    main()
