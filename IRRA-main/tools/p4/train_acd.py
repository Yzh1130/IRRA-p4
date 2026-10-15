"""Train an explicitly IR-only LoRA adapter on the frozen development base.

No existing training or evaluation entry point is modified. All four encodings
run in eval mode; only the fourth, IR student encoding, enables the adapter.
"""
from __future__ import annotations

import argparse
import hashlib
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
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.p4.acd_lora import file_sha256, inject_lora, save_adapter
from tools.p4.acd_losses import acd_loss


def worker_init_seed(worker_id):
    """DataLoader's private generator supplies initial_seed, not global RNG."""
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clone_training_loader(loader, seed):
    """Keep the repository dataset/transforms/collate and sampler semantics."""
    from torch.utils.data import DataLoader, RandomSampler
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sampler = loader.sampler
    if isinstance(sampler, RandomSampler):
        sampler = RandomSampler(loader.dataset, replacement=sampler.replacement,
                                num_samples=sampler.num_samples, generator=generator)
    options = dict(dataset=loader.dataset, batch_size=loader.batch_size,
                   sampler=sampler, num_workers=loader.num_workers,
                   collate_fn=loader.collate_fn, drop_last=loader.drop_last,
                   pin_memory=loader.pin_memory, timeout=loader.timeout,
                   generator=generator, worker_init_fn=worker_init_seed)
    if loader.num_workers:
        options.update(prefetch_factor=loader.prefetch_factor,
                       persistent_workers=loader.persistent_workers)
    return DataLoader(**options)


def base_state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if ".acd_lora." in name:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().reshape(-1)
                      .view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def batch_hashes(batch, step):
    def digest_tensor(tensor, algorithm):
        return hashlib.new(algorithm, tensor.detach().cpu().contiguous().reshape(-1)
                           .view(torch.uint8).numpy().tobytes()).hexdigest()
    image_ids = batch["image_ids"].detach().cpu().reshape(-1).tolist()
    return {"step": step, "image_ids_sha1": hashlib.sha1(
        json.dumps(image_ids, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "rgb_sha256": digest_tensor(batch["images"], "sha256"),
        "ir_sha256": digest_tensor(batch["thermal_images"], "sha256")}


def encode_training_batch(model, controller, batch):
    """Exactly one text/RGB/IR teacher encoding and one IR student encoding."""
    with controller.enabled(False), torch.no_grad():
        text = F.normalize(model.encode_text_deterministic(batch["caption_ids"]).float(), dim=1)
        rgb, ir, _, _, _, _ = model.encode_rgbt_global(
            batch["images"], batch["thermal_images"])
        fused = F.normalize(rgb.float() + ir.float(), dim=1)
    with controller.enabled():
        student = F.normalize(model.encode_image(
            batch["thermal_images"], modality="thermal").float(), dim=1)
    # The base's temperature=0.02 is fixed, not a trained scale.
    return 50.0 * (text @ fused.t()), 50.0 * (text @ student.t())


def learning_rate_factor(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_epochs(model, controller, loader, cli, output, record):
    """Shared core also exercised by synthetic tests; no development selection."""
    model.eval()
    parameters = list(controller.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cli.lr, weight_decay=0.0)
    total_steps = cli.epochs * len(loader)
    if not total_steps:
        raise ValueError("Training loader contains no batches")
    target_steps = min(total_steps, cli.smoke_steps) if cli.smoke_steps else total_steps
    before = base_state_sha256(model)
    record.update(base_state_sha256_before=before, planned_total_steps=total_steps,
                  target_steps=target_steps, steps=0, epochs=[],
                  encoding_calls_per_step={"text_teacher": 1, "rgb_teacher": 1,
                                           "ir_teacher": 1, "ir_student": 1},
                  encodings_total={"text_teacher": 0, "rgb_teacher": 0,
                                   "ir_teacher": 0, "ir_student": 0})
    device = parameters[0].device
    step = 0
    with (output / "step_log.jsonl").open("w", encoding="utf-8") as steps_file, \
         (output / "step_hashes.txt").open("w", encoding="utf-8") as hashes_file, \
         (output / "train_log.txt").open("w", encoding="utf-8") as log_file:
        for epoch in range(cli.epochs):
            epoch_start = time.perf_counter()
            loss_sum, batches = 0.0, 0
            for batch in loader:
                if step >= target_steps:
                    break
                if step < 20:
                    hashes_file.write(json.dumps(batch_hashes(batch, step + 1), sort_keys=True) + "\n")
                    hashes_file.flush()
                batch = {key: value.to(device) if torch.is_tensor(value) else value
                         for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                lr = cli.lr * learning_rate_factor(step, total_steps, len(loader))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                z_f, z_i = encode_training_batch(model, controller, batch)
                loss, diagnostics = acd_loss(z_f.float(), z_i.float(), batch["pids"],
                                             cli.variant, cli.a1_impl)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite ACD loss at step {step + 1}")
                loss.backward()
                if any(parameter.grad is not None for name, parameter in model.named_parameters()
                       if ".acd_lora." not in name):
                    raise RuntimeError("A frozen base parameter acquired a gradient")
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                loss_sum += float(loss.detach())
                batches += 1
                for key in record["encodings_total"]:
                    record["encodings_total"][key] += 1
                if step <= 20 or step % 10 == 0:
                    values = {name: float(value) if torch.isfinite(value) else None
                              for name, value in diagnostics.items()}
                    values.update(step=step, epoch=epoch + 1, lr=lr,
                                  lora_gradient_norm=float(grad_norm),
                                  lora_parameter_norm=math.sqrt(sum(float(
                                      p.detach().double().square().sum()) for p in parameters)))
                    steps_file.write(json.dumps(values, allow_nan=False) + "\n")
                    steps_file.flush()
                record["steps"] = step
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - epoch_start
            summary = dict(epoch=epoch + 1, steps=batches,
                           loss_mean=loss_sum / batches if batches else None,
                           lr=optimizer.param_groups[0]["lr"], elapsed_seconds=elapsed)
            record["epochs"].append(summary)
            line = json.dumps(summary, allow_nan=False)
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()
            if step >= target_steps:
                break
    after = base_state_sha256(model)
    record.update(base_state_sha256_after=after, base_state_unchanged=before == after)
    if before != after:
        raise RuntimeError("Frozen base state changed during ACD training")
    save_adapter(controller, output / "adapter.pt", record["base_checkpoint_sha256"],
                 {"variant": cli.variant, "a1_impl": cli.a1_impl, "steps": step,
                  "epochs_requested": cli.epochs, "git_commit": record.get("git_commit")})
    return record


def prepare_training(cli):
    from datasets import build_dataloader
    from model import build_model
    from train import set_seed
    from tools.p4.train_batch_stats import _load_training_args
    from tools.p4.export_dev_records import (
        _resolve_annotation, _resolve_saved_split, load_strict_checkpoint,
        validate_dev_split)
    base_run = Path(cli.base_run).resolve()
    checkpoint, config = base_run / "last.pth", base_run / "configs.yaml"
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError("base_run must contain last.pth and configs.yaml")
    args = _load_training_args(config)
    if args.dataset_name != "RGBT-PEDES" or args.batch_size != 64:
        raise ValueError("ACD requires the historical RGBT-PEDES batch-64 training configuration")
    if not bool(getattr(args, "use_rgbt_add_fusion", False)):
        raise ValueError("ACD requires additive base fusion")
    if args.sampler != "random":
        raise ValueError("ACD requires the historical base sampler=random")
    # Historical batch/ID uncertainty switches are inert without use_mum and
    # remain inherited; all other optional use_* routes must be disabled.
    forbidden = {"mraf_enabled", "mdrc_enable"} | {
        name for name in vars(args) if name.startswith("use_") and name not in {
            "use_rgbt_add_fusion", "use_batch_uncertainty", "use_id_uncertainty"}}
    for flag in sorted(forbidden):
        if bool(getattr(args, flag, False)):
            raise ValueError(f"ACD requires the frozen plain additive base: {flag}")
    if str(getattr(args, "deg_retrieval_loss", "none")) != "none":
        raise ValueError("base_run must be base_clean, without degradation retrieval")
    if not math.isclose(float(args.temperature), 0.02, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("ACD requires the historical fixed temperature=0.02")
    args.root_dir = str(Path(cli.root_dir).resolve())
    requested_split = Path(cli.dev_split_file).resolve()
    annotation = _resolve_annotation(args.root_dir)
    # Compare identity membership and annotation hash before replacing the
    # checkpoint's recorded split path with a relocated command-line path.
    saved_split, saved_split_recorded = _resolve_saved_split(args, config, None)
    split = validate_dev_split(annotation, requested_split)
    saved_split_data = validate_dev_split(annotation, saved_split)
    if split["split_identity_sha256"] != saved_split_data["split_identity_sha256"]:
        raise ValueError("Requested development split differs from base checkpoint training split")
    if split["n_train"] != 1076 or split["n_dev"] != 190:
        raise ValueError("ACD requires the verified 1076/190 identity development split")
    args.dev_split_file = str(requested_split)
    # Build the full original checkpoint structure before changing loader-only
    # controls; loss_names stays intact so MLM/ID weights load strictly.
    set_seed(cli.data_seed)
    args.training, args.distributed = True, False
    args.MLM = False
    args.mdrc_enable = False
    args.degrade_aug_mode = "none"
    args.deg_retrieval_loss = "none"
    loader, _, _, num_classes = build_dataloader(args)
    if num_classes != 1076:
        raise ValueError("Repository loader did not construct the 1076-ID training subset")
    loader = clone_training_loader(loader, cli.data_seed)
    model = build_model(args, num_classes)
    load_strict_checkpoint(model, checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("Real ACD training requires CUDA; synthetic CPU tests remain supported")
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    controller = inject_lora(model, cli.lora_rank, cli.lora_alpha, cli.lora_init_seed)
    return model, controller, loader, {
        "base_checkpoint": str(checkpoint), "base_checkpoint_sha256": file_sha256(checkpoint),
        "base_config": str(config), "base_config_sha256": file_sha256(config),
        "effective_training_args": vars(args), "development_split": split,
        "base_training_split_recorded": saved_split_recorded,
        "base_training_split_verified": saved_split_data["split"],
        "sampler_type": type(loader.sampler).__name__,
        "training_batch_size": loader.batch_size, "training_num_classes": num_classes,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_run", required=True)
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--dev_split_file", required=True)
    parser.add_argument("--variant", required=True, choices=("A0", "A0bal", "A1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_init_seed", type=int, default=20261015)
    parser.add_argument("--data_seed", type=int, default=101)
    parser.add_argument("--a1_impl", choices=("joint", "decomposed"), default="joint")
    parser.add_argument("--smoke_steps", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    return parser


def main(argv=None):
    cli = build_parser().parse_args(argv)
    if cli.epochs <= 0 or cli.smoke_steps < 0 or cli.lr <= 0 or not math.isfinite(cli.lr):
        raise ValueError("epochs and lr must be positive; smoke_steps must be nonnegative")
    output = Path(cli.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT,
                                     text=True).strip()
    record = dict(schema_version=1, tool="train_acd", git_commit=commit,
                  arguments=vars(cli), argv=list(sys.argv if argv is None else argv),
                  seeds={"data_seed": cli.data_seed, "dataloader_seed": cli.data_seed,
                         "lora_init_seed": cli.lora_init_seed}, status="running",
                  precision="native frozen teacher and visual student / FP32 LoRA weight delta and loss",
                  checkpoint_selection="last adapter only; no development evaluation")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        model, controller, loader, details = prepare_training(cli)
        record.update(details)
        train_epochs(model, controller, loader, cli, output, record)
        record["status"] = "smoke_completed" if cli.smoke_steps else "completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        record["peak_cuda_memory_allocated_bytes"] = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0)
        (output / "run_record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return record


if __name__ == "__main__":
    main()
