"""Collect a fixed, full-training-epoch teacher reference for A1const.

Run in a separate process before ACD training. The base and data loader are
prepared by train_acd.prepare_training; the injected LoRA remains disabled.
No student encoding, optimizer, backward pass, or development evaluation runs.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.p4.acd_losses import positive_targets
from tools.p4.grad_conflict import preserve_global_rng
from tools.p4.train_acd import prepare_training


def collect_teacher_stats(model, controller, loader, max_batches=0):
    """Compute teacher-only FP32 logits and aggregate row statistics in FP64.

    The caller supplies an eval-mode model. This collector never changes its
    parameters, buffers, requires_grad flags, or existing .grad values. Its RNG
    context also restores random draws made by single-worker augmentations.
    The positive-logit statistic averages uniform positive targets within each
    row first, then averages equally across text rows (not positive pairs).
    """
    if (not isinstance(max_batches, int) or isinstance(max_batches, bool)
            or max_batches < 0):
        raise ValueError("max_batches must be a nonnegative integer")
    if model.training:
        raise ValueError("Teacher statistics require model.eval()")
    expected_batches = len(loader)
    try:
        device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("Teacher model must contain parameters") from error
    reference_rows, reference_b64, positive_rows = [], [], []
    batch_sizes = []
    with preserve_global_rng(), controller.enabled(False), torch.no_grad():
        # Construct the iterator inside the RNG guard: DataLoader iteration can
        # consume a base seed even before the first batch is requested.
        batches = islice(loader, max_batches) if max_batches else loader
        for batch in batches:
            batch = {key: value.to(device) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            # Same frozen complete-column computation and operation order as
            # encode_training_batch; deliberately omit the IR student encoding.
            text = F.normalize(model.encode_text_deterministic(
                batch["caption_ids"]).float(), dim=1)
            rgb, ir, _, _, _, _ = model.encode_rgbt_global(
                batch["images"], batch["thermal_images"])
            fused = F.normalize(rgb.float() + ir.float(), dim=1)
            z_f = 50.0 * (text @ fused.t())
            if (z_f.ndim != 2 or z_f.shape[0] == 0
                    or z_f.shape[0] != z_f.shape[1]
                    or z_f.shape[0] != batch["pids"].numel()):
                raise ValueError("Expected nonempty square teacher batch logits")
            a = torch.logsumexp(z_f, dim=1).detach().cpu().double()
            q = positive_targets(batch["pids"].to(z_f.device), z_f.dtype)
            positive = (q * z_f).sum(dim=1).detach().cpu().double()
            if not torch.isfinite(a).all() or not torch.isfinite(positive).all():
                raise FloatingPointError("Teacher reference contains nonfinite values")
            reference_rows.append(a)
            positive_rows.append(positive)
            size = z_f.shape[0]
            batch_sizes.append(size)
            if size == 64:
                reference_b64.append(a)
    if not reference_rows:
        raise ValueError("Teacher training loader produced no batches")
    values = torch.cat(reference_rows)
    full_values = torch.cat(reference_b64) if reference_b64 else None
    quantiles = torch.quantile(values, values.new_tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    complete = len(batch_sizes) == expected_batches
    return {
        "ref_mean_b64": float(full_values.mean()) if full_values is not None else None,
        "ref_mean_all": float(values.mean()),
        "ref_median_all": float(quantiles[2]),
        "ref_std_all": float(values.std(unbiased=False)),
        "ref_quantiles_all": {name: float(quantiles[index]) for name, index in (
            ("p05", 0), ("p25", 1), ("p75", 3), ("p95", 4))},
        "row_count": values.numel(), "batch_count": len(batch_sizes),
        "full_batch64_row_count": full_values.numel() if full_values is not None else 0,
        "full_batch64_count": len(reference_b64), "batch_sizes": batch_sizes,
        "expected_batch_count": expected_batches, "complete_epoch": complete,
        "statistics_scope": "full_epoch" if complete else "smoke_partial_epoch",
        "valid_as_full_epoch_reference": complete and full_values is not None,
        "teacher_positive_logit_mean": float(torch.cat(positive_rows).mean()),
        "teacher_positive_logit_definition":
            "mean_over_text_rows(sum_j(q_ij * z_F_ij)); uniform same-PID q per row",
        "reference_definition": "a_i=logsumexp_j(50*cos(text_i, rgb_j+ir_j))",
        "reference_precision": "FP32 logits and row logsumexp; CPU FP64 aggregation",
        "standard_deviation_definition": "population, ddof=0",
        "quantile_definition": "linear interpolation; median=quantile(0.5)",
        "missing_full_batch_reference_reason":
            None if full_values is not None else "No batch with exactly 64 text rows",
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_run", required=True,
                        help="Historical base_clean run containing last.pth and configs.yaml")
    parser.add_argument("--root_dir", required=True, help="Relocated dataset root")
    parser.add_argument("--dev_split_file", required=True,
                        help="Verified 1076/190 training/development identity split")
    parser.add_argument("--data_seed", type=int, default=101,
                        help="Same training loader and augmentation seed as train_acd (default: 101)")
    parser.add_argument("--output_json", required=True,
                        help="New JSON path; an existing output is never overwritten")
    parser.add_argument("--num_batches", type=int, default=0,
                        help="0: required full epoch; positive: smoke limit only. A partial-epoch JSON "
                             "is marked incomplete and must not replace the production reference")
    return parser


def main(argv=None):
    cli = build_parser().parse_args(argv)
    if cli.num_batches < 0:
        raise ValueError("num_batches must be nonnegative")
    output = Path(cli.output_json).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {output}")
    started = time.perf_counter()
    record = {
        "schema_version": 1, "tool": "teacher_ref_stats",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True).strip(),
        "arguments": vars(cli),
        "argv": list(sys.argv if argv is None else argv),
        "seeds": {"data_seed": cli.data_seed, "dataloader_seed": cli.data_seed,
                  "lora_init_seed": 20261015},
        "status": "running", "teacher_adapter_enabled": False,
        "student_encodings": 0,
    }
    prepare_cli = Namespace(base_run=cli.base_run, root_dir=cli.root_dir,
                            dev_split_file=cli.dev_split_file, data_seed=cli.data_seed,
                            lora_rank=8, lora_alpha=16, lora_init_seed=20261015)
    record["preparation_arguments"] = vars(prepare_cli)
    with preserve_global_rng():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            model, controller, loader, details = prepare_training(prepare_cli)
            record.update(details)
            model.eval().requires_grad_(False)
            record.update(collect_teacher_stats(model, controller, loader,
                                                max_batches=cli.num_batches))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            record["status"] = "completed" if record["complete_epoch"] else "smoke_completed"
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            record["peak_cuda_memory_allocated_bytes"] = (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0)
            output.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive create also guards against a concurrent writer after
            # the initial existence check. Never rewrite a previous reference.
            with output.open("x", encoding="utf-8") as destination:
                json.dump(record, destination, ensure_ascii=False, indent=2, allow_nan=False)
                destination.write("\n")
    print(json.dumps({name: record[name] for name in (
        "status", "ref_mean_b64", "ref_mean_all", "row_count", "batch_count",
        "complete_epoch", "elapsed_seconds")}, ensure_ascii=False, allow_nan=False), flush=True)
    return record


if __name__ == "__main__":
    main()
