"""Run bounded real-model none/SDM/CE training without checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config_mapping
from tools.p4.prepare_configs import derive_configs


def local_engineering_config():
    """Explicit local engineering settings; never claim historical A parity."""
    from utils.options import get_args
    from tools.protocol.gen_configs import FORMAL_COMMON_OVERRIDES, arm_overrides
    previous = sys.argv
    try:
        sys.argv = ["train.py"]
        config = vars(get_args()).copy()
    finally:
        sys.argv = previous
    config.update(FORMAL_COMMON_OVERRIDES)
    config.update(arm_overrides(0.2, [0.7284522740054223, 0.2715477259945776])["A"])
    config.update(dataset_name="RGBT-PEDES", root_dir=str(PROJECT_ROOT / "datasets/TextReID"),
                  MLM=True, img_aug=True, sampler="random", num_workers=0,
                  use_rgbt_add_fusion=True, use_mum=False, use_global_uncertainty=False,
                  use_precision_fusion=False, val_dataset="val",
                  dev_split_file=str(PROJECT_ROOT / "splits/dev_190.json"))
    return config


def run_smokes(args):
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output_dir to preserve earlier smoke artifacts")
    output.mkdir(parents=True, exist_ok=True)
    base = local_engineering_config() if args.local_defaults else load_config_mapping(args.config_file)
    if not args.local_defaults:
        # Validate full A before the deliberate small-batch smoke override.
        base = derive_configs(base, base.get("dev_split_file") or "splits/dev_190.json",
                              args.deg_lambda, args.deg_rng_seed)["base_clean"]
    source = "Explicit local engineering settings; historical A configs.yaml unavailable" if args.local_defaults else str(Path(args.config_file).resolve())
    summaries = []
    for name, loss in (("base_clean", "none"), ("c0_sdm", "sdm"), ("c0_ce", "ce")):
        directory = output / name
        directory.mkdir()
        config = dict(base, deg_retrieval_loss=loss, deg_lambda=args.deg_lambda,
                      deg_rng_seed=args.deg_rng_seed, deg_branch_prob=1.0,
                      smoke_steps=args.smoke_steps, batch_size=args.batch_size,
                      num_workers=args.num_workers, name=name,
                      save_last_checkpoint=False, save_all_epoch_ckpt=False,
                      save_metric_best_ckpt=False, save_full_eval_state_ckpt=False,
                      verify_ckpt_immediately_after_save=False)
        if args.root_dir:
            config["root_dir"] = str(Path(args.root_dir).resolve())
        config_path = directory / "smoke_config.yaml"
        serializable = {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()}
        config_path.write_text(yaml.safe_dump(serializable, sort_keys=True), encoding="utf-8")
        command = [sys.executable, "-B", "-u", "train.py", "--config_file", str(config_path),
                   "--output_dir", str(directory), "--smoke_steps", str(args.smoke_steps),
                   "--batch_size", str(args.batch_size), "--MLM"]
        env = dict(os.environ, TRAIN_SEED=str(args.train_seed), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONIOENCODING="utf-8")
        with (directory / "console.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                if "[P4_STEP]" in line or "[P4_SMOKE]" in line:
                    print(f"{name}: {line.strip()}", flush=True)
            returncode = process.wait()
        checkpoint_files = [p for p in directory.rglob("*") if p.suffix in {".pth", ".ckpt"}]
        if checkpoint_files:
            raise RuntimeError(f"Smoke incorrectly saved checkpoints: {checkpoint_files}")
        if returncode:
            (output / "smoke_summary.md").write_text(
                f"# Smoke failed\n\n{name} exited {returncode}; see {directory / 'console.log'}.\n"
                f"Configuration source: {source}.\n", encoding="utf-8")
            raise RuntimeError(f"{name} failed with exit code {returncode}; see console.log")
        records = list(directory.rglob("run_record.json"))
        if len(records) != 1:
            raise RuntimeError(f"Expected one run_record.json, got {len(records)}")
        record = json.loads(records[0].read_text(encoding="utf-8"))
        steps = record.get("steps", [])
        if len(steps) != args.smoke_steps:
            raise RuntimeError(f"Expected {args.smoke_steps} recorded steps, got {len(steps)}")
        summaries.append({"name": name, "record": str(records[0]), "steps": steps,
                          "deg_rng_seed": record.get("deg_rng_seed")})
    lines = ["# Local bounded smoke", "", f"Configuration source: {source}.",
             f"batch_size={args.batch_size}; train_seed={args.train_seed}; deg_lambda={args.deg_lambda}; no checkpoint saved.",
             "", "| configuration | steps | mean L_clean | mean L_deg | all finite | mean seconds/step | peak MiB |",
             "|---|---:|---:|---:|---|---:|---:|"]
    for summary in summaries:
        steps = summary["steps"]
        def average(key):
            return sum(float(s[key]) for s in steps) / len(steps)
        peak = max(int(s["peak_memory_bytes"]) for s in steps) / (1024 ** 2)
        lines.append(f"| {summary['name']} | {len(steps)} | {average('L_clean'):.8f} | "
                     f"{average('L_deg'):.8f} | {all(s['finite'] for s in steps)} | "
                     f"{average('seconds'):.4f} | {peak:.2f} |")
    lines += ["", "Per-step losses, timing, component losses and epoch counts are in the run records:", ""]
    lines += [f"- {s['name']}: {s['record']} (deg_rng_seed={s['deg_rng_seed']})" for s in summaries]
    (output / "smoke_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if not all(step["finite"] for summary in summaries for step in summary["steps"]):
        raise FloatingPointError("Smoke recorded a nonfinite loss or gradient; see smoke_summary.md")
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--config_file")
    sources.add_argument("--local-defaults", dest="local_defaults", action="store_true",
                         help="explicit engineering smoke only; historical A fields are unverified")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--smoke_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_seed", type=int, default=101)
    parser.add_argument("--deg_rng_seed", type=int)
    parser.add_argument("--deg_lambda", type=float, default=1.0)
    parser.add_argument("--root_dir")
    args = parser.parse_args(argv)
    if args.smoke_steps < 1 or args.batch_size < 1:
        parser.error("smoke_steps and batch_size must be positive")
    run_smokes(args)


if __name__ == "__main__":
    main()
