#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "proto2_scratch_self005_cross000_align000_60",
        "mode": "scratch",
        "self_v": "0.05",
        "self_t": "0.05",
        "cross_t2v": "0",
        "cross_v2t": "0",
        "align": "0",
        "temp": "0.07",
        "momentum": "0.9",
        "epochs": "60",
        "lr": "1e-5",
    },
    {
        "name": "proto2_scratch_self005_cross005_align000_60",
        "mode": "scratch",
        "self_v": "0.05",
        "self_t": "0.05",
        "cross_t2v": "0.05",
        "cross_v2t": "0.05",
        "align": "0",
        "temp": "0.07",
        "momentum": "0.9",
        "epochs": "60",
        "lr": "1e-5",
    },
    {
        "name": "proto2_scratch_self005_cross005_align001_60",
        "mode": "scratch",
        "self_v": "0.05",
        "self_t": "0.05",
        "cross_t2v": "0.05",
        "cross_v2t": "0.05",
        "align": "0.01",
        "temp": "0.07",
        "momentum": "0.9",
        "epochs": "60",
        "lr": "1e-5",
    },
    {
        "name": "proto2_scratch_self010_cross005_align001_60",
        "mode": "scratch",
        "self_v": "0.10",
        "self_t": "0.10",
        "cross_t2v": "0.05",
        "cross_v2t": "0.05",
        "align": "0.01",
        "temp": "0.07",
        "momentum": "0.9",
        "epochs": "60",
        "lr": "1e-5",
    },
    {
        "name": "proto2_warm_self005_cross000_align000_10_lr1e6",
        "mode": "warm",
        "self_v": "0.05",
        "self_t": "0.05",
        "cross_t2v": "0",
        "cross_v2t": "0",
        "align": "0",
        "temp": "0.07",
        "momentum": "0.9",
        "epochs": "10",
        "lr": "1e-6",
    },
    {
        "name": "proto2_warm_self005_cross005_align000_10_lr1e6",
        "mode": "warm",
        "self_v": "0.05",
        "self_t": "0.05",
        "cross_t2v": "0.05",
        "cross_v2t": "0.05",
        "align": "0",
        "temp": "0.07",
        "momentum": "0.9",
        "epochs": "10",
        "lr": "1e-6",
    },
    {
        "name": "proto2_warm_self005_cross005_align001_10_lr1e6",
        "mode": "warm",
        "self_v": "0.05",
        "self_t": "0.05",
        "cross_t2v": "0.05",
        "cross_v2t": "0.05",
        "align": "0.01",
        "temp": "0.07",
        "momentum": "0.9",
        "epochs": "10",
        "lr": "1e-6",
    },
    {
        "name": "proto2_warm_self010_cross005_align001_10_lr1e6",
        "mode": "warm",
        "self_v": "0.10",
        "self_t": "0.10",
        "cross_t2v": "0.05",
        "cross_v2t": "0.05",
        "align": "0.01",
        "temp": "0.07",
        "momentum": "0.9",
        "epochs": "10",
        "lr": "1e-6",
    },
]

PROTO_KEYS = [
    "loss_proto_total",
    "loss_proto_self_v",
    "loss_proto_self_t",
    "loss_proto_cross_t2v",
    "loss_proto_cross_v2t",
    "loss_proto_align",
]

FIELDNAMES = [
    "name",
    "mode",
    "self_v",
    "self_t",
    "cross_t2v",
    "cross_v2t",
    "align",
    "temp",
    "momentum",
    "epochs",
    "lr",
    "best_R1",
    "best_epoch",
    *PROTO_KEYS,
    "ended",
]


def find_latest_train_log(run_dir: Path):
    logs = [
        Path(path)
        for path in glob.glob(str(run_dir / "**" / "train_log.txt"), recursive=True)
    ]
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def last_metric(lines, key):
    pattern = re.compile(rf"{re.escape(key)}:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
    value = ""
    for line in lines:
        match = pattern.search(line)
        if match:
            value = match.group(1)
    return value


def parse_best(lines):
    best_r1 = ""
    best_epoch = ""
    pattern = re.compile(r"best R1:\s*([0-9.]+)\s+at epoch\s+([0-9]+)")
    for line in lines:
        match = pattern.search(line)
        if match:
            best_r1 = match.group(1)
            best_epoch = match.group(2)
    return best_r1, best_epoch


def summarize_one(runs_root: Path, meta):
    row = dict(meta)
    run_dir = runs_root / meta["name"]
    log_path = find_latest_train_log(run_dir)
    if log_path is None:
        row.update({"best_R1": "", "best_epoch": "", "ended": "false"})
        for key in PROTO_KEYS:
            row[key] = ""
        return row

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    best_r1, best_epoch = parse_best(lines)
    row["best_R1"] = best_r1
    row["best_epoch"] = best_epoch
    for key in PROTO_KEYS:
        row[key] = last_metric(lines, key)
    row["ended"] = "true" if best_r1 else "false"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", default="pack/runs")
    parser.add_argument("--output", default="summary.csv")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    rows = [summarize_one(runs_root, meta) for meta in EXPERIMENTS]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {output}")
    for row in rows:
        print(
            "{name}\tmode={mode}\tbest_R1={best_R1}\tbest_epoch={best_epoch}\t"
            "loss_proto_total={loss_proto_total}\tended={ended}".format(**row)
        )


if __name__ == "__main__":
    main()
