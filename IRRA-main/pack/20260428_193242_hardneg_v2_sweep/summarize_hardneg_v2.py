#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "v2_smoke_l005_m005_top3_temp007_2",
        "mode": "smoke",
        "lambda_hn": "0.05",
        "margin": "0.05",
        "topn": "3",
        "temperature": "0.07",
        "direction": "both",
        "start": "1",
        "epochs": "2",
        "lr": "1e-5",
    },
    {
        "name": "v2_scratch_l005_m005_top3_temp007_60",
        "mode": "scratch",
        "lambda_hn": "0.05",
        "margin": "0.05",
        "topn": "3",
        "temperature": "0.07",
        "direction": "both",
        "start": "5",
        "epochs": "60",
        "lr": "1e-5",
    },
    {
        "name": "v2_scratch_l010_m005_top3_temp007_60",
        "mode": "scratch",
        "lambda_hn": "0.10",
        "margin": "0.05",
        "topn": "3",
        "temperature": "0.07",
        "direction": "both",
        "start": "5",
        "epochs": "60",
        "lr": "1e-5",
    },
    {
        "name": "v2_scratch_l005_m010_top3_temp007_60",
        "mode": "scratch",
        "lambda_hn": "0.05",
        "margin": "0.10",
        "topn": "3",
        "temperature": "0.07",
        "direction": "both",
        "start": "5",
        "epochs": "60",
        "lr": "1e-5",
    },
    {
        "name": "v2_warm_l005_m005_top3_temp007_10_lr1e6",
        "mode": "warmstart",
        "lambda_hn": "0.05",
        "margin": "0.05",
        "topn": "3",
        "temperature": "0.07",
        "direction": "both",
        "start": "1",
        "epochs": "10",
        "lr": "1e-6",
    },
    {
        "name": "v2_warm_l010_m005_top3_temp007_10_lr1e6",
        "mode": "warmstart",
        "lambda_hn": "0.10",
        "margin": "0.05",
        "topn": "3",
        "temperature": "0.07",
        "direction": "both",
        "start": "1",
        "epochs": "10",
        "lr": "1e-6",
    },
    {
        "name": "v2_warm_l005_m010_top3_temp007_10_lr1e6",
        "mode": "warmstart",
        "lambda_hn": "0.05",
        "margin": "0.10",
        "topn": "3",
        "temperature": "0.07",
        "direction": "both",
        "start": "1",
        "epochs": "10",
        "lr": "1e-6",
    },
]


FIELDNAMES = [
    "name",
    "mode",
    "lambda_hn",
    "margin",
    "topn",
    "temperature",
    "direction",
    "start",
    "epochs",
    "lr",
    "best_R1",
    "best_epoch",
    "ended",
]


BEST_RE = re.compile(r"best R1:\s*([0-9.+\-eE]+)\s+at epoch\s+([0-9.+\-eE]+)")


def latest_train_log(run_dir):
    logs = list(run_dir.glob("**/train_log.txt"))
    if not logs:
        return None
    return max(logs, key=lambda path: path.stat().st_mtime)


def parse_train_log(path):
    if path is None:
        return "", "", False
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = BEST_RE.findall(text)
    if not matches:
        return "", "", False
    best_r1, best_epoch = matches[-1]
    tail = text[-10000:]
    ended = ("Traceback (most recent call last)" not in tail) and ("RuntimeError:" not in tail)
    return best_r1, best_epoch, ended


def render_table(rows):
    widths = {
        field: max(len(field), *(len(str(row[field])) for row in rows))
        for field in FIELDNAMES
    }
    header = " | ".join(field.ljust(widths[field]) for field in FIELDNAMES)
    sep = "-+-".join("-" * widths[field] for field in FIELDNAMES)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row[field]).ljust(widths[field]) for field in FIELDNAMES))


def main():
    pack_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
    runs_dir = pack_dir / "runs"
    rows = []
    for exp in EXPERIMENTS:
        row = dict(exp)
        log_path = latest_train_log(runs_dir / exp["name"])
        best_r1, best_epoch, ended = parse_train_log(log_path)
        row["best_R1"] = best_r1
        row["best_epoch"] = best_epoch
        row["ended"] = str(bool(ended))
        rows.append(row)

    summary_csv = pack_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"summary_csv={summary_csv}")
    render_table(rows)


if __name__ == "__main__":
    main()
