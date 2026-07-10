"""Element-wise comparison for two IRRA score payloads."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.revision.score_io import (  # noqa: E402
    load_score_file,
    require_outputs_absent,
    score_consistency_report,
    write_json,
)


def _metrics_difference(metrics_a, metrics_b):
    result = {}
    for key in sorted(set(metrics_a) | set(metrics_b)):
        value_a = metrics_a.get(key)
        value_b = metrics_b.get(key)
        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            result[key] = {
                "file_a": float(value_a),
                "file_b": float(value_b),
                "difference_b_minus_a": float(value_b) - float(value_a),
            }
        elif value_a != value_b:
            result[key] = {"file_a": value_a, "file_b": value_b}
    return result


def compare_score_files(file_a, file_b, matrix_key="sim", rtol=0.0, atol=0.0):
    rtol = float(rtol)
    atol = float(atol)
    if not math.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"rtol must be finite and >= 0, got {rtol!r}.")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError(f"atol must be finite and >= 0, got {atol!r}.")
    item_a = load_score_file(file_a, matrix_key=matrix_key)
    item_b = load_score_file(file_b, matrix_key=matrix_key)
    consistency = score_consistency_report([item_a, item_b])
    if consistency["score_shape"]:
        difference = (item_a["matrix"] - item_b["matrix"]).abs()
        max_abs_diff = float(difference.max().item())
        mean_abs_diff = float(difference.mean().item())
        median_abs_diff = float(torch.quantile(difference.reshape(-1), 0.5).item())
        exact_equal = bool(item_a["matrix"].equal(item_b["matrix"]))
        allclose = bool(
            item_a["matrix"].allclose(item_b["matrix"], rtol=rtol, atol=atol)
        )
    else:
        max_abs_diff = None
        mean_abs_diff = None
        median_abs_diff = None
        exact_equal = False
        allclose = False
    matched = bool(consistency["compatible"] and allclose)
    return {
        "file_a": item_a["path"],
        "file_b": item_b["path"],
        "matrix_key": matrix_key,
        "rtol": rtol,
        "atol": atol,
        "shape_a": list(item_a["matrix"].shape),
        "shape_b": list(item_b["matrix"].shape),
        "shape_consistent": consistency["score_shape"],
        "txt_pids_consistent": consistency["txt_pids"],
        "img_pids_consistent": consistency["img_pids"],
        "txt_indices_consistent": consistency["txt_indices"],
        "img_indices_consistent": consistency["img_indices"],
        "order_hash_consistent": consistency["order_hash"],
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "median_abs_diff": median_abs_diff,
        "exact_equal": exact_equal,
        "allclose": allclose,
        "metrics_difference": _metrics_difference(
            item_a["metrics"], item_b["metrics"]
        ),
        "matched": matched,
        "mismatches": consistency["mismatches"],
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare score matrices and all ordering fields element by element."
    )
    parser.add_argument("--file-a", required=True)
    parser.add_argument("--file-b", required=True)
    parser.add_argument("--matrix-key", choices=["sim", "raw_sim"], default="sim")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--output-json", default="")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = compare_score_files(
        args.file_a,
        args.file_b,
        matrix_key=args.matrix_key,
        rtol=args.rtol,
        atol=args.atol,
    )
    if args.output_json:
        output_json = os.path.abspath(args.output_json)
        require_outputs_absent([output_json])
        parent = os.path.dirname(output_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        write_json(output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
