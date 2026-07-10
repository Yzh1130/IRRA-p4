"""Strict equal-weight score fusion with an explicitly separated oracle mode."""

import argparse
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.revision.score_io import (  # noqa: E402
    METRIC_KEYS,
    compute_retrieval_metrics,
    fuse_score_items,
    load_score_file,
    prepare_output_parents,
    report_paths,
    require_outputs_absent,
    require_score_consistency,
    write_json,
    write_markdown,
    write_tsv,
)


def build_ensemble_payload(
    score_files,
    matrix_key="sim",
    fusion_method="row_zscore_equal",
    epsilon=1e-6,
    oracle_weights=None,
):
    if not score_files:
        raise ValueError("--score-files must contain at least one score file.")
    items = [load_score_file(path, matrix_key=matrix_key) for path in score_files]
    if len(items) > 1:
        require_score_consistency(items)

    oracle_only = oracle_weights is not None
    if oracle_only and (len(items) != 7 or len(oracle_weights) != 7):
        raise ValueError(
            "--oracle-weights is reserved for the fixed seven-model oracle upper "
            "bound and requires exactly 7 score files and 7 weights."
        )
    fused, normalized_weights = fuse_score_items(
        items,
        method=fusion_method,
        epsilon=epsilon,
        weights=oracle_weights,
    )
    metrics = compute_retrieval_metrics(
        fused, items[0]["txt_pids"], items[0]["img_pids"]
    )
    equal_weights = oracle_weights is None
    effective_method = (
        f"{fusion_method}_oracle_weighted" if oracle_only else fusion_method
    )
    metadata = {
        "fusion_method": effective_method,
        "base_fusion_method": fusion_method,
        "matrix_key": matrix_key,
        "model_count": len(items),
        "equal_weights": equal_weights,
        "normalized_weights": normalized_weights,
        "input_files": [item["path"] for item in items],
        "epsilon": float(epsilon),
        "oracle_only": oracle_only,
        "test_set_tuned_or_preselected_weights": oracle_only,
        "score_semantics": "offline_score_ensemble",
        "score_shape": list(fused.shape),
    }
    payload = {
        "sim": fused.clone(),
        "raw_sim": fused.clone(),
        "txt_pids": items[0]["txt_pids"].clone(),
        "img_pids": items[0]["img_pids"].clone(),
        "txt_indices": items[0]["txt_indices"].clone(),
        "img_indices": items[0]["img_indices"].clone(),
        "metrics": metrics,
        "metadata": metadata,
        "order_hash": items[0]["order_hash"],
    }
    return payload


def _markdown_summary(payload):
    metrics = payload["metrics"]
    metadata = payload["metadata"]
    lines = [
        "# Score ensemble summary",
        "",
        f"- Fusion method: `{metadata['fusion_method']}`",
        f"- Model count: {metadata['model_count']}",
        f"- Equal weights: {str(metadata['equal_weights']).lower()}",
        f"- Oracle only: {str(metadata['oracle_only']).lower()}",
        f"- Matrix key: `{metadata['matrix_key']}`",
        f"- Epsilon: {metadata['epsilon']}",
        "",
        "| R1 | R5 | R10 | mAP | mINP |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| " + " | ".join(f"{metrics[key]:.6f}" for key in METRIC_KEYS) + " |",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in metadata["input_files"])
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fuse strictly aligned score files. Formal default is row-wise "
            "z-score followed by equal averaging."
        )
    )
    parser.add_argument("--score-files", nargs="+", required=True)
    parser.add_argument("--matrix-key", choices=["sim", "raw_sim"], default="sim")
    parser.add_argument(
        "--fusion-method",
        choices=["raw_equal", "row_zscore_equal"],
        default="row_zscore_equal",
    )
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--oracle-weights",
        nargs="+",
        type=float,
        default=None,
        help="seven fixed weights for an explicitly labelled oracle upper bound",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_prefix = os.path.abspath(args.output_prefix)
    is_oracle = args.oracle_weights is not None
    has_oracle_label = "oracle_upper_bound" in os.path.basename(output_prefix).lower()
    if is_oracle and not has_oracle_label:
        raise ValueError(
            "Oracle output prefix must contain 'oracle_upper_bound' so it cannot be "
            "mistaken for a formal result."
        )
    if not is_oracle and has_oracle_label:
        raise ValueError(
            "Formal equal-weight output prefix must not contain 'oracle_upper_bound'."
        )

    payload = build_ensemble_payload(
        args.score_files,
        matrix_key=args.matrix_key,
        fusion_method=args.fusion_method,
        epsilon=args.epsilon,
        oracle_weights=args.oracle_weights,
    )
    paths = report_paths(output_prefix, include_score=True)
    require_outputs_absent(paths.values())
    prepare_output_parents(paths.values())
    torch.save(payload, paths["score"])
    summary = {
        "metrics": payload["metrics"],
        "metadata": payload["metadata"],
        "order_hash": payload["order_hash"],
        "outputs": paths,
    }
    write_json(paths["json"], summary)
    row = {**payload["metadata"], **payload["metrics"]}
    fields = [
        "fusion_method",
        "model_count",
        "equal_weights",
        "oracle_only",
        "matrix_key",
        "epsilon",
        *METRIC_KEYS,
    ]
    write_tsv(paths["tsv"], [row], fields)
    write_markdown(paths["markdown"], _markdown_summary(payload))
    print(f"Saved ensemble score: {paths['score']}")
    for key in METRIC_KEYS:
        print(f"{key}={payload['metrics'][key]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
