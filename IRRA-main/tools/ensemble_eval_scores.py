import argparse
import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.metrics import rank  # noqa: E402


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_score_file(path):
    data = _torch_load_cpu(path)
    if not isinstance(data, dict):
        raise ValueError(f"Score file must contain a dict: {path}")
    required = ["sim", "txt_pids", "img_pids", "order_hash"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Score file {path} is missing keys: {missing}")

    sim = data["sim"]
    txt_pids = data["txt_pids"]
    img_pids = data["img_pids"]
    if not torch.is_tensor(sim) or sim.dim() != 2:
        raise ValueError(f"Score file {path} has invalid sim shape.")
    if not torch.is_tensor(txt_pids) or txt_pids.view(-1).numel() != sim.shape[0]:
        raise ValueError(f"Score file {path} has invalid txt_pids.")
    if not torch.is_tensor(img_pids) or img_pids.view(-1).numel() != sim.shape[1]:
        raise ValueError(f"Score file {path} has invalid img_pids.")

    return {
        "path": path,
        "sim": sim.detach().cpu().float(),
        "txt_pids": txt_pids.detach().cpu().long().view(-1),
        "img_pids": img_pids.detach().cpu().long().view(-1),
        "order_hash": str(data["order_hash"]),
        "metrics": data.get("metrics", {}),
        "metadata": data.get("metadata", {}),
    }


def _validate_score_files(items):
    ref = items[0]
    for item in items[1:]:
        if tuple(item["sim"].shape) != tuple(ref["sim"].shape):
            raise ValueError(
                "Score shape mismatch: "
                f"{item['path']} has {tuple(item['sim'].shape)}, "
                f"expected {tuple(ref['sim'].shape)}"
            )
        if item["order_hash"] != ref["order_hash"]:
            raise ValueError(
                "order_hash mismatch: "
                f"{item['path']} has {item['order_hash']}, "
                f"expected {ref['order_hash']}"
            )
        if not torch.equal(item["txt_pids"], ref["txt_pids"]):
            raise ValueError(f"txt_pids mismatch in {item['path']}")
        if not torch.equal(item["img_pids"], ref["img_pids"]):
            raise ValueError(f"img_pids mismatch in {item['path']}")


def _zscore(sim, dim, eps):
    mean = sim.mean(dim=dim, keepdim=True)
    std = sim.std(dim=dim, unbiased=False, keepdim=True).clamp_min(eps)
    return (sim - mean) / std


def _normalize(sim, mode, eps):
    mode = str(mode).lower()
    if mode == "none":
        return sim.float()
    if mode == "row_zscore":
        return _zscore(sim.float(), dim=1, eps=eps)
    if mode == "row_col_zscore":
        return _zscore(_zscore(sim.float(), dim=1, eps=eps), dim=0, eps=eps)
    raise ValueError(f"Unsupported norm={mode!r}")


def _compute_metrics(sim, txt_pids, img_pids):
    cmc, map_value, minp, _ = rank(
        similarity=sim,
        q_pids=txt_pids,
        g_pids=img_pids,
        max_rank=10,
        get_mAP=True,
    )
    return {
        "R1": float(cmc[0].item()),
        "R5": float(cmc[4].item()),
        "R10": float(cmc[9].item()),
        "mAP": float(map_value.item()),
        "mINP": float(minp.item()),
    }


def _parse_weights(raw_weights, num_files):
    if raw_weights is None:
        weights = [1.0] * num_files
    else:
        weights = [float(weight) for weight in raw_weights]
    if len(weights) != num_files:
        raise ValueError(
            f"--weights length must match --score_files: {len(weights)} vs {num_files}"
        )
    if not all(torch.isfinite(torch.tensor(weights)).tolist()):
        raise ValueError("--weights must be finite numbers.")
    weight_sum = float(sum(weights))
    if abs(weight_sum) < 1e-12:
        raise ValueError("--weights must not sum to zero.")
    return weights


def _score_semantics(metadata):
    semantics = metadata.get("score_semantics", "")
    return str(semantics) if semantics is not None else ""


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble dumped IRRA eval score matrices."
    )
    parser.add_argument(
        "--score_files",
        nargs="+",
        required=True,
        help="input .pt score files produced by test.py --save_eval_scores",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="per-file ensemble weights; defaults to all ones",
    )
    parser.add_argument(
        "--norm",
        choices=["none", "row_zscore", "row_col_zscore"],
        default="row_zscore",
        help="per-score normalization before weighted averaging",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="output .pt file for the ensemble result",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="minimum std used by zscore normalization",
    )
    args = parser.parse_args()

    if args.eps <= 0:
        raise ValueError("--eps must be > 0.")

    score_files = [os.path.abspath(path) for path in args.score_files]
    items = [_load_score_file(path) for path in score_files]
    if not items:
        raise ValueError("--score_files must not be empty.")
    _validate_score_files(items)

    weights = _parse_weights(args.weights, len(items))
    normalized_sims = [
        _normalize(item["sim"], mode=args.norm, eps=args.eps)
        for item in items
    ]
    weight_tensor = torch.tensor(weights, dtype=torch.float32).view(-1, 1, 1)
    stacked = torch.stack(normalized_sims, dim=0)
    ensemble_sim = (stacked * weight_tensor).sum(dim=0) / weight_tensor.sum()

    txt_pids = items[0]["txt_pids"]
    img_pids = items[0]["img_pids"]
    metrics = _compute_metrics(ensemble_sim, txt_pids, img_pids)
    semantics = [_score_semantics(item["metadata"]) for item in items]
    unique_semantics = sorted(set(semantics))

    output = {
        "sim": ensemble_sim.float(),
        "txt_pids": txt_pids.clone(),
        "img_pids": img_pids.clone(),
        "txt_indices": torch.arange(ensemble_sim.shape[0], dtype=torch.long),
        "img_indices": torch.arange(ensemble_sim.shape[1], dtype=torch.long),
        "weights": weights,
        "normalized_weights": [
            float(weight / sum(weights)) for weight in weights
        ],
        "norm": str(args.norm),
        "metrics": metrics,
        "source_files": score_files,
        "source_order_hash": items[0]["order_hash"],
        "source_score_semantics": semantics,
        "source_metadata": [item["metadata"] for item in items],
        "metadata": {
            "score_semantics": "ensemble_weighted_average",
            "score_shape": list(ensemble_sim.shape),
            "norm": str(args.norm),
            "eps": float(args.eps),
            "num_sources": len(items),
        },
    }

    output_file = os.path.abspath(args.output_file)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(output, output_file)

    if len(unique_semantics) > 1:
        print(
            "Warning: mixed source score_semantics: "
            + ", ".join(unique_semantics)
        )
    print(
        "Ensemble result: "
        f"R1={metrics['R1']:.3f} "
        f"R5={metrics['R5']:.3f} "
        f"R10={metrics['R10']:.3f} "
        f"mAP={metrics['mAP']:.3f} "
        f"mINP={metrics['mINP']:.3f}"
    )
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
