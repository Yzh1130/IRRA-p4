"""
weight_split_eval.py

Split-half weighted fusion evaluation for multiple random-seed IRRA models.

Given N (default 7) per-model text-to-image similarity score files produced by
`test.py --save_eval_scores --eval_scores_file ...` (each a .pt dict with keys
`sim` [Nt x Ni], `txt_pids` [Nt], `img_pids` [Ni], `order_hash`, ...), this
script:

  (1) Fixes a random seed and splits the *text query* indices [0, Nt) into two
      disjoint halves A and B. The gallery (image columns / img_pids) is never
      split -- only the text queries are partitioned.
  (2) Row-wise z-score normalizes each model's similarity matrix
      ((x - row_mean) / max(row_std, eps), eps = 1e-6, population std;
      matches tools/ensemble_eval_scores.py).
  (3) On half A, searches fusion weights for the N models with random search
      (each weight ~ U[0, 2], 2000 trials by default), maximizing R1.
  (4) Evaluates the searched weights on half B, reporting R1/R5/R10/mAP/mINP.
  (5) Also reports half-B results using uniform (average) weights as a control.
  (6) Reuses the project's existing metric function `utils.metrics.rank` so that
      R1/R5/R10/mAP/mINP are computed identically to training/test evaluation.

This is a standalone analysis tool. It does NOT import or modify any training
code and does not touch checkpoints, logs, or datasets.

Example:
    python tools/weight_split_eval.py \
        --score_files pack/scores/seed0.pt pack/scores/seed1.pt ... seed6.pt \
        --seed 42 --num_search 2000 --output_json pack/scores/weight_split_eval.json

If you have not dumped the score matrices yet, produce each one with:
    python test.py --config_file <model_dir>/configs.yaml \
        --save_eval_scores --eval_scores_file pack/scores/seedX.pt
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Reuse the exact metric implementation used by training/test evaluation so
# R1/R5/R10/mAP/mINP stay byte-for-byte consistent with the rest of the project.
from utils.metrics import rank  # noqa: E402


# --------------------------------------------------------------------------- #
# Score file loading / validation (mirrors tools/ensemble_eval_scores.py)
# --------------------------------------------------------------------------- #
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
        raise ValueError(f"Score file {path} has invalid sim shape (expected 2D Nt x Ni).")
    if not torch.is_tensor(txt_pids) or txt_pids.view(-1).numel() != sim.shape[0]:
        raise ValueError(f"Score file {path} has invalid txt_pids (must match sim rows).")
    if not torch.is_tensor(img_pids) or img_pids.view(-1).numel() != sim.shape[1]:
        raise ValueError(f"Score file {path} has invalid img_pids (must match sim cols).")

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
    """All matrices must share shape, evaluation order, and pid vectors."""
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
                "order_hash mismatch (matrices come from different eval orders): "
                f"{item['path']} has {item['order_hash']}, expected {ref['order_hash']}"
            )
        if not torch.equal(item["txt_pids"], ref["txt_pids"]):
            raise ValueError(f"txt_pids mismatch in {item['path']}")
        if not torch.equal(item["img_pids"], ref["img_pids"]):
            raise ValueError(f"img_pids mismatch in {item['path']}")


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def _row_zscore(sim, eps):
    """Row-wise z-score: (x - row_mean) / max(row_std, eps), population std.

    Uses std.clamp_min(eps) to match tools/ensemble_eval_scores.py exactly.
    """
    mean = sim.mean(dim=1, keepdim=True)
    std = sim.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    return (sim - mean) / std


# --------------------------------------------------------------------------- #
# Metric helpers (reuse utils.metrics.rank)
# --------------------------------------------------------------------------- #
def _r1_only(sim, q_pids, g_pids):
    """Fast R1 for the inner search loop (no mAP/mINP)."""
    all_cmc, _ = rank(
        similarity=sim,
        q_pids=q_pids,
        g_pids=g_pids,
        max_rank=10,
        get_mAP=False,
    )
    return float(all_cmc[0].item())


def _full_metrics(sim, q_pids, g_pids):
    """Full R1/R5/R10/mAP/mINP, identical to project Evaluator."""
    all_cmc, mAP, mINP, _ = rank(
        similarity=sim,
        q_pids=q_pids,
        g_pids=g_pids,
        max_rank=10,
        get_mAP=True,
    )
    return {
        "R1": float(all_cmc[0].item()),
        "R5": float(all_cmc[4].item()),
        "R10": float(all_cmc[9].item()),
        "mAP": float(mAP.item()),
        "mINP": float(mINP.item()),
    }


def _fuse(stacked_sim, weights):
    """Weighted sum over models: stacked_sim [N, Nq, Ni], weights [N] -> [Nq, Ni].

    Per-query ranking is invariant to a global positive scale, so we do a plain
    weighted sum (dividing by sum(weights) would not change any metric).
    """
    w = weights.view(-1, 1, 1)
    return (stacked_sim * w).sum(dim=0)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Split-half weighted fusion evaluation over multiple "
        "random-seed IRRA similarity matrices."
    )
    parser.add_argument(
        "--score_files",
        nargs="+",
        required=True,
        help="input .pt score files produced by test.py --save_eval_scores "
        "(default use case: 7 random-seed models)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed controlling both the query split and the weight search",
    )
    parser.add_argument(
        "--num_search",
        type=int,
        default=2000,
        help="number of random-search trials for the fusion weights",
    )
    parser.add_argument(
        "--weight_low",
        type=float,
        default=0.0,
        help="lower bound of the uniform weight sampling range",
    )
    parser.add_argument(
        "--weight_high",
        type=float,
        default=2.0,
        help="upper bound of the uniform weight sampling range",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="epsilon added to row std during z-score normalization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device for the search/eval tensors ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="output .json file for all results",
    )
    args = parser.parse_args()

    if args.num_search <= 0:
        raise ValueError("--num_search must be > 0.")
    if args.eps <= 0:
        raise ValueError("--eps must be > 0.")
    if not args.weight_high > args.weight_low:
        raise ValueError("--weight_high must be greater than --weight_low.")

    device = torch.device(args.device)

    # --- Load + validate ---------------------------------------------------- #
    score_files = [os.path.abspath(path) for path in args.score_files]
    items = [_load_score_file(path) for path in score_files]
    if len(items) < 2:
        raise ValueError("Provide at least 2 score files to fuse.")
    _validate_score_files(items)

    num_models = len(items)
    txt_pids = items[0]["txt_pids"]
    img_pids = items[0]["img_pids"]
    num_txt = int(txt_pids.numel())
    num_img = int(img_pids.numel())

    print("=" * 70)
    print(f"Loaded {num_models} score matrices, each sim shape = "
          f"[Nt={num_txt}, Ni={num_img}]")
    print(f"order_hash = {items[0]['order_hash']}")
    for i, item in enumerate(items):
        base_r1 = item.get("metrics", {}).get("R1", None)
        extra = f"  (saved R1={base_r1:.3f})" if isinstance(base_r1, (int, float)) else ""
        print(f"  [{i}] {item['path']}{extra}")
    print("=" * 70)

    # --- Row z-score normalize each matrix (full matrix; rows are independent
    #     so normalize-then-split == split-then-normalize) -------------------- #
    normalized = [_row_zscore(item["sim"], eps=args.eps) for item in items]
    stacked = torch.stack(normalized, dim=0).to(device)  # [N, Nt, Ni]
    # NOTE: utils.metrics.rank moves the sort indices to CPU and indexes the pid
    # vectors with them, so q_pids/g_pids must stay on CPU (only the similarity
    # matrix benefits from living on the GPU). This matches how Evaluator calls
    # rank in the project.
    img_pids_cpu = img_pids  # already CPU long

    # --- Fixed-seed split of TEXT QUERY indices into disjoint halves A / B --- #
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(num_txt)
    half = num_txt // 2
    idx_a = np.sort(perm[:half])
    idx_b = np.sort(perm[half:])
    idx_a_t = torch.from_numpy(idx_a).long().to(device)
    idx_b_t = torch.from_numpy(idx_b).long().to(device)

    print(f"Query split (seed={args.seed}): "
          f"|A|={idx_a_t.numel()} |B|={idx_b_t.numel()} "
          f"(gallery unchanged: {num_img} images)")

    # Pre-slice the query rows for each half (gallery/columns untouched).
    stacked_a = stacked[:, idx_a_t, :]  # [N, Na, Ni]
    stacked_b = stacked[:, idx_b_t, :]  # [N, Nb, Ni]
    q_pids_a = txt_pids[torch.from_numpy(idx_a).long()]  # CPU
    q_pids_b = txt_pids[torch.from_numpy(idx_b).long()]  # CPU

    # --- Random search on half A, maximizing R1 ----------------------------- #
    # Reseed a dedicated stream for the weight sampling (still deterministic).
    search_rng = np.random.default_rng(args.seed + 1)
    best_weights = None
    best_r1_a = -1.0
    t0 = time.time()
    for trial in range(args.num_search):
        w_np = search_rng.uniform(args.weight_low, args.weight_high, size=num_models)
        weights = torch.from_numpy(w_np).float().to(device)
        fused_a = _fuse(stacked_a, weights)
        r1_a = _r1_only(fused_a, q_pids_a, img_pids_cpu)
        if r1_a > best_r1_a:
            best_r1_a = r1_a
            best_weights = w_np.copy()
    search_secs = time.time() - t0
    print(f"Random search done: {args.num_search} trials in {search_secs:.1f}s, "
          f"best A-half R1 = {best_r1_a:.3f}")

    best_weights_t = torch.from_numpy(best_weights).float().to(device)
    avg_weights_t = torch.ones(num_models, dtype=torch.float32, device=device)

    # --- Evaluate on half B ------------------------------------------------- #
    fused_b_searched = _fuse(stacked_b, best_weights_t)
    fused_b_avg = _fuse(stacked_b, avg_weights_t)
    metrics_b_searched = _full_metrics(fused_b_searched, q_pids_b, img_pids_cpu)
    metrics_b_avg = _full_metrics(fused_b_avg, q_pids_b, img_pids_cpu)

    # For reference: searched weights re-scored on half A with full metrics.
    metrics_a_searched = _full_metrics(
        _fuse(stacked_a, best_weights_t), q_pids_a, img_pids_cpu
    )

    # --- Report ------------------------------------------------------------- #
    def _fmt(m):
        return (f"R1={m['R1']:.3f}  R5={m['R5']:.3f}  R10={m['R10']:.3f}  "
                f"mAP={m['mAP']:.3f}  mINP={m['mINP']:.3f}")

    print("-" * 70)
    print(f"Searched weights (from A-half R1 search): "
          f"{[round(float(w), 4) for w in best_weights]}")
    print(f"[A-half | searched weights] {_fmt(metrics_a_searched)}")
    print(f"[B-half | searched weights] {_fmt(metrics_b_searched)}")
    print(f"[B-half | average weights ] {_fmt(metrics_b_avg)}")
    print("-" * 70)
    d_r1 = metrics_b_searched["R1"] - metrics_b_avg["R1"]
    print(f"B-half R1 improvement (searched - average): {d_r1:+.3f}")
    print("=" * 70)

    # --- Persist ------------------------------------------------------------ #
    result = {
        "config": {
            "score_files": score_files,
            "num_models": num_models,
            "seed": int(args.seed),
            "num_search": int(args.num_search),
            "weight_range": [float(args.weight_low), float(args.weight_high)],
            "eps": float(args.eps),
            "device": str(device),
            "norm": "row_zscore_std_clamp_min_eps",
        },
        "data": {
            "num_txt_queries": num_txt,
            "num_gallery_images": num_img,
            "order_hash": items[0]["order_hash"],
            "split_size_A": int(idx_a_t.numel()),
            "split_size_B": int(idx_b_t.numel()),
            "split_indices_A": idx_a.tolist(),
            "split_indices_B": idx_b.tolist(),
        },
        "searched_weights": [float(w) for w in best_weights],
        "search_best_R1_on_A": float(best_r1_a),
        "metrics": {
            "A_half_searched_weights": metrics_a_searched,
            "B_half_searched_weights": metrics_b_searched,
            "B_half_average_weights": metrics_b_avg,
        },
        "B_half_R1_improvement_searched_minus_average": float(d_r1),
        "per_model_saved_metrics": [
            {"path": item["path"], "metrics": item.get("metrics", {})}
            for item in items
        ],
    }

    output_json = os.path.abspath(args.output_json)
    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved results: {output_json}")


if __name__ == "__main__":
    main()
