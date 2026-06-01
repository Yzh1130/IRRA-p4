import argparse
import json
import math
import os
from collections import defaultdict


MODES = ("concat", "rgb", "thermal", "avg")


def _safe_float(value, default=None):
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _safe_int(value, default=None):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_diag(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _query_maps(payload):
    queries = payload.get("queries", [])
    by_query = {int(item["query_index"]): item for item in queries}
    candidates_by_query = defaultdict(list)
    for candidate in payload.get("candidates", []):
        candidates_by_query[int(candidate["query_index"])].append(candidate)
    return by_query, candidates_by_query


def _raw_oracle_summary(payload):
    summary = payload.get("summary", {})
    total = int(summary.get("total_queries", len(payload.get("queries", []))))
    old_top1_correct = int(summary.get("old_top1_correct", 0))
    gt_in_topk = int(summary.get("gt_in_topk_count", 0))
    wrong_old_top1_but_gt = int(
        summary.get("wrong_old_top1_but_gt_in_topk", 0)
    )
    raw_r1 = 100.0 * old_top1_correct / total if total else 0.0
    oracle_r1 = 100.0 * gt_in_topk / total if total else 0.0
    return {
        "raw_R1": raw_r1,
        "gt_in_topk_count": gt_in_topk,
        "wrong_old_top1_but_gt_in_topk": wrong_old_top1_but_gt,
        "oracle_if_choose_any_gt_in_topk_R1": oracle_r1,
    }


def _local_rank_field(mode):
    return f"local_{mode}_rank_within_topk"


def _local_gap_field(mode):
    return f"local_gap_to_old_top1_{mode}"


def _local_ability(by_query, candidates_by_query):
    target_queries = [
        qidx
        for qidx, query in by_query.items()
        if (not bool(query.get("old_top1_is_gt", False)))
        and bool(query.get("has_gt_in_topk", False))
    ]
    out = {"query_count": len(target_queries)}
    for mode in MODES:
        rank_field = _local_rank_field(mode)
        counts = {"top1": 0, "top3": 0, "top5": 0}
        ranks = []
        for qidx in target_queries:
            gt_ranks = [
                _safe_int(candidate.get(rank_field))
                for candidate in candidates_by_query.get(qidx, [])
                if bool(candidate.get("is_gt_match", False))
            ]
            gt_ranks = [rank for rank in gt_ranks if rank is not None]
            if not gt_ranks:
                continue
            best_rank = min(gt_ranks)
            ranks.append(best_rank)
            if best_rank <= 1:
                counts["top1"] += 1
            if best_rank <= 3:
                counts["top3"] += 1
            if best_rank <= 5:
                counts["top5"] += 1
        out[mode] = {
            "gt_local_top1_count": counts["top1"],
            "gt_local_top3_count": counts["top3"],
            "gt_local_top5_count": counts["top5"],
            "best_gt_local_rank_avg": (
                sum(ranks) / len(ranks) if ranks else None
            ),
        }
    return out


def _choose_candidate(candidates, mode, gap_threshold, margin_threshold, max_rank, max_local_rank):
    gap_field = _local_gap_field(mode)
    rank_field = _local_rank_field(mode)
    accepted = []
    for candidate in candidates:
        if bool(candidate.get("is_old_top1", False)):
            continue
        gap = _safe_float(candidate.get(gap_field))
        margin = _safe_float(candidate.get("global_top12_margin"))
        global_rank = _safe_int(candidate.get("candidate_rank_global"))
        local_rank = _safe_int(candidate.get(rank_field))
        if gap is None or margin is None or global_rank is None or local_rank is None:
            continue
        if gap <= gap_threshold:
            continue
        if margin >= margin_threshold:
            continue
        if global_rank > max_rank:
            continue
        if local_rank > max_local_rank:
            continue
        accepted.append((candidate, gap, local_rank, global_rank))
    if not accepted:
        return None
    accepted.sort(key=lambda item: (-item[1], item[2], item[3]))
    return accepted[0][0]


def _threshold_sweep(by_query, candidates_by_query, total_queries, raw_correct):
    gap_thresholds = [
        -0.05,
        -0.02,
        -0.01,
        0.0,
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
    ]
    margin_thresholds = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    rank_limits = [1, 2, 3, 5, 10, 20, 50]
    local_rank_limits = [1, 2, 3, 5, 8, 16]
    rules = []

    for mode in MODES:
        for gap_threshold in gap_thresholds:
            for margin_threshold in margin_thresholds:
                for max_rank in rank_limits:
                    for max_local_rank in local_rank_limits:
                        accepted_count = 0
                        improved_count = 0
                        hurt_count = 0
                        neutral_count = 0
                        for qidx, query in by_query.items():
                            selected = _choose_candidate(
                                candidates_by_query.get(qidx, []),
                                mode=mode,
                                gap_threshold=gap_threshold,
                                margin_threshold=margin_threshold,
                                max_rank=max_rank,
                                max_local_rank=max_local_rank,
                            )
                            if selected is None:
                                continue
                            accepted_count += 1
                            old_correct = bool(query.get("old_top1_is_gt", False))
                            new_correct = bool(selected.get("is_gt_match", False))
                            if (not old_correct) and new_correct:
                                improved_count += 1
                            elif old_correct and (not new_correct):
                                hurt_count += 1
                            else:
                                neutral_count += 1
                        net = improved_count - hurt_count
                        estimated_r1 = (
                            100.0 * (raw_correct + net) / total_queries
                            if total_queries
                            else 0.0
                        )
                        rules.append(
                            {
                                "accepted_count": int(accepted_count),
                                "improved_count": int(improved_count),
                                "hurt_count": int(hurt_count),
                                "neutral_count": int(neutral_count),
                                "net": int(net),
                                "estimated_R1": float(estimated_r1),
                                "rule": (
                                    f"mode={mode}, "
                                    f"local_gap_to_old_top1_{mode}>{gap_threshold}, "
                                    f"global_top12_margin<{margin_threshold}, "
                                    f"candidate_global_rank<={max_rank}, "
                                    f"local_{mode}_rank<={max_local_rank}"
                                ),
                                "mode": mode,
                                "gap_threshold": gap_threshold,
                                "margin_threshold": margin_threshold,
                                "candidate_global_rank_lte": max_rank,
                                "local_rank_lte": max_local_rank,
                            }
                        )

    rules.sort(
        key=lambda item: (
            item["net"],
            item["estimated_R1"],
            item["improved_count"],
            -item["hurt_count"],
            -item["accepted_count"],
        ),
        reverse=True,
    )
    return rules[:30]


def analyze(diag_file):
    payload = _load_diag(diag_file)
    by_query, candidates_by_query = _query_maps(payload)
    summary = payload.get("summary", {})
    total_queries = int(summary.get("total_queries", len(by_query)))
    raw_correct = int(summary.get("old_top1_correct", 0))
    return {
        "diag_file": os.path.abspath(diag_file),
        "topk_oracle_upper_bound": _raw_oracle_summary(payload),
        "local_ranking_ability": _local_ability(by_query, candidates_by_query),
        "top_30_rules": _threshold_sweep(
            by_query=by_query,
            candidates_by_query=candidates_by_query,
            total_queries=total_queries,
            raw_correct=raw_correct,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze QALR v3 candidate-oracle diagnostics."
    )
    parser.add_argument("--diag_file", required=True, help="candidate_diag.json")
    parser.add_argument("--out", required=True, help="analysis json output")
    args = parser.parse_args()

    result = analyze(args.diag_file)
    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(f"[QALR_V3_ANALYSIS] wrote {out_path}")


if __name__ == "__main__":
    main()
