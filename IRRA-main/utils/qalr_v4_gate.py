import math
from collections import OrderedDict

import torch


QALR_V4_FEATURE_NAMES = [
    "global_score",
    "global_rank",
    "global_rank_norm",
    "global_top1_score",
    "global_top2_score",
    "global_top12_margin",
    "global_score_gap_to_old_top1",
    "local_avg_score",
    "local_concat_score",
    "local_rgb_score",
    "local_thermal_score",
    "local_avg_rank",
    "local_concat_rank",
    "local_rgb_rank",
    "local_thermal_rank",
    "local_avg_rank_norm",
    "local_concat_rank_norm",
    "local_rgb_rank_norm",
    "local_thermal_rank_norm",
    "local_avg_gap_to_old_top1",
    "local_concat_gap_to_old_top1",
    "local_rgb_gap_to_old_top1",
    "local_thermal_gap_to_old_top1",
    "avg_concat_agreement",
    "rgb_thermal_gap",
    "rgb_avg_gap",
    "thermal_avg_gap",
]


QALR_V4_FEATURE_DEFINITIONS = OrderedDict(
    [
        ("global_score", "Candidate global cosine score from the pre-QALR matrix."),
        ("global_rank", "1-based candidate rank inside the query topK global pool."),
        (
            "global_rank_norm",
            "(global_rank - 1) / max(topk - 1, 1), so top1 is 0 and the last topK item is near 1.",
        ),
        ("global_top1_score", "Global score of the original top1 candidate."),
        ("global_top2_score", "Global score of the original top2 candidate."),
        ("global_top12_margin", "global_top1_score - global_top2_score."),
        (
            "global_score_gap_to_old_top1",
            "candidate global_score - original top1 global_score.",
        ),
        ("local_avg_score", "Average of RGB and thermal local matching scores."),
        ("local_concat_score", "Local matching score over concatenated RGB/T patch tokens."),
        ("local_rgb_score", "Local matching score over RGB patch tokens."),
        ("local_thermal_score", "Local matching score over thermal patch tokens."),
        ("local_avg_rank", "1-based rank by local_avg_score inside topK."),
        ("local_concat_rank", "1-based rank by local_concat_score inside topK."),
        ("local_rgb_rank", "1-based rank by local_rgb_score inside topK."),
        ("local_thermal_rank", "1-based rank by local_thermal_score inside topK."),
        (
            "local_avg_rank_norm",
            "(local_avg_rank - 1) / max(topk - 1, 1).",
        ),
        (
            "local_concat_rank_norm",
            "(local_concat_rank - 1) / max(topk - 1, 1).",
        ),
        ("local_rgb_rank_norm", "(local_rgb_rank - 1) / max(topk - 1, 1)."),
        (
            "local_thermal_rank_norm",
            "(local_thermal_rank - 1) / max(topk - 1, 1).",
        ),
        (
            "local_avg_gap_to_old_top1",
            "candidate local_avg_score - original top1 local_avg_score.",
        ),
        (
            "local_concat_gap_to_old_top1",
            "candidate local_concat_score - original top1 local_concat_score.",
        ),
        (
            "local_rgb_gap_to_old_top1",
            "candidate local_rgb_score - original top1 local_rgb_score.",
        ),
        (
            "local_thermal_gap_to_old_top1",
            "candidate local_thermal_score - original top1 local_thermal_score.",
        ),
        (
            "avg_concat_agreement",
            "1 - abs(local_avg_rank - local_concat_rank) / max(topk - 1, 1), clipped to [0, 1].",
        ),
        ("rgb_thermal_gap", "local_rgb_score - local_thermal_score."),
        ("rgb_avg_gap", "local_rgb_score - local_avg_score."),
        ("thermal_avg_gap", "local_thermal_score - local_avg_score."),
    ]
)


def qalr_v4_feature_meta(topk):
    return {
        "feature_names": list(QALR_V4_FEATURE_NAMES),
        "feature_definitions": dict(QALR_V4_FEATURE_DEFINITIONS),
        "rank_norm": {
            "formula": "(rank - 1) / max(topk - 1, 1)",
            "topk": int(topk),
        },
        "label": "candidate_pid == q_pid",
        "leakage_guard": (
            "q_pid, candidate_pid, candidate_gallery_index and is_gt_match are "
            "stored only as labels/metadata and are not included in features."
        ),
    }


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(value):
        return float(default)
    return float(value)


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _rank_norm(rank, topk):
    denom = max(int(topk) - 1, 1)
    return float(max(_safe_int(rank, 1) - 1, 0)) / float(denom)


def build_qalr_v4_feature_vector(candidate, topk):
    global_score = _safe_float(candidate.get("global_score"))
    global_top1_score = _safe_float(candidate.get("global_top1_score"))

    local_avg_score = _safe_float(candidate.get("local_avg_score"))
    local_concat_score = _safe_float(candidate.get("local_concat_score"))
    local_rgb_score = _safe_float(candidate.get("local_rgb_score"))
    local_thermal_score = _safe_float(candidate.get("local_thermal_score"))

    global_rank = _safe_int(candidate.get("candidate_rank_global"), 1)
    local_avg_rank = _safe_int(candidate.get("local_avg_rank_within_topk"), 1)
    local_concat_rank = _safe_int(candidate.get("local_concat_rank_within_topk"), 1)
    local_rgb_rank = _safe_int(candidate.get("local_rgb_rank_within_topk"), 1)
    local_thermal_rank = _safe_int(
        candidate.get("local_thermal_rank_within_topk"), 1
    )

    agreement = 1.0 - abs(local_avg_rank - local_concat_rank) / float(
        max(int(topk) - 1, 1)
    )
    agreement = min(1.0, max(0.0, agreement))

    values = OrderedDict(
        [
            ("global_score", global_score),
            ("global_rank", float(global_rank)),
            ("global_rank_norm", _rank_norm(global_rank, topk)),
            ("global_top1_score", global_top1_score),
            ("global_top2_score", _safe_float(candidate.get("global_top2_score"))),
            (
                "global_top12_margin",
                _safe_float(candidate.get("global_top12_margin")),
            ),
            ("global_score_gap_to_old_top1", global_score - global_top1_score),
            ("local_avg_score", local_avg_score),
            ("local_concat_score", local_concat_score),
            ("local_rgb_score", local_rgb_score),
            ("local_thermal_score", local_thermal_score),
            ("local_avg_rank", float(local_avg_rank)),
            ("local_concat_rank", float(local_concat_rank)),
            ("local_rgb_rank", float(local_rgb_rank)),
            ("local_thermal_rank", float(local_thermal_rank)),
            ("local_avg_rank_norm", _rank_norm(local_avg_rank, topk)),
            ("local_concat_rank_norm", _rank_norm(local_concat_rank, topk)),
            ("local_rgb_rank_norm", _rank_norm(local_rgb_rank, topk)),
            ("local_thermal_rank_norm", _rank_norm(local_thermal_rank, topk)),
            (
                "local_avg_gap_to_old_top1",
                _safe_float(candidate.get("local_gap_to_old_top1_avg")),
            ),
            (
                "local_concat_gap_to_old_top1",
                _safe_float(candidate.get("local_gap_to_old_top1_concat")),
            ),
            (
                "local_rgb_gap_to_old_top1",
                _safe_float(candidate.get("local_gap_to_old_top1_rgb")),
            ),
            (
                "local_thermal_gap_to_old_top1",
                _safe_float(candidate.get("local_gap_to_old_top1_thermal")),
            ),
            ("avg_concat_agreement", agreement),
            ("rgb_thermal_gap", local_rgb_score - local_thermal_score),
            ("rgb_avg_gap", local_rgb_score - local_avg_score),
            ("thermal_avg_gap", local_thermal_score - local_avg_score),
        ]
    )
    return [float(values[name]) for name in QALR_V4_FEATURE_NAMES]


def build_qalr_v4_tensor_payload(candidates, metadata, topk):
    feature_rows = []
    labels = []
    query_indices = []
    candidate_indices = []
    q_pids = []
    candidate_pids = []
    candidate_global_ranks = []

    group_query_indices = []
    group_starts = []
    group_ends = []
    group_q_pids = []
    current_query = None
    current_start = 0
    current_q_pid = None

    for row_index, candidate in enumerate(candidates):
        query_index = _safe_int(candidate.get("query_index"), 0)
        q_pid = _safe_int(candidate.get("q_pid"), -1)
        candidate_pid = _safe_int(candidate.get("candidate_pid"), -1)

        if current_query is None:
            current_query = query_index
            current_start = row_index
            current_q_pid = q_pid
        elif query_index != current_query:
            group_query_indices.append(int(current_query))
            group_starts.append(int(current_start))
            group_ends.append(int(row_index))
            group_q_pids.append(int(current_q_pid))
            current_query = query_index
            current_start = row_index
            current_q_pid = q_pid

        feature_rows.append(build_qalr_v4_feature_vector(candidate, topk=topk))
        labels.append(1 if candidate_pid == q_pid else 0)
        query_indices.append(query_index)
        candidate_indices.append(
            _safe_int(candidate.get("candidate_gallery_index"), -1)
        )
        q_pids.append(q_pid)
        candidate_pids.append(candidate_pid)
        candidate_global_ranks.append(
            _safe_int(candidate.get("candidate_rank_global"), 0)
        )

    if current_query is not None:
        group_query_indices.append(int(current_query))
        group_starts.append(int(current_start))
        group_ends.append(int(len(candidates)))
        group_q_pids.append(int(current_q_pid))

    num_features = len(QALR_V4_FEATURE_NAMES)
    if feature_rows:
        features = torch.tensor(feature_rows, dtype=torch.float32)
    else:
        features = torch.empty((0, num_features), dtype=torch.float32)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    query_indices_tensor = torch.tensor(query_indices, dtype=torch.long)
    candidate_indices_tensor = torch.tensor(candidate_indices, dtype=torch.long)
    q_pids_tensor = torch.tensor(q_pids, dtype=torch.long)
    candidate_pids_tensor = torch.tensor(candidate_pids, dtype=torch.long)
    candidate_global_ranks_tensor = torch.tensor(
        candidate_global_ranks, dtype=torch.long
    )

    groups = {
        "query_indices": torch.tensor(group_query_indices, dtype=torch.long),
        "starts": torch.tensor(group_starts, dtype=torch.long),
        "ends": torch.tensor(group_ends, dtype=torch.long),
        "q_pids": torch.tensor(group_q_pids, dtype=torch.long),
    }

    positive_count = int(labels_tensor.sum().item()) if labels_tensor.numel() else 0
    query_positive_count = 0
    for start, end in zip(group_starts, group_ends):
        if any(labels[start:end]):
            query_positive_count += 1

    payload_metadata = dict(metadata)
    payload_metadata.update(
        {
            "num_candidates": int(features.shape[0]),
            "num_features": int(features.shape[1]),
            "positive_count": int(positive_count),
            "positive_ratio": (
                float(positive_count) / float(features.shape[0])
                if features.shape[0] > 0
                else 0.0
            ),
            "num_queries": int(len(group_query_indices)),
            "queries_with_positive_in_topk": int(query_positive_count),
        }
    )

    return {
        "features": features,
        "labels": labels_tensor,
        "query_indices": query_indices_tensor,
        "candidate_indices": candidate_indices_tensor,
        "q_pids": q_pids_tensor,
        "candidate_pids": candidate_pids_tensor,
        "candidate_global_ranks": candidate_global_ranks_tensor,
        "groups": groups,
        "feature_names": list(QALR_V4_FEATURE_NAMES),
        "metadata": payload_metadata,
    }
