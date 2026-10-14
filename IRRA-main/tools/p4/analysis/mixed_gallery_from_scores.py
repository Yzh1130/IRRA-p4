"""CPU-only mixed-gallery diagnostics from verified, pairwise cosine scores.

No model, GPU, optimizer or data writer is used. A query's score for a gallery
image is copied from that image's assigned homogeneous degradation scenario.
Every original homogeneous query rank/AP is checked exactly before analysis.
"""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import torch
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from tools.p4.export_dev_records import (
    CORE_SCENARIOS, SCENARIO_NAMES, TRAIN_STATE_SCENARIOS, TRAIN_STATE_SCENARIO_NAMES,
    per_query_records, validate_dev_split,
)
from tools.p4.paired_bootstrap import load_records

POSITIVE_STATES = ("clean", "rgb_noise_010", "rgb_missing", "ir_noise_010", "ir_missing")
MIXTURES = {
    "M1": (("clean", "rgb_noise_010", "ir_noise_010", "rgb_missing", "ir_missing", "rgb_blur_k7"),
           (1 / 6,) * 6),
    "M2": (("clean", "rgb_noise_010"), (0.5, 0.5)),
    "M3": (("clean", "rgb_missing"), (0.8, 0.2)),
}
RERANK_FLAGS = (
    "use_qalr", "use_qalr_v2", "use_qalr_v3_gate", "use_qalr_v3_two_stage_gate",
    "use_qalr_v3_candidate_diag", "use_transport_rerank", "use_uspc_slot_rerank",
    "use_modality_text_prompt",
)


def _annotation_path(export_dir, annotation_file=None):
    if annotation_file is not None:
        return Path(annotation_file).resolve()
    summary_path = Path(export_dir) / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        recorded = summary.get("provenance", {}).get("annotation", {}).get("path", "")
        if recorded and Path(recorded).is_file():
            return Path(recorded).resolve()
    for relative in ("RGBT-PEDES/RGBT-PEDES-Final/Caption.json", "RGBT-PEDES/RGBT-PEDES/caption.json"):
        candidate = CODE_ROOT / "datasets" / "TextReID" / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Annotation unavailable; pass --annotation_file with the split's original annotation")


def _validate_pairwise_payload(payload, scenario):
    required = {"sim", "txt_pids", "img_pids", "txt_indices", "img_indices", "metadata"}
    if not isinstance(payload, dict):
        raise ValueError(f"{scenario}: score payload must be a dictionary")
    if not required <= set(payload):
        raise ValueError(f"{scenario}: score payload missing {sorted(required - set(payload))}")
    metadata = payload["metadata"]
    if metadata.get("score_semantics") != "global_cosine_similarity":
        raise ValueError(f"{scenario}: pairwise column selection requires global_cosine_similarity")
    active = [flag for flag in RERANK_FLAGS if bool(metadata.get(flag, False))]
    if active:
        raise ValueError(f"{scenario}: gallery postprocessing/alternative fusion is unsupported: {active}")
    core_definition = {name: (mode, strength, kernel) for name, mode, strength, kernel in CORE_SCENARIOS}
    if scenario in core_definition:
        mode, strength, kernel = core_definition[scenario]
        expected = {"eval_degrade_mode": mode, "eval_degrade_strength": strength,
                    "eval_degrade_blur_kernel": kernel, "eval_degrade_seed": 2026}
    else:
        mode = dict(TRAIN_STATE_SCENARIOS)[scenario]
        expected = {"eval_degrade_mode": mode,
                    "eval_degrade_strength": .1 if mode.endswith("_noise") else 1.,
                    "eval_degrade_blur_kernel": 7}
    for field, value in expected.items():
        if field in metadata and metadata[field] != value:
            raise ValueError(f"{scenario}: degradation metadata disagrees: {field}")
    sim = payload["sim"]
    if not torch.is_tensor(sim) or sim.ndim != 2 or sim.device.type != "cpu":
        raise ValueError(f"{scenario}: score matrix must be a CPU [queries, gallery] tensor")
    if metadata.get("score_shape") != list(sim.shape):
        raise ValueError(f"{scenario}: recorded score shape disagrees")
    if sim.dtype != torch.float32:
        raise ValueError(f"{scenario}: expected Evaluator's float32 score dump")
    for field in ("txt_pids", "img_pids", "txt_indices", "img_indices"):
        value = payload[field]
        if not torch.is_tensor(value) or value.device.type != "cpu" or value.dtype != torch.long or value.ndim != 1:
            raise ValueError(f"{scenario}: {field} must be Evaluator's CPU one-dimensional int64 tensor")
    order = {"txt_pids": payload["txt_pids"].view(-1).tolist(),
             "img_pids": payload["img_pids"].view(-1).tolist(), "score_shape": list(sim.shape)}
    order_hash = hashlib.sha1(json.dumps(order, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    for recorded in (payload.get("order_hash"), metadata.get("order_hash")):
        if recorded is not None and recorded != order_hash:
            raise ValueError(f"{scenario}: score order hash disagrees")


def verify_homogeneous_records(payloads, dev_data, recorded_rows):
    """Exact ranks and CSV-precision AP, using the exporter's order checks."""
    by_key = {}
    for row in recorded_rows:
        key = (str(row["scenario"]), str(row["query_id"]))
        if key in by_key:
            raise ValueError(f"Duplicate per-query record: {key}")
        by_key[key] = row
    expected_queries = {str(q["query_id"]) for q in dev_data["queries"]}
    core = set(SCENARIO_NAMES)
    all_scenarios = core | set(TRAIN_STATE_SCENARIO_NAMES)
    names = set(payloads)
    if names not in (core, all_scenarios):
        raise ValueError("Scores require exactly eleven core or all seventeen scenarios")
    if {scenario for scenario, _ in by_key} != names:
        raise ValueError("Scores and per_query.csv contain different scenarios")
    reconstructed = {}
    for scenario in (*SCENARIO_NAMES, *TRAIN_STATE_SCENARIO_NAMES):
        if scenario not in names:
            continue
        if scenario not in payloads:
            raise ValueError(f"Missing homogeneous scores: {scenario}")
        _validate_pairwise_payload(payloads[scenario], scenario)
        keys = {query for (name, query) in by_key if name == scenario}
        if keys != expected_queries:
            raise ValueError(f"{scenario}: per-query keys differ from development annotation")
        rows = per_query_records(payloads[scenario], dev_data, scenario)
        for actual in rows:
            recorded = by_key[(scenario, actual["query_id"])]
            for field in ("pid", "caption", "n_pos", "best_pos_rank"):
                if actual[field] != recorded[field]:
                    raise ValueError(f"Homogeneous self-check failed: {scenario}/{actual['query_id']} {field}: "
                                     f"recomputed={actual[field]!r}, recorded={recorded[field]!r}")
            # csv.DictWriter serializes the Python float with round-trip repr;
            # parsing it recovers that exact float32-derived value, not 6 digits.
            if actual["AP"] != recorded["AP"]:
                raise ValueError(f"Homogeneous self-check failed: {scenario}/{actual['query_id']} AP: "
                                 f"recomputed={actual['AP']!r}, recorded={recorded['AP']!r}")
        reconstructed[scenario] = rows
    return reconstructed


def load_verified_export(export_dir, split_file, annotation_file=None):
    """Load eleven or seventeen CPU matrices after validating split/query order."""
    export_dir = Path(export_dir).resolve()
    annotation_path = _annotation_path(export_dir, annotation_file)
    dev_data = validate_dev_split(annotation_path, split_file)
    annos = json.loads(annotation_path.read_text(encoding="utf-8"))
    dev_ids = set(dev_data["gallery_raw_pids"])
    dev_data["gallery_annotation_indices"] = [index for index, anno in enumerate(annos)
                                              if anno.get("split", "train") == "train" and int(anno["id"]) in dev_ids]
    summary_path = export_dir / "summary.json"
    if summary_path.is_file():
        provenance = json.loads(summary_path.read_text(encoding="utf-8")).get("provenance", {})
        semantic_hash = provenance.get("split_identity_sha256")
        if semantic_hash is not None and semantic_hash != dev_data["split_identity_sha256"]:
            raise ValueError("Export summary development split differs from supplied split")
    score_names = {path.stem for path in (export_dir / "scores").glob("*.pt")}
    core, all_names = set(SCENARIO_NAMES), set(SCENARIO_NAMES) | set(TRAIN_STATE_SCENARIO_NAMES)
    if score_names not in (core, all_names):
        raise ValueError("Scores directory requires exactly eleven core or all seventeen scenarios")
    payloads = {
        scenario: torch.load(export_dir / "scores" / f"{scenario}.pt", map_location="cpu", weights_only=True)
        for scenario in (*SCENARIO_NAMES, *TRAIN_STATE_SCENARIO_NAMES) if scenario in score_names
    }
    rows = load_records(export_dir / "per_query.csv")
    reconstructed = verify_homogeneous_records(payloads, dev_data, rows)
    for field in ("test_ckpt_file", "config_file", "model_name"):
        recorded_values = {payload["metadata"][field] for payload in payloads.values()
                           if field in payload["metadata"]}
        if len(recorded_values) > 1:
            raise ValueError(f"Homogeneous score files mix checkpoint/config sources: {field}")
    train_seeds = {payloads[name]["metadata"]["eval_degrade_seed"] for name in TRAIN_STATE_SCENARIO_NAMES
                   if name in payloads and "eval_degrade_seed" in payloads[name]["metadata"]}
    if len(train_seeds) > 1:
        raise ValueError("Training-state score files use inconsistent degradation seeds")
    checked_configs = []
    for config in {payload["metadata"].get("config_file", "") for payload in payloads.values()}:
        if config and Path(config).is_file():
            from utils.config import load_config_mapping

            saved_config = load_config_mapping(config)
            active = [field for field in ("use_global_uncertainty", "use_mum", "use_prob_sampling", "mum_sample_enabled")
                      if bool(saved_config.get(field, False))]
            if active:
                raise ValueError(f"Only verified deterministic A/base/c0 feature paths are supported: {active}")
            checked_configs.append(str(Path(config).resolve()))
    return {"export_dir": str(export_dir), "dev_data": dev_data, "payloads": payloads,
            "scores": {name: item["sim"] for name, item in payloads.items()},
            "records_by_scenario": reconstructed, "source_configs_checked": checked_configs}


def ranking_arrays(sim, qids, gids):
    """Full argsort and float32 AP, identical to the exporter's convention."""
    sim = torch.as_tensor(sim, device="cpu")
    qids, gids = torch.as_tensor(qids, device="cpu"), torch.as_tensor(gids, device="cpu")
    if sim.shape != (qids.numel(), gids.numel()) or not torch.isfinite(sim).all():
        raise ValueError("Invalid score shape or nonfinite scores")
    from utils.metrics import rank

    if not qids.numel() or not gids.numel() or not bool(qids[:, None].eq(gids[None, :]).any(1).all()):
        raise ValueError("Query without a positive gallery item")
    _, _, _, order = rank(sim, qids, gids, max_rank=min(10, len(gids)), get_mAP=True)
    matches = gids[order].eq(qids[:, None])
    n_pos = matches.sum(1)
    if bool((n_pos == 0).any()):
        raise ValueError("Query without a positive gallery item")
    precision = matches.cumsum(1).float() / torch.arange(1, sim.shape[1] + 1).float()
    ap = (precision * matches).sum(1) / n_pos
    best = matches.long().argmax(1) + 1
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, sim.shape[1] + 1).expand_as(order))
    return {"best_pos_rank": best, "AP": ap, "n_pos": n_pos,
            "top1_index": order[:, 0], "ranks": ranks}


def select_state_columns(scores, assignments):
    """Each gallery column has one state, shared by every query."""
    first = next(iter(scores.values()))
    if len(assignments) != first.shape[1]:
        raise ValueError("Assignments must contain one state per gallery image pair")
    mixed = torch.empty_like(first)
    for state in set(assignments):
        if state not in scores or scores[state].shape != first.shape:
            raise ValueError(f"Missing or mismatched score state: {state}")
        columns = torch.tensor([index for index, name in enumerate(assignments) if name == state], dtype=torch.long)
        mixed[:, columns] = scores[state][:, columns]
    return mixed


def positive_negative_mix(positive_scores, negative_scores, qids, gids):
    """All positives of a query take S0, all negatives take S1."""
    positive = torch.as_tensor(qids)[:, None].eq(torch.as_tensor(gids)[None, :])
    return torch.where(positive, positive_scores, negative_scores)


def assign_gallery_states(n_gallery, mixture, seed):
    if mixture not in MIXTURES or n_gallery < 1:
        raise ValueError("Unknown mixture or empty gallery")
    names, probabilities = MIXTURES[mixture]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.multinomial(torch.tensor(probabilities, dtype=torch.float64), n_gallery,
                                replacement=True, generator=generator)
    return [names[index] for index in indices.tolist()]


def clustered_differences(values, pids, n_resamples=1000, seed=2026, weights=None):
    """Query-weighted paired differences; resample complete PID clusters.

    The input contains already paired differences, one row per observation.
    All observations of one PID travel together, including query/state strata.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    pids = np.asarray(pids)
    if values.shape[0] != len(pids) or not len(pids) or not np.isfinite(values).all() or n_resamples < 1:
        raise ValueError("Invalid paired bootstrap observations/resample count")
    weights = np.ones(len(pids), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(pids),) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Bootstrap observation weights must be finite and positive")
    _, inverse = np.unique(pids, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    cluster_sums = np.zeros((n_clusters, values.shape[1]), dtype=np.float64)
    np.add.at(cluster_sums, inverse, values * weights[:, None])
    cluster_sizes = np.bincount(inverse, weights=weights).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    samples = np.empty((n_resamples, values.shape[1]), dtype=np.float64)
    for start in range(0, n_resamples, 256):
        count = min(256, n_resamples - start)
        draws = rng.integers(0, n_clusters, size=(count, n_clusters))
        multiplicities = np.zeros((count, n_clusters), dtype=np.float64)
        np.add.at(multiplicities, (np.repeat(np.arange(count), n_clusters), draws.ravel()), 1)
        # Reductions avoid loading a second vendor BLAS/OpenMP runtime in
        # Windows Python environments that also load PyTorch's runtime.
        numerators = (multiplicities[:, :, None] * cluster_sums[None, :, :]).sum(axis=1)
        denominators = (multiplicities * cluster_sizes[None, :]).sum(axis=1)
        samples[start:start + count] = numerators / denominators[:, None]
    bounds = np.quantile(samples, (0.025, 0.975), axis=0)
    return [{"difference": float(np.average(values[:, index], weights=weights)),
             "ci95": [float(bounds[0, index]), float(bounds[1, index])],
             "n_queries": len(pids), "n_pids": n_clusters}
            for index in range(values.shape[1])]


def clustered_difference(values, pids, n_resamples=1000, seed=2026):
    return clustered_differences(values, pids, n_resamples, seed)[0]


def _metric_vectors(ranked):
    return np.column_stack(((ranked["best_pos_rank"].numpy() == 1) * 100.0,
                            ranked["AP"].numpy().astype(np.float64) * 100.0,
                            ranked["best_pos_rank"].numpy().astype(np.float64)))


def _difference_fields(vectors, baseline, pids, n_resamples, seed):
    differences = clustered_differences(vectors - baseline, pids, n_resamples, seed)
    result = {}
    for column, name in enumerate(("R1", "mAP", "best_pos_rank_mean")):
        result[name] = float(vectors[:, column].mean())
        result[f"delta_{name}"] = differences[column]["difference"]
        result[f"delta_{name}_ci95_low"], result[f"delta_{name}_ci95_high"] = differences[column]["ci95"]
    return result


def positive_pair_ranks(sim, homogeneous_scores, qids, gids, assignments, queries):
    """Each positive ranks only against negatives; equal scores do not outrank it."""
    rows = []
    for query_index, query in enumerate(queries):
        negative = gids.ne(qids[query_index])
        mixed_negative = sim[query_index, negative].sort().values
        for gallery_index in torch.nonzero(~negative).view(-1).tolist():
            state = assignments[gallery_index]
            homogeneous = homogeneous_scores[state][query_index]
            same_negative = homogeneous[negative].sort().values
            # right=True counts <=; only strictly greater negative scores compete.
            mixed_rank = 1 + len(mixed_negative) - int(torch.searchsorted(
                mixed_negative, sim[query_index, gallery_index], right=True))
            homogeneous_rank = 1 + len(same_negative) - int(torch.searchsorted(
                same_negative, homogeneous[gallery_index], right=True))
            rows.append({"query_index": query_index, "query_id": query["query_id"], "pid": query["pid"],
                         "gallery_index": gallery_index, "positive_state": state,
                         "mixed_positive_pair_rank": mixed_rank,
                         "homogeneous_positive_pair_rank": homogeneous_rank,
                         "rank_difference": mixed_rank - homogeneous_rank})
    return rows


def top1_error_state_rates(ranked, qids, gids, assignments, states=None, queries=None):
    """Normalize each wrong top-1 by that query's negative-state proportion.

    The aggregate is mean_q[I(top1 state=s)/negative_share(q,s)] over incorrect
    queries. If a state is absent from a query's negatives its term is zero.
    A second aggregate ratio is supplied with a distinct, explicit name.
    """
    qids, gids = torch.as_tensor(qids), torch.as_tensor(gids)
    top1 = ranked["top1_index"]
    incorrect = gids[top1].ne(qids)
    n_errors = int(incorrect.sum())
    rows, query_rows = [], []
    states = list(states) if states is not None else list(dict.fromkeys(assignments))
    incorrect_indices = torch.nonzero(incorrect).view(-1).tolist()
    for state in states:
        gallery_count = assignments.count(state)
        error_count = 0
        shares, normalized_terms = [], []
        available = 0
        for index in incorrect_indices:
            negative_indices = torch.nonzero(gids.ne(qids[index])).view(-1).tolist()
            n_negative_state = sum(assignments[column] == state for column in negative_indices)
            negative_share = n_negative_state / len(negative_indices)
            is_state = int(assignments[int(top1[index])] == state)
            term = is_state / negative_share if negative_share else 0.0
            error_count += is_state
            available += int(n_negative_state > 0)
            shares.append(negative_share)
            normalized_terms.append(term)
            query_rows.append({"query_index": index,
                               "query_id": queries[index]["query_id"] if queries is not None else str(index),
                               "pid": queries[index]["pid"] if queries is not None else int(qids[index]),
                               "state": state, "top1_gallery_index": int(top1[index]),
                               "top1_state": assignments[int(top1[index])],
                               "negative_count": len(negative_indices), "negative_state_count": n_negative_state,
                               "negative_state_share": negative_share,
                               "top1_from_state": is_state, "query_normalized_top1": term})
        gallery_share = gallery_count / len(assignments)
        error_share = error_count / n_errors if n_errors else None
        mean_negative_share = float(np.mean(shares)) if shares else None
        rows.append({"state": state, "gallery_count": gallery_count, "gallery_share": gallery_share,
                     "top1_error_count": error_count, "all_top1_errors": n_errors,
                     "top1_error_share": error_share, "available_error_queries": available,
                     "mean_negative_state_share": mean_negative_share,
                     "query_normalized_error_ratio": float(np.mean(normalized_terms)) if normalized_terms else None,
                     "error_share_over_mean_negative_share": (error_share / mean_negative_share
                                                               if mean_negative_share else None)})
    return rows, query_rows


def analyze_verified_export(verified, seeds=(2026, 2027, 2028), n_resamples=1000, bootstrap_seed=2026):
    data, scores = verified["dev_data"], verified["scores"]
    qids = torch.tensor([q["eval_pid"] for q in data["queries"]])
    gids = torch.tensor(data["gallery_pids"])
    pids = [q["pid"] for q in data["queries"]]
    homogeneous = {name: ranking_arrays(score, qids, gids) for name, score in scores.items()}
    homogeneous_vectors = {name: _metric_vectors(ranked) for name, ranked in homogeneous.items()}
    matrix_rows, matrix_vectors = [], {}
    for positive_state, negative_state in itertools.product(POSITIVE_STATES, SCENARIO_NAMES):
        ranked = ranking_arrays(positive_negative_mix(scores[positive_state], scores[negative_state], qids, gids), qids, gids)
        vectors = _metric_vectors(ranked)
        matrix_vectors[(positive_state, negative_state)] = vectors
        matrix_rows.append({"positive_state": positive_state, "negative_state": negative_state,
                            "baseline": positive_state, "n_queries": len(qids),
                            **_difference_fields(vectors, homogeneous_vectors[positive_state], pids,
                                                 n_resamples, bootstrap_seed)})
    random_rows, stratum_rows, stratum_queries, error_rows, error_query_rows, assignment_rows = [], [], [], [], [], []
    random_vectors, stratum_vectors = {}, {}
    for mixture, seed in itertools.product(MIXTURES, seeds):
        assignments = assign_gallery_states(len(gids), mixture, seed)
        mixed_scores = select_state_columns(scores, assignments)
        ranked = ranking_arrays(mixed_scores, qids, gids)
        vectors = _metric_vectors(ranked)
        random_vectors[(mixture, int(seed))] = vectors
        random_rows.append({"mixture": mixture, "seed": int(seed), "baseline": "clean", "n_queries": len(qids),
                            **_difference_fields(vectors, homogeneous_vectors["clean"], pids, n_resamples, bootstrap_seed)})
        pair_rows = positive_pair_ranks(mixed_scores, scores, qids, gids, assignments, data["queries"])
        for state in MIXTURES[mixture][0]:
            pairs = [row for row in pair_rows if row["positive_state"] == state]
            if not pairs:
                stratum_rows.append({"mixture": mixture, "seed": int(seed), "positive_state": state,
                                     "n_positive_pairs": 0, "n_pids": 0, "mixed_positive_pair_rank_mean": None,
                                     "homogeneous_positive_pair_rank_mean": None, "rank_difference": None,
                                     "rank_difference_ci95_low": None, "rank_difference_ci95_high": None})
                continue
            pair_pids = [row["pid"] for row in pairs]
            pair_deltas = [row["rank_difference"] for row in pairs]
            delta = clustered_difference(pair_deltas, pair_pids, n_resamples, bootstrap_seed)
            stratum_vectors[(mixture, int(seed), state)] = pairs
            stratum_rows.append({"mixture": mixture, "seed": int(seed), "positive_state": state,
                                 "n_positive_pairs": len(pairs), "n_pids": delta["n_pids"],
                                 "mixed_positive_pair_rank_mean": float(np.mean([row["mixed_positive_pair_rank"] for row in pairs])),
                                 "homogeneous_positive_pair_rank_mean": float(np.mean([row["homogeneous_positive_pair_rank"] for row in pairs])),
                                 "rank_difference": delta["difference"],
                                 "rank_difference_ci95_low": delta["ci95"][0], "rank_difference_ci95_high": delta["ci95"][1]})
        stratum_queries.extend({"mixture": mixture, "seed": int(seed), **row} for row in pair_rows)
        rates, rate_queries = top1_error_state_rates(ranked, qids, gids, assignments,
                                                   MIXTURES[mixture][0], data["queries"])
        error_rows.extend({"mixture": mixture, "seed": int(seed), **row} for row in rates)
        error_query_rows.extend({"mixture": mixture, "seed": int(seed), **row} for row in rate_queries)
        assignment_rows.extend({"mixture": mixture, "seed": int(seed), "gallery_index": index,
                                "image_id": index, "annotation_index": data.get("gallery_annotation_indices", list(range(len(gids))))[index],
                                "gallery_pid": data["gallery_raw_pids"][index], "state": state}
                               for index, state in enumerate(assignments))
    for mixture in MIXTURES:
        mean_vectors = np.mean([random_vectors[(mixture, int(seed))] for seed in seeds], axis=0)
        random_vectors[(mixture, "mean")] = mean_vectors
        random_rows.append({"mixture": mixture, "seed": "mean", "baseline": "clean", "n_queries": len(qids),
                            **_difference_fields(mean_vectors, homogeneous_vectors["clean"], pids, n_resamples, bootstrap_seed)})
        for state in MIXTURES[mixture][0]:
            rows_by_seed = [stratum_vectors.get((mixture, int(seed), state), []) for seed in seeds]
            nonempty = [rows for rows in rows_by_seed if rows]
            pairs = [row for rows in nonempty for row in rows]
            if not pairs:
                continue
            weights = [1 / len(rows) for rows in nonempty for _ in rows]
            delta = clustered_differences([row["rank_difference"] for row in pairs],
                                          [row["pid"] for row in pairs], n_resamples, bootstrap_seed, weights)[0]
            stratum_vectors[(mixture, "mean", state)] = {"pairs": pairs, "weights": weights}
            stratum_rows.append({"mixture": mixture, "seed": "mean", "positive_state": state,
                                 "n_positive_pairs": len(pairs), "n_pids": delta["n_pids"],
                                 "mixed_positive_pair_rank_mean": float(np.average(
                                     [row["mixed_positive_pair_rank"] for row in pairs], weights=weights)),
                                 "homogeneous_positive_pair_rank_mean": float(np.average(
                                     [row["homogeneous_positive_pair_rank"] for row in pairs], weights=weights)),
                                 "rank_difference": delta["difference"],
                                 "rank_difference_ci95_low": delta["ci95"][0], "rank_difference_ci95_high": delta["ci95"][1]})
    return {"matrix": matrix_rows, "random_metrics": random_rows, "positive_state_strata": stratum_rows,
            "positive_state_queries": stratum_queries, "top1_error_states": error_rows,
            "top1_error_queries": error_query_rows, "assignments": assignment_rows,
            "_matrix_vectors": matrix_vectors, "_random_vectors": random_vectors,
            "_stratum_vectors": stratum_vectors}


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(export_dirs, split_file, output_dir, seeds=(2026, 2027, 2028), annotation_file=None,
                 n_resamples=1000, bootstrap_seed=2026, names=None):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Analysis output is nonempty; choose a new or empty directory: {output_dir}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must be nonempty and unique")
    exports = [load_verified_export(path, split_file, annotation_file) for path in export_dirs]
    if not exports:
        raise ValueError("No export directories supplied")
    names = [Path(export["export_dir"]).name for export in exports] if names is None else list(names)
    if len(names) != len(exports) or len(set(names)) != len(names):
        raise ValueError("--names must contain one unique display name per export directory")
    reference = exports[0]["dev_data"]
    for export in exports[1:]:
        for field in ("queries", "gallery_raw_pids", "gallery_pids", "gallery_annotation_indices", "split_identity_sha256"):
            if export["dev_data"][field] != reference[field]:
                raise ValueError(f"Multiple exports disagree on development ordering: {field}")
    # All eleven rank/AP self-checks above must finish before any output is created.
    results = [analyze_verified_export(export, seeds, n_resamples, bootstrap_seed) for export in exports]
    output_dir.mkdir(parents=True, exist_ok=True)
    public_results = []
    for index, (export, result) in enumerate(zip(exports, results)):
        label = f"{index:02d}_{Path(export['export_dir']).name}"
        run_dir = output_dir / label
        run_dir.mkdir()
        public = {key: value for key, value in result.items() if not key.startswith("_")}
        for name, rows in public.items():
            _write_csv(run_dir / f"{name}.csv", rows)
        public_results.append({"label": label, "name": names[index], "export_dir": export["export_dir"],
                               "n_scenarios_checked": len(export["scores"]),
                               "source_configs_checked": export["source_configs_checked"], **public})
        (run_dir / "results.json").write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparisons = []
    pids = [query["pid"] for query in reference["queries"]]
    for a, b in itertools.combinations(range(len(results)), 2):
        for kind, field in (("positive_negative", "_matrix_vectors"), ("random", "_random_vectors")):
            for key, vectors_a in results[a][field].items():
                vectors_b = results[b][field][key]
                differences = clustered_differences(vectors_b - vectors_a, pids, n_resamples, bootstrap_seed)
                for metric, item in zip(("R1", "mAP", "best_pos_rank_mean"), differences):
                    comparisons.append({"export_A": exports[a]["export_dir"], "export_B": exports[b]["export_dir"],
                                        "name_A": names[a], "name_B": names[b],
                                        "kind": kind, "condition_1": key[0], "condition_2": key[1], "metric": metric,
                                        "difference_B_minus_A": item["difference"],
                                        "ci95_low": item["ci95"][0], "ci95_high": item["ci95"][1]})
        for key, selected_a in results[a]["_stratum_vectors"].items():
            selected_b = results[b]["_stratum_vectors"][key]
            if key[1] == "mean":
                pairs_a, pairs_b, weights = selected_a["pairs"], selected_b["pairs"], selected_a["weights"]
                if weights != selected_b["weights"]:
                    raise ValueError("Cross-model mixture strata disagree on seed weights")
            else:
                pairs_a, pairs_b, weights = selected_a, selected_b, None
            pairing_a = [(row["query_id"], row["gallery_index"], row["positive_state"]) for row in pairs_a]
            pairing_b = [(row["query_id"], row["gallery_index"], row["positive_state"]) for row in pairs_b]
            if pairing_a != pairing_b:
                raise ValueError("Cross-model positive-pair ordering differs")
            pair_pids = [row["pid"] for row in pairs_a]
            for metric in ("mixed_positive_pair_rank", "rank_difference"):
                delta = clustered_differences([brow[metric] - arow[metric] for arow, brow in zip(pairs_a, pairs_b)],
                                              pair_pids, n_resamples, bootstrap_seed, weights)[0]
                comparisons.append({"export_A": exports[a]["export_dir"], "export_B": exports[b]["export_dir"],
                                    "name_A": names[a], "name_B": names[b],
                                    "kind": "positive_state_stratum", "condition_1": key[0],
                                    "condition_2": f"{key[1]}:{key[2]}", "metric": metric,
                                    "difference_B_minus_A": delta["difference"],
                                    "ci95_low": delta["ci95"][0], "ci95_high": delta["ci95"][1]})
    _write_csv(output_dir / "checkpoint_comparisons.csv", comparisons)
    summary = {
        "device": "cpu", "homogeneous_selfcheck": "all exported scenarios: exact best_pos_rank and round-trip CSV AP",
        "split": reference["split"], "annotation": reference["annotation"],
        "metric_units": "R1/mAP/differences in percentage points; ranks in positions",
        "bootstrap": {"cluster": "original pid", "weighting": "text queries", "n_resamples": n_resamples,
                      "seed": bootstrap_seed, "interval": "percentile 95%"},
        "positive_state_stratum_definition": (
            "One observation per (query, positive gallery image pair). Rank is 1 plus the number of strictly higher "
            "scoring negatives; other positives and tied negatives do not count. Subtract the pair's rank in its "
            "assigned state's homogeneous gallery from its mixed-gallery rank. Mean rows average nonempty "
            "seed-level means equally; PID-bootstrap resamples all paired observations with 1/n_pairs_seed weights."
        ),
        "top1_error_state_definition": (
            "Over wrong-top1 queries, average I(top1 state=s)/fraction of that query's NEGATIVES assigned s. "
            "Absent states contribute zero; no-error mixtures have undefined ratios. The separately named aggregate "
            "ratio divides error state share by mean negative state share. Per-query denominators are exported."
        ),
        "image_id_semantics": "canonical evaluation img_indices/gallery row ID; not training-batch image_ids",
        "seed_average_definition": "Mean each query's metric across assignment seeds before PID-cluster bootstrap",
        "applicability": "Verified deterministic A/base_clean/c0 feature paths and pairwise cosine metadata only; "
                         "unavailable source configs are not independently verified",
        "mixtures": {name: {"states": list(states), "probabilities": list(probabilities)}
                     for name, (states, probabilities) in MIXTURES.items()},
        "assignment_seeds": list(seeds), "exports": public_results, "checkpoint_comparisons": comparisons,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export_dirs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None, help="One unique display name per export directory")
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--output_dir", required=True, help="Must be new or empty; existing files are never overwritten")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--annotation_file", default=None, help="Relocated original annotation (MD5 verified against split)")
    parser.add_argument("--n_resamples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    return parser


if __name__ == "__main__":
    cli = build_parser().parse_args()
    summary = run_analysis(cli.export_dirs, cli.split_file, cli.output_dir, cli.seeds,
                           cli.annotation_file, cli.n_resamples, cli.bootstrap_seed, cli.names)
    print(json.dumps({"output_dir": str(Path(cli.output_dir).resolve()), "n_exports": len(summary["exports"]),
                      "homogeneous_selfcheck": summary["homogeneous_selfcheck"]}, ensure_ascii=False, indent=2))
