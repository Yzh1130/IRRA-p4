"""CPU analysis of exact additive score contributions and P-feat retrieval.

P-feat normalizes a saved modality feature. It does not zero the other modality's
input and encode again (P-zero). Signed contribution shares can be negative or
exceed one; nonpositive total scores have undefined shares, saved as null/blank.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F

from tools.p4.paired_bootstrap import paired_bootstrap
from tools.p4.export_dev_records import SCENARIO_NAMES, TRAIN_STATE_SCENARIO_NAMES
from tools.p4.analysis.mixed_gallery_from_scores import _validate_pairwise_payload, ranking_arrays


def retrieval_statistics(scores, query_pids, gallery_pids):
    """Match Evaluator's full descending argsort and float32 AP convention."""
    if scores.ndim != 2 or scores.shape != (len(query_pids), len(gallery_pids)):
        raise ValueError("Scores/PID shapes disagree")
    if not torch.isfinite(scores).all():
        raise ValueError("Retrieval scores must be finite")
    ranked = ranking_arrays(scores, query_pids, gallery_pids)
    return {**ranked, "R1": float((ranked["best_pos_rank"] == 1).double().mean() * 100),
            "mAP": float(ranked["AP"].double().mean() * 100)}


def decompose_features(features, score_payload=None):
    """Return score matrices and per-query scalar contributions on CPU."""
    required = ("f_rgb", "f_ir", "f_fused", "text_features", "txt_pids", "img_pids")
    missing = [key for key in required if key not in features]
    if missing:
        raise ValueError(f"Feature payload missing fields: {missing}")
    rgb, ir, fused, text = [features[name].detach().cpu().float()
                            for name in required[:4]]
    qids, gids = [features[name].detach().cpu().long().view(-1) for name in required[4:]]
    if (rgb.ndim != 2 or rgb.shape != ir.shape or rgb.shape != fused.shape
            or text.ndim != 2 or text.shape[1] != rgb.shape[1]
            or rgb.shape[0] != len(gids) or text.shape[0] != len(qids)):
        raise ValueError("Additive modality/query/PID shapes disagree")
    if not all(bool(torch.isfinite(value).all()) for value in (rgb, ir, fused, text)):
        raise ValueError("Nonfinite modality features")
    if not torch.equal(fused, rgb + ir):
        raise ValueError("Saved fused tensor is not exactly f_rgb + f_ir")
    text_normalized = F.normalize(text, dim=1)
    denominator = fused.norm(dim=1).clamp_min(1e-12)
    fused_scores = text_normalized @ F.normalize(fused, dim=1).T
    rgb_scores = text_normalized @ (rgb / denominator[:, None]).T
    ir_scores = text_normalized @ (ir / denominator[:, None]).T
    max_error = float((rgb_scores + ir_scores - fused_scores).abs().max())
    if not math.isfinite(max_error) or max_error >= 1e-5:
        raise ValueError(f"Score decomposition fails tolerance: {max_error}")
    if score_payload is not None:
        for key, expected in (("txt_pids", qids), ("img_pids", gids)):
            if not torch.equal(score_payload[key].cpu().long().view(-1), expected):
                raise ValueError(f"Saved scores and features disagree on {key}")
        reference = score_payload["sim"].detach().cpu().float()
        if reference.shape != fused_scores.shape:
            raise ValueError("Saved score shape differs from feature score shape")
        score_error = float((fused_scores - reference).abs().max())
        if not math.isfinite(score_error) or score_error >= 1e-5:
            raise ValueError(f"Features do not reconstruct saved scores: {score_error}")
    else:
        score_error = None
    # Preserve original fused rankings when CPU reconstruction and GPU scores
    # differ slightly near a tie. Contributions retain their exact feature formula.
    ranking_scores = reference if score_payload is not None else fused_scores
    rgb_probe = text_normalized @ F.normalize(rgb, dim=1).T
    ir_probe = text_normalized @ F.normalize(ir, dim=1).T
    positives = qids[:, None].eq(gids)
    if bool((positives.sum(1) == 0).any()) or bool((~positives).sum(1).eq(0).any()):
        raise ValueError("Each query needs positive and negative candidates for margin analysis")
    fused_retrieval = retrieval_statistics(ranking_scores, qids, gids)
    order = torch.argsort(fused_retrieval["ranks"], dim=1)
    ranked_positive = positives.gather(1, order)
    best_positive = order.gather(1, ranked_positive.long().argmax(1)[:, None]).squeeze(1)
    strongest_negative = order.gather(1, (~ranked_positive).long().argmax(1)[:, None]).squeeze(1)
    def share(part, total):
        return None if total <= 0 else part / total

    query_rows = []
    for index in range(len(qids)):
        pos = positives[index]
        positive_total = float(ranking_scores[index, pos].mean())
        positive_rgb = float(rgb_scores[index, pos].mean())
        positive_ir = float(ir_scores[index, pos].mean())
        p, n = int(best_positive[index]), int(strongest_negative[index])
        best_total, best_rgb, best_ir = (float(matrix[index, p])
                                        for matrix in (ranking_scores, rgb_scores, ir_scores))
        negative_total, negative_rgb, negative_ir = (float(matrix[index, n])
                                                    for matrix in (ranking_scores, rgb_scores, ir_scores))
        query_rows.append({
            "positive_mean_score": positive_total,
            "positive_mean_rgb_contribution": positive_rgb,
            "positive_mean_ir_contribution": positive_ir,
            "positive_mean_rgb_share": share(positive_rgb, positive_total),
            "positive_mean_ir_share": share(positive_ir, positive_total),
            "best_positive_gallery_index": p, "strongest_negative_gallery_index": n,
            "best_positive_score": best_total, "best_positive_rgb_contribution": best_rgb,
            "best_positive_ir_contribution": best_ir,
            "best_positive_rgb_share": share(best_rgb, best_total),
            "best_positive_ir_share": share(best_ir, best_total),
            "best_positive_rgb_negative": int(best_rgb < 0),
            "best_positive_ir_negative": int(best_ir < 0),
            "best_positive_reconstruction_residual": best_rgb + best_ir - best_total,
            "strongest_negative_score": negative_total,
            "strongest_negative_rgb_contribution": negative_rgb,
            "strongest_negative_ir_contribution": negative_ir,
            "margin": best_total - negative_total,
            "margin_rgb_contribution": best_rgb - negative_rgb,
            "margin_ir_contribution": best_ir - negative_ir,
            "margin_reconstruction_residual": best_rgb + best_ir - negative_rgb - negative_ir - best_total + negative_total,
            "n_pos": int(pos.sum()),
        })
    return {
        "fused_scores": ranking_scores, "reconstructed_fused_scores": fused_scores,
        "rgb_contribution_scores": rgb_scores,
        "ir_contribution_scores": ir_scores,
        "P_feat_rgb_scores": rgb_probe, "P_feat_ir_scores": ir_probe,
        "query_rows": query_rows,
        "retrieval": {name: retrieval_statistics(matrix, qids, gids) for name, matrix in
                      (("fused", ranking_scores), ("P-feat-rgb", rgb_probe), ("P-feat-ir", ir_probe))},
        "geometry": {"rgb_norm": distribution(rgb.norm(dim=1)),
                     "ir_norm": distribution(ir.norm(dim=1)),
                     "rgb_ir_cosine": distribution(F.cosine_similarity(rgb, ir, dim=1, eps=1e-12))},
        "negative_contribution_fractions": {
            "best_positive_rgb": sum(row["best_positive_rgb_negative"] for row in query_rows) / len(query_rows),
            "best_positive_ir": sum(row["best_positive_ir_negative"] for row in query_rows) / len(query_rows),
        },
        "max_decomposition_error": max_error, "max_saved_score_error": score_error,
    }


def distribution(values):
    values = values.detach().cpu().double()
    probabilities = (0., .05, .25, .5, .75, .95, 1.)
    quantiles = torch.quantile(values, torch.tensor(probabilities, dtype=torch.float64))
    return {"mean": float(values.mean()), "std": float(values.std(unbiased=False)),
            **{f"q{int(probability * 100):02d}": float(value)
               for probability, value in zip(probabilities, quantiles)}}


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty analysis: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_export_dirs(export_dirs, output_dir, n_bootstrap=1000, bootstrap_seed=2026):
    """Analyze saved features only, using the first export as comparison reference."""
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite analysis output: {output_dir}")
    directories = [Path(directory).resolve() for directory in export_dirs]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("At least one distinct export directory is required")
    if n_bootstrap < 1:
        raise ValueError("--n_bootstrap must be positive")
    query_rows, metric_rows, geometry_rows, bootstrap_records, sources = [], [], [], [], []
    checkpoint_summaries, ordering_reference = [], None
    for export_index, directory in enumerate(directories):
        feature_paths = sorted((directory / "features").glob("*.pt"))
        if not feature_paths:
            raise FileNotFoundError(f"No modality features in {directory}; re-export with --save_modality_features")
        scenario_names = {path.stem for path in feature_paths}
        core = set(SCENARIO_NAMES)
        if scenario_names not in (core, core | set(TRAIN_STATE_SCENARIO_NAMES)):
            raise ValueError("Feature analysis requires all 11 core scenarios, optionally all six training states")
        records, scenario_summaries, checkpoint_ordering = [], {}, {}
        for feature_path in feature_paths:
            features = torch.load(feature_path, map_location="cpu", weights_only=False)
            scenario = feature_path.stem
            if (features.get("format") != "p4_additive_modality_features_v1"
                    or features.get("scenario") != scenario):
                raise ValueError(f"Feature scenario mismatch: {feature_path}")
            payload = torch.load(directory / "scores" / f"{scenario}.pt", map_location="cpu", weights_only=False)
            _validate_pairwise_payload(payload, scenario)
            result = decompose_features(features, payload)
            n_queries, n_gallery = result["fused_scores"].shape
            for key, count in (("txt_indices", n_queries), ("img_indices", n_gallery)):
                expected_indices = torch.arange(count)
                if (not torch.equal(features[key].cpu().long(), expected_indices)
                        or not torch.equal(payload[key].cpu().long(), expected_indices)):
                    raise ValueError(f"Noncanonical feature order: {feature_path}: {key}")
            query_ids, raw_pids = features["query_ids"], features["query_raw_pids"].tolist()
            captions = features["captions"]
            if (len(query_ids) != n_queries or len(set(query_ids)) != n_queries
                    or len(raw_pids) != n_queries or len(captions) != n_queries
                    or len(features["gallery_raw_pids"]) != n_gallery
                    or len(features["image_ids"]) != n_gallery
                    or len(set(features["image_ids"].tolist())) != n_gallery):
                raise ValueError(f"Invalid feature query identities: {feature_path}")
            if (len(features["gallery_annotation_indices"]) != n_gallery
                    or features["image_ids"].tolist() != features["gallery_annotation_indices"]):
                raise ValueError("Feature image IDs must match source annotation indices")
            ordering = {
                "query_ids": query_ids, "query_raw_pids": raw_pids, "captions": captions,
                "gallery_raw_pids": features["gallery_raw_pids"].tolist(),
                "txt_pids": features["txt_pids"].tolist(), "img_pids": features["img_pids"].tolist(),
                "image_ids": features["image_ids"].tolist(),
                "gallery_annotation_indices": features.get("gallery_annotation_indices", []),
                "gallery_file_paths": features.get("gallery_file_paths", []),
                "split_identity_sha256": features["metadata"].get("split_identity_sha256"),
            }
            checkpoint_ordering[scenario] = ordering
            if len(checkpoint_ordering) > 1 and ordering != next(iter(checkpoint_ordering.values())):
                raise ValueError("Query/gallery ordering changes across feature scenarios")
            for i, decomposition in enumerate(result["query_rows"]):
                query_rows.append({"checkpoint_index": export_index, "export_dir": str(directory),
                                   "scenario": scenario, "query_id": query_ids[i], "pid": raw_pids[i],
                                   **decomposition,
                                   **{f"{probe}_{key}": int(stats[key][i]) if key == "best_pos_rank" else float(stats[key][i])
                                      for probe, stats in result["retrieval"].items()
                                      for key in ("best_pos_rank", "AP")}})
            for probe, stats in result["retrieval"].items():
                metric_rows.append({"checkpoint_index": export_index, "export_dir": str(directory),
                                    "scenario": scenario, "probe": probe,
                                    "R1": stats["R1"], "mAP": stats["mAP"]})
                for i in range(n_queries):
                    records.append({"scenario": f"{scenario}/{probe}", "query_id": query_ids[i],
                                    "pid": raw_pids[i], "caption": captions[i],
                                    "n_pos": int(stats["n_pos"][i]),
                                    "best_pos_rank": int(stats["best_pos_rank"][i]), "AP": float(stats["AP"][i])})
            for measurement, stats in result["geometry"].items():
                geometry_rows.append({"checkpoint_index": export_index, "export_dir": str(directory),
                                      "scenario": scenario, "measurement": measurement, **stats})
            scenario_summaries[scenario] = {
                "max_decomposition_error": result["max_decomposition_error"],
                "max_saved_score_error": result["max_saved_score_error"],
                "metrics": {probe: {key: stats[key] for key in ("R1", "mAP")}
                            for probe, stats in result["retrieval"].items()},
                "geometry": result["geometry"],
                "negative_contribution_fractions": result["negative_contribution_fractions"],
            }
            sources.append(str(feature_path))
        checkpoint_identity = next(iter(checkpoint_ordering.values()))
        if ordering_reference is None:
            ordering_reference = checkpoint_identity
        elif checkpoint_identity != ordering_reference:
            raise ValueError("Checkpoint canonical query/gallery ordering differs")
        bootstrap_records.append(records)
        checkpoint_summaries.append({"export_dir": str(directory), "scenarios": scenario_summaries})
    comparisons = []
    for index in range(1, len(bootstrap_records)):
        common = {row["scenario"] for row in bootstrap_records[0]} & {row["scenario"] for row in bootstrap_records[index]}
        reference = [row for row in bootstrap_records[0] if row["scenario"] in common]
        candidate = [row for row in bootstrap_records[index] if row["scenario"] in common]
        comparison = paired_bootstrap(reference, candidate,
                                      n_resamples=n_bootstrap, seed=bootstrap_seed)
        # This analysis reports each scenario/probe separately. The legacy
        # bootstrap helper's aggregate assumes plain core names, so omit it here.
        comparison.pop("mean_all", None)
        comparison.pop("mean_degraded", None)
        comparison.pop("scenario_weighting", None)
        comparisons.append({"reference_export": str(directories[0]), "comparison_export": str(directories[index]),
                            "comparison_scenarios": sorted({name.rsplit("/", 1)[0] for name in common}),
                            "missing_train_scenarios_are_not_imputed": True,
                            **comparison})
    summary = {
        "format": "p4_modality_decomposition_v1", "device": "cpu", "sources": sources,
        "score_semantics": "L2-normalized text dot L2-normalized f_rgb+f_ir; no logit scale",
        "contribution_share_semantics": "signed part/total only when total>0; otherwise null; shares are not probabilities",
        "positive_semantics": "all-positive mean and best positive under fused score; margin uses strongest fused-score negative",
        "fusion_ranking_semantics": "saved sim anchors original fused ranks, AP and candidate ties; CPU feature scores only reconstruct within 1e-5",
        "P_feat_semantics": "normalize each saved single modality feature; no other-modality input zeroing or reencoding",
        "zero_norm_semantics": "F.normalize eps=1e-12; a zero feature has zero cosine scores",
        "checkpoints": checkpoint_summaries, "paired_checkpoint_comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_query_contributions.csv", query_rows)
    _write_csv(output_dir / "probe_metrics.csv", metric_rows)
    _write_csv(output_dir / "feature_geometry.csv", geometry_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    cli = parser.parse_args()
    result = analyze_export_dirs(cli.export_dirs, cli.output_dir, cli.n_bootstrap, cli.bootstrap_seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
