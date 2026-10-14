"""CPU-only summary of saved P4 query records and training logs.

No score matrices, checkpoints, images or GPU are needed. All new output goes
to a new/empty directory. Recorded checkpoint hashes are provenance claims,
not an independent verification of unavailable checkpoint weights.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

import torch

CODE_ROOT = Path(__file__).resolve().parents[3]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.p4.analysis.mixed_gallery_from_scores import clustered_differences
from tools.p4.export_dev_records import SCENARIO_NAMES, TRAIN_STATE_SCENARIO_NAMES
from tools.p4.paired_bootstrap import load_records
from tools.p4.train_batch_stats import TRAIN_STATES, load_kappa

GROUPS = {
    "infrared_degraded": ("ir_missing", "ir_noise_005", "ir_noise_010", "ir_blur_k7"),
    "rgb_degraded": ("rgb_missing", "rgb_noise_005", "rgb_noise_010", "rgb_blur_k7"),
    "both_degraded": ("both_noise_005", "both_blur_k7"),
    "degraded_core10": SCENARIO_NAMES[1:],
}
METRICS = ("R1", "R5", "R10", "mAP", "mINP")
QUERY_METRICS = ("R1", "R5", "R10", "mAP")
QUANTILES = (.05, .25, .50, .75, .95)


def file_source(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def distribution(values):
    values = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if values.ndim != 1 or not len(values) or not torch.isfinite(values).all():
        raise ValueError("Distribution requires nonempty finite observations")
    return {"n": len(values), "mean": float(values.mean()), "std": float(values.std(unbiased=False)),
            "std_definition": "population", "min": float(values.min()), "max": float(values.max()),
            "quantiles": {str(q): float(torch.quantile(values, q)) for q in QUANTILES}}


def load_export_records(export_dir):
    """Validate 11/17 scenes, identity/query order and aggregate metric agreement."""
    export_dir = Path(export_dir).resolve()
    rows = load_records(export_dir / "per_query.csv")
    by_scene, seen = {}, set()
    for row in rows:
        key = (row["scenario"], row["query_id"])
        if key in seen:
            raise ValueError(f"Duplicate scenario/query row: {key}")
        seen.add(key)
        for field in ("n_tokens", "n_color", "rho"):
            if field not in row:
                raise ValueError(f"Missing query attribute: {field}")
        row["n_tokens"], row["n_color"], row["rho"] = int(row["n_tokens"]), int(row["n_color"]), float(row["rho"])
        if (row["n_tokens"] < 0 or not 0 <= row["n_color"] <= row["n_tokens"] or not math.isfinite(row["rho"])
                or not 0 <= row["rho"] <= 1):
            raise ValueError("Invalid color/token query attributes")
        expected_rho = row["n_color"] / row["n_tokens"] if row["n_tokens"] else 0.
        if not math.isclose(row["rho"], expected_rho, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("rho disagrees with n_color/n_tokens")
        by_scene.setdefault(row["scenario"], []).append(row)
    core = set(SCENARIO_NAMES)
    complete = core | set(TRAIN_STATE_SCENARIO_NAMES)
    if set(by_scene) not in (core, complete):
        raise ValueError("Export requires exactly eleven core or all seventeen scenarios")
    query_fields = ("query_id", "pid", "caption", "n_pos", "n_tokens", "n_color", "rho")
    queries = by_scene["clean"]
    expected_order = [tuple(row[field] for field in query_fields) for row in queries]
    for scenario, records in by_scene.items():
        if [tuple(row[field] for field in query_fields) for row in records] != expected_order:
            raise ValueError(f"Query identity/attribute/order differs across scenarios: {scenario}")
    with (export_dir / "metrics.tsv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"scenario", *METRICS} <= set(reader.fieldnames or []):
            raise ValueError("metrics.tsv missing retrieval fields")
        metrics = {}
        for row in reader:
            if row["scenario"] in metrics:
                raise ValueError("Duplicate metrics scenario")
            parsed = {metric: float(row[metric]) for metric in METRICS}
            if any(not math.isfinite(value) or not 0 <= value <= 100 for value in parsed.values()):
                raise ValueError("Invalid percentage metrics")
            metrics[row["scenario"]] = parsed
    if set(metrics) != set(by_scene):
        raise ValueError("metrics.tsv/per_query.csv scenario lists disagree")
    for scenario, records in by_scene.items():
        actual = {"R1": sum(row["best_pos_rank"] <= 1 for row in records) / len(records) * 100,
                  "R5": sum(row["best_pos_rank"] <= 5 for row in records) / len(records) * 100,
                  "R10": sum(row["best_pos_rank"] <= 10 for row in records) / len(records) * 100,
                  "mAP": sum(row["AP"] for row in records) / len(records) * 100}
        for metric, value in actual.items():
            if abs(value - metrics[scenario][metric]) > 2e-4:
                raise ValueError(f"Query/metrics disagreement: {scenario}/{metric}")
    summary_path = export_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    if summary is not None:
        provenance = summary.get("provenance", {})
        if "n_queries" in provenance and int(provenance["n_queries"]) != len(queries):
            raise ValueError("Export summary query count disagrees")
        for key, actual in (("clean_R1", metrics["clean"]["R1"]), ("clean_mAP", metrics["clean"]["mAP"]),
                            ("degraded_R1_mean", sum(metrics[scene]["R1"] for scene in SCENARIO_NAMES[1:]) / 10)):
            if key in summary and abs(float(summary[key]) - actual) > 2e-4:
                raise ValueError(f"Export summary metric disagrees: {key}")
    sources = {"per_query": file_source(export_dir / "per_query.csv"),
               "metrics": file_source(export_dir / "metrics.tsv")}
    if summary_path.is_file():
        sources["summary"] = file_source(summary_path)
    return {"export_dir": str(export_dir), "queries": queries, "by_scene": by_scene, "metrics": metrics,
            "export_summary_recorded": summary, "sources": sources}


def available_probes(scenario):
    if scenario == "clean":
        return (("P-zero_RGB-only", "ir_missing"), ("P-zero_IR-only", "rgb_missing"))
    if scenario in ("rgb_noise_005", "rgb_noise_010", "rgb_blur_k7"):
        return (("P-zero_IR-only", "rgb_missing"),)
    if scenario in ("ir_noise_005", "ir_noise_010", "ir_blur_k7"):
        return (("P-zero_RGB-only", "ir_missing"),)
    return ()


def failure_labels(by_scene):
    """Keep fusion failure and A/B flags independently; denominator is all queries."""
    summaries, records = [], []
    for scenario, rows in by_scene.items():
        probes = available_probes(scenario)
        computed = bool(probes)
        failures = a_count = b_count = oracle_hits = 0
        for index, row in enumerate(rows):
            rank = row["best_pos_rank"]
            failure = rank > 1
            failures += int(failure)
            probe_ranks = {name: by_scene[source][index]["best_pos_rank"] for name, source in probes}
            a_label = failure and any(value == 1 for value in probe_ranks.values()) if computed else None
            b_label = failure and all(value > 1 for value in probe_ranks.values()) and rank <= 10 if computed else None
            oracle_rank = min(rank, *probe_ranks.values()) if computed else None
            if computed:
                a_count += int(a_label)
                b_count += int(b_label)
                oracle_hits += int(oracle_rank == 1)
            records.append({"scenario": scenario, "query_id": row["query_id"], "pid": row["pid"],
                            "fusion_rank": rank, "fusion_failure": failure, "label_A": a_label, "label_B": b_label,
                            "oracle_rank": oracle_rank, "probe_RGB_only_rank": probe_ranks.get("P-zero_RGB-only"),
                            "probe_IR_only_rank": probe_ranks.get("P-zero_IR-only"),
                            "probe_status": "computed" if computed else "not_computed_no_defined_probe"})
        n = len(rows)
        summaries.append({"scenario": scenario, "n_queries": n, "fusion_failures": failures,
                          "fusion_failure_percent": failures / n * 100,
                          "label_A_count": a_count if computed else None,
                          "label_A_percent": a_count / n * 100 if computed else None,
                          "label_B_count": b_count if computed else None,
                          "label_B_percent": b_count / n * 100 if computed else None,
                          "oracle_R1": oracle_hits / n * 100 if computed else None,
                          "probe_status": "computed" if computed else "not_computed_no_defined_probe"})
    return summaries, records


def fit_regressions(rho, n_tokens, responses, pids, n_resamples=1000, seed=2026):
    """OLS y=(scenario-clean) AP percentage points; standardize unique queries once."""
    rho, n_tokens, responses = [torch.as_tensor(value, dtype=torch.float64, device="cpu")
                                 for value in (rho, n_tokens, responses)]
    if responses.ndim == 1:
        responses = responses[:, None]
    if len(rho) != len(n_tokens) or responses.shape[0] != len(rho) or len(pids) != len(rho) or n_resamples < 1:
        raise ValueError("Regression input sizes/resample count disagree")
    if not all(bool(torch.isfinite(value).all()) for value in (rho, n_tokens, responses)):
        raise ValueError("Regression observations must be finite")
    std_rho, std_tokens = rho.std(unbiased=False), n_tokens.std(unbiased=False)
    standardization = {"rho_mean": float(rho.mean()), "rho_std": float(std_rho),
                       "n_tokens_mean": float(n_tokens.mean()), "n_tokens_std": float(std_tokens),
                       "std_definition": "population over unique queries"}
    if not std_rho > 0 or not std_tokens > 0:
        return {"status": "not_identifiable_constant_covariate", "standardization": standardization,
                "coefficients": None, "ci95": None, "bootstrap_valid": 0, "n_queries": len(rho)}
    x = torch.stack((torch.ones_like(rho), (rho - rho.mean()) / std_rho,
                     (n_tokens - n_tokens.mean()) / std_tokens), dim=1)
    if int(torch.linalg.matrix_rank(x)) < 3:
        return {"status": "not_identifiable_collinear_covariates", "standardization": standardization,
                "coefficients": None, "ci95": None, "bootstrap_valid": 0, "n_queries": len(rho)}
    coefficients = torch.linalg.lstsq(x, responses, driver="gelsd").solution
    unique_pids = sorted(set(int(pid) for pid in pids))
    mapping = {pid: index for index, pid in enumerate(unique_pids)}
    inverse = torch.tensor([mapping[int(pid)] for pid in pids], dtype=torch.long)
    n_clusters = len(unique_pids)
    xtx = torch.zeros((n_clusters, 3, 3), dtype=torch.float64)
    xty = torch.zeros((n_clusters, 3, responses.shape[1]), dtype=torch.float64)
    xtx.index_add_(0, inverse, x[:, :, None] * x[:, None, :])
    xty.index_add_(0, inverse, x[:, :, None] * responses[:, None, :])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    draws = torch.randint(n_clusters, (n_resamples, n_clusters), generator=generator)
    counts = torch.zeros((n_resamples, n_clusters), dtype=torch.float64)
    counts.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
    # Use CPU torch reductions, not NumPy BLAS (avoids duplicate OpenMP runtimes).
    bootstrap_xtx = (counts[:, :, None, None] * xtx[None, :, :, :]).sum(1)
    bootstrap_xty = (counts[:, :, None, None] * xty[None, :, :, :]).sum(1)
    eigenvalues = torch.linalg.eigvalsh(bootstrap_xtx)
    valid = eigenvalues[:, 0] > eigenvalues[:, -1].clamp_min(1.) * 1e-12
    if bool(valid.any()):
        samples = torch.linalg.solve(bootstrap_xtx[valid], bootstrap_xty[valid])
        intervals = torch.quantile(samples, torch.tensor([.025, .975], dtype=torch.float64), dim=0)
        ci95 = intervals.tolist()
    else:
        ci95 = None
    return {"status": "ok" if bool(valid.any()) else "point_estimate_only_no_full_rank_bootstrap",
            "standardization": standardization, "coefficients": coefficients.tolist(), "ci95": ci95,
            "bootstrap_valid": int(valid.sum()), "bootstrap_requested": n_resamples,
            "n_queries": len(rho), "n_pids": n_clusters}


def parse_validation_log(path):
    """A validation marker owns only its first subsequent t2i table row."""
    pending_epoch, pending_line = None, None
    rows, seen = [], set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        marker = re.search(r"Validation Results - Epoch:\s*(\d+)", line)
        if marker:
            if pending_epoch is not None:
                raise ValueError(f"Validation epoch {pending_epoch} has no t2i table")
            pending_epoch, pending_line = int(marker.group(1)), line_number
            continue
        if pending_epoch is not None and re.search(r"\|\s*t2i\s*\|", line):
            parts = [item.strip() for item in line.split("|")]
            try:
                index = parts.index("t2i")
                values = [float(value) for value in parts[index + 1:index + 6]]
            except (ValueError, IndexError) as error:
                raise ValueError(f"Malformed validation t2i table at line {line_number}") from error
            if len(values) != 5 or any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
                raise ValueError("Invalid validation metrics")
            if pending_epoch in seen:
                raise ValueError(f"Duplicate validation epoch {pending_epoch}")
            seen.add(pending_epoch)
            rows.append({"epoch": pending_epoch, **dict(zip(METRICS, values)),
                         "marker_line": pending_line, "table_line": line_number})
            pending_epoch, pending_line = None, None
    if pending_epoch is not None:
        raise ValueError(f"Validation epoch {pending_epoch} has no t2i table")
    return rows


def load_training_run(run_dir):
    run_dir = Path(run_dir).resolve()
    record_path, log_path = run_dir / "run_record.json", run_dir / "train_log.txt"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("epochs"), list):
        raise ValueError("run_record.json must contain recorded epochs")
    epoch_rows, seen = [], set()
    totals = {"text_samples": 0, "epoch_unique_image_exposures": 0, "degraded_copies": 0,
              "unique_images_union": None,
              "recorded_step_rows": len(record["steps"]) if isinstance(record.get("steps"), list) else None,
              "image_encoder_calls": {}, "degradation_mode_counts": {}}
    for row in record["epochs"]:
        required = {"epoch", "text_samples", "unique_images", "degraded_copies", "image_encoder_calls", "degradation_mode_counts"}
        if not required <= set(row):
            raise ValueError(f"Recorded epoch missing counters: {sorted(required - set(row))}")
        epoch = int(row["epoch"])
        if epoch in seen:
            raise ValueError("run_record contains duplicate epoch")
        seen.add(epoch)
        flat = {"epoch": epoch, "partial": row.get("partial")}
        for field in ("text_samples", "unique_images", "degraded_copies"):
            value = row[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("Training counter must be a nonnegative recorded integer")
            totals["epoch_unique_image_exposures" if field == "unique_images" else field] += value
            flat[field] = value
        for field in ("image_encoder_calls", "degradation_mode_counts"):
            if not isinstance(row[field], dict):
                raise ValueError("Training call/mode counters must be mappings")
            for key, value in row[field].items():
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError("Training counter must be a nonnegative recorded integer")
                totals[field][key] = totals[field].get(key, 0) + value
                flat[f"{field}.{key}"] = value
        flat["seconds"] = row.get("seconds")
        flat["peak_memory_bytes"] = row.get("peak_memory_bytes")
        epoch_rows.append(flat)
    validation = parse_validation_log(log_path)
    if any(row["epoch"] not in seen for row in validation):
        raise ValueError("Validation log epoch absent from run_record")
    return {"run_dir": str(run_dir), "recorded_run": record, "totals": totals,
            "epoch_counts": epoch_rows, "validation_metrics": validation,
            "sources": {"run_record": file_source(record_path), "train_log": file_source(log_path)},
            "recorded_step_rows_semantics": "Number of detailed rows serialized in run_record.steps; "
                                            "zero does not mean zero training batches when per-step recording is disabled",
            "unique_images_semantics": "Per-epoch unique_images retained; epoch_unique_image_exposures sums those counts; "
                                       "the distinct-image union over the entire run was not recorded"}


def measured_kappas(kappa_json=None, manual=None):
    result = load_kappa(kappa_json) if kappa_json is not None else {}
    for specification in manual or []:
        if "=" not in specification:
            raise ValueError("Manual --kappa must be state=value")
        state, value = specification.split("=", 1)
        if state not in TRAIN_STATES or state in result:
            raise ValueError(f"Unknown/duplicate kappa state: {state}")
        raw = float(value)
        if not math.isfinite(raw):
            raise ValueError("Kappa must be finite")
        result[state] = {"raw": raw, "clipped": min(1., max(0., raw)),
                         "source": {"kind": "manual", "argument": specification}}
    return result


def _vectors(rows):
    return torch.tensor([[int(row["best_pos_rank"] <= 1) * 100., int(row["best_pos_rank"] <= 5) * 100.,
                          int(row["best_pos_rank"] <= 10) * 100., row["AP"] * 100.]
                         for row in rows], dtype=torch.float64)


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_one(export, run, kappas, n_resamples, seed):
    queries, by_scene = export["queries"], export["by_scene"]
    rho, tokens, pids = [row["rho"] for row in queries], [row["n_tokens"] for row in queries], [row["pid"] for row in queries]
    scenes = [name for name in (*SCENARIO_NAMES, *TRAIN_STATE_SCENARIO_NAMES) if name in by_scene]
    metrics = [{"scenario": scene, **export["metrics"][scene]} for scene in scenes]
    groups = [{"group": group, "n_scenarios": len(names),
               **{metric: sum(export["metrics"][scene][metric] for scene in names) / len(names) for metric in METRICS}}
              for group, names in GROUPS.items()]
    bins = [{"scenario": scene,
             "rank_1": sum(row["best_pos_rank"] == 1 for row in by_scene[scene]),
             "rank_2_5": sum(2 <= row["best_pos_rank"] <= 5 for row in by_scene[scene]),
             "rank_6_10": sum(6 <= row["best_pos_rank"] <= 10 for row in by_scene[scene]),
             "rank_gt10": sum(row["best_pos_rank"] > 10 for row in by_scene[scene]), "n_queries": len(queries)} for scene in scenes]
    labels, label_queries = failure_labels(by_scene)
    responses = torch.tensor([[row["AP"] - clean["AP"] for row, clean in zip(by_scene[scene], by_scene["clean"])]
                              for scene in scenes], dtype=torch.float64).T * 100
    fitted = fit_regressions(rho, tokens, responses, pids, n_resamples, seed)
    regressions = []
    for scene_index, scene in enumerate(scenes):
        for coefficient_index, coefficient in enumerate(("intercept", "z_rho", "z_n_tokens")):
            regressions.append({"scenario": scene, "coefficient": coefficient, "status": fitted["status"],
                                "value": fitted["coefficients"][coefficient_index][scene_index] if fitted["coefficients"] else None,
                                "ci95_low": fitted["ci95"][0][coefficient_index][scene_index] if fitted["ci95"] else None,
                                "ci95_high": fitted["ci95"][1][coefficient_index][scene_index] if fitted["ci95"] else None,
                                "bootstrap_valid": fitted["bootstrap_valid"]})
    k2 = [{"state": state, "raw_kappa": item["raw"], "clipped_kappa": item["clipped"],
           "source": item["source"], **distribution([max(.05, 1 - value * item["clipped"]) for value in rho])}
          for state, item in kappas.items()]
    # Logs are printed to 3 decimals; last exports preserve float32 metrics.
    log_consistency = "no_validation_rows_recorded"
    if run["validation_metrics"]:
        last = run["validation_metrics"][-1]
        if abs(last["R1"] - export["metrics"]["clean"]["R1"]) > .00051:
            log_consistency = "last_validation_R1_differs_from_export_clean_R1"
        else:
            log_consistency = "last_validation_R1_matches_export_with_3_decimal_tolerance"
    return {"metrics": metrics, "group_means": groups, "rank_bins": bins, "failure_labels": labels,
            "failure_query_records": label_queries, "rho_distribution": distribution(rho),
            "n_tokens_distribution": distribution(tokens), "regressions": regressions,
            "regression_definition": "(scene AP-clean AP)*100 ~ intercept + z(rho) + z(n_tokens)",
            "regression_standardization": fitted["standardization"], "k2": k2,
            "training": run, "last_validation_export_consistency": log_consistency}


def run_summary(export_dirs, run_dirs, output_dir, names=None, kappa_json=None, kappa=None,
                n_resamples=1000, seed=2026):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Output directory is nonempty; existing outputs are never overwritten")
    if not export_dirs or len(export_dirs) != len(run_dirs) or n_resamples < 1:
        raise ValueError("Supply one run directory per export and positive bootstrap count")
    names = [Path(path).name for path in export_dirs] if names is None else list(names)
    if len(names) != len(export_dirs) or len(set(names)) != len(names):
        raise ValueError("Names must be unique and correspond one-to-one with exports")
    exports = [load_export_records(path) for path in export_dirs]
    runs = [load_training_run(path) for path in run_dirs]
    query_fields = ("query_id", "pid", "caption", "n_pos", "n_tokens", "n_color", "rho")
    reference = [tuple(row[field] for field in query_fields) for row in exports[0]["queries"]]
    for export in exports[1:]:
        if [tuple(row[field] for field in query_fields) for row in export["queries"]] != reference:
            raise ValueError("Cross-model query order/identity/attributes disagree")
    semantic_hashes = {export["export_summary_recorded"].get("provenance", {}).get("split_identity_sha256")
                       for export in exports if export["export_summary_recorded"] is not None}
    semantic_hashes.discard(None)
    if len(semantic_hashes) > 1:
        raise ValueError("Cross-model recorded development split hashes disagree")
    for export, run in zip(exports, runs):
        provenance = (export["export_summary_recorded"] or {}).get("provenance", {})
        checkpoint = provenance.get("checkpoint")
        recorded_output = run["recorded_run"].get("args", {}).get("output_dir")
        if checkpoint and recorded_output and Path(checkpoint).parent.name != Path(recorded_output).name:
            raise ValueError("Supplied training run does not match export's recorded checkpoint directory")
        config_path = Path(run["run_dir"]) / "configs.yaml"
        expected_config_hash = provenance.get("config", {}).get("sha256")
        if expected_config_hash and config_path.is_file() and file_source(config_path)["sha256"] != expected_config_hash:
            raise ValueError("Training run config hash differs from export's recorded config")
    kappas = measured_kappas(kappa_json, kappa)
    results = [summarize_one(export, run, kappas, n_resamples, seed) for export, run in zip(exports, runs)]
    comparisons, query_changes, point_only_differences = [], [], []
    pids = [row["pid"] for row in exports[0]["queries"]]
    for a, b in itertools.combinations(range(len(exports)), 2):
        common = [scene for scene in (*SCENARIO_NAMES, *TRAIN_STATE_SCENARIO_NAMES)
                  if scene in exports[a]["by_scene"] and scene in exports[b]["by_scene"]]
        deltas = {scene: _vectors(exports[b]["by_scene"][scene]) - _vectors(exports[a]["by_scene"][scene]) for scene in common}
        for condition, scenes in [(scene, (scene,)) for scene in common] + list(GROUPS.items()):
            values = torch.stack([deltas[scene] for scene in scenes]).mean(0).numpy()
            intervals = clustered_differences(values, pids, n_resamples, seed)
            changed_correct = changed_wrong = None
            if len(scenes) == 1:
                scene = scenes[0]
                changed_correct = changed_wrong = 0
                for row_a, row_b in zip(exports[a]["by_scene"][scene], exports[b]["by_scene"][scene]):
                    hit_a, hit_b = row_a["best_pos_rank"] == 1, row_b["best_pos_rank"] == 1
                    became_correct, became_wrong = not hit_a and hit_b, hit_a and not hit_b
                    changed_correct += int(became_correct)
                    changed_wrong += int(became_wrong)
                    query_changes.append({"name_A": names[a], "name_B": names[b], "scenario": scene,
                                          "query_id": row_a["query_id"], "pid": row_a["pid"],
                                          "rank_A": row_a["best_pos_rank"], "rank_B": row_b["best_pos_rank"],
                                          "AP_difference_pp": (row_b["AP"] - row_a["AP"]) * 100,
                                          "became_correct": became_correct, "became_wrong": became_wrong})
            for metric, item in zip(QUERY_METRICS, intervals):
                comparisons.append({"name_A": names[a], "name_B": names[b], "condition": condition,
                                    "metric": metric, "difference_B_minus_A": item["difference"],
                                    "ci95_low": item["ci95"][0], "ci95_high": item["ci95"][1],
                                    "changed_correct": changed_correct, "changed_wrong": changed_wrong,
                                    "n_queries": len(pids), "n_scenarios": len(scenes)})
            point_only_differences.append({"name_A": names[a], "name_B": names[b], "condition": condition,
                                          "metric": "mINP",
                                          "point_difference_B_minus_A": sum(exports[b]["metrics"][scene]["mINP"]
                                                                             - exports[a]["metrics"][scene]["mINP"]
                                                                             for scene in scenes) / len(scenes),
                                          "ci95": None,
                                          "reason": "per_query.csv omits last-positive rank; query mINP is not identifiable"})
    output_dir.mkdir(parents=True, exist_ok=True)
    public = []
    for index, (name, export, result) in enumerate(zip(names, exports, results)):
        run_output = output_dir / f"{index:02d}_{Path(export['export_dir']).name}"
        run_output.mkdir()
        for field in ("metrics", "group_means", "rank_bins", "failure_labels", "failure_query_records", "regressions"):
            _write_csv(run_output / f"{field}.csv", result[field])
        _write_csv(run_output / "epoch_counts.csv", result["training"]["epoch_counts"])
        _write_csv(run_output / "validation_metrics.csv", result["training"]["validation_metrics"])
        _write_csv(run_output / "k2.csv", [{**item, "source": json.dumps(item["source"], ensure_ascii=False),
                                           "quantiles": json.dumps(item["quantiles"])} for item in result["k2"]])
        item = {"name": name, "export_dir": export["export_dir"], "sources": export["sources"],
                "recorded_export_provenance": export["export_summary_recorded"], **result}
        public.append(item)
        (run_output / "summary.json").write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "pairwise_comparisons.csv", comparisons)
    _write_csv(output_dir / "query_changes.csv", query_changes)
    _write_csv(output_dir / "point_only_differences.csv", point_only_differences)
    summary = {"device": "cpu", "bootstrap": {"n_resamples": n_resamples, "seed": seed,
                "cluster": "original pid", "interval": "percentile 95%", "weighting": "text queries"},
               "n_unique_queries": len(reference), "groups": {name: list(scenes) for name, scenes in GROUPS.items()},
               "kappa_source": file_source(kappa_json) if kappa_json else None,
               "provenance_limit": "Recorded export/run metadata and local file hashes only; no unavailable checkpoint weights verified",
               "exports": public, "pairwise_comparisons": comparisons,
               "point_only_differences": point_only_differences}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export_dirs", nargs="+", required=True)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--output_dir", required=True, help="New or empty directory")
    parser.add_argument("--kappa_json", default=None)
    parser.add_argument("--kappa", action="append", default=[], help="Measured/manual state=value; repeat for multiple states")
    parser.add_argument("--n_resamples", type=int, default=1000)
    parser.add_argument("--seed", "--bootstrap_seed", dest="seed", type=int, default=2026)
    return parser


if __name__ == "__main__":
    cli = build_parser().parse_args()
    result = run_summary(cli.export_dirs, cli.run_dirs, cli.output_dir, cli.names,
                         cli.kappa_json, cli.kappa, cli.n_resamples, cli.seed)
    print(json.dumps({"output_dir": str(Path(cli.output_dir).resolve()), "n_exports": len(result["exports"]),
                      "n_unique_queries": result["n_unique_queries"]}, ensure_ascii=False, indent=2))
