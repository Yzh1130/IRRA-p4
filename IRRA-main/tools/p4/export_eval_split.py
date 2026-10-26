"""Export dev or official-test features using the existing P4 evaluation path.

The split must be explicit. Use --save_modality_features to save the exact
pre-addition modality features. This entry point never trains or updates a model.
Existing development helpers and all eleven scenario calculations are reused.
"""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.p4 import export_dev_records as _development
from tools.p4.export_dev_records import (
    CORE_SCENARIOS, METRIC_FIELDS, QUERY_FIELDS, SCENARIO_NAMES,
    TRAIN_STATE_SCENARIOS, TRAIN_STATE_SCENARIO_NAMES,
    _eval_with_optional_features, _feature_scenarios, _file_hashes,
    _resolve_annotation, _resolve_saved_split, _write_opt_in_record,
    acd_missing_ir_route, build_summary, compare_exports, dropblank_metrics,
    load_acd_for_export, load_strict_checkpoint, load_word_list,
    make_train_state_evaluator, per_query_records, validate_dev_split,
)


def build_test_data(annotation, dev_split_file):
    """Build annotation-ordered official-test metadata without reading any images.

    The checkpoint's reduced-training identity signature is deliberately retained:
    official-test evaluation must still verify that it uses the same dev protocol.
    Contiguous test labels follow RGBTPEDES._process_anno's sorted raw IDs.
    """
    development = validate_dev_split(annotation, dev_split_file)
    annos = json.loads(Path(annotation).read_text(encoding="utf-8"))
    split = json.loads(Path(dev_split_file).read_text(encoding="utf-8"))
    development_ids = {int(pid) for pid in split["dev_ids"]}
    training_ids = {int(pid) for pid in split["train_reduced_ids"]}
    selected = [(index, anno) for index, anno in enumerate(annos)
                if anno.get("split", "train") == "test"]
    test_ids = {int(anno["id"]) for _, anno in selected}
    if test_ids & (development_ids | training_ids):
        raise ValueError("Official-test identities overlap training/development identities")
    if not test_ids:
        raise ValueError("Official-test split has no gallery identities")
    pid_map = {pid: index for index, pid in enumerate(sorted(test_ids))}
    gallery_counts = {pid: 0 for pid in test_ids}
    caption_counts = {pid: 0 for pid in test_ids}
    queries, gallery_pids, gallery_raw_pids = [], [], []
    for annotation_index, anno in selected:
        raw_pid = int(anno["id"])
        gallery_counts[raw_pid] += 1
        gallery_raw_pids.append(raw_pid)
        gallery_pids.append(pid_map[raw_pid])
        for caption_index, caption in enumerate(anno["captions"]):
            caption_counts[raw_pid] += 1
            queries.append({
                "query_id": f"{annotation_index}:{caption_index}",
                "pid": raw_pid, "caption": caption, "eval_pid": pid_map[raw_pid],
            })
    if any(gallery_counts[pid] == 0 or caption_counts[pid] == 0 for pid in test_ids):
        raise ValueError("An official-test identity has no gallery pair or zero captions")
    return {
        **development,
        "queries": queries, "gallery_pids": gallery_pids,
        "gallery_raw_pids": gallery_raw_pids,
        "test_records": len(selected), "test_captions": len(queries),
        "n_test_ids": len(test_ids),
    }


def _refuse_existing_output(output_dir):
    """The new entry point rejects an existing output before loading a model."""
    if Path(output_dir).exists():
        raise FileExistsError(f"Refusing to overwrite existing export output: {output_dir}")


def _test_provenance(data):
    # Test-only additions leave existing dev JSONs byte-for-byte unchanged.
    return {"split": "test", "n_test_ids": data["n_test_ids"],
            "test_records": data["test_records"], "test_captions": data["test_captions"]}


def ap_tie_bounds(score_payload):
    """Mean AP (percent) with exactly tied scores ordered worst and best for each query.

    An exact float tie between a positive and a negative gallery item makes AP depend on
    the sort implementation: the Evaluator sorts on the GPU, per_query_records on the CPU.
    Returns (worst, best, number of queries whose AP depends on the tie order). Labels and
    order must already have been verified by per_query_records.
    """
    import numpy as np

    sim = score_payload["sim"].detach().cpu().double().numpy()
    qids = score_payload["txt_pids"].detach().cpu().long().view(-1).numpy()
    gids = score_payload["img_pids"].detach().cpu().long().view(-1).numpy()
    best, worst, tied = [], [], 0
    for row, qid in zip(sim, qids):
        positive = gids == qid
        values = []
        for later in (~positive, positive):   # best: positives first among ties; worst: positives last
            hits = np.flatnonzero(positive[np.lexsort((later, -row))]) + 1
            values.append(float(np.mean(np.arange(1, len(hits) + 1) / hits)))
        best.append(values[0])
        worst.append(values[1])
        tied += values[0] != values[1]
    return 100.0 * float(np.mean(worst)), 100.0 * float(np.mean(best)), int(tied)


def check_per_query_ap(records, score_payload, metrics, scenario, split, logger, tie_checks):
    """Per-query AP must reproduce the Evaluator mAP; exact score ties may only move it within their bounds.

    Development split: unchanged when the original 2e-4 check passes (no extra output).
    Official test split: tie statistics are always recorded in tie_checks.
    """
    mean_ap = sum(row["AP"] for row in records) / len(records) * 100
    evaluator = float(metrics["mAP"])
    strict = abs(mean_ap - evaluator) <= 2e-4
    if strict and split != "test":
        return
    worst, best, tied = ap_tie_bounds(score_payload)
    if not strict and not (tied > 0 and worst - 2e-4 <= evaluator <= best + 2e-4):
        raise ValueError(
            f"Per-query AP disagrees with existing Evaluator mAP for {scenario}: per-query {mean_ap:.6f}, "
            f"tie bounds [{worst:.6f}, {best:.6f}] over {tied} tie-dependent queries, Evaluator {evaluator:.6f}")
    if not strict:
        logger.info("[P4_EXPORT_TIE] scenario=%s tie_dependent_queries=%d per_query_mAP=%.6f bounds=[%.6f, %.6f] evaluator_mAP=%.6f",
                    scenario, tied, mean_ap, worst, best, evaluator)
    if split == "test":
        tie_checks[scenario] = {"tie_dependent_queries": tied, "per_query_mean_ap": mean_ap,
                                "ap_bounds": [worst, best], "evaluator_map": evaluator}


def export_split(cli):
    _refuse_existing_output(cli.output_dir)
    # Deferred imports keep CSV/lexicon/bootstrap helpers independently CPU testable.
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    if cli.probe == "zero":
        raise NotImplementedError("P-zero probe is not added; existing rgb_missing/ir_missing encode zeroed inputs")
    if set(cli.scenarios) != set(SCENARIO_NAMES) or len(cli.scenarios) != len(SCENARIO_NAMES):
        raise ValueError("--scenarios must contain exactly the eleven P3 core scenario names")
    if int(cli.eval_degrade_seed) != 2026 and not bool(getattr(cli, "train_state_scenarios", False)):
        raise ValueError("P4 main development export fixes --eval_degrade_seed=2026")
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    import torch
    from datasets import build_dataloader
    from model import build_model
    from processor.processor import apply_img_local_filter_stage, restore_eval_extra_state
    from utils.iotools import load_train_configs
    from utils.logger import setup_logger
    from utils.metrics import Evaluator

    adapter_path = getattr(cli, "acd_adapter", None)
    comparison_dir = getattr(cli, "compare_export", None)
    save_features = bool(getattr(cli, "save_modality_features", False) or cli.probe == "feat"
                         or adapter_path or comparison_dir)
    if comparison_dir:
        # Fail before loading the visual encoder for missing/incomplete references.
        reference_paths = _feature_scenarios(comparison_dir)
        expected_names = set(SCENARIO_NAMES + (TRAIN_STATE_SCENARIO_NAMES if getattr(cli, "train_state_scenarios", False) else ()))
        if set(reference_paths) != expected_names:
            raise ValueError("Reference feature scenarios do not match requested export scenarios")
    if adapter_path and not Path(adapter_path).is_file():
        raise FileNotFoundError(adapter_path)

    config_path = Path(cli.config_file or Path(cli.checkpoint).parent / "configs.yaml").resolve()
    args = load_train_configs(str(config_path))
    if getattr(args, "dataset_name", None) != "RGBT-PEDES":
        raise ValueError("P4 development export is restricted to RGBT-PEDES")
    if cli.root_dir is not None:
        args.root_dir = str(Path(cli.root_dir).resolve())
    annotation = _resolve_annotation(args.root_dir)
    dev_data = (validate_dev_split(annotation, cli.dev_split_file) if cli.split == "dev"
                else build_test_data(annotation, cli.dev_split_file))
    training_split, saved_training_split = _resolve_saved_split(args, config_path, cli.training_split_file)
    training_data = validate_dev_split(annotation, training_split)
    if training_data["split_identity_sha256"] != dev_data["split_identity_sha256"]:
        raise ValueError("Checkpoint training split does not match supplied development split")
    # Prevent optional research rerank/diagnostic paths stored in another run from
    # writing outside this export. These are outside the base_clean/c0 protocol.
    unsupported = [name for name in (
        "use_qalr", "use_qalr_v2", "use_qalr_v3_gate", "use_qalr_v3_two_stage_gate",
        "qalr_v3_candidate_diag", "use_transport_rerank", "use_uspc_slot_rerank",
    ) if bool(getattr(args, name, False))]
    if unsupported:
        raise ValueError(f"P4 base/c0 export expects global retrieval without optional rerankers: {unsupported}")
    output_dir = Path(cli.output_dir).resolve()
    checkpoint_path = Path(cli.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if save_features:
        for scenario in SCENARIO_NAMES + (TRAIN_STATE_SCENARIO_NAMES if getattr(cli, "train_state_scenarios", False) else ()):
            if (output_dir / "features" / f"{scenario}.pt").exists():
                raise FileExistsError(f"Refusing to overwrite modality features: {scenario}")
        for filename in ("metrics_dropblank.tsv", "export_run_record.json") + (("invariance.json",) if comparison_dir else ()):
            if (output_dir / filename).exists():
                raise FileExistsError(f"Refusing to overwrite opt-in export output: {filename}")
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
    args.training = False
    args.distributed = False
    args.eval_dataset = "val" if cli.split == "dev" else "test"
    args.dev_split_file = str(Path(cli.dev_split_file).resolve())
    args.output_dir = str(output_dir)
    args.config_file = str(config_path)
    args.test_ckpt_file = str(checkpoint_path)
    args.save_eval_scores = True
    args.eval_scores_save_raw = False
    if cli.batch_size is not None:
        args.test_batch_size = cli.batch_size
    if cli.num_workers is not None:
        args.num_workers = cli.num_workers
    img_loader, txt_loader, num_classes = build_dataloader(args)
    if num_classes != dev_data["n_train"]:
        raise ValueError("Evaluation loader did not use the verified reduced training split")
    loader_captions = list(getattr(txt_loader.dataset, "captions", []))
    if loader_captions != [q["caption"] for q in dev_data["queries"]]:
        raise ValueError("Caption loader order does not match annotation order")
    model = build_model(args, num_classes=num_classes)
    checkpoint = load_strict_checkpoint(model, checkpoint_path)
    apply_img_local_filter_stage(model, args, int(checkpoint.get("epoch", 1)), logger=None, prefix="P4_EXPORT")
    if checkpoint.get("full_eval_state_checkpoint", False):
        restore_eval_extra_state(model, checkpoint.get("extra_eval_state", {}), logger=None)
    model.requires_grad_(False)
    model.to(device).eval()
    acd_controller = load_acd_for_export(model, adapter_path, checkpoint_path) if adapter_path else None
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("IRRA", str(output_dir), if_train=False)
    color_words, stopwords = load_word_list(cli.color_lexicon), load_word_list(cli.stopwords)
    if save_features:
        # Only opt-in output paths are checked; all existing default writes remain unchanged.
        for scenario in SCENARIO_NAMES + (TRAIN_STATE_SCENARIO_NAMES if getattr(cli, "train_state_scenarios", False) else ()):
            if (output_dir / "features" / f"{scenario}.pt").exists():
                raise FileExistsError(f"Refusing to overwrite modality features: {scenario}")
        annos = json.loads(annotation.read_text(encoding="utf-8"))
        dev_ids = set(dev_data["gallery_raw_pids"])
        selected = [(i, a) for i, a in enumerate(annos)
                    if ((cli.split == "dev" and a.get("split", "train") == "train" and int(a["id"]) in dev_ids)
                        or (cli.split == "test" and a.get("split", "train") == "test"))]
        dev_data["gallery_annotation_indices"] = [i for i, _ in selected]
        dev_data["gallery_file_paths"] = [a["file_path"] for _, a in selected]
    metric_rows, query_rows, train_state_records, dropblank_rows = [], [], {}, []
    tie_checks = {}
    for scenario, mode, strength, kernel in CORE_SCENARIOS:
        args.eval_degrade_mode = mode
        args.eval_degrade_strength = strength
        args.eval_degrade_blur_kernel = kernel
        args.eval_degrade_seed = 2026
        args.eval_scores_file = str(output_dir / "scores" / f"{scenario}.pt")
        evaluator = Evaluator(img_loader, txt_loader, source="p4_dev_export")
        with acd_missing_ir_route(model, acd_controller, scenario):
            recorder = _eval_with_optional_features(evaluator, model, save_features)
        metrics = dict(evaluator.last_eval_result)
        if not all(math.isfinite(float(metrics[key])) for key in METRIC_FIELDS[1:]):
            raise ValueError(f"Nonfinite metrics for {scenario}")
        metric_rows.append({"scenario": scenario, **metrics})
        payload = torch.load(args.eval_scores_file, map_location="cpu", weights_only=False)
        if recorder is not None:
            features = recorder.save(output_dir / "features" / f"{scenario}.pt", payload, dev_data, scenario)
            dropblank_rows.append(dropblank_metrics(features, metrics, scenario))
        records = per_query_records(payload, dev_data, scenario, color_words, stopwords)
        check_per_query_ap(records, payload, metrics, scenario, cli.split, logger, tie_checks)
        query_rows.extend(records)
        logger.info("[P4_EXPORT] scenario=%s R1=%.6f mAP=%.6f", scenario, metrics["R1"], metrics["mAP"])
    if bool(getattr(cli, "train_state_scenarios", False)):
        for scenario, mode in TRAIN_STATE_SCENARIOS:
            # This mode is recorded in score metadata. Only this new evaluator's
            # hook interprets it; the eleven original evaluations above are intact.
            args.eval_degrade_mode = mode
            args.eval_degrade_strength = 0.1 if mode.endswith("_noise") else 1.0
            args.eval_degrade_blur_kernel = 7
            args.eval_degrade_seed = int(cli.eval_degrade_seed)
            args.eval_scores_file = str(output_dir / "scores" / f"{scenario}.pt")
            evaluator = make_train_state_evaluator(
                img_loader, txt_loader, mode, seed=int(cli.eval_degrade_seed),
                expected_gallery_size=len(dev_data["gallery_pids"]),
            )
            recorder = _eval_with_optional_features(evaluator, model, save_features)
            train_state_records[scenario] = evaluator.training_state_record()
            metrics = dict(evaluator.last_eval_result)
            if not all(math.isfinite(float(metrics[key])) for key in METRIC_FIELDS[1:]):
                raise ValueError(f"Nonfinite metrics for {scenario}")
            metric_rows.append({"scenario": scenario, **metrics})
            payload = torch.load(args.eval_scores_file, map_location="cpu", weights_only=False)
            if recorder is not None:
                features = recorder.save(output_dir / "features" / f"{scenario}.pt", payload, dev_data, scenario)
                dropblank_rows.append(dropblank_metrics(features, metrics, scenario))
            records = per_query_records(payload, dev_data, scenario, color_words, stopwords)
            check_per_query_ap(records, payload, metrics, scenario, cli.split, logger, tie_checks)
            query_rows.extend(records)
            logger.info("[P4_EXPORT] scenario=%s R1=%.6f mAP=%.6f", scenario, metrics["R1"], metrics["mAP"])
    provenance = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _file_hashes(checkpoint_path)["sha256"],
        "config": _file_hashes(config_path), "training_split_recorded": saved_training_split,
        "training_split_verified": training_data["split"],
        "development_split": dev_data["split"], "annotation": dev_data["annotation"],
        "split_identity_sha256": dev_data["split_identity_sha256"],
        "eval_degrade_seed": 2026, "n_dev": dev_data["n_dev"],
        "n_gallery_pairs": len(dev_data["gallery_pids"]), "n_queries": len(dev_data["queries"]),
        "color_lexicon": _file_hashes(cli.color_lexicon), "stopwords": _file_hashes(cli.stopwords),
        "query_id_semantics": "zero-based annotation index:caption index",
        "pid_semantics": "original annotation identity; evaluator labels are contiguous",
        "token_semantics": "casefolded English regex word occurrences after fixed stopword removal",
    }
    if cli.split == "test":
        provenance.update(_test_provenance(dev_data))
        provenance["ap_tie_check"] = tie_checks
    if train_state_records:
        provenance["train_state_eval_degrade_seed"] = int(cli.eval_degrade_seed)
        provenance["training_state_degradation"] = train_state_records
    with (output_dir / "metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(metric_rows)
    with (output_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUERY_FIELDS)
        writer.writeheader()
        writer.writerows(query_rows)
    summary = build_summary(metric_rows, provenance)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if cli.probe == "feat":
        from tools.p4.analysis.modality_decomposition import analyze_export_dirs

        analyze_export_dirs([output_dir], output_dir / "probes" / "feat")
    if save_features:
        with (output_dir / "metrics_dropblank.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(dropblank_rows)
        extra = {"checkpoint": _file_hashes(checkpoint_path),
                 "adapter": _file_hashes(adapter_path) if adapter_path else None,
                 "adapter_config": dict(acd_controller.config) if acd_controller is not None else {},
                 "adapter_route": "only rgb_missing _encode_visual_tokens(modality=thermal)" if adapter_path else None,
                 "feature_export": True, "dropblank_metrics": "metrics_dropblank.tsv"}
        if cli.split == "test":
            extra.update(_test_provenance(dev_data))
            extra["ap_tie_check"] = tie_checks
        try:
            if comparison_dir:
                comparison = compare_exports(comparison_dir, output_dir, allow_acd_changes=bool(adapter_path))
                extra["invariance_overall_pass"] = comparison["overall_pass"]
        except Exception:
            extra["invariance_overall_pass"] = False
            raise
        finally:
            _write_opt_in_record(output_dir, cli, args, started, started_utc, device, extra)
    return summary


def build_parser():
    parser = _development.build_parser()
    parser.add_argument("--split", choices=("dev", "test"), required=True,
                        help="Explicit evaluation split; checkpoint training-split checks remain enabled")
    return parser


if __name__ == "__main__":
    cli = build_parser().parse_args()
    if cli.batch_size is not None and cli.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if cli.num_workers is not None and cli.num_workers < 0:
        raise ValueError("--num_workers must be nonnegative")
    print(json.dumps(export_split(cli), ensure_ascii=False, indent=2))
