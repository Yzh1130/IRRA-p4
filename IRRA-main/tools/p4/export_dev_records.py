"""Export P4 development retrieval records through the existing Evaluator.

Only the supplied output directory is written. The model is loaded once, strictly;
a new Evaluator resets the independent degradation RNG for each P3 core scenario.
AP in per_query.csv is a fraction; metrics.tsv and summary.json use percentages.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys


CODE_ROOT = Path(__file__).resolve().parents[2]
CORE_SCENARIOS = (
    ("clean", "none", 1.0, 7),
    ("rgb_missing", "rgb_missing", 1.0, 7),
    ("ir_missing", "ir_missing", 1.0, 7),
    ("rgb_noise_005", "rgb_noise", 0.05, 7),
    ("rgb_noise_010", "rgb_noise", 0.10, 7),
    ("ir_noise_005", "ir_noise", 0.05, 7),
    ("ir_noise_010", "ir_noise", 0.10, 7),
    ("rgb_blur_k7", "rgb_blur", 1.0, 7),
    ("ir_blur_k7", "ir_blur", 1.0, 7),
    ("both_noise_005", "both_noise", 0.05, 7),
    ("both_blur_k7", "both_blur", 1.0, 7),
)
SCENARIO_NAMES = tuple(item[0] for item in CORE_SCENARIOS)
QUERY_FIELDS = (
    "query_id", "pid", "caption", "scenario", "best_pos_rank", "AP",
    "n_pos", "n_tokens", "n_color", "rho",
)
METRIC_FIELDS = ("scenario", "R1", "R5", "R10", "mAP", "mINP")


def load_word_list(path):
    """Load reproducible lowercase token lists, ignoring blank/comment lines."""
    return {
        line.strip().casefold()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def caption_token_counts(caption, color_words=None, stopwords=None):
    """Count English word occurrences after fixed stopword removal (not BPE)."""
    if color_words is None:
        color_words = load_word_list(Path(__file__).with_name("color_lexicon.txt"))
    if stopwords is None:
        stopwords = load_word_list(Path(__file__).with_name("stopwords.txt"))
    tokens = [
        token for token in re.findall(r"[a-z]+(?:'[a-z]+)?", str(caption).casefold())
        if token not in stopwords
    ]
    n_color = sum(token in color_words for token in tokens)
    return {
        "n_tokens": len(tokens), "n_color": n_color,
        "rho": n_color / len(tokens) if tokens else 0.0,
    }


def _file_hashes(path):
    content = Path(path).read_bytes()
    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "md5": hashlib.md5(content).hexdigest(),
    }


def validate_dev_split(annotation_path, split_path):
    """Verify identities, annotation hash and recorded counts before evaluation.

    Raw identity labels are retained for exports. Contiguous PID labels mirror
    datasets/rgbtpedes.py:_process_anno exactly and verify saved score ordering.
    """
    annotation_path, split_path = Path(annotation_path), Path(split_path)
    annos = json.loads(annotation_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    required = ("dev_ids", "train_reduced_ids", "n_dev", "n_train_reduced", "anno_md5")
    missing = [key for key in required if key not in split]
    if missing:
        raise ValueError(f"Development split missing fields: {missing}")
    dev_list = [int(pid) for pid in split["dev_ids"]]
    train_list = [int(pid) for pid in split["train_reduced_ids"]]
    dev_ids, train_ids = set(dev_list), set(train_list)
    if len(dev_ids) != len(dev_list) or len(train_ids) != len(train_list):
        raise ValueError("Development split contains duplicate identities")
    official_train = {int(a["id"]) for a in annos if a.get("split", "train") == "train"}
    official_test = {int(a["id"]) for a in annos if a.get("split", "train") == "test"}
    if dev_ids & train_ids or (dev_ids | train_ids) & official_test:
        raise ValueError("Training/development/test identities overlap")
    if dev_ids | train_ids != official_train:
        raise ValueError("Development split does not exactly partition official training identities")
    if len(dev_ids) != int(split["n_dev"]) or len(train_ids) != int(split["n_train_reduced"]):
        raise ValueError("Recorded development/train identity counts disagree")
    annotation_hash = _file_hashes(annotation_path)
    if annotation_hash["md5"] != str(split["anno_md5"]).lower():
        raise ValueError(
            f"Annotation MD5 mismatch: expected={split['anno_md5']} actual={annotation_hash['md5']}"
        )
    # Existing RGBTPEDES also includes official 'val' rows in dataset.val; reject
    # those here rather than silently exporting identities outside dev_ids.
    if any(a.get("split", "train") not in {"train", "test"} for a in annos):
        raise ValueError("Annotation contains additional validation rows outside the explicit dev split")
    selected = [(index, a) for index, a in enumerate(annos)
                if a.get("split", "train") == "train" and int(a["id"]) in dev_ids]
    train_annos = [a for a in annos
                   if a.get("split", "train") == "train" and int(a["id"]) in train_ids]
    actual_counts = {
        "dev_records": len(selected),
        "dev_captions": sum(len(a["captions"]) for _, a in selected),
        "train_reduced_records": len(train_annos),
        "train_reduced_captions": sum(len(a["captions"]) for a in train_annos),
    }
    for key, count in actual_counts.items():
        if key in split and int(split[key]) != count:
            raise ValueError(f"Recorded {key}={split[key]} disagrees with actual {count}")
    pid_map = {pid: index for index, pid in enumerate(sorted(dev_ids))}
    queries, gallery_raw_pids, gallery_pids = [], [], []
    for annotation_index, anno in selected:
        raw_pid = int(anno["id"])
        gallery_raw_pids.append(raw_pid)
        gallery_pids.append(pid_map[raw_pid])
        for caption_index, caption in enumerate(anno["captions"]):
            queries.append({
                "query_id": f"{annotation_index}:{caption_index}",
                "pid": raw_pid, "caption": caption,
                "eval_pid": pid_map[raw_pid],
            })
    semantic_content = json.dumps({
        "dev_ids": sorted(dev_ids), "train_reduced_ids": sorted(train_ids),
        "anno_md5": annotation_hash["md5"],
    }, sort_keys=True).encode("utf-8")
    return {
        "queries": queries, "gallery_pids": gallery_pids,
        "gallery_raw_pids": gallery_raw_pids, "n_train": len(train_ids),
        "n_dev": len(dev_ids), **actual_counts,
        "annotation": annotation_hash, "split": _file_hashes(split_path),
        "split_identity_sha256": hashlib.sha256(semantic_content).hexdigest(),
    }


def load_strict_checkpoint(model, checkpoint_path):
    """Load exactly matching keys/shapes; never fill missing weights from init."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a model state dictionary")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Checkpoint model state must be a nonempty tensor dictionary")
    if all(str(key).startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state, strict=True)
    return checkpoint


def per_query_records(score_payload, dev_data, scenario, color_words=None, stopwords=None):
    """Produce per-query AP using the same full argsort convention as rank()."""
    import torch

    sim = score_payload["sim"].detach().cpu()
    qids = score_payload["txt_pids"].detach().cpu().long().view(-1)
    gids = score_payload["img_pids"].detach().cpu().long().view(-1)
    expected_qids = torch.tensor([q["eval_pid"] for q in dev_data["queries"]], dtype=torch.long)
    expected_gids = torch.tensor(dev_data["gallery_pids"], dtype=torch.long)
    if sim.shape != (len(expected_qids), len(expected_gids)):
        raise ValueError("Score shape does not match development annotation ordering")
    if not torch.equal(qids, expected_qids) or not torch.equal(gids, expected_gids):
        raise ValueError("Score PID order does not match development annotation ordering")
    for field, count in (("txt_indices", len(qids)), ("img_indices", len(gids))):
        if field in score_payload and not torch.equal(
            score_payload[field].cpu().long(), torch.arange(count)
        ):
            raise ValueError(f"Noncanonical score order: {field}")
    if not torch.isfinite(sim).all():
        raise ValueError("Nonfinite development retrieval scores")
    indices = torch.argsort(sim, dim=1, descending=True)
    matches = gids[indices].eq(qids[:, None])
    n_pos = matches.sum(dim=1)
    if bool((n_pos == 0).any()):
        raise ValueError("A development query has no positive gallery image pair")
    precision = matches.cumsum(1).float() / torch.arange(1, sim.shape[1] + 1).float()
    ap = (precision * matches).sum(1) / n_pos
    best_rank = matches.to(torch.int64).argmax(1) + 1
    records = []
    for index, query in enumerate(dev_data["queries"]):
        records.append({
            "query_id": query["query_id"], "pid": query["pid"],
            "caption": query["caption"], "scenario": scenario,
            "best_pos_rank": int(best_rank[index]), "AP": float(ap[index]),
            "n_pos": int(n_pos[index]),
            **caption_token_counts(query["caption"], color_words, stopwords),
        })
    return records


def build_summary(metric_rows, provenance=None):
    by_name = {row["scenario"]: row for row in metric_rows}
    if len(by_name) != len(metric_rows) or set(by_name) != set(SCENARIO_NAMES):
        raise ValueError("Summary requires exactly clean and all ten P3 core degradation scenarios")
    return {
        "metric_units": "percent", "per_query_AP_units": "fraction",
        "degraded_R1_mean": sum(float(by_name[name]["R1"]) for name in SCENARIO_NAMES[1:]) / 10,
        "clean_R1": float(by_name["clean"]["R1"]),
        "clean_mAP": float(by_name["clean"]["mAP"]),
        "degraded_scenarios": list(SCENARIO_NAMES[1:]),
        "provenance": provenance or {},
    }


def _resolve_annotation(root_dir):
    dataset_root = Path(root_dir) / "RGBT-PEDES"
    for data_dir in (dataset_root / "RGBT-PEDES-Final", dataset_root / "RGBT-PEDES", dataset_root):
        if (data_dir / "RGB").is_dir() and (data_dir / "T").is_dir():
            for name in ("Caption.json", "caption.json"):
                if (data_dir / name).is_file():
                    return data_dir / name
            raise FileNotFoundError(f"No caption annotation in {data_dir}")
    raise FileNotFoundError(f"No registered RGB/T dataset under {dataset_root}")


def _resolve_saved_split(config, config_path, explicit_training_split):
    saved = str(getattr(config, "dev_split_file", "") or "")
    if not saved:
        raise ValueError("Saved training config has no dev_split_file; official 1266-ID models are unsafe for dev evaluation")
    if explicit_training_split:
        return Path(explicit_training_split).resolve(), saved
    candidates = (Path(saved), config_path.parent / saved, CODE_ROOT / saved)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), saved
    raise FileNotFoundError(
        "Cannot verify saved training split. Supply --training_split_file with its relocated actual JSON: " + saved
    )


def export(cli):
    # Deferred imports keep CSV/lexicon/bootstrap helpers independently CPU testable.
    if cli.probe is not None:
        raise NotImplementedError("Optional feat/zero probe is not implemented in this first P4 export tool")
    if set(cli.scenarios) != set(SCENARIO_NAMES) or len(cli.scenarios) != len(SCENARIO_NAMES):
        raise ValueError("--scenarios must contain exactly the eleven P3 core scenario names")
    if int(cli.eval_degrade_seed) != 2026:
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

    config_path = Path(cli.config_file or Path(cli.checkpoint).parent / "configs.yaml").resolve()
    args = load_train_configs(str(config_path))
    if getattr(args, "dataset_name", None) != "RGBT-PEDES":
        raise ValueError("P4 development export is restricted to RGBT-PEDES")
    if cli.root_dir is not None:
        args.root_dir = str(Path(cli.root_dir).resolve())
    annotation = _resolve_annotation(args.root_dir)
    dev_data = validate_dev_split(annotation, cli.dev_split_file)
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
    args.training = False
    args.distributed = False
    args.eval_dataset = "val"
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
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("IRRA", str(output_dir), if_train=False)
    color_words, stopwords = load_word_list(cli.color_lexicon), load_word_list(cli.stopwords)
    metric_rows, query_rows = [], []
    for scenario, mode, strength, kernel in CORE_SCENARIOS:
        args.eval_degrade_mode = mode
        args.eval_degrade_strength = strength
        args.eval_degrade_blur_kernel = kernel
        args.eval_degrade_seed = 2026
        args.eval_scores_file = str(output_dir / "scores" / f"{scenario}.pt")
        evaluator = Evaluator(img_loader, txt_loader, source="p4_dev_export")
        evaluator.eval(model)
        metrics = dict(evaluator.last_eval_result)
        if not all(math.isfinite(float(metrics[key])) for key in METRIC_FIELDS[1:]):
            raise ValueError(f"Nonfinite metrics for {scenario}")
        metric_rows.append({"scenario": scenario, **metrics})
        payload = torch.load(args.eval_scores_file, map_location="cpu", weights_only=False)
        records = per_query_records(payload, dev_data, scenario, color_words, stopwords)
        if abs(sum(row["AP"] for row in records) / len(records) * 100 - metrics["mAP"]) > 2e-4:
            raise ValueError(f"Per-query AP disagrees with existing Evaluator mAP for {scenario}")
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
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config_file", default=None, help="Defaults to configs.yaml beside the checkpoint")
    parser.add_argument("--dev_split_file", default=str(CODE_ROOT / "splits" / "dev_190.json"))
    parser.add_argument("--training_split_file", default=None, help="Relocated saved training split JSON, if its recorded path is unavailable")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--root_dir", default=None, help="Explicit relocated dataset root")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIO_NAMES, default=list(SCENARIO_NAMES))
    parser.add_argument("--eval_degrade_seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=None, help="Evaluation loader batch size")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--color_lexicon", default=str(Path(__file__).with_name("color_lexicon.txt")))
    parser.add_argument("--stopwords", default=str(Path(__file__).with_name("stopwords.txt")))
    parser.add_argument("--probe", choices=("feat", "zero"), default=None,
                        help="Reserved optional probe; explicitly unsupported in the first implementation")
    return parser


if __name__ == "__main__":
    cli = build_parser().parse_args()
    if cli.batch_size is not None and cli.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if cli.num_workers is not None and cli.num_workers < 0:
        raise ValueError("--num_workers must be nonnegative")
    print(json.dumps(export(cli), ensure_ascii=False, indent=2))
