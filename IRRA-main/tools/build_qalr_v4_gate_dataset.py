import argparse
import json
import os
import os.path as op
import random
import sys
from copy import deepcopy


REPO_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = None
F = None
DataLoader = None
dataset_build = None
ImageDataset = None
ImageDatasetRGBT = None
TextDataset = None
build_model = None
apply_img_local_filter_stage = None
is_full_eval_state_checkpoint = None
load_full_eval_state_checkpoint = None
restore_eval_extra_state = None
Checkpointer = None
load_train_configs = None
Evaluator = None
QALR_V4_FEATURE_NAMES = None
build_qalr_v4_tensor_payload = None
qalr_v4_feature_meta = None
QueryAdaptiveLocalReranker = None


def load_project_modules():
    global torch
    global F
    global DataLoader
    global dataset_build
    global ImageDataset
    global ImageDatasetRGBT
    global TextDataset
    global build_model
    global apply_img_local_filter_stage
    global is_full_eval_state_checkpoint
    global load_full_eval_state_checkpoint
    global restore_eval_extra_state
    global Checkpointer
    global load_train_configs
    global Evaluator
    global QALR_V4_FEATURE_NAMES
    global build_qalr_v4_tensor_payload
    global qalr_v4_feature_meta
    global QueryAdaptiveLocalReranker

    import torch as torch_module
    import torch.nn.functional as functional_module
    from torch.utils.data import DataLoader as data_loader_class

    import datasets.build as dataset_build_module
    from datasets.bases import ImageDataset as image_dataset_class
    from datasets.bases import ImageDatasetRGBT as image_dataset_rgbt_class
    from datasets.bases import TextDataset as text_dataset_class
    from model import build_model as build_model_fn
    from processor.processor import (
        apply_img_local_filter_stage as apply_img_local_filter_stage_fn,
    )
    from processor.processor import (
        is_full_eval_state_checkpoint as is_full_eval_state_checkpoint_fn,
    )
    from processor.processor import (
        load_full_eval_state_checkpoint as load_full_eval_state_checkpoint_fn,
    )
    from processor.processor import restore_eval_extra_state as restore_eval_extra_state_fn
    from utils.checkpoint import Checkpointer as checkpointer_class
    from utils.iotools import load_train_configs as load_train_configs_fn
    from utils.metrics import Evaluator as evaluator_class
    from utils.qalr_v4_gate import (
        QALR_V4_FEATURE_NAMES as feature_names,
    )
    from utils.qalr_v4_gate import (
        build_qalr_v4_tensor_payload as build_tensor_payload_fn,
    )
    from utils.qalr_v4_gate import qalr_v4_feature_meta as feature_meta_fn
    from utils.query_adaptive_rerank import (
        QueryAdaptiveLocalReranker as qalr_reranker_class,
    )

    torch = torch_module
    F = functional_module
    DataLoader = data_loader_class
    dataset_build = dataset_build_module
    ImageDataset = image_dataset_class
    ImageDatasetRGBT = image_dataset_rgbt_class
    TextDataset = text_dataset_class
    build_model = build_model_fn
    apply_img_local_filter_stage = apply_img_local_filter_stage_fn
    is_full_eval_state_checkpoint = is_full_eval_state_checkpoint_fn
    load_full_eval_state_checkpoint = load_full_eval_state_checkpoint_fn
    restore_eval_extra_state = restore_eval_extra_state_fn
    Checkpointer = checkpointer_class
    load_train_configs = load_train_configs_fn
    Evaluator = evaluator_class
    QALR_V4_FEATURE_NAMES = feature_names
    build_qalr_v4_tensor_payload = build_tensor_payload_fn
    qalr_v4_feature_meta = feature_meta_fn
    QueryAdaptiveLocalReranker = qalr_reranker_class


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build QALR v4 trainable candidate gate feature dataset."
    )
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--ckpt_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--token_topm", type=int, default=8)
    parser.add_argument("--patch_topr", type=int, default=4)
    parser.add_argument("--holdout_pid_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_queries", type=int, default=0)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    return parser.parse_args()


def log_line(lines, message):
    print(message)
    lines.append(str(message))


def set_seed(seed):
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def instantiate_dataset(args):
    factory = getattr(dataset_build, "__factory")
    if args.dataset_name not in factory:
        raise KeyError(f"Unsupported dataset_name={args.dataset_name!r}.")
    return factory[args.dataset_name](root=args.root_dir)


def build_eval_style_split(dataset, split):
    split = str(split).lower()
    if split == "train":
        if not hasattr(dataset, "train_annos") or not hasattr(dataset, "_process_anno"):
            raise RuntimeError(
                "Dataset does not expose train_annos/_process_anno required for "
                "safe train split eval-style construction."
            )
        eval_dict, _ = dataset._process_anno(dataset.train_annos, training=False)
        return eval_dict
    if split == "val":
        return deepcopy(dataset.val)
    if split == "test":
        return deepcopy(dataset.test)
    raise ValueError(f"Unsupported split={split!r}.")


def split_pids(eval_dict, holdout_pid_ratio, seed):
    if holdout_pid_ratio < 0.0 or holdout_pid_ratio >= 1.0:
        raise ValueError(
            "holdout_pid_ratio must be in [0, 1), got "
            f"{holdout_pid_ratio}."
        )

    pids = sorted({int(pid) for pid in eval_dict["image_pids"]})
    rng = random.Random(int(seed))
    shuffled = list(pids)
    rng.shuffle(shuffled)

    if len(shuffled) <= 1 or holdout_pid_ratio <= 0.0:
        holdout_count = 0
    else:
        holdout_count = int(round(len(shuffled) * float(holdout_pid_ratio)))
        holdout_count = max(1, min(holdout_count, len(shuffled) - 1))

    holdout_pids = set(shuffled[:holdout_count])
    train_pids = set(shuffled[holdout_count:])
    return train_pids, holdout_pids


def filter_eval_dict(eval_dict, pid_set, source_query_keep=None):
    pid_set = {int(pid) for pid in pid_set}
    source_query_keep = (
        None if source_query_keep is None else {int(idx) for idx in source_query_keep}
    )

    img_paths_t = eval_dict.get("img_paths_t", None)
    out = {
        "image_pids": [],
        "img_paths": [],
        "caption_pids": [],
        "captions": [],
        "source_caption_indices": [],
    }
    if img_paths_t is not None:
        out["img_paths_t"] = []

    for idx, pid in enumerate(eval_dict["image_pids"]):
        if int(pid) not in pid_set:
            continue
        out["image_pids"].append(int(pid))
        out["img_paths"].append(eval_dict["img_paths"][idx])
        if img_paths_t is not None:
            out["img_paths_t"].append(img_paths_t[idx])

    for idx, pid in enumerate(eval_dict["caption_pids"]):
        if int(pid) not in pid_set:
            continue
        if source_query_keep is not None and idx not in source_query_keep:
            continue
        out["caption_pids"].append(int(pid))
        out["captions"].append(eval_dict["captions"][idx])
        out["source_caption_indices"].append(int(idx))

    return out


def build_loaders(eval_dict, args):
    transform = dataset_build.build_transforms(img_size=args.img_size, is_train=False)
    if "img_paths_t" in eval_dict:
        img_set = ImageDatasetRGBT(
            eval_dict["image_pids"],
            eval_dict["img_paths"],
            eval_dict["img_paths_t"],
            transform,
        )
    else:
        img_set = ImageDataset(eval_dict["image_pids"], eval_dict["img_paths"], transform)

    txt_set = TextDataset(
        eval_dict["caption_pids"],
        eval_dict["captions"],
        text_length=args.text_length,
    )
    batch_size = int(getattr(args, "test_batch_size", getattr(args, "batch_size", 64)))
    num_workers = int(getattr(args, "num_workers", 0))
    img_loader = DataLoader(
        img_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    txt_loader = DataLoader(
        txt_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return img_loader, txt_loader


def empty_payload(split_name, metadata, topk):
    return build_qalr_v4_tensor_payload(
        candidates=[],
        metadata={
            **metadata,
            "dataset_split_name": split_name,
            "effective_topk": int(topk),
        },
        topk=topk,
    )


def build_candidate_payload(model, eval_dict, args, split_name, split_pids):
    metadata = {
        "dataset_split_name": str(split_name),
        "dataset_name": str(getattr(args, "dataset_name", "")),
        "source_split": str(getattr(args, "qalr_v4_source_split", "train")),
        "num_pids": int(len(split_pids)),
        "num_images": int(len(eval_dict["image_pids"])),
        "num_text_queries": int(len(eval_dict["caption_pids"])),
        "source_caption_indices": list(eval_dict.get("source_caption_indices", [])),
        "requested_topk": int(args.qalr_v4_topk),
        "token_topm": int(args.qalr_v4_token_topm),
        "patch_topr": int(args.qalr_v4_patch_topr),
    }
    if len(eval_dict["image_pids"]) == 0 or len(eval_dict["caption_pids"]) == 0:
        return empty_payload(split_name, metadata, topk=args.qalr_v4_topk)

    img_loader, txt_loader = build_loaders(eval_dict, args)
    evaluator = Evaluator(img_loader, txt_loader, source=f"qalr_v4_gate_{split_name}")
    embedding_outputs = evaluator._compute_embedding(
        model,
        return_local_tokens=True,
        use_mtp_eval_fusion=False,
    )

    qfeats = F.normalize(embedding_outputs["qfeats"], p=2, dim=1)
    gfeats = F.normalize(embedding_outputs["gfeats"], p=2, dim=1)
    global_similarity = qfeats @ gfeats.t()

    reranker = QueryAdaptiveLocalReranker(
        topk=int(args.qalr_v4_topk),
        alpha=0.0,
        token_topm=int(args.qalr_v4_token_topm),
        patch_topr=int(args.qalr_v4_patch_topr),
        vocab_size=int(getattr(args, "vocab_size", 49408)),
    )
    diag_payload = reranker.candidate_oracle_diagnostics(
        query_token_embeddings=embedding_outputs["query_token_embeddings"],
        query_token_ids=embedding_outputs["query_token_ids"],
        query_global_embeddings=embedding_outputs["query_global_embeddings"],
        gallery_rgb_patch_tokens=embedding_outputs["gallery_rgb_patch_tokens"],
        gallery_thermal_patch_tokens=embedding_outputs["gallery_thermal_patch_tokens"],
        gallery_rgb_precision_weights=embedding_outputs.get(
            "gallery_rgb_precision_weights", None
        ),
        gallery_thermal_precision_weights=embedding_outputs.get(
            "gallery_thermal_precision_weights", None
        ),
        gallery_precision_weight_valid=embedding_outputs.get(
            "gallery_precision_weight_valid", None
        ),
        global_similarity=global_similarity,
        qids=embedding_outputs["qids"],
        gids=embedding_outputs["gids"],
        save_scores=True,
    )

    effective_topk = int(diag_payload.get("config", {}).get("topk", args.qalr_v4_topk))
    metadata.update(
        {
            "effective_topk": int(effective_topk),
            "candidate_diag_summary": dict(diag_payload.get("summary", {})),
        }
    )
    return build_qalr_v4_tensor_payload(
        candidates=list(diag_payload.get("candidates", [])),
        metadata=metadata,
        topk=effective_topk,
    )


def load_checkpoint_for_eval(model, args, ckpt_file):
    ckpt_file = op.abspath(ckpt_file)
    raw_checkpoint = torch.load(ckpt_file, map_location=torch.device("cpu"))
    if is_full_eval_state_checkpoint(raw_checkpoint):
        checkpoint = load_full_eval_state_checkpoint(
            model,
            ckpt_file,
            checkpoint=raw_checkpoint,
        )
        extra_eval_state = checkpoint.get("extra_eval_state", {})
        best_epoch = int(checkpoint.get("epoch", 1))
        apply_img_local_filter_stage(
            model,
            args,
            best_epoch,
            logger=None,
            prefix="QALR_V4_GATE_DATASET",
        )
        restore_eval_extra_state(model, extra_eval_state, logger=None)
        return checkpoint

    checkpointer = Checkpointer(model)
    checkpoint = checkpointer.load(f=ckpt_file)
    best_epoch = int(checkpoint.get("epoch", 1)) if isinstance(checkpoint, dict) else 1
    apply_img_local_filter_stage(
        model,
        args,
        best_epoch,
        logger=None,
        prefix="QALR_V4_GATE_DATASET",
    )
    return checkpoint


def payload_stats(payload):
    metadata = payload.get("metadata", {})
    return {
        "num_queries": int(metadata.get("num_queries", 0)),
        "num_candidates": int(metadata.get("num_candidates", 0)),
        "num_features": int(metadata.get("num_features", len(QALR_V4_FEATURE_NAMES))),
        "positive_count": int(metadata.get("positive_count", 0)),
        "positive_ratio": float(metadata.get("positive_ratio", 0.0)),
        "queries_with_positive_in_topk": int(
            metadata.get("queries_with_positive_in_topk", 0)
        ),
    }


def log_payload_stats(lines, name, payload):
    stats = payload_stats(payload)
    log_line(
        lines,
        (
            f"[{name}] num_queries={stats['num_queries']} "
            f"num_candidates={stats['num_candidates']} "
            f"num_features={stats['num_features']} "
            f"positive_count={stats['positive_count']} "
            f"positive_ratio={stats['positive_ratio']:.8f} "
            f"queries_with_positive_in_topk={stats['queries_with_positive_in_topk']}"
        ),
    )


def main():
    cli_args = parse_args()
    if cli_args.topk <= 0:
        raise ValueError(f"--topk must be > 0, got {cli_args.topk}.")
    if cli_args.token_topm <= 0:
        raise ValueError(f"--token_topm must be > 0, got {cli_args.token_topm}.")
    if cli_args.patch_topr <= 0:
        raise ValueError(f"--patch_topr must be > 0, got {cli_args.patch_topr}.")
    if cli_args.limit_queries < 0:
        raise ValueError(
            f"--limit_queries must be >= 0, got {cli_args.limit_queries}."
        )

    os.makedirs(cli_args.out_dir, exist_ok=True)
    log_lines = []
    load_project_modules()
    set_seed(cli_args.seed)

    args = load_train_configs(cli_args.config_file)
    args.training = False
    args.distributed = False
    args.config_file = op.abspath(cli_args.config_file)
    args.test_ckpt_file = op.abspath(cli_args.ckpt_file)
    args.qalr_v4_source_split = str(cli_args.split)
    args.qalr_v4_topk = int(cli_args.topk)
    args.qalr_v4_token_topm = int(cli_args.token_topm)
    args.qalr_v4_patch_topr = int(cli_args.patch_topr)

    log_line(log_lines, "[QALR_V4_GATE_DATASET] start")
    log_line(log_lines, f"config_file={args.config_file}")
    log_line(log_lines, f"ckpt_file={args.test_ckpt_file}")
    log_line(log_lines, f"out_dir={op.abspath(cli_args.out_dir)}")
    log_line(log_lines, f"split={cli_args.split}")
    log_line(log_lines, f"topk={cli_args.topk}")
    log_line(log_lines, f"token_topm={cli_args.token_topm}")
    log_line(log_lines, f"patch_topr={cli_args.patch_topr}")
    log_line(log_lines, f"holdout_pid_ratio={cli_args.holdout_pid_ratio}")
    log_line(log_lines, f"seed={cli_args.seed}")
    log_line(log_lines, f"limit_queries={cli_args.limit_queries}")

    dataset = instantiate_dataset(args)
    full_eval_dict = build_eval_style_split(dataset, cli_args.split)
    train_pids, holdout_pids = split_pids(
        full_eval_dict,
        holdout_pid_ratio=float(cli_args.holdout_pid_ratio),
        seed=int(cli_args.seed),
    )

    source_query_keep = None
    if cli_args.limit_queries > 0:
        source_query_keep = set(
            range(min(int(cli_args.limit_queries), len(full_eval_dict["captions"])))
        )

    train_eval_dict = filter_eval_dict(
        full_eval_dict,
        train_pids,
        source_query_keep=source_query_keep,
    )
    holdout_eval_dict = filter_eval_dict(
        full_eval_dict,
        holdout_pids,
        source_query_keep=source_query_keep,
    )

    num_classes = len(dataset.train_id_container)
    model = build_model(args, num_classes=num_classes)
    load_checkpoint_for_eval(model, args, cli_args.ckpt_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with torch.no_grad():
        train_payload = build_candidate_payload(
            model=model,
            eval_dict=train_eval_dict,
            args=args,
            split_name="train",
            split_pids=train_pids,
        )
        holdout_payload = build_candidate_payload(
            model=model,
            eval_dict=holdout_eval_dict,
            args=args,
            split_name="holdout",
            split_pids=holdout_pids,
        )

    train_path = op.join(cli_args.out_dir, "train_candidates.pt")
    holdout_path = op.join(cli_args.out_dir, "holdout_candidates.pt")
    meta_path = op.join(cli_args.out_dir, "feature_meta.json")
    log_path = op.join(cli_args.out_dir, "build_log.txt")

    torch.save(train_payload, train_path)
    torch.save(holdout_payload, holdout_path)

    train_stats = payload_stats(train_payload)
    holdout_stats = payload_stats(holdout_payload)
    total_queries = train_stats["num_queries"] + holdout_stats["num_queries"]
    total_candidates = train_stats["num_candidates"] + holdout_stats["num_candidates"]
    total_positive = train_stats["positive_count"] + holdout_stats["positive_count"]
    total_queries_with_positive = (
        train_stats["queries_with_positive_in_topk"]
        + holdout_stats["queries_with_positive_in_topk"]
    )
    total_positive_ratio = (
        float(total_positive) / float(total_candidates) if total_candidates else 0.0
    )

    feature_meta = qalr_v4_feature_meta(topk=cli_args.topk)
    feature_meta.update(
        {
            "builder": "tools/build_qalr_v4_gate_dataset.py",
            "config_file": args.config_file,
            "ckpt_file": args.test_ckpt_file,
            "dataset_name": str(getattr(args, "dataset_name", "")),
            "source_split": str(cli_args.split),
            "topk": int(cli_args.topk),
            "token_topm": int(cli_args.token_topm),
            "patch_topr": int(cli_args.patch_topr),
            "holdout_pid_ratio": float(cli_args.holdout_pid_ratio),
            "seed": int(cli_args.seed),
            "limit_queries": int(cli_args.limit_queries),
            "train_pid_count": int(len(train_pids)),
            "holdout_pid_count": int(len(holdout_pids)),
            "train_candidates_file": op.abspath(train_path),
            "holdout_candidates_file": op.abspath(holdout_path),
            "total": {
                "num_queries": int(total_queries),
                "num_candidates": int(total_candidates),
                "num_features": int(len(QALR_V4_FEATURE_NAMES)),
                "positive_count": int(total_positive),
                "positive_ratio": float(total_positive_ratio),
                "queries_with_positive_in_topk": int(total_queries_with_positive),
            },
            "train": train_stats,
            "holdout": holdout_stats,
        }
    )
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(feature_meta, handle, indent=2, sort_keys=True)

    log_line(log_lines, f"train_pid_count={len(train_pids)}")
    log_line(log_lines, f"holdout_pid_count={len(holdout_pids)}")
    log_payload_stats(log_lines, "TRAIN", train_payload)
    log_payload_stats(log_lines, "HOLDOUT", holdout_payload)
    log_line(
        log_lines,
        (
            f"[TOTAL] num_queries={total_queries} "
            f"num_candidates={total_candidates} "
            f"num_features={len(QALR_V4_FEATURE_NAMES)} "
            f"positive_count={total_positive} "
            f"positive_ratio={total_positive_ratio:.8f} "
            f"queries_with_positive_in_topk={total_queries_with_positive}"
        ),
    )
    log_line(log_lines, "feature_names=" + ", ".join(QALR_V4_FEATURE_NAMES))
    log_line(log_lines, f"wrote_train={op.abspath(train_path)}")
    log_line(log_lines, f"wrote_holdout={op.abspath(holdout_path)}")
    log_line(log_lines, f"wrote_feature_meta={op.abspath(meta_path)}")

    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(log_lines) + "\n")
    print(f"wrote_build_log={op.abspath(log_path)}")


if __name__ == "__main__":
    main()
