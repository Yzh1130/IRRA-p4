import argparse
import json
import os
import os.path as op
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.qalr_v4_gate import (
    QALRV4GateMLP,
    apply_feature_normalizer,
    compute_feature_normalizer,
    save_gate_checkpoint,
)


THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98]
CANDIDATE_RANK_MAX_VALUES = [2, 3, 5, 10]


def parse_args():
    parser = argparse.ArgumentParser(description="Train QALR v4 MLP candidate gate.")
    parser.add_argument("--train_pt", required=True)
    parser.add_argument("--holdout_pt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument(
        "--loss",
        type=str,
        default="weighted_bce",
        choices=["weighted_bce", "focal"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--target_type", type=str, default="promote", choices=["match", "promote"])
    parser.add_argument(
        "--ignore_old_top1_candidate",
        default=True,
        action="store_true",
        help="ignore candidate_global_rank=1 rows when target_type=promote",
    )
    parser.add_argument("--old_correct_negative_weight", type=float, default=1.0)
    parser.add_argument("--neutral_negative_weight", type=float, default=0.25)
    parser.add_argument("--hurt_negative_weight", type=float, default=1.0)
    return parser.parse_args()


def set_seed(seed):
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device_arg):
    device_arg = str(device_arg).lower()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested, but CUDA is not available.")
    return device


def log_line(lines, message):
    print(message)
    lines.append(str(message))


def load_payload(path):
    payload = torch.load(path, map_location="cpu")
    required = [
        "features",
        "labels",
        "q_pids",
        "candidate_pids",
        "candidate_global_ranks",
        "groups",
        "feature_names",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")
    return payload


def validate_feature_names(train_payload, holdout_payload):
    train_names = list(train_payload["feature_names"])
    holdout_names = list(holdout_payload["feature_names"])
    if train_names != holdout_names:
        raise ValueError("train/holdout feature_names mismatch.")
    return train_names


def weighted_bce_loss(logits, labels, pos_weight, sample_weight=None):
    loss = F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        pos_weight=pos_weight.to(logits.device),
        reduction="none",
    )
    if sample_weight is not None:
        loss = loss * sample_weight.to(logits.device).float()
    return loss.mean()


def focal_loss(logits, labels, pos_weight, sample_weight=None, gamma=2.0):
    labels = labels.float()
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight.to(logits.device),
        reduction="none",
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(labels > 0.5, probs, 1.0 - probs)
    loss = (1.0 - pt).pow(float(gamma)) * bce
    if sample_weight is not None:
        loss = loss * sample_weight.to(logits.device).float()
    return loss.mean()


def compute_loss(logits, labels, pos_weight, loss_name, sample_weight=None):
    if loss_name == "weighted_bce":
        return weighted_bce_loss(logits, labels, pos_weight, sample_weight=sample_weight)
    if loss_name == "focal":
        return focal_loss(logits, labels, pos_weight, sample_weight=sample_weight)
    raise ValueError(f"Unsupported loss={loss_name!r}.")


@torch.no_grad()
def predict_logits(model, features, batch_size, device):
    model.eval()
    outputs = []
    for start in range(0, features.shape[0], int(batch_size)):
        end = min(start + int(batch_size), features.shape[0])
        outputs.append(model(features[start:end].to(device)).detach().cpu())
    if outputs:
        return torch.cat(outputs, dim=0)
    return torch.empty((0,), dtype=torch.float32)


@torch.no_grad()
def evaluate_loss(model, features, labels, sample_weights, pos_weight, loss_name, batch_size, device):
    if features.shape[0] == 0:
        return 0.0
    logits = predict_logits(model, features, batch_size=batch_size, device=device)
    loss = compute_loss(
        logits,
        labels.float(),
        pos_weight.cpu(),
        loss_name,
        sample_weight=sample_weights,
    )
    return float(loss.item())


def binary_auc(scores, labels):
    labels = labels.long()
    pos_mask = labels.eq(1)
    neg_mask = labels.eq(0)
    pos_count = int(pos_mask.sum().item())
    neg_count = int(neg_mask.sum().item())
    if pos_count == 0 or neg_count == 0:
        return None
    order = torch.argsort(scores.float(), stable=True)
    sorted_scores = scores[order].float()
    ranks = torch.empty(scores.numel(), dtype=torch.float64)
    start = 0
    rank_base = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and sorted_scores[end].item() == sorted_scores[start].item():
            end += 1
        avg_rank = rank_base[start:end].mean()
        ranks[order[start:end]] = avg_rank
        start = end
    pos_rank_sum = ranks[pos_mask].sum()
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (
        pos_count * neg_count
    )
    return float(auc.item())


def prepare_training_targets(payload, args):
    match_labels = payload["labels"].float()
    num_rows = int(match_labels.numel())
    if str(args.target_type) == "match":
        return {
            "labels": match_labels,
            "sample_weights": torch.ones(num_rows, dtype=torch.float32),
            "train_mask": torch.ones(num_rows, dtype=torch.bool),
            "stats": {
                "target_type": "match",
                "action_positive_count": int(match_labels.sum().item()),
                "action_negative_count": int(num_rows - int(match_labels.sum().item())),
                "ignored_old_top1_count": 0,
            },
        }

    q_pids = payload["q_pids"].long()
    candidate_pids = payload["candidate_pids"].long()
    candidate_global_ranks = payload["candidate_global_ranks"].long()
    groups = payload["groups"]
    starts = groups["starts"].long()
    ends = groups["ends"].long()

    labels = torch.zeros(num_rows, dtype=torch.float32)
    sample_weights = torch.ones(num_rows, dtype=torch.float32)
    train_mask = torch.ones(num_rows, dtype=torch.bool)
    ignored_old_top1_count = 0

    for start, end in zip(starts.tolist(), ends.tolist()):
        if end <= start:
            continue
        q_pid = int(q_pids[start].item())
        ranks = candidate_global_ranks[start:end]
        old_rel = ranks.eq(1).nonzero(as_tuple=False).view(-1)
        old_offset = int(old_rel[0].item()) if old_rel.numel() else 0
        old_index = int(start + old_offset)
        old_top1_correct = int(candidate_pids[old_index].item()) == q_pid

        for row_index in range(int(start), int(end)):
            is_old_top1 = int(candidate_global_ranks[row_index].item()) == 1
            is_match = int(candidate_pids[row_index].item()) == q_pid
            if is_old_top1 and bool(args.ignore_old_top1_candidate):
                train_mask[row_index] = False
                ignored_old_top1_count += 1
                continue
            if (not old_top1_correct) and is_match and (not is_old_top1):
                labels[row_index] = 1.0
                sample_weights[row_index] = 1.0
            else:
                labels[row_index] = 0.0
                if old_top1_correct and is_match:
                    sample_weights[row_index] = float(args.neutral_negative_weight)
                elif old_top1_correct and (not is_match):
                    sample_weights[row_index] = float(args.hurt_negative_weight)
                elif (not old_top1_correct) and (not is_match):
                    sample_weights[row_index] = float(args.hurt_negative_weight)
                else:
                    sample_weights[row_index] = float(args.old_correct_negative_weight)

    valid_labels = labels[train_mask]
    positive_count = int(valid_labels.sum().item())
    negative_count = int(valid_labels.numel() - positive_count)
    return {
        "labels": labels,
        "sample_weights": sample_weights,
        "train_mask": train_mask,
        "stats": {
            "target_type": "promote",
            "action_positive_count": int(positive_count),
            "action_negative_count": int(negative_count),
            "ignored_old_top1_count": int(ignored_old_top1_count),
        },
    }


def threshold_search(payload, probs):
    groups = payload["groups"]
    q_pids = payload["q_pids"].long()
    candidate_pids = payload["candidate_pids"].long()
    candidate_global_ranks = payload["candidate_global_ranks"].long()
    starts = groups["starts"].long()
    ends = groups["ends"].long()

    rows = []
    best = None
    for candidate_rank_max in CANDIDATE_RANK_MAX_VALUES:
        for threshold in THRESHOLDS:
            changed = 0
            improved = 0
            hurt = 0
            neutral = 0
            for start, end in zip(starts.tolist(), ends.tolist()):
                if end <= start:
                    continue
                q_pid = int(q_pids[start].item())
                ranks = candidate_global_ranks[start:end]
                old_rel = ranks.eq(1).nonzero(as_tuple=False).view(-1)
                old_offset = int(old_rel[0].item()) if old_rel.numel() else 0
                old_pid = int(candidate_pids[start + old_offset].item())
                old_correct = old_pid == q_pid

                local_probs = probs[start:end]
                eligible = (
                    ranks.gt(1)
                    & ranks.le(int(candidate_rank_max))
                    & local_probs.ge(float(threshold))
                )
                if not bool(eligible.any().item()):
                    continue

                masked_probs = local_probs.clone()
                masked_probs[~eligible] = -1.0
                selected_offset = int(torch.argmax(masked_probs).item())
                if selected_offset == old_offset:
                    continue

                changed += 1
                selected_pid = int(candidate_pids[start + selected_offset].item())
                new_correct = selected_pid == q_pid
                if (not old_correct) and new_correct:
                    improved += 1
                elif old_correct and (not new_correct):
                    hurt += 1
                else:
                    neutral += 1

            net = improved - hurt
            score = net - 2 * hurt
            row = {
                "candidate_rank_max": int(candidate_rank_max),
                "threshold": float(threshold),
                "changed": int(changed),
                "improved": int(improved),
                "hurt": int(hurt),
                "neutral": int(neutral),
                "net": int(net),
                "score": int(score),
            }
            rows.append(row)
            if best is None or (
                row["score"],
                row["net"],
                -row["hurt"],
                row["improved"],
                -row["changed"],
            ) > (
                best["score"],
                best["net"],
                -best["hurt"],
                best["improved"],
                -best["changed"],
            ):
                best = row
    return rows, best


def write_threshold_tsv(path, rows):
    header = [
        "epoch",
        "candidate_rank_max",
        "threshold",
        "changed",
        "improved",
        "hurt",
        "neutral",
        "net",
        "score",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write(
                "\t".join(
                    [
                        str(row["epoch"]),
                        str(row["candidate_rank_max"]),
                        f"{row['threshold']:.2f}",
                        str(row["changed"]),
                        str(row["improved"]),
                        str(row["hurt"]),
                        str(row["neutral"]),
                        str(row["net"]),
                        str(row["score"]),
                    ]
                )
                + "\n"
            )


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be > 0, got {args.epochs}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be > 0, got {args.batch_size}.")
    if args.hidden_dim < 2:
        raise ValueError(f"--hidden_dim must be >= 2, got {args.hidden_dim}.")
    if args.patience < 0:
        raise ValueError(f"--patience must be >= 0, got {args.patience}.")

    os.makedirs(args.out_dir, exist_ok=True)
    log_lines = []
    threshold_rows_all = []
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_payload = load_payload(args.train_pt)
    holdout_payload = load_payload(args.holdout_pt)
    feature_names = validate_feature_names(train_payload, holdout_payload)

    train_features = train_payload["features"].float()
    holdout_features = holdout_payload["features"].float()
    train_targets = prepare_training_targets(train_payload, args)
    holdout_targets = prepare_training_targets(holdout_payload, args)
    train_labels_all = train_targets["labels"].float()
    train_sample_weights_all = train_targets["sample_weights"].float()
    train_mask = train_targets["train_mask"].bool()
    holdout_labels_all = holdout_targets["labels"].float()
    holdout_sample_weights_all = holdout_targets["sample_weights"].float()
    holdout_mask = holdout_targets["train_mask"].bool()

    if train_features.shape[0] == 0:
        raise ValueError("train payload has no candidates.")
    if holdout_features.shape[0] == 0:
        raise ValueError("holdout payload has no candidates.")

    mean, std = compute_feature_normalizer(train_features)
    train_features_norm = apply_feature_normalizer(train_features, mean, std)
    holdout_features_norm = apply_feature_normalizer(holdout_features, mean, std)

    train_features_for_loss = train_features_norm[train_mask]
    train_labels = train_labels_all[train_mask]
    train_sample_weights = train_sample_weights_all[train_mask]
    holdout_features_for_loss = holdout_features_norm[holdout_mask]
    holdout_labels = holdout_labels_all[holdout_mask]
    holdout_sample_weights = holdout_sample_weights_all[holdout_mask]

    positive_count = int(train_labels.sum().item())
    negative_count = int(train_labels.numel() - positive_count)
    if positive_count <= 0:
        raise ValueError("train payload has no positive action candidates.")
    pos_weight = torch.tensor(
        float(negative_count) / float(positive_count),
        dtype=torch.float32,
        device=device,
    )

    input_dim = int(train_features.shape[1])
    model = QALRV4GateMLP(
        input_dim=input_dim,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    train_set = TensorDataset(train_features_for_loss, train_labels, train_sample_weights)
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
    )

    config = {
        "input_dim": input_dim,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "loss": str(args.loss),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "target_type": str(args.target_type),
        "ignore_old_top1_candidate": bool(args.ignore_old_top1_candidate),
        "old_correct_negative_weight": float(args.old_correct_negative_weight),
        "neutral_negative_weight": float(args.neutral_negative_weight),
        "hurt_negative_weight": float(args.hurt_negative_weight),
        "train_pt": op.abspath(args.train_pt),
        "holdout_pt": op.abspath(args.holdout_pt),
        "pos_weight": float(pos_weight.item()),
        "training_args": dict(vars(args)),
    }

    log_line(log_lines, "[QALR_V4_GATE_TRAIN] start")
    log_line(log_lines, f"device={device}")
    log_line(log_lines, f"train_candidates={train_features.shape[0]}")
    log_line(log_lines, f"holdout_candidates={holdout_features.shape[0]}")
    log_line(log_lines, f"num_features={input_dim}")
    log_line(log_lines, f"target_type={args.target_type}")
    log_line(log_lines, f"action_positive_count={positive_count}")
    log_line(log_lines, f"action_negative_count={negative_count}")
    log_line(
        log_lines,
        f"ignored_old_top1_count={train_targets['stats'].get('ignored_old_top1_count', 0)}",
    )
    log_line(
        log_lines,
        f"holdout_ignored_old_top1_count={holdout_targets['stats'].get('ignored_old_top1_count', 0)}",
    )
    log_line(
        log_lines,
        (
            f"negative_weights old_correct={args.old_correct_negative_weight:.6f} "
            f"neutral={args.neutral_negative_weight:.6f} "
            f"hurt={args.hurt_negative_weight:.6f}"
        ),
    )
    log_line(log_lines, f"pos_weight={float(pos_weight.item()):.6f}")
    log_line(log_lines, "feature_names=" + ", ".join(feature_names))

    best_key = None
    best_metrics = None
    stale_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_features, batch_labels, batch_weights in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            batch_weights = batch_weights.to(device)
            logits = model(batch_features)
            loss = compute_loss(
                logits,
                batch_labels,
                pos_weight,
                args.loss,
                sample_weight=batch_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_features.shape[0])
            loss_sum += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = loss_sum / float(max(seen, 1))
        holdout_loss = evaluate_loss(
            model,
            holdout_features_for_loss,
            holdout_labels,
            holdout_sample_weights,
            pos_weight,
            args.loss,
            args.batch_size,
            device,
        )
        holdout_logits = predict_logits(
            model,
            holdout_features_norm,
            batch_size=args.batch_size,
            device=device,
        )
        holdout_probs = torch.sigmoid(holdout_logits)
        holdout_auc = binary_auc(holdout_probs[holdout_mask], holdout_labels.long())
        rows, best_row = threshold_search(holdout_payload, holdout_probs)
        for row in rows:
            threshold_rows_all.append({"epoch": int(epoch), **row})

        current_key = (
            int(best_row["score"]),
            int(best_row["net"]),
            -int(best_row["hurt"]),
            int(best_row["improved"]),
            -int(best_row["changed"]),
        )
        improved = best_key is None or current_key > best_key
        if improved:
            best_key = current_key
            stale_epochs = 0
            best_metrics = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "holdout_loss": float(holdout_loss),
                "holdout_auc": holdout_auc,
                "best_threshold": dict(best_row),
                "best_candidate_rank_max": int(best_row["candidate_rank_max"]),
            }
            config["best_threshold"] = float(best_row["threshold"])
            config["best_candidate_rank_max"] = int(best_row["candidate_rank_max"])
            save_gate_checkpoint(
                op.join(args.out_dir, "best_gate.pt"),
                model=model,
                mean=mean,
                std=std,
                feature_names=feature_names,
                config=config,
                metrics=best_metrics,
            )
        else:
            stale_epochs += 1

        auc_text = "NA" if holdout_auc is None else f"{holdout_auc:.6f}"
        log_line(
            log_lines,
            (
                f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"holdout_loss={holdout_loss:.6f} holdout_auc={auc_text} "
                f"best_rank_max={best_row['candidate_rank_max']} "
                f"best_threshold={best_row['threshold']:.2f} "
                f"changed={best_row['changed']} improved={best_row['improved']} "
                f"hurt={best_row['hurt']} neutral={best_row['neutral']} "
                f"net={best_row['net']} score={best_row['score']}"
            ),
        )

        if args.patience > 0 and stale_epochs >= int(args.patience):
            log_line(
                log_lines,
                f"[EARLY_STOP] epoch={epoch} stale_epochs={stale_epochs}",
            )
            break

    last_metrics = {
        "epoch": int(epoch),
        "best_metrics": dict(best_metrics or {}),
    }
    save_gate_checkpoint(
        op.join(args.out_dir, "last_gate.pt"),
        model=model,
        mean=mean,
        std=std,
        feature_names=feature_names,
        config=config,
        metrics=last_metrics,
    )

    write_threshold_tsv(op.join(args.out_dir, "threshold_search.tsv"), threshold_rows_all)
    feature_meta = {
        "feature_names": list(feature_names),
        "normalizer": {
            "mean": mean.detach().cpu().float().tolist(),
            "std": std.detach().cpu().float().tolist(),
        },
        "config": config,
        "best_metrics": dict(best_metrics or {}),
        "target_stats": {
            "train": dict(train_targets["stats"]),
            "holdout": dict(holdout_targets["stats"]),
        },
        "train_metadata": dict(train_payload.get("metadata", {})),
        "holdout_metadata": dict(holdout_payload.get("metadata", {})),
    }
    with open(op.join(args.out_dir, "feature_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(feature_meta, handle, indent=2, sort_keys=True)

    if best_metrics:
        log_line(log_lines, "[BEST] " + json.dumps(best_metrics, sort_keys=True))
    with open(op.join(args.out_dir, "train_log.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
