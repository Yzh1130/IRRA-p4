import argparse
import json
import os
import os.path as op
import random
import sys
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.bases import tokenize
from datasets.build import build_transforms
from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
from datasets.rgbtpedes import RGBTPEDES
from datasets.rstpreid import RSTPReid
from model import build_model
from utils.checkpoint import load_state_dict
from utils.iotools import mkdir_if_missing, read_image
from utils.simple_tokenizer import SimpleTokenizer


DATASET_FACTORY = {
    "CUHK-PEDES": CUHKPEDES,
    "ICFG-PEDES": ICFGPEDES,
    "RSTPReid": RSTPReid,
    "RGBT-PEDES": RGBTPEDES,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an offline hard-negative image index for train captions."
    )
    parser.add_argument("--dataset_name", default="RGBT-PEDES", choices=list(DATASET_FACTORY.keys()))
    parser.add_argument("--root_dir", default="datasets/TextReID")
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=1)

    # Model/eval options needed to reproduce the visual/text feature path.
    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--img_size", type=int, nargs=2, default=[384, 128])
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)
    parser.add_argument("--loss_names", default="sdm")
    parser.add_argument("--cmt_depth", type=int, default=4)
    parser.add_argument("--id_loss_weight", type=float, default=1.0)
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0)

    parser.set_defaults(use_rgbt_add_fusion=True)
    parser.add_argument("--use_rgbt_add_fusion", dest="use_rgbt_add_fusion", action="store_true")
    parser.add_argument("--no_use_rgbt_add_fusion", dest="use_rgbt_add_fusion", action="store_false")
    parser.add_argument("--use_global_uncertainty", action="store_true", default=False)
    parser.add_argument("--uncertainty_dist_type", default="diag_gaussian")
    parser.set_defaults(use_precision_fusion=None)
    parser.add_argument("--use_precision_fusion", dest="use_precision_fusion", action="store_true")
    parser.add_argument("--no_use_precision_fusion", dest="use_precision_fusion", action="store_false")
    parser.add_argument("--uncertainty_loss_type", default="w2")
    parser.add_argument("--eval_use_fused_mu", action="store_true", default=True)
    parser.add_argument("--no_eval_use_fused_mu", dest="eval_use_fused_mu", action="store_false")
    parser.add_argument("--eval_use_mu_only", action="store_true", default=False)

    return parser.parse_args()


class TrainCaptionDataset(Dataset):
    def __init__(self, caption_records, text_length):
        self.caption_records = caption_records
        self.text_length = int(text_length)
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_records)

    def __getitem__(self, index):
        record = self.caption_records[index]
        return {
            "record_index": index,
            "caption_ids": tokenize(
                record["caption"],
                tokenizer=self.tokenizer,
                text_length=self.text_length,
                truncate=True,
            ),
        }


class TrainImageDataset(Dataset):
    def __init__(self, image_records, transform):
        self.image_records = image_records
        self.transform = transform

    def __len__(self):
        return len(self.image_records)

    def __getitem__(self, index):
        record = self.image_records[index]
        rgb_img = read_image(record["rgb_path"])
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        if record["thermal_path"] is None:
            return {
                "record_index": index,
                "images": rgb_img,
            }

        thermal_img = read_image(record["thermal_path"])
        if self.transform is not None:
            thermal_img = self.transform(thermal_img)

        return {
            "record_index": index,
            "images": rgb_img,
            "thermal_images": thermal_img,
        }


def collate_indexed_tensor_batch(batch):
    out = {"record_index": torch.tensor([item["record_index"] for item in batch], dtype=torch.long)}
    for key in batch[0].keys():
        if key == "record_index":
            continue
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def collect_train_records(dataset):
    image_records_by_id = OrderedDict()
    caption_records = []
    image_key_to_id = {}

    for caption_index, sample in enumerate(dataset.train):
        if len(sample) == 5:
            pid, image_id, rgb_path, thermal_path, caption = sample
        elif len(sample) == 4:
            pid, image_id, rgb_path, caption = sample
            thermal_path = None
        else:
            raise ValueError(
                "Unsupported train sample format. Expected len=4 or len=5, "
                f"got len={len(sample)}."
            )

        image_key = op.basename(str(rgb_path))
        pid = int(pid)
        image_id = int(image_id)

        previous_image_id = image_key_to_id.get(image_key)
        if previous_image_id is not None and previous_image_id != image_id:
            raise ValueError(
                "Image basename is not unique in train split: "
                f"{image_key} maps to image_id={previous_image_id} and {image_id}."
            )
        image_key_to_id[image_key] = image_id

        if image_id not in image_records_by_id:
            image_records_by_id[image_id] = {
                "image_id": image_id,
                "image_key": image_key,
                "pid": pid,
                "rgb_path": str(rgb_path),
                "thermal_path": None if thermal_path is None else str(thermal_path),
            }
        else:
            existing = image_records_by_id[image_id]
            if existing["pid"] != pid or existing["image_key"] != image_key:
                raise ValueError(
                    "Inconsistent repeated image_id in train split: "
                    f"image_id={image_id}, existing={existing}, "
                    f"new_pid={pid}, new_key={image_key}."
                )

        caption_records.append(
            {
                "caption_index": int(caption_index),
                "image_id": image_id,
                "image_key": image_key,
                "pid": pid,
                "caption": str(caption),
            }
        )

    return list(image_records_by_id.values()), caption_records


def load_checkpoint_weights(model, checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint format in {checkpoint_file!r}: "
            f"{type(checkpoint)}"
        )
    load_state_dict(model, state_dict)
    return checkpoint


def resolve_feature_name(args):
    if bool(getattr(args, "eval_use_mu_only", False)):
        return "rgb_mu"
    if bool(getattr(args, "eval_use_fused_mu", True)):
        return "fused_mu"
    if bool(getattr(args, "use_rgbt_add_fusion", True)):
        return "rgb_plus_thermal_mu"
    return "rgb_mu"


def encode_text_features(model, caption_records, args, device):
    dataset = TrainCaptionDataset(caption_records, text_length=args.text_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_indexed_tensor_batch,
    )

    features = [None] * len(caption_records)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            record_indices = batch["record_index"].tolist()
            caption_ids = batch["caption_ids"].to(device)
            if hasattr(model, "encode_text_eval"):
                text_features = model.encode_text_eval(caption_ids)
            elif hasattr(model, "encode_text_deterministic"):
                text_features = model.encode_text_deterministic(caption_ids)
            else:
                text_features = model.encode_text(caption_ids)
            if isinstance(text_features, dict):
                text_features = text_features["global_feat"]
            text_features = text_features.float().cpu()
            for row, record_index in enumerate(record_indices):
                features[record_index] = text_features[row]

    if any(feature is None for feature in features):
        raise RuntimeError("Failed to encode all train captions.")
    return torch.stack(features, dim=0)


def encode_image_features(model, image_records, args, device):
    transform = build_transforms(img_size=tuple(args.img_size), is_train=False)
    dataset = TrainImageDataset(image_records, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_indexed_tensor_batch,
    )

    features = [None] * len(image_records)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            record_indices = batch["record_index"].tolist()
            images = batch["images"].to(device)
            thermal_images = batch.get("thermal_images", None)
            if thermal_images is not None:
                thermal_images = thermal_images.to(device)

            if hasattr(model, "encode_rgbt_eval"):
                image_features = model.encode_rgbt_eval(images, thermal_images)
            elif thermal_images is not None and hasattr(model, "encode_rgbt_deterministic"):
                image_features = model.encode_rgbt_deterministic(images, thermal_images)
            elif thermal_images is not None:
                image_features = model.encode_image(images) + model.encode_image(thermal_images)
            else:
                image_features = model.encode_image(images)
            if isinstance(image_features, dict):
                image_features = image_features["global_feat"]
            image_features = image_features.float().cpu()
            for row, record_index in enumerate(record_indices):
                features[record_index] = image_features[row]

    if any(feature is None for feature in features):
        raise RuntimeError("Failed to encode all train images.")
    return torch.stack(features, dim=0)


def mine_caption_to_image_negatives(
    caption_features,
    image_features,
    caption_records,
    image_records,
    topk,
    device,
):
    caption_features = F.normalize(caption_features.float(), p=2, dim=1)
    image_features = F.normalize(image_features.float(), p=2, dim=1)

    image_pids = torch.tensor([record["pid"] for record in image_records], dtype=torch.long)
    image_keys = [record["image_key"] for record in image_records]
    image_ids = [record["image_id"] for record in image_records]

    topk = int(topk)
    if topk <= 0:
        raise ValueError(f"topk must be > 0, got {topk}.")
    effective_topk = min(topk, len(image_records))

    # Aggregate per-caption retrieval results by anchor image key. For each
    # anchor image, keep the highest score observed across its captions.
    candidates_by_anchor = OrderedDict()
    for record in image_records:
        candidates_by_anchor[record["image_key"]] = {}

    image_features_device = image_features.to(device)
    image_pids_device = image_pids.to(device)
    chunk_size = 1024

    with torch.no_grad():
        for start in range(0, len(caption_records), chunk_size):
            end = min(start + chunk_size, len(caption_records))
            caption_chunk = caption_features[start:end].to(device)
            caption_pids = torch.tensor(
                [record["pid"] for record in caption_records[start:end]],
                dtype=torch.long,
                device=device,
            )

            scores = caption_chunk @ image_features_device.t()
            same_pid = caption_pids.unsqueeze(1).eq(image_pids_device.unsqueeze(0))
            scores = scores.masked_fill(same_pid, float("-inf"))
            top_scores, top_indices = torch.topk(
                scores,
                k=effective_topk,
                dim=1,
                largest=True,
                sorted=True,
            )

            top_scores = top_scores.cpu()
            top_indices = top_indices.cpu()
            for row, caption_record in enumerate(caption_records[start:end]):
                anchor_key = caption_record["image_key"]
                anchor_pid = int(caption_record["pid"])
                anchor_candidates = candidates_by_anchor[anchor_key]
                for score, image_index in zip(top_scores[row].tolist(), top_indices[row].tolist()):
                    if not torch.isfinite(torch.tensor(score)):
                        continue
                    neg_pid = int(image_pids[image_index].item())
                    if neg_pid == anchor_pid:
                        raise RuntimeError(
                            "same-pid negative survived mining mask: "
                            f"anchor_key={anchor_key}, anchor_pid={anchor_pid}, "
                            f"neg_key={image_keys[image_index]}, neg_pid={neg_pid}."
                        )
                    neg_key = image_keys[image_index]
                    previous = anchor_candidates.get(neg_key)
                    if previous is None or float(score) > previous["score"]:
                        anchor_candidates[neg_key] = {
                            "neg_image_key": neg_key,
                            "neg_image_id": int(image_ids[image_index]),
                            "neg_pid": neg_pid,
                            "score": float(score),
                        }

    items = OrderedDict()
    anchors = OrderedDict()
    image_record_by_key = {record["image_key"]: record for record in image_records}
    for anchor_key, candidate_map in candidates_by_anchor.items():
        anchor_record = image_record_by_key[anchor_key]
        sorted_candidates = sorted(
            candidate_map.values(),
            key=lambda item: item["score"],
            reverse=True,
        )[:topk]
        anchor_pid = int(anchor_record["pid"])
        anchor_image_id = int(anchor_record["image_id"])
        anchors[anchor_key] = {
            "anchor_pid": anchor_pid,
            "anchor_image_id": anchor_image_id,
        }
        items[anchor_key] = [
            {
                "anchor_pid": anchor_pid,
                "anchor_image_id": anchor_image_id,
                **candidate,
            }
            for candidate in sorted_candidates
        ]

    return anchors, items


def print_index_stats(items, topk):
    counts = [len(negatives) for negatives in items.values()]
    if counts:
        avg_count = sum(counts) / len(counts)
        min_count = min(counts)
        max_count = max(counts)
        zero_count = sum(1 for count in counts if count == 0)
    else:
        avg_count = 0.0
        min_count = 0
        max_count = 0
        zero_count = 0

    print(f"[OFFHN_INDEX] num_anchors {len(items)}")
    print(f"[OFFHN_INDEX] avg_negatives_per_anchor {avg_count:.4f}")
    print(f"[OFFHN_INDEX] min_negatives_per_anchor {min_count}")
    print(f"[OFFHN_INDEX] max_negatives_per_anchor {max_count}")
    print(f"[OFFHN_INDEX] zero_negative_anchors {zero_count}")
    print(f"[OFFHN_INDEX] topk {int(topk)}")


def print_random_examples(anchors, items, num_examples=5, seed=1):
    rng = random.Random(seed)
    anchor_keys = list(items.keys())
    if not anchor_keys:
        return
    sample_count = min(int(num_examples), len(anchor_keys))
    for anchor_key in rng.sample(anchor_keys, sample_count):
        negatives = items[anchor_key]
        anchor_pid = anchors[anchor_key]["anchor_pid"]
        if not negatives:
            print(
                "[OFFHN_INDEX][SAMPLE] "
                f"anchor_key={anchor_key} anchor_pid={anchor_pid} "
                "neg_key=NA neg_pid=NA score=NA"
            )
            continue
        neg = negatives[0]
        print(
            "[OFFHN_INDEX][SAMPLE] "
            f"anchor_key={anchor_key} "
            f"anchor_pid={anchor_pid} "
            f"neg_key={neg['neg_image_key']} "
            f"neg_pid={neg['neg_pid']} "
            f"score={neg['score']:.6f}"
        )


def validate_items(items):
    for anchor_key, negatives in items.items():
        for negative in negatives:
            if int(negative["neg_pid"]) == int(negative["anchor_pid"]):
                raise RuntimeError(
                    "Invalid same-pid negative in output index: "
                    f"anchor_key={anchor_key}, negative={negative}"
                )


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_cls = DATASET_FACTORY[args.dataset_name]
    dataset = dataset_cls(root=args.root_dir)
    image_records, caption_records = collect_train_records(dataset)
    num_classes = len(dataset.train_id_container)

    print(f"[OFFHN_INDEX] num_train_images {len(image_records)}")
    print(f"[OFFHN_INDEX] num_train_captions {len(caption_records)}")

    model = build_model(args, num_classes=num_classes)
    load_checkpoint_weights(model, args.checkpoint_file)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    text_features = encode_text_features(model, caption_records, args, device)
    image_features = encode_image_features(model, image_records, args, device)

    anchors, items = mine_caption_to_image_negatives(
        caption_features=text_features,
        image_features=image_features,
        caption_records=caption_records,
        image_records=image_records,
        topk=args.topk,
        device=device,
    )
    validate_items(items)

    index = {
        "meta": {
            "dataset_name": args.dataset_name,
            "topk": int(args.topk),
            "checkpoint_file": op.abspath(args.checkpoint_file),
            "feature": resolve_feature_name(args),
            "text_feature": "text_eval_global",
            "anchor_key": "image_basename",
            "anchor_mode": "caption_query_topk_aggregated_by_image_key",
            "num_train_images": len(image_records),
            "num_train_captions": len(caption_records),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "anchors": anchors,
        "items": items,
    }

    output_dir = op.dirname(op.abspath(args.output_file))
    if output_dir:
        mkdir_if_missing(output_dir)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)

    print_index_stats(items, args.topk)
    print_random_examples(anchors, items, num_examples=5, seed=args.seed)
    print(f"[OFFHN_INDEX] output_file {op.abspath(args.output_file)}")


if __name__ == "__main__":
    main()
