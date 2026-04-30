import argparse
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
        description="Build an offline prototype init bank from train-set features."
    )
    parser.add_argument(
        "--dataset_name", default="RGBT-PEDES", choices=list(DATASET_FACTORY.keys())
    )
    parser.add_argument("--root_dir", default="datasets/TextReID")
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--visual_aggregate_mode",
        default="unique_image",
        choices=["unique_image", "sample"],
    )
    parser.set_defaults(normalize_before_mean=True, normalize_after_mean=True)
    parser.add_argument(
        "--normalize_before_mean",
        dest="normalize_before_mean",
        action="store_true",
        help="L2-normalize each feature before per-pid averaging.",
    )
    parser.add_argument(
        "--no_normalize_before_mean",
        dest="normalize_before_mean",
        action="store_false",
        help="Disable feature normalization before per-pid averaging.",
    )
    parser.add_argument(
        "--normalize_after_mean",
        dest="normalize_after_mean",
        action="store_true",
        help="L2-normalize each prototype after per-pid averaging.",
    )
    parser.add_argument(
        "--no_normalize_after_mean",
        dest="normalize_after_mean",
        action="store_false",
        help="Disable prototype normalization after per-pid averaging.",
    )

    # Model options needed to instantiate the same global feature path.
    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--img_size", type=int, nargs=2, default=[384, 128])
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)
    parser.add_argument("--MLM", default=False, action="store_true")
    parser.add_argument("--loss_names", default="sdm")
    parser.add_argument("--cmt_depth", type=int, default=4)
    parser.add_argument("--id_loss_weight", type=float, default=1.0)
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0)
    parser.add_argument("--lambda_mlm", type=float, default=1.0)

    parser.set_defaults(use_rgbt_add_fusion=True)
    parser.add_argument(
        "--use_rgbt_add_fusion", dest="use_rgbt_add_fusion", action="store_true"
    )
    parser.add_argument(
        "--no_use_rgbt_add_fusion",
        dest="use_rgbt_add_fusion",
        action="store_false",
    )
    parser.add_argument("--use_global_uncertainty", action="store_true", default=False)
    parser.add_argument("--uncertainty_dist_type", default="diag_gaussian")
    parser.set_defaults(use_precision_fusion=None)
    parser.add_argument(
        "--use_precision_fusion", dest="use_precision_fusion", action="store_true"
    )
    parser.add_argument(
        "--no_use_precision_fusion",
        dest="use_precision_fusion",
        action="store_false",
    )
    parser.add_argument("--uncertainty_loss_type", default="w2")
    parser.add_argument("--logvar_min", type=float, default=-6.0)
    parser.add_argument("--logvar_max", type=float, default=4.0)
    parser.add_argument("--uncertainty_eps", type=float, default=1e-6)
    parser.add_argument("--lambda_unc", type=float, default=0.1)
    parser.add_argument("--lambda_tr", type=float, default=1.0)
    parser.add_argument("--lambda_rt", type=float, default=1.0)
    parser.add_argument("--lambda_tt", type=float, default=0.0)
    parser.add_argument("--lambda_var_reg", type=float, default=1e-4)
    parser.add_argument("--eval_use_fused_mu", action="store_true", default=True)
    parser.add_argument(
        "--no_eval_use_fused_mu", dest="eval_use_fused_mu", action="store_false"
    )
    parser.add_argument("--eval_use_mu_only", action="store_true", default=False)

    # Keep memory-bank shapes configurable for checkpoints trained with non-default
    # MUM settings, even though sampled/probabilistic features are not used here.
    parser.add_argument("--use_mum", default=False, action="store_true")
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--uncertainty_scale", type=float, default=1.0)
    parser.add_argument("--memory_bank_size", type=int, default=0)
    parser.set_defaults(use_batch_uncertainty=True, use_id_uncertainty=True)
    parser.add_argument(
        "--use_batch_uncertainty", dest="use_batch_uncertainty", action="store_true"
    )
    parser.add_argument(
        "--no_use_batch_uncertainty",
        dest="use_batch_uncertainty",
        action="store_false",
    )
    parser.add_argument(
        "--use_id_uncertainty", dest="use_id_uncertainty", action="store_true"
    )
    parser.add_argument(
        "--no_use_id_uncertainty", dest="use_id_uncertainty", action="store_false"
    )
    parser.add_argument("--use_prob_sampling", default=False, action="store_true")

    # Accepted for command compatibility. The script does not need prototype banks
    # to extract deterministic features, so these default to disabled.
    parser.add_argument("--use_proto_calib", default=False, action="store_true")
    parser.add_argument("--proto_version", default="v1", choices=["v1", "v2"])
    parser.add_argument("--lambda_proto_v", type=float, default=1.0)
    parser.add_argument("--lambda_proto_t", type=float, default=1.0)
    parser.add_argument("--lambda_proto_self_v", type=float, default=None)
    parser.add_argument("--lambda_proto_self_t", type=float, default=None)
    parser.add_argument("--lambda_proto_cross_t2v", type=float, default=0.0)
    parser.add_argument("--lambda_proto_cross_v2t", type=float, default=0.0)
    parser.add_argument("--lambda_proto_align", type=float, default=0.0)
    parser.add_argument("--proto_temp", type=float, default=0.07)
    parser.add_argument("--proto_momentum", type=float, default=0.9)
    parser.add_argument("--proto_warmup_epochs", type=int, default=0)
    parser.add_argument(
        "--proto_update_source",
        type=str,
        default="detached",
        choices=["detached", "raw"],
    )
    parser.add_argument("--debug_proto_stats", default=False, action="store_true")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(
            "[PROTO_INIT][WARNING] ignoring unsupported arguments: "
            + " ".join(unknown)
        )
    args.img_size = tuple(args.img_size)
    return args


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
    out = {
        "record_index": torch.tensor(
            [item["record_index"] for item in batch], dtype=torch.long
        )
    }
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


def validate_pid_space(image_records, caption_records, num_classes):
    expected = set(range(int(num_classes)))
    observed = set()
    for record in list(image_records) + list(caption_records):
        pid = int(record["pid"])
        if pid < 0 or pid >= int(num_classes):
            raise ValueError(
                f"pid={pid} is outside valid range [0, {int(num_classes) - 1}]."
            )
        observed.add(pid)
    missing = sorted(expected - observed)
    if missing:
        preview = ",".join(str(pid) for pid in missing[:20])
        raise ValueError(
            "Train-set pids are not contiguous with num_classes: "
            f"missing {len(missing)} pid(s), first_missing={preview}."
        )


def load_checkpoint_weights(model, checkpoint_file):
    if not op.isfile(checkpoint_file):
        raise RuntimeError(f"checkpoint_file='{checkpoint_file}' does not exist.")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint format in {checkpoint_file!r}: "
            f"{type(checkpoint)}"
        )
    load_state_dict(model, state_dict)
    return checkpoint


def extract_visual_det_feat(model, images, thermal_images):
    if not (
        hasattr(model, "encode_rgbt_global")
        and hasattr(model, "_encode_visual_global_gaussian")
        and hasattr(model, "_fuse_rgbt_global")
    ):
        raise AttributeError(
            "Model must provide encode_rgbt_global(), "
            "_encode_visual_global_gaussian(), and _fuse_rgbt_global() "
            "to extract fused_det_feat."
        )
    rgb_global_feat, thermal_global_feat, _, _, _, _ = model.encode_rgbt_global(
        images, thermal_images
    )
    rgb_mu, rgb_logvar, thermal_mu, thermal_logvar = (
        model._encode_visual_global_gaussian(
            rgb_global_feat,
            thermal_global_feat,
            thermal_available=thermal_images is not None,
        )
    )
    fused_mu, _ = model._fuse_rgbt_global(
        rgb_mu=rgb_mu,
        rgb_logvar=rgb_logvar,
        thermal_mu=thermal_mu,
        thermal_logvar=thermal_logvar,
        thermal_available=thermal_images is not None,
    )
    return fused_mu.float()


def extract_text_det_feat(model, caption_ids):
    if hasattr(model, "_encode_caption_ids_with_uncertainty"):
        _, _, text_mu, _ = model._encode_caption_ids_with_uncertainty(caption_ids)
        return text_mu.float()
    if hasattr(model, "encode_text_deterministic"):
        return model.encode_text_deterministic(caption_ids).float()
    return model.encode_text(caption_ids).float()


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
            image_features = extract_visual_det_feat(model, images, thermal_images)
            image_features = image_features.float().cpu()
            for row, record_index in enumerate(record_indices):
                features[record_index] = image_features[row]

    if any(feature is None for feature in features):
        raise RuntimeError("Failed to encode all train images.")
    return torch.stack(features, dim=0)


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
            text_features = extract_text_det_feat(model, caption_ids)
            text_features = text_features.float().cpu()
            for row, record_index in enumerate(record_indices):
                features[record_index] = text_features[row]

    if any(feature is None for feature in features):
        raise RuntimeError("Failed to encode all train captions.")
    return torch.stack(features, dim=0)


def aggregate_by_pid(features, pids, num_classes, normalize_before_mean, normalize_after_mean):
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N, C], got {tuple(features.shape)}.")
    pids = torch.as_tensor(pids, dtype=torch.long)
    if pids.ndim != 1 or pids.shape[0] != features.shape[0]:
        raise ValueError(
            "pids must have shape [N] matching features, got "
            f"{tuple(pids.shape)} and {tuple(features.shape)}."
        )
    if pids.numel() == 0:
        raise ValueError("Cannot aggregate an empty feature set.")
    if pids.min().item() < 0 or pids.max().item() >= int(num_classes):
        raise ValueError(
            "Found pid outside valid range while aggregating: "
            f"min={int(pids.min().item())}, max={int(pids.max().item())}, "
            f"num_classes={int(num_classes)}."
        )

    features = features.float()
    if normalize_before_mean:
        features = F.normalize(features, p=2, dim=1)

    feature_dim = features.shape[1]
    sums = features.new_zeros(int(num_classes), feature_dim)
    counts = torch.zeros(int(num_classes), dtype=torch.long)
    sums.index_add_(0, pids, features)
    counts.index_add_(0, pids, torch.ones_like(pids, dtype=torch.long))

    initialized = counts.gt(0)
    prototypes = sums.clone()
    prototypes[initialized] = (
        sums[initialized] / counts[initialized].to(sums.dtype).unsqueeze(1)
    )
    if normalize_after_mean and initialized.any():
        prototypes[initialized] = F.normalize(prototypes[initialized], p=2, dim=1)
    return prototypes.cpu(), initialized.cpu(), counts.cpu()


def build_visual_aggregation_inputs(image_features, image_records, caption_records, mode):
    if mode == "unique_image":
        visual_pids = [int(record["pid"]) for record in image_records]
        return image_features, visual_pids

    print(
        "[PROTO_INIT][WARNING] visual_aggregate_mode=sample repeats the same "
        "image once per train caption when an image has multiple captions."
    )
    image_id_to_index = {
        int(record["image_id"]): index for index, record in enumerate(image_records)
    }
    sample_indices = []
    visual_pids = []
    for record in caption_records:
        image_id = int(record["image_id"])
        if image_id not in image_id_to_index:
            raise ValueError(f"image_id={image_id} is missing from image records.")
        sample_indices.append(image_id_to_index[image_id])
        visual_pids.append(int(record["pid"]))
    sample_indices = torch.tensor(sample_indices, dtype=torch.long)
    return image_features.index_select(0, sample_indices), visual_pids


def count_stats(counts):
    counts_float = counts.float()
    return (
        int(counts.min().item()),
        int(counts.max().item()),
        float(counts_float.mean().item()),
    )


def initialized_norm_mean(prototypes, initialized):
    if not initialized.any():
        return 0.0
    return float(prototypes[initialized].norm(p=2, dim=1).mean().item())


def print_summary(args, num_classes, feature_dim, output, rng):
    visual_counts = output["visual_counts"]
    text_counts = output["text_counts"]
    visual_initialized = output["visual_initialized"]
    text_initialized = output["text_initialized"]
    visual_min, visual_max, visual_mean = count_stats(visual_counts)
    text_min, text_max, text_mean = count_stats(text_counts)

    print(f"[PROTO_INIT] dataset_name={args.dataset_name}")
    print(f"[PROTO_INIT] checkpoint_file={args.checkpoint_file}")
    print(f"[PROTO_INIT] output_file={args.output_file}")
    print(f"[PROTO_INIT] num_classes={num_classes}")
    print(f"[PROTO_INIT] feature_dim={feature_dim}")
    print(f"[PROTO_INIT] visual_aggregate_mode={args.visual_aggregate_mode}")
    print(
        "[PROTO_INIT] visual_initialized_count="
        f"{int(visual_initialized.sum().item())}"
    )
    print(
        "[PROTO_INIT] text_initialized_count="
        f"{int(text_initialized.sum().item())}"
    )
    print(
        "[PROTO_INIT] visual_zero_count="
        f"{int((~visual_initialized).sum().item())}"
    )
    print(
        "[PROTO_INIT] text_zero_count="
        f"{int((~text_initialized).sum().item())}"
    )
    print(
        "[PROTO_INIT] visual_count_min/max/mean="
        f"{visual_min}/{visual_max}/{visual_mean:.6f}"
    )
    print(
        "[PROTO_INIT] text_count_min/max/mean="
        f"{text_min}/{text_max}/{text_mean:.6f}"
    )
    print(
        "[PROTO_INIT] visual_proto_norm_mean="
        f"{initialized_norm_mean(output['visual_prototypes'], visual_initialized):.6f}"
    )
    print(
        "[PROTO_INIT] text_proto_norm_mean="
        f"{initialized_norm_mean(output['text_prototypes'], text_initialized):.6f}"
    )

    sample_count = min(5, int(num_classes))
    for pid in rng.sample(range(int(num_classes)), sample_count):
        print(
            "[PROTO_INIT][PID] "
            f"pid={pid} "
            f"visual_count={int(visual_counts[pid].item())} "
            f"text_count={int(text_counts[pid].item())} "
            f"visual_initialized={bool(visual_initialized[pid].item())} "
            f"text_initialized={bool(text_initialized[pid].item())}"
        )


def main():
    args = parse_args()
    rng = random.Random(int(args.seed))
    torch.manual_seed(int(args.seed))

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {args.batch_size}.")
    if args.num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {args.num_workers}.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    device = torch.device(args.device)

    if op.exists(args.output_file):
        print(
            "[PROTO_INIT][WARNING] output_file already exists and will be "
            f"overwritten: {args.output_file}"
        )

    dataset_cls = DATASET_FACTORY[args.dataset_name]
    dataset = dataset_cls(root=args.root_dir)
    image_records, caption_records = collect_train_records(dataset)
    num_classes = len(dataset.train_id_container)
    validate_pid_space(image_records, caption_records, num_classes)

    model = build_model(args, num_classes=num_classes)
    load_checkpoint_weights(model, args.checkpoint_file)
    model.to(device)
    model.eval()

    image_features = encode_image_features(model, image_records, args, device)
    text_features = encode_text_features(model, caption_records, args, device)
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError(
            "Visual/text feature dimension mismatch: "
            f"{image_features.shape[1]} vs {text_features.shape[1]}."
        )
    feature_dim = int(image_features.shape[1])

    visual_features_for_agg, visual_pids = build_visual_aggregation_inputs(
        image_features=image_features,
        image_records=image_records,
        caption_records=caption_records,
        mode=args.visual_aggregate_mode,
    )
    text_pids = [int(record["pid"]) for record in caption_records]

    visual_prototypes, visual_initialized, visual_counts = aggregate_by_pid(
        visual_features_for_agg,
        visual_pids,
        num_classes=num_classes,
        normalize_before_mean=bool(args.normalize_before_mean),
        normalize_after_mean=bool(args.normalize_after_mean),
    )
    text_prototypes, text_initialized, text_counts = aggregate_by_pid(
        text_features,
        text_pids,
        num_classes=num_classes,
        normalize_before_mean=bool(args.normalize_before_mean),
        normalize_after_mean=bool(args.normalize_after_mean),
    )

    output = {
        "meta": {
            "dataset_name": args.dataset_name,
            "root_dir": args.root_dir,
            "checkpoint_file": args.checkpoint_file,
            "num_classes": int(num_classes),
            "feature_dim": feature_dim,
            "visual_aggregate_mode": args.visual_aggregate_mode,
            "normalize_before_mean": bool(args.normalize_before_mean),
            "normalize_after_mean": bool(args.normalize_after_mean),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "feature_names": {
                "visual": "fused_det_feat",
                "text": "text_det_feat",
            },
        },
        "visual_prototypes": visual_prototypes.float(),
        "text_prototypes": text_prototypes.float(),
        "visual_initialized": visual_initialized.bool(),
        "text_initialized": text_initialized.bool(),
        "visual_counts": visual_counts.long(),
        "text_counts": text_counts.long(),
    }

    output_dir = op.dirname(op.abspath(args.output_file))
    if output_dir:
        mkdir_if_missing(output_dir)
    torch.save(output, args.output_file)
    print_summary(args, num_classes, feature_dim, output, rng)


if __name__ == "__main__":
    main()
