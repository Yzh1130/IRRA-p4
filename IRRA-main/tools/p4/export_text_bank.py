"""Export every caption in the verified 1076-identity training subset.

No images are read or encoded; the base text tower is frozen in eval mode.
An existing output, including its companion run record, is never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def training_caption_records(annos, train_ids):
    """Mirror RGBTPEDES training annotation/caption order, retaining raw IDs."""
    train_ids = set(int(pid) for pid in train_ids)
    pid_map = {pid: index for index, pid in enumerate(sorted(train_ids))}
    rows, image_id = [], 0
    for annotation_index, anno in enumerate(annos):
        raw_pid = int(anno["id"])
        if anno.get("split", "train") != "train" or raw_pid not in train_ids:
            continue
        for caption_index, caption in enumerate(anno["captions"]):
            rows.append({"pid": pid_map[raw_pid], "raw_pid": raw_pid, "image_id": image_id,
                         "annotation_index": annotation_index, "caption_index": caption_index,
                         "caption": str(caption)})
        image_id += 1
    if {row["raw_pid"] for row in rows} != train_ids:
        raise ValueError("Training identities do not all have captions")
    return rows


def encode_text_batches(model, loader, device):
    features, pids = [], []
    with torch.no_grad():
        for pid, tokens in loader:
            feature = model.encode_text_eval(tokens.to(device)).float()
            if feature.ndim != 2 or feature.shape[0] != len(pid) or not torch.isfinite(feature).all():
                raise ValueError("Text encoder returned invalid global features")
            features.append(feature.cpu())
            pids.extend(torch.as_tensor(pid).tolist())
    if not features:
        raise ValueError("Empty training text bank")
    raw = torch.cat(features)
    return raw.numpy(), F.normalize(raw, p=2, dim=1).numpy(), pids


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--dev_split_file", required=True)
    parser.add_argument("--output", required=True, help="New .npz filename; companion <stem>_run_record.json is also written")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv=None):
    cli = build_parser().parse_args(argv)
    output = Path(cli.output).resolve()
    record_path = output.with_name(output.stem + "_run_record.json")
    if output.suffix.lower() != ".npz" or cli.batch_size < 1 or cli.num_workers < 0:
        raise ValueError("Require .npz output, positive batch_size and nonnegative num_workers")
    if output.exists() or record_path.exists():
        raise FileExistsError("Text bank or its run record already exists")
    from datasets.bases import TextDataset
    from model import build_model
    from train import set_seed
    from tools.p4.export_dev_records import (_file_hashes, _resolve_annotation, _resolve_saved_split,
                                             load_strict_checkpoint, validate_dev_split)
    from tools.p4.grad_conflict import preserve_global_rng
    from tools.p4.train_batch_stats import _load_training_args

    start = time.perf_counter()
    with preserve_global_rng():
        set_seed(cli.seed)
        args = _load_training_args(cli.config_file)
        if args.dataset_name != "RGBT-PEDES":
            raise ValueError("Training text bank is restricted to RGBT-PEDES")
        annotation = _resolve_annotation(cli.root_dir)
        verified = validate_dev_split(annotation, cli.dev_split_file)
        saved_split, _ = _resolve_saved_split(args, Path(cli.config_file).resolve(), None)
        base_split = validate_dev_split(annotation, saved_split)
        if verified["split_identity_sha256"] != base_split["split_identity_sha256"]:
            raise ValueError("Requested training split differs from the checkpoint configuration")
        if verified["n_train"] != 1076:
            raise ValueError("Expected the verified 1076-identity reduced training subset")
        annos = json.loads(Path(annotation).read_text(encoding="utf-8"))
        split = json.loads(Path(cli.dev_split_file).read_text(encoding="utf-8"))
        rows = training_caption_records(annos, split["train_reduced_ids"])
        if len(rows) != verified["train_reduced_captions"]:
            raise ValueError("Training caption count does not match split validation")
        device = "cuda" if cli.device == "auto" and torch.cuda.is_available() else "cpu" if cli.device == "auto" else cli.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        args.root_dir, args.dev_split_file = cli.root_dir, cli.dev_split_file
        model = build_model(args, num_classes=verified["n_train"])
        load_strict_checkpoint(model, cli.checkpoint)
        model.requires_grad_(False)
        if device == "cpu":
            model.float()
        model.to(device).eval()
        dataset = TextDataset([row["pid"] for row in rows], [row["caption"] for row in rows], text_length=args.text_length)
        loader = torch.utils.data.DataLoader(dataset, batch_size=cli.batch_size, num_workers=cli.num_workers,
                                             shuffle=False, generator=torch.Generator().manual_seed(cli.seed))
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        raw, normalized, encoded_pids = encode_text_batches(model, loader, device)
        if encoded_pids != [row["pid"] for row in rows]:
            raise ValueError("Text loader order changed")
        if device == "cuda":
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
        record = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, capture_output=True,
                                               text=True, check=True).stdout.strip(),
                  "parameters": vars(cli), "seeds": {"seed": cli.seed}, "device": device,
                  "elapsed_seconds": time.perf_counter() - start, "peak_cuda_memory_bytes": peak,
                  "checkpoint": _file_hashes(cli.checkpoint), "config": _file_hashes(cli.config_file),
                  "annotation": verified["annotation"], "split": verified["split"],
                  "split_identity_sha256": verified["split_identity_sha256"],
                  "n_identities": verified["n_train"], "n_captions": len(rows),
                  "order": "training annotation order, then caption order; zero-based indices"}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            np.savez_compressed(handle, text_features=raw, text_features_normalized=normalized,
                                **{field: np.asarray([row[field] for row in rows]) for field in
                                   ("pid", "raw_pid", "image_id", "annotation_index", "caption_index", "caption")})
        record["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        with record_path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(f"{output}: captions={len(rows)}, identities=1076")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
