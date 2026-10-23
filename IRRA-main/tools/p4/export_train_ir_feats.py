"""Export the 1076-identity reduced-training gallery for R13 1b inference.

Images are visited once in annotation order with the existing deterministic
evaluation transforms. The existing gallery encoder and modality recorder are
observed, without a second modality forward or feature normalization. This tool
does not train, optimize, or evaluate retrieval on the official test split.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tools.p4.export_dev_records import (
    ModalityFeatureRecorder, _resolve_annotation, _resolve_saved_split,
    load_strict_checkpoint, validate_dev_split,
)


def _load_config(path):
    # Keep metadata, verification, and archive helpers usable without a model.
    from utils.iotools import load_train_configs

    return load_train_configs(str(path))


def prepare_metadata(cli, config):
    """Validate the saved/relocated split, then form one entry per training image.

    Raw identity labels and relative filenames retain train_annos ordering;
    the contiguous gallery labels are produced by RGBTPEDES itself.
    """
    from datasets.rgbtpedes import RGBTPEDES

    if getattr(config, "dataset_name", None) != "RGBT-PEDES":
        raise ValueError("R13 training-feature export requires RGBT-PEDES")
    config_path = Path(cli.config_file or Path(cli.checkpoint).parent / "configs.yaml").resolve()
    root_dir = Path(cli.root_dir or config.root_dir).resolve()
    split_path = Path(cli.dev_split_file).resolve()
    annotation = _resolve_annotation(root_dir)
    dev_data = validate_dev_split(annotation, split_path)
    if dev_data["n_train"] != 1076:
        raise ValueError(f"Expected exactly 1076 train_reduced_ids, got {dev_data['n_train']}")
    # Check the saved file when accessible. Only a relocated, unavailable path
    # falls back to the supplied local split, as allowed by the existing export.
    # An accessible different partition must not be silently overridden.
    try:
        training_split, saved_training_split = _resolve_saved_split(config, config_path, None)
    except FileNotFoundError:
        training_split, saved_training_split = _resolve_saved_split(config, config_path, split_path)
    training_data = validate_dev_split(annotation, training_split)
    if training_data["split_identity_sha256"] != dev_data["split_identity_sha256"]:
        raise ValueError("Checkpoint training split does not match supplied development split")
    dataset = RGBTPEDES(root=str(root_dir), dev_split_file=str(split_path), verbose=False)
    train_gallery, identities = dataset._process_anno(dataset.train_annos, training=False)
    raw_pids = [int(anno["id"]) for anno in dataset.train_annos]
    file_paths = [str(anno["file_path"]) for anno in dataset.train_annos]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected_train = {int(pid) for pid in split["train_reduced_ids"]}
    if set(raw_pids) != expected_train or len(identities) != 1076:
        raise ValueError("Training gallery does not match the 1076 reduced-training identities")
    if list(dataset.val["image_pids"]) != dev_data["gallery_pids"]:
        raise ValueError("Development gallery order disagrees with validated annotations")
    if [int(anno["id"]) for anno in dataset.val_annos] != dev_data["gallery_raw_pids"]:
        raise ValueError("Development raw identity order disagrees with validated annotations")
    if not (len(raw_pids) == len(file_paths) == len(train_gallery["image_pids"])):
        raise ValueError("Expected one training gallery item per annotation/image pair")
    limit = getattr(cli, "limit", None)
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("--limit must be positive")
        count = min(int(limit), len(raw_pids))
        # Caption fields are irrelevant to the image gallery and intentionally
        # not truncated or used to expand the gallery into image-caption pairs.
        train_gallery = dict(train_gallery)
        for key in ("image_pids", "img_paths", "img_paths_t"):
            train_gallery[key] = train_gallery[key][:count]
        raw_pids, file_paths = raw_pids[:count], file_paths[:count]
    return {
        "train_gallery": train_gallery, "train_raw_pids": raw_pids,
        "train_file_paths": file_paths, "dataset": dataset, "dev_data": dev_data,
        "config_path": config_path, "annotation": annotation, "root_dir": root_dir,
        "training_split": training_split, "saved_training_split": saved_training_split,
    }


def build_gallery_loader(gallery, args, cli):
    """Match the evaluation ImageDatasetRGBT/resize/normalize loader path."""
    from torch.utils.data import DataLoader
    from datasets.bases import ImageDatasetRGBT
    from datasets.build import build_transforms

    batch_size = getattr(cli, "batch_size", None) or getattr(args, "test_batch_size", 64)
    workers = getattr(cli, "num_workers", None)
    workers = getattr(args, "num_workers", 0) if workers is None else workers
    transforms = build_transforms(img_size=args.img_size, is_train=False)
    images = ImageDatasetRGBT(
        gallery["image_pids"], gallery["img_paths"], gallery["img_paths_t"], transforms,
    )
    return DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=workers)


def build_inference_model(args, checkpoint_path, device):
    """Strictly load the base checkpoint using the existing export operations."""
    from model import build_model
    from processor.processor import apply_img_local_filter_stage, restore_eval_extra_state

    model = build_model(args, num_classes=1076)
    checkpoint = load_strict_checkpoint(model, checkpoint_path)
    apply_img_local_filter_stage(
        model, args, int(checkpoint.get("epoch", 1)), logger=None, prefix="R13_TRAIN_FEATURES",
    )
    if checkpoint.get("full_eval_state_checkpoint", False):
        restore_eval_extra_state(model, checkpoint.get("extra_eval_state", {}), logger=None)
    model.requires_grad_(False)
    return model.to(device).eval()


def encode_gallery_features(model, loader, expected_pids):
    """Observe the exact pre-addition RGB and IR tensors during clean encoding."""
    import numpy as np
    import torch
    from utils.metrics import Evaluator

    evaluator = Evaluator(loader, [], source="r13_train_feature_export")
    device = next(model.parameters()).device
    observed_pids = []
    with ModalityFeatureRecorder(evaluator, model) as recorder, torch.no_grad():
        for pid, rgb, ir in loader:
            evaluator._encode_gallery_feature(model, rgb.to(device), ir.to(device))
            observed_pids.extend(pid.reshape(-1).cpu().tolist())
    if observed_pids != [int(pid) for pid in expected_pids]:
        raise ValueError("Encoded gallery PID ordering differs from validated annotation ordering")
    if not recorder.rgb or not recorder.ir:
        raise ValueError("Gallery encoding returned no images")
    f_rgb = torch.cat(recorder.rgb).float().numpy()
    f_ir = torch.cat(recorder.ir).float().numpy()
    if f_rgb.shape != f_ir.shape or f_rgb.ndim != 2 or len(f_ir) != len(expected_pids):
        raise ValueError("Modality feature shape differs from the validated gallery")
    if not (np.isfinite(f_rgb).all() and np.isfinite(f_ir).all()):
        raise ValueError("Nonfinite modality features")
    return f_rgb, f_ir


def _numpy(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def verify_dev_ir(features, pids, npz_path, tolerance=1e-5):
    """Compare clean pre-addition IR features and contiguous dev PID ordering."""
    import numpy as np

    actual, actual_pids = _numpy(features), _numpy(pids)
    with np.load(npz_path, allow_pickle=False) as reference:
        for key in ("clean/f_ir", "clean/img_pids"):
            if key not in reference:
                raise ValueError(f"Development reference npz missing {key}")
        expected = reference["clean/f_ir"]
        expected_pids = reference["clean/img_pids"]
        if actual.shape != expected.shape or actual.ndim != 2:
            raise ValueError(f"Development feature shape mismatch: {actual.shape} != {expected.shape}")
        if (actual_pids.shape != expected_pids.shape
                or not np.array_equal(actual_pids, expected_pids)):
            raise ValueError("Development gallery PID ordering differs from clean/img_pids")
        if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
            raise ValueError("Nonfinite development IR features")
        # Float64 comparison avoids rounding a just-over-threshold difference
        # back down when the reference and exported arrays have mixed dtypes.
        difference = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    if difference > tolerance:
        raise ValueError(f"Development IR feature max absolute difference {difference} exceeds {tolerance}")
    return difference


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def make_provenance(cli, metadata, verification=None):
    """Record checkpoint, split identity set, repository, and row semantics."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=CODE_ROOT, text=True,
    )
    return {
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(cli.checkpoint),
        "config": str(metadata["config_path"]),
        "config_sha256": _sha256(metadata["config_path"]),
        "split_identity_sha256": metadata["dev_data"]["split_identity_sha256"],
        "development_split": metadata["dev_data"]["split"],
        "training_split_recorded": metadata["saved_training_split"],
        "training_split_verified": str(metadata["training_split"]),
        "git_head": commit, "git_dirty": bool(status.strip()),
        "git_status_porcelain": status.splitlines(),
        "n_train_reduced_identities": 1076,
        "n_images": len(metadata["train_raw_pids"]),
        "arguments": vars(cli), "verify_dev": verification,
        "row_order": "RGBTPEDES.train_annos, one image pair per row, no caption expansion",
        "raw_pids": "original train_annos id (not contiguous training labels)",
        "file_paths": "original annotation file_path, aligned to raw_pids",
        "features": "exact clean evaluation pre-addition modality tensors, converted to float32",
    }


def write_npz(output, f_rgb, f_ir, raw_pids, file_paths, provenance):
    """Write only a new archive, without pickle-dependent object arrays."""
    import numpy as np

    path = Path(output)
    if path.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    rgb, ir = _numpy(f_rgb).astype(np.float32), _numpy(f_ir).astype(np.float32)
    pids, paths = _numpy(raw_pids), np.asarray([str(p) for p in file_paths], dtype=str)
    if rgb.ndim != 2 or ir.shape != rgb.shape or rgb.shape[0] == 0:
        raise ValueError("Expected nonempty aligned RGB/IR feature matrices")
    if pids.shape != (len(ir),) or paths.shape != (len(ir),):
        raise ValueError("Output metadata must contain one PID and path per image")
    if pids.dtype.kind not in "iu":
        raise ValueError("raw_pids must contain integer identity labels")
    if not (np.isfinite(rgb).all() and np.isfinite(ir).all()):
        raise ValueError("Nonfinite output modality features")
    payload = json.dumps(provenance, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle, f_ir=ir, f_rgb=rgb, raw_pids=pids.astype(np.int64),
            file_paths=paths, provenance=np.asarray(payload),
        )


def export(cli):
    """Run clean inference; verify dev first and write no archive on mismatch."""
    started = time.perf_counter()
    output = Path(cli.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    if cli.batch_size is not None and cli.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if cli.num_workers is not None and cli.num_workers < 0:
        raise ValueError("--num_workers must be nonnegative")
    if getattr(cli, "limit", None) is not None and cli.limit <= 0:
        raise ValueError("--limit must be positive")
    if not getattr(cli, "verify_dev_npz", None):
        raise ValueError("--verify_dev_npz is required for verified R13 1b exports")
    if not Path(cli.verify_dev_npz).is_file():
        raise FileNotFoundError(cli.verify_dev_npz)
    checkpoint = Path(cli.checkpoint).resolve()
    config_path = Path(cli.config_file or checkpoint.parent / "configs.yaml").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args = _load_config(config_path)
    metadata = prepare_metadata(cli, args)
    import torch

    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.training = False
    args.distributed = False
    args.root_dir = str(metadata["root_dir"])
    args.dev_split_file = str(Path(cli.dev_split_file).resolve())
    args.eval_dataset = "val"
    args.config_file = str(config_path)
    args.test_ckpt_file = str(checkpoint)
    args.eval_degrade_mode = "none"
    args.save_eval_scores = False
    args.eval_scores_save_raw = False
    model = build_inference_model(args, checkpoint, device)
    verification = None
    if getattr(cli, "verify_dev_npz", None):
        dev_pids = metadata["dev_data"]["gallery_pids"]
        loader = build_gallery_loader(metadata["dataset"].val, args, cli)
        _, dev_ir = encode_gallery_features(model, loader, dev_pids)
        difference = verify_dev_ir(dev_ir, dev_pids, cli.verify_dev_npz)
        verification = {
            "path": str(Path(cli.verify_dev_npz).resolve()),
            "sha256": _sha256(cli.verify_dev_npz),
            "feature_key": "clean/f_ir", "pid_key": "clean/img_pids",
            "max_absolute_difference": difference, "tolerance": 1e-5,
        }
    loader = build_gallery_loader(metadata["train_gallery"], args, cli)
    rgb, ir = encode_gallery_features(model, loader, metadata["train_gallery"]["image_pids"])
    provenance = make_provenance(cli, metadata, verification)
    provenance["elapsed_seconds_before_write"] = time.perf_counter() - started
    write_npz(output, rgb, ir, metadata["train_raw_pids"], metadata["train_file_paths"], provenance)
    return {"output": str(output), "n_images": len(ir), "provenance": provenance}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config_file", default=None, help="Default: configs.yaml beside checkpoint")
    parser.add_argument("--root_dir", default=None, help="Relocated dataset root from the base config")
    parser.add_argument("--dev_split_file", required=True)
    parser.add_argument("--output", required=True, help="New .npz archive; existing output is refused")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verify_dev_npz", required=True, help="Corresponding R13 base npz; mandatory clean/f_ir verification")
    parser.add_argument("--limit", type=int, default=None,
                        help="Export only first N training image pairs; dev verification always uses its full gallery")
    return parser


if __name__ == "__main__":
    print(json.dumps(export(build_parser().parse_args()), ensure_ascii=False, indent=2, default=str))
