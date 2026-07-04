import argparse
import os
import os.path as op
import sys
from types import SimpleNamespace

import torch


PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import build_dataloader
from datasets.rgbtpedes import RGBTPEDES
from utils.iotools import load_train_configs, mkdir_if_missing


DEFAULT_SCORE_FILE = (
    "/root/autodl-tmp/RGBT_REPRODUCE_RESULTS/route1_scores/"
    "seed0_epoch43_R1_57.7108/seed0_epoch43_R1_57.7108_scores.pt"
)
DEFAULT_OUTPUT_FILE = "outputs/paper_figures/rgbt_pedes_test_metadata.pt"


def _default_args():
    return SimpleNamespace(
        dataset_name="RGBT-PEDES",
        root_dir="datasets/TextReID",
        training=False,
        eval_dataset="test",
        img_size=(384, 128),
        text_length=77,
        test_batch_size=512,
        num_workers=0,
    )


def _load_args(config_file, root_dir):
    if config_file:
        args = load_train_configs(config_file)
    else:
        args = _default_args()
    args.dataset_name = "RGBT-PEDES"
    args.training = False
    args.eval_dataset = "test"
    if root_dir:
        args.root_dir = root_dir
    if not hasattr(args, "root_dir"):
        args.root_dir = "datasets/TextReID"
    if not hasattr(args, "img_size"):
        args.img_size = (384, 128)
    if not hasattr(args, "text_length"):
        args.text_length = 77
    if not hasattr(args, "test_batch_size"):
        args.test_batch_size = 512
    if not hasattr(args, "num_workers"):
        args.num_workers = 0
    return args


def _as_list(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return list(value)


def _require_sequential(indices, expected_len, name):
    values = _as_list(indices)
    expected = list(range(expected_len))
    if values != expected:
        raise AssertionError(
            f"{name} is not sequential 0..{expected_len - 1}. "
            f"head={values[:10]} tail={values[-10:] if values else []}"
        )


def _load_score(score_file):
    if not op.isfile(score_file):
        raise FileNotFoundError(
            f"Score file not found: {score_file}. "
            "Pass --score_file if the seed0 clean score file is elsewhere."
        )
    score = torch.load(score_file, map_location=torch.device("cpu"))
    if not isinstance(score, dict):
        raise TypeError(f"Score file must contain a dict, got {type(score)}.")
    for key in ["sim", "txt_indices", "img_indices"]:
        if key not in score:
            raise KeyError(f"Score file is missing required key: {key}")
    return score


def _build_metadata(args):
    # Reuse the same pure-test dataloader construction used by test.py.
    test_img_loader, test_txt_loader, _ = build_dataloader(args)
    txt_dataset = test_txt_loader.dataset
    img_dataset = test_img_loader.dataset

    if not hasattr(txt_dataset, "captions"):
        raise AttributeError("TextDataset does not expose captions.")
    if not hasattr(img_dataset, "img_paths_rgb") or not hasattr(img_dataset, "img_paths_t"):
        raise AttributeError(
            "ImageDatasetRGBT must expose img_paths_rgb and img_paths_t."
        )

    captions = list(txt_dataset.captions)
    rgb_paths = list(img_dataset.img_paths_rgb)
    ir_paths = list(img_dataset.img_paths_t)

    dataset = RGBTPEDES(root=args.root_dir, verbose=False)
    raw_image_pids = [int(anno["id"]) for anno in dataset.test_annos]
    raw_text_pids = [
        int(anno["id"])
        for anno in dataset.test_annos
        for _ in anno["captions"]
    ]

    return {
        "captions": captions,
        "rgb_paths": rgb_paths,
        "ir_paths": ir_paths,
        "txt_indices": torch.arange(len(captions), dtype=torch.long),
        "img_indices": torch.arange(len(rgb_paths), dtype=torch.long),
        "eval_text_pids": list(dataset.test["caption_pids"]),
        "eval_image_pids": list(dataset.test["image_pids"]),
        "raw_text_pids": raw_text_pids,
        "raw_image_pids": raw_image_pids,
        "source": "RGBT-PEDES test split metadata",
        "root_dir": str(args.root_dir),
        "annotation_file": str(dataset.anno_path),
    }


def _validate_metadata(metadata, score):
    sim = score["sim"]
    if not torch.is_tensor(sim) or sim.dim() != 2:
        raise AssertionError(f"score['sim'] must be a 2D tensor, got {type(sim)}.")
    num_text, num_image = int(sim.shape[0]), int(sim.shape[1])

    captions = metadata["captions"]
    rgb_paths = metadata["rgb_paths"]
    ir_paths = metadata["ir_paths"]
    if len(captions) != 2490:
        raise AssertionError(f"Expected 2490 captions, got {len(captions)}.")
    if len(rgb_paths) != 1473:
        raise AssertionError(f"Expected 1473 RGB paths, got {len(rgb_paths)}.")
    if len(ir_paths) != 1473:
        raise AssertionError(f"Expected 1473 IR paths, got {len(ir_paths)}.")
    if len(captions) != num_text:
        raise AssertionError(
            f"caption count {len(captions)} != score sim rows {num_text}."
        )
    if len(rgb_paths) != num_image:
        raise AssertionError(
            f"RGB path count {len(rgb_paths)} != score sim cols {num_image}."
        )
    if len(ir_paths) != num_image:
        raise AssertionError(
            f"IR path count {len(ir_paths)} != score sim cols {num_image}."
        )

    _require_sequential(metadata["txt_indices"], num_text, "metadata txt_indices")
    _require_sequential(metadata["img_indices"], num_image, "metadata img_indices")
    _require_sequential(score["txt_indices"], num_text, "score txt_indices")
    _require_sequential(score["img_indices"], num_image, "score img_indices")
    if len(_as_list(score["txt_indices"])) != len(_as_list(metadata["txt_indices"])):
        raise AssertionError("metadata txt_indices and score txt_indices length mismatch.")
    if len(_as_list(score["img_indices"])) != len(_as_list(metadata["img_indices"])):
        raise AssertionError("metadata img_indices and score img_indices length mismatch.")

    missing_rgb = [path for path in rgb_paths[:5] if not op.isfile(path)]
    missing_ir = [path for path in ir_paths[:5] if not op.isfile(path)]
    if missing_rgb:
        raise FileNotFoundError(f"First RGB path(s) missing: {missing_rgb}")
    if missing_ir:
        raise FileNotFoundError(f"First IR path(s) missing: {missing_ir}")


def main():
    parser = argparse.ArgumentParser(
        description="Dump RGBT-PEDES test split captions/RGB/IR paths in eval order."
    )
    parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--score_file", type=str, default=DEFAULT_SCORE_FILE)
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    build_args = _load_args(args.config_file, args.root_dir)
    score = _load_score(args.score_file)
    metadata = _build_metadata(build_args)
    _validate_metadata(metadata, score)

    output_file = op.abspath(args.output_file)
    mkdir_if_missing(op.dirname(output_file))
    torch.save(metadata, output_file)

    print("[RGBT_METADATA] validation=passed")
    print(f"[RGBT_METADATA] captions={len(metadata['captions'])}")
    print(f"[RGBT_METADATA] rgb_paths={len(metadata['rgb_paths'])}")
    print(f"[RGBT_METADATA] ir_paths={len(metadata['ir_paths'])}")
    print("[RGBT_METADATA] first5_captions:")
    for idx, caption in enumerate(metadata["captions"][:5]):
        print(f"  {idx}: {caption}")
    print("[RGBT_METADATA] first5_rgb_paths:")
    for idx, path in enumerate(metadata["rgb_paths"][:5]):
        print(f"  {idx}: {path}")
    print("[RGBT_METADATA] first5_ir_paths:")
    for idx, path in enumerate(metadata["ir_paths"][:5]):
        print(f"  {idx}: {path}")
    print(f"[RGBT_METADATA] saved={output_file}")


if __name__ == "__main__":
    main()
