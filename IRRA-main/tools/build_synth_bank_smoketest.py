"""
Smoke test for the synthetic caption bank append pipeline.

Purpose:
  Your previous bank (pseudo_synth_bank_from_alt_caps.json) ended up empty, so
  the training log showed `added_synth_captions=0`. That by itself does NOT
  tell us whether the *bug* is in the bank generator or in the dataloader-side
  append logic. This script isolates the two.

What it does:
  For every image in the train split, fabricate K identical-length dummy
  captions with a unique marker (e.g. "SMOKETEST_1"). The bank is GUARANTEED
  non-empty and every key matches the train split (same key rule as
  check_synth_caption_bank.py: basename(img_path)).

How to verify after building:

  1. Run check_synth_caption_bank.py and confirm matched_train_keys == bank_total_keys.

  2. Run ONE quick training epoch with:
       --use_synth_caption_bank
       --synth_caption_bank_file <this bank>
       --synth_caption_keep_per_image 2
     and grep master.log for:
       added_synth_captions   -> should be ~= 2 * num_train_images (non-zero)
       total_train_captions_after_append  -> should be > 5497

  If added_synth_captions is still 0 with THIS bank, the bug is in the
  dataloader-side append logic (not the generator), and we have to go fix
  build_dataloader directly.

  If added_synth_captions is non-zero with THIS bank, the append path is fine
  and we can trust build_large_synth_bank_v5_llm.py to actually work once its
  output is non-empty.
"""
import argparse
import json
import os
import os.path as op
import sys
from collections import OrderedDict

# Match the sys.path convention used by build_large_synth_bank_v2.py / v3 / v4
# so we can import the project's dataset loaders the same way.
PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
TOOL_ROOT = op.abspath(op.dirname(__file__))
for path in (PROJECT_ROOT, TOOL_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
from datasets.rstpreid import RSTPReid
from datasets.rgbtpedes import RGBTPEDES


DATASET_FACTORY = {
    "CUHK-PEDES": CUHKPEDES,
    "ICFG-PEDES": ICFGPEDES,
    "RSTPReid": RSTPReid,
    "RGBT-PEDES": RGBTPEDES,
}


def _extract_img_path_from_train_sample(sample):
    # Legacy: (pid, image_id, img_path, caption)
    # RGBT:   (pid, image_id, rgb_path, thermal_path, caption)
    if len(sample) in (4, 5):
        return sample[2]
    raise ValueError(
        "Unsupported train sample format. Expected len=4 or 5, got "
        f"len={len(sample)}."
    )


def _collect_unique_image_keys(train_samples):
    """One base key per image_id. This is the same rule as
    check_synth_caption_bank.py and (we assume) the dataloader append path."""
    seen = OrderedDict()
    for sample in train_samples:
        image_id = sample[1]
        if image_id in seen:
            continue
        img_path = _extract_img_path_from_train_sample(sample)
        seen[image_id] = op.basename(str(img_path))
    return list(seen.values())


def build_smoketest_bank(train_keys, captions_per_image):
    bank = OrderedDict()
    for key in train_keys:
        bank[key] = [
            f"SMOKETEST_{i+1} a person wearing a shirt and pants, carrying a bag."
            for i in range(captions_per_image)
        ]
    return bank


def parse_args():
    parser = argparse.ArgumentParser(
        description="Produce a guaranteed non-empty smoke-test bank to "
        "validate the dataloader-side synth caption append path."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="RGBT-PEDES",
        choices=list(DATASET_FACTORY.keys()),
    )
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument(
        "--output_file",
        type=str,
        default="./synth_caption_bank_smoketest.json",
    )
    parser.add_argument(
        "--captions_per_image",
        type=int,
        default=2,
        help="number of dummy captions per image. Keep >= 2 to stress-test "
        "synth_caption_keep_per_image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.captions_per_image < 1:
        raise ValueError("captions_per_image must be >= 1")

    dataset = DATASET_FACTORY[args.dataset_name](root=args.root_dir, verbose=False)
    train_keys = _collect_unique_image_keys(dataset.train)
    bank = build_smoketest_bank(train_keys, args.captions_per_image)

    output_file = op.abspath(args.output_file)
    output_dir = op.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(bank, f, indent=2)

    total_captions = sum(len(v) for v in bank.values())

    print(f"[SMOKETEST] dataset_name={args.dataset_name}")
    print(f"[SMOKETEST] root_dir={args.root_dir}")
    print(f"[SMOKETEST] caption_file={dataset.anno_path}")
    print(f"[SMOKETEST] output_file={output_file}")
    print("[SMOKETEST] key_rule=basename(img_path)")
    print(f"[SMOKETEST] num_unique_train_images={len(train_keys)}")
    print(f"[SMOKETEST] captions_per_image={args.captions_per_image}")
    print(f"[SMOKETEST] bank_total_keys={len(bank)}")
    print(f"[SMOKETEST] bank_total_captions={total_captions}")
    print("[SMOKETEST] first_3_keys_and_values:")
    for k in list(bank.keys())[:3]:
        print(f"  - {k}: {bank[k]}")
    print(
        "[SMOKETEST] NEXT STEPS:\n"
        f"  1) python check_synth_caption_bank.py --dataset_name {args.dataset_name} "
        f"--root_dir {args.root_dir} --synth_caption_bank_file {output_file}\n"
        "     -> expect matched_train_keys == bank_total_keys\n"
        "  2) run 1 train epoch with --use_synth_caption_bank "
        f"--synth_caption_bank_file {output_file} "
        f"--synth_caption_keep_per_image {args.captions_per_image}\n"
        f"     -> expect added_synth_captions ~= "
        f"{args.captions_per_image * len(train_keys)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
