import argparse
import os.path as op
import re
import sys

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.iotools import read_json
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

FORBIDDEN_PHRASES = (
    "the image shows",
    "appears to",
    "maybe",
    "possibly",
    "as an ai",
)
COLOR_WORDS = {
    "black", "white", "red", "blue", "green", "yellow", "gray", "grey",
    "brown", "orange", "pink", "purple", "beige",
}
BAG_ITEM_WORDS = {"bag", "backpack", "handbag", "shoulder bag", "crossbody bag"}
SHOE_ITEM_WORDS = {"shoes", "sneakers", "boots"}
MALE_WORDS = {"man", "male", "boy"}
FEMALE_WORDS = {"woman", "female", "girl"}


def _extract_img_path_from_train_sample(sample):
    # Legacy: (pid, image_id, img_path, caption)
    # RGBT:  (pid, image_id, rgb_path, thermal_path, caption)
    if len(sample) == 5:
        return sample[2]
    if len(sample) == 4:
        return sample[2]
    raise ValueError(
        "Unsupported train sample format for key extraction. "
        f"Expected len=4 or len=5, got len={len(sample)}."
    )


def _collect_train_image_keys(train_samples):
    # Keep one base sample per image_id (same rule as train-time append logic).
    base_sample_by_image_id = {}
    for sample in train_samples:
        image_id = sample[1]
        if image_id not in base_sample_by_image_id:
            base_sample_by_image_id[image_id] = sample

    train_keys = []
    for sample in base_sample_by_image_id.values():
        img_path = _extract_img_path_from_train_sample(sample)
        train_keys.append(op.basename(str(img_path)))
    return train_keys


def _contains_phrase(text, phrase):
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def _color_item_phrases(text, item_words):
    text = str(text).lower()
    phrases = set()
    for color in COLOR_WORDS:
        for item in item_words:
            pattern = r"\b" + re.escape(color) + r"(?:\s+\w+){0,2}\s+" + re.escape(item) + r"\b"
            if re.search(pattern, text):
                phrases.add((color, item))
    return phrases


def _quality_scan(bank):
    quality = {
        "keys_with_more_than_3_captions": 0,
        "captions_with_forbidden_phrases": 0,
        "possible_hair_conflicts": 0,
        "possible_gender_conflicts": 0,
        "possible_bag_color_conflicts": 0,
        "possible_shoe_color_conflicts": 0,
        "examples": [],
    }

    for key, captions in bank.items():
        if not isinstance(captions, list):
            continue
        if len(captions) > 3:
            quality["keys_with_more_than_3_captions"] += 1

        lowered_caps = [str(c).lower() for c in captions if isinstance(c, str)]
        for caption in lowered_caps:
            if any(phrase in caption for phrase in FORBIDDEN_PHRASES):
                quality["captions_with_forbidden_phrases"] += 1
                if len(quality["examples"]) < 5:
                    quality["examples"].append((key, "forbidden_phrase", caption))

        joined = " ".join(lowered_caps)
        hair_conflict = _contains_phrase(joined, "long hair") and _contains_phrase(joined, "short hair")
        male = any(_contains_phrase(joined, word) for word in MALE_WORDS)
        female = any(_contains_phrase(joined, word) for word in FEMALE_WORDS)
        bag_colors = {color for color, _item in _color_item_phrases(joined, BAG_ITEM_WORDS)}
        shoe_colors = {color for color, _item in _color_item_phrases(joined, SHOE_ITEM_WORDS)}

        checks = [
            ("possible_hair_conflicts", hair_conflict, "hair_conflict"),
            ("possible_gender_conflicts", male and female, "gender_conflict"),
            ("possible_bag_color_conflicts", len(bag_colors) > 1, "bag_color_conflict"),
            ("possible_shoe_color_conflicts", len(shoe_colors) > 1, "shoe_color_conflict"),
        ]
        for counter, hit, reason in checks:
            if hit:
                quality[counter] += 1
                if len(quality["examples"]) < 5:
                    quality["examples"].append((key, reason, " | ".join(captions[:3])))

    return quality


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check synthetic caption bank key alignment against train split."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="RGBT-PEDES",
        choices=list(DATASET_FACTORY.keys()),
        help="dataset name used to build train split",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./data",
        help="dataset root dir (same as training --root_dir)",
    )
    parser.add_argument(
        "--synth_caption_bank_file",
        type=str,
        required=True,
        help="path to synthetic caption bank json",
    )
    parser.add_argument(
        "--show_unmatched_topk",
        type=int,
        default=5,
        help="number of unmatched key examples to print",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not op.isfile(args.synth_caption_bank_file):
        print(
            f"[SYNTH-CHECK] error: bank file not found: {args.synth_caption_bank_file}"
        )
        return 1

    bank = read_json(args.synth_caption_bank_file)
    if not isinstance(bank, dict):
        print(
            "[SYNTH-CHECK] error: bank file must be a JSON object "
            "{<stable_image_key>: [caption1, caption2, ...], ...}"
        )
        return 1

    dataset = DATASET_FACTORY[args.dataset_name](root=args.root_dir, verbose=False)
    train_keys = _collect_train_image_keys(dataset.train)
    train_key_set = set(train_keys)

    bank_keys = list(bank.keys())
    matched_keys = [k for k in bank_keys if k in train_key_set]
    unmatched_keys = [k for k in bank_keys if k not in train_key_set]

    total_synth_captions = 0
    invalid_value_key_count = 0
    for key in bank_keys:
        captions = bank[key]
        if isinstance(captions, list):
            total_synth_captions += len(captions)
        else:
            invalid_value_key_count += 1

    topk = max(0, int(args.show_unmatched_topk))
    matched_examples = matched_keys[:5]
    unmatched_examples = unmatched_keys[:topk]
    avg_caps_per_image = (
        total_synth_captions / float(len(bank_keys)) if bank_keys else 0.0
    )
    quality = _quality_scan(bank)

    print(f"[SYNTH-CHECK] dataset_name={args.dataset_name}")
    print(f"[SYNTH-CHECK] root_dir={args.root_dir}")
    print(f"[SYNTH-CHECK] bank_file={args.synth_caption_bank_file}")
    print("[SYNTH-CHECK] train_key_rule=basename(img_path)")
    print(f"[SYNTH-CHECK] train_image_keys={len(train_keys)}")
    print(f"[SYNTH-CHECK] bank_total_keys={len(bank_keys)}")
    print(f"[SYNTH-CHECK] matched_train_keys={len(matched_keys)}")
    print(f"[SYNTH-CHECK] unmatched_bank_keys={len(unmatched_keys)}")
    print(f"[SYNTH-CHECK] total_synthetic_captions={total_synth_captions}")
    print(f"[SYNTH-CHECK] avg_captions_per_image={avg_caps_per_image:.3f}")
    print(
        "[SYNTH-CHECK] invalid_value_key_count="
        f"{invalid_value_key_count} (non-list values)"
    )
    print(
        "[SYNTH-CHECK] keys_with_more_than_3_captions="
        f"{quality['keys_with_more_than_3_captions']}"
    )
    print(
        "[SYNTH-CHECK] captions_with_forbidden_phrases="
        f"{quality['captions_with_forbidden_phrases']}"
    )
    print(
        "[SYNTH-CHECK] possible_hair_conflicts="
        f"{quality['possible_hair_conflicts']}"
    )
    print(
        "[SYNTH-CHECK] possible_gender_conflicts="
        f"{quality['possible_gender_conflicts']}"
    )
    print(
        "[SYNTH-CHECK] possible_bag_color_conflicts="
        f"{quality['possible_bag_color_conflicts']}"
    )
    print(
        "[SYNTH-CHECK] possible_shoe_color_conflicts="
        f"{quality['possible_shoe_color_conflicts']}"
    )
    if quality["examples"]:
        print("[SYNTH-CHECK] possible_conflict_examples_top5:")
        for key, reason, preview in quality["examples"][:5]:
            print(f"  - {key} [{reason}]: {preview}")
    else:
        print("[SYNTH-CHECK] possible_conflict_examples_top5: []")
    if matched_examples:
        print("[SYNTH-CHECK] matched_examples_top5:")
        for key in matched_examples:
            captions = bank.get(key, [])
            preview = captions[0] if isinstance(captions, list) and captions else ""
            print(f"  - {key}: {preview}")
    else:
        print("[SYNTH-CHECK] matched_examples_top5: []")
    if unmatched_examples:
        print(f"[SYNTH-CHECK] unmatched_examples_top{topk}:")
        for key in unmatched_examples:
            captions = bank.get(key, [])
            preview = captions[0] if isinstance(captions, list) and captions else ""
            print(f"  - {key}: {preview}")
    else:
        print(f"[SYNTH-CHECK] unmatched_examples_top{topk}: []")

    return 0


if __name__ == "__main__":
    sys.exit(main())
