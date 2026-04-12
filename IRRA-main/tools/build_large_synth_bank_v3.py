import argparse
import json
import os
import os.path as op
import re
import sys
from collections import OrderedDict

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
TOOL_ROOT = op.abspath(op.dirname(__file__))
for path in (PROJECT_ROOT, TOOL_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from build_large_synth_bank_v2 import (
    COLOR_WORDS,
    DATASET_FACTORY,
    _choose_canonical_source,
    build_normal_caption,
    build_thermal_caption,
)


COLOR_GROUP_PATTERN = r"(?:" + "|".join(sorted(COLOR_WORDS)) + r")"
ATTR_GROUP_PATTERNS = OrderedDict(
    [
        (
            "hair",
            (
                rf"\blong(?:\s+{COLOR_GROUP_PATTERN}){{0,2}}\s+hair\b",
                rf"\bshort(?:\s+{COLOR_GROUP_PATTERN}){{0,2}}\s+hair\b",
            ),
        ),
        ("upper", (r"\btop\b", r"\bshirt\b", r"\bhoodie\b", r"\bjacket\b", r"\bcoat\b")),
        ("lower", (r"\bpants\b", r"\bjeans\b", r"\bskirt\b", r"\bshorts\b")),
        ("shoes", (r"\bshoes\b",)),
        ("bag", (r"\bbackpack\b", r"\bshoulder bag\b", r"\bbag\b")),
        ("accessory", (r"\bglasses\b", r"\bhat\b")),
        (
            "action",
            (
                r"\bwalking\b",
                r"\bstanding\b",
                r"\briding a bicycle\b",
                r"\bcarrying (?:a )?(?:[a-z]+ )?(?:bag|backpack|shoulder bag)\b",
            ),
        ),
    ]
)

COLOR_RE = re.compile(r"\b(?:" + "|".join(sorted(COLOR_WORDS)) + r")\b", re.IGNORECASE)
NOISE_RE = re.compile(
    r"\b(?:he|she|his|her|their|young|old|slim|thin|obese|fat|moderate|"
    r"medium|rather|relatively|man|woman|boy|girl|male|female)\b",
    re.IGNORECASE,
)
BAD_STRUCTURE_RE = re.compile(
    r"\bA figure person\b|\bA figure (?:he|she)\b|\bA person of\b|"
    r"\bwearing a dress in (?:his|her|their)\s+(?:left|right)?\s*hand\b",
    re.IGNORECASE,
)


def _normalize_spaces(text):
    text = re.sub(r"\s+", " ", str(text))
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip(" ,.;:")


def _extract_img_path_from_train_sample(sample):
    if len(sample) in (4, 5):
        return sample[2]
    raise ValueError(
        "Unsupported train sample format for synthetic captions. "
        f"Expected len=4 or len=5, got len={len(sample)}."
    )


def _collect_train_groups(train_samples):
    groups = OrderedDict()
    for sample in train_samples:
        image_id = sample[1]
        if image_id not in groups:
            img_path = _extract_img_path_from_train_sample(sample)
            groups[image_id] = {
                "key": op.basename(str(img_path)),
                "captions": [],
            }
        groups[image_id]["captions"].append(sample[-1])
    return groups


def _word_count(caption):
    return len(re.findall(r"[a-z-]+", caption.lower()))


def _caption_body(caption, prefix):
    caption = _normalize_spaces(caption).rstrip(".").lower()
    prefix = prefix.lower()
    if caption.startswith(prefix):
        return caption[len(prefix):].strip()
    return caption


def _token_set(text):
    return set(re.findall(r"[a-z-]+", text.lower()))


def _similarity(text_a, text_b):
    tokens_a = _token_set(text_a)
    tokens_b = _token_set(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / float(len(tokens_a | tokens_b))


def count_attr_groups(caption):
    text = caption.lower()
    groups = set()
    for group, patterns in ATTR_GROUP_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            groups.add(group)
    return groups


def _is_bad_rewrite(normal, thermal):
    normal = _normalize_spaces(normal)
    thermal = _normalize_spaces(thermal)
    if not normal.startswith("A person "):
        return True, "bad_normal_prefix"
    if not thermal.startswith("A figure "):
        return True, "bad_thermal_prefix"
    if _word_count(normal) < 6:
        return True, "normal_too_short"
    if _word_count(thermal) < 5:
        return True, "thermal_too_short"
    if BAD_STRUCTURE_RE.search(normal) or BAD_STRUCTURE_RE.search(thermal):
        return True, "bad_structure"
    if NOISE_RE.search(normal) or NOISE_RE.search(thermal):
        return True, "noise_word_residual"
    if COLOR_RE.search(thermal):
        return True, "thermal_color_residual"

    normal_body = _caption_body(normal, "A person")
    thermal_body = _caption_body(thermal, "A figure")
    if (
        _similarity(normal_body, thermal_body) >= 0.92
        and _word_count(normal) <= 8
        and _word_count(thermal) <= 8
    ):
        return True, "near_duplicate_short_pair"
    return False, ""


def _drop_low_info(normal_groups, thermal_groups):
    if len(normal_groups) < 3:
        return True, "normal_attr_groups_lt_3"
    if len(thermal_groups) < 2:
        return True, "thermal_attr_groups_lt_2"
    weak_groups = {"hair", "accessory"}
    if thermal_groups and thermal_groups <= weak_groups:
        return True, "thermal_weak_groups_only"
    return False, ""


def build_bank(train_samples):
    groups = _collect_train_groups(train_samples)
    bank = OrderedDict()
    kept_examples = []
    dropped_examples = []
    dropped_low_info_keys = 0
    dropped_bad_rewrite_keys = 0
    kept_normal_group_total = 0
    kept_thermal_group_total = 0

    for group in groups.values():
        key = group["key"]
        source = _choose_canonical_source(group["captions"])
        normal = build_normal_caption(source)
        thermal = build_thermal_caption(source)
        normal_groups = count_attr_groups(normal)
        thermal_groups = count_attr_groups(thermal)

        bad_rewrite, bad_reason = _is_bad_rewrite(normal, thermal)
        low_info, low_info_reason = _drop_low_info(normal_groups, thermal_groups)
        record = {
            "key": key,
            "reason": bad_reason if bad_rewrite else low_info_reason,
            "source": source,
            "normal": normal,
            "thermal": thermal,
            "normal_attr_groups": sorted(normal_groups),
            "thermal_attr_groups": sorted(thermal_groups),
        }

        if bad_rewrite:
            dropped_bad_rewrite_keys += 1
            if len(dropped_examples) < 5:
                dropped_examples.append(record)
            continue
        if low_info:
            dropped_low_info_keys += 1
            if len(dropped_examples) < 5:
                dropped_examples.append(record)
            continue

        bank[key] = [normal, thermal]
        kept_normal_group_total += len(normal_groups)
        kept_thermal_group_total += len(thermal_groups)
        if len(kept_examples) < 5:
            kept_examples.append(record)

    kept_keys = len(bank)
    stats = {
        "total_train_keys_seen": len(groups),
        "kept_keys": kept_keys,
        "dropped_low_info_keys": dropped_low_info_keys,
        "dropped_bad_rewrite_keys": dropped_bad_rewrite_keys,
        "average_normal_attr_groups": (
            kept_normal_group_total / float(kept_keys) if kept_keys else 0.0
        ),
        "average_thermal_attr_groups": (
            kept_thermal_group_total / float(kept_keys) if kept_keys else 0.0
        ),
        "kept_examples": kept_examples,
        "dropped_examples": dropped_examples,
    }
    return bank, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a high-confidence v3 synthetic caption bank for the train split."
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
        "--output_file",
        type=str,
        default="./synth_caption_bank_large_v3.json",
        help="output json path for the synthetic caption bank",
    )
    return parser.parse_args()


def _print_example(example):
    print(f"  - {example['key']}")
    print(f"    reason: {example['reason']}")
    print(f"    original canonical source: {example['source']}")
    print(f"    rewritten normal: {example['normal']}")
    print(f"    rewritten thermal: {example['thermal']}")
    print(f"    normal_attr_groups: {example['normal_attr_groups']}")
    print(f"    thermal_attr_groups: {example['thermal_attr_groups']}")


def main():
    args = parse_args()

    dataset = DATASET_FACTORY[args.dataset_name](root=args.root_dir, verbose=False)
    bank, stats = build_bank(dataset.train)

    output_file = op.abspath(args.output_file)
    output_dir = op.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(bank, f, indent=4, separators=(",", ": "))

    print(f"[SYNTH-V3] dataset_name={args.dataset_name}")
    print(f"[SYNTH-V3] root_dir={args.root_dir}")
    print(f"[SYNTH-V3] caption_file={dataset.anno_path}")
    print(f"[SYNTH-V3] output_file={output_file}")
    print("[SYNTH-V3] split=train")
    print("[SYNTH-V3] key_rule=basename(img_path)")
    print(f"[SYNTH-V3] total_train_keys_seen={stats['total_train_keys_seen']}")
    print(f"[SYNTH-V3] kept_keys={stats['kept_keys']}")
    print(f"[SYNTH-V3] dropped_low_info_keys={stats['dropped_low_info_keys']}")
    print(f"[SYNTH-V3] dropped_bad_rewrite_keys={stats['dropped_bad_rewrite_keys']}")
    print(
        "[SYNTH-V3] average_normal_attr_groups="
        f"{stats['average_normal_attr_groups']:.3f}"
    )
    print(
        "[SYNTH-V3] average_thermal_attr_groups="
        f"{stats['average_thermal_attr_groups']:.3f}"
    )
    print("[SYNTH-V3] first_5_kept_examples:")
    for example in stats["kept_examples"]:
        _print_example(example)
    print("[SYNTH-V3] first_5_dropped_examples:")
    for example in stats["dropped_examples"]:
        _print_example(example)

    return 0


if __name__ == "__main__":
    sys.exit(main())
