"""Build rule-based MGA pseudo JSONL from existing RGBT-PEDES captions.

This script is intentionally offline-only: it does not call any API and does
not generate captions from images. It converts train split human captions into
structured pseudo text fields compatible with ``datasets/mga_pseudo.py``.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

COLOR_WORDS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "gray",
    "grey",
    "pink",
    "purple",
    "orange",
    "brown",
    "dark",
    "light",
}

VISIBLE_TERMS = {
    "shirt": "shirt",
    "clothes": "clothes",
    "clothing": "clothing",
    "t-shirt": "t-shirt",
    "tshirt": "t-shirt",
    "coat": "coat",
    "jacket": "jacket",
    "pants": "pants",
    "jeans": "jeans",
    "trousers": "trousers",
    "shorts": "shorts",
    "dress": "dress",
    "skirt": "skirt",
    "shoes": "shoes",
    "shoe": "shoes",
    "sneaker": "sneakers",
    "sneakers": "sneakers",
    "backpack": "backpack",
    "bag": "bag",
    "handbag": "handbag",
    "hat": "hat",
    "cap": "cap",
}

THERMAL_PHRASES = (
    "long sleeve",
    "short sleeve",
    "long hair",
    "short hair",
)

THERMAL_TERMS = {
    "trousers": "trousers",
    "pants": "pants",
    "shorts": "shorts",
    "dress": "dress",
    "skirt": "skirt",
    "backpack": "backpack",
    "bag": "bag",
    "handbag": "handbag",
    "hat": "hat",
    "cap": "cap",
    "carrying": "carrying",
    "holding": "holding",
    "standing": "standing",
    "walking": "walking",
}

SHORT_CAPTION_WORDS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build rule-based MGA pseudo JSONL from RGBT-PEDES captions."
    )
    parser.add_argument("--root_dir", required=True, help="Dataset root directory.")
    parser.add_argument("--dataset_name", default="RGBT-PEDES")
    parser.add_argument("--out_json", required=True, help="Output JSONL path.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--max_items",
        default=0,
        type=int,
        help="Maximum number of train samples to write. 0 means all samples.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--debug_examples", default=5, type=int)
    return parser.parse_args()


def clean_caption(caption: Any) -> str:
    text = "" if caption is None else str(caption)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"([,.;:!?])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text if text else "person"


def caption_hash(caption: str) -> str:
    normalized = " ".join(str(caption).split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def json_safe_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def normalize_for_matching(text: str) -> str:
    text = text.lower()
    text = text.replace("t shirt", "tshirt")
    text = text.replace("t-shirt", "tshirt")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_tokens(text: str) -> List[str]:
    normalized = normalize_for_matching(text)
    if not normalized:
        return []
    return normalized.split()


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        item = " ".join(str(item).split()).strip()
        if item == "" or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def has_phrase(normalized_text: str, phrase: str) -> bool:
    phrase = normalize_for_matching(phrase)
    if phrase == "":
        return False
    return bool(re.search(rf"(^|\s){re.escape(phrase)}($|\s)", normalized_text))


def make_attr_sentence(attributes: Sequence[str]) -> str:
    attrs = dedupe_keep_order(attributes)
    if not attrs:
        return ""
    if len(attrs) == 1:
        attr_text = attrs[0]
    elif len(attrs) == 2:
        attr_text = f"{attrs[0]} and {attrs[1]}"
    else:
        attr_text = ", ".join(attrs[:-1]) + f" and {attrs[-1]}"
    return f"The person has {attr_text}."


def make_short_caption(caption: str) -> str:
    words = caption.split()
    if len(words) <= SHORT_CAPTION_WORDS:
        short = caption
    else:
        short = " ".join(words[:SHORT_CAPTION_WORDS])
    if short and short[-1] not in ".!?":
        short += "."
    return short


def extract_rgb_color_attributes(caption: str) -> List[str]:
    tokens = word_tokens(caption)
    attrs = []
    color_positions = [
        (idx, token)
        for idx, token in enumerate(tokens)
        if token in COLOR_WORDS
    ]
    used_color_positions = set()

    for idx, token in enumerate(tokens):
        canonical_term = VISIBLE_TERMS.get(token)
        if canonical_term is None:
            continue

        nearest_color = None
        nearest_color_idx = None
        for color_idx, color in reversed(color_positions):
            if color_idx >= idx:
                continue
            if idx - color_idx <= 3:
                nearest_color = color
                nearest_color_idx = color_idx
                break

        if nearest_color is not None:
            attrs.append(f"{nearest_color} {canonical_term}")
            used_color_positions.add(nearest_color_idx)
        else:
            attrs.append(canonical_term)

    for color_idx, color in color_positions:
        if color_idx not in used_color_positions:
            attrs.append(color)

    return dedupe_keep_order(attrs)


def extract_thermal_shape_attributes(caption: str) -> List[str]:
    normalized = normalize_for_matching(caption)
    tokens = word_tokens(caption)
    attrs = []

    for phrase in THERMAL_PHRASES:
        if has_phrase(normalized, phrase):
            attrs.append(phrase)

    for token in tokens:
        canonical_term = THERMAL_TERMS.get(token)
        if canonical_term is not None:
            attrs.append(canonical_term)

    return dedupe_keep_order(attrs)


def extract_general_attributes(caption: str) -> List[str]:
    rgb_attrs = extract_rgb_color_attributes(caption)
    thermal_attrs = extract_thermal_shape_attributes(caption)

    filtered_thermal_attrs = []
    for thermal_attr in thermal_attrs:
        covered_by_rgb = any(
            rgb_attr == thermal_attr or rgb_attr.endswith(f" {thermal_attr}")
            for rgb_attr in rgb_attrs
        )
        if not covered_by_rgb:
            filtered_thermal_attrs.append(thermal_attr)

    return dedupe_keep_order([*rgb_attrs, *filtered_thermal_attrs])


def build_pseudo_record(sample: Tuple[Any, ...]) -> Dict[str, Any]:
    if len(sample) == 5:
        pid, image_id, rgb_path, thermal_path, caption = sample
    elif len(sample) == 4:
        pid, image_id, rgb_path, caption = sample
        thermal_path = ""
    else:
        raise ValueError(
            "Unsupported train sample format. Expected len=4 or len=5, "
            f"got len={len(sample)}."
        )

    original_caption = "" if caption is None else str(caption)
    cleaned_caption = clean_caption(original_caption)
    short_caption = make_short_caption(cleaned_caption)

    general_attrs = extract_general_attributes(cleaned_caption)
    attr_sentence = make_attr_sentence(general_attrs)
    if attr_sentence:
        attr_text = attr_sentence
        attr_weight = 0.7
    else:
        attr_text = cleaned_caption
        attr_weight = 0.5

    rgb_attrs = extract_rgb_color_attributes(cleaned_caption)
    rgb_color_texts = []
    rgb_sentence = make_attr_sentence(rgb_attrs)
    if rgb_sentence:
        rgb_color_texts.append(
            {
                "text": rgb_sentence,
                "weight": 0.8,
                "source": "rule_rgb_color",
            }
        )

    thermal_attrs = extract_thermal_shape_attributes(cleaned_caption)
    thermal_shape_texts = []
    thermal_sentence = make_attr_sentence(thermal_attrs)
    if thermal_sentence:
        thermal_shape_texts.append(
            {
                "text": thermal_sentence,
                "weight": 0.8,
                "source": "rule_thermal_shape",
            }
        )

    return {
        "pid": json_safe_scalar(pid),
        "image_id": json_safe_scalar(image_id),
        "rgb_path": "" if rgb_path is None else str(rgb_path),
        "thermal_path": "" if thermal_path is None else str(thermal_path),
        "caption": original_caption,
        "caption_hash": caption_hash(original_caption),
        "pseudo_texts": [
            {
                "text": cleaned_caption,
                "kind": "full",
                "weight": 1.0,
                "source": "rule_original",
            },
            {
                "text": short_caption,
                "kind": "short",
                "weight": 0.8,
                "source": "rule_short",
            },
            {
                "text": attr_text,
                "kind": "attribute",
                "weight": attr_weight,
                "source": "rule_attribute",
            },
        ],
        "rgb_color_texts": rgb_color_texts,
        "thermal_shape_texts": thermal_shape_texts,
        "negative_texts": [],
    }


def candidate_root_dirs(root_dir: str) -> List[str]:
    root_dir = os.path.abspath(root_dir)
    candidates = [root_dir]
    parent = os.path.dirname(root_dir)
    grandparent = os.path.dirname(parent)
    candidates.extend([parent, grandparent])

    out = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def load_rgbtpedes_dataset(root_dir: str) -> Any:
    from datasets.rgbtpedes import RGBTPEDES

    errors = []
    for candidate in candidate_root_dirs(root_dir):
        try:
            return RGBTPEDES(root=candidate, verbose=False)
        except RuntimeError as exc:
            errors.append(f"{candidate}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(
        "Could not load RGBT-PEDES with the provided --root_dir or its parents. "
        f"Tried:\n{joined}"
    )


def split_samples(dataset: Any, split: str) -> List[Tuple[Any, ...]]:
    split = str(split).lower()
    if split == "train":
        return list(dataset.train)
    if split == "val":
        samples, _ = dataset._process_anno(dataset.val_annos, training=True)
        return list(samples)
    if split == "test":
        samples, _ = dataset._process_anno(dataset.test_annos, training=True)
        return list(samples)
    raise ValueError(f"Unsupported split={split!r}.")


def select_indices(num_items: int, max_items: int, seed: int) -> List[int]:
    indices = list(range(num_items))
    if int(max_items) <= 0 or int(max_items) >= num_items:
        return indices

    rng = random.Random(int(seed))
    rng.shuffle(indices)
    return sorted(indices[: int(max_items)])


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def print_stats(
    total_samples: int,
    written_rows: int,
    stats: Counter,
    examples: Sequence[Dict[str, Any]],
) -> None:
    print(f"total samples: {total_samples}")
    print(f"written rows: {written_rows}")
    print(f"pseudo_texts count: {stats['pseudo_texts']}")
    print(f"rgb_color_texts count: {stats['rgb_color_texts']}")
    print(f"thermal_shape_texts count: {stats['thermal_shape_texts']}")
    print(f"missing rgb_color count: {stats['missing_rgb_color']}")
    print(f"missing thermal_shape count: {stats['missing_thermal_shape']}")
    print("debug examples:")
    for idx, example in enumerate(examples, start=1):
        print(f"[{idx}] {json.dumps(example, ensure_ascii=False)}")


def main() -> None:
    args = parse_args()
    if args.dataset_name != "RGBT-PEDES":
        raise ValueError(
            "This rule builder currently supports only --dataset_name RGBT-PEDES."
        )

    dataset = load_rgbtpedes_dataset(args.root_dir)
    samples = split_samples(dataset, args.split)
    indices = select_indices(len(samples), args.max_items, args.seed)

    ensure_parent_dir(args.out_json)
    stats = Counter()
    examples = []

    with open(args.out_json, "w", encoding="utf-8") as handle:
        for index in indices:
            record = build_pseudo_record(samples[index])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats["pseudo_texts"] += len(record["pseudo_texts"])
            stats["rgb_color_texts"] += len(record["rgb_color_texts"])
            stats["thermal_shape_texts"] += len(record["thermal_shape_texts"])
            if not record["rgb_color_texts"]:
                stats["missing_rgb_color"] += 1
            if not record["thermal_shape_texts"]:
                stats["missing_thermal_shape"] += 1
            if len(examples) < int(args.debug_examples):
                examples.append(record)

    print_stats(
        total_samples=len(samples),
        written_rows=len(indices),
        stats=stats,
        examples=examples,
    )
    print(f"out_json: {args.out_json}")


if __name__ == "__main__":
    main()
