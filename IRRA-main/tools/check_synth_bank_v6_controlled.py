"""
Check SynthCap-v6 controlled metadata or flat caption banks for RGBT-PEDES.
"""
import argparse
import json
import os.path as op
import random
import re
import statistics
import sys
from collections import Counter, OrderedDict


SPLIT = "train"
DEFAULT_ROOT_DIR = "datasets/TextReID"

FORBIDDEN_PHRASES = (
    "maybe",
    "probably",
    "possibly",
    "appears",
    "appears to",
    "seems",
    "seems to",
    "the image",
    "the photo",
    "the picture",
    "this image",
    "this photo",
    "as an ai",
    "thermal",
    "infrared",
    "rgb",
    "modality",
    "dataset",
)

COLOR_WORDS = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "gray",
    "grey",
    "brown",
    "orange",
    "pink",
    "purple",
    "beige",
    "khaki",
    "cyan",
    "gold",
    "golden",
    "silver",
    "blond",
    "blonde",
    "dark",
    "light",
}

CLOTHING_WORDS = {
    "shirt",
    "tshirt",
    "t-shirt",
    "clothing",
    "clothes",
    "top",
    "coat",
    "jacket",
    "hoodie",
    "sweater",
    "jumper",
    "dress",
    "skirt",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "shoes",
    "sneakers",
    "boots",
    "hat",
    "cap",
    "uniform",
}

BAG_TERMS = {
    "bag",
    "backpack",
    "handbag",
    "purse",
    "satchel",
    "school bag",
    "shoulder bag",
    "crossbody bag",
    "schoolbag",
    "briefcase",
}

SHOE_TERMS = {"shoes", "sneakers", "boots", "trainers", "footwear"}
BODY_BUILD_TERMS = {
    "thin",
    "slim",
    "skinny",
    "heavy",
    "fat",
    "obese",
    "stout",
    "medium",
    "average",
    "well-proportioned",
    "well proportioned",
    "medium build",
    "average build",
    "heavy build",
    "slim build",
    "thin build",
}
VIEWPOINT_TERMS = {
    "back",
    "behind",
    "from behind",
    "back view",
    "rear",
    "front",
    "from front",
    "front view",
    "side",
    "side view",
    "profile",
}
PHONE_TERMS = {"phone", "mobile phone", "cell phone", "cellphone", "telephone"}
HELD_ITEM_TERMS = PHONE_TERMS | BAG_TERMS | {"umbrella", "book", "paper", "bottle"}
ACTION_VARIANTS = {
    "walking": {"walk", "walks", "walking", "walked"},
    "standing": {"stand", "stands", "standing", "stood"},
    "holding": {"hold", "holds", "holding", "held"},
    "carrying": {"carry", "carries", "carrying", "carried", "take", "takes", "taking"},
    "using": {"use", "uses", "using", "used", "texting"},
    "riding": {"ride", "rides", "riding", "rode"},
}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_caption_json(root_dir, caption_json):
    if caption_json:
        path = op.abspath(caption_json)
        if op.isfile(path):
            return path
        raise FileNotFoundError(f"caption_json does not exist: {path}")

    candidates = [
        op.join(root_dir, "RGBT-PEDES", "Caption.json"),
        op.join(root_dir, "RGBT-PEDES", "caption.json"),
        op.join(root_dir, "RGBT-PEDES", "RGBT-PEDES-Final", "Caption.json"),
        op.join(root_dir, "RGBT-PEDES", "RGBT-PEDES-Final", "caption.json"),
        op.join(root_dir, "RGBT-PEDES", "RGBT-PEDES", "Caption.json"),
        op.join(root_dir, "RGBT-PEDES", "RGBT-PEDES", "caption.json"),
    ]
    for path in candidates:
        if op.isfile(path):
            return op.abspath(path)
    raise FileNotFoundError(
        "Could not find RGBT-PEDES Caption.json. Tried: "
        + ", ".join(op.abspath(path) for path in candidates)
    )


def _basename_from_file_path(file_path):
    return op.basename(str(file_path).replace("\\", "/"))


def _normalize_text(text):
    text = str(text or "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _norm_key(text):
    text = _normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_count(text):
    return len(re.findall(r"[A-Za-z0-9-]+", str(text or "")))


def _term_set(text, terms):
    lowered = str(text or "").lower()
    found = set()
    for term in sorted(terms, key=len, reverse=True):
        if re.search(r"\b" + re.escape(term.lower()) + r"\b", lowered):
            found.add(term.lower())
    return found


def _color_set(text):
    found = _term_set(text, COLOR_WORDS)
    if "grey" in found:
        found.remove("grey")
        found.add("gray")
    if "blonde" in found:
        found.remove("blonde")
        found.add("blond")
    return found


def _bag_set(text):
    found = _term_set(text, BAG_TERMS)
    if any(term in found for term in found if term != "bag"):
        found.add("bag")
    return found


def _required_bag_set(text):
    return _term_set(text, BAG_TERMS)


def _shoe_set(text):
    return _term_set(text, SHOE_TERMS)


def _clothing_set(text):
    return _term_set(text, CLOTHING_WORDS)


def _body_build_set(text):
    return _term_set(text, BODY_BUILD_TERMS)


def _viewpoint_set(text):
    return _term_set(text, VIEWPOINT_TERMS)


def _action_set(text):
    lowered = str(text or "").lower()
    found = set()
    for canonical, variants in ACTION_VARIANTS.items():
        if any(re.search(r"\b" + re.escape(term) + r"\b", lowered) for term in variants):
            found.add(canonical)
    return found


def _source_allows_hold_or_carry(source_caption):
    return bool(
        _action_set(source_caption) & {"holding", "carrying", "using"}
        or _term_set(source_caption, HELD_ITEM_TERMS)
    )


def _has_forbidden(text):
    lowered = str(text or "").lower()
    return any(re.search(r"\b" + re.escape(term) + r"\b", lowered) for term in FORBIDDEN_PHRASES)


def collect_train_sources(caption_json_path):
    annos = _read_json(caption_json_path)
    if not isinstance(annos, list):
        raise ValueError(f"Caption file must contain a JSON list: {caption_json_path}")

    sources = OrderedDict()
    for anno in annos:
        if not isinstance(anno, dict):
            continue
        if str(anno.get("split", "train")).lower() != SPLIT:
            continue
        file_path = str(anno.get("file_path", anno.get("img_path", "")))
        if not file_path:
            continue
        key = _basename_from_file_path(file_path)
        captions = [
            _normalize_text(caption)
            for caption in anno.get("captions", [])
            if isinstance(caption, str) and _normalize_text(caption)
        ]
        if key not in sources:
            sources[key] = {
                "file_path": file_path,
                "pid": anno.get("id", None),
                "captions": captions,
            }
    return sources


def _select_default_source(source_record):
    captions = source_record.get("captions", [])
    if not captions:
        return ""
    return max(captions, key=lambda caption: (_word_count(caption), len(caption)))


def load_bank(path):
    obj = _read_json(path)
    if not isinstance(obj, dict):
        raise ValueError("bank file must be a JSON object")

    rows = []
    if isinstance(obj.get("items"), dict):
        bank_type = "metadata"
        for key, entries in obj["items"].items():
            if not isinstance(entries, list):
                rows.append(
                    {
                        "key": str(key),
                        "caption": "",
                        "source_caption": "",
                        "invalid": "metadata_entries_not_list",
                    }
                )
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    rows.append(
                        {
                            "key": str(key),
                            "caption": _normalize_text(entry.get("caption", "")),
                            "source_caption": _normalize_text(entry.get("source_caption", "")),
                            "quality_filter": _normalize_text(
                                entry.get("quality", {}).get("filter", "")
                                if isinstance(entry.get("quality", {}), dict)
                                else ""
                            ),
                            "quality_reason": _normalize_text(
                                entry.get("quality", {}).get("reason", "")
                                if isinstance(entry.get("quality", {}), dict)
                                else ""
                            ),
                            "invalid": "",
                        }
                    )
                else:
                    rows.append(
                        {
                            "key": str(key),
                            "caption": "",
                            "source_caption": "",
                            "quality_filter": "",
                            "quality_reason": "",
                            "invalid": "metadata_entry_not_object",
                        }
                    )
    else:
        bank_type = "flat"
        for key, captions in obj.items():
            if not isinstance(captions, list):
                rows.append(
                    {
                        "key": str(key),
                        "caption": "",
                        "source_caption": "",
                        "invalid": "flat_value_not_list",
                    }
                )
                continue
            for caption in captions:
                rows.append(
                    {
                        "key": str(key),
                        "caption": _normalize_text(caption) if isinstance(caption, str) else "",
                        "source_caption": "",
                        "quality_filter": "",
                        "quality_reason": "",
                        "invalid": "" if isinstance(caption, str) else "flat_caption_not_string",
                    }
                )
    return bank_type, rows


def _word_stats(lengths):
    if not lengths:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
        }
    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / float(len(lengths)),
        "median": float(statistics.median(lengths)),
    }


def analyze(rows, sources, bank_type):
    train_keys = set(sources.keys())
    bank_keys = [row["key"] for row in rows]
    bank_key_set = set(bank_keys)
    matched_keys = sorted(bank_key_set & train_keys)
    unmatched_keys = sorted(bank_key_set - train_keys)

    valid_rows = [row for row in rows if row.get("caption")]
    lengths = [_word_count(row["caption"]) for row in valid_rows]
    captions_by_key = Counter(row["key"] for row in valid_rows)
    avg_caps = (
        sum(captions_by_key.values()) / float(len(captions_by_key))
        if captions_by_key
        else 0.0
    )

    invalid_count = sum(1 for row in rows if row.get("invalid") or not row.get("caption"))
    normalized = [_norm_key(row["caption"]) for row in valid_rows]
    duplicate_count = sum(count - 1 for count in Counter(normalized).values() if count > 1)
    forbidden_count = sum(1 for row in valid_rows if _has_forbidden(row["caption"]))
    word_length_out_of_range_count = sum(
        1 for length in lengths if length < 8 or length > 45
    )

    strict_counts = OrderedDict(
        [
            ("new_color", 0),
            ("deleted_color", 0),
            ("new_bag", 0),
            ("deleted_bag", 0),
            ("new_shoe", 0),
            ("deleted_shoe", 0),
            ("new_clothing", 0),
            ("deleted_clothing", 0),
            ("new_body", 0),
            ("new_view", 0),
            ("new_action", 0),
            ("missing_source_caption", 0),
        ]
    )
    examples = {name: [] for name in strict_counts}

    if bank_type == "metadata":
        for row in valid_rows:
            source_caption = row.get("source_caption", "")
            if not source_caption:
                strict_counts["missing_source_caption"] += 1
                if len(examples["missing_source_caption"]) < 5:
                    examples["missing_source_caption"].append(
                        (row["key"], [], "", row["caption"])
                    )
                continue

            checks = []
            source_colors = _color_set(source_caption)
            synth_colors = _color_set(row["caption"])
            checks.append(("new_color", synth_colors - source_colors))
            checks.append(("deleted_color", source_colors - synth_colors))

            source_bags = _bag_set(source_caption)
            synth_bags = _bag_set(row["caption"])
            checks.append(("new_bag", synth_bags - source_bags))
            checks.append(
                (
                    "deleted_bag",
                    _required_bag_set(source_caption) - _required_bag_set(row["caption"]),
                )
            )

            source_shoes = _shoe_set(source_caption)
            synth_shoes = _shoe_set(row["caption"])
            checks.append(("new_shoe", synth_shoes - source_shoes))
            checks.append(("deleted_shoe", source_shoes - synth_shoes))

            source_clothing = _clothing_set(source_caption)
            synth_clothing = _clothing_set(row["caption"])
            checks.append(("new_clothing", synth_clothing - source_clothing))
            checks.append(("deleted_clothing", source_clothing - synth_clothing))

            checks.append(
                (
                    "new_body",
                    _body_build_set(row["caption"]) - _body_build_set(source_caption),
                )
            )
            checks.append(
                (
                    "new_view",
                    _viewpoint_set(row["caption"]) - _viewpoint_set(source_caption),
                )
            )

            source_actions = _action_set(source_caption)
            synth_actions = _action_set(row["caption"])
            new_actions = synth_actions - source_actions
            if _source_allows_hold_or_carry(source_caption):
                new_actions = new_actions - {"holding", "carrying", "using"}
            checks.append(("new_action", new_actions))

            for name, terms in checks:
                if terms:
                    strict_counts[name] += 1
                    if len(examples[name]) < 5:
                        examples[name].append(
                            (row["key"], sorted(terms), source_caption, row["caption"])
                        )

    return {
        "total_train_images": len(train_keys),
        "bank_keys": len(bank_key_set),
        "matched_keys": len(matched_keys),
        "unmatched_keys": len(unmatched_keys),
        "unmatched_key_examples": unmatched_keys[:10],
        "total_captions": len(valid_rows),
        "average_captions_per_image": avg_caps,
        "word_stats": _word_stats(lengths),
        "duplicate_count": duplicate_count,
        "forbidden_word_count": forbidden_count,
        "word_length_out_of_range_count": word_length_out_of_range_count,
        "potential_new_color_hallucination_count": strict_counts["new_color"],
        "potential_new_bag_hallucination_count": strict_counts["new_bag"],
        "invalid_count": invalid_count,
        "strict_counts": strict_counts,
        "strict_examples": examples,
        "source_strict_checks_enabled": bank_type == "metadata",
    }


def print_samples(rows, sources, sample_count, seed):
    valid_rows = [row for row in rows if row.get("caption") and row["key"] in sources]
    rng = random.Random(seed)
    if len(valid_rows) > sample_count:
        valid_rows = rng.sample(valid_rows, sample_count)
    else:
        rng.shuffle(valid_rows)

    print(f"[SYNTH-V6-CHECK] random_samples={len(valid_rows)}")
    for row in valid_rows:
        source_caption = row.get("source_caption") or _select_default_source(sources[row["key"]])
        print(f"  - key={row['key']}")
        print(f"    source: {source_caption}")
        print(f"    synth:  {row['caption']}")
        print(
            "    quality: "
            f"filter={row.get('quality_filter', '')} reason={row.get('quality_reason', '')}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check SynthCap-v6 controlled metadata or flat bank."
    )
    parser.add_argument("--bank_file", required=True)
    parser.add_argument("--root_dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument(
        "--caption_json",
        default="",
        help="Caption.json path. Empty means auto-resolve under --root_dir.",
    )
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--strict", default=False, action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    caption_json = resolve_caption_json(args.root_dir, args.caption_json)
    bank_type, rows = load_bank(args.bank_file)
    sources = collect_train_sources(caption_json)
    report = analyze(rows, sources, bank_type)

    print(f"[SYNTH-V6-CHECK] bank_file={op.abspath(args.bank_file)}")
    print(f"[SYNTH-V6-CHECK] bank_type={bank_type}")
    print(f"[SYNTH-V6-CHECK] caption_json={caption_json}")
    print(f"[SYNTH-V6-CHECK] total_train_images={report['total_train_images']}")
    print(f"[SYNTH-V6-CHECK] bank_keys={report['bank_keys']}")
    print(f"[SYNTH-V6-CHECK] matched_keys={report['matched_keys']}")
    print(f"[SYNTH-V6-CHECK] unmatched_keys={report['unmatched_keys']}")
    print(f"[SYNTH-V6-CHECK] total_captions={report['total_captions']}")
    print(
        "[SYNTH-V6-CHECK] average_captions_per_image="
        f"{report['average_captions_per_image']:.3f}"
    )
    ws = report["word_stats"]
    print(
        "[SYNTH-V6-CHECK] word_length_stats="
        f"min={ws['min']} max={ws['max']} mean={ws['mean']:.3f} median={ws['median']:.3f}"
    )
    print(f"[SYNTH-V6-CHECK] duplicate_count={report['duplicate_count']}")
    print(f"[SYNTH-V6-CHECK] forbidden_word_count={report['forbidden_word_count']}")
    print(
        "[SYNTH-V6-CHECK] word_length_out_of_range_count="
        f"{report['word_length_out_of_range_count']}"
    )
    print(
        "[SYNTH-V6-CHECK] potential_new_color_hallucination_count="
        f"{report['potential_new_color_hallucination_count']}"
    )
    print(
        "[SYNTH-V6-CHECK] potential_new_bag_hallucination_count="
        f"{report['potential_new_bag_hallucination_count']}"
    )
    print(f"[SYNTH-V6-CHECK] invalid_count={report['invalid_count']}")
    print(
        "[SYNTH-V6-CHECK] source_strict_checks_enabled="
        f"{report['source_strict_checks_enabled']}"
    )
    print(f"[SYNTH-V6-CHECK] unmatched_key_examples={report['unmatched_key_examples']}")

    for reason, count in report["strict_counts"].items():
        print(f"[SYNTH-V6-CHECK] strict_reason_{reason}={count}")

    for reason, examples in report["strict_examples"].items():
        if not examples:
            continue
        print(f"[SYNTH-V6-CHECK] {reason}_examples_top5:")
        for key, terms, source, synth in examples[:5]:
            print(f"  - {key} terms={terms}")
            print(f"    source: {source}")
            print(f"    synth:  {synth}")

    print_samples(rows, sources, max(0, int(args.sample)), args.seed)

    strict_reasons = OrderedDict()
    strict_reasons["unmatched_keys"] = report["unmatched_keys"]
    strict_reasons["invalid_count"] = report["invalid_count"]
    strict_reasons["duplicate_count"] = report["duplicate_count"]
    strict_reasons["forbidden_word_count"] = report["forbidden_word_count"]
    strict_reasons["word_length_out_of_range_count"] = report[
        "word_length_out_of_range_count"
    ]
    if bank_type == "metadata":
        strict_reasons.update(report["strict_counts"])

    severe = any(count > 0 for count in strict_reasons.values())
    print("[SYNTH-V6-CHECK] strict_issue_reasons:")
    for reason, count in strict_reasons.items():
        if count > 0:
            print(f"  - {reason}: {count}")
    if args.strict and severe:
        print("[SYNTH-V6-CHECK] strict=True severe_issues_found=True")
        return 1
    print("[SYNTH-V6-CHECK] strict_severe_issues_found=" + str(bool(severe)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
