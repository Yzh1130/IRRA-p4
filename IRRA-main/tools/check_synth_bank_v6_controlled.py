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

BAG_TERMS = {
    "bag",
    "backpack",
    "handbag",
    "purse",
    "satchel",
    "shoulder bag",
    "crossbody bag",
    "schoolbag",
    "briefcase",
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
                            "invalid": "",
                        }
                    )
                else:
                    rows.append(
                        {
                            "key": str(key),
                            "caption": "",
                            "source_caption": "",
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


def analyze(rows, sources):
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

    color_hallucination = 0
    bag_hallucination = 0
    color_examples = []
    bag_examples = []

    for row in valid_rows:
        source_caption = row.get("source_caption") or _select_default_source(
            sources.get(row["key"], {})
        )
        if not source_caption:
            continue
        new_colors = _color_set(row["caption"]) - _color_set(source_caption)
        if new_colors:
            color_hallucination += 1
            if len(color_examples) < 5:
                color_examples.append((row["key"], sorted(new_colors), source_caption, row["caption"]))
        new_bags = _bag_set(row["caption"]) - _bag_set(source_caption)
        if new_bags:
            bag_hallucination += 1
            if len(bag_examples) < 5:
                bag_examples.append((row["key"], sorted(new_bags), source_caption, row["caption"]))

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
        "potential_new_color_hallucination_count": color_hallucination,
        "potential_new_bag_hallucination_count": bag_hallucination,
        "invalid_count": invalid_count,
        "color_examples": color_examples,
        "bag_examples": bag_examples,
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
    report = analyze(rows, sources)

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
        "[SYNTH-V6-CHECK] potential_new_color_hallucination_count="
        f"{report['potential_new_color_hallucination_count']}"
    )
    print(
        "[SYNTH-V6-CHECK] potential_new_bag_hallucination_count="
        f"{report['potential_new_bag_hallucination_count']}"
    )
    print(f"[SYNTH-V6-CHECK] invalid_count={report['invalid_count']}")
    print(f"[SYNTH-V6-CHECK] unmatched_key_examples={report['unmatched_key_examples']}")

    if report["color_examples"]:
        print("[SYNTH-V6-CHECK] color_hallucination_examples_top5:")
        for key, new_terms, source, synth in report["color_examples"]:
            print(f"  - {key} new={new_terms}")
            print(f"    source: {source}")
            print(f"    synth:  {synth}")
    if report["bag_examples"]:
        print("[SYNTH-V6-CHECK] bag_hallucination_examples_top5:")
        for key, new_terms, source, synth in report["bag_examples"]:
            print(f"  - {key} new={new_terms}")
            print(f"    source: {source}")
            print(f"    synth:  {synth}")

    print_samples(rows, sources, max(0, int(args.sample)), args.seed)

    severe = (
        report["unmatched_keys"] > 0
        or report["invalid_count"] > 0
        or report["duplicate_count"] > 0
        or report["forbidden_word_count"] > 0
        or report["potential_new_color_hallucination_count"] > 0
        or report["potential_new_bag_hallucination_count"] > 0
    )
    if args.strict and severe:
        print("[SYNTH-V6-CHECK] strict=True severe_issues_found=True")
        return 1
    print("[SYNTH-V6-CHECK] strict_severe_issues_found=" + str(bool(severe)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
