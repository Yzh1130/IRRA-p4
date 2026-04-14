import argparse
import json
import os
import os.path as op
import re
import sys
import types
from collections import OrderedDict

PROJECT_ROOT = op.abspath(op.join(op.dirname(__file__), ".."))
TOOL_ROOT = op.abspath(op.dirname(__file__))
for path in (PROJECT_ROOT, TOOL_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

DATASET_IMPORTS = (
    ("cuhkpedes", "CUHKPEDES"),
    ("icfgpedes", "ICFGPEDES"),
    ("rstpreid", "RSTPReid"),
    ("rgbtpedes", "RGBTPEDES"),
)


def _make_missing_dataset_class(class_name):
    class MissingDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{class_name} loader is unavailable. Restore the project datasets "
                "Python package before building the synthetic caption bank."
            )

    MissingDataset.__name__ = class_name
    return MissingDataset


def _install_dataset_import_stubs():
    datasets_pkg = sys.modules.get("datasets")
    if datasets_pkg is None:
        datasets_pkg = types.ModuleType("datasets")
        datasets_pkg.__path__ = []
        sys.modules["datasets"] = datasets_pkg

    for module_suffix, class_name in DATASET_IMPORTS:
        module_name = "datasets." + module_suffix
        if module_name in sys.modules:
            continue
        stub_module = types.ModuleType(module_name)
        setattr(stub_module, class_name, _make_missing_dataset_class(class_name))
        sys.modules[module_name] = stub_module
        setattr(datasets_pkg, module_suffix, stub_module)


try:
    from build_large_synth_bank_v2 import (
        COLOR_WORDS,
        DATASET_FACTORY,
        _choose_canonical_source,
        build_normal_caption,
        build_thermal_caption,
    )
except ModuleNotFoundError as exc:
    if not str(exc.name).startswith("datasets"):
        raise
    sys.modules.pop("build_large_synth_bank_v2", None)
    _install_dataset_import_stubs()
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

COLOR_RE = re.compile(
    r"\b(?:" + "|".join(sorted(COLOR_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
NOISE_RE = re.compile(
    r"\b(?:he|she|his|her|their|young|old|slim|thin|obese|fat|moderate|"
    r"medium|rather|relatively|man|woman|boy|girl|male|female|visible clothing)\b",
    re.IGNORECASE,
)
BAD_STRUCTURE_RE = re.compile(
    r"\bA figure person\b|\bA figure (?:he|she)\b|\bA person of\b|"
    r"\bwearing a dress in (?:his|her|their)\s+(?:left|right)?\s*hand\b|"
    r"\bwith with\b|\bwearing wearing\b|\bcarrying carrying\b|"
    r"\b(?:this|that|it)\s+(?:is|was)\b",
    re.IGNORECASE,
)

CORE_GROUPS = {"upper", "lower"}
DISCRIMINATIVE_GROUPS = {"shoes", "bag", "accessory", "hair"}
WEAK_GROUPS = {"hair", "accessory"}


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


def _collect_train_groups(train_samples, max_train_keys=-1):
    groups = OrderedDict()
    for sample in train_samples:
        image_id = sample[1]
        if image_id not in groups:
            if max_train_keys > 0 and len(groups) >= max_train_keys:
                continue
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


def _has_color(caption):
    return COLOR_RE.search(caption) is not None


def _is_bad_rewrite(normal, thermal, normal_groups, thermal_groups):
    normal = _normalize_spaces(normal)
    thermal = _normalize_spaces(thermal)
    normal_word_count = _word_count(normal)
    thermal_word_count = _word_count(thermal)

    if not normal.startswith("A person "):
        return True, "bad_normal_prefix"
    if not thermal.startswith("A figure "):
        return True, "bad_thermal_prefix"
    if normal_word_count < 8:
        return True, "normal_too_short"
    if thermal_word_count < 7:
        return True, "thermal_too_short"
    if thermal_word_count > 22:
        return True, "thermal_too_long"
    if BAD_STRUCTURE_RE.search(normal) or BAD_STRUCTURE_RE.search(thermal):
        return True, "bad_structure"
    if NOISE_RE.search(normal) or NOISE_RE.search(thermal):
        return True, "noise_word_residual"
    if _has_color(thermal):
        return True, "thermal_color_residual"

    normal_body = _caption_body(normal, "A person")
    thermal_body = _caption_body(thermal, "A figure")
    if (
        _similarity(normal_body, thermal_body) >= 0.92
        and normal_word_count <= 10
        and thermal_word_count <= 10
    ):
        return True, "near_duplicate_short_pair"

    if thermal_groups and thermal_groups <= WEAK_GROUPS:
        return True, "thermal_weak_groups_only"
    if len(normal_groups) - len(thermal_groups) >= 3 and len(thermal_groups) <= 3:
        return True, "thermal_attr_density_gap"
    return False, ""


def _drop_low_info(groups, prefix):
    if len(groups) < 3:
        return True, f"{prefix}_attr_groups_lt_3"
    for group in sorted(CORE_GROUPS):
        if group not in groups:
            return True, f"{prefix}_missing_{group}"
    if not (groups & DISCRIMINATIVE_GROUPS):
        return True, f"{prefix}_missing_discriminative_group"
    if groups <= (CORE_GROUPS | {"action"}):
        return True, f"{prefix}_only_core_action_groups"
    return False, ""


def _score_sample(normal, thermal, normal_groups, thermal_groups):
    score = 10.0 * (len(normal_groups) + len(thermal_groups))

    if {"upper", "lower", "shoes"} <= normal_groups:
        score += 6.0
    if {"upper", "lower", "shoes"} <= thermal_groups:
        score += 6.0
    if "bag" in normal_groups or "bag" in thermal_groups:
        score += 4.0
    if "accessory" in normal_groups or "accessory" in thermal_groups:
        score += 2.0
    if _has_color(normal) and not _has_color(thermal):
        score += 3.0
    if "action" in normal_groups or "action" in thermal_groups:
        score += 1.0

    normal_word_count = _word_count(normal)
    thermal_word_count = _word_count(thermal)
    if normal_word_count < 10:
        score -= 3.0
    if thermal_word_count < 9:
        score -= 3.0
    if len(normal_groups) == 3:
        score -= 2.0
    if len(thermal_groups) == 3:
        score -= 2.0
    if normal_groups <= (CORE_GROUPS | WEAK_GROUPS):
        score -= 2.0
    if thermal_groups <= (CORE_GROUPS | WEAK_GROUPS):
        score -= 3.0

    density_gap = len(normal_groups) - len(thermal_groups)
    if density_gap > 0:
        score -= 2.0 * density_gap
    if thermal_word_count + 3 < normal_word_count:
        score -= 1.5

    return round(score, 3)


def _make_record(key, source, normal, thermal, normal_groups, thermal_groups, reason="", score=None):
    record = {
        "key": key,
        "reason": reason,
        "source": source,
        "normal": normal,
        "thermal": thermal,
        "normal_attr_groups": sorted(normal_groups),
        "thermal_attr_groups": sorted(thermal_groups),
    }
    if score is not None:
        record["score"] = score
    return record


def build_bank(train_samples, max_train_keys=-1, max_keep_keys=-1):
    groups = _collect_train_groups(train_samples, max_train_keys=max_train_keys)
    candidates = []
    dropped_examples = []
    dropped_low_info_keys = 0
    dropped_bad_rewrite_keys = 0

    for group in groups.values():
        key = group["key"]
        source = _choose_canonical_source(group["captions"])
        normal = build_normal_caption(source)
        thermal = build_thermal_caption(source)
        normal_groups = count_attr_groups(normal)
        thermal_groups = count_attr_groups(thermal)

        bad_rewrite, bad_reason = _is_bad_rewrite(
            normal, thermal, normal_groups, thermal_groups
        )
        normal_low_info, normal_low_info_reason = _drop_low_info(normal_groups, "normal")
        thermal_low_info, thermal_low_info_reason = _drop_low_info(
            thermal_groups, "thermal"
        )

        if bad_rewrite:
            dropped_bad_rewrite_keys += 1
            if len(dropped_examples) < 5:
                dropped_examples.append(
                    _make_record(
                        key,
                        source,
                        normal,
                        thermal,
                        normal_groups,
                        thermal_groups,
                        reason=bad_reason,
                    )
                )
            continue

        if normal_low_info or thermal_low_info:
            dropped_low_info_keys += 1
            low_info_reason = normal_low_info_reason or thermal_low_info_reason
            if len(dropped_examples) < 5:
                dropped_examples.append(
                    _make_record(
                        key,
                        source,
                        normal,
                        thermal,
                        normal_groups,
                        thermal_groups,
                        reason=low_info_reason,
                    )
                )
            continue

        score = _score_sample(normal, thermal, normal_groups, thermal_groups)
        candidates.append(
            _make_record(
                key,
                source,
                normal,
                thermal,
                normal_groups,
                thermal_groups,
                score=score,
            )
        )

    candidates.sort(key=lambda item: (-item["score"], item["key"]))
    kept_keys_before_cap = len(candidates)
    if max_keep_keys > 0:
        candidates = candidates[:max_keep_keys]

    bank = OrderedDict()
    kept_normal_group_total = 0
    kept_thermal_group_total = 0
    for record in candidates:
        bank[record["key"]] = [record["normal"], record["thermal"]]
        kept_normal_group_total += len(record["normal_attr_groups"])
        kept_thermal_group_total += len(record["thermal_attr_groups"])

    kept_keys_after_cap = len(bank)
    kept_tail10 = list(reversed(candidates[-10:]))
    stats = {
        "total_train_keys_seen": len(groups),
        "kept_keys_before_cap": kept_keys_before_cap,
        "kept_keys_after_cap": kept_keys_after_cap,
        "keep_cap_applied": max_keep_keys > 0,
        "keep_cap_satisfied": max_keep_keys <= 0 or kept_keys_after_cap == max_keep_keys,
        "dropped_low_info_keys": dropped_low_info_keys,
        "dropped_bad_rewrite_keys": dropped_bad_rewrite_keys,
        "average_normal_attr_groups": (
            kept_normal_group_total / float(kept_keys_after_cap)
            if kept_keys_after_cap
            else 0.0
        ),
        "average_thermal_attr_groups": (
            kept_thermal_group_total / float(kept_keys_after_cap)
            if kept_keys_after_cap
            else 0.0
        ),
        "kept_examples": candidates[:5],
        "kept_score_top10": candidates[:10],
        "kept_score_tail10": kept_tail10,
        "dropped_examples": dropped_examples,
    }
    return bank, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a stricter v4 synthetic caption bank for the train split."
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
        default="./synth_caption_bank_large_v4.json",
        help="output json path for the synthetic caption bank",
    )
    parser.add_argument(
        "--max_train_keys",
        type=int,
        default=-1,
        help="limit train keys for smoke tests; -1 keeps all train keys",
    )
    parser.add_argument(
        "--max_keep_keys",
        type=int,
        default=-1,
        help="keep only top-K scored keys after filtering; -1 keeps all",
    )
    return parser.parse_args()


def _print_example(example):
    print(f"  - {example['key']}")
    print(f"    reason: {example['reason']}")
    print(f"    original canonical source: {example['source']}")
    print(f"    rewritten normal: {example['normal']}")
    print(f"    rewritten thermal: {example['thermal']}")
    if "score" in example:
        print(f"    score: {example['score']:.3f}")
    print(f"    normal_attr_groups: {example['normal_attr_groups']}")
    print(f"    thermal_attr_groups: {example['thermal_attr_groups']}")


def _print_score_row(rank, example):
    print(
        "  - "
        f"rank={rank} key={example['key']} score={example['score']:.3f} "
        f"normal_groups={example['normal_attr_groups']} "
        f"thermal_groups={example['thermal_attr_groups']}"
    )


def main():
    args = parse_args()

    dataset = DATASET_FACTORY[args.dataset_name](root=args.root_dir, verbose=False)
    bank, stats = build_bank(
        dataset.train,
        max_train_keys=args.max_train_keys,
        max_keep_keys=args.max_keep_keys,
    )

    output_file = op.abspath(args.output_file)
    output_dir = op.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(bank, f, indent=4, separators=(",", ": "))

    print(f"[SYNTH-V4] dataset_name={args.dataset_name}")
    print(f"[SYNTH-V4] root_dir={args.root_dir}")
    print(f"[SYNTH-V4] caption_file={dataset.anno_path}")
    print(f"[SYNTH-V4] output_file={output_file}")
    print("[SYNTH-V4] split=train")
    print("[SYNTH-V4] key_rule=basename(img_path)")
    print(f"[SYNTH-V4] max_train_keys={args.max_train_keys}")
    print(f"[SYNTH-V4] max_keep_keys={args.max_keep_keys}")
    print(f"[SYNTH-V4] total_train_keys_seen={stats['total_train_keys_seen']}")
    print(f"[SYNTH-V4] kept_keys_before_cap={stats['kept_keys_before_cap']}")
    print(f"[SYNTH-V4] kept_keys_after_cap={stats['kept_keys_after_cap']}")
    print(f"[SYNTH-V4] keep_cap_applied={stats['keep_cap_applied']}")
    print(f"[SYNTH-V4] keep_cap_satisfied={stats['keep_cap_satisfied']}")
    print(f"[SYNTH-V4] dropped_low_info_keys={stats['dropped_low_info_keys']}")
    print(f"[SYNTH-V4] dropped_bad_rewrite_keys={stats['dropped_bad_rewrite_keys']}")
    print(
        "[SYNTH-V4] average_normal_attr_groups="
        f"{stats['average_normal_attr_groups']:.3f}"
    )
    print(
        "[SYNTH-V4] average_thermal_attr_groups="
        f"{stats['average_thermal_attr_groups']:.3f}"
    )
    print("[SYNTH-V4] first_5_kept_examples:")
    for example in stats["kept_examples"]:
        _print_example(example)
    print("[SYNTH-V4] kept_score_top10:")
    for rank, example in enumerate(stats["kept_score_top10"], start=1):
        _print_score_row(rank, example)
    print("[SYNTH-V4] kept_score_tail10_lowest_first:")
    for rank, example in enumerate(stats["kept_score_tail10"], start=1):
        _print_score_row(rank, example)
    print("[SYNTH-V4] first_5_dropped_examples:")
    for example in stats["dropped_examples"]:
        _print_example(example)

    return 0


if __name__ == "__main__":
    sys.exit(main())
