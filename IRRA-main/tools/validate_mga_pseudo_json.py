import argparse
import json
import os
import random
import sys
from collections import Counter


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.mga_pseudo import (  # noqa: E402
    TEXT_LIST_FIELDS,
    _merge_record,
    normalize_mga_pseudo_record,
)


def _empty_weight_stats():
    return {
        "count": 0,
        "mean": 0.0,
        "min": 0.0,
        "max": 0.0,
    }


def _weight_stats(weights):
    if not weights:
        return _empty_weight_stats()
    return {
        "count": len(weights),
        "mean": sum(weights) / float(len(weights)),
        "min": min(weights),
        "max": max(weights),
    }


def _compact_example(record):
    example = {
        "key": record.get("key", ""),
        "pid": record.get("pid", None),
        "image_id": record.get("image_id", None),
        "rgb_path": record.get("rgb_path", None),
        "caption": record.get("caption", None),
    }
    for field_name in TEXT_LIST_FIELDS:
        items = record.get(field_name, [])
        example[field_name] = items[:2]
    return example


def validate_jsonl(path, min_weight=0.0):
    if not os.path.isfile(path):
        raise RuntimeError(f"MGA pseudo JSONL file does not exist: {path}")

    total_lines = 0
    valid_lines = 0
    invalid_json_lines = 0
    invalid_schema_lines = 0
    counts = {field_name: 0 for field_name in TEXT_LIST_FIELDS}
    kind_counter = Counter()
    weights = []
    records_by_key = {}
    valid_line_records = []

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped == "":
                continue
            total_lines += 1
            try:
                raw_record = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_json_lines += 1
                continue

            normalized = normalize_mga_pseudo_record(
                raw_record,
                min_weight=min_weight,
            )
            if normalized is None:
                invalid_schema_lines += 1
                continue

            key, record = normalized
            valid_lines += 1
            valid_line_records.append(record)
            if key in records_by_key:
                _merge_record(records_by_key[key], record)
            else:
                records_by_key[key] = record

            for field_name in TEXT_LIST_FIELDS:
                items = record.get(field_name, [])
                counts[field_name] += len(items)
                for item in items:
                    weights.append(float(item.get("weight", 1.0)))
                    kind_counter[str(item.get("kind", "unknown"))] += 1

    return {
        "total_lines": total_lines,
        "valid_lines": valid_lines,
        "invalid_json_lines": invalid_json_lines,
        "invalid_schema_lines": invalid_schema_lines,
        "merged_record_count": len(records_by_key),
        "counts": counts,
        "kind_distribution": dict(kind_counter),
        "weight_stats": _weight_stats(weights),
        "examples": valid_line_records,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate a structured MGA pseudo-caption JSONL file."
    )
    parser.add_argument("--json", required=True, help="path to MGA pseudo JSONL")
    parser.add_argument(
        "--show_examples",
        type=int,
        default=0,
        help="number of random normalized samples to print",
    )
    parser.add_argument(
        "--min_weight",
        type=float,
        default=0.0,
        help="minimum pseudo-text weight kept during validation",
    )
    args = parser.parse_args()

    stats = validate_jsonl(args.json, min_weight=args.min_weight)
    weights = stats["weight_stats"]

    print(f"json={args.json}")
    print(f"total_lines={stats['total_lines']}")
    print(f"valid_lines={stats['valid_lines']}")
    print(f"invalid_json_lines={stats['invalid_json_lines']}")
    print(f"invalid_schema_lines={stats['invalid_schema_lines']}")
    print(f"merged_record_count={stats['merged_record_count']}")
    print(f"pseudo_texts={stats['counts']['pseudo_texts']}")
    print(f"rgb_color_texts={stats['counts']['rgb_color_texts']}")
    print(f"thermal_shape_texts={stats['counts']['thermal_shape_texts']}")
    print(f"negative_texts={stats['counts']['negative_texts']}")
    print(
        "weight count={count} mean={mean:.6f} min={min:.6f} max={max:.6f}".format(
            **weights
        )
    )
    print(
        "kind_distribution={}".format(
            json.dumps(
                stats["kind_distribution"],
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    )

    show_examples = max(0, int(args.show_examples))
    if show_examples > 0 and stats["examples"]:
        rng = random.Random(1)
        examples = list(stats["examples"])
        rng.shuffle(examples)
        examples = examples[:show_examples]
        print("examples:")
        for idx, record in enumerate(examples, start=1):
            print(
                json.dumps(
                    {"index": idx, "record": _compact_example(record)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
