"""Evaluate color-word and no-color query groups on RGBT-PEDES scores."""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.rgbtpedes import RGBTPEDES  # noqa: E402
from tools.revision.score_io import (  # noqa: E402
    METRIC_KEYS,
    compute_retrieval_metrics,
    load_score_file,
    prepare_output_parents,
    report_paths,
    require_outputs_absent,
    write_json,
    write_markdown,
    write_tsv,
)


DEFAULT_TERMS_FILE = Path(__file__).with_name("color_terms_en.txt")
EXPECTED_QUERY_COUNT = 2490
EXPECTED_IMAGE_COUNT = 1473


def load_color_terms(path=DEFAULT_TERMS_FILE):
    with open(path, "r", encoding="utf-8") as handle:
        terms = [line.strip().lower() for line in handle if line.strip()]
    if not terms:
        raise ValueError(f"Color-term file is empty: {path}")
    if len(terms) != len(set(terms)):
        raise ValueError(f"Color-term file contains duplicates: {path}")
    return terms


def compile_color_patterns(terms):
    return {
        term: re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
            flags=re.IGNORECASE,
        )
        for term in terms
    }


def match_color_terms(caption, patterns):
    return {
        term: len(pattern.findall(str(caption)))
        for term, pattern in patterns.items()
        if pattern.search(str(caption))
    }


def group_captions(captions, terms):
    patterns = compile_color_patterns(terms)
    color_indices = []
    no_color_indices = []
    frequencies = Counter()
    matched_terms_by_query = {}
    for index, caption in enumerate(captions):
        matches = match_color_terms(caption, patterns)
        if matches:
            color_indices.append(index)
            frequencies.update(matches)
            matched_terms_by_query[index] = sorted(matches)
        else:
            no_color_indices.append(index)
    return {
        "color_indices": color_indices,
        "no_color_indices": no_color_indices,
        "term_frequencies": dict(sorted(frequencies.items())),
        "matched_terms_by_query": matched_terms_by_query,
    }


def restore_test_order(score_item, root_dir):
    dataset = RGBTPEDES(root=root_dir, verbose=False)
    captions = list(dataset.test["captions"])
    caption_pids = torch.tensor(dataset.test["caption_pids"], dtype=torch.long)
    image_pids = torch.tensor(dataset.test["image_pids"], dtype=torch.long)
    txt_indices = score_item["txt_indices"]
    img_indices = score_item["img_indices"]

    if score_item["matrix"].shape[0] != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_QUERY_COUNT} score queries, got "
            f"{score_item['matrix'].shape[0]}."
        )
    if score_item["matrix"].shape[1] != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_IMAGE_COUNT} gallery images, got "
            f"{score_item['matrix'].shape[1]}."
        )
    if len(captions) != EXPECTED_QUERY_COUNT or image_pids.numel() != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            "Parsed RGBT-PEDES test split has unexpected size: "
            f"queries={len(captions)}, images={image_pids.numel()}."
        )
    if txt_indices.numel() != EXPECTED_QUERY_COUNT or len(set(txt_indices.tolist())) != EXPECTED_QUERY_COUNT:
        raise ValueError("txt_indices must contain 2490 unique query indices.")
    if int(txt_indices.min()) < 0 or int(txt_indices.max()) >= len(captions):
        raise IndexError("txt_indices contains an out-of-range RGBT-PEDES query index.")
    if int(img_indices.min()) < 0 or int(img_indices.max()) >= image_pids.numel():
        raise IndexError("img_indices contains an out-of-range RGBT-PEDES image index.")
    if img_indices.numel() != EXPECTED_IMAGE_COUNT or len(set(img_indices.tolist())) != EXPECTED_IMAGE_COUNT:
        raise ValueError("img_indices must contain 1473 unique gallery indices.")

    expected_txt_pids = caption_pids[txt_indices]
    expected_img_pids = image_pids[img_indices]
    if not torch.equal(expected_txt_pids, score_item["txt_pids"]):
        raise ValueError(
            "score txt_pids do not match RGBT-PEDES captions selected by txt_indices."
        )
    if not torch.equal(expected_img_pids, score_item["img_pids"]):
        raise ValueError(
            "score img_pids do not match RGBT-PEDES images selected by img_indices."
        )
    ordered_captions = [captions[index] for index in txt_indices.tolist()]
    return ordered_captions, dataset.anno_path


def evaluate_groups(score_item, captions, terms, example_count=5):
    grouping = group_captions(captions, terms)
    color_indices = grouping["color_indices"]
    no_color_indices = grouping["no_color_indices"]
    if len(color_indices) + len(no_color_indices) != EXPECTED_QUERY_COUNT:
        raise AssertionError("Color/no-color grouping did not cover exactly 2490 queries.")

    groups = {}
    for name, indices in (
        ("color_word", color_indices),
        ("no_color_word", no_color_indices),
    ):
        if not indices:
            raise ValueError(f"Query group {name!r} is empty.")
        index_tensor = torch.tensor(indices, dtype=torch.long)
        metrics = compute_retrieval_metrics(
            score_item["matrix"][index_tensor],
            score_item["txt_pids"][index_tensor],
            score_item["img_pids"],
        )
        examples = []
        for index in indices[: int(example_count)]:
            examples.append(
                {
                    "score_query_position": index,
                    "dataset_txt_index": int(score_item["txt_indices"][index]),
                    "caption": captions[index],
                    "matched_terms": grouping["matched_terms_by_query"].get(index, []),
                }
            )
        groups[name] = {
            "count": len(indices),
            "proportion": len(indices) / EXPECTED_QUERY_COUNT,
            "metrics": metrics,
            "examples": examples,
        }
    return {"groups": groups, "term_frequencies": grouping["term_frequencies"]}


def _markdown(result):
    lines = [
        "# RGBT-PEDES color-word group evaluation",
        "",
        "| Group | Count | Proportion | R1 | R5 | R10 | mAP | mINP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, group in result["groups"].items():
        metric_values = " | ".join(
            f"{group['metrics'][key]:.6f}" for key in METRIC_KEYS
        )
        lines.append(
            f"| {name} | {group['count']} | {group['proportion']:.6f} | "
            f"{metric_values} |"
        )
    lines.extend(["", "## Color-term frequencies", ""])
    lines.extend(
        f"- `{term}`: {count}" for term, count in result["term_frequencies"].items()
    )
    lines.extend(["", "## Example queries", ""])
    for name, group in result["groups"].items():
        lines.append(f"### {name}")
        lines.append("")
        for example in group["examples"]:
            lines.append(
                f"- [{example['dataset_txt_index']}] {example['caption']} "
                f"(terms={example['matched_terms']})"
            )
        lines.append("")
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate RGBT-PEDES queries with and without bounded color terms."
    )
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--matrix-key", choices=["sim", "raw_sim"], default="sim")
    parser.add_argument("--root-dir", default="datasets/TextReID")
    parser.add_argument("--color-terms-file", default=str(DEFAULT_TERMS_FILE))
    parser.add_argument("--example-count", type=int, default=5)
    parser.add_argument("--output-prefix", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.example_count < 0:
        raise ValueError("--example-count must be >= 0.")
    score_item = load_score_file(args.score_file, matrix_key=args.matrix_key)
    captions, annotation_file = restore_test_order(score_item, args.root_dir)
    terms = load_color_terms(args.color_terms_file)
    result = evaluate_groups(
        score_item, captions, terms, example_count=args.example_count
    )
    result.update(
        {
            "score_file": score_item["path"],
            "matrix_key": args.matrix_key,
            "annotation_file": annotation_file,
            "color_terms_file": os.path.abspath(args.color_terms_file),
            "query_count": EXPECTED_QUERY_COUNT,
            "group_count_sum": sum(group["count"] for group in result["groups"].values()),
        }
    )
    paths = report_paths(args.output_prefix)
    require_outputs_absent(paths.values())
    prepare_output_parents(paths.values())
    write_json(paths["json"], result)
    rows = [
        {"group": name, "count": group["count"], "proportion": group["proportion"], **group["metrics"]}
        for name, group in result["groups"].items()
    ]
    write_tsv(
        paths["tsv"],
        rows,
        ["group", "count", "proportion", *METRIC_KEYS],
    )
    write_markdown(paths["markdown"], _markdown(result))
    print(f"Saved color-group evaluation: {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
