#!/usr/bin/env python3
"""Create non-fabricated MRAF experiment summaries from an explicit manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


SUMMARY_FIELDS: Tuple[str, ...] = (
    "method",
    "train_seed",
    "checkpoint",
    "best_epoch",
    "clean_r1",
    "clean_r5",
    "clean_r10",
    "clean_map",
    "clean_minp",
    "mean_abs_delta_r1",
    "mean_degraded_r1",
    "rgb_missing_r1",
    "ir_missing_r1",
    "rgb_noise_010_r1",
    "ir_noise_010_r1",
    "both_blur_k7_r1",
    "parameter_count",
    "inference_time",
    "test_elapsed_time",
    "alpha_statistics_file",
    "git_head",
    "config",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize completed MRAF experiments. Missing values remain null/NA; "
            "the tool never invents metrics."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSON file containing a list or {'runs': [...]}",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="directory for JSON, CSV, and Markdown outputs",
    )
    parser.add_argument(
        "--basename",
        default="mraf_seed1_summary",
        help="output basename without extension",
    )
    return parser


def _load_runs(manifest_path: Path) -> List[Dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    runs = payload.get("runs") if isinstance(payload, dict) else payload
    if not isinstance(runs, list):
        raise ValueError("Manifest must be a list or an object with a 'runs' list.")
    normalized: List[Dict[str, Any]] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"Manifest run {index} must be a JSON object.")
        if not str(run.get("method", "")).strip():
            raise ValueError(f"Manifest run {index} is missing 'method'.")
        if "train_seed" not in run:
            raise ValueError(f"Manifest run {index} is missing 'train_seed'.")
        normalized.append(
            {field: run.get(field, None) for field in SUMMARY_FIELDS}
        )
    return normalized


def _csv_value(field: str, value: Any) -> Any:
    if value is None:
        return ""
    if field == "config":
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _markdown_value(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, ensure_ascii=False, sort_keys=True)}`"
    return str(value).replace("|", "\\|")


def write_summary(
    runs: Sequence[Dict[str, Any]],
    output_dir: Path,
    basename: str = "mraf_seed1_summary",
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{basename}.json"
    csv_path = output_dir / f"{basename}.csv"
    markdown_path = output_dir / f"{basename}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"runs": list(runs)},
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    field: _csv_value(field, run.get(field))
                    for field in SUMMARY_FIELDS
                }
            )

    table_fields = (
        "method",
        "train_seed",
        "best_epoch",
        "clean_r1",
        "clean_map",
        "mean_abs_delta_r1",
        "mean_degraded_r1",
        "rgb_missing_r1",
        "ir_missing_r1",
        "rgb_noise_010_r1",
        "ir_noise_010_r1",
        "both_blur_k7_r1",
        "parameter_count",
        "inference_time",
    )
    lines = [
        "# MRAF seed1 experiment summary",
        "",
        "尚未提供的实验值以 `NA` 表示；`NA` 不代表 0。",
        "",
        "| " + " | ".join(table_fields) + " |",
        "|" + "|".join("---" for _ in table_fields) + "|",
    ]
    for run in runs:
        lines.append(
            "| "
            + " | ".join(_markdown_value(run.get(field)) for field in table_fields)
            + " |"
        )
    lines.extend(["", "## Provenance", ""])
    for run in runs:
        lines.extend(
            [
                f"- method: {_markdown_value(run.get('method'))}",
                f"  - checkpoint: {_markdown_value(run.get('checkpoint'))}",
                "  - alpha_statistics_file: "
                f"{_markdown_value(run.get('alpha_statistics_file'))}",
                f"  - git_head: {_markdown_value(run.get('git_head'))}",
                f"  - config: {_markdown_value(run.get('config'))}",
            ]
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    runs = _load_runs(manifest_path)
    paths = write_summary(
        runs,
        Path(args.output_dir).resolve(),
        basename=str(args.basename),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
