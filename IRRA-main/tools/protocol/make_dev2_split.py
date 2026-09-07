"""Create the deterministic RGBNT201-PEDES train2/dev2 identity split.

The locked 141/30/30 train2/dev2/test identity partition follows the existing
split convention used for RGBNT201-PEDES in the literature.  Keeping dev2 and
the untouched official test split at the same size also makes modality-utility
ratios measured on dev2 directly comparable with those later measured on test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


OFFICIAL_BASELINE = {
    "records": 4787,
    "identities": 201,
    "identity_min": 1,
    "identity_max": 201,
    "train_identities": 171,
    "train_records": 3951,
    "train_caption_slots": 3951,
    "train_effective_captions": 3948,
    "train_empty_captions": 3,
    "test_identities": 30,
    "test_records": 836,
    "test_caption_slots": 836,
    "test_effective_captions": 836,
    "test_empty_captions": 0,
}


class SplitValidationError(ValueError):
    """Raised when the annotation or requested split violates the protocol."""


def _file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _caption_stats(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    caption_slots = 0
    effective_captions = 0
    empty_captions = 0
    for index, row in enumerate(rows):
        captions = row.get("captions")
        if not isinstance(captions, list) or not all(
            isinstance(caption, str) for caption in captions
        ):
            raise SplitValidationError(
                f"annotation row {index} has a non-list or non-string captions field"
            )
        caption_slots += len(captions)
        for caption in captions:
            if caption.strip():
                effective_captions += 1
            else:
                empty_captions += 1
    return {
        "caption_slots": caption_slots,
        "effective_captions": effective_captions,
        "empty_captions": empty_captions,
    }


def _load_and_validate_annotation(
    anno_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not anno_path.is_file():
        raise SplitValidationError(f"annotation file does not exist: {anno_path}")

    try:
        with anno_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SplitValidationError(
            f"failed to read UTF-8 JSON annotation {anno_path}: {exc}"
        ) from exc

    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SplitValidationError("annotation root must be a list of objects")

    missing_id = [index for index, row in enumerate(rows) if "id" not in row]
    if missing_id:
        raise SplitValidationError(
            f"annotation rows missing id (first indices: {missing_id[:5]})"
        )

    train_rows = [row for row in rows if row.get("split") == "train"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    unexpected_splits = sorted(
        {str(row.get("split")) for row in rows} - {"train", "test"}
    )
    if unexpected_splits:
        raise SplitValidationError(
            "official annotation contains unexpected split values: "
            + ", ".join(unexpected_splits)
        )

    try:
        train_ids = {int(row["id"]) for row in train_rows}
        test_ids = {int(row["id"]) for row in test_rows}
    except (TypeError, ValueError) as exc:
        raise SplitValidationError(f"annotation contains a non-integer id: {exc}") from exc

    overlap = train_ids & test_ids
    if overlap:
        raise SplitValidationError(
            "train/test identity sets overlap; refusing to touch test identities "
            f"(first ids: {sorted(overlap)[:5]})"
        )

    train_caption_stats = _caption_stats(train_rows)
    test_caption_stats = _caption_stats(test_rows)
    non_single_caption_rows = [
        index
        for index, row in enumerate(rows)
        if len(row["captions"]) != 1
    ]
    if non_single_caption_rows:
        raise SplitValidationError(
            "official annotation must contain exactly one caption slot per "
            f"record (first bad indices: {non_single_caption_rows[:5]})"
        )
    empty_train_ids = {
        int(row["id"])
        for row in train_rows
        if any(not caption.strip() for caption in row["captions"])
    }
    if empty_train_ids != {31, 82, 108}:
        raise SplitValidationError(
            "official train empty-caption identities must be exactly "
            f"[31, 82, 108], got {sorted(empty_train_ids)}"
        )
    all_ids = train_ids | test_ids
    actual = {
        "records": len(rows),
        "identities": len(all_ids),
        "identity_min": min(all_ids) if all_ids else None,
        "identity_max": max(all_ids) if all_ids else None,
        "train_identities": len(train_ids),
        "train_records": len(train_rows),
        "train_caption_slots": train_caption_stats["caption_slots"],
        "train_effective_captions": train_caption_stats["effective_captions"],
        "train_empty_captions": train_caption_stats["empty_captions"],
        "test_identities": len(test_ids),
        "test_records": len(test_rows),
        "test_caption_slots": test_caption_stats["caption_slots"],
        "test_effective_captions": test_caption_stats["effective_captions"],
        "test_empty_captions": test_caption_stats["empty_captions"],
    }
    if actual != OFFICIAL_BASELINE:
        differences = [
            f"{key}: expected {OFFICIAL_BASELINE[key]}, got {actual[key]}"
            for key in OFFICIAL_BASELINE
            if actual[key] != OFFICIAL_BASELINE[key]
        ]
        raise SplitValidationError(
            "annotation does not match the confirmed official RGBNT201-PEDES "
            "baseline: " + "; ".join(differences)
        )

    return train_rows, test_rows


def _select_dev_ids(
    train_ids: Sequence[int], n_dev: int, seed: int
) -> tuple[list[int], list[int]]:
    ordered_ids = np.asarray(sorted(int(pid) for pid in train_ids), dtype=np.int64)
    if n_dev <= 0:
        raise SplitValidationError(f"n_dev must be positive, got {n_dev}")
    if n_dev >= len(ordered_ids):
        raise SplitValidationError(
            f"n_dev must be smaller than the {len(ordered_ids)} train identities, "
            f"got {n_dev}"
        )

    permutation = np.random.default_rng(int(seed)).permutation(ordered_ids)
    dev_ids = sorted(int(pid) for pid in permutation[:n_dev].tolist())
    dev_set = set(dev_ids)
    train_reduced_ids = [
        int(pid) for pid in ordered_ids.tolist() if int(pid) not in dev_set
    ]
    return dev_ids, train_reduced_ids


def _partition_stats(
    train_rows: Sequence[dict[str, Any]], selected_ids: set[int]
) -> dict[str, int]:
    selected_rows = [row for row in train_rows if int(row["id"]) in selected_ids]
    caption_stats = _caption_stats(selected_rows)
    return {
        "records": len(selected_rows),
        "effective_captions": caption_stats["effective_captions"],
    }


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    return text.encode("utf-8")


def _existing_generated_at(
    out_path: Path, anno_md5: str, n_dev: int, seed: int
) -> str | None:
    """Keep a prior timestamp when regenerating an identical protocol file."""
    if not out_path.is_file():
        return None
    try:
        with out_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        existing.get("anno_md5") == anno_md5
        and existing.get("n_dev") == n_dev
        and existing.get("seed") == seed
        and isinstance(existing.get("generated_at"), str)
    ):
        return existing["generated_at"]
    return None


def _build_payload(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    n_dev: int,
    seed: int,
    anno_md5: str,
    generated_at: str,
) -> dict[str, Any]:
    train_ids = sorted({int(row["id"]) for row in train_rows})
    test_ids = {int(row["id"]) for row in test_rows}
    dev_ids, train_reduced_ids = _select_dev_ids(train_ids, n_dev, seed)

    dev_set = set(dev_ids)
    reduced_set = set(train_reduced_ids)
    if dev_set & reduced_set:
        raise SplitValidationError("dev and train_reduced identity sets overlap")
    if dev_set | reduced_set != set(train_ids):
        raise SplitValidationError("dev and train_reduced do not conserve train identities")
    if dev_set & test_ids:
        raise SplitValidationError(
            "dev_ids overlap official test identities; refusing to touch test"
        )
    if reduced_set & test_ids:
        raise SplitValidationError(
            "train_reduced_ids overlap official test identities; refusing to touch test"
        )

    if len(dev_ids) + len(train_reduced_ids) != OFFICIAL_BASELINE["train_identities"]:
        raise SplitValidationError(
            "identity conservation check failed: n_dev + n_train_reduced must "
            f"equal {OFFICIAL_BASELINE['train_identities']}"
        )

    dev_stats = _partition_stats(train_rows, dev_set)
    reduced_stats = _partition_stats(train_rows, reduced_set)
    if dev_stats["records"] + reduced_stats["records"] != len(train_rows):
        raise SplitValidationError("record conservation check failed")
    effective_train_captions = _caption_stats(train_rows)["effective_captions"]
    if (
        dev_stats["effective_captions"]
        + reduced_stats["effective_captions"]
        != effective_train_captions
    ):
        raise SplitValidationError("effective-caption conservation check failed")

    return {
        "anno_md5": anno_md5,
        "dev_effective_captions": dev_stats["effective_captions"],
        "dev_ids": dev_ids,
        "dev_records": dev_stats["records"],
        "generated_at": generated_at,
        "n_dev": len(dev_ids),
        "n_train_reduced": len(train_reduced_ids),
        "seed": int(seed),
        "train_reduced_effective_captions": reduced_stats["effective_captions"],
        "train_reduced_ids": train_reduced_ids,
        "train_reduced_records": reduced_stats["records"],
    }


def _write_lf_utf8(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data.decode("utf-8"))


def make_dev2_split(
    anno_path: Path, out_path: Path, n_dev: int = 30, seed: int = 20260907
) -> dict[str, Any]:
    """Validate the official annotation, create dev2, and self-check its MD5."""
    train_rows, test_rows = _load_and_validate_annotation(anno_path)
    anno_md5 = _file_md5(anno_path)
    generated_at = _existing_generated_at(out_path, anno_md5, n_dev, seed)
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )

    # Build the complete artifact twice.  Identical MD5 values are the in-process
    # reproducibility gate for both identity selection and JSON serialization.
    first = _build_payload(
        train_rows,
        test_rows,
        n_dev=n_dev,
        seed=seed,
        anno_md5=anno_md5,
        generated_at=generated_at,
    )
    second = _build_payload(
        train_rows,
        test_rows,
        n_dev=n_dev,
        seed=seed,
        anno_md5=anno_md5,
        generated_at=generated_at,
    )
    first_bytes = _canonical_json_bytes(first)
    second_bytes = _canonical_json_bytes(second)
    first_md5 = _bytes_md5(first_bytes)
    second_md5 = _bytes_md5(second_bytes)
    if first_md5 != second_md5:
        raise SplitValidationError(
            "determinism self-check failed: repeated generation produced "
            f"{first_md5} and {second_md5}"
        )

    _write_lf_utf8(out_path, first_bytes)
    if _file_md5(out_path) != first_md5:
        raise SplitValidationError("written artifact MD5 does not match generated bytes")

    print("RGBNT201-PEDES DEV2 SPLIT SUMMARY")
    print(f"  annotation: {anno_path}")
    print(f"  annotation_md5: {anno_md5}")
    print(
        "  dev2: "
        f"identities={first['n_dev']} records={first['dev_records']} "
        f"effective_captions={first['dev_effective_captions']}"
    )
    print(
        "  train2: "
        f"identities={first['n_train_reduced']} "
        f"records={first['train_reduced_records']} "
        "effective_captions="
        f"{first['train_reduced_effective_captions']}"
    )
    print(
        "  official_test_untouched: "
        f"identities={OFFICIAL_BASELINE['test_identities']} "
        f"records={OFFICIAL_BASELINE['test_records']}"
    )
    print(f"  seed: {seed}")
    print(f"  reproducibility_md5_run_1: {first_md5}")
    print(f"  reproducibility_md5_run_2: {second_md5}")
    print(f"  output: {out_path}")
    print("END RGBNT201-PEDES DEV2 SPLIT SUMMARY")
    return first


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the deterministic 141/30 train2/dev2 identity split from "
            "the official RGBNT201-PEDES captionform0_8B.json annotation."
        )
    )
    parser.add_argument(
        "--anno", type=Path, required=True, help="Path to captionform0_8B.json"
    )
    parser.add_argument(
        "--n-dev", type=int, default=30, help="Number of train identities for dev2"
    )
    parser.add_argument(
        "--seed", type=int, default=20260907, help="NumPy PCG64 selection seed"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        make_dev2_split(
            anno_path=args.anno.expanduser().resolve(),
            out_path=args.out.expanduser().resolve(),
            n_dev=args.n_dev,
            seed=args.seed,
        )
    except SplitValidationError as exc:
        parser.exit(2, f"make_dev2_split.py: error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
