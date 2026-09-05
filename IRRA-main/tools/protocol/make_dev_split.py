"""Create a deterministic identity-level development split for RGBT-PEDES."""

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
    "records": 4723,
    "identities": 1822,
    "train_identities": 1266,
    "train_records": 3250,
    "train_captions": 5497,
    "test_identities": 556,
    "test_records": 1473,
    "test_captions": 2490,
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


def _caption_count(rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    for index, row in enumerate(rows):
        captions = row.get("captions")
        if not isinstance(captions, list) or not all(
            isinstance(caption, str) for caption in captions
        ):
            raise SplitValidationError(
                f"annotation row {index} has a non-list or non-string captions field"
            )
        count += len(captions)
    return count


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
        {str(row.get("split")) for row in rows}
        - {"train", "test"}
    )
    if unexpected_splits:
        raise SplitValidationError(
            "official annotation contains unexpected split values: "
            + ", ".join(unexpected_splits)
        )

    train_ids = {int(row["id"]) for row in train_rows}
    test_ids = {int(row["id"]) for row in test_rows}
    overlap = train_ids & test_ids
    if overlap:
        raise SplitValidationError(
            "train/test identity sets overlap; refusing to touch test identities "
            f"(first ids: {sorted(overlap)[:5]})"
        )

    actual = {
        "records": len(rows),
        "identities": len(train_ids | test_ids),
        "train_identities": len(train_ids),
        "train_records": len(train_rows),
        "train_captions": _caption_count(train_rows),
        "test_identities": len(test_ids),
        "test_records": len(test_rows),
        "test_captions": _caption_count(test_rows),
    }
    if actual != OFFICIAL_BASELINE:
        differences = [
            f"{key}: expected {OFFICIAL_BASELINE[key]}, got {actual[key]}"
            for key in OFFICIAL_BASELINE
            if actual[key] != OFFICIAL_BASELINE[key]
        ]
        raise SplitValidationError(
            "annotation does not match the confirmed official RGBT-PEDES baseline: "
            + "; ".join(differences)
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
    return {
        "records": len(selected_rows),
        "captions": _caption_count(selected_rows),
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
    if (dev_set | reduced_set) & test_ids:
        raise SplitValidationError("generated split touches official test identities")

    dev_stats = _partition_stats(train_rows, dev_set)
    reduced_stats = _partition_stats(train_rows, reduced_set)
    if dev_stats["records"] + reduced_stats["records"] != len(train_rows):
        raise SplitValidationError("record conservation check failed")
    if dev_stats["captions"] + reduced_stats["captions"] != _caption_count(train_rows):
        raise SplitValidationError("caption conservation check failed")

    return {
        "anno_md5": anno_md5,
        "dev_captions": dev_stats["captions"],
        "dev_ids": dev_ids,
        "dev_records": dev_stats["records"],
        "generated_at": generated_at,
        "n_dev": len(dev_ids),
        "n_train_reduced": len(train_reduced_ids),
        "seed": int(seed),
        "train_reduced_captions": reduced_stats["captions"],
        "train_reduced_ids": train_reduced_ids,
        "train_reduced_records": reduced_stats["records"],
    }


def _write_lf_utf8(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data.decode("utf-8"))


def make_dev_split(anno_path: Path, out_path: Path, n_dev: int, seed: int) -> dict[str, Any]:
    train_rows, test_rows = _load_and_validate_annotation(anno_path)
    anno_md5 = _file_md5(anno_path)
    generated_at = _existing_generated_at(out_path, anno_md5, n_dev, seed)
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )

    # Build the complete artifact twice. Identical MD5 values are the in-process
    # reproducibility gate for the identity selection and JSON serialization.
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

    print("RGBT-PEDES DEV SPLIT SUMMARY")
    print(f"  annotation: {anno_path}")
    print(f"  annotation_md5: {anno_md5}")
    print(
        "  dev: "
        f"identities={first['n_dev']} records={first['dev_records']} "
        f"captions={first['dev_captions']}"
    )
    print(
        "  train_reduced: "
        f"identities={first['n_train_reduced']} "
        f"records={first['train_reduced_records']} "
        f"captions={first['train_reduced_captions']}"
    )
    print(f"  seed: {seed}")
    print(f"  reproducibility_md5_run_1: {first_md5}")
    print(f"  reproducibility_md5_run_2: {second_md5}")
    print(f"  output: {out_path}")
    print("END RGBT-PEDES DEV SPLIT SUMMARY")
    return first


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic identity-level development split from the "
            "official RGBT-PEDES Caption.json annotation."
        )
    )
    parser.add_argument("--anno", type=Path, required=True, help="Path to Caption.json")
    parser.add_argument(
        "--n-dev", type=int, default=190, help="Number of train identities for dev"
    )
    parser.add_argument(
        "--seed", type=int, default=20260905, help="NumPy PCG64 selection seed"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        make_dev_split(
            anno_path=args.anno.expanduser().resolve(),
            out_path=args.out.expanduser().resolve(),
            n_dev=args.n_dev,
            seed=args.seed,
        )
    except SplitValidationError as exc:
        parser.exit(2, f"make_dev_split.py: error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
