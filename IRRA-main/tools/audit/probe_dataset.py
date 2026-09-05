"""只读探测 RGB-T 文本行人检索数据、标注及 loader 口径。

本模块只读取数据和源码；唯一输出为 ``--out`` 下的 JSON dump 与
``docs/audit/DATA_AUDIT_<dataset-name>.md``。它不会构造 DataLoader、模型或执行 forward。
"""

from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import hashlib
import importlib
import itertools
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__:
    from .common import (
        AuditFinding,
        get_load_error,
        resolve_audit_output_dir,
        safe_json_load,
        safe_pickle_load,
        write_report,
    )
else:  # 兼容 ``python tools/audit/probe_dataset.py --help``。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.audit.common import (  # type: ignore
        AuditFinding,
        get_load_error,
        resolve_audit_output_dir,
        safe_json_load,
        safe_pickle_load,
        write_report,
    )


MIN_CANDIDATE_BYTES = 10 * 1024
ANNOTATION_SUFFIXES = {".json", ".pkl", ".pickle", ".csv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

IDENTITY_ALIASES = (
    "person_id",
    "personid",
    "identity_id",
    "identity",
    "pid",
    "label",
    "id",
)
SPLIT_ALIASES = ("split", "subset", "partition", "set", "phase")
TEXT_ALIASES = (
    "captions",
    "caption",
    "descriptions",
    "description",
    "sentences",
    "sentence",
    "texts",
    "text",
)
RGB_HINTS = ("rgb", "vis", "visible", "color", "colour")
THERMAL_HINTS = ("thermal", "infrared", "ir", "t")
GENERIC_PATH_HINTS = ("image", "img", "file", "path", "photo")

SUPPORTED_DATASETS = ("RGBT-PEDES", "RGBNT201-PEDES")

# 任务给定的已确证基线。键名与 _baseline_measurements() 的实测口径一一对应。
CONFIRMED_BASELINES: dict[str, tuple[tuple[str, str, Any], ...]] = {
    "RGBT-PEDES": (
        ("total_records", "总记录", 4723),
        ("total_identities", "ID 唯一数", 1822),
        ("train_identities", "train 身份", 1266),
        ("train_records", "train 记录", 3250),
        ("train_captions", "train caption", 5497),
        ("test_identities", "test 身份", 556),
        ("test_records", "test 记录", 1473),
        ("test_captions", "test caption", 2490),
        ("train_test_intersection", "train∩test 身份数", 0),
        ("caption_histogram", "caption 数分布", {1: 1459, 2: 3264}),
    ),
    "RGBNT201-PEDES": (
        ("total_records", "总记录", 4787),
        ("total_identities", "ID 唯一数", 201),
        ("train_identities", "train 身份", 171),
        ("train_records", "train 记录", 3951),
        ("train_captions", "train caption", 3951),
        ("test_identities", "test 身份", 30),
        ("test_records", "test 记录", 836),
        ("test_captions", "test caption", 836),
        ("train_test_intersection", "train∩test 身份数", 0),
        ("caption_histogram", "caption 数分布", {1: 4787}),
        ("empty_captions", "空 caption", 3),
        ("train_empty_captions", "train 空 caption", 3),
        ("test_empty_captions", "test 空 caption", 0),
    ),
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _stable_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _short_value(value: Any, limit: int = 200) -> str:
    rendered = _stable_value(value).replace("\r", "\\r").replace("\n", "\\n")
    return rendered if len(rendered) <= limit else rendered[: limit - 3] + "..."


def _sort_identity(value: Any) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _source_label(path: Path, repo: Path | None = None, line: int | None = None) -> str:
    path = path.resolve(strict=False)
    if repo is not None:
        try:
            label = path.relative_to(repo.resolve(strict=False)).as_posix()
        except ValueError:
            label = str(path)
    else:
        label = str(path)
    return f"{label}:{line}" if line is not None else label


def _load_error_for(path: Path) -> str:
    """兼容 common.get_load_error 的按路径或最近一次错误两种实现。"""
    try:
        value = get_load_error(path)
    except TypeError:
        try:
            value = get_load_error()
        except Exception as exc:  # pragma: no cover - compatibility guard
            return str(exc)
    except Exception as exc:  # pragma: no cover - compatibility guard
        return str(exc)
    if isinstance(value, Mapping):
        return str(value.get(str(path), value.get("reason", value)))
    return "" if value is None else str(value)


def _read_csv(path: Path) -> tuple[list[dict[str, str]] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _load_candidate(path: Path) -> tuple[Any | None, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = safe_json_load(path)
        return value, None if value is not None else (_load_error_for(path) or "JSON 读取失败")
    if suffix in {".pkl", ".pickle"}:
        value = safe_pickle_load(path)
        return value, None if value is not None else (_load_error_for(path) or "pickle 读取失败")
    if suffix == ".csv":
        return _read_csv(path)
    return None, f"不支持的后缀：{suffix}"


def _records_from_object(value: Any) -> tuple[list[Mapping[str, Any]], str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)], "顶层列表"
    if not isinstance(value, Mapping):
        return [], "无法识别记录容器"

    preferred = ("annotations", "annotation", "records", "items", "data", "samples")
    normalized = {_normalize_key(key): key for key in value}
    for name in preferred:
        original = normalized.get(name)
        candidate = value.get(original) if original is not None else None
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, Mapping)], f"顶层键 {original}"

    list_candidates: list[tuple[int, str, list[Mapping[str, Any]]]] = []
    for key, candidate in value.items():
        if isinstance(candidate, list):
            records = [item for item in candidate if isinstance(item, Mapping)]
            if records:
                list_candidates.append((len(records), str(key), records))
    if list_candidates:
        _, key, records = max(list_candidates, key=lambda item: item[0])
        return records, f"最大记录列表键 {key}"

    if value and all(isinstance(item, Mapping) for item in value.values()):
        return list(value.values()), "顶层映射值"
    return [], "无法识别记录容器"


def _all_record_keys(records: Sequence[Mapping[str, Any]], limit: int = 200) -> list[str]:
    keys: set[str] = set()
    for record in records[:limit]:
        keys.update(str(key) for key in record)
    return sorted(keys)


def _pick_alias(keys: Sequence[str], aliases: Sequence[str]) -> str | None:
    by_normalized = {_normalize_key(key): key for key in keys}
    for alias in aliases:
        found = by_normalized.get(_normalize_key(alias))
        if found is not None:
            return found
    return None


def _looks_like_path_key(key: str) -> bool:
    normalized = _normalize_key(key)
    tokens = set(normalized.split("_"))
    return bool(tokens.intersection(GENERIC_PATH_HINTS)) or any(
        hint in normalized for hint in ("filepath", "imagepath", "imgpath")
    )


def _infer_fields(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = _all_record_keys(records)
    identity = _pick_alias(keys, IDENTITY_ALIASES)
    split = _pick_alias(keys, SPLIT_ALIASES)
    text = _pick_alias(keys, TEXT_ALIASES)
    rgb_fields: list[str] = []
    thermal_fields: list[str] = []
    generic_fields: list[str] = []
    for key in keys:
        if not _looks_like_path_key(key):
            continue
        normalized = _normalize_key(key)
        tokens = set(normalized.split("_"))
        if any(hint in tokens or hint in normalized for hint in RGB_HINTS):
            rgb_fields.append(key)
        elif any(hint in tokens for hint in THERMAL_HINTS) or "thermal" in normalized:
            thermal_fields.append(key)
        else:
            generic_fields.append(key)
    return {
        "identity": identity,
        "split": split,
        "text": text,
        "rgb_paths": rgb_fields,
        "thermal_paths": thermal_fields,
        "generic_paths": generic_fields,
        "all_keys": keys,
    }


def _sample_structure(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index in (0, 1):
        if index >= len(records):
            continue
        record = records[index]
        samples.append(
            {
                "index": index,
                "keys": [str(key) for key in record.keys()],
                "fields": {
                    str(key): {
                        "type": type(value).__name__,
                        "sample": _short_value(value),
                    }
                    for key, value in record.items()
                },
            }
        )
    return samples


def _probe_candidate(path: Path, repo: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError:
        size = None
    value, error = _load_candidate(path)
    records, container = _records_from_object(value)
    top_keys = [str(key) for key in value.keys()] if isinstance(value, Mapping) else []
    top_length = len(value) if isinstance(value, (list, Mapping)) else None
    mapping = _infer_fields(records)
    probe = {
        "path": str(path.resolve(strict=False)),
        "source": _source_label(path, repo),
        "size_bytes": size,
        "status": "OK" if value is not None else "UNKNOWN",
        "error": error,
        "top_type": type(value).__name__ if value is not None else "无法判定",
        "top_length": top_length,
        "top_keys": top_keys,
        "record_container": container,
        "record_count": len(records),
        "field_mapping": mapping,
        "samples": _sample_structure(records),
        "_records": records,
    }
    printable = {key: val for key, val in probe.items() if key != "_records"}
    print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
    return probe


def _walk_identity(path: Path) -> tuple[Any, ...]:
    """Return a stable directory identity while following symlinks."""
    try:
        stat_result = path.stat()
        if int(getattr(stat_result, "st_ino", 0)) != 0:
            return ("inode", int(stat_result.st_dev), int(stat_result.st_ino))
    except (OSError, RuntimeError):
        pass
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = Path(os.path.abspath(str(path)))
    return ("path", os.path.normcase(str(resolved)))


def _is_walk_excluded(path: Path, exclusions: Sequence[str]) -> bool:
    candidates = [path]
    try:
        candidates.append(path.resolve(strict=False))
    except (OSError, RuntimeError):
        pass
    normalized = [candidate.as_posix().casefold() for candidate in candidates]
    return any(
        token and any(token in candidate for candidate in normalized)
        for token in (
            Path(str(exclusion)).as_posix().casefold()
            for exclusion in exclusions
        )
    )


def _find_files_followlinks(
    root: Path,
    patterns: Sequence[str],
    exclude_substrings: Sequence[str],
    max_depth: int,
) -> list[Path]:
    """Find files below ``root``, following links without revisiting targets."""
    base = root.expanduser()
    if not base.is_dir():
        return []

    found: list[Path] = []
    visited: set[tuple[Any, ...]] = set()
    pending: list[tuple[Path, int]] = [(base, 0)]
    while pending:
        directory, depth = pending.pop()
        identity = _walk_identity(directory)
        if identity in visited:
            continue
        visited.add(identity)
        if _is_walk_excluded(directory, exclude_substrings):
            continue
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(
                    iterator,
                    key=lambda entry: entry.name.casefold(),
                    reverse=True,
                )
        except (OSError, RuntimeError):
            continue
        for entry in entries:
            path = Path(entry.path)
            if _is_walk_excluded(path, exclude_substrings):
                continue
            try:
                if entry.is_dir(follow_symlinks=True):
                    if depth < max_depth:
                        pending.append((path, depth + 1))
                    continue
                if not entry.is_file(follow_symlinks=True):
                    continue
            except OSError:
                continue
            try:
                relative = path.relative_to(base).as_posix()
            except ValueError:
                relative = path.as_posix()
            if any(
                fnmatch.fnmatchcase(entry.name, pattern)
                or fnmatch.fnmatchcase(relative, Path(pattern).as_posix())
                for pattern in patterns
            ):
                found.append(path)
    return sorted(found, key=lambda item: item.as_posix().casefold())


def _find_candidates(data_root: Path, explicit: Path | None) -> tuple[list[Path], list[str]]:
    notes: list[str] = []
    if explicit is not None:
        if explicit.is_file():
            return [explicit.resolve(strict=False)], notes
        notes.append(f"显式标注未找到：{explicit.resolve(strict=False)}")
        return [], notes
    if not data_root.exists():
        notes.append(f"数据根未找到：{data_root.resolve(strict=False)}")
        return [], notes
    found = _find_files_followlinks(
        data_root,
        patterns=("*.json", "*.pkl", "*.pickle", "*.csv"),
        exclude_substrings=("site-packages", "__pycache__", ".git/objects", ".cache"),
        max_depth=12,
    )
    candidates: list[Path] = []
    for item in found or []:
        path = Path(item)
        try:
            if path.is_file() and path.suffix.lower() in ANNOTATION_SUFFIXES and path.stat().st_size > MIN_CANDIDATE_BYTES:
                candidates.append(path.resolve(strict=False))
        except OSError:
            continue
    return sorted(set(candidates), key=lambda path: str(path).lower()), notes


def _find_repo(start: Path) -> Path:
    start = start.resolve(strict=False)
    for candidate in (start, *start.parents):
        if (candidate / "datasets").is_dir() and (candidate / "model").is_dir():
            return candidate
        nested = candidate / "IRRA-main"
        if (nested / "datasets").is_dir() and (nested / "model").is_dir():
            return nested
    return Path(__file__).resolve().parents[2]


def _loader_resolution(repo: Path, data_root: Path, dataset_name: str) -> dict[str, Any]:
    dataset_dir = (
        data_root
        if data_root.name.casefold() == dataset_name.casefold()
        else data_root / dataset_name
    )
    if dataset_name == "RGBT-PEDES":
        data_candidates = (
            dataset_dir / "RGBT-PEDES-Final",
            dataset_dir / "RGBT-PEDES",
            dataset_dir,
        )
        annotation_names = ("Caption.json", "caption.json")
    else:
        data_candidates = (dataset_dir,)
        annotation_names = ("captionform0_8B.json",)
    selected_data: Path | None = None
    for candidate in data_candidates:
        if (candidate / "RGB").is_dir() and (candidate / "T").is_dir():
            selected_data = candidate
            break
    selected_anno: Path | None = None
    if selected_data is not None:
        for name in annotation_names:
            candidate = selected_data / name
            if candidate.exists():
                selected_anno = candidate
                break
    return {
        "source": _source_label(repo / "datasets" / "rgbtpedes.py", repo),
        "dataset_name": dataset_name,
        "root_argument": str(data_root.resolve(strict=False)),
        "dataset_dir": str(dataset_dir.resolve(strict=False)),
        "data_candidates_in_order": [str(path.resolve(strict=False)) for path in data_candidates],
        "selected_data_dir": str(selected_data.resolve(strict=False)) if selected_data else None,
        "annotation_names_in_order": list(annotation_names),
        "selected_annotation": str(selected_anno.resolve(strict=False)) if selected_anno else None,
        "status": "OK" if selected_anno else "UNKNOWN",
    }


def _candidate_score(probe: Mapping[str, Any], loader_anno: str | None) -> tuple[int, int]:
    mapping = probe["field_mapping"]
    score = 0
    if loader_anno and Path(probe["path"]) == Path(loader_anno):
        score += 100
    if mapping.get("identity"):
        score += 10
    if mapping.get("split"):
        score += 10
    if mapping.get("text"):
        score += 10
    if mapping.get("rgb_paths") or mapping.get("thermal_paths") or mapping.get("generic_paths"):
        score += 10
    if "caption" in Path(probe["path"]).name.lower():
        score += 5
    return score, int(probe.get("record_count", 0))


def _select_primary(
    probes: Sequence[dict[str, Any]], explicit: Path | None, loader: Mapping[str, Any]
) -> dict[str, Any] | None:
    if explicit is not None:
        explicit_resolved = explicit.resolve(strict=False)
        for probe in probes:
            if Path(probe["path"]) == explicit_resolved:
                return probe
    usable = [probe for probe in probes if probe.get("status") == "OK"]
    if not usable:
        return None
    return max(usable, key=lambda probe: _candidate_score(probe, loader.get("selected_annotation")))


def _record_texts(record: Mapping[str, Any], text_field: str | None) -> list[str]:
    if text_field is None or text_field not in record:
        return []
    value = record.get(text_field)
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        texts: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                texts.append(item)
            elif isinstance(item, Mapping):
                candidate = item.get("text", item.get("caption", item.get("description")))
                if isinstance(candidate, str) and candidate.strip():
                    texts.append(candidate)
        return texts
    return []


def _record_caption_entries(record: Mapping[str, Any], text_field: str | None) -> list[Any]:
    """按 loader 的遍历口径返回 caption 条目，不过滤空字符串。"""

    if text_field is None or text_field not in record:
        return []
    value = record.get(text_field)
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [value]
    return []


def _record_split(record: Mapping[str, Any], split_field: str | None) -> str:
    if split_field is None:
        return "未标注"
    value = str(record.get(split_field, "未标注")).strip().lower()
    return value or "未标注"


def _record_identity(record: Mapping[str, Any], identity_field: str | None) -> Any | None:
    if identity_field is None:
        return None
    value = record.get(identity_field)
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return value


def _discover_modality_pairs(roots: Iterable[Path], max_depth: int = 7) -> list[dict[str, Path]]:
    rgb_names = {"rgb", "vis", "visible", "color", "colour"}
    thermal_names = {"t", "ir", "thermal", "infrared"}
    queue: deque[tuple[Path, int]] = deque()
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve(strict=False)
        if root.exists() and root.is_dir() and root not in seen:
            queue.append((root, 0))
            seen.add(root)
    pairs: list[dict[str, Path]] = []
    while queue:
        directory, depth = queue.popleft()
        try:
            children = [
                item
                for item in directory.iterdir()
                if item.is_dir() and not item.is_symlink()
            ]
        except OSError:
            continue
        by_name = {item.name.lower(): item for item in children}
        rgb = next((by_name[name] for name in rgb_names if name in by_name), None)
        thermal = next((by_name[name] for name in thermal_names if name in by_name), None)
        if rgb is not None or thermal is not None:
            pairs.append({"parent": directory, "rgb": rgb, "thermal": thermal})
        if depth >= max_depth:
            continue
        for child in children:
            if child.name.lower() in rgb_names | thermal_names:
                continue
            resolved = child.resolve(strict=False)
            if resolved not in seen:
                seen.add(resolved)
                queue.append((child, depth + 1))
    return pairs


def _choose_modality_pair(
    pairs: Sequence[Mapping[str, Path | None]], annotation: Path
) -> Mapping[str, Path | None] | None:
    if not pairs:
        return None
    annotation_parent = annotation.parent.resolve(strict=False)

    def score(pair: Mapping[str, Path | None]) -> tuple[int, int]:
        parent = Path(pair["parent"]).resolve(strict=False)
        exact = int(parent == annotation_parent)
        try:
            distance = len(annotation_parent.relative_to(parent).parts)
        except ValueError:
            try:
                distance = len(parent.relative_to(annotation_parent).parts)
            except ValueError:
                distance = 999
        complete = int(pair.get("rgb") is not None and pair.get("thermal") is not None)
        return exact * 100 + complete * 10 - distance, -distance

    return max(pairs, key=score)


def _directory_image_names(directory: Path | None) -> set[str] | None:
    if directory is None or not directory.is_dir():
        return None
    names: set[str] = set()
    try:
        for item in directory.iterdir():
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
                names.add(item.name)
    except OSError:
        return None
    return names


def _first_record_path(record: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _annotation_stats(
    primary: Mapping[str, Any], data_root: Path, repo: Path
) -> dict[str, Any]:
    records: Sequence[Mapping[str, Any]] = primary["_records"]
    mapping: Mapping[str, Any] = primary["field_mapping"]
    identity_field = mapping.get("identity")
    split_field = mapping.get("split")
    text_field = mapping.get("text")
    annotation = Path(primary["path"])

    pairs = _discover_modality_pairs((annotation.parent, data_root))
    chosen_pair = _choose_modality_pair(pairs, annotation)
    rgb_dir = Path(chosen_pair["rgb"]) if chosen_pair and chosen_pair.get("rgb") else None
    thermal_dir = (
        Path(chosen_pair["thermal"]) if chosen_pair and chosen_pair.get("thermal") else None
    )
    rgb_names = _directory_image_names(rgb_dir)
    thermal_names = _directory_image_names(thermal_dir)
    modality_inventory_known = rgb_names is not None and thermal_names is not None

    per_split: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "records": 0,
            "identities": set(),
            "captions": 0,
            "empty_captions": 0,
            "caption_histogram": Counter(),
            "query_identities": set(),
            "gallery_identities": set(),
            "records_with_both_derived_paths": 0,
            "records_with_both_files": 0,
        }
    )
    per_identity: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rgb": 0, "thermal": 0, "descriptions": 0, "records": 0, "splits": set()}
    )
    anomalies: dict[str, list[dict[str, Any]]] = {
        "missing_identity": [],
        "missing_split": [],
        "missing_path_annotation": [],
        "only_rgb_annotation": [],
        "only_thermal_annotation": [],
        "rgb_missing_on_disk": [],
        "thermal_missing_on_disk": [],
        "no_description": [],
    }
    annotated_names: set[str] = set()

    for index, record in enumerate(records):
        split = _record_split(record, split_field)
        pid = _record_identity(record, identity_field)
        caption_entries = _record_caption_entries(record, text_field)
        texts = _record_texts(record, text_field)
        rgb_value = _first_record_path(record, mapping.get("rgb_paths", ()))
        thermal_value = _first_record_path(record, mapping.get("thermal_paths", ()))
        generic_value = _first_record_path(record, mapping.get("generic_paths", ()))
        basename_source = rgb_value or thermal_value or generic_value
        basename = Path(basename_source).name if basename_source else None
        if basename:
            annotated_names.add(basename)

        bucket = per_split[split]
        bucket["records"] += 1
        bucket["captions"] += len(caption_entries)
        bucket["empty_captions"] += sum(
            1 for caption in caption_entries
            if isinstance(caption, str) and not caption.strip()
        )
        bucket["caption_histogram"][len(caption_entries)] += 1
        if pid is not None:
            pid_key = str(pid)
            bucket["identities"].add(pid_key)
            bucket["gallery_identities"].add(pid_key)
            if caption_entries:
                bucket["query_identities"].add(pid_key)
            ident = per_identity[pid_key]
            ident["records"] += 1
            ident["descriptions"] += len(caption_entries)
            ident["splits"].add(split)
        else:
            anomalies["missing_identity"].append({"index": index})
        if split == "未标注":
            anomalies["missing_split"].append({"index": index, "identity": pid})
        if not texts:
            anomalies["no_description"].append({"index": index, "identity": pid, "split": split})

        # RGBT-PEDES 的 `_pair_paths` 会把一个 generic file_path 的 basename
        # 同时映射到 RGB/ 与 T/，因此 generic 路径在 loader 语义上具备两侧派生路径。
        has_rgb_annotation = bool(rgb_value or generic_value)
        has_thermal_annotation = bool(thermal_value or generic_value)
        if has_rgb_annotation and has_thermal_annotation:
            bucket["records_with_both_derived_paths"] += 1
        if not rgb_value and not thermal_value and not generic_value:
            anomalies["missing_path_annotation"].append(
                {"index": index, "identity": pid, "split": split}
            )
        elif has_rgb_annotation and not has_thermal_annotation:
            anomalies["only_rgb_annotation"].append(
                {"index": index, "identity": pid, "split": split, "path": rgb_value or generic_value}
            )
        elif has_thermal_annotation and not has_rgb_annotation:
            anomalies["only_thermal_annotation"].append(
                {"index": index, "identity": pid, "split": split, "path": thermal_value or generic_value}
            )

        if basename and rgb_names is not None:
            if basename in rgb_names:
                if pid is not None:
                    per_identity[str(pid)]["rgb"] += 1
            else:
                anomalies["rgb_missing_on_disk"].append(
                    {
                        "index": index,
                        "identity": pid,
                        "split": split,
                        "basename": basename,
                        "expected_path": str((rgb_dir / basename).resolve(strict=False)) if rgb_dir else None,
                    }
                )
        if basename and thermal_names is not None:
            if basename in thermal_names:
                if pid is not None:
                    per_identity[str(pid)]["thermal"] += 1
            else:
                anomalies["thermal_missing_on_disk"].append(
                    {
                        "index": index,
                        "identity": pid,
                        "split": split,
                        "basename": basename,
                        "expected_path": str((thermal_dir / basename).resolve(strict=False)) if thermal_dir else None,
                    }
                )
        if (
            basename
            and rgb_names is not None
            and thermal_names is not None
            and basename in rgb_names
            and basename in thermal_names
        ):
            bucket["records_with_both_files"] += 1

    split_output: dict[str, Any] = {}
    all_ids: set[str] = set()
    all_query_ids: set[str] = set()
    all_gallery_ids: set[str] = set()
    for split, values in sorted(per_split.items()):
        identities = set(values["identities"])
        all_ids.update(identities)
        all_query_ids.update(values["query_identities"])
        all_gallery_ids.update(values["gallery_identities"])
        ordered_ids = sorted(identities, key=_sort_identity)
        pid2label = {pid: index for index, pid in enumerate(ordered_ids)}
        split_output[split] = {
            "records": values["records"],
            "unique_original_identities": len(identities),
            "captions": values["captions"],
            "empty_captions": values["empty_captions"],
            "caption_per_record_histogram": dict(sorted(values["caption_histogram"].items())),
            "records_with_both_derived_paths": values["records_with_both_derived_paths"],
            "records_with_both_files": values["records_with_both_files"]
            if modality_inventory_known
            else None,
            "query_unique_identities": len(values["query_identities"]),
            "gallery_unique_identities": len(values["gallery_identities"]),
            "query_minus_gallery": sorted(
                set(values["query_identities"]) - set(values["gallery_identities"]),
                key=_sort_identity,
            ),
            "gallery_minus_query": sorted(
                set(values["gallery_identities"]) - set(values["query_identities"]),
                key=_sort_identity,
            ),
            "renumbered_unique_labels": len(pid2label),
            "renumbered_contiguous": list(pid2label.values()) == list(range(len(pid2label))),
            "raw_to_label_first_20": list(pid2label.items())[:20],
            "raw_to_label_total": len(pid2label),
        }

    train_ids = set(per_split.get("train", {}).get("identities", set()))
    test_ids = set(per_split.get("test", {}).get("identities", set()))
    zero_by_identity: list[dict[str, Any]] = []
    if modality_inventory_known:
        for pid, values in sorted(per_identity.items(), key=lambda item: _sort_identity(item[0])):
            if values["rgb"] == 0 or values["thermal"] == 0 or values["descriptions"] == 0:
                zero_by_identity.append(
                    {
                        "identity": pid,
                        "rgb": values["rgb"],
                        "thermal": values["thermal"],
                        "descriptions": values["descriptions"],
                        "splits": sorted(values["splits"]),
                    }
                )

    caption_counts = [_record_caption_entries(record, text_field) for record in records]
    caption_lengths = [len(items) for items in caption_counts]
    numeric_ids: list[int] = []
    for value in all_ids:
        try:
            numeric_ids.append(int(value))
        except (TypeError, ValueError):
            continue
    disk = {
        "rgb_dir": str(rgb_dir.resolve(strict=False)) if rgb_dir else None,
        "thermal_dir": str(thermal_dir.resolve(strict=False)) if thermal_dir else None,
        "rgb_file_count": len(rgb_names) if rgb_names is not None else None,
        "thermal_file_count": len(thermal_names) if thermal_names is not None else None,
        "paired_basenames": len(rgb_names & thermal_names)
        if rgb_names is not None and thermal_names is not None
        else None,
        "rgb_only_basenames": sorted(rgb_names - thermal_names)
        if rgb_names is not None and thermal_names is not None
        else None,
        "thermal_only_basenames": sorted(thermal_names - rgb_names)
        if rgb_names is not None and thermal_names is not None
        else None,
        "annotated_basenames": len(annotated_names),
        "paired_basenames_not_annotated": sorted((rgb_names & thermal_names) - annotated_names)
        if rgb_names is not None and thermal_names is not None
        else None,
    }
    return {
        "source": _source_label(annotation, repo),
        "field_mapping": mapping,
        "splits": split_output,
        "total_records": len(records),
        "total_unique_original_identities": len(all_ids),
        "original_identity_min": min(numeric_ids) if numeric_ids else None,
        "original_identity_max": max(numeric_ids) if numeric_ids else None,
        "total_captions": sum(caption_lengths),
        "total_empty_captions": sum(
            values["empty_captions"] for values in per_split.values()
        ),
        "caption_per_record_histogram": dict(Counter(caption_lengths)),
        "caption_per_record_min": min(caption_lengths) if caption_lengths else None,
        "caption_per_record_max": max(caption_lengths) if caption_lengths else None,
        "caption_per_record_median": statistics.median(caption_lengths) if caption_lengths else None,
        "train_test_identity_intersection": sorted(train_ids & test_ids, key=_sort_identity),
        "train_test_identity_union_count": len(train_ids | test_ids),
        "all_query_unique_identities": len(all_query_ids),
        "all_gallery_unique_identities": len(all_gallery_ids),
        "per_identity": {
            pid: {**values, "splits": sorted(values["splits"])}
            for pid, values in per_identity.items()
        },
        "identities_with_any_zero": zero_by_identity,
        "anomalies": anomalies,
        "disk": disk,
        "modality_inventory_known": modality_inventory_known,
    }


def _record_key(
    record: Mapping[str, Any], mapping: Mapping[str, Any], index: int
) -> tuple[str, str, str, int]:
    split = _record_split(record, mapping.get("split"))
    pid = _record_identity(record, mapping.get("identity"))
    path_fields = (
        list(mapping.get("rgb_paths", ()))
        + list(mapping.get("thermal_paths", ()))
        + list(mapping.get("generic_paths", ()))
    )
    path = _first_record_path(record, path_fields) or ""
    # loader 的 `_pair_paths` 只使用 basename；用同一口径对齐多个标注，
    # 使目录前缀变化体现为 file_path 字段差异，而不是整条记录失配。
    basename = Path(path.replace("\\", "/")).name if path else ""
    if basename:
        # basename 是当前 loader 配对 RGB/T 及辨识图像记录的实际键；split/id
        # 本身正是需要逐字段比较的内容，不能把它们放进对齐键。
        return "basename", "", basename, 0
    return "index", "", "", index


def _canonical_record_map(probe: Mapping[str, Any]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    output: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    duplicates: Counter[tuple[Any, ...]] = Counter()
    for index, record in enumerate(probe["_records"]):
        base = _record_key(record, probe["field_mapping"], index)
        occurrence = duplicates[base]
        duplicates[base] += 1
        output[(*base, occurrence)] = record
    return output


def _md5(path: Path) -> str | None:
    try:
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _compare_annotations(probes: Sequence[Mapping[str, Any]], repo: Path) -> list[dict[str, Any]]:
    annotation_like = [
        probe
        for probe in probes
        if probe.get("status") == "OK"
        and probe.get("field_mapping", {}).get("identity")
        and probe.get("field_mapping", {}).get("text")
        and (
            probe.get("field_mapping", {}).get("rgb_paths")
            or probe.get("field_mapping", {}).get("thermal_paths")
            or probe.get("field_mapping", {}).get("generic_paths")
        )
    ]
    rgbt_like = [
        probe
        for probe in annotation_like
        if "rgbt-pedes" in Path(str(probe.get("path", ""))).as_posix().lower()
    ]
    if len(rgbt_like) >= 2:
        annotation_like = rgbt_like
    comparisons: list[dict[str, Any]] = []
    for left, right in itertools.combinations(annotation_like, 2):
        left_map = _canonical_record_map(left)
        right_map = _canonical_record_map(right)
        left_keys, right_keys = set(left_map), set(right_map)
        shared = left_keys & right_keys
        field_names = sorted(
            set(itertools.chain.from_iterable(record.keys() for record in left_map.values()))
            | set(itertools.chain.from_iterable(record.keys() for record in right_map.values())),
            key=str,
        )
        fields: list[dict[str, Any]] = []
        for field in field_names:
            same = mismatch = only_left = only_right = 0
            examples: list[dict[str, Any]] = []
            mismatch_record_keys: list[list[Any]] = []
            for key in shared:
                left_record, right_record = left_map[key], right_map[key]
                left_has, right_has = field in left_record, field in right_record
                if left_has and not right_has:
                    only_left += 1
                elif right_has and not left_has:
                    only_right += 1
                elif _stable_value(left_record.get(field)) == _stable_value(right_record.get(field)):
                    same += 1
                else:
                    mismatch += 1
                    mismatch_record_keys.append(list(key))
                    if len(examples) < 5:
                        examples.append(
                            {
                                "record_key": list(key),
                                "left": _short_value(left_record.get(field)),
                                "right": _short_value(right_record.get(field)),
                            }
                        )
            fields.append(
                {
                    "field": str(field),
                    "same": same,
                    "value_mismatch": mismatch,
                    "only_left": only_left,
                    "only_right": only_right,
                    "examples": examples,
                    "mismatch_record_keys": mismatch_record_keys,
                }
            )
        comparisons.append(
            {
                "left": left["source"],
                "right": right["source"],
                "source": f"逐记录逐字段比较：{left['source']} ↔ {right['source']}",
                "left_md5": _md5(Path(left["path"])),
                "right_md5": _md5(Path(right["path"])),
                "byte_identical": _md5(Path(left["path"])) == _md5(Path(right["path"])),
                "left_records": len(left_map),
                "right_records": len(right_map),
                "shared_records": len(shared),
                "only_left_records": len(left_keys - right_keys),
                "only_right_records": len(right_keys - left_keys),
                "only_left_record_keys": [list(key) for key in sorted(left_keys - right_keys, key=str)],
                "only_right_record_keys": [list(key) for key in sorted(right_keys - left_keys, key=str)],
                "fields": fields,
                "status": "OK"
                if left_keys == right_keys
                and all(
                    row["value_mismatch"] == row["only_left"] == row["only_right"] == 0
                    for row in fields
                )
                else "WARN",
            }
        )
    return comparisons


def _record_expected_annotation_files(
    probes: Sequence[Mapping[str, Any]],
    findings: list[AuditFinding],
    data_root: Path,
    dataset_name: str,
) -> None:
    """记录任务指定的数据集标注是否可在本机复核。"""

    paths = [Path(str(probe.get("path", ""))) for probe in probes]
    if dataset_name == "RGBT-PEDES":
        expected = (
            ("RGBT-PEDES-Final/Caption.json", lambda path: path.name == "Caption.json" and path.parent.name == "RGBT-PEDES-Final"),
            ("RGBT-PEDES/caption.json", lambda path: path.name == "caption.json" and path.parent.name == "RGBT-PEDES"),
            ("caption_2_no_color.json", lambda path: path.name == "caption_2_no_color.json"),
        )
    else:
        expected = (
            ("RGBNT201-PEDES/captionform0_8B.json", lambda path: path.name == "captionform0_8B.json" and path.parent.name == "RGBNT201-PEDES"),
        )
    for label, predicate in expected:
        matches = [str(path.resolve(strict=False)) for path in paths if predicate(path)]
        _finding(
            findings,
            f"指定标注文件：{label}",
            matches if matches else "未找到",
            "; ".join(matches) if matches else str(data_root),
            "OK" if matches else "UNKNOWN",
            "用于三份标注的逐对逐字段复核；缺失时保留 UNKNOWN，供服务器数据目录复现",
        )


def _read_source_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return []


def _first_source_match(
    path: Path, pattern: str, repo: Path, context_after: int = 0
) -> dict[str, Any]:
    regex = re.compile(pattern)
    lines = _read_source_lines(path)
    for index, line in enumerate(lines, start=1):
        if regex.search(line):
            excerpt = [
                {"line": number, "text": lines[number - 1]}
                for number in range(index, min(len(lines), index + context_after) + 1)
            ]
            return {
                "source": _source_label(path, repo, index),
                "line": index,
                "text": line,
                "excerpt": excerpt,
                "status": "OK",
            }
    return {
        "source": _source_label(path, repo),
        "line": None,
        "text": "未找到",
        "excerpt": [],
        "status": "UNKNOWN",
    }


def _loader_static_audit(repo: Path, inferred_train_ids: int | None) -> dict[str, Any]:
    loader_path = repo / "datasets" / "rgbtpedes.py"
    build_path = repo / "datasets" / "build.py"
    facts = [
        {
            "item": "数据目录解析优先级",
            **_first_source_match(loader_path, r"^\s*candidates\s*=\s*\[", repo, 4),
        },
        {
            "item": "标注文件名解析优先级",
            **_first_source_match(loader_path, r"for\s+fname\s+in\s+\(.*Caption\.json", repo),
        },
        {
            "item": "val 为空时复用 test",
            **_first_source_match(
                loader_path,
                r"if\s+len\(val_annos\)\s*==\s*0",
                repo,
                1,
            ),
        },
        {
            "item": "每个 split 独立身份重编号",
            **_first_source_match(loader_path, r"raw_pids\s*=\s*sorted", repo, 1),
        },
    ]
    loader_lines = _read_source_lines(loader_path)
    skip_patterns = (
        re.compile(r"\bcontinue\b"),
        re.compile(r"\bfilter\w*\b", re.IGNORECASE),
        re.compile(r"\bskip\w*\b", re.IGNORECASE),
    )
    skip_matches = [
        {"line": index, "text": line}
        for index, line in enumerate(loader_lines, start=1)
        if any(pattern.search(line) for pattern in skip_patterns)
    ]
    len_zero_matches = [
        {"line": index, "text": line}
        for index, line in enumerate(loader_lines, start=1)
        if re.search(r"if.*len\s*\(.*\)\s*==\s*0", line, re.IGNORECASE)
    ]
    len_zero_skip_matches = [
        row
        for row in len_zero_matches
        if row["line"] < len(loader_lines)
        and re.search(
            r"\b(?:continue|return)\b",
            loader_lines[row["line"]],
            re.IGNORECASE,
        )
    ]
    len_zero_evidence = "; ".join(
        f"{row['line']}: {row['text'].strip()}"
        + (
            f"；{row['line'] + 1}: {loader_lines[row['line']].strip()}"
            if row["line"] < len(loader_lines)
            else ""
        )
        for row in len_zero_matches
    ) or "无"
    facts.append(
        {
            "item": "rgbtpedes.py 全文过滤/跳过逻辑",
            "source": (
                f"{_source_label(loader_path, repo)}:1-{len(loader_lines)}；"
                "正则 continue|filter|skip 及 if.*len(...)==0 下一条语句 continue/return"
            ),
            "line": None,
            "text": (
                f"continue/filter/skip 匹配数={len(skip_matches)}；"
                f"len(...)==0 命中：{len_zero_evidence}；"
                f"其下一条语句 continue/return 命中数={len(len_zero_skip_matches)}"
            ),
            "excerpt": skip_matches,
            "status": "OK" if not skip_matches and not len_zero_skip_matches else "WARN",
        }
    )
    num_classes = _first_source_match(
        build_path, r"num_classes\s*=\s*len\(dataset\.train_id_container\)", repo
    )
    num_classes.update(
        {
            "expression": "len(dataset.train_id_container)",
            "inferred_value": inferred_train_ids,
            "status": "OK"
            if num_classes.get("line") is not None and inferred_train_ids is not None
            else "UNKNOWN",
        }
    )

    filters: list[dict[str, Any]] = []
    patterns = (
        re.compile(r"if.*len\s*\(.*\)\s*==\s*0", re.IGNORECASE),
        re.compile(r"\bcontinue\b", re.IGNORECASE),
        re.compile(r"\bfilter\w*\b", re.IGNORECASE),
        re.compile(r"\bskip\w*\b", re.IGNORECASE),
    )
    datasets_dir = repo / "datasets"
    if datasets_dir.is_dir():
        for path in sorted(datasets_dir.rglob("*.py"), key=lambda item: str(item).lower()):
            for line_number, line in enumerate(_read_source_lines(path), start=1):
                if any(pattern.search(line) for pattern in patterns):
                    filters.append(
                        {
                            "source": _source_label(path, repo, line_number),
                            "line": line_number,
                            "text": line,
                            "status": "OK",
                        }
                    )
    train_expand = _first_source_match(
        loader_path, r"for\s+caption\s+in\s+anno\[.captions.\]", repo, 3
    )
    return {
        "facts": facts,
        "num_classes": num_classes,
        "filter_candidates": filters,
        "training_expansion": train_expand,
    }


def _discover_loader_modules(repo: Path) -> list[dict[str, Any]]:
    """先静态搜索数据构建候选模块，再由动态探测选择最小安全入口。"""

    candidates: list[dict[str, Any]] = []
    content_pattern = re.compile(
        r"(?:\bclass\s+\w*Dataset\w*|\bdef\s+\w*(?:build|loader)\w*\s*\(|"
        r"\b\w*DataLoader\w*\b|\b\w*Dataset\w*\b)"
    )
    for directory_name in ("datasets", "data", "utils"):
        directory = repo / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py"), key=lambda item: item.as_posix().lower()):
            lines = _read_source_lines(path)
            name_match = bool(re.search(r"(?:build|dataset|loader)", path.stem, re.IGNORECASE))
            evidence = [
                {"line": index, "text": line}
                for index, line in enumerate(lines, start=1)
                if content_pattern.search(line)
            ][:5]
            if not name_match and not evidence:
                continue
            relative = path.relative_to(repo).with_suffix("")
            candidates.append(
                {
                    "module": ".".join(relative.parts),
                    "source": _source_label(path, repo),
                    "name_match": name_match,
                    "evidence": evidence,
                }
            )
    return candidates


def _loader_dynamic_probe(repo: Path, data_root: Path, dataset_name: str) -> dict[str, Any]:
    """best-effort 构造 Dataset 对象；不创建 DataLoader，也不访问任何样本。"""

    source = f"import datasets.rgbtpedes.RGBTPEDES(root={data_root}, verbose=False)"
    discovered_modules = _discover_loader_modules(repo)
    if dataset_name != "RGBT-PEDES":
        return {
            "status": "UNKNOWN",
            "source": "datasets/build.py::__factory（当前未注册 RGBNT201-PEDES）",
            "train_length": None,
            "test_gallery_length": None,
            "test_query_length": None,
            "train_unique_labels": None,
            "test_unique_labels": None,
            "train_labels_contiguous": None,
            "test_labels_contiguous": None,
            "note": "仅完成标注与磁盘静态探测；当前仓库没有 RGBNT201-PEDES Dataset 类可构造。",
            "discovered_modules": discovered_modules,
        }
    original_path = list(sys.path)
    original_no_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(repo))
        module = importlib.import_module("datasets.rgbtpedes")
        module_path = Path(getattr(module, "__file__", "")).resolve(strict=False)
        try:
            module_path.relative_to(repo.resolve(strict=False))
        except ValueError as exc:
            raise ImportError(f"导入到了仓库外同名模块：{module_path}") from exc
        dataset_class = getattr(module, "RGBTPEDES")
        dataset = dataset_class(root=str(data_root), verbose=False)
        train_labels = sorted(getattr(dataset, "train_id_container", set()))
        test_labels = sorted(getattr(dataset, "test_id_container", set()))
        test_data = getattr(dataset, "test", {})
        return {
            "status": "OK",
            "source": source,
            "train_length": len(getattr(dataset, "train", [])),
            "test_gallery_length": len(test_data.get("image_pids", [])),
            "test_query_length": len(test_data.get("caption_pids", [])),
            "train_unique_labels": len(train_labels),
            "test_unique_labels": len(test_labels),
            "train_labels_contiguous": train_labels == list(range(len(train_labels))),
            "test_labels_contiguous": test_labels == list(range(len(test_labels))),
            "note": "仅构造 RGBTPEDES；未构造 DataLoader、未索引 dataset、未执行 forward。",
            "discovered_modules": discovered_modules,
        }
    except Exception as exc:
        return {
            "status": "UNKNOWN",
            "source": source,
            "train_length": None,
            "test_gallery_length": None,
            "test_query_length": None,
            "train_unique_labels": None,
            "test_unique_labels": None,
            "train_labels_contiguous": None,
            "test_labels_contiguous": None,
            "note": f"{type(exc).__name__}: {exc}",
            "discovered_modules": discovered_modules,
        }
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_no_bytecode


def _static_mraf_parameters(repo: Path) -> list[dict[str, Any]]:
    path = repo / "model" / "mraf.py"
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError):
        return [
            {
                "scope": "model/mraf.py",
                "parameter": "无法判定",
                "default": None,
                "source": _source_label(path, repo),
                "status": "UNKNOWN",
            }
        ]
    output: list[dict[str, Any]] = []

    def collect(function: ast.FunctionDef, scope: str) -> None:
        args = list(function.args.args)
        defaults = [None] * (len(args) - len(function.args.defaults)) + list(function.args.defaults)
        for argument, default in zip(args, defaults):
            if argument.arg == "self":
                continue
            default_text = "无默认值" if default is None else ast.unparse(default)
            output.append(
                {
                    "scope": scope,
                    "parameter": argument.arg,
                    "default": default_text,
                    "source": _source_label(path, repo, argument.lineno),
                    "status": "OK",
                }
            )

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ModalityReliabilityAdaptiveFusion":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "__init__":
                    collect(member, f"{node.name}.__init__")
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "compute_mraf_reliability_targets",
            "resolve_mraf_rel_lambda",
        }:
            collect(node, node.name)

    # CLI 使用带 mraf_ 前缀的公开变量名；它们并不直接定义在 model/mraf.py
    # 的函数签名中，因此另行静态列出来源，避免把两层名称混为一谈。
    options_path = repo / "utils" / "options.py"
    options_lines = _read_source_lines(options_path)
    option_pattern = re.compile(
        r"add_argument\(\s*[\"']--(?P<name>mraf_[A-Za-z0-9_]+)[\"'](?P<body>.*)"
    )
    for line_number, line in enumerate(options_lines, start=1):
        match = option_pattern.search(line)
        if not match:
            continue
        body = match.group("body")
        default_match = re.search(r"\bdefault\s*=\s*([^,)]+)", body)
        if default_match:
            default_text = default_match.group(1).strip()
        elif re.search(r"action\s*=\s*[\"']store_true[\"']", body):
            default_text = "False（store_true）"
        elif re.search(r"action\s*=\s*[\"']store_false[\"']", body):
            default_text = "True（store_false）"
        else:
            default_text = "无法判定"
        output.append(
            {
                "scope": "utils/options.py CLI（传入 model/mraf.py 或训练逻辑）",
                "parameter": match.group("name"),
                "default": default_text,
                "source": _source_label(options_path, repo, line_number),
                "status": "OK",
            }
        )
    return output


def _finding(
    findings: list[AuditFinding], item: str, value: Any, source: str, status: str, note: str = ""
) -> None:
    findings.append(
        AuditFinding(item=item, value=value, source=source, status=status, note=note)
    )


def _format_list(values: Sequence[Any] | None, empty: str = "无") -> str:
    if values is None:
        return "无法判定"
    if not values:
        return empty
    return ", ".join(str(value) for value in values)


def _md_escape(value: Any) -> str:
    if value is None:
        return "无法判定"
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(_md_escape(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_md_escape(item) for item in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _baseline_measurements(stats: Mapping[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {}
    train = stats.get("splits", {}).get("train", {})
    test = stats.get("splits", {}).get("test", {})
    return {
        "total_records": stats.get("total_records"),
        "total_identities": stats.get("train_test_identity_union_count"),
        "train_identities": train.get("unique_original_identities"),
        "train_records": train.get("records"),
        "train_captions": train.get("captions"),
        "test_identities": test.get("unique_original_identities"),
        "test_records": test.get("records"),
        "test_captions": test.get("captions"),
        "train_test_intersection": len(stats.get("train_test_identity_intersection", ())),
        "caption_histogram": stats.get("caption_per_record_histogram"),
        "empty_captions": stats.get("total_empty_captions"),
        "train_empty_captions": train.get("empty_captions"),
        "test_empty_captions": test.get("empty_captions"),
    }


def _render_report(
    repo: Path,
    dataset_name: str,
    dump_path: Path,
    probes: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any] | None,
    stats: Mapping[str, Any] | None,
    comparisons: Sequence[Mapping[str, Any]],
    loader_resolution: Mapping[str, Any],
    static_loader: Mapping[str, Any],
    dynamic_loader: Mapping[str, Any],
    mraf_parameters: Sequence[Mapping[str, Any]],
    findings: list[AuditFinding],
    discovery_notes: Sequence[str],
) -> Path:
    report_path = repo / "docs" / "audit" / f"DATA_AUDIT_{dataset_name}.md"
    sections: list[tuple[str, str]] = []

    probe_rows = []
    for probe in probes:
        mapping = probe["field_mapping"]
        probe_rows.append(
            (
                probe["source"],
                probe["top_type"],
                probe["record_count"],
                mapping.get("identity") or "未找到",
                mapping.get("split") or "未找到",
                mapping.get("text") or "未找到",
                _format_list(
                    list(mapping.get("rgb_paths", ()))
                    + list(mapping.get("thermal_paths", ()))
                    + list(mapping.get("generic_paths", ()))
                ),
                _md5(Path(probe["path"])) or "无法判定",
                probe["status"],
            )
        )
    if not probe_rows:
        probe_rows.append(("；".join(discovery_notes) or str(loader_resolution["root_argument"]), "未找到", 0, "未找到", "未找到", "未找到", "未找到", "无法判定", "UNKNOWN"))
    comparison_lines = []
    if comparisons:
        for comparison in comparisons:
            comparison_lines.append(
                f"- `{comparison['left']}` ↔ `{comparison['right']}`："
                f"共享记录 {comparison['shared_records']}，仅左 {comparison['only_left_records']}，"
                f"仅右 {comparison['only_right_records']}，逐字段状态 {comparison['status']}。"
                f" MD5(left)={comparison['left_md5']}，MD5(right)={comparison['right_md5']}，"
                f"字节相同={comparison['byte_identical']}。"
                f"source={comparison['source']}；status={comparison['status']}"
            )
            if comparison["only_left_record_keys"] or comparison["only_right_record_keys"]:
                comparison_lines.append(
                    "  - 仅左记录键："
                    + _format_list(comparison["only_left_record_keys"])
                    + "；仅右记录键："
                    + _format_list(comparison["only_right_record_keys"])
                    + "。完整清单见 `audit_out/dataset_probe.json`。"
                )
            mismatch_rows = [
                row for row in comparison["fields"]
                if row["value_mismatch"] or row["only_left"] or row["only_right"]
            ]
            if mismatch_rows:
                comparison_lines.append(
                    _table(
                        ("字段", "值不同", "仅左有", "仅右有", "不同记录键（前20）", "source", "status"),
                        [
                            (
                                row["field"], row["value_mismatch"], row["only_left"],
                                row["only_right"],
                                _format_list(row.get("mismatch_record_keys", [])[:20]),
                                comparison["source"], "WARN"
                            )
                            for row in mismatch_rows
                        ],
                    )
                )
    else:
        comparison_lines.append(
            f"- 可逐对比较的标注少于 2 个。source={loader_resolution['root_argument']}；status=UNKNOWN"
        )
    sections.append(
        (
            "一、探测到的标注文件与字段映射",
            _table(
                ("标注/source", "顶层类型", "记录数", "身份字段", "划分字段", "文本字段", "路径字段", "MD5", "status"),
                probe_rows,
            )
            + "\n\n### 标注逐对逐字段比对\n\n"
            + "\n\n".join(comparison_lines)
            + "\n\n### 仓库 loader 的解析优先级推断\n\n"
            + _table(
                ("项目", "值", "source", "status"),
                (
                    ("传给 loader 的 root", loader_resolution["root_argument"], loader_resolution["source"], loader_resolution["status"]),
                    ("data_dir 候选顺序", " → ".join(loader_resolution["data_candidates_in_order"]), loader_resolution["source"], loader_resolution["status"]),
                    ("推断选中 data_dir", loader_resolution["selected_data_dir"], loader_resolution["source"], loader_resolution["status"]),
                    ("标注文件名顺序", " → ".join(loader_resolution["annotation_names_in_order"]), loader_resolution["source"], loader_resolution["status"]),
                    ("推断实际加载标注", loader_resolution["selected_annotation"], loader_resolution["source"], loader_resolution["status"]),
                ),
            ),
        )
    )

    stats_rows: list[Sequence[Any]] = []
    if stats:
        for split in ("train", "test"):
            row = stats["splits"].get(split)
            if row:
                stats_rows.append(
                    (
                        split,
                        row["unique_original_identities"],
                        row["renumbered_unique_labels"],
                        row["records"],
                        row["records_with_both_derived_paths"],
                        row["records_with_both_files"] if row["records_with_both_files"] is not None else "无法判定",
                        row["captions"],
                        row["caption_per_record_histogram"],
                        stats["source"],
                        "OK",
                    )
                )
            else:
                stats_rows.append((split, "未找到", "未找到", "未找到", "未找到", "无法判定", "未找到", "未找到", stats["source"], "UNKNOWN"))
        stats_rows.append(
            (
                "合计",
                stats["total_unique_original_identities"],
                "各 split 独立编号",
                stats["total_records"],
                sum(row["records_with_both_derived_paths"] for row in stats["splits"].values()),
                (
                    sum(row["records_with_both_files"] for row in stats["splits"].values())
                    if stats["modality_inventory_known"]
                    else "无法判定"
                ),
                stats["total_captions"],
                stats["caption_per_record_histogram"],
                stats["source"],
                "OK",
            )
        )
    else:
        stats_rows.append(("train/test/合计", "未找到", "无法判定", "未找到", "未找到", "无法判定", "未找到", "未找到", loader_resolution["root_argument"], "UNKNOWN"))
    sections.append(
        (
            "二、身份、图像对与文本统计表",
            _table(
                ("划分", "原始唯一身份", "重编号标签数", "标注记录", "具备两侧派生路径", "磁盘实际配对", "文本描述", "每记录 caption 分布", "source", "status"),
                stats_rows,
            )
            + (
                "\n\n"
                + _table(
                    ("项目", "值", "source", "status"),
                    (
                        ("train ∩ test 身份", _format_list(stats["train_test_identity_intersection"]), stats["source"], "WARN" if stats["train_test_identity_intersection"] else "OK"),
                        ("train ∪ test 身份数", stats["train_test_identity_union_count"], stats["source"], "OK"),
                        ("原始身份 ID 最小值", stats["original_identity_min"], stats["source"], "OK" if stats["original_identity_min"] is not None else "UNKNOWN"),
                        ("原始身份 ID 最大值", stats["original_identity_max"], stats["source"], "OK" if stats["original_identity_max"] is not None else "UNKNOWN"),
                        ("caption/记录 最小值", stats["caption_per_record_min"], stats["source"], "OK"),
                        ("caption/记录 中位数", stats["caption_per_record_median"], stats["source"], "OK"),
                        ("caption/记录 最大值", stats["caption_per_record_max"], stats["source"], "OK"),
                    ),
                )
                if stats
                else ""
            ),
        )
    )

    measured = _baseline_measurements(stats)
    reference_rows = []
    for key, item, baseline_value in CONFIRMED_BASELINES[dataset_name]:
        actual = measured.get(key)
        if actual is None:
            consistency, status = "无法判定", "UNKNOWN"
        elif actual == baseline_value:
            consistency, status = "逐值一致", "OK"
        else:
            consistency, status = "不一致", "WARN"
        reference_rows.append(
            (item, actual, baseline_value, consistency, stats["source"] if stats else loader_resolution["root_argument"], status)
        )
    sections.append(
        (
            "三、本次实测 vs 已确证基线",
            _table(
                ("项目", "本次实测", "已确证基线", "是否一致", "source", "status"),
                reference_rows,
            )
            + f"\n\n基线名称={dataset_name}；source=任务给定已确证基线。",
        )
    )

    anomaly_rows: list[Sequence[Any]] = []
    if stats:
        anomaly_labels = {
            "missing_identity": "缺身份字段",
            "missing_split": "缺 split",
            "missing_path_annotation": "无路径标注",
            "only_rgb_annotation": "仅 RGB 路径标注",
            "only_thermal_annotation": "仅 T 路径标注",
            "rgb_missing_on_disk": "RGB 文件未找到",
            "thermal_missing_on_disk": "T 文件未找到",
            "no_description": "无描述",
        }
        for key, label in anomaly_labels.items():
            values = stats["anomalies"][key]
            identities = sorted(
                {str(row.get("identity")) for row in values if row.get("identity") is not None},
                key=_sort_identity,
            )
            if key in {"rgb_missing_on_disk", "thermal_missing_on_disk"}:
                detail_values = sorted(
                    {
                        str(row.get("expected_path") or row.get("basename"))
                        for row in values
                        if row.get("expected_path") or row.get("basename")
                    }
                )
            elif key in {"only_rgb_annotation", "only_thermal_annotation"}:
                detail_values = sorted(
                    {str(row["path"]) for row in values if row.get("path")}
                )
            else:
                detail_values = identities
            anomaly_rows.append((label, len(values), _format_list(detail_values), stats["source"], "WARN" if values else "OK"))
        disk = stats["disk"]
        for label, key in (("磁盘仅 RGB basename", "rgb_only_basenames"), ("磁盘仅 T basename", "thermal_only_basenames"), ("成对文件未见于标注", "paired_basenames_not_annotated")):
            values = disk.get(key)
            anomaly_rows.append((label, len(values) if values is not None else "无法判定", _format_list(values), f"{disk.get('rgb_dir')}；{disk.get('thermal_dir')}", "UNKNOWN" if values is None else ("WARN" if values else "OK")))
        zero_values = stats["identities_with_any_zero"]
        anomaly_rows.append(("任一 RGB/T/描述计数为 0 的身份", len(zero_values), _format_list([row["identity"] for row in zero_values]), stats["source"], "UNKNOWN" if not stats["modality_inventory_known"] else ("WARN" if zero_values else "OK")))
    else:
        anomaly_rows.append(("异常记录", "无法判定", "未找到标注", loader_resolution["root_argument"], "UNKNOWN"))
    sections.append(
        (
            "四、异常记录清单",
            (
                _table(
                    ("项目", "值", "source", "status"),
                    (
                        (
                            "RGB 目录文件数",
                            stats["disk"]["rgb_file_count"] if stats else "无法判定",
                            stats["disk"]["rgb_dir"] if stats else loader_resolution["root_argument"],
                            "OK" if stats and stats["disk"]["rgb_file_count"] is not None else "UNKNOWN",
                        ),
                        (
                            "T 目录文件数",
                            stats["disk"]["thermal_file_count"] if stats else "无法判定",
                            stats["disk"]["thermal_dir"] if stats else loader_resolution["root_argument"],
                            "OK" if stats and stats["disk"]["thermal_file_count"] is not None else "UNKNOWN",
                        ),
                        (
                            "RGB/T 同名配对文件数",
                            stats["disk"]["paired_basenames"] if stats else "无法判定",
                            (
                                f"{stats['disk']['rgb_dir']}；{stats['disk']['thermal_dir']}"
                                if stats
                                else loader_resolution["root_argument"]
                            ),
                            "OK" if stats and stats["disk"]["paired_basenames"] is not None else "UNKNOWN",
                        ),
                    ),
                )
                + "\n\n"
                + _table(("检查项", "记录/文件数", "身份 / basename / 预期路径", "source", "status"), anomaly_rows)
            ),
        )
    )

    loader_rows = []
    for fact in static_loader["facts"]:
        original = "<br>".join(f"{row['line']}: {row['text']}" for row in fact["excerpt"])
        loader_rows.append((fact["item"], original or fact.get("text", "未找到"), fact["source"], fact["status"]))
    num_classes = static_loader["num_classes"]
    loader_rows.append(("ID 分类头 num_classes", f"{num_classes['expression']}；推断值={num_classes['inferred_value']}", num_classes["source"], num_classes["status"]))
    expansion = static_loader["training_expansion"]
    train_captions = stats.get("splits", {}).get("train", {}).get("captions") if stats else None
    loader_rows.append(("训练样本按 caption 展开", f"原文={expansion['text']}；推断展开样本数={train_captions}", expansion["source"], "OK" if expansion["line"] and train_captions is not None else "UNKNOWN"))
    dynamic_rows = [
        (
            "预先搜索的数据构建候选模块",
            _format_list([row["module"] for row in dynamic_loader.get("discovered_modules", [])]),
            "datasets/、data/、utils/ 静态扫描；逐项 source 见 audit_out/dataset_probe.json",
            "OK" if dynamic_loader.get("discovered_modules") else "UNKNOWN",
        ),
        ("构造 Dataset 状态", dynamic_loader["status"], dynamic_loader["source"], dynamic_loader["status"]),
        ("train dataset 长度", dynamic_loader["train_length"], dynamic_loader["source"], dynamic_loader["status"]),
        ("test gallery/image_pids 长度", dynamic_loader["test_gallery_length"], dynamic_loader["source"], dynamic_loader["status"]),
        ("test query/caption_pids 长度", dynamic_loader["test_query_length"], dynamic_loader["source"], dynamic_loader["status"]),
        ("train 实际唯一 label 数", dynamic_loader["train_unique_labels"], dynamic_loader["source"], dynamic_loader["status"]),
        ("test 实际唯一 label 数", dynamic_loader["test_unique_labels"], dynamic_loader["source"], dynamic_loader["status"]),
        ("train label 连续", dynamic_loader["train_labels_contiguous"], dynamic_loader["source"], dynamic_loader["status"]),
        ("test label 连续", dynamic_loader["test_labels_contiguous"], dynamic_loader["source"], dynamic_loader["status"]),
        ("内省说明", dynamic_loader["note"], dynamic_loader["source"], dynamic_loader["status"]),
    ]
    filter_rows = [
        (row["source"], row["text"], row["status"]) for row in static_loader["filter_candidates"]
    ] or [("datasets/", "未找到匹配行", "UNKNOWN")]
    renumber_rows: list[Sequence[Any]] = []
    if stats:
        for split, values in stats["splits"].items():
            label_range = (
                f"0..{values['raw_to_label_total'] - 1}"
                if values["raw_to_label_total"]
                else "空"
            )
            renumber_rows.append((split, f"{values['renumbered_contiguous']}；范围={label_range}", values["raw_to_label_first_20"], values["raw_to_label_total"], stats["source"], "OK"))
    else:
        renumber_rows.append(("无法判定", "无法判定", "未找到", 0, loader_resolution["root_argument"], "UNKNOWN"))
    mraf_rows = [
        (row["scope"], row["parameter"], row["default"], row["source"], row["status"])
        for row in mraf_parameters
    ]
    sections.append(
        (
            "五、loader 内省结果",
            "### 四项专项事实（精确行号与原文）\n\n"
            + _table(("事实", "源码原文", "source", "status"), loader_rows)
            + "\n\n### 原始 ID 到重编号标签\n\n"
            + _table(("split", "连续整数", "映射前20条", "映射总数", "source", "status"), renumber_rows)
            + "\n\n### Dataset 对象 best-effort 内省\n\n"
            + _table(("检查项", "值", "source", "status"), dynamic_rows)
            + "\n\n### 数据构建路径中可能的过滤/跳过行（仅列事实）\n\n"
            + _table(("source", "原文", "status"), filter_rows)
            + "\n\n### `model/mraf.py` 参数静态清点\n\n"
            + _table(("作用域", "参数", "默认值", "source", "status"), mraf_rows),
        )
    )

    query_rows: list[Sequence[Any]] = []
    if stats:
        test = stats["splits"].get("test")
        if test:
            equal = test["query_unique_identities"] == test["gallery_unique_identities"]
            query_rows.extend(
                (
                    ("gallery/image_pids 数量", test["records"], stats["source"], "OK"),
                    ("query/caption_pids 数量", test["captions"], stats["source"], "OK"),
                    ("query（文本）唯一身份", test["query_unique_identities"], stats["source"], "OK"),
                    ("gallery（图像）唯一身份", test["gallery_unique_identities"], stats["source"], "OK"),
                    ("两侧是否相等", equal, stats["source"], "OK" if equal else "WARN"),
                    ("query - gallery", _format_list(test["query_minus_gallery"]), stats["source"], "WARN" if test["query_minus_gallery"] else "OK"),
                    ("gallery - query", _format_list(test["gallery_minus_query"]), stats["source"], "WARN" if test["gallery_minus_query"] else "OK"),
                )
            )
        else:
            query_rows.append(("test query/gallery", "未找到 test split", stats["source"], "UNKNOWN"))
    else:
        query_rows.append(("test query/gallery", "无法判定", loader_resolution["root_argument"], "UNKNOWN"))
    sections.append(("六、query/gallery 口径", _table(("检查项", "值", "source", "status"), query_rows)))

    unknowns = [finding for finding in findings if str(finding.status).upper() == "UNKNOWN"]
    unresolved_rows = [
        (finding.item, finding.value, finding.source, finding.note or "未解决")
        for finding in unknowns
    ]
    if not unresolved_rows:
        unresolved_rows.append(("UNKNOWN 检查项", "无", "本报告 findings 汇总", "当前无 UNKNOWN"))
    sections.append(("七、未解决项", _table(("检查项", "值", "source", "说明"), unresolved_rows)))
    sections.append(
        (
            "附录：原始探测输出路径",
            _table(
                ("文件", "source", "status"),
                ((str(dump_path.resolve(strict=False)), str(dump_path.resolve(strict=False)), "OK"),),
            ),
        )
    )
    write_report(report_path, sections)
    return report_path


def _json_ready_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in probe.items() if key != "_records"}


def _write_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读探测数据标注、RGB/T 文件完整性、身份重编号与评测口径。"
    )
    parser.add_argument(
        "--dataset-name",
        choices=SUPPORTED_DATASETS,
        default="RGBT-PEDES",
        help="待审计数据集（默认：RGBT-PEDES）",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="待探测的数据根目录")
    parser.add_argument("--anno", type=Path, default=None, help="显式指定标注文件")
    parser.add_argument("--dump-schema", action="store_true", help="仅探测候选标注结构后退出")
    parser.add_argument("--out", type=Path, default=Path("audit_out"), help="原始 JSON 输出目录")
    parser.add_argument("--repo", type=Path, default=None, help="源码仓库根；默认自动定位")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = _find_repo(args.repo if args.repo is not None else Path(__file__).resolve().parents[2])
    data_root = args.data_root.expanduser().resolve(strict=False)
    explicit = args.anno.expanduser().resolve(strict=False) if args.anno else None
    out_dir = resolve_audit_output_dir(repo, args.out)
    dump_path = out_dir / "dataset_probe.json"

    findings: list[AuditFinding] = []
    candidates, discovery_notes = _find_candidates(data_root, explicit)
    if not candidates:
        _finding(
            findings,
            "候选标注文件",
            "未找到",
            str(explicit or data_root),
            "UNKNOWN",
            "；".join(discovery_notes) or "没有大于 10KB 的 JSON/PKL/CSV 候选",
        )
    probes = [_probe_candidate(path, repo) for path in candidates]
    for probe in probes:
        if probe["status"] == "UNKNOWN":
            _finding(findings, "标注读取", "无法判定", probe["source"], "UNKNOWN", probe.get("error") or "读取失败")
    _record_expected_annotation_files(probes, findings, data_root, args.dataset_name)

    if args.dump_schema:
        payload: dict[str, Any] = {
            "generated_at": _now_iso(),
            "mode": "schema_only",
            "arguments": {
                "data_root": str(data_root),
                "dataset_name": args.dataset_name,
                "anno": str(explicit) if explicit else None,
                "out": str(out_dir),
                "repo": str(repo),
            },
            "discovery_notes": discovery_notes,
            "annotation_candidates": [_json_ready_probe(probe) for probe in probes],
            "findings": [asdict(finding) for finding in findings],
            "unknown_items": [
                asdict(finding)
                for finding in findings
                if str(finding.status).upper() == "UNKNOWN"
            ],
        }
        _write_dump(dump_path, payload)
        print(f"schema dump: {dump_path}")
        return 0

    loader = _loader_resolution(repo, data_root, args.dataset_name)
    if loader["status"] == "UNKNOWN":
        _finding(
            findings,
            "仓库 loader 实际加载标注",
            "无法判定",
            loader["source"],
            "UNKNOWN",
            "按 datasets/rgbtpedes.py 的候选优先级未找到完整 RGB/T 数据目录和标注",
        )
    primary = _select_primary(probes, explicit, loader)
    if primary is None:
        _finding(findings, "主标注选择", "未找到", str(explicit or data_root), "UNKNOWN", "无法执行统计阶段")

    comparisons = _compare_annotations(probes, repo)
    payload: dict[str, Any] = {
        "generated_at": _now_iso(),
        "mode": "schema_only" if args.dump_schema else "full",
        "arguments": {
            "data_root": str(data_root),
            "dataset_name": args.dataset_name,
            "anno": str(explicit) if explicit else None,
            "out": str(out_dir),
            "repo": str(repo),
        },
        "discovery_notes": discovery_notes,
        "annotation_candidates": [_json_ready_probe(probe) for probe in probes],
        "primary_annotation": primary["source"] if primary else None,
        "loader_resolution": loader,
        "annotation_pairwise_comparisons": comparisons,
    }
    stats = _annotation_stats(primary, data_root, repo) if primary is not None else None
    train_ids = (
        stats.get("splits", {}).get("train", {}).get("unique_original_identities")
        if stats
        else None
    )
    static_loader = _loader_static_audit(repo, train_ids)
    dynamic_loader = _loader_dynamic_probe(repo, data_root, args.dataset_name)
    mraf_parameters = _static_mraf_parameters(repo)

    if stats:
        source = stats["source"]
        for split in ("train", "test"):
            row = stats["splits"].get(split)
            _finding(
                findings,
                f"{split} 统计",
                row if row is not None else "未找到",
                source,
                "OK" if row is not None else "UNKNOWN",
                "原始标注层与 split 内重编号层",
            )
        intersection = stats["train_test_identity_intersection"]
        _finding(
            findings,
            "train∩test 身份",
            intersection,
            source,
            "WARN" if intersection else "OK",
            "非空时列出全部交集身份",
        )
        test = stats["splits"].get("test")
        if test:
            equal = test["query_unique_identities"] == test["gallery_unique_identities"]
            _finding(
                findings,
                "test query/gallery 身份覆盖",
                {
                    "query": test["query_unique_identities"],
                    "gallery": test["gallery_unique_identities"],
                },
                source,
                "OK" if equal else "WARN",
                "不相等时差集见统计结构",
            )
        else:
            _finding(findings, "test query/gallery 身份覆盖", "未找到", source, "UNKNOWN", "test split 缺失")
        if not stats["modality_inventory_known"]:
            _finding(findings, "RGB/T 磁盘清点", "无法判定", source, "UNKNOWN", "未找到 RGB 或 T 目录")

    for fact in static_loader["facts"]:
        if fact["status"] == "UNKNOWN":
            _finding(findings, fact["item"], "未找到源码行", fact["source"], "UNKNOWN", "静态定位失败")
    if static_loader["num_classes"]["status"] == "UNKNOWN":
        _finding(findings, "num_classes", "无法判定", static_loader["num_classes"]["source"], "UNKNOWN", "表达式或训练身份数缺失")
    if dynamic_loader["status"] == "UNKNOWN":
        _finding(
            findings,
            "Dataset 对象 best-effort 内省",
            "无法判定",
            dynamic_loader["source"],
            "UNKNOWN",
            dynamic_loader["note"],
        )
    for row in mraf_parameters:
        if row["status"] == "UNKNOWN":
            _finding(findings, "model/mraf.py 参数", "无法判定", row["source"], "UNKNOWN", "静态解析失败")

    payload.update(
        {
            "statistics": stats,
            "loader_static_audit": static_loader,
            "loader_dynamic_probe": dynamic_loader,
            "mraf_parameters": mraf_parameters,
            "findings": [asdict(finding) for finding in findings],
            "unknown_items": [
                asdict(finding)
                for finding in findings
                if str(finding.status).upper() == "UNKNOWN"
            ],
        }
    )
    _write_dump(dump_path, payload)
    report_path = _render_report(
        repo=repo,
        dataset_name=args.dataset_name,
        dump_path=dump_path,
        probes=probes,
        primary=primary,
        stats=stats,
        comparisons=comparisons,
        loader_resolution=loader,
        static_loader=static_loader,
        dynamic_loader=dynamic_loader,
        mraf_parameters=mraf_parameters,
        findings=findings,
        discovery_notes=discovery_notes,
    )
    print(f"raw dump: {dump_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
