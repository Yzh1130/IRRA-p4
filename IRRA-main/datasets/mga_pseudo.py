"""Utilities for structured MGA pseudo-caption JSONL files.

This module only reads, validates, and normalizes externally generated pseudo
caption records. It is intentionally not wired into the training dataset.
"""

import json
import os.path as op
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


SUPPORTED_PSEUDO_KINDS = {
    "full",
    "short",
    "long",
    "attribute",
    "rgb_color",
    "thermal_shape",
    "object",
    "negative",
}

TEXT_LIST_FIELDS = (
    "pseudo_texts",
    "rgb_color_texts",
    "thermal_shape_texts",
    "negative_texts",
)

IMPLIED_KIND_BY_FIELD = {
    "rgb_color_texts": "rgb_color",
    "thermal_shape_texts": "thermal_shape",
    "negative_texts": "negative",
}

SCORE_WEIGHTS = {
    "clip_score": 0.4,
    "caption_consensus": 0.3,
    "attr_confidence": 0.3,
}

CANONICAL_JSONL_SCHEMA = {
    "pid": "optional",
    "image_id": "preferred key when available",
    "rgb_path": "optional fallback key source",
    "thermal_path": "optional",
    "caption": "optional original caption",
    "caption_hash": "optional fallback key source",
    "pseudo_texts": [
        {
            "text": "required",
            "kind": sorted(SUPPORTED_PSEUDO_KINDS),
            "clip_score": "optional float",
            "caption_consensus": "optional float",
            "attr_confidence": "optional float",
            "weight": "optional float",
            "source": "optional string",
        }
    ],
    "rgb_color_texts": [{"text": "required", "weight": "optional float"}],
    "thermal_shape_texts": [{"text": "required", "weight": "optional float"}],
    "negative_texts": [{"text": "required", "weight": "optional float"}],
}


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    if out == float("inf") or out == float("-inf"):
        return None
    return out


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def normalize_rgb_path_key(rgb_path: Any) -> Optional[str]:
    """Return a stable basename key from an RGB path."""
    if _is_missing(rgb_path):
        return None
    normalized = op.normpath(str(rgb_path).strip())
    basename = op.basename(normalized)
    if basename == "":
        return None
    return basename


def make_mga_pseudo_key(record: Dict[str, Any]) -> Optional[str]:
    """Build the canonical lookup key for a raw or normalized record."""
    image_id = record.get("image_id", None)
    if not _is_missing(image_id):
        return str(image_id).strip()

    rgb_key = normalize_rgb_path_key(record.get("rgb_path", None))
    if rgb_key is not None:
        return rgb_key

    caption_hash = record.get("caption_hash", None)
    if not _is_missing(caption_hash):
        return str(caption_hash).strip()
    return None


def compute_pseudo_weight(item: Dict[str, Any]) -> float:
    """Compute a normalized pseudo-text confidence weight in [0, 1]."""
    if not isinstance(item, dict):
        return 1.0

    explicit_weight = _as_float(item.get("weight", None))
    if explicit_weight is not None:
        return _clamp01(explicit_weight)

    weighted_sum = 0.0
    weight_sum = 0.0
    for score_name, score_weight in SCORE_WEIGHTS.items():
        score = _as_float(item.get(score_name, None))
        if score is None:
            continue
        weighted_sum += float(score_weight) * _clamp01(score)
        weight_sum += float(score_weight)

    if weight_sum <= 0.0:
        return 1.0
    return _clamp01(weighted_sum / weight_sum)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _normalize_text_item(
    item: Any,
    field_name: str,
    min_weight: float,
) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None

    text = _safe_text(item.get("text", ""))
    if text == "":
        return None

    if field_name == "pseudo_texts":
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in SUPPORTED_PSEUDO_KINDS:
            return None
    else:
        kind = IMPLIED_KIND_BY_FIELD[field_name]

    weight = compute_pseudo_weight(item)
    if weight < float(min_weight):
        return None

    normalized = {
        "text": text,
        "kind": kind,
        "weight": weight,
    }
    for optional_key in (
        "clip_score",
        "caption_consensus",
        "attr_confidence",
    ):
        value = _as_float(item.get(optional_key, None))
        if value is not None:
            normalized[optional_key] = _clamp01(value)

    source = item.get("source", None)
    if not _is_missing(source):
        normalized["source"] = str(source)
    return normalized


def _normalize_text_list(
    raw_list: Any,
    field_name: str,
    min_weight: float,
) -> List[Dict[str, Any]]:
    if raw_list is None:
        return []
    if not isinstance(raw_list, list):
        return []
    normalized = []
    for item in raw_list:
        normalized_item = _normalize_text_item(
            item,
            field_name=field_name,
            min_weight=min_weight,
        )
        if normalized_item is not None:
            normalized.append(normalized_item)
    return normalized


def normalize_mga_pseudo_record(
    raw_record: Any,
    min_weight: float = 0.0,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Normalize one raw JSONL object into the canonical schema.

    Returns:
        ``(key, record)`` when the row has a usable lookup key and at least one
        valid candidate text; otherwise ``None``.
    """
    if not isinstance(raw_record, dict):
        return None

    key = make_mga_pseudo_key(raw_record)
    if key is None:
        return None

    normalized = {
        "key": key,
        "pid": raw_record.get("pid", None),
        "image_id": raw_record.get("image_id", None),
        "rgb_path": raw_record.get("rgb_path", None),
        "thermal_path": raw_record.get("thermal_path", None),
        "caption": raw_record.get("caption", None),
        "caption_hash": raw_record.get("caption_hash", None),
    }
    for field_name in TEXT_LIST_FIELDS:
        normalized[field_name] = _normalize_text_list(
            raw_record.get(field_name, []),
            field_name=field_name,
            min_weight=min_weight,
        )

    total_texts = sum(len(normalized[field]) for field in TEXT_LIST_FIELDS)
    if total_texts <= 0:
        return None
    return key, normalized


def _merge_record(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("pid", "image_id", "rgb_path", "thermal_path", "caption", "caption_hash"):
        if dst.get(key, None) is None and src.get(key, None) is not None:
            dst[key] = src[key]
    for field_name in TEXT_LIST_FIELDS:
        dst.setdefault(field_name, [])
        dst[field_name].extend(src.get(field_name, []))
    return dst


def load_mga_pseudo_json(path: str, min_weight: float = 0.0) -> Dict[str, Dict[str, Any]]:
    """Load a structured MGA pseudo-caption JSONL file.

    The returned dict is keyed by ``image_id`` when available, then RGB basename,
    then ``caption_hash``. Duplicate keys are merged by concatenating text lists.
    """
    if path is None or str(path).strip() == "":
        return {}
    json_path = str(path)
    if not op.isfile(json_path):
        raise RuntimeError(f"MGA pseudo JSONL file does not exist: {json_path}")

    records: Dict[str, Dict[str, Any]] = {}
    with open(json_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped == "":
                continue
            try:
                raw_record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in MGA pseudo file {json_path}:{line_no}: {exc}"
                ) from exc

            normalized = normalize_mga_pseudo_record(
                raw_record,
                min_weight=min_weight,
            )
            if normalized is None:
                continue
            key, record = normalized
            if key in records:
                _merge_record(records[key], record)
            else:
                records[key] = record
    return records


def select_weighted_text(
    text_list: Iterable[Dict[str, Any]],
    min_weight: float,
    rng: Optional[random.Random],
) -> Optional[str]:
    """Select one candidate text using normalized confidence weights."""
    if text_list is None:
        return None
    if rng is None:
        rng = random

    candidates = []
    for item in text_list:
        if not isinstance(item, dict):
            continue
        text = _safe_text(item.get("text", ""))
        if text == "":
            continue
        weight = compute_pseudo_weight(item)
        if weight < float(min_weight):
            continue
        candidates.append((text, weight))

    if not candidates:
        return None

    total_weight = sum(weight for _, weight in candidates)
    if total_weight <= 0.0:
        index = min(int(rng.random() * len(candidates)), len(candidates) - 1)
        return candidates[index][0]

    threshold = rng.random() * total_weight
    cumulative = 0.0
    for text, weight in candidates:
        cumulative += weight
        if cumulative >= threshold:
            return text
    return candidates[-1][0]


def collect_mga_pseudo_stats(records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Collect lightweight validation stats for normalized records."""
    counts = {field_name: 0 for field_name in TEXT_LIST_FIELDS}
    kind_counter = Counter()
    weights = []
    for record in records.values():
        for field_name in TEXT_LIST_FIELDS:
            items = record.get(field_name, [])
            counts[field_name] += len(items)
            for item in items:
                kind_counter[str(item.get("kind", "unknown"))] += 1
                weights.append(float(item.get("weight", 1.0)))

    if weights:
        weight_mean = sum(weights) / float(len(weights))
        weight_min = min(weights)
        weight_max = max(weights)
    else:
        weight_mean = 0.0
        weight_min = 0.0
        weight_max = 0.0

    return {
        "record_count": len(records),
        "counts": counts,
        "kind_distribution": dict(kind_counter),
        "weight_count": len(weights),
        "weight_mean": weight_mean,
        "weight_min": weight_min,
        "weight_max": weight_max,
    }
