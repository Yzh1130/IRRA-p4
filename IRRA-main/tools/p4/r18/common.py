"""Shared round-18 file/metadata operations; no data or model is fetched."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[3]
BASE = "41ef03cf6a5f9c5bd153bc9f00defa7d84acae4b"
DATASETS = ("RGBT-PEDES", "RGBNT201-PEDES")

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8", newline="\n")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def require_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Unsupported P4 dataset: {name!r}")
    return name

def dataset_class(name):
    require_dataset(name)
    if name == "RGBNT201-PEDES":
        from datasets.rgbnt201pedes import RGBNT201PEDES
        return RGBNT201PEDES
    from datasets.rgbtpedes import RGBTPEDES
    return RGBTPEDES

def nonblank_annotations(annos):
    """Keep every image, retaining only its existing nonblank caption strings."""
    return [dict(a, captions=[c for c in a["captions"] if c.strip()]) for a in annos]

def expected_training_ids(name, split_path=None):
    # Retain R13's original RGBT-PEDES contract, sourced from its frozen split,
    # rather than relaxing its 1076-ID check on a supplied synthetic partition.
    require_dataset(name)
    path = CODE / "splits/dev_190.json" if name == "RGBT-PEDES" else Path(split_path)
    return int(read_json(path)["n_train_reduced"])

def item_meta(annotation, evaluation, name, split):
    import re
    annos = read_json(annotation)
    rows = []
    for index in evaluation.get("gallery_annotation_indices", []):
        rows.append(annos[index])
    if not rows:
        raise ValueError("Item metadata requires verified gallery annotation indices")
    filenames = [Path(a["file_path"]).name for a in rows]
    ids = [int(a["id"]) for a in rows]
    if ids != evaluation["gallery_raw_pids"]:
        raise ValueError("Item metadata gallery raw PID order mismatch")
    cameras = []
    for filename in filenames:
        match = re.fullmatch(r"(\d+)_(cam\d+)_(\d+)_(\d+)\.jpg", filename)
        if name == "RGBNT201-PEDES" and match is None:
            raise ValueError(f"Invalid RGBNT201 camera filename: {filename}")
        cameras.append(match.group(2) if name == "RGBNT201-PEDES" else None)
    query_names = [Path(annos[int(q["query_id"].split(":")[0])]["file_path"]).name for q in evaluation["queries"]]
    return {"dataset_name": name, "split": split, "annotation_sha256": sha256(annotation),
            "gallery_filenames": filenames, "gallery_raw_pids": ids, "gallery_cameras": cameras,
            "query_filenames": query_names, "query_raw_pids": [q["pid"] for q in evaluation["queries"]]}

def protocol_prefix(directory):
    matches = list(Path(directory).glob("r12_*_selection.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one protocol run in {directory}, found {len(matches)}")
    return str(matches[0])[:-len("selection.json")]
