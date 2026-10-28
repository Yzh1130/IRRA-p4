"""Deterministic identity split; reads annotations, never samples or edits text."""
from pathlib import Path
import argparse
import hashlib
import sys
from datetime import datetime, timezone
CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
from tools.p4.r18.common import read_json, write_json, sha256

EXPECTED_DEV = [6,20,21,23,25,26,27,38,40,45,58,63,69,81,82,83,101,102,111,117,120,121,129,132,139,143,146,147,163,168]
RULE = 'Sort sha256(("RGBNT201-PEDES|dev|2026|%d" % id).encode("utf-8")).hexdigest(); first 30 identities; no RNG.'

def generate(annotation, output):
    annotation = Path(annotation)
    annos = read_json(annotation)
    official = sorted({int(a["id"]) for a in annos if a["split"] == "train"})
    if official != list(range(1,172)):
        raise ValueError("Expected official train IDs 1..171")
    ranked = sorted(official, key=lambda pid: hashlib.sha256(("RGBNT201-PEDES|dev|2026|%d" % pid).encode("utf-8")).hexdigest())
    dev = sorted(ranked[:30]); train = sorted(ranked[30:])
    if dev != EXPECTED_DEV:
        raise ValueError(f"Development identity list mismatch: {dev}")
    selected = [a for a in annos if a["split"] == "train" and int(a["id"]) in dev]
    reduced = [a for a in annos if a["split"] == "train" and int(a["id"]) in train]
    counts = [len(selected),sum(bool(c.strip()) for a in selected for c in a["captions"]),len(reduced),sum(bool(c.strip()) for a in reduced for c in a["captions"])]
    if counts != [682,681,3269,3267]:
        raise ValueError(f"Recorded partition sizes mismatch: {counts}")
    result = dict(anno_md5=hashlib.md5(annotation.read_bytes()).hexdigest(), dev_ids=dev, train_reduced_ids=train,
                  n_dev=30, n_train_reduced=141, dev_records=counts[0], dev_captions=counts[1], train_reduced_records=counts[2], train_reduced_captions=counts[3],
                  seed=2026, generated_at=datetime.now(timezone.utc).isoformat(), dataset="RGBNT201-PEDES", annotation_file=annotation.name,
                  annotation_sha256=sha256(annotation), rule=RULE)
    write_json(output,result)
    return result

if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotation',type=Path,default=CODE/'datasets/TextReID/RGBNT201/captionform0_8B.json')
    ap.add_argument('--output',type=Path,default=CODE/'splits/rgbnt201_dev30.json')
    cli=ap.parse_args(); print(generate(cli.annotation,cli.output))
