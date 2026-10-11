"""Generate six measured training-state kappas from base_clean dev metrics.

R1 is a percentage, matching export_dev_records.py. The caller must supply
metrics produced by base_clean; a TSV alone cannot establish checkpoint
provenance. Missing states are errors, never substituted with P3 scenarios.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


STATE_SCENARIO_PAIRS = (
    ("rgb_drop", "rgb_drop_train"),
    ("ir_drop", "ir_drop_train"),
    ("rgb_noise", "rgb_noise_train"),
    ("ir_noise", "ir_noise_train"),
    ("rgb_attenuate", "rgb_attenuate_train"),
    ("ir_attenuate", "ir_attenuate_train"),
)


def build_kappa(metrics_tsv):
    """Require clean and six measured rows, retaining raw values and sources."""
    source = Path(metrics_tsv).expanduser().resolve(strict=True)
    required = {"clean", *(scenario for _, scenario in STATE_SCENARIO_PAIRS)}
    measured = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"scenario", "R1"}.issubset(reader.fieldnames):
            raise ValueError("metrics.tsv requires scenario and R1 columns")
        for row in reader:
            scenario = row.get("scenario")
            if scenario not in required:
                continue
            if scenario in measured:
                raise ValueError(f"Duplicate measured scenario: {scenario}")
            try:
                r1 = float(row["R1"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid R1 for {scenario}: {row.get('R1')!r}") from error
            if not math.isfinite(r1) or not 0.0 <= r1 <= 100.0:
                raise ValueError(f"R1 for {scenario} must be finite and in [0, 100]")
            measured[scenario] = r1
    missing = sorted(required.difference(measured))
    if missing:
        raise ValueError("Missing required measured scenarios: " + ", ".join(missing))
    clean_r1 = measured["clean"]
    if clean_r1 <= 0.0:
        raise ValueError("R1_clean must be positive")

    source_path = str(source)
    states = {}
    for state, scenario in STATE_SCENARIO_PAIRS:
        r1 = measured[scenario]
        raw = 1.0 - r1 / clean_r1
        states[state] = {
            "R1": r1,
            "raw": raw,
            "clipped": min(1.0, max(0.0, raw)),
            "scenario": scenario,
            "source_file": source_path,
        }
    return {
        "schema_version": 1,
        "R1_clean": clean_r1,
        "clean_scenario": "clean",
        "states": states,
        "source_metrics": source_path,
        "source_metrics_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_requirement": "base_clean checkpoint evaluated on the development split",
        "checkpoint_provenance_verified_by_this_tool": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_tsv", "--metrics", dest="metrics_tsv", required=True,
                        help="base_clean export metrics.tsv with clean and all six *_train rows")
    parser.add_argument("--output", required=True, help="new output kappa.json path")
    args = parser.parse_args(argv)
    result = build_kappa(args.metrics_tsv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"Wrote six measured training-state kappas: {output}")
    return result


if __name__ == "__main__":
    main()
