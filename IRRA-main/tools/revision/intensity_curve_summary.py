"""Summarize fixed RGB/IR noise and blur intensity curves."""

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.revision.score_io import (  # noqa: E402
    compute_retrieval_metrics,
    load_json_manifest,
    load_score_file,
    prepare_output_parents,
    report_paths,
    require_outputs_absent,
    require_score_consistency,
    resolve_manifest_path,
    write_json,
    write_markdown,
    write_tsv,
)


EXPECTED_SERIES = {
    "rgb_noise": (0.02, 0.05, 0.10, 0.20),
    "ir_noise": (0.02, 0.05, 0.10, 0.20),
    "rgb_blur": (3.0, 7.0, 11.0),
    "ir_blur": (3.0, 7.0, 11.0),
}


def sort_intensity_records(records):
    return sorted(
        records,
        key=lambda row: (str(row["series"]), float(row["intensity"])),
    )


def validate_intensity_records(records):
    observed = {}
    for record in records:
        series = str(record.get("series", ""))
        if series not in EXPECTED_SERIES:
            raise ValueError(
                f"Unsupported intensity series={series!r}; expected "
                f"{sorted(EXPECTED_SERIES)}."
            )
        intensity = float(record["intensity"])
        key = (series, intensity)
        if key in observed:
            raise ValueError(f"Duplicate intensity record: {series} {intensity}")
        observed[key] = record
    missing = []
    for series, values in EXPECTED_SERIES.items():
        for value in values:
            if (series, float(value)) not in observed:
                missing.append(f"{series}:{value:g}")
    if missing:
        raise ValueError(f"Intensity manifest is missing required records: {missing}")
    if len(observed) != sum(len(values) for values in EXPECTED_SERIES.values()):
        raise ValueError("Intensity manifest contains unsupported extra intensities.")
    return sort_intensity_records(list(observed.values()))


def load_records(manifest, manifest_path, default_matrix_key="sim"):
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list):
        raise TypeError("Intensity manifest requires a 'records' list.")
    raw_records = validate_intensity_records(raw_records)
    rows = []
    items = []
    for raw in raw_records:
        if not str(raw.get("score_file", "")).strip():
            raise ValueError(f"Intensity record requires score_file: {raw}")
        item = load_score_file(
            resolve_manifest_path(manifest_path, raw["score_file"]),
            matrix_key=str(raw.get("matrix_key", default_matrix_key)),
        )
        items.append(item)
        metrics = compute_retrieval_metrics(
            item["matrix"], item["txt_pids"], item["img_pids"]
        )
        rows.append(
            {
                "series": str(raw["series"]),
                "intensity": float(raw["intensity"]),
                "R1": metrics["R1"],
                "mAP": metrics["mAP"],
                "score_file": item["path"],
            }
        )
    if len(items) > 1:
        require_score_consistency(items)
    return sort_intensity_records(rows)


def _markdown(model_name, rows):
    lines = [
        f"# Intensity curves: {model_name}",
        "",
        "| Series | Numeric intensity | R1 | mAP |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['series']} | {row['intensity']:g} | "
            f"{row['R1']:.6f} | {row['mAP']:.6f} |"
        )
    return "\n".join(lines)


def _write_plots(rows, plot_dir):
    import matplotlib.pyplot as plt

    plot_dir = os.path.abspath(plot_dir)
    output_paths = []
    for series in sorted(EXPECTED_SERIES):
        series_rows = [row for row in rows if row["series"] == series]
        for metric in ("R1", "mAP"):
            output_paths.append(os.path.join(plot_dir, f"{series}_{metric}.png"))
    require_outputs_absent(output_paths)
    os.makedirs(plot_dir, exist_ok=True)
    for series in sorted(EXPECTED_SERIES):
        series_rows = [row for row in rows if row["series"] == series]
        x_values = [row["intensity"] for row in series_rows]
        for metric in ("R1", "mAP"):
            output_path = os.path.join(plot_dir, f"{series}_{metric}.png")
            plt.figure()
            plt.plot(x_values, [row[metric] for row in series_rows], marker="o")
            plt.xlabel("Noise std" if series.endswith("noise") else "Blur kernel")
            plt.ylabel(metric)
            plt.title(f"{series}: {metric}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=160)
            plt.close()
    return output_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize fixed RGB/IR noise and blur intensity curves."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--matrix-key", choices=["sim", "raw_sim"], default="sim")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--plot-dir",
        default="",
        help="optional directory for separate matplotlib figures (no subplots)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest, manifest_path = load_json_manifest(args.manifest)
    rows = load_records(manifest, manifest_path, default_matrix_key=args.matrix_key)
    model_name = str(manifest.get("model_name", ""))
    paths = report_paths(args.output_prefix)
    require_outputs_absent(paths.values())
    prepare_output_parents(paths.values())
    plots = _write_plots(rows, args.plot_dir) if args.plot_dir else []
    result = {
        "model_name": model_name,
        "manifest_file": manifest_path,
        "records": rows,
        "plots": plots,
    }
    write_json(paths["json"], result)
    write_tsv(paths["tsv"], rows, ["series", "intensity", "R1", "mAP", "score_file"])
    write_markdown(paths["markdown"], _markdown(model_name, rows))
    print(f"Saved intensity summary: {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
