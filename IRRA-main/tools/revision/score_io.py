"""Strict CPU score loading, consistency checks, metrics and report helpers."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import torch


REQUIRED_ORDER_KEYS = (
    "txt_pids",
    "img_pids",
    "txt_indices",
    "img_indices",
    "order_hash",
)
METRIC_KEYS = ("R1", "R5", "R10", "mAP", "mINP")


class ScoreConsistencyError(ValueError):
    """Raised when score matrices cannot be compared or fused safely."""


def torch_load_cpu(path):
    path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Score file does not exist: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _require_tensor(payload, key, path):
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise TypeError(
            f"Score file {path} has invalid {key}={type(value).__name__}; "
            "expected a torch.Tensor."
        )
    return value.detach().cpu()


def load_score_file(path, matrix_key="sim"):
    """Load and validate one score payload entirely on CPU."""
    path = os.path.abspath(os.fspath(path))
    payload = torch_load_cpu(path)
    if not isinstance(payload, dict):
        raise TypeError(
            f"Score file {path} must contain a dict, got {type(payload).__name__}."
        )
    if matrix_key not in payload:
        raise KeyError(f"Score file {path} is missing matrix key {matrix_key!r}.")
    missing = [key for key in REQUIRED_ORDER_KEYS if key not in payload]
    if missing:
        raise KeyError(f"Score file {path} is missing required fields: {missing}")

    matrix = _require_tensor(payload, matrix_key, path)
    if matrix.dim() != 2:
        raise ValueError(
            f"Score file {path} has {matrix_key}.shape={tuple(matrix.shape)}; "
            "expected a 2D matrix."
        )
    if not matrix.is_floating_point():
        matrix = matrix.float()
    else:
        matrix = matrix.float()
    finite_mask = torch.isfinite(matrix)
    if not bool(finite_mask.all()):
        nan_count = int(torch.isnan(matrix).sum().item())
        inf_count = int(torch.isinf(matrix).sum().item())
        raise ValueError(
            f"Score file {path} matrix {matrix_key!r} contains non-finite values: "
            f"NaN={nan_count}, Inf={inf_count}."
        )

    txt_pids = _require_tensor(payload, "txt_pids", path).long().view(-1)
    img_pids = _require_tensor(payload, "img_pids", path).long().view(-1)
    txt_indices = _require_tensor(payload, "txt_indices", path).long().view(-1)
    img_indices = _require_tensor(payload, "img_indices", path).long().view(-1)
    rows, columns = matrix.shape
    expected_lengths = {
        "txt_pids": rows,
        "txt_indices": rows,
        "img_pids": columns,
        "img_indices": columns,
    }
    actual_tensors = {
        "txt_pids": txt_pids,
        "txt_indices": txt_indices,
        "img_pids": img_pids,
        "img_indices": img_indices,
    }
    for key, expected in expected_lengths.items():
        actual = actual_tensors[key].numel()
        if actual != expected:
            raise ValueError(
                f"Score file {path} has len({key})={actual}, but "
                f"{matrix_key}.shape={tuple(matrix.shape)} requires {expected}."
            )

    metrics = payload.get("metrics", {})
    metadata = payload.get("metadata", {})
    if metrics is None:
        metrics = {}
    if metadata is None:
        metadata = {}
    if not isinstance(metrics, dict):
        raise TypeError(f"Score file {path} metrics must be a dict.")
    if not isinstance(metadata, dict):
        raise TypeError(f"Score file {path} metadata must be a dict.")

    return {
        "path": path,
        "matrix_key": str(matrix_key),
        "matrix": matrix,
        "txt_pids": txt_pids,
        "img_pids": img_pids,
        "txt_indices": txt_indices,
        "img_indices": img_indices,
        "metrics": dict(metrics),
        "metadata": dict(metadata),
        "order_hash": str(payload["order_hash"]),
    }


def score_consistency_report(items):
    if len(items) < 2:
        raise ValueError("At least two score items are required for consistency checks.")
    reference = items[0]
    checks = {
        "score_shape": True,
        "txt_pids": True,
        "img_pids": True,
        "txt_indices": True,
        "img_indices": True,
        "order_hash": True,
    }
    mismatches = []
    for item in items[1:]:
        item_checks = {
            "score_shape": tuple(item["matrix"].shape)
            == tuple(reference["matrix"].shape),
            "txt_pids": torch.equal(item["txt_pids"], reference["txt_pids"]),
            "img_pids": torch.equal(item["img_pids"], reference["img_pids"]),
            "txt_indices": torch.equal(
                item["txt_indices"], reference["txt_indices"]
            ),
            "img_indices": torch.equal(
                item["img_indices"], reference["img_indices"]
            ),
            "order_hash": item["order_hash"] == reference["order_hash"],
        }
        for key, matched in item_checks.items():
            checks[key] = checks[key] and matched
            if not matched:
                mismatches.append(
                    {
                        "file": item["path"],
                        "reference_file": reference["path"],
                        "field": key,
                    }
                )
    return {
        **checks,
        "compatible": all(checks.values()),
        "mismatches": mismatches,
    }


def require_score_consistency(items):
    report = score_consistency_report(items)
    if not report["compatible"]:
        details = ", ".join(
            f"{entry['field']} in {entry['file']}" for entry in report["mismatches"]
        )
        raise ScoreConsistencyError(
            "Score files are not safe to fuse; no reordering or silent repair is "
            f"allowed. Mismatches: {details}"
        )
    return report


def row_zscore(matrix, epsilon=1e-6):
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and > 0, got {epsilon!r}.")
    matrix = matrix.float()
    row_mean = matrix.mean(dim=1, keepdim=True)
    row_std = matrix.std(dim=1, unbiased=False, keepdim=True)
    return (matrix - row_mean) / (row_std + epsilon)


def fuse_score_items(items, method="row_zscore_equal", epsilon=1e-6, weights=None):
    if not items:
        raise ValueError("At least one score item is required for fusion.")
    if len(items) > 1:
        require_score_consistency(items)
    method = str(method).lower()
    if method not in {"raw_equal", "row_zscore_equal"}:
        raise ValueError(
            f"Unsupported fusion method={method!r}; expected raw_equal or "
            "row_zscore_equal."
        )
    matrices = [item["matrix"].float() for item in items]
    if method == "row_zscore_equal":
        matrices = [row_zscore(matrix, epsilon=epsilon) for matrix in matrices]

    if weights is None:
        parsed_weights = [1.0] * len(items)
    else:
        parsed_weights = [float(value) for value in weights]
        if len(parsed_weights) != len(items):
            raise ValueError(
                f"weights length={len(parsed_weights)} does not match "
                f"model_count={len(items)}."
            )
    if not all(math.isfinite(value) and value >= 0.0 for value in parsed_weights):
        raise ValueError("Fusion weights must be finite and non-negative.")
    weight_sum = sum(parsed_weights)
    if weight_sum <= 0.0:
        raise ValueError("Fusion weights must contain at least one positive value.")
    normalized_weights = [value / weight_sum for value in parsed_weights]
    weight_tensor = torch.tensor(normalized_weights, dtype=torch.float32).view(-1, 1, 1)
    fused = (torch.stack(matrices, dim=0) * weight_tensor).sum(dim=0)
    return fused.float(), normalized_weights


def compute_retrieval_metrics(similarity, txt_pids, img_pids):
    similarity = similarity.detach().cpu().float()
    txt_pids = txt_pids.detach().cpu().long().view(-1)
    img_pids = img_pids.detach().cpu().long().view(-1)
    if similarity.dim() != 2:
        raise ValueError("similarity must be a 2D tensor.")
    if similarity.shape != (txt_pids.numel(), img_pids.numel()):
        raise ValueError(
            "similarity shape does not match txt_pids/img_pids lengths: "
            f"{tuple(similarity.shape)} vs ({txt_pids.numel()}, {img_pids.numel()})."
        )
    if similarity.shape[0] == 0 or similarity.shape[1] == 0:
        raise ValueError("similarity must contain at least one query and gallery item.")

    indices = torch.argsort(similarity, dim=1, descending=True)
    matches = img_pids[indices].eq(txt_pids.view(-1, 1))
    relevant_count = matches.sum(dim=1)
    if bool((relevant_count == 0).any()):
        bad_queries = (relevant_count == 0).nonzero(as_tuple=False).view(-1).tolist()
        raise ValueError(f"Queries without a positive gallery match: {bad_queries[:10]}")

    cumulative = matches.cumsum(dim=1)
    cmc = cumulative.clamp(max=1).float().mean(dim=0) * 100.0
    inp = [
        cumulative[index][row.nonzero()[-1]] / (row.nonzero()[-1] + 1.0)
        for index, row in enumerate(matches)
    ]
    minp = torch.cat(inp).mean() * 100.0
    precision_by_rank = [
        cumulative[:, index] / (index + 1.0)
        for index in range(cumulative.shape[1])
    ]
    precision_by_rank = torch.stack(precision_by_rank, dim=1) * matches
    average_precision = precision_by_rank.sum(dim=1) / relevant_count
    mean_average_precision = average_precision.mean() * 100.0

    def rank_value(rank):
        index = min(int(rank), int(cmc.numel())) - 1
        return float(cmc[index].item())

    return {
        "R1": rank_value(1),
        "R5": rank_value(5),
        "R10": rank_value(10),
        "mAP": float(mean_average_precision.item()),
        "mINP": float(minp.item()),
    }


def json_safe(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def report_paths(output_prefix, include_score=False, include_csv=False):
    prefix = os.path.abspath(os.fspath(output_prefix))
    paths = {
        "json": prefix + ".json",
        "tsv": prefix + ".tsv",
        "markdown": prefix + ".md",
    }
    if include_score:
        paths["score"] = prefix + ".pt"
    if include_csv:
        paths["csv"] = prefix + ".csv"
    return paths


def require_outputs_absent(paths):
    existing = [os.path.abspath(os.fspath(path)) for path in paths if os.path.exists(path)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing revision output(s): " + ", ".join(existing)
        )


def prepare_output_parents(paths):
    for path in paths:
        parent = os.path.dirname(os.path.abspath(os.fspath(path)))
        if parent:
            os.makedirs(parent, exist_ok=True)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_tsv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key, "")) for key in fieldnames})


def write_markdown(path, content):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content.rstrip() + "\n")


def load_json_manifest(path):
    path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Manifest {path} must contain a JSON object.")
    return payload, path


def resolve_manifest_path(manifest_path, value):
    value = os.fspath(value)
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(os.path.dirname(manifest_path), value))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate one score file on CPU and print its ordering summary."
    )
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--matrix-key", choices=["sim", "raw_sim"], default="sim")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    item = load_score_file(args.score_file, matrix_key=args.matrix_key)
    summary = {
        "path": item["path"],
        "matrix_key": item["matrix_key"],
        "shape": list(item["matrix"].shape),
        "txt_pids": item["txt_pids"].numel(),
        "img_pids": item["img_pids"].numel(),
        "txt_indices": item["txt_indices"].numel(),
        "img_indices": item["img_indices"].numel(),
        "order_hash": item["order_hash"],
        "metrics": item["metrics"],
        "metadata": item["metadata"],
    }
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
