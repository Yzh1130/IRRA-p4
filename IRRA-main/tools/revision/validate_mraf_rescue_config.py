#!/usr/bin/env python3
"""Read-only validation for current and historical MRAF YAML configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ConfigLoadError, load_config_mapping  # noqa: E402


GATE_RESCUE_PROTOCOL: Mapping[str, Any] = {
    "batch_size": 64,
    "loss_names": "sdm+mlm+id",
    "MLM": True,
    "warmup_epochs": 5,
    "eval_use_fused_mu": True,
    "mdrc_loss_scale": 100,
    "mdrc_lambda": 0.05,
    "mdrc_prob": 0.5,
    "mdrc_warmup_epoch": 3,
}

REPORT_FIELDS = (
    "dataset_name",
    "num_epoch",
    "batch_size",
    "mraf_rel_lambda",
    "mraf_gate_fp32",
    "mraf_gate_base_grad_scale",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely validate a saved MRAF configs.yaml, including the historical "
            "!!python/tuple tag, without loading a model or accessing a GPU."
        )
    )
    parser.add_argument("--config", required=True, help="configs.yaml to validate")
    parser.add_argument(
        "--require-gate-rescue-protocol",
        action="store_true",
        help="also require the confirmed formal MDRC+MRAF protocol fields",
    )
    return parser


def _display(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _concise(error: BaseException) -> str:
    return " ".join(str(error).split())


def validate_protocol(config: Mapping[str, Any], config_path: Path) -> None:
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in GATE_RESCUE_PROTOCOL.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ConfigLoadError(
            f"Historical formal config mismatch: {config_path}: {mismatches}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().absolute()
    try:
        config = load_config_mapping(config_path)
        if args.require_gate_rescue_protocol:
            validate_protocol(config, config_path)
    except (ConfigLoadError, OSError) as error:
        print("config_valid=false", file=sys.stderr)
        print(f"config_path={config_path}", file=sys.stderr)
        print(f"config_error={_concise(error)}", file=sys.stderr)
        return 2

    print("config_valid=true")
    print("top_level_type=dict")
    print("tuple_tag_supported=true")
    print(f"config_path={config_path}")
    for field in REPORT_FIELDS:
        print(f"{field}={_display(config.get(field))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
