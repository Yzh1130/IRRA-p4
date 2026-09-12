#!/usr/bin/env python3
"""Generate the locked formal or development-grid training configurations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (  # noqa: E402
    ConfigLoadError,
    HistoricalConfigSafeLoader,
    load_config_mapping,
)


ARM_NAMES = ("A", "B", "C", "D", "E", "F", "H1", "H2", "U")
E_TEMPERATURE_CANDIDATES = (0.2, 0.3, 0.5)
DEV_ARM_NAMES = ("devA", "devT010", "devT020", "devT030", "devT050")
SUPPLEMENTARY_ARM_NAMES = ("Dblur", "DnoNoise")
DEV_SPLIT_FILE = "splits/dev_190.json"
DEV_REQUIRED_INHERITED_FIELDS = (
    "MLM",
    "img_aug",
    "sampler",
    "num_instance",
    "batch_size",
    "lr",
    "optimizer",
    "loss_names",
    "img_size",
    "stride_size",
    "text_length",
    "pretrain_choice",
    "cmt_depth",
    "temperature",
)

FORMAL_COMMON_OVERRIDES: dict[str, Any] = {
    "num_epoch": 60,
    "batch_size": 64,
    "lr": 1e-5,
    "optimizer": "Adam",
    "loss_names": "sdm+mlm+id",
    "degrade_aug_mode": "none",
    "eval_period": 1,
    "save_last_checkpoint": True,
    "save_all_epoch_ckpt": False,
    "save_metric_best_ckpt": False,
    "save_model_only": True,
    "mdrc_lambda": 0.05,
    "mdrc_loss_scale": 100,
    "mdrc_prob": 0.5,
    "mdrc_warmup_epoch": 3,
    "mdrc_temp": 0.07,
    "mraf_hidden_dim": 128,
    "mraf_dropout": 0.1,
    "mraf_gate_temperature": 1.0,
    "mraf_gate_base_grad_scale": 1.0,
    "mraf_rel_warmup_epoch": 1,
    "mraf_balance_lambda": 0.0,
}

GATE_ENABLED_OVERRIDES: dict[str, Any] = {
    "mraf_scorer_lr_multiplier": 50.0,
    "mraf_gate_logit_scale": 5.0,
    "mraf_gate_fp32": True,
}

GATE_DISABLED_OVERRIDES: dict[str, Any] = {
    "mraf_scorer_lr_multiplier": 1.0,
    "mraf_gate_logit_scale": 1.0,
    "mraf_gate_fp32": False,
}

DEV_PROTOCOL_OVERRIDES: dict[str, Any] = {
    "num_epoch": 60,
    "eval_period": 1,
    "degrade_aug_mode": "none",
    "save_last_checkpoint": True,
    "save_all_epoch_ckpt": False,
    "save_metric_best_ckpt": False,
    "save_model_only": True,
    "training": True,
    "val_dataset": "val",
    "dev_split_file": DEV_SPLIT_FILE,
}

DEV_FIXED_OVERRIDES: dict[str, Any] = {
    "mdrc_lambda": 0.05,
    "mdrc_loss_scale": 100,
    "mdrc_prob": 0.5,
    "mdrc_warmup_epoch": 3,
    "mdrc_temp": 0.07,
    "mraf_hidden_dim": 128,
    "mraf_dropout": 0.1,
    "mraf_gate_temperature": 1.0,
    "mraf_gate_base_grad_scale": 1.0,
    "mraf_rel_warmup_epoch": 1,
    "mraf_balance_lambda": 0.0,
}

DEV_VARIANT_OVERRIDES: dict[str, dict[str, Any]] = {
    "devA": {
        "mdrc_enable": False,
        "mraf_enabled": False,
        "mraf_rel_lambda": 0.0,
        "mraf_target_temperature": 0.1,
        "mraf_equalize_modality_norm": False,
        "mraf_freeze_gate": False,
        "mraf_utility_prior": None,
    },
    "devT010": {
        "mdrc_enable": True,
        "mraf_enabled": True,
        "mraf_rel_lambda": 1.0,
        "mraf_target_temperature": 0.1,
        "mraf_equalize_modality_norm": False,
        "mraf_freeze_gate": False,
        "mraf_utility_prior": None,
    },
    "devT020": {
        "mdrc_enable": True,
        "mraf_enabled": True,
        "mraf_rel_lambda": 1.0,
        "mraf_target_temperature": 0.2,
        "mraf_equalize_modality_norm": False,
        "mraf_freeze_gate": False,
        "mraf_utility_prior": None,
    },
    "devT030": {
        "mdrc_enable": True,
        "mraf_enabled": True,
        "mraf_rel_lambda": 1.0,
        "mraf_target_temperature": 0.3,
        "mraf_equalize_modality_norm": False,
        "mraf_freeze_gate": False,
        "mraf_utility_prior": None,
    },
    "devT050": {
        "mdrc_enable": True,
        "mraf_enabled": True,
        "mraf_rel_lambda": 1.0,
        "mraf_target_temperature": 0.5,
        "mraf_equalize_modality_norm": False,
        "mraf_freeze_gate": False,
        "mraf_utility_prior": None,
    },
}

DEV_EXPERIMENT_FIELDS = (
    "mdrc_enable",
    "mraf_enabled",
    "mraf_rel_lambda",
    "mraf_target_temperature",
    "mraf_scorer_lr_multiplier",
    "mraf_gate_logit_scale",
    "mraf_gate_fp32",
    "mraf_equalize_modality_norm",
    "mraf_freeze_gate",
    "mraf_utility_prior",
)

DEV_GENERATOR_OWNED_FIELDS = frozenset(
    (
        "name",
        *DEV_PROTOCOL_OVERRIDES.keys(),
        *DEV_FIXED_OVERRIDES.keys(),
        *DEV_EXPERIMENT_FIELDS,
    )
)

DEV_VALIDATION_FIELDS = (
    *DEV_REQUIRED_INHERITED_FIELDS,
    "name",
    *DEV_PROTOCOL_OVERRIDES.keys(),
    *DEV_FIXED_OVERRIDES.keys(),
    *DEV_EXPERIMENT_FIELDS,
)


class ProtocolConfigError(ValueError):
    """Raised when locked protocol inputs or generated configs are invalid."""


class HistoricalConfigSafeDumper(yaml.SafeDumper):
    """Safe dumper preserving the tuple tag emitted by save_train_configs."""


def _represent_python_tuple(
    dumper: yaml.SafeDumper,
    value: tuple[Any, ...],
) -> yaml.nodes.SequenceNode:
    return dumper.represent_sequence(
        "tag:yaml.org,2002:python/tuple",
        value,
    )


HistoricalConfigSafeDumper.add_representer(tuple, _represent_python_tuple)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolConfigError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolConfigError(f"{label} must be a finite number.")
    return result


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolConfigError(f"{label} must be a JSON object.")
    return value


def load_frozen_hparams(path: Path) -> tuple[float, list[float]]:
    """Read the two dev-selected values without any numerical fallback."""
    frozen_path = path.expanduser().resolve()
    if not frozen_path.is_file():
        raise ProtocolConfigError(f"Frozen hyperparameter file not found: {frozen_path}")
    try:
        with frozen_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolConfigError(
            f"Failed to read frozen hyperparameters {frozen_path}: {exc}"
        ) from exc

    root = _require_mapping(payload, "FROZEN_HPARAMS.json top level")
    e_values = _require_mapping(root.get("E"), "FROZEN_HPARAMS.json['E']")
    u_values = _require_mapping(root.get("U"), "FROZEN_HPARAMS.json['U']")

    if "mraf_target_temperature" not in e_values:
        raise ProtocolConfigError(
            "FROZEN_HPARAMS.json['E']['mraf_target_temperature'] is required."
        )
    e_temperature = _finite_number(
        e_values["mraf_target_temperature"],
        "FROZEN_HPARAMS.json['E']['mraf_target_temperature']",
    )
    if e_temperature not in E_TEMPERATURE_CANDIDATES:
        raise ProtocolConfigError(
            "FROZEN_HPARAMS.json['E']['mraf_target_temperature'] must be one of "
            f"{list(E_TEMPERATURE_CANDIDATES)}."
        )

    if "mraf_utility_prior" not in u_values:
        raise ProtocolConfigError(
            "FROZEN_HPARAMS.json['U']['mraf_utility_prior'] is required."
        )
    prior_value = u_values["mraf_utility_prior"]
    if (
        isinstance(prior_value, (str, bytes))
        or not isinstance(prior_value, Sequence)
        or len(prior_value) != 2
    ):
        raise ProtocolConfigError(
            "FROZEN_HPARAMS.json['U']['mraf_utility_prior'] must contain exactly "
            "two positive finite numbers."
        )
    utility_prior = [
        _finite_number(item, f"FROZEN_HPARAMS.json['U']['mraf_utility_prior'][{index}]")
        for index, item in enumerate(prior_value)
    ]
    if any(item <= 0.0 for item in utility_prior):
        raise ProtocolConfigError(
            "FROZEN_HPARAMS.json['U']['mraf_utility_prior'] values must be > 0."
        )
    return e_temperature, utility_prior


def load_base_config(path: Path) -> dict[str, Any]:
    base_path = path.expanduser().resolve()
    try:
        return load_config_mapping(base_path)
    except ConfigLoadError as exc:
        raise ProtocolConfigError(str(exc)) from exc


def arm_overrides(
    e_temperature: float,
    utility_prior: list[float],
) -> dict[str, dict[str, Any]]:
    return {
        "A": {
            "mdrc_enable": False,
            "mraf_enabled": False,
            "mraf_rel_lambda": 0.0,
            "mraf_target_temperature": 0.1,
            "mraf_equalize_modality_norm": False,
            "mraf_freeze_gate": False,
            "mraf_utility_prior": None,
        },
        "B": {
            "mdrc_enable": True,
            "mraf_enabled": False,
            "mraf_rel_lambda": 0.0,
            "mraf_target_temperature": 0.1,
            "mraf_equalize_modality_norm": False,
            "mraf_freeze_gate": False,
            "mraf_utility_prior": None,
        },
        "C": {
            "mdrc_enable": False,
            "mraf_enabled": True,
            "mraf_rel_lambda": 1.0,
            "mraf_target_temperature": 0.1,
            "mraf_equalize_modality_norm": False,
            "mraf_freeze_gate": False,
            "mraf_utility_prior": None,
        },
        "D": {
            "mdrc_enable": True,
            "mraf_enabled": True,
            "mraf_rel_lambda": 1.0,
            "mraf_target_temperature": 0.1,
            "mraf_equalize_modality_norm": False,
            "mraf_freeze_gate": False,
            "mraf_utility_prior": None,
        },
        "E": {
            "mdrc_enable": True,
            "mraf_enabled": True,
            "mraf_rel_lambda": 1.0,
            "mraf_target_temperature": e_temperature,
            "mraf_equalize_modality_norm": False,
            "mraf_freeze_gate": False,
            "mraf_utility_prior": None,
        },
        "F": {
            "mdrc_enable": True,
            "mraf_enabled": True,
            "mraf_rel_lambda": 0.0,
            "mraf_target_temperature": 0.1,
            "mraf_equalize_modality_norm": False,
            "mraf_freeze_gate": False,
            "mraf_utility_prior": None,
        },
        "H1": {
            "mdrc_enable": True,
            "mraf_enabled": True,
            "mraf_rel_lambda": 0.0,
            "mraf_target_temperature": 0.1,
            "mraf_equalize_modality_norm": True,
            "mraf_freeze_gate": True,
            "mraf_utility_prior": None,
        },
        "H2": {
            "mdrc_enable": True,
            "mraf_enabled": True,
            "mraf_rel_lambda": 1.0,
            "mraf_target_temperature": 0.1,
            "mraf_equalize_modality_norm": True,
            "mraf_freeze_gate": False,
            "mraf_utility_prior": None,
        },
        "U": {
            "mdrc_enable": True,
            "mraf_enabled": True,
            "mraf_rel_lambda": 1.0,
            "mraf_target_temperature": 0.1,
            "mraf_equalize_modality_norm": False,
            "mraf_freeze_gate": False,
            "mraf_utility_prior": list(utility_prior),
        },
    }


def supplementary_arm_overrides(
    e_temperature: float,
    utility_prior: list[float],
) -> dict[str, dict[str, Any]]:
    return {
        "Dblur": dict(
            arm_overrides(e_temperature, utility_prior)["D"],
            mdrc_degrade_modes=[
                "rgb_drop",
                "ir_drop",
                "rgb_blur",
                "ir_blur",
                "rgb_attenuate",
                "ir_attenuate",
            ],
        ),
        "DnoNoise": dict(
            arm_overrides(e_temperature, utility_prior)["D"],
            mdrc_degrade_modes=[
                "rgb_drop",
                "ir_drop",
                "rgb_attenuate",
                "ir_attenuate",
            ],
        ),
    }


def build_configs(
    base_config: Mapping[str, Any],
    e_temperature: float,
    utility_prior: list[float],
) -> dict[str, dict[str, Any]]:
    per_arm = arm_overrides(e_temperature, utility_prior)
    configs: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        config = dict(base_config)
        config.update(FORMAL_COMMON_OVERRIDES)
        config.update(
            GATE_ENABLED_OVERRIDES
            if bool(per_arm[arm]["mraf_enabled"])
            else GATE_DISABLED_OVERRIDES
        )
        config.update(per_arm[arm])
        configs[arm] = config
    validate_configs(configs, e_temperature, utility_prior)
    return configs


def generate_supplementary_configs(
    base_config: Mapping[str, Any],
    e_temperature: float,
    utility_prior: list[float],
) -> dict[str, dict[str, Any]]:
    per_arm = supplementary_arm_overrides(e_temperature, utility_prior)
    configs: dict[str, dict[str, Any]] = {}
    for arm in SUPPLEMENTARY_ARM_NAMES:
        config = dict(base_config)
        config.update(FORMAL_COMMON_OVERRIDES)
        config.update(
            GATE_ENABLED_OVERRIDES
            if bool(per_arm[arm]["mraf_enabled"])
            else GATE_DISABLED_OVERRIDES
        )
        config.update(per_arm[arm])
        configs[arm] = config
    validate_supplementary_configs(
        configs,
        base_config,
        e_temperature,
        utility_prior,
    )
    return configs


def _validate_dev_base_config(base_config: Mapping[str, Any]) -> None:
    missing = [
        field
        for field in DEV_REQUIRED_INHERITED_FIELDS
        if field not in base_config
    ]
    if missing:
        raise ProtocolConfigError(
            "Base config is missing required inherited development-grid field(s): "
            + ", ".join(missing)
        )
    if base_config["MLM"] is not True:
        raise ProtocolConfigError(
            "Base config field MLM must be the boolean true for the development "
            f"grid, got {base_config['MLM']!r}."
        )


def _dev_generator_values(arm: str) -> dict[str, Any]:
    if arm not in DEV_VARIANT_OVERRIDES:
        raise ProtocolConfigError(f"Unknown development-grid arm: {arm!r}.")

    values = dict(DEV_PROTOCOL_OVERRIDES)
    values["name"] = arm
    values.update(DEV_FIXED_OVERRIDES)
    values.update(
        GATE_ENABLED_OVERRIDES
        if bool(DEV_VARIANT_OVERRIDES[arm]["mraf_enabled"])
        else GATE_DISABLED_OVERRIDES
    )
    values.update(DEV_VARIANT_OVERRIDES[arm])
    actual_owned_fields = frozenset(values)
    if actual_owned_fields != DEV_GENERATOR_OWNED_FIELDS:
        missing = sorted(DEV_GENERATOR_OWNED_FIELDS - actual_owned_fields)
        unexpected = sorted(actual_owned_fields - DEV_GENERATOR_OWNED_FIELDS)
        raise ProtocolConfigError(
            "Internal development-grid ownership mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )
    return values


def _dev_expected_values(
    base_config: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    _validate_dev_base_config(base_config)
    expected = {
        field: base_config[field]
        for field in DEV_REQUIRED_INHERITED_FIELDS
    }
    expected.update(_dev_generator_values(arm))
    return expected


def _assert_dev_inheritance(
    configs: Mapping[str, Mapping[str, Any]],
    base_config: Mapping[str, Any],
) -> None:
    """Require every non-owned generated value to equal the base value."""
    missing_marker = object()
    differences: list[tuple[str, Any, Any, Any]] = []
    for arm in DEV_ARM_NAMES:
        config = configs[arm]
        for key, base_value in base_config.items():
            if key in DEV_GENERATOR_OWNED_FIELDS:
                continue
            generated_value = config.get(key, missing_marker)
            if (
                generated_value is missing_marker
                or generated_value != base_value
            ):
                differences.append((arm, key, base_value, generated_value))
        for key, generated_value in config.items():
            if key not in base_config and key not in DEV_GENERATOR_OWNED_FIELDS:
                differences.append((arm, key, missing_marker, generated_value))

    if not differences:
        return

    difference_keys = sorted({str(item[1]) for item in differences})

    def render(value: Any) -> str:
        return "<missing>" if value is missing_marker else repr(value)

    details = "; ".join(
        f"{arm}.{key}: base={render(base_value)}, "
        f"generated={render(generated_value)}"
        for arm, key, base_value, generated_value in differences
    )
    raise ProtocolConfigError(
        "Development-grid inheritance mismatch for key(s): "
        f"{', '.join(difference_keys)}. Details: {details}"
    )


def build_dev_configs(
    base_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the five pre-freeze configurations without reading frozen values."""
    _validate_dev_base_config(base_config)
    configs: dict[str, dict[str, Any]] = {}
    for arm in DEV_ARM_NAMES:
        config = dict(base_config)
        config.update(_dev_generator_values(arm))
        configs[arm] = config
    validate_dev_configs(configs, base_config)
    return configs


def validate_dev_configs(
    configs: Mapping[str, Mapping[str, Any]],
    base_config: Mapping[str, Any],
) -> None:
    if tuple(configs) != DEV_ARM_NAMES:
        raise ProtocolConfigError(
            "Generated development-grid order must be "
            f"{list(DEV_ARM_NAMES)}, got {list(configs)}."
        )
    _validate_dev_base_config(base_config)
    _assert_dev_inheritance(configs, base_config)
    for arm in DEV_ARM_NAMES:
        config = configs[arm]
        expected_values = _dev_expected_values(base_config, arm)
        for key in DEV_VALIDATION_FIELDS:
            expected = expected_values[key]
            if key not in config or config[key] != expected:
                raise ProtocolConfigError(
                    f"{arm}.{key} must be {expected!r}, got {config.get(key)!r}."
                )


def validate_configs(
    configs: Mapping[str, Mapping[str, Any]],
    e_temperature: float,
    utility_prior: list[float],
) -> None:
    if tuple(configs) != ARM_NAMES:
        raise ProtocolConfigError(
            f"Generated arm order must be {list(ARM_NAMES)}, got {list(configs)}."
        )
    expected_arm_values = arm_overrides(e_temperature, utility_prior)
    for arm in ARM_NAMES:
        config = configs[arm]
        for key, expected in FORMAL_COMMON_OVERRIDES.items():
            if config.get(key) != expected:
                raise ProtocolConfigError(
                    f"{arm}.{key} must be {expected!r}, got {config.get(key)!r}."
                )
        expected_gate = (
            GATE_ENABLED_OVERRIDES
            if bool(expected_arm_values[arm]["mraf_enabled"])
            else GATE_DISABLED_OVERRIDES
        )
        for key, expected in expected_gate.items():
            if config.get(key) != expected:
                raise ProtocolConfigError(
                    f"{arm}.{key} must be {expected!r}, got {config.get(key)!r}."
                )
        for key, expected in expected_arm_values[arm].items():
            if config.get(key) != expected:
                raise ProtocolConfigError(
                    f"{arm}.{key} must be {expected!r}, got {config.get(key)!r}."
                )


def validate_supplementary_configs(
    configs: Mapping[str, Mapping[str, Any]],
    base_config: Mapping[str, Any],
    e_temperature: float,
    utility_prior: list[float],
) -> None:
    if tuple(configs) != SUPPLEMENTARY_ARM_NAMES:
        raise ProtocolConfigError(
            "Generated supplementary arm order must be "
            f"{list(SUPPLEMENTARY_ARM_NAMES)}, got {list(configs)}."
        )

    expected_modes_by_arm = {
        "Dblur": [
            "rgb_drop",
            "ir_drop",
            "rgb_blur",
            "ir_blur",
            "rgb_attenuate",
            "ir_attenuate",
        ],
        "DnoNoise": [
            "rgb_drop",
            "ir_drop",
            "rgb_attenuate",
            "ir_attenuate",
        ],
    }
    expected_d_without_modes = dict(
        build_configs(base_config, e_temperature, utility_prior)["D"]
    )
    expected_d_without_modes.pop("mdrc_degrade_modes", None)
    for arm in SUPPLEMENTARY_ARM_NAMES:
        expected_modes = expected_modes_by_arm[arm]
        supplementary_config = dict(configs[arm])
        if supplementary_config.get("mdrc_degrade_modes") != expected_modes:
            raise ProtocolConfigError(
                f"{arm}.mdrc_degrade_modes must be {expected_modes!r}, "
                f"got {supplementary_config.get('mdrc_degrade_modes')!r}."
            )

        supplementary_config.pop("mdrc_degrade_modes")
        if supplementary_config != expected_d_without_modes:
            differing_keys = sorted(
                key
                for key in supplementary_config.keys()
                | expected_d_without_modes.keys()
                if supplementary_config.get(key) != expected_d_without_modes.get(key)
            )
            raise ProtocolConfigError(
                f"{arm} must equal formal D except for mdrc_degrade_modes; "
                f"differing keys: {differing_keys}."
            )


def serialize_configs(
    configs: Mapping[str, Mapping[str, Any]],
    config_names: Sequence[str] = ARM_NAMES,
) -> dict[str, str]:
    serialized: dict[str, str] = {}
    for arm in config_names:
        try:
            text = yaml.dump(
                dict(configs[arm]),
                Dumper=HistoricalConfigSafeDumper,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
            loaded = yaml.load(text, Loader=HistoricalConfigSafeLoader)
        except yaml.YAMLError as exc:
            raise ProtocolConfigError(f"Failed to serialize {arm}.yaml: {exc}") from exc
        if loaded != dict(configs[arm]):
            raise ProtocolConfigError(f"In-memory YAML round-trip failed for {arm}.yaml.")
        serialized[arm] = text
    return serialized


def write_configs(
    out_dir: Path,
    serialized: Mapping[str, str],
    config_names: Sequence[str] = ARM_NAMES,
) -> None:
    output_dir = out_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    try:
        for arm in config_names:
            temp_path = output_dir / f".{arm}.yaml.tmp"
            with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized[arm])
            temporary_paths.append(temp_path)
        for arm in config_names:
            (output_dir / f".{arm}.yaml.tmp").replace(output_dir / f"{arm}.yaml")
    finally:
        for temp_path in temporary_paths:
            if temp_path.exists():
                temp_path.unlink()


def read_back_configs(
    out_dir: Path,
    config_names: Sequence[str] = ARM_NAMES,
) -> dict[str, dict[str, Any]]:
    output_dir = out_dir.expanduser().resolve()
    configs: dict[str, dict[str, Any]] = {}
    for arm in config_names:
        path = output_dir / f"{arm}.yaml"
        try:
            configs[arm] = load_config_mapping(path)
        except ConfigLoadError as exc:
            raise ProtocolConfigError(str(exc)) from exc
    return configs


def print_validation_table(configs: Mapping[str, Mapping[str, Any]]) -> None:
    print(
        "| arm | mdrc | mraf | rel_lambda | T_r | equalize | freeze | "
        "utility_prior | scorer_lr_multiplier | gate_logit_scale | gate_fp32 |"
    )
    print(
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |"
    )
    for arm in ARM_NAMES:
        config = configs[arm]
        utility = config["mraf_utility_prior"]
        utility_text = "None" if utility is None else json.dumps(utility)
        print(
            f"| {arm} | {config['mdrc_enable']} | {config['mraf_enabled']} | "
            f"{config['mraf_rel_lambda']} | {config['mraf_target_temperature']} | "
            f"{config['mraf_equalize_modality_norm']} | "
            f"{config['mraf_freeze_gate']} | {utility_text} | "
            f"{config['mraf_scorer_lr_multiplier']} | "
            f"{config['mraf_gate_logit_scale']} | {config['mraf_gate_fp32']} |"
        )


def print_supplementary_validation_table(
    configs: Mapping[str, Mapping[str, Any]],
) -> None:
    print(
        "| arm | mdrc | mraf | rel_lambda | T_r | equalize | freeze | "
        "utility_prior | scorer_lr_multiplier | gate_logit_scale | gate_fp32 |"
    )
    print(
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |"
    )
    for arm in SUPPLEMENTARY_ARM_NAMES:
        config = configs[arm]
        utility = config["mraf_utility_prior"]
        utility_text = "None" if utility is None else json.dumps(utility)
        print(
            f"| {arm} | {config['mdrc_enable']} | {config['mraf_enabled']} | "
            f"{config['mraf_rel_lambda']} | {config['mraf_target_temperature']} | "
            f"{config['mraf_equalize_modality_norm']} | "
            f"{config['mraf_freeze_gate']} | {utility_text} | "
            f"{config['mraf_scorer_lr_multiplier']} | "
            f"{config['mraf_gate_logit_scale']} | {config['mraf_gate_fp32']} |"
        )


def _table_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def print_dev_validation_table(
    configs: Mapping[str, Mapping[str, Any]],
    base_config: Mapping[str, Any],
) -> None:
    """Print every locked/inherited dev parameter after the YAML read-back."""
    print("| config | parameter | source | expected | read_back | status |")
    print("| --- | --- | --- | --- | --- | --- |")
    for arm in DEV_ARM_NAMES:
        expected_values = _dev_expected_values(base_config, arm)
        for key in DEV_VALIDATION_FIELDS:
            expected = expected_values[key]
            actual = configs[arm].get(key)
            status = "OK" if key in configs[arm] and actual == expected else "MISMATCH"
            source = (
                "生成器设定"
                if key in DEV_GENERATOR_OWNED_FIELDS
                else "继承自 base"
            )
            print(
                f"| {arm} | {key} | {source} | {_table_value(expected)} | "
                f"{_table_value(actual)} | {status} |"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate either locked A/B/C/D/E/F/H1/H2/U formal configs or the "
            "five pre-freeze development-grid configs from a saved base config."
        )
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="flat configs.yaml produced by save_train_configs",
    )
    parser.add_argument(
        "--frozen-hparams",
        type=Path,
        default=Path("FROZEN_HPARAMS.json"),
        help=(
            "committed JSON containing E.mraf_target_temperature and "
            "U.mraf_utility_prior; ignored and never read with --dev-grid"
        ),
    )
    parser.add_argument(
        "--dev-grid",
        default=False,
        action="store_true",
        help=(
            "generate devA/devT010/devT020/devT030/devT050 without reading "
            "FROZEN_HPARAMS.json"
        ),
    )
    parser.add_argument(
        "--supplementary",
        default=False,
        action="store_true",
        help="generate only the supplementary Dblur/DnoNoise configurations",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="output directory for the generated YAML files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dev_grid and args.supplementary:
        parser.error("--dev-grid and --supplementary cannot be used together")
    try:
        base_config = load_base_config(args.base_config)
        if args.dev_grid:
            configs = build_dev_configs(base_config)
            serialized = serialize_configs(configs, DEV_ARM_NAMES)
            write_configs(args.out_dir, serialized, DEV_ARM_NAMES)
            read_back = read_back_configs(args.out_dir, DEV_ARM_NAMES)
            validate_dev_configs(read_back, base_config)
        elif args.supplementary:
            e_temperature, utility_prior = load_frozen_hparams(args.frozen_hparams)
            configs = generate_supplementary_configs(
                base_config,
                e_temperature,
                utility_prior,
            )
            serialized = serialize_configs(configs, SUPPLEMENTARY_ARM_NAMES)
            write_configs(args.out_dir, serialized, SUPPLEMENTARY_ARM_NAMES)
            read_back = read_back_configs(args.out_dir, SUPPLEMENTARY_ARM_NAMES)
            validate_supplementary_configs(
                read_back,
                base_config,
                e_temperature,
                utility_prior,
            )
        else:
            e_temperature, utility_prior = load_frozen_hparams(args.frozen_hparams)
            configs = build_configs(base_config, e_temperature, utility_prior)
            serialized = serialize_configs(configs)
            write_configs(args.out_dir, serialized)
            read_back = read_back_configs(args.out_dir)
            validate_configs(read_back, e_temperature, utility_prior)
    except ProtocolConfigError as exc:
        parser.exit(2, f"error: {exc}\n")
    if args.dev_grid:
        print_dev_validation_table(read_back, base_config)
    elif args.supplementary:
        print_supplementary_validation_table(read_back)
    else:
        print_validation_table(read_back)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
