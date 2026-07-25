"""Safe loading helpers for current and historical training YAML configs."""

from __future__ import annotations

import os
import os.path as osp
from collections.abc import Mapping

import yaml


class HistoricalConfigSafeLoader(yaml.SafeLoader):
    """Safe YAML loader with the one legacy tag emitted by this project."""


def _construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node, deep=True))


HistoricalConfigSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)


class ConfigLoadError(ValueError):
    """Raised when a saved training configuration is absent or incompatible."""


def load_config_mapping(path):
    """Load a config mapping without enabling arbitrary Python constructors."""
    config_path = osp.abspath(osp.expanduser(os.fspath(path)))
    if not osp.isfile(config_path):
        raise ConfigLoadError(f"Config file not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=HistoricalConfigSafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigLoadError(
            f"Failed to load config {config_path}: {error}"
        ) from error
    if config is None:
        raise ConfigLoadError(f"Config file is empty: {config_path}")
    if not isinstance(config, Mapping):
        raise ConfigLoadError(
            f"Config top level must be a mapping: {config_path}; "
            f"got {type(config).__name__}"
        )
    return dict(config)
