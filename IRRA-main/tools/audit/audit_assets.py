"""只读盘点论文实验资产，并生成 CSV 与 Markdown 报告。

本模块只读取传入的实验目录和 Git 元数据。它不会加载 checkpoint 或
score matrix，不会跟随目录符号链接；唯一写入位置是 ``--out`` 和仓库内的
``docs/audit``。
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import re
import statistics
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

if __package__:
    from .common import resolve_audit_output_dir, run_git
else:  # 兼容 ``python tools/audit/audit_assets.py ...``。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.audit.common import resolve_audit_output_dir, run_git  # type: ignore


_SKIP_DIR_NAMES = {
    ".git",
    ".cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "site-packages",
}
_DATA_IMAGE_DIR_NAMES = {"rgb", "thermal", "images"}
_CHECKPOINT_SUFFIXES = {".pth", ".ckpt", ".pt"}
_TEXT_SUFFIXES = {".txt", ".log", ".out", ".yaml", ".yml", ".json", ".csv", ".tsv"}
_METRIC_SUFFIXES = {".json", ".csv", ".tsv", ".txt", ".log"}
_SCORE_SUFFIXES = {".pt", ".pth", ".npy", ".npz", ".csv", ".tsv", ".json"}
_MAX_STRUCTURED_TEXT_BYTES = 20 * 1024 * 1024
_HEAD_TEXT_CHARS = 1024 * 1024
_TAIL_TEXT_CHARS = 2 * 1024 * 1024
_SHA_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{7,40})(?![0-9a-f])", re.IGNORECASE)
_LOGGER_TIME_RE = re.compile(
    r"(?<!\d)(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[,.]\d{1,6})?(?:Z|[+-]\d{2}:?\d{2})?)"
)
_ISO_TIME_RE = re.compile(
    r"(?<!\d)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)
_NORMAL_END_PATTERNS = (
    re.compile(r"\bbest\s+R1\s*:", re.IGNORECASE),
    re.compile(r"\btraining\s+(?:is\s+)?(?:complete|completed|finished)\b", re.IGNORECASE),
    re.compile(r"\bevaluation\s+(?:is\s+)?(?:complete|completed|finished)\b", re.IGNORECASE),
    re.compile(r"\bEpoch\s+\d+\s+done\b", re.IGNORECASE),
    re.compile(r"\bValidation\s+Results\s*-\s*Epoch\b", re.IGNORECASE),
    re.compile(r"\bSTATUS\s*=\s*COMPLETED\b", re.IGNORECASE),
    re.compile(r'"status"\s*:\s*"COMPLETED"', re.IGNORECASE),
)
_CLEAN_SCENARIOS = {"", "none", "clean", "baseline", "original", "normal", "undegraded"}


@dataclass(frozen=True)
class FileMeta:
    path: Path
    size: int
    mtime: float
    is_symlink: bool
    kinds: frozenset[str]


@dataclass
class ScanResult:
    root: Path
    files: list[FileMeta] = field(default_factory=list)
    direct_sizes: dict[Path, int] = field(default_factory=lambda: defaultdict(int))
    child_dirs: dict[Path, set[Path]] = field(default_factory=lambda: defaultdict(set))
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MetricRecord:
    scenario: str
    r1: Optional[float]
    map_value: Optional[float]
    minp: Optional[float]
    source: str
    mtime: float
    priority: int


@dataclass(frozen=True)
class ValueMatch:
    value: Any
    actual_key: str
    source: str
    priority: int


def _absolute(path: Path) -> Path:
    """返回不要求目标存在的绝对路径，不解析目录符号链接。"""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(os.fspath(_absolute(path))))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        _absolute(path).relative_to(_absolute(parent))
        return True
    except ValueError:
        return False


def _normalise_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.replace("λ", "lambda").replace("τ", "tau")
    return re.sub(r"[^a-z0-9]+", "", text)


def _safe_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}" if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _format_bytes(size: int) -> str:
    value = float(max(0, size))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    return f"{value:.2f} {unit}"


def _format_mtime(timestamp: float) -> str:
    try:
        return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return ""


def _classify_file(path: Path) -> frozenset[str]:
    name = path.name.lower()
    stem = path.stem.lower()
    suffix = path.suffix.lower()
    kinds: set[str] = set()

    under_scores_dir = any(
        part.casefold() == "scores" for part in path.parts[:-1]
    )
    score_like = suffix in _SCORE_SUFFIXES and (
        "score" in stem or under_scores_dir
    )
    if score_like:
        kinds.add("scores")

    if suffix in _CHECKPOINT_SUFFIXES and not score_like:
        if not any(token in stem for token in ("diagnostic", "feature", "sample_level")):
            kinds.add("checkpoint")

    exact_configs = {
        "config.yaml",
        "configs.yaml",
        "config.yml",
        "configs.yml",
        "args.yaml",
        "args.yml",
        "args.json",
        "hparams.yaml",
        "hparams.yml",
        "hyperparameters.yaml",
        "unit.json",
        "candidate.json",
    }
    if name in exact_configs or (
        suffix in {".yaml", ".yml", ".json"}
        and any(token in stem for token in ("config", "hparam", "hyperparam"))
    ):
        kinds.add("config")

    if suffix == ".log" or name in {
        "train_log.txt",
        "test_log.txt",
        "train.log",
        "eval.log",
        "driver.log",
    } or stem.endswith("_log"):
        kinds.add("log")

    metric_tokens = ("metric", "result", "summary", "robustness", "evaluation")
    if suffix in _METRIC_SUFFIXES and any(token in stem for token in metric_tokens):
        kinds.add("metrics")
    if name in {"diagnosis.json", "checkpoint_selection.tsv", "mraf_epoch_metrics.tsv"}:
        kinds.add("metrics")

    if "gate" in stem and suffix in {".csv", ".tsv", ".json", ".npy", ".npz", ".pt"}:
        kinds.add("gate")
    if "drift" in stem and suffix in {".csv", ".tsv", ".json", ".npy", ".npz", ".pt"}:
        kinds.add("drift")

    metadata_names = {
        "command.txt",
        "launch_metadata.txt",
        "manifest.txt",
        "status.txt",
        "commit.txt",
        "git_commit.txt",
        "git_head.txt",
        "version.txt",
    }
    if name in metadata_names:
        kinds.add("metadata")

    return frozenset(kinds)


def _looks_like_dataset_image_dir(path: Path) -> bool:
    name = path.name.lower()
    if name not in _DATA_IMAGE_DIR_NAMES and name != "t":
        return False
    parent_name = path.parent.name.lower()
    return (
        "pedes" in parent_name
        or "rgbt" in parent_name
        or "rgbnt" in parent_name
        or name in {"rgb", "thermal", "images"}
    )


def _scan_root(root: Path) -> ScanResult:
    result = ScanResult(root=_absolute(root))
    if not result.root.exists():
        result.warnings.append(f"根目录未找到：{result.root}")
        return result
    if result.root.is_symlink():
        result.warnings.append(f"根目录是符号链接，按只读安全规则未跟随：{result.root}")
        return result
    if not result.root.is_dir():
        result.warnings.append(f"根路径不是目录：{result.root}")
        return result

    stack = [result.root]
    seen_dirs: set[str] = set()
    while stack:
        directory = stack.pop()
        key = _path_key(directory)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    child = Path(entry.path)
                    try:
                        if entry.is_symlink():
                            # 文件链接作为资产入口记录；目录链接绝不递归。
                            kinds = _classify_file(child)
                            if kinds:
                                stat_result = entry.stat(follow_symlinks=False)
                                result.files.append(
                                    FileMeta(
                                        path=child,
                                        size=int(stat_result.st_size),
                                        mtime=float(stat_result.st_mtime),
                                        is_symlink=True,
                                        kinds=kinds,
                                    )
                                )
                                result.direct_sizes[directory] += int(stat_result.st_size)
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            result.child_dirs[directory].add(child)
                            if entry.name in _SKIP_DIR_NAMES or _looks_like_dataset_image_dir(child):
                                continue
                            stack.append(child)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        stat_result = entry.stat(follow_symlinks=False)
                        result.direct_sizes[directory] += int(stat_result.st_size)
                        kinds = _classify_file(child)
                        if kinds:
                            result.files.append(
                                FileMeta(
                                    path=child,
                                    size=int(stat_result.st_size),
                                    mtime=float(stat_result.st_mtime),
                                    is_symlink=False,
                                    kinds=kinds,
                                )
                            )
                    except (OSError, ValueError) as exc:
                        result.warnings.append(f"无法读取目录项 {child}：{exc}")
        except (OSError, ValueError) as exc:
            result.warnings.append(f"无法遍历目录 {directory}：{exc}")
    return result


def _subtree_sizes(scan: ScanResult) -> dict[Path, int]:
    sizes: dict[Path, int] = defaultdict(int)
    for directory, size in scan.direct_sizes.items():
        sizes[directory] += size
    directories = set(scan.child_dirs)
    for children in scan.child_dirs.values():
        directories.update(children)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        parent = directory.parent
        if _is_relative_to(parent, scan.root):
            sizes[parent] += sizes.get(directory, 0)
    return dict(sizes)


def _is_primary_config(meta: FileMeta) -> bool:
    return meta.path.name.lower() in {
        "configs.yaml",
        "config.yaml",
        "configs.yml",
        "config.yml",
        "args.yaml",
        "args.yml",
        "args.json",
        "hparams.yaml",
        "hparams.yml",
        "unit.json",
        "candidate.json",
    }


def _dataset_parent(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ("pedes", "reid", "rgbnt", "cuhk", "icfg", "rstp"))


def _logical_wrapper(leaf: Path) -> Path:
    """折叠 ``wrapper/[train_output/]RGBT-PEDES/<leaf>`` 两层输出。"""

    parent = leaf.parent
    if _dataset_parent(parent):
        before_dataset = parent.parent
        if before_dataset.name.lower() == "train_output":
            return before_dataset.parent
        return before_dataset
    if parent.name.lower() in {"train_output", "output", "outputs"}:
        return parent.parent
    return leaf


def _discover_runs(scans: Sequence[ScanResult]) -> tuple[list[Path], dict[Path, set[Path]], list[str]]:
    direct: dict[Path, list[FileMeta]] = defaultdict(list)
    all_files: list[FileMeta] = []
    for scan in scans:
        all_files.extend(scan.files)
        for meta in scan.files:
            direct[meta.path.parent].append(meta)

    anchor_dirs: set[Path] = set()
    leaf_to_wrapper: dict[Path, Path] = {}
    for directory, metas in direct.items():
        has_checkpoint = any("checkpoint" in meta.kinds for meta in metas)
        has_primary_config = any(_is_primary_config(meta) for meta in metas)
        if has_checkpoint or has_primary_config:
            wrapper = _logical_wrapper(directory)
            leaf_to_wrapper[directory] = wrapper
            anchor_dirs.add(wrapper)

    # 仅日志/指标的残缺运行也应进入审计，但汇总容器不能因自身 summary.log
    # 而覆盖其下多个真实运行。
    for directory, metas in direct.items():
        if directory in leaf_to_wrapper:
            continue
        has_run_signal = any(
            bool(meta.kinds.intersection({"log", "metrics", "scores"})) for meta in metas
        )
        if not has_run_signal:
            continue
        descendants = [anchor for anchor in anchor_dirs if anchor != directory and _is_relative_to(anchor, directory)]
        ancestors = [anchor for anchor in anchor_dirs if _is_relative_to(directory, anchor)]
        if ancestors or descendants:
            continue
        anchor_dirs.add(directory)
        leaf_to_wrapper[directory] = directory

    # 若父 anchor 只是由内层训练叶目录折叠得到，则直接 config/checkpoint
    # anchor 与它合并；保留真正嵌套的 unit，因为其 unit.json 表示独立评估配置。
    wrappers = sorted(anchor_dirs, key=lambda item: (len(item.parts), str(item).lower()))
    merged_leafs: dict[Path, set[Path]] = defaultdict(set)
    for leaf, wrapper in leaf_to_wrapper.items():
        merged_leafs[wrapper].add(leaf)
    warnings: list[str] = []
    if not wrappers:
        warnings.append("未识别到包含 config、checkpoint、log、metrics 或 score 的实验运行目录")
    return wrappers, merged_leafs, warnings


def _assign_files(wrappers: Sequence[Path], files: Sequence[FileMeta]) -> dict[Path, list[FileMeta]]:
    assignments: dict[Path, list[FileMeta]] = defaultdict(list)
    by_depth = sorted(wrappers, key=lambda item: len(item.parts), reverse=True)
    for meta in files:
        for wrapper in by_depth:
            if _is_relative_to(meta.path, wrapper):
                assignments[wrapper].append(meta)
                break
    for metas in assignments.values():
        metas.sort(key=lambda item: (str(item.path).lower(), item.mtime))
    return assignments


def _read_small_text(path: Path, limit: int = _MAX_STRUCTURED_TEXT_BYTES) -> tuple[Optional[str], Optional[str]]:
    try:
        if path.stat().st_size > limit:
            return None, f"文件超过 {limit} 字节，未读取内容"
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return handle.read(), None
    except (OSError, UnicodeError) as exc:
        return None, str(exc)


def _read_head_tail(path: Path) -> tuple[list[tuple[str, str]], Optional[str]]:
    """读取日志开头和末尾窗口；标签用于构造可追溯 source。"""

    chunks: list[tuple[str, str]] = []
    try:
        size = path.stat().st_size
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            head = handle.read(_HEAD_TEXT_CHARS)
            chunks.append(("全文" if size <= _HEAD_TEXT_CHARS else "开头", head))
            if size > _HEAD_TEXT_CHARS:
                # TextIOWrapper 接受从文件起点计算的 cookie；errors=replace 使落在
                # 多字节字符中间时也能安全降级。
                handle.seek(max(0, size - _TAIL_TEXT_CHARS))
                tail = handle.read(_TAIL_TEXT_CHARS)
                chunks.append(("末尾", tail))
        return chunks, None
    except (OSError, UnicodeError) as exc:
        return chunks, str(exc)


def _scalar_from_text(value: str) -> Any:
    stripped = value.strip().strip(",")
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        stripped = stripped[1:-1]
    lowered = stripped.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null", "~"}:
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        value_float = float(stripped)
        return value_float if math.isfinite(value_float) else stripped
    except ValueError:
        return stripped


def _flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for raw_key, value in mapping.items():
        key = str(raw_key)
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            yield from _flatten_mapping(value, full_key)
        else:
            yield full_key, value


def _parse_simple_mapping(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([\w.\-/λτ]+)\s*[:=]\s*(.*?)\s*$", line, re.UNICODE)
        if match:
            result[match.group(1)] = _scalar_from_text(match.group(2))
    return result


def _parse_cli_flags(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pattern = re.compile(
        r"--([A-Za-z0-9_-]+)(?:\s*=\s*|\s+)(?:'([^']*)'|\"([^\"]*)\"|([^\s\\]+))"
    )
    for match in pattern.finditer(text.replace("\\\n", " ")):
        raw_value = next((group for group in match.groups()[1:] if group is not None), "")
        result[match.group(1)] = _scalar_from_text(raw_value)
    for flag in re.findall(r"(?:^|\s)--([A-Za-z0-9_-]+)(?=\s|$)", text):
        result.setdefault(flag, True)
    return result


def _mapping_from_file(meta: FileMeta) -> tuple[dict[str, Any], Optional[str]]:
    text, error = _read_small_text(meta.path)
    if text is None:
        return {}, error
    name = meta.path.name.lower()
    mapping: Any = None
    if meta.path.suffix.lower() == ".json":
        try:
            mapping = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            error = str(exc)
    elif meta.path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            mapping = yaml.safe_load(text)
        except Exception as exc:  # PyYAML 缺失或历史 tuple tag 均回退到逐行解析。
            error = str(exc)
    if isinstance(mapping, Mapping):
        result = dict(_flatten_mapping(mapping))
    else:
        result = _parse_simple_mapping(text)
    if name in {"command.txt", "launch_metadata.txt"} or "training_command" in result:
        command_text = str(result.get("training_command", text))
        for key, value in _parse_cli_flags(command_text).items():
            result[f"cli.{key}"] = value
    return result, error


def _config_priority(path: Path) -> int:
    name = path.name.lower()
    if name == "unit.json":
        return 80
    if name == "candidate.json":
        return 70
    if name in {"command.txt", "launch_metadata.txt"}:
        return 90
    if name in {"configs.yaml", "config.yaml", "configs.yml", "config.yml"}:
        return 60
    if name.startswith("args") or "hparam" in name:
        return 50
    return 20


def _collect_config_values(metas: Sequence[FileMeta]) -> tuple[list[ValueMatch], list[str]]:
    values: list[ValueMatch] = []
    warnings: list[str] = []
    candidates = [
        meta
        for meta in metas
        if "config" in meta.kinds or "metadata" in meta.kinds
    ]
    for meta in candidates:
        mapping, error = _mapping_from_file(meta)
        if error and not mapping:
            warnings.append(f"配置/元数据无法解析 {meta.path}：{error}")
        priority = _config_priority(meta.path)
        for key, value in mapping.items():
            values.append(
                ValueMatch(
                    value=value,
                    actual_key=key,
                    source=f"{meta.path}:{key}",
                    priority=priority,
                )
            )
    return values, warnings


_ALIASES: dict[str, tuple[str, ...]] = {
    "target_temperature": (
        "Tr",
        "target_temperature",
        "mraf_target_temperature",
        "reliability_target_temperature",
    ),
    "tau_s": (
        "tau_s",
        "τs",
        "mraf_gate_logit_scale",
        "gate_logit_scale",
        "gate_scale",
    ),
    "lambda_rel": (
        "lambda_rel",
        "λrel",
        "mraf_rel_lambda",
        "rel_lambda",
        "reliability_lambda",
    ),
    "scorer_lr_multiplier": (
        "mraf_scorer_lr_multiplier",
        "scorer_lr_multiplier",
        "scorer_lr_mult",
        "gate_lr_multiplier",
    ),
    "mdrc_lambda_max": (
        "mdrc_lambda_max",
        "mdrc_max_lambda",
        "mdrc_lambda",
        "lambda_mdrc",
        "lambda_max",
        "λmax",
    ),
    "mdrc_scale": (
        "mdrc_loss_scale",
        "mdrc_scale",
        "mdrc_scale_factor",
        "mdrc_coefficient",
        "mdrc_coef",
    ),
    "mdrc_probability": (
        "mdrc_prob",
        "mdrc_probability",
        "mdrc_enable_prob",
        "mdrc_degrade_prob",
        "mdrc_apply_prob",
    ),
    "mdrc_warmup": (
        "mdrc_warmup",
        "mdrc_warmup_epoch",
        "mdrc_warmup_epochs",
        "warmup_mdrc_epochs",
    ),
    "batch_size": ("batch_size", "train_batch_size", "ims_per_batch"),
    "optimizer": ("optimizer", "optim", "optimizer_name"),
    "learning_rate": ("learning_rate", "lr", "base_lr"),
    "epochs": ("num_epoch", "num_epochs", "epochs", "max_epoch", "max_epochs"),
    "image_size": ("img_size", "image_size", "input_size", "resolution"),
    "text_length": ("text_length", "max_text_length", "max_length", "context_length"),
    "seed": ("seed", "random_seed", "torch_seed"),
    "config_label": ("case", "case_name", "candidate", "label", "config_name", "experiment", "name"),
}


def _resolve_value(values: Sequence[ValueMatch], canonical: str) -> Optional[ValueMatch]:
    aliases = {_normalise_key(alias) for alias in _ALIASES[canonical]}
    matches = []
    for item in values:
        leaf_key = item.actual_key.rsplit(".", 1)[-1]
        if _normalise_key(leaf_key) in aliases or _normalise_key(item.actual_key) in aliases:
            matches.append(item)
    if not matches:
        return None
    return max(matches, key=lambda item: (item.priority, len(item.actual_key)))


def _infer_seed(wrapper: Path, values: Sequence[ValueMatch]) -> ValueMatch:
    match = _resolve_value(values, "seed")
    if match is not None:
        return match
    for part in reversed(wrapper.parts):
        found = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:[_-]|$)", part, re.IGNORECASE)
        if found:
            return ValueMatch(found.group(1), "path:seed", f"{wrapper}:目录名", 0)
    return ValueMatch("无法判定", "", f"{wrapper}:未找到 seed 字段或目录标记", 0)


def _infer_label(wrapper: Path, values: Sequence[ValueMatch]) -> ValueMatch:
    seed_suffix_pattern = re.compile(
        r"^(?P<label>.+?)[_-]seed[_-]?\d+$", re.IGNORECASE
    )

    match = _resolve_value(values, "config_label")
    if match is not None and str(match.value).strip():
        raw_label = str(match.value).strip()
        found = seed_suffix_pattern.fullmatch(raw_label)
        if found:
            label = re.sub(r"^\d{8}_\d{6}_?", "", found.group("label"))
            if label:
                return ValueMatch(
                    label,
                    match.actual_key,
                    f"{match.source}; 配置值 _seed 后缀归一化",
                    match.priority,
                )
        return match

    for part in reversed(wrapper.parts):
        found = seed_suffix_pattern.fullmatch(part)
        if found:
            label = re.sub(r"^\d{8}_\d{6}_?", "", found.group("label"))
            if label:
                return ValueMatch(
                    label,
                    "path:config_label",
                    f"{wrapper}:目录名 _seed 后缀推断",
                    0,
                )
    label_pattern = re.compile(r"(?:^|[_\-.])(H1|H2|A|B|C|D|E|F)(?:[_\-.]|$)", re.IGNORECASE)
    for part in reversed(wrapper.parts):
        found = label_pattern.search(part)
        if found:
            return ValueMatch(found.group(1).upper(), "path:config_label", f"{wrapper}:目录名", 0)
    cleaned = re.sub(r"^\d{8}_\d{6}_?", "", wrapper.name)
    cleaned = re.sub(r"(?:^|[_-])seed[_-]?\d+(?:[_-]|$)", "_", cleaned, flags=re.IGNORECASE).strip("_-")
    value = cleaned or "其它/无法判定"
    return ValueMatch(value, "path:directory_name", f"{wrapper}:目录名推断", 0)


def _text_header_flags(meta: FileMeta) -> tuple[bool, bool, str]:
    if meta.path.suffix.lower() not in _TEXT_SUFFIXES or "scores" in meta.kinds:
        return False, False, ""
    text, error = _read_small_text(meta.path, limit=256 * 1024)
    if text is None:
        return False, False, f"{meta.path}:无法读取表头（{error}）"
    sample = "\n".join(text.splitlines()[:5]).lower()
    has_gate = bool(re.search(r"(?:^|[^a-z])(gate|alpha_(?:rgb|ir)|mraf_logits?)", sample))
    has_drift = "drift_rgb" in sample or "drift_ir" in sample or "drift_" in sample
    return has_gate, has_drift, f"{meta.path}:前5行字段"


def _parse_datetime(raw: str) -> Optional[datetime]:
    text = raw.strip().replace(",", ".")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if re.search(r"[+-]\d{4}$", text):
        text = text[:-5] + text[-5:-2] + ":" + text[-2:]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _log_facts(logs: Sequence[FileMeta]) -> dict[str, Any]:
    timestamps: list[tuple[datetime, str]] = []
    end_sources: list[str] = []
    warnings: list[str] = []
    chunks_by_path: dict[Path, list[tuple[str, str]]] = {}
    for meta in logs:
        chunks, error = _read_head_tail(meta.path)
        chunks_by_path[meta.path] = chunks
        if error:
            warnings.append(f"日志无法读取 {meta.path}：{error}")
        for chunk_label, text in chunks:
            for time_match in _LOGGER_TIME_RE.finditer(text):
                parsed = _parse_datetime(time_match.group(1))
                if parsed is not None:
                    timestamps.append((parsed, f"{meta.path}:{chunk_label}"))
            for time_match in _ISO_TIME_RE.finditer(text):
                parsed = _parse_datetime(time_match.group(1))
                if parsed is not None:
                    timestamps.append((parsed, f"{meta.path}:{chunk_label}"))
            if chunk_label in {"末尾", "全文"}:
                for pattern in _NORMAL_END_PATTERNS:
                    if pattern.search(text):
                        end_sources.append(f"{meta.path}:{chunk_label}:匹配 {pattern.pattern}")
                        break

    aware_timestamps = [item for item in timestamps if item[0].utcoffset() is not None]
    naive_timestamps = [item for item in timestamps if item[0].utcoffset() is None]
    if aware_timestamps and naive_timestamps:
        explicit_offsets = {item[0].utcoffset() for item in aware_timestamps}
        if len(explicit_offsets) == 1:
            assumed_timezone = timezone(next(iter(explicit_offsets)))
            timestamps = [
                (
                    value if value.utcoffset() is not None else value.replace(tzinfo=assumed_timezone),
                    source,
                )
                for value, source in timestamps
            ]
            warnings.append(
                "日志混合带时区/不带时区时间戳；按唯一显式 UTC offset 补齐不带时区项"
            )
        else:
            timestamps = aware_timestamps
            warnings.append(
                "日志混合不带时区项及多个显式 UTC offset；首末时间与时长仅采用带时区项"
            )
    first_time = min(timestamps, key=lambda item: item[0]) if timestamps else None
    last_time = max(timestamps, key=lambda item: item[0]) if timestamps else None
    duration_seconds: Optional[float] = None
    duration_source = ""
    if first_time and last_time:
        try:
            duration_seconds = max(0.0, (last_time[0] - first_time[0]).total_seconds())
            duration_source = f"{first_time[1]} 至 {last_time[1]}"
        except TypeError:
            warnings.append("日志同时含有带时区和不带时区的时间戳，墙钟时长无法判定")
    return {
        "first_time": first_time[0].isoformat() if first_time else "无法判定",
        "first_time_source": first_time[1] if first_time else "扫描日志开头/末尾：未找到时间戳",
        "last_time": last_time[0].isoformat() if last_time else "无法判定",
        "last_time_source": last_time[1] if last_time else "扫描日志开头/末尾：未找到时间戳",
        "duration_seconds": duration_seconds,
        "duration_source": duration_source or "日志时间戳不足，无法推算",
        "normal_end": bool(end_sources),
        "normal_end_source": "; ".join(end_sources) if end_sources else "扫描日志末尾/全文：未匹配正常结束标记",
        "chunks": chunks_by_path,
        "warnings": warnings,
    }


def _metric_key(mapping: Mapping[str, Any], aliases: Iterable[str]) -> tuple[Optional[float], str]:
    normalised = {_normalise_key(alias) for alias in aliases}
    for key, value in mapping.items():
        if _normalise_key(str(key)) in normalised:
            return _safe_number(value), str(key)
    return None, ""


def _scenario_from_mapping(mapping: Mapping[str, Any], fallback: str = "") -> str:
    for key in ("scenario", "condition", "degradation", "degrade_mode", "eval_degrade_mode", "mode"):
        for actual, value in mapping.items():
            if _normalise_key(str(actual)) == _normalise_key(key) and str(value).strip():
                scenario = str(value).strip()
                strength = next(
                    (
                        str(v)
                        for k, v in mapping.items()
                        if _normalise_key(str(k))
                        in {_normalise_key("strength"), _normalise_key("eval_degrade_strength")}
                    ),
                    "",
                )
                return f"{scenario}@{strength}" if strength else scenario
    return fallback


def _scenario_from_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    if "eval" in parts:
        index = len(parts) - 1 - parts[::-1].index("eval")
        if index + 1 < len(path.parts) - 1:
            return path.parts[index + 1]
    for part in reversed(path.parts[:-1]):
        if part.lower() in _CLEAN_SCENARIOS or any(
            token in part.lower() for token in ("blur", "noise", "black", "drop", "ir_", "rgb_")
        ):
            return part
    return ""


def _records_from_mapping(
    mapping: Mapping[str, Any],
    source: str,
    mtime: float,
    priority: int,
    scenario_hint: str = "",
) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    scenario = _scenario_from_mapping(mapping, scenario_hint)
    r1, _ = _metric_key(mapping, ("R1", "rank1", "rank-1", "top1"))
    map_value, _ = _metric_key(mapping, ("mAP", "mean_average_precision"))
    minp, _ = _metric_key(mapping, ("mINP", "mean_inp"))
    if any(value is not None for value in (r1, map_value, minp)):
        records.append(MetricRecord(scenario, r1, map_value, minp, source, mtime, priority))

    # 支持 clean_R1、ir_black_mAP 等宽表。
    wide: dict[str, dict[str, Optional[float]]] = defaultdict(dict)
    for raw_key, value in mapping.items():
        key = str(raw_key)
        found = re.match(r"^(.+?)[_.-](R1|rank1|mAP|mINP)$", key, re.IGNORECASE)
        if found:
            metric_name = _normalise_key(found.group(2))
            canonical = "r1" if metric_name in {"r1", "rank1"} else metric_name
            wide[found.group(1)][canonical] = _safe_number(value)
    for wide_scenario, values in wide.items():
        records.append(
            MetricRecord(
                wide_scenario,
                values.get("r1"),
                values.get("map"),
                values.get("minp"),
                f"{source}:{wide_scenario}_*",
                mtime,
                priority,
            )
        )
    return records


def _records_from_json(meta: FileMeta) -> list[MetricRecord]:
    if "scores" in meta.kinds:
        return []
    text, _ = _read_small_text(meta.path)
    if text is None:
        return []
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    records: list[MetricRecord] = []

    def visit(value: Any, pointer: str, scenario_hint: str, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(value, Mapping):
            current_hint = _scenario_from_mapping(value, scenario_hint)
            records.extend(
                _records_from_mapping(
                    value,
                    f"{meta.path}:{pointer or '/'}",
                    meta.mtime,
                    40,
                    current_hint,
                )
            )
            for key, child in value.items():
                child_hint = current_hint
                if isinstance(child, Mapping) and str(key).lower() not in {"metrics", "metadata", "summary"}:
                    child_hint = str(key)
                visit(child, f"{pointer}/{key}", child_hint, depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{pointer}/{index}", scenario_hint, depth + 1)

    visit(payload, "", _scenario_from_path(meta.path), 0)
    return records


def _records_from_delimited(meta: FileMeta) -> list[MetricRecord]:
    if "scores" in meta.kinds:
        return []
    try:
        if meta.path.stat().st_size > _MAX_STRUCTURED_TEXT_BYTES:
            return []
        delimiter = "\t" if meta.path.suffix.lower() == ".tsv" else ","
        with meta.path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError):
        return []
    records: list[MetricRecord] = []
    for index, row in enumerate(rows, start=2):
        records.extend(
            _records_from_mapping(
                row,
                f"{meta.path}:第{index}行",
                meta.mtime + index * 1e-9,
                50 if "final" in meta.path.name.lower() else 45,
                _scenario_from_path(meta.path),
            )
        )
    return records


def _records_from_logs(logs: Sequence[FileMeta], log_facts: Mapping[str, Any]) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    chunks_by_path = log_facts.get("chunks", {})
    key_value_patterns = {
        "r1": re.compile(r"(?<![A-Za-z0-9_])R1\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
        "map": re.compile(r"(?<![A-Za-z0-9_])mAP\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
        "minp": re.compile(r"(?<![A-Za-z0-9_])mINP\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    }
    pretty = re.compile(
        r"\|\s*t2i\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|\s*(-?\d+(?:\.\d+)?)\s*\|",
        re.IGNORECASE,
    )
    for meta in logs:
        scenario_hint = _scenario_from_path(meta.path)
        for chunk_label, text in chunks_by_path.get(meta.path, []):
            for line_index, line in enumerate(text.splitlines(), start=1):
                pretty_match = pretty.search(line)
                source = f"{meta.path}:{chunk_label}第{line_index}行"
                if pretty_match:
                    records.append(
                        MetricRecord(
                            scenario_hint,
                            float(pretty_match.group(1)),
                            float(pretty_match.group(4)),
                            float(pretty_match.group(5)),
                            source,
                            meta.mtime + line_index * 1e-9,
                            30,
                        )
                    )
                    continue
                values: dict[str, Optional[float]] = {}
                for name, pattern in key_value_patterns.items():
                    found = pattern.search(line)
                    values[name] = float(found.group(1)) if found else None
                if values["r1"] is not None and (
                    values["map"] is not None or values["minp"] is not None
                ):
                    scenario_match = re.search(
                        r"(?:scenario|eval_degrade_mode|degrade_mode)\s*[:=]\s*([A-Za-z0-9_.-]+)",
                        line,
                        re.IGNORECASE,
                    )
                    scenario = scenario_match.group(1) if scenario_match else scenario_hint
                    records.append(
                        MetricRecord(
                            scenario,
                            values["r1"],
                            values["map"],
                            values["minp"],
                            source,
                            meta.mtime + line_index * 1e-9,
                            25,
                        )
                    )
    return records


def _choose_metrics(metas: Sequence[FileMeta], logs: Sequence[FileMeta], log_facts: Mapping[str, Any]) -> dict[str, Any]:
    records: list[MetricRecord] = []
    for meta in metas:
        if "metrics" not in meta.kinds or "scores" in meta.kinds:
            continue
        suffix = meta.path.suffix.lower()
        if suffix == ".json":
            records.extend(_records_from_json(meta))
        elif suffix in {".csv", ".tsv"}:
            records.extend(_records_from_delimited(meta))
    records.extend(_records_from_logs(logs, log_facts))

    # 相同来源窗口可能重叠；按内容去重。
    unique: dict[tuple[Any, ...], MetricRecord] = {}
    for record in records:
        key = (record.scenario, record.r1, record.map_value, record.minp, record.source)
        unique[key] = record
    records = list(unique.values())

    clean_records = [record for record in records if record.scenario.strip().lower() in _CLEAN_SCENARIOS]
    if not clean_records:
        unlabelled = [record for record in records if not record.scenario.strip()]
        clean_records = unlabelled
    clean = max(clean_records, key=lambda item: (item.priority, item.mtime), default=None)

    degraded_by_scenario: dict[str, MetricRecord] = {}
    for record in records:
        scenario = record.scenario.strip()
        if scenario.lower() in _CLEAN_SCENARIOS:
            continue
        previous = degraded_by_scenario.get(scenario)
        if previous is None or (record.priority, record.mtime) > (previous.priority, previous.mtime):
            degraded_by_scenario[scenario] = record
    degraded_payload = [
        {
            "scenario": scenario,
            "R1": record.r1,
            "mAP": record.map_value,
            "mINP": record.minp,
            "source": record.source,
        }
        for scenario, record in sorted(degraded_by_scenario.items())
    ]
    return {
        "clean_r1": clean.r1 if clean else None,
        "clean_map": clean.map_value if clean else None,
        "clean_minp": clean.minp if clean else None,
        "clean_source": clean.source if clean else "metrics 文件与日志中未解析到 clean R1/mAP/mINP",
        "degraded_metrics": degraded_payload,
        "degraded_source": "; ".join(item["source"] for item in degraded_payload)
        if degraded_payload
        else "metrics 文件与日志中未解析到退化场景指标",
        "record_count": len(records),
    }


def _extract_commit_from_text(path: Path, text: str, location: str) -> Optional[tuple[str, str]]:
    patterns = (
        re.compile(r"(?:git[_ -]?(?:head|commit)|commit(?:[_ -]?(?:hash|sha))?)\s*[:=]\s*([0-9a-f]{7,40})", re.IGNORECASE),
        re.compile(r"\bHEAD\s*[:=]\s*([0-9a-f]{7,40})", re.IGNORECASE),
    )
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern in patterns:
            found = pattern.search(line)
            if found:
                return found.group(1), f"{path}:{location}第{line_number}行"
    return None


def _normalise_git_output(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, str) and item.strip():
                return item
    if hasattr(result, "stdout"):
        return str(getattr(result, "stdout", ""))
    if isinstance(result, Mapping):
        for key in ("stdout", "output", "text"):
            if key in result:
                return str(result[key])
    return ""


def _git_history(repo: Optional[Path]) -> tuple[list[tuple[int, str]], str]:
    if repo is None:
        return [], "未提供 --repo"
    try:
        output = _normalise_git_output(
            run_git(repo, "log", "--all", "--format=%H%x09%ct")
        )
    except Exception as exc:
        return [], f"git log 无法执行：{exc}"
    history: list[tuple[int, str]] = []
    for line in output.splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 2:
            parts = line.strip().split()
        if len(parts) >= 2 and _SHA_RE.fullmatch(parts[0]):
            try:
                history.append((int(parts[1]), parts[0]))
            except ValueError:
                continue
    history.sort()
    return history, "git log --all --format=%H%x09%ct"


def _commit_for_run(
    metas: Sequence[FileMeta],
    logs: Sequence[FileMeta],
    log_facts: Mapping[str, Any],
    git_history: Sequence[tuple[int, str]],
    git_source: str,
) -> dict[str, str]:
    # 优先级①：目录内显式保存的 commit/manifest/config/launch metadata。
    record_metas = [
        meta
        for meta in metas
        if "metadata" in meta.kinds or "config" in meta.kinds
    ]
    record_metas.sort(
        key=lambda item: (
            0 if any(token in item.path.name.lower() for token in ("commit", "git", "manifest", "launch")) else 1,
            str(item.path).lower(),
        )
    )
    for meta in record_metas:
        text, _ = _read_small_text(meta.path, limit=2 * 1024 * 1024)
        if text:
            found = _extract_commit_from_text(meta.path, text, "文件")
            if found:
                return {
                    "commit": found[0],
                    "source": found[1],
                    "status": "OK",
                    "candidates": found[0],
                    "method": "目录内 commit 记录",
                }

    # 优先级②：训练/评估日志打印的 commit。
    chunks_by_path = log_facts.get("chunks", {})
    for meta in logs:
        for location, text in chunks_by_path.get(meta.path, []):
            found = _extract_commit_from_text(meta.path, text, location)
            if found:
                return {
                    "commit": found[0],
                    "source": found[1],
                    "status": "OK",
                    "candidates": found[0],
                    "method": "日志打印 commit",
                }

    # 优先级③：只按日志 mtime 找到前后两个 commit，必须标 WARN。
    if logs and git_history:
        chosen_log = max(logs, key=lambda item: item.mtime)
        times = [item[0] for item in git_history]
        index = bisect.bisect_left(times, int(chosen_log.mtime))
        nearby: list[tuple[int, str]] = []
        if index > 0:
            nearby.append(git_history[index - 1])
        if index < len(git_history):
            nearby.append(git_history[index])
        if len(nearby) == 1 and len(git_history) > 1:
            nearby = git_history[-2:] if index >= len(git_history) else git_history[:2]
        candidates = [f"{sha}@{_format_mtime(timestamp)}" for timestamp, sha in nearby]
        return {
            "commit": "无法唯一判定",
            "source": f"{git_source}; {chosen_log.path}:mtime={_format_mtime(chosen_log.mtime)}",
            "status": "WARN",
            "candidates": "; ".join(candidates),
            "method": "日志 mtime 所在 commit 时间区间",
        }
    return {
        "commit": "无法判定",
        "source": git_source if not logs else f"{git_source}; 日志中未打印 commit",
        "status": "UNKNOWN",
        "candidates": "",
        "method": "未找到证据",
    }


def _dedupe_paths(metas: Sequence[FileMeta]) -> list[FileMeta]:
    seen: set[str] = set()
    result: list[FileMeta] = []
    for meta in metas:
        try:
            identity = _path_key(meta.path.resolve(strict=False)) if meta.is_symlink else _path_key(meta.path)
        except (OSError, RuntimeError):
            identity = _path_key(meta.path)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(meta)
    return result


def _path_list(metas: Sequence[FileMeta]) -> str:
    return "; ".join(str(meta.path) for meta in metas)


def _source_for_presence(kind: str, metas: Sequence[FileMeta], wrapper: Path) -> str:
    matching = [meta.path for meta in metas if kind in meta.kinds]
    if matching:
        return "; ".join(str(path) for path in matching)
    return f"扫描 {wrapper}：未找到 {kind} 类型文件"


def _run_size(wrapper: Path, scans: Sequence[ScanResult], sizes_by_root: Mapping[str, Mapping[Path, int]]) -> tuple[int, str]:
    for scan in scans:
        if _is_relative_to(wrapper, scan.root):
            size = int(sizes_by_root.get(_path_key(scan.root), {}).get(wrapper, 0))
            return size, f"文件系统遍历 {wrapper}（不跟随目录符号链接）"
    return 0, f"{wrapper}:无法匹配扫描根"


def _build_row(
    wrapper: Path,
    leafs: Sequence[Path],
    metas: Sequence[FileMeta],
    scans: Sequence[ScanResult],
    sizes_by_root: Mapping[str, Mapping[Path, int]],
    git_history: Sequence[tuple[int, str]],
    git_source: str,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    config_values, config_warnings = _collect_config_values(metas)
    warnings.extend(config_warnings)
    label = _infer_label(wrapper, config_values)
    seed = _infer_seed(wrapper, config_values)

    configs = [meta for meta in metas if "config" in meta.kinds]
    checkpoints_all = [meta for meta in metas if "checkpoint" in meta.kinds]
    checkpoints = _dedupe_paths(checkpoints_all)
    logs = [meta for meta in metas if "log" in meta.kinds]
    scores = [meta for meta in metas if "scores" in meta.kinds]
    metric_files = [meta for meta in metas if "metrics" in meta.kinds]
    gate_files = [meta for meta in metas if "gate" in meta.kinds]
    drift_files = [meta for meta in metas if "drift" in meta.kinds]

    header_sources: list[str] = []
    for meta in metric_files:
        header_gate, header_drift, header_source = _text_header_flags(meta)
        if header_gate and meta not in gate_files:
            gate_files.append(meta)
            header_sources.append(header_source)
        if header_drift and meta not in drift_files:
            drift_files.append(meta)
            header_sources.append(header_source)

    log_facts = _log_facts(logs)
    warnings.extend(log_facts["warnings"])
    metrics = _choose_metrics(metas, logs, log_facts)
    commit = _commit_for_run(metas, logs, log_facts, git_history, git_source)
    directory_size, size_source = _run_size(wrapper, scans, sizes_by_root)

    checkpoint_sizes: list[str] = []
    checkpoint_mtimes: list[str] = []
    for meta in checkpoints:
        try:
            target_size = meta.path.stat().st_size if meta.is_symlink else meta.size
            target_mtime = meta.path.stat().st_mtime if meta.is_symlink else meta.mtime
        except OSError:
            target_size = meta.size
            target_mtime = meta.mtime
        checkpoint_sizes.append(f"{meta.path}={target_size}")
        checkpoint_mtimes.append(f"{meta.path}={_format_mtime(target_mtime)}")

    row: dict[str, Any] = {
        "run_dir": str(wrapper),
        "run_dir_source": "实验资产目录结构扫描",
        "artifact_leaf_dirs": "; ".join(str(path) for path in sorted(set(leafs))),
        "artifact_leaf_dirs_source": "合并 wrapper/[train_output/]RGBT-PEDES/<leaf> 结构",
        "directory_size_bytes": directory_size,
        "directory_size_human": _format_bytes(directory_size),
        "directory_size_source": size_source,
        "config_label": _display_value(label.value),
        "config_label_key": label.actual_key,
        "config_label_source": label.source,
        "seed": _display_value(seed.value),
        "seed_key": seed.actual_key,
        "seed_source": seed.source,
        "config_paths": _path_list(configs),
        "config_paths_source": _source_for_presence("config", metas, wrapper),
    }
    for canonical in (
        "target_temperature",
        "tau_s",
        "lambda_rel",
        "scorer_lr_multiplier",
        "mdrc_lambda_max",
        "mdrc_scale",
        "mdrc_probability",
        "mdrc_warmup",
        "batch_size",
        "optimizer",
        "learning_rate",
        "epochs",
        "image_size",
        "text_length",
    ):
        match = _resolve_value(config_values, canonical)
        row[canonical] = _display_value(match.value) if match else "无法判定"
        row[f"{canonical}_key"] = match.actual_key if match else ""
        row[f"{canonical}_source"] = match.source if match else f"配置与命令中未匹配 {_ALIASES[canonical]}"

    if row["image_size"] == "无法判定":
        height_aliases = {_normalise_key(value) for value in ("img_h", "image_height", "height")}
        width_aliases = {_normalise_key(value) for value in ("img_w", "image_width", "width")}
        height = next((value for value in reversed(config_values) if _normalise_key(value.actual_key.rsplit('.', 1)[-1]) in height_aliases), None)
        width = next((value for value in reversed(config_values) if _normalise_key(value.actual_key.rsplit('.', 1)[-1]) in width_aliases), None)
        if height and width:
            row["image_size"] = f"{_display_value(height.value)}x{_display_value(width.value)}"
            row["image_size_key"] = f"{height.actual_key}+{width.actual_key}"
            row["image_size_source"] = f"{height.source}; {width.source}"

    row.update(
        {
            "checkpoint_paths": _path_list(checkpoints),
            "checkpoint_paths_source": _source_for_presence("checkpoint", metas, wrapper),
            "checkpoint_count": len(checkpoints),
            "checkpoint_count_source": "checkpoint 路径按符号链接目标去重",
            "checkpoint_sizes": "; ".join(checkpoint_sizes),
            "checkpoint_sizes_source": "文件系统 stat；未读取 checkpoint 内容",
            "checkpoint_mtimes": "; ".join(checkpoint_mtimes),
            "checkpoint_mtimes_source": "文件系统 stat",
            "log_paths": _path_list(logs),
            "log_paths_source": _source_for_presence("log", metas, wrapper),
            "log_first_timestamp": log_facts["first_time"],
            "log_first_timestamp_source": log_facts["first_time_source"],
            "log_last_timestamp": log_facts["last_time"],
            "log_last_timestamp_source": log_facts["last_time_source"],
            "wallclock_seconds": "" if log_facts["duration_seconds"] is None else round(log_facts["duration_seconds"], 3),
            "wallclock_hours": "" if log_facts["duration_seconds"] is None else round(log_facts["duration_seconds"] / 3600.0, 6),
            "wallclock_source": log_facts["duration_source"],
            "normal_end": bool(log_facts["normal_end"]),
            "normal_end_source": log_facts["normal_end_source"],
            "clean_R1": "" if metrics["clean_r1"] is None else metrics["clean_r1"],
            "clean_mAP": "" if metrics["clean_map"] is None else metrics["clean_map"],
            "clean_mINP": "" if metrics["clean_minp"] is None else metrics["clean_minp"],
            "clean_metrics_source": metrics["clean_source"],
            "degraded_metrics": json.dumps(metrics["degraded_metrics"], ensure_ascii=False, sort_keys=True),
            "degraded_metrics_source": metrics["degraded_source"],
            "score_paths": _path_list(scores),
            "score_paths_source": _source_for_presence("scores", metas, wrapper),
            "gate_stats_paths": _path_list(_dedupe_paths(gate_files)),
            "gate_stats_source": "; ".join(header_sources) or _source_for_presence("gate", metas, wrapper),
            "drift_stats_paths": _path_list(_dedupe_paths(drift_files)),
            "drift_stats_source": "; ".join(header_sources) or _source_for_presence("drift", metas, wrapper),
            "git_commit": commit["commit"],
            "git_commit_method": commit["method"],
            "git_commit_status": commit["status"],
            "git_commit_candidates": commit["candidates"],
            "git_commit_source": commit["source"],
            "has_ckpt": bool(checkpoints),
            "has_ckpt_source": _source_for_presence("checkpoint", metas, wrapper),
            "has_log": bool(logs),
            "has_log_source": _source_for_presence("log", metas, wrapper),
            "has_config": bool(configs),
            "has_config_source": _source_for_presence("config", metas, wrapper),
            "has_metrics": bool(metric_files or metrics["record_count"]),
            "has_metrics_source": _path_list(metric_files) or metrics["clean_source"],
            "has_scores": bool(scores),
            "has_scores_source": _source_for_presence("scores", metas, wrapper),
            "has_gate_stats": bool(gate_files),
            "has_gate_stats_source": _path_list(gate_files) or _source_for_presence("gate", metas, wrapper),
            "has_drift_stats": bool(drift_files),
            "has_drift_stats_source": _path_list(drift_files) or _source_for_presence("drift", metas, wrapper),
            "reuse_decision": "",
        }
    )
    return row, warnings


_CSV_FIELDS = (
    "run_dir",
    "run_dir_source",
    "artifact_leaf_dirs",
    "artifact_leaf_dirs_source",
    "directory_size_bytes",
    "directory_size_human",
    "directory_size_source",
    "config_label",
    "config_label_key",
    "config_label_source",
    "seed",
    "seed_key",
    "seed_source",
    "config_paths",
    "config_paths_source",
    "target_temperature",
    "target_temperature_key",
    "target_temperature_source",
    "tau_s",
    "tau_s_key",
    "tau_s_source",
    "lambda_rel",
    "lambda_rel_key",
    "lambda_rel_source",
    "scorer_lr_multiplier",
    "scorer_lr_multiplier_key",
    "scorer_lr_multiplier_source",
    "mdrc_lambda_max",
    "mdrc_lambda_max_key",
    "mdrc_lambda_max_source",
    "mdrc_scale",
    "mdrc_scale_key",
    "mdrc_scale_source",
    "mdrc_probability",
    "mdrc_probability_key",
    "mdrc_probability_source",
    "mdrc_warmup",
    "mdrc_warmup_key",
    "mdrc_warmup_source",
    "batch_size",
    "batch_size_key",
    "batch_size_source",
    "optimizer",
    "optimizer_key",
    "optimizer_source",
    "learning_rate",
    "learning_rate_key",
    "learning_rate_source",
    "epochs",
    "epochs_key",
    "epochs_source",
    "image_size",
    "image_size_key",
    "image_size_source",
    "text_length",
    "text_length_key",
    "text_length_source",
    "checkpoint_paths",
    "checkpoint_paths_source",
    "checkpoint_count",
    "checkpoint_count_source",
    "checkpoint_sizes",
    "checkpoint_sizes_source",
    "checkpoint_mtimes",
    "checkpoint_mtimes_source",
    "log_paths",
    "log_paths_source",
    "log_first_timestamp",
    "log_first_timestamp_source",
    "log_last_timestamp",
    "log_last_timestamp_source",
    "wallclock_seconds",
    "wallclock_hours",
    "wallclock_source",
    "normal_end",
    "normal_end_source",
    "clean_R1",
    "clean_mAP",
    "clean_mINP",
    "clean_metrics_source",
    "degraded_metrics",
    "degraded_metrics_source",
    "score_paths",
    "score_paths_source",
    "gate_stats_paths",
    "gate_stats_source",
    "drift_stats_paths",
    "drift_stats_source",
    "git_commit",
    "git_commit_method",
    "git_commit_status",
    "git_commit_candidates",
    "git_commit_source",
    "has_ckpt",
    "has_ckpt_source",
    "has_log",
    "has_log_source",
    "has_config",
    "has_config_source",
    "has_metrics",
    "has_metrics_source",
    "has_scores",
    "has_scores_source",
    "has_gate_stats",
    "has_gate_stats_source",
    "has_drift_stats",
    "has_drift_stats_source",
    "reuse_decision",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in _CSV_FIELDS})


def _md_cell(value: Any) -> str:
    text = _display_value(value).replace("\r", " ").replace("\n", "<br>")
    return text.replace("|", "\\|")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(_md_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
    return lines


def _coverage_cell(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    has_ckpt = any(bool(row["has_ckpt"]) for row in rows)
    has_metrics = any(bool(row["has_metrics"]) for row in rows)
    if has_ckpt and has_metrics:
        value = "checkpoint + 指标"
    elif has_ckpt:
        value = "仅 checkpoint"
    elif has_metrics:
        value = "仅指标"
    else:
        value = "无 checkpoint/指标"
    sources = sorted({str(row["run_dir"]) for row in rows})
    return value, "; ".join(sources)


def _report_lines(
    rows: Sequence[Mapping[str, Any]],
    roots: Sequence[Path],
    csv_path: Path,
    warnings: Sequence[str],
) -> list[str]:
    lines = [
        "# 实验资产审计",
        "",
        "> 本报告仅记录脚本可复核的事实；`复用裁定` 始终留空，由人工填写。",
        "",
        "## 一、扫描范围",
        "",
    ]
    scope_rows = [[str(root), "命令行参数 `--roots`"] for root in roots]
    lines.extend(_markdown_table(("扫描根", "source"), scope_rows or [["未提供", "命令行参数 `--roots`"]]))
    lines.extend(["", f"原始明细：`{csv_path}`（source: 本次审计输出）", ""])

    lines.extend(["## 二、按配置分组汇总", ""])
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_label"])].append(row)
    group_rows: list[list[Any]] = []
    for label, group in sorted(grouped.items()):
        duration_values = [float(row["wallclock_hours"]) for row in group if row["wallclock_hours"] != ""]
        group_rows.append(
            [
                label,
                len(group),
                ", ".join(sorted({str(row["seed"]) for row in group})),
                sum(bool(row["has_ckpt"]) for row in group),
                sum(bool(row["has_metrics"]) for row in group),
                f"{statistics.median(duration_values):.3f}" if duration_values else "无法判定",
                "; ".join(str(row["run_dir"]) for row in group),
                "",
            ]
        )
    if not group_rows:
        group_rows.append(["未找到", 0, "未找到", 0, 0, "无法判定", "扫描根：未识别运行目录", ""])
    lines.extend(
        _markdown_table(
            ("配置", "运行数", "seed", "有 checkpoint", "有指标", "时长中位数(h)", "source", "复用裁定"),
            group_rows,
        )
    )
    lines.append("")

    lines.extend(["## 三、配置 × seed 覆盖矩阵", ""])
    seeds = sorted({str(row["seed"]) for row in rows})
    coverage_rows: list[list[Any]] = []
    for label, group in sorted(grouped.items()):
        line: list[Any] = [label]
        source_parts: list[str] = []
        for seed in seeds:
            cell_rows = [row for row in group if str(row["seed"]) == seed]
            if cell_rows:
                value, source = _coverage_cell(cell_rows)
                line.append(value)
                source_parts.append(f"seed={seed}: {source}")
            else:
                line.append("—")
        line.extend(["; ".join(source_parts) or "无记录", ""])
        coverage_rows.append(line)
    headers = ["配置", *[f"seed {seed}" for seed in seeds], "source", "复用裁定"]
    if not coverage_rows:
        coverage_rows.append(["未找到", "扫描根：未识别运行目录", ""])
    lines.extend(_markdown_table(headers, coverage_rows))
    lines.append("")

    lines.extend(["## 四、单次训练墙钟时长", ""])
    durations = [
        (float(row["wallclock_hours"]), str(row["wallclock_source"]))
        for row in rows
        if row["wallclock_hours"] != ""
    ]
    if durations:
        duration_values = [value for value, _ in durations]
        duration_source = "; ".join(source for _, source in durations)
        duration_rows = [
            ["最小值", f"{min(duration_values):.3f}", duration_source],
            ["中位数", f"{statistics.median(duration_values):.3f}", duration_source],
            ["最大值", f"{max(duration_values):.3f}", duration_source],
        ]
    else:
        duration_rows = [["最小值/中位数/最大值", "无法判定", "日志时间戳不足或未找到运行目录"]]
    lines.extend(_markdown_table(("统计项", "小时", "source"), duration_rows))
    lines.append("")

    lines.extend(["## 五、运行明细", ""])
    detail_rows = []
    for row in rows:
        detail_rows.append(
            [
                row["run_dir"],
                row["config_label"],
                row["seed"],
                row["has_ckpt"],
                row["has_log"],
                row["has_config"],
                row["has_metrics"],
                row["has_scores"],
                row["has_gate_stats"],
                row["has_drift_stats"],
                row["clean_R1"] if row["clean_R1"] != "" else "无法判定",
                row["clean_mAP"] if row["clean_mAP"] != "" else "无法判定",
                row["clean_mINP"] if row["clean_mINP"] != "" else "无法判定",
                f"{row['git_commit']} ({row['git_commit_status']})",
                "; ".join(
                    (
                        str(row["config_label_source"]),
                        str(row["seed_source"]),
                        str(row["clean_metrics_source"]),
                        str(row["git_commit_source"]),
                    )
                ),
                "",
            ]
        )
    if not detail_rows:
        detail_rows.append(
            [
                "未找到",
                "无法判定",
                "无法判定",
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                "无法判定",
                "无法判定",
                "无法判定",
                "无法判定 (UNKNOWN)",
                "扫描根：未识别运行目录",
                "",
            ]
        )
    lines.extend(
        _markdown_table(
            (
                "运行目录",
                "配置",
                "seed",
                "ckpt",
                "log",
                "config",
                "metrics",
                "分数矩阵",
                "gate",
                "drift",
                "clean R1",
                "clean mAP",
                "clean mINP",
                "commit",
                "source",
                "复用裁定",
            ),
            detail_rows,
        )
    )
    lines.append("")

    lines.extend(["## 六、退化场景指标", ""])
    degraded_rows = []
    for row in rows:
        try:
            payload = json.loads(str(row["degraded_metrics"]))
        except json.JSONDecodeError:
            payload = []
        for item in payload:
            degraded_rows.append(
                [
                    row["run_dir"],
                    item.get("scenario", "无法判定"),
                    item.get("R1"),
                    item.get("mAP"),
                    item.get("mINP"),
                    item.get("source", row["degraded_metrics_source"]),
                ]
            )
    if not degraded_rows:
        degraded_rows.append(["未找到", "未找到", "", "", "", "metrics 文件与日志扫描"])
    lines.extend(_markdown_table(("运行目录", "场景", "R1", "mAP", "mINP", "source"), degraded_rows))
    lines.append("")

    lines.extend(["## 七、未解决项", ""])
    unresolved = list(dict.fromkeys(warnings))
    for row in rows:
        if row["git_commit_status"] == "UNKNOWN":
            unresolved.append(f"commit 无法判定：{row['run_dir']}（source: {row['git_commit_source']}）")
        for field_name, label in (
            ("target_temperature", "Tr"),
            ("tau_s", "tau_s"),
            ("lambda_rel", "lambda_rel"),
            ("scorer_lr_multiplier", "scorer LR multiplier"),
        ):
            if row[field_name] == "无法判定":
                unresolved.append(
                    f"{label} 无法判定：{row['run_dir']}（source: {row[field_name + '_source']}）"
                )
    if unresolved:
        lines.extend(f"- {item}" for item in dict.fromkeys(unresolved))
    else:
        lines.append("- 无（source: 本报告所有运行明细）")
    lines.append("")
    return lines


def _write_markdown(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines).rstrip() + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _project_dir() -> Path:
    # tools/audit/audit_assets.py -> 项目代码根（IRRA-main）
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读盘点实验 config/log/checkpoint/metrics/score/gate/drift 资产。"
    )
    parser.add_argument("--roots", nargs="+", required=True, type=Path, help="一个或多个实验资产根目录")
    parser.add_argument("--out", type=Path, default=Path("audit_out"), help="CSV 输出目录（默认 audit_out/）")
    parser.add_argument("--repo", type=Path, default=None, help="用于 commit 时间线反推的 Git 仓库路径")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    roots: list[Path] = []
    seen_roots: set[str] = set()
    for raw_root in args.roots:
        root = _absolute(raw_root)
        key = _path_key(root)
        if key not in seen_roots:
            roots.append(root)
            seen_roots.add(key)

    scans = [_scan_root(root) for root in roots]
    wrappers, merged_leafs, discovery_warnings = _discover_runs(scans)
    all_files = [meta for scan in scans for meta in scan.files]
    assignments = _assign_files(wrappers, all_files)
    sizes_by_root = {_path_key(scan.root): _subtree_sizes(scan) for scan in scans}
    repo = _absolute(args.repo) if args.repo is not None else _project_dir()
    git_history, git_source = _git_history(repo)

    rows: list[dict[str, Any]] = []
    warnings = [warning for scan in scans for warning in scan.warnings]
    warnings.extend(discovery_warnings)
    if not git_history:
        warnings.append(f"Git commit 时间线不可用：{git_source}")
    for wrapper in wrappers:
        row, run_warnings = _build_row(
            wrapper,
            sorted(merged_leafs.get(wrapper, {wrapper})),
            assignments.get(wrapper, []),
            scans,
            sizes_by_root,
            git_history,
            git_source,
        )
        rows.append(row)
        warnings.extend(run_warnings)
    rows.sort(key=lambda row: (str(row["config_label"]), str(row["seed"]), str(row["run_dir"])))

    out_dir = resolve_audit_output_dir(_project_dir(), args.out)
    csv_path = out_dir / "assets.csv"
    report_path = _project_dir() / "docs" / "audit" / "EXPERIMENT_ASSET_AUDIT.md"
    _write_csv(csv_path, rows)
    _write_markdown(report_path, _report_lines(rows, roots, csv_path, warnings))
    print(f"已写入：{csv_path}")
    print(f"已写入：{report_path}")
    print(f"识别运行目录：{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
