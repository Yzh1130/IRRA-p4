"""还原 MRAF/MDRC 超参数探索历史（只读审计）。"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from .common import AuditFinding, find_files, resolve_audit_output_dir, run_git, write_report
else:  # 兼容 ``python tools/audit/audit_hparams.py --help``。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.audit.common import (  # type: ignore
        AuditFinding,
        find_files,
        resolve_audit_output_dir,
        run_git,
        write_report,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARAMETER_ALIASES: dict[str, tuple[str, ...]] = {
    "mraf_target_temperature": (
        "mraf_target_temperature",
        "target_temperature",
        "Tr",
    ),
    "mraf_gate_logit_scale": (
        "mraf_gate_logit_scale",
        "gate_logit_scale",
        "tau_s",
        "τs",
    ),
    "mraf_scorer_lr_multiplier": (
        "mraf_scorer_lr_multiplier",
        "mraf_scorer_lr_mult",
        "selected_scorer_lr_multiplier",
        "scorer_lr_multiplier",
        "scorer_multiplier",
    ),
    "mraf_rel_lambda": (
        "mraf_rel_lambda",
        "lambda_rel",
        "λrel",
        "rel_lambda",
    ),
    "mraf_freeze_gate": ("mraf_freeze_gate", "freeze_gate"),
    "mraf_gate_temperature": (
        "mraf_gate_temperature",
        "gate_temperature",
    ),
    "mraf_gate_base_grad_scale": (
        "mraf_gate_base_grad_scale",
        "gate_base_grad_scale",
    ),
    "mdrc_lambda": ("mdrc_lambda", "lambda_mdrc", "mdrc_lambda_max"),
    "mdrc_loss_scale": (
        "mdrc_loss_scale",
        "effective_mdrc_loss_scale",
        "mdrc_scale",
    ),
    "mdrc_prob": ("mdrc_prob", "mdrc_probability"),
    "mdrc_warmup_epoch": (
        "mdrc_warmup_epoch",
        "mdrc_warmup_epochs",
        "mdrc_warmup",
    ),
}

KNOWN_ANCHORS: dict[str, dict[str, Any]] = {
    "mraf_target_temperature": {
        "counts": {"0.1": 7, "0.2": 2, "0.3": 1, "0.5": 1},
        "source": "用户提供的 Stage 0 修正 6",
    },
    "mraf_gate_logit_scale": {
        "counts": {"5.0": 9, "1.0": 2},
        "source": "用户提供的 Stage 0 修正 6",
    },
    "mraf_scorer_lr_multiplier": {
        "values": {"1.0", "50.0"},
        "source": "用户提供的 Stage 0 修正 6",
    },
    "mraf_freeze_gate": {
        "counts": {"false": 10, "true": 1},
        "source": "用户提供的 Stage 0 修正 6",
    },
}

MRAF_FOCUS = (
    "mraf_target_temperature",
    "mraf_gate_logit_scale",
    "mraf_scorer_lr_multiplier",
    "mraf_rel_lambda",
    "mraf_freeze_gate",
    "mraf_gate_temperature",
    "mraf_gate_base_grad_scale",
)

TIME_RE = re.compile(
    r"(?P<stamp>20\d{2}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2})"
)
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


@dataclass
class ValueEvidence:
    value: Any
    alias: str
    source: str


@dataclass
class RunRecord:
    run_id: str
    config_path: str
    label: str
    seed: str
    first_time: str
    last_time: str
    time_source: str
    values: dict[str, ValueEvidence]
    metrics: dict[str, str]
    metric_source: str

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["values"] = {
            key: asdict(evidence) for key, evidence in self.values.items()
        }
        return data


def _resolve_project_root(repo: Path) -> Path:
    candidate = repo.expanduser().resolve(strict=False)
    if (candidate / "model" / "mraf.py").is_file():
        return candidate
    nested = candidate / "IRRA-main"
    if (nested / "model" / "mraf.py").is_file():
        return nested
    # 无效 --repo 只影响只读 Git/源码检查；报告仍写回本工具所在项目的 docs/audit。
    return PROJECT_ROOT


def _display(path: Path, base: Path | None = None) -> str:
    path = path.resolve(strict=False)
    if base is not None:
        try:
            return path.relative_to(base.resolve(strict=False)).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _read_text(path: Path, limit: int = 8_000_000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except (OSError, UnicodeError):
        return ""


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    if re.fullmatch(NUMBER_RE, value):
        try:
            return float(value) if any(ch in value.lower() for ch in ".e") else int(value)
        except ValueError:
            pass
    return value.strip("\"'")


def _simple_mapping(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"^\s*([A-Za-z_][\w.-]*)\s*:\s*(.*?)\s*$", line)
        if match and match.group(2):
            result[match.group(1)] = _parse_scalar(match.group(2))
    return result


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(_flatten(value, dotted))
            else:
                flat[dotted] = value
                flat[str(key)] = value
    return flat


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    text = _read_text(path)
    if not text:
        return {}, "无法读取"
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
            return _flatten(data), "json"
        except (json.JSONDecodeError, TypeError):
            return _simple_mapping(text), "文本回退"
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return _flatten(data), "yaml.safe_load"
    except Exception:
        pass
    return _simple_mapping(text), "文本回退"


def _normal_key(key: str) -> str:
    return key.rsplit(".", 1)[-1].strip().lower()


def _lookup_mapping(
    mapping: dict[str, Any], aliases: Iterable[str], source: str
) -> ValueEvidence | None:
    indexed = {_normal_key(str(key)): (str(key), value) for key, value in mapping.items()}
    for alias in aliases:
        item = indexed.get(alias.lower())
        if item is not None:
            actual, value = item
            return ValueEvidence(value=value, alias=actual, source=f"{source}:{actual}")
    return None


def _lookup_text(text: str, aliases: Iterable[str], source: str) -> ValueEvidence | None:
    for alias in aliases:
        pattern = re.compile(
            rf"(?<![\w]){re.escape(alias)}\s*[:=]\s*([^\s,;\]\}}]+)",
            flags=re.IGNORECASE,
        )
        matches = list(pattern.finditer(text))
        if matches:
            raw = matches[-1].group(1).strip("\"'")
            return ValueEvidence(
                value=_parse_scalar(raw),
                alias=alias,
                source=f"{source}:匹配 `{alias}`",
            )
    return None


def _time_strings(paths: Iterable[Path]) -> tuple[str, str, str]:
    stamps: list[datetime] = []
    stamp_sources: list[str] = []
    fallback: list[datetime] = []
    fallback_sources: list[str] = []
    for path in paths:
        text = _read_text(path, limit=2_000_000)
        for match in TIME_RE.finditer(text):
            raw = match.group("stamp").replace("/", "-")
            try:
                stamps.append(datetime.fromisoformat(raw))
                stamp_sources.append(path.as_posix())
            except ValueError:
                continue
        try:
            fallback.append(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
            fallback_sources.append(path.as_posix())
        except OSError:
            continue
    if stamps:
        return (
            min(stamps).isoformat(sep=" "),
            max(stamps).isoformat(sep=" "),
            "日志时间戳: " + ", ".join(sorted(set(stamp_sources))),
        )
    if fallback:
        return (
            min(fallback).isoformat(sep=" "),
            max(fallback).isoformat(sep=" "),
            "文件 mtime（无法取得日志时间戳）: "
            + ", ".join(sorted(set(fallback_sources))),
        )
    return "无法判定", "无法判定", "未找到可用时间来源"


def _seed_and_label(path: Path, mapping: dict[str, Any]) -> tuple[str, str]:
    seed = "无法判定"
    for key in ("seed", "training_seed", "random_seed"):
        evidence = _lookup_mapping(mapping, (key,), path.as_posix())
        if evidence is not None:
            seed = str(evidence.value)
            break
    if seed == "无法判定":
        for part in reversed(path.parts):
            match = re.search(r"(?i)(?:^|[_-])seed[_-]?(\d+)(?:$|[_-])", part)
            if match:
                seed = match.group(1)
                break
    label_match = next(
        (
            match
            for part in reversed(path.parts)
            if (
                match := re.search(
                    r"(?i)(?:^|[_-])(SCR0[235]|A|B|C|D|E|F|H1|H2)(?:[_-]|$)",
                    part,
                )
            )
        ),
        None,
    )
    if label_match:
        label = label_match.group(1).upper()
    else:
        label = path.parent.name or "无法判定"
    return seed, label


def _extract_metrics(texts: list[tuple[Path, str]]) -> tuple[dict[str, str], str]:
    metrics: dict[str, str] = {}
    sources: list[str] = []
    metric_patterns = {
        "R1": re.compile(rf"(?i)(?:R1|Rank[- ]?1)\s*[:=]\s*({NUMBER_RE})"),
        "mAP": re.compile(rf"(?i)mAP\s*[:=]\s*({NUMBER_RE})"),
        "mINP": re.compile(rf"(?i)mINP\s*[:=]\s*({NUMBER_RE})"),
    }
    for path, text in texts:
        scenario = "clean"
        lowered_path = path.as_posix().lower()
        for candidate in (
            "rgb_drop",
            "ir_drop",
            "rgb_noise",
            "ir_noise",
            "rgb_attenuate",
            "ir_attenuate",
            "moddrop",
        ):
            if candidate in lowered_path:
                scenario = candidate
                break
        for name, pattern in metric_patterns.items():
            matches = pattern.findall(text)
            if matches:
                metrics[f"{scenario}.{name}"] = matches[-1]
                sources.append(f"{path.as_posix()}:末次 `{name}`")
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            flat = _flatten(parsed)
            for key, value in flat.items():
                lowered = key.lower()
                if lowered.endswith("r1") or lowered.endswith("map") or lowered.endswith("minp"):
                    metrics[key] = str(value)
                    sources.append(f"{path.as_posix()}:{key}")
    return metrics, "; ".join(sorted(set(sources))) or "未找到指标"


def _nearby_logs(config: Path, log_files: list[Path], root: Path) -> list[Path]:
    result: list[Path] = []
    config_parent = config.parent.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    ancestors = {config_parent}
    current = config_parent
    for _ in range(3):
        if current == root_resolved or current.parent == current:
            break
        current = current.parent
        ancestors.add(current)
    for path in log_files:
        parent = path.parent.resolve(strict=False)
        if parent in ancestors:
            result.append(path)
            continue
        try:
            relative = parent.relative_to(config_parent)
            if len(relative.parts) <= 2:
                result.append(path)
        except ValueError:
            pass
    return sorted(set(result), key=lambda item: item.as_posix())


def _discover_runs(roots: list[Path]) -> tuple[list[RunRecord], list[AuditFinding]]:
    records: list[RunRecord] = []
    findings: list[AuditFinding] = []
    seen_configs: set[Path] = set()
    claimed_logs: set[Path] = set()
    seen_log_run_dirs: set[Path] = set()
    for root in roots:
        source_root = root.expanduser().resolve(strict=False)
        if not source_root.exists():
            findings.append(
                AuditFinding(
                    item="超参数扫描根目录",
                    value="未找到",
                    source=source_root.as_posix(),
                    status="UNKNOWN",
                    note="目录不存在，已继续扫描其它根目录。",
                )
            )
            continue
        configs = find_files(
            source_root,
            ("*.yaml", "*.yml", "*config*.json", "*args*.json", "*hparam*.json"),
            max_depth=None,
        )
        logs = find_files(
            source_root,
            ("*.log", "*log*.txt", "*.out", "*.tsv", "*metrics*.json", "*summary*.json"),
            max_depth=None,
        )
        resolved_logs = sorted(
            {path.resolve(strict=False) for path in logs}, key=lambda item: item.as_posix()
        )
        config_groups: dict[Path, list[Path]] = defaultdict(list)
        for raw_config in configs:
            config = raw_config.resolve(strict=False)
            if config not in seen_configs:
                seen_configs.add(config)
                config_groups[config.parent].append(config)

        for run_dir, run_configs in sorted(
            config_groups.items(), key=lambda item: item[0].as_posix()
        ):
            config_items = [
                (config, *_load_config(config))
                for config in sorted(run_configs, key=lambda item: item.as_posix())
            ]
            merged_mapping: dict[str, Any] = {}
            for _, mapping, _ in config_items:
                merged_mapping.update(mapping)
            nearby = sorted(
                {
                    log.resolve(strict=False)
                    for config in run_configs
                    for log in _nearby_logs(config, resolved_logs, source_root)
                },
                key=lambda item: item.as_posix(),
            )
            claimed_logs.update(nearby)
            text_items = [(path, _read_text(path)) for path in nearby]
            combined = "\n".join(text for _, text in text_items)
            values: dict[str, ValueEvidence] = {}
            for canonical, aliases in PARAMETER_ALIASES.items():
                evidence = next(
                    (
                        found
                        for config, mapping, _ in config_items
                        if (found := _lookup_mapping(mapping, aliases, config.as_posix()))
                        is not None
                    ),
                    None,
                )
                if evidence is None and combined:
                    evidence = _lookup_text(
                        combined,
                        aliases,
                        ", ".join(path.as_posix() for path, _ in text_items),
                    )
                if evidence is not None:
                    values[canonical] = evidence
            primary_config = run_configs[0]
            seed, label = _seed_and_label(primary_config, merged_mapping)
            metrics, metric_source = _extract_metrics(text_items)
            first_time, last_time, time_source = _time_strings([*run_configs, *nearby])
            records.append(
                RunRecord(
                    run_id=_display(run_dir, source_root),
                    config_path="; ".join(config.as_posix() for config in run_configs),
                    label=label,
                    seed=seed,
                    first_time=first_time,
                    last_time=last_time,
                    time_source=time_source,
                    values=values,
                    metrics=metrics,
                    metric_source=metric_source,
                )
            )
            findings.append(
                AuditFinding(
                    item="配置解析",
                    value="; ".join(
                        f"{config.name}={parser_name}"
                        for config, _, parser_name in config_items
                    ),
                    source="; ".join(config.as_posix() for config in run_configs),
                    status="OK" if merged_mapping else "WARN",
                    note=(
                        f"同一运行目录合并 config {len(run_configs)} 个，"
                        f"关联日志/指标文件 {len(nearby)} 个，只计为 1 次运行。"
                    ),
                )
            )

        log_groups: dict[Path, list[Path]] = defaultdict(list)
        for log_path in resolved_logs:
            if log_path not in claimed_logs:
                log_groups[log_path.parent].append(log_path)
        for run_dir, run_logs in sorted(
            log_groups.items(), key=lambda item: item[0].as_posix()
        ):
            if run_dir in seen_log_run_dirs:
                continue
            seen_log_run_dirs.add(run_dir)
            claimed_logs.update(run_logs)
            text_items = [(path, _read_text(path)) for path in run_logs]
            combined = "\n".join(text for _, text in text_items)
            values: dict[str, ValueEvidence] = {}
            source_text = ", ".join(path.as_posix() for path in run_logs)
            for canonical, aliases in PARAMETER_ALIASES.items():
                evidence = _lookup_text(combined, aliases, source_text)
                if evidence is not None:
                    values[canonical] = evidence
            seed, label = _seed_and_label(run_logs[0], {})
            metrics, metric_source = _extract_metrics(text_items)
            first_time, last_time, time_source = _time_strings(run_logs)
            records.append(
                RunRecord(
                    run_id=_display(run_dir, source_root),
                    config_path="未找到 config；logs=" + source_text,
                    label=label,
                    seed=seed,
                    first_time=first_time,
                    last_time=last_time,
                    time_source=time_source,
                    values=values,
                    metrics=metrics,
                    metric_source=metric_source,
                )
            )
            findings.append(
                AuditFinding(
                    item="仅日志运行解析",
                    value=f"超参 {len(values)} 项，指标 {len(metrics)} 项",
                    source=source_text,
                    status="OK" if values or metrics else "WARN",
                    note="该目录未找到 config，按日志所在目录合并为 1 次运行。",
                )
            )
    return records, findings


def _value_key(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(float(value))
    if isinstance(value, int):
        return str(value)
    return str(value)


def _history(records: list[RunRecord]) -> dict[str, dict[str, dict[str, Any]]]:
    history: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        for parameter, evidence in record.values.items():
            value = _value_key(evidence.value)
            bucket = history[parameter].setdefault(
                value,
                {
                    "runs": [],
                    "seeds": set(),
                    "times": [],
                    "metrics": [],
                    "sources": [],
                },
            )
            bucket["runs"].append(record.run_id)
            if record.seed != "无法判定":
                bucket["seeds"].add(record.seed)
            if record.first_time != "无法判定":
                bucket["times"].append(record.first_time)
            if record.last_time != "无法判定":
                bucket["times"].append(record.last_time)
            if record.metrics:
                bucket["metrics"].append(
                    f"{record.run_id}: "
                    + ", ".join(f"{key}={value}" for key, value in record.metrics.items())
                )
            bucket["sources"].append(evidence.source)
    return history


def _literal_default(node: ast.AST | None) -> str:
    if node is None:
        return "无默认值"
    try:
        return repr(ast.literal_eval(node))
    except (ValueError, TypeError):
        return ast.unparse(node) if hasattr(ast, "unparse") else "无法判定"


def _scan_model_parameters(project_root: Path) -> list[dict[str, str]]:
    path = project_root / "model" / "mraf.py"
    if not path.is_file():
        return []
    text = _read_text(path)
    try:
        tree = ast.parse(text, filename=path.as_posix())
    except SyntaxError:
        return []
    rows: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = [*node.args.posonlyargs, *node.args.args]
        defaults = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)
        for argument, default in zip(args, defaults):
            name = argument.arg
            if name == "self":
                continue
            if any(
                token in name
                for token in (
                    "dim",
                    "dropout",
                    "eps",
                    "fp32",
                    "freeze",
                    "gate",
                    "equalize",
                    "lambda",
                    "scale",
                    "temperature",
                    "warmup",
                )
            ):
                rows.append(
                    {
                        "name": name,
                        "default": _literal_default(default),
                        "source": f"{path.as_posix()}:{argument.lineno}",
                        "scope": node.name,
                    }
                )
    return rows


def _scan_cli_defaults(project_root: Path) -> list[dict[str, str]]:
    path = project_root / "utils" / "options.py"
    text = _read_text(path)
    rows: list[dict[str, str]] = []
    if not text:
        return rows
    pattern = re.compile(
        r"add_argument\(\s*[\"']--(?P<name>mraf_[A-Za-z0-9_]+)[\"'](?P<body>.*?)\)",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        name = match.group("name")
        body = match.group("body")
        default_match = re.search(r"\bdefault\s*=\s*([^,\n\)]+)", body)
        if default_match:
            default = default_match.group(1).strip()
        elif "action='store_true'" in body or 'action="store_true"' in body:
            default = "False（store_true）"
        elif "action='store_false'" in body or 'action="store_false"' in body:
            default = "True（store_false）"
        else:
            default = "无法判定"
        line = text.count("\n", 0, match.start()) + 1
        rows.append(
            {
                "name": name,
                "default": default,
                "source": f"{path.as_posix()}:{line}",
                "scope": "CLI",
            }
        )
    return rows


def _git_timelines(repo: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for parameter in PARAMETER_ALIASES:
        try:
            result = run_git(
                repo,
                "log",
                "--all",
                f"-S{parameter}",
                "--format=%h\t%ci\t%s",
            )
        except (OSError, ValueError) as exc:
            rows.append(
                {
                    "parameter": parameter,
                    "timeline": "无法判定",
                    "source": f"git log -S{parameter}: {exc}",
                }
            )
            continue
        if result.returncode != 0:
            detail = result.stderr.strip() or f"git exit={result.returncode}"
            rows.append(
                {
                    "parameter": parameter,
                    "timeline": "无法判定",
                    "source": f"git log --all -S{parameter} 失败：{detail}",
                }
            )
            continue
        output = result.stdout.strip()
        rows.append(
            {
                "parameter": parameter,
                "timeline": output.replace("\n", "<br>") if output else "未找到相关 commit",
                "source": f"git log --all -S{parameter} --format=%h %ci %s",
            }
        )
    return rows


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _history_markdown(history: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "| 超参 | 取值 | 运行次数 | seed | 首次出现时间 | 最后出现时间 | 关联指标 | source |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    if not history:
        lines.append("| 未找到 | 无法判定 | 0 | 无法判定 | 无法判定 | 无法判定 | 未找到 | 扫描根目录 |")
        return "\n".join(lines)
    for parameter in PARAMETER_ALIASES:
        for value, bucket in sorted(history.get(parameter, {}).items()):
            times = sorted(bucket["times"])
            lines.append(
                "| {parameter} | {value} | {count} | {seeds} | {first} | {last} | {metrics} | {source} |".format(
                    parameter=_escape(parameter),
                    value=_escape(value),
                    count=len(bucket["runs"]),
                    seeds=_escape(", ".join(sorted(bucket["seeds"])) or "无法判定"),
                    first=_escape(times[0] if times else "无法判定"),
                    last=_escape(times[-1] if times else "无法判定"),
                    metrics=_escape("; ".join(bucket["metrics"]) or "未找到"),
                    source=_escape("; ".join(sorted(set(bucket["sources"])))),
                )
            )
    return "\n".join(lines)


def _anchor_markdown(
    records: list[RunRecord], history: dict[str, dict[str, dict[str, Any]]]
) -> tuple[str, list[AuditFinding]]:
    lines = [
        "| 已知锚点 | 服务器确认值 | 本次实测 | 是否一致 | source |",
        "| --- | --- | --- | --- | --- |",
    ]
    findings: list[AuditFinding] = []
    for parameter, anchor in KNOWN_ANCHORS.items():
        observed_buckets = history.get(parameter, {})
        observed_counts = {
            _value_key(value): len(bucket["runs"])
            for value, bucket in observed_buckets.items()
        }
        if "counts" in anchor:
            expected = {_value_key(key): value for key, value in anchor["counts"].items()}
            observed = observed_counts
            consistent = observed == expected if observed_buckets else None
            expected_text = ", ".join(f"{k}×{v}" for k, v in expected.items())
            observed_text = (
                ", ".join(f"{k}×{v}" for k, v in observed.items()) or "未找到"
            )
        else:
            expected_set = {_value_key(item) for item in anchor["values"]}
            observed_set = set(observed_counts)
            consistent = observed_set == expected_set if observed_buckets else None
            expected_text = "{" + ", ".join(sorted(expected_set)) + "}"
            observed_text = "{" + ", ".join(sorted(observed_set)) + "}" if observed_set else "未找到"
        status = "一致" if consistent is True else "不一致" if consistent is False else "无法判定"
        source = "; ".join(
            sorted(
                {
                    evidence.source
                    for record in records
                    for name, evidence in record.values.items()
                    if name == parameter
                }
            )
        ) or "扫描根目录未找到对应配置"
        lines.append(
            f"| {_escape(parameter)} | {_escape(expected_text)} | {_escape(observed_text)} | {status} | {_escape(source)} |"
        )
        findings.append(
            AuditFinding(
                item=f"已知锚点：{parameter}",
                value=observed_text,
                source=source,
                status="OK" if consistent is True else "WARN" if consistent is False else "UNKNOWN",
                note=f"锚点来源：{anchor['source']}；期望 {expected_text}。",
            )
        )

    for label in ("SCR02", "SCR03", "SCR05"):
        matched = [record for record in records if label.lower() in record.run_id.lower()]
        seeds = sorted({record.seed for record in matched if record.seed != "无法判定"})
        consistent = seeds == ["1", "2"] if matched else None
        status = "一致" if consistent is True else "不一致" if consistent is False else "无法判定"
        source = "; ".join(record.config_path for record in matched) or "扫描根目录未找到该筛选组"
        lines.append(
            f"| {label} 训练种子 | 1、2（2 个） | {_escape(', '.join(seeds) or '未找到')} | {status} | {_escape(source)} |"
        )
        findings.append(
            AuditFinding(
                item=f"已知锚点：{label} seeds",
                value=", ".join(seeds) or "未找到",
                source=source,
                status="OK" if consistent is True else "WARN" if consistent is False else "UNKNOWN",
                note="服务器锚点为 seed 1、2。",
            )
        )
    return "\n".join(lines), findings


def _introduction_markdown(history: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "| 超参 | 新取值 | 首次出现时间 | 前一取值批次最晚完成时间 | source |",
        "| --- | --- | --- | --- | --- |",
    ]
    any_row = False
    for parameter, values in history.items():
        timeline: list[tuple[str, str, str, list[str]]] = []
        for value, bucket in values.items():
            times = sorted(bucket["times"])
            if times:
                timeline.append((times[0], value, times[-1], bucket["sources"]))
        timeline.sort()
        prior_end = "无前序取值"
        for first, value, last, sources in timeline:
            any_row = True
            lines.append(
                f"| {_escape(parameter)} | {_escape(value)} | {_escape(first)} | {_escape(prior_end)} | {_escape('; '.join(sorted(set(sources))))} |"
            )
            if prior_end == "无前序取值" or last > prior_end:
                prior_end = last
    if not any_row:
        lines.append("| 未找到 | 无法判定 | 无法判定 | 无法判定 | 扫描根目录 |")
    return "\n".join(lines)


def _model_markdown(project_root: Path) -> str:
    model_rows = _scan_model_parameters(project_root)
    cli_rows = _scan_cli_defaults(project_root)
    lines = [
        "| 变量名 | 默认值 | 作用域 | source |",
        "| --- | --- | --- | --- |",
    ]
    for row in [*cli_rows, *model_rows]:
        lines.append(
            f"| {_escape(row['name'])} | {_escape(row['default'])} | {_escape(row['scope'])} | {_escape(row['source'])} |"
        )
    if not cli_rows and not model_rows:
        lines.append("| 未找到 | 无法判定 | 无法判定 | model/mraf.py、utils/options.py |")
    lines.extend(
        [
            "",
            "说明：带 `mraf_` 前缀的 CLI 变量定义在 `utils/options.py`；"
            "`model/mraf.py` 内部使用去前缀参数（例如 `gate_logit_scale`、"
            "`target_temperature`）。表格按源码实际名称分别列出，不将二者伪装成同一行。",
        ]
    )
    return "\n".join(lines)


def _git_markdown(rows: list[dict[str, str]]) -> str:
    lines = [
        "| 超参 | git 时间线 | source |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {_escape(row['parameter'])} | {_escape(row['timeline'])} | {_escape(row['source'])} |"
        )
    return "\n".join(lines)


def _declared_range_markdown(
    records: list[RunRecord], history: dict[str, dict[str, dict[str, Any]]]
) -> str:
    observed = set(history.get("mraf_target_temperature", {}))
    declared = {"0.2", "0.3", "0.5"}
    only_observed = sorted(observed - declared)
    missing = sorted(declared - observed)
    sources = sorted(
        {
            evidence.source
            for record in records
            for name, evidence in record.values.items()
            if name == "mraf_target_temperature"
        }
    )
    return "\n".join(
        [
            "| 项目 | 值 | source |",
            "| --- | --- | --- |",
            "| 论文声明的 Tr 预筛选范围 | {0.2, 0.3, 0.5}；论文表述为 3 个种子 | 任务书给定的论文声明 |",
            f"| 本次实测 Tr 取值集合 | {_escape('{' + ', '.join(sorted(observed)) + '}' if observed else '未找到')} | {_escape('; '.join(sources) or '扫描根目录')} |",
            f"| 实测存在、声明范围外 | {_escape(', '.join(only_observed) or '无')} | {_escape('; '.join(sources) or '扫描根目录')} |",
            f"| 声明范围内但本次未找到 | {_escape(', '.join(missing) or '无')} | {_escape('; '.join(sources) or '扫描根目录')} |",
            "| 服务器确认的 SCR02/SCR03/SCR05 种子 | 各为 seed 1、2，共 2 个训练种子 | 用户提供的 Stage 0 修正 6 |",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="只读扫描配置、日志与 Git 历史，还原 MRAF/MDRC 超参数探索记录。"
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        type=Path,
        help="一个或多个实验资产根目录。",
    )
    parser.add_argument(
        "--repo",
        required=True,
        type=Path,
        help="MRAF 仓库根或其 IRRA-main Python 项目根。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("audit_out"),
        help="原始 JSON 输出目录，默认 audit_out/。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = _resolve_project_root(args.repo)
    requested_repo = args.repo.expanduser().resolve(strict=False)
    roots = [path.expanduser().resolve(strict=False) for path in args.roots]
    records, scan_findings = _discover_runs(roots)
    history = _history(records)
    git_rows = _git_timelines(requested_repo)
    anchor_table, anchor_findings = _anchor_markdown(records, history)
    findings = [*scan_findings, *anchor_findings]
    direct_mraf = requested_repo / "model" / "mraf.py"
    nested_mraf = requested_repo / "IRRA-main" / "model" / "mraf.py"
    if not direct_mraf.is_file() and not nested_mraf.is_file():
        findings.append(
            AuditFinding(
                item="MRAF 仓库定位",
                value="无法判定",
                source=requested_repo.as_posix(),
                status="UNKNOWN",
                note=f"未找到 model/mraf.py；报告安全写入 {project_root.as_posix()}/docs/audit。",
            )
        )
    if not records:
        findings.append(
            AuditFinding(
                item="超参数运行记录",
                value="未找到",
                source=", ".join(path.as_posix() for path in roots),
                status="UNKNOWN",
                note="未发现可解析的 config；其余静态与 Git 检查仍已执行。",
            )
        )

    unknowns = [finding for finding in findings if finding.status == "UNKNOWN"]
    unknown_lines = [
        "| 检查项 | 值 | source | note |",
        "| --- | --- | --- | --- |",
    ]
    if unknowns:
        for finding in unknowns:
            unknown_lines.append(
                f"| {_escape(finding.item)} | {_escape(finding.value)} | {_escape(finding.source)} | {_escape(finding.note)} |"
            )
    else:
        unknown_lines.append("| 无 | 无 | 本次审计结果 | 所有检查项均取得可报告结果 |")

    markdown = "\n\n".join(
        [
            "# MRAF 超参数探索历史审计",
            "本报告只汇总文件与 Git 中可复核的事实，不对候选选择过程作主观判断。",
            "## 一、各超参历史取值总表\n\n" + _history_markdown(history),
            "## 二、代码默认值变更时间线（Git）\n\n" + _git_markdown(git_rows),
            "## 三、取值引入时间线\n\n" + _introduction_markdown(history),
            "## 四、论文声明范围与实测历史的差异\n\n"
            + _declared_range_markdown(records, history),
            "## 五、已知锚点比对\n\n" + anchor_table,
            "## 六、MRAF 代码侧静态参数清点\n\n" + _model_markdown(project_root),
            "## 七、未解决项\n\n" + "\n".join(unknown_lines),
        ]
    ) + "\n"

    report_path = project_root / "docs" / "audit" / "HPARAM_HISTORY.md"
    write_report(report_path, [markdown])

    out_dir = resolve_audit_output_dir(project_root, args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "hparam_history.json"
    payload = {
        "roots": [path.as_posix() for path in roots],
        "repo": requested_repo.as_posix(),
        "runs": [record.to_json() for record in records],
        "history": {
            parameter: {
                value: {
                    **bucket,
                    "seeds": sorted(bucket["seeds"]),
                }
                for value, bucket in values.items()
            }
            for parameter, values in history.items()
        },
        "findings": [asdict(finding) for finding in findings],
    }
    with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")

    print(f"已写入报告：{report_path}")
    print(f"已写入原始记录：{raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
