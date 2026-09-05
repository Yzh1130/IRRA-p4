"""审计脚本共用的只读、安全降级工具。"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import pickle
import subprocess
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableSequence, Optional, Sequence, Tuple, Union


LOGGER = logging.getLogger(__name__)

_DEFAULT_EXCLUDES = (
    "site-packages",
    "__pycache__",
    str(Path(".git") / "objects"),
    ".cache",
)
_READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {
        "branch",
        "diff",
        "log",
        "ls-files",
        "remote",
        "rev-parse",
        "show",
        "status",
        "tag",
        "worktree",
    }
)
_LOAD_ERRORS: dict[str, str] = {}


class _RestrictedUnpickler(pickle.Unpickler):
    """只允许由基础容器/标量组成的标注 pickle，拒绝任意全局对象。"""

    _ALLOWED_BUILTINS = {
        "bool",
        "bytearray",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "set",
        "str",
        "tuple",
    }

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins" and name in self._ALLOWED_BUILTINS:
            import builtins

            return getattr(builtins, name)
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        raise pickle.UnpicklingError(f"拒绝不安全的 pickle 全局对象：{module}.{name}")


@dataclass(frozen=True)
class AuditFinding:
    """一条可追溯的审计结论。"""

    item: str
    value: Any
    source: str
    status: str = "UNKNOWN"
    note: str = ""

    def __post_init__(self) -> None:
        normalized = str(self.status).upper()
        if normalized not in {"OK", "WARN", "UNKNOWN"}:
            raise ValueError("status 必须是 OK、WARN 或 UNKNOWN")
        object.__setattr__(self, "status", normalized)

    def to_dict(self) -> dict[str, Any]:
        """返回可直接写入 JSON 的字典。"""

        return asdict(self)


def _path_key(path: Union[str, os.PathLike[str]]) -> str:
    try:
        return str(Path(path).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, TypeError, ValueError):
        return str(path)


def _record_load_error(
    path: Union[str, os.PathLike[str]],
    exc: BaseException,
    errors: Optional[MutableSequence[str]],
) -> None:
    reason = f"{type(exc).__name__}: {exc}"
    key = _path_key(path)
    _LOAD_ERRORS[key] = reason
    if errors is not None:
        errors.append(f"{key}: {reason}")
    LOGGER.warning("读取失败 %s：%s", key, reason)


def get_load_error(
    path: Optional[Union[str, os.PathLike[str]]] = None,
) -> Union[Optional[str], dict[str, str]]:
    """取得安全加载器记录的失败原因；不传路径时返回全部记录的副本。"""

    if path is None:
        return dict(_LOAD_ERRORS)
    return _LOAD_ERRORS.get(_path_key(path))


def safe_json_load(
    path: Union[str, os.PathLike[str]],
    *,
    errors: Optional[MutableSequence[str]] = None,
) -> Any:
    """以 UTF-8 读取 JSON；任何失败都记录原因并返回 ``None``。"""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline=None) as handle:
            value = json.load(handle)
    except Exception as exc:  # 审计工具必须对未知、损坏格式优雅降级。
        _record_load_error(path, exc, errors)
        return None
    _LOAD_ERRORS.pop(_path_key(path), None)
    return value


def safe_pickle_load(
    path: Union[str, os.PathLike[str]],
    *,
    errors: Optional[MutableSequence[str]] = None,
) -> Any:
    """以二进制方式读取 pickle；任何失败都记录原因并返回 ``None``。

    pickle 是二进制格式，无法指定文本编码；调用者只应审计可信的本地实验资产。
    """

    source = Path(path)
    try:
        with source.open("rb") as handle:
            value = _RestrictedUnpickler(handle).load()
    except Exception as exc:  # 审计工具必须对未知、损坏格式优雅降级。
        _record_load_error(path, exc, errors)
        return None
    _LOAD_ERRORS.pop(_path_key(path), None)
    return value


def _is_excluded(path: Path, exclusions: Sequence[str]) -> bool:
    normalized = path.as_posix().casefold()
    for exclusion in exclusions:
        token = Path(str(exclusion)).as_posix().casefold()
        if token and token in normalized:
            return True
    return False


def find_files(
    root: Union[str, os.PathLike[str]],
    patterns: Union[str, Sequence[str]],
    exclude_substrings: Optional[Sequence[str]] = None,
    max_depth: Optional[int] = None,
) -> list[Path]:
    """按 glob 模式查找文件，不跟随目录符号链接，访问失败时跳过。

    ``max_depth=0`` 只检查 ``root`` 直属文件；``None`` 表示不限制深度。
    """

    base = Path(root)
    if not base.is_dir():
        return []
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth 不能小于 0")

    wanted = (patterns,) if isinstance(patterns, str) else tuple(str(pattern) for pattern in patterns)
    if not wanted:
        return []
    exclusions = tuple(_DEFAULT_EXCLUDES if exclude_substrings is None else exclude_substrings)
    if _is_excluded(base, exclusions):
        return []

    found: list[Path] = []
    pending: list[Tuple[Path, int]] = [(base, 0)]
    while pending:
        directory, depth = pending.pop()
        try:
            entries = list(directory.iterdir())
        except (OSError, RuntimeError):
            continue
        for entry in entries:
            if _is_excluded(entry, exclusions):
                continue
            try:
                is_directory = entry.is_dir()
                is_symlink = entry.is_symlink()
            except OSError:
                continue
            if is_directory:
                if not is_symlink and (max_depth is None or depth < max_depth):
                    pending.append((entry, depth + 1))
                continue
            try:
                relative = entry.relative_to(base)
            except ValueError:
                relative = entry
            if any(
                fnmatch.fnmatchcase(entry.name, pattern)
                or fnmatch.fnmatchcase(relative.as_posix(), Path(pattern).as_posix())
                for pattern in wanted
            ):
                found.append(entry)
    return sorted(found, key=lambda item: item.as_posix().casefold())


def _validate_git_command(arguments: Sequence[str]) -> None:
    if not arguments:
        raise ValueError("run_git 至少需要一个 Git 子命令")
    subcommand = arguments[0]
    if subcommand not in _READ_ONLY_GIT_SUBCOMMANDS:
        allowed = ", ".join(sorted(_READ_ONLY_GIT_SUBCOMMANDS))
        raise ValueError(f"拒绝非白名单 Git 子命令 {subcommand!r}；允许：{allowed}")

    remainder = tuple(arguments[1:])
    output_flags = {"-o", "--output"}
    if any(arg in output_flags or arg.startswith("--output=") for arg in remainder):
        raise ValueError("拒绝可能写文件的 Git 输出参数")

    if subcommand == "worktree":
        actions = [arg for arg in remainder if not arg.startswith("-")]
        if not actions or actions[0] != "list":
            raise ValueError("run_git 仅允许只读的 `git worktree list`")
    elif subcommand == "branch":
        forbidden = {
            "-c",
            "-C",
            "-d",
            "-D",
            "-f",
            "-m",
            "-M",
            "--copy",
            "--delete",
            "--edit-description",
            "--force",
            "--move",
            "--set-upstream-to",
            "--unset-upstream",
        }
        if any(arg in forbidden or arg.startswith("--set-upstream-to=") for arg in remainder):
            raise ValueError("拒绝可能修改分支的 Git 参数")
        positional = [arg for arg in remainder if not arg.startswith("-")]
        listing_requested = any(arg in {"-l", "--list"} for arg in remainder)
        if positional and not listing_requested:
            raise ValueError("拒绝可能创建分支的位置参数；列出匹配项时必须显式使用 --list")
    elif subcommand == "tag":
        forbidden = {
            "-a",
            "-d",
            "-f",
            "-F",
            "-m",
            "-s",
            "-u",
            "--annotate",
            "--delete",
            "--file",
            "--force",
            "--local-user",
            "--message",
            "--sign",
        }
        if any(arg in forbidden for arg in remainder):
            raise ValueError("拒绝可能创建、修改或删除 tag 的 Git 参数")
        positional = [arg for arg in remainder if not arg.startswith("-")]
        listing_requested = any(arg in {"-l", "--list"} for arg in remainder)
        if positional and not listing_requested:
            raise ValueError("拒绝可能创建 tag 的位置参数；列出匹配项时必须显式使用 --list")
    elif subcommand == "remote":
        positional = [arg for arg in remainder if not arg.startswith("-")]
        if positional and positional[0] not in {"get-url"}:
            raise ValueError("run_git 的 remote 子命令仅允许列表或 get-url")
        if any(arg not in {"-v", "--verbose", "--all", "--push", "get-url"} and arg.startswith("-") for arg in remainder):
            raise ValueError("拒绝未获准的 git remote 参数")
    elif any(arg == "--ext-diff" for arg in remainder):
        raise ValueError("拒绝调用外部 diff 程序")


def run_git(
    repo_dir: Union[str, os.PathLike[str]],
    *args: Union[str, os.PathLike[str]],
    timeout: float = 60.0,
) -> subprocess.CompletedProcess[str]:
    """在 ``repo_dir`` 调用经过白名单校验的只读 Git 命令。"""

    arguments = tuple(str(arg) for arg in args)
    _validate_git_command(arguments)
    command = ["git", "-C", str(Path(repo_dir)), arguments[0]]
    if arguments[0] == "diff":
        command.append("--no-ext-diff")
    command.extend(arguments[1:])
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(
            args=command,
            returncode=127,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
        )


def resolve_audit_output_dir(
    project_root: Union[str, os.PathLike[str]],
    requested: Union[str, os.PathLike[str]],
) -> Path:
    """把 ``--out`` 限定在目标项目的 ``audit_out/`` 内。"""

    project = Path(project_root).expanduser().resolve(strict=False)
    allowed_root = (project / "audit_out").resolve(strict=False)
    candidate = Path(requested).expanduser()
    if not candidate.is_absolute():
        candidate = project / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            f"--out 只能位于 {allowed_root} 内；收到 {resolved}"
        ) from exc
    return resolved


Section = Union[str, Tuple[str, str]]


def write_report(
    path: Union[str, os.PathLike[str]],
    sections: Union[Mapping[str, str], Iterable[Section]],
) -> Path:
    """把有序章节写为 UTF-8 Markdown，并统一使用 LF 换行。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    items: Iterable[Section]
    if isinstance(sections, Mapping):
        items = sections.items()
    else:
        items = sections

    chunks: list[str] = []
    for section in items:
        if isinstance(section, str):
            chunk = section
        else:
            title, body = section
            heading = title if title.lstrip().startswith("#") else f"## {title}"
            chunk = f"{heading}\n\n{body}" if body else heading
        chunks.append(chunk.replace("\r\n", "\n").replace("\r", "\n").strip())
    content = "\n\n".join(chunk for chunk in chunks if chunk)
    if content:
        content += "\n"
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    return destination
