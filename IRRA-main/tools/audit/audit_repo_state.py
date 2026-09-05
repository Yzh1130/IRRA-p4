"""生成只读、可追溯的 Git 仓库状态审计报告。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

if __package__:
    from .common import AuditFinding, resolve_audit_output_dir, run_git, write_report
else:  # 兼容 ``python tools/audit/audit_repo_state.py --help``。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.audit.common import (  # type: ignore
        AuditFinding,
        resolve_audit_output_dir,
        run_git,
        write_report,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_AUDIT_DIR = PROJECT_ROOT / "docs" / "audit"
_TEXT_SUFFIXES = {
    ".bat",
    ".cfg",
    ".csv",
    ".ini",
    ".iml",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_TEXT_NAMES = {".gitignore", ".gitattributes", "LICENSE", "Makefile"}
_EOL_SCAN_MAX_BYTES = 25 * 1024 * 1024


def _source(repo: Path, *arguments: str) -> str:
    rendered = " ".join(arguments)
    return f"git {rendered}（repo={repo}）"


def _error_note(returncode: int, stderr: str) -> str:
    detail = stderr.strip() or "Git 未返回错误详情"
    return f"returncode={returncode}; {detail}"


def _cell(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, (list, tuple)):
        text = "<br>".join(str(item) for item in value)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\r\n", "<br>").replace("\r", "<br>").replace("\n", "<br>")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    escaped_headers = [_cell(header) for header in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in escaped_headers) + " |",
    ]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _finding_table(findings: Sequence[AuditFinding]) -> str:
    return _table(
        ("检查项", "值", "状态", "来源", "备注"),
        [(item.item, item.value, item.status, item.source, item.note) for item in findings],
    )


def _parse_remote_lines(output: str, source: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "\t" in stripped:
            name, remainder = stripped.split("\t", 1)
        else:
            pieces = stripped.split(maxsplit=1)
            name = pieces[0]
            remainder = pieces[1] if len(pieces) > 1 else ""
        kind = ""
        url = remainder
        if remainder.endswith(")") and " (" in remainder:
            url, kind_text = remainder.rsplit(" (", 1)
            kind = kind_text[:-1]
        rows.append(
            {
                "name": name,
                "url": url,
                "kind": kind or "无法判定",
                "status": "OK",
                "source": source,
                "note": "",
            }
        )
    return rows


def _parse_worktrees(output: str, source: str) -> list[dict[str, str]]:
    worktrees: list[dict[str, str]] = []
    block: dict[str, str] = {}
    flags: set[str] = set()

    def flush() -> None:
        if not block and not flags:
            return
        branch = block.get("branch", "")
        prefix = "refs/heads/"
        if branch.startswith(prefix):
            branch = branch[len(prefix) :]
        if "detached" in flags:
            branch = "detached HEAD"
        worktrees.append(
            {
                "path": block.get("worktree", "无法判定"),
                "head": block.get("HEAD", "无法判定"),
                "branch": branch or "无法判定",
                "locked": block.get("locked", "是") if "locked" in flags else "否",
                "prunable": block.get("prunable", "是") if "prunable" in flags else "否",
                "status": "WARN" if "prunable" in flags else "OK",
                "source": source,
                "note": "",
            }
        )

    for line in list(output.splitlines()) + [""]:
        if not line.strip():
            flush()
            block = {}
            flags = set()
            continue
        key, separator, value = line.partition(" ")
        if separator:
            block[key] = value
            flags.add(key)
        else:
            flags.add(key)
    return worktrees


def _parse_status_z(output: str) -> list[dict[str, str]]:
    records = output.split("\0")
    changes: list[dict[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        code = record[:2] if len(record) >= 2 else "??"
        path = record[3:] if len(record) >= 4 else record
        original_path = ""
        if "R" in code or "C" in code:
            if index < len(records):
                original_path = records[index]
                index += 1
        changes.append(
            {
                "code": code,
                "path": path,
                "original_path": original_path,
            }
        )
    return changes


def _parse_log(output: str, source: str) -> list[dict[str, str]]:
    commits: list[dict[str, str]] = []
    for record in output.split("\x1e"):
        cleaned = record.strip()
        if not cleaned:
            continue
        fields = cleaned.split("\x1f", 3)
        if len(fields) != 4:
            commits.append(
                {
                    "hash": "无法判定",
                    "time": "无法判定",
                    "author": "无法判定",
                    "subject": cleaned,
                    "status": "UNKNOWN",
                    "source": source,
                    "note": "commit 记录无法按预期字段解析",
                }
            )
            continue
        commit_hash, timestamp, author, subject = fields
        commits.append(
            {
                "hash": commit_hash,
                "time": timestamp,
                "author": author,
                "subject": subject,
                "status": "OK",
                "source": source,
                "note": "",
            }
        )
    return commits


def _normalise_eol(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _scan_tracked_text_differences(repo: Path) -> dict[str, Any]:
    """绕过 index stat 缓存，按 HEAD blob 与工作树文本内容核实 EOL 差异。"""

    list_args = ("ls-files", "-z")
    list_source = _source(repo, *list_args)
    result = run_git(repo, *list_args)
    if result.returncode != 0:
        return {
            "status": "UNKNOWN",
            "source": list_source,
            "note": _error_note(result.returncode, result.stderr),
            "checked": 0,
            "eol_only_paths": [],
            "eol_plus_audit_ignore_paths": [],
            "semantic_paths": [],
            "skipped": [],
        }

    tracked = [item for item in result.stdout.split("\0") if item]
    eol_only: list[str] = []
    eol_plus_ignore: list[str] = []
    semantic: list[str] = []
    skipped: list[str] = []
    checked = 0
    for relative in tracked:
        relative_posix = relative.replace("\\", "/")
        path = repo / Path(relative)
        if relative_posix.startswith("IRRA-main/datasets/TextReID/"):
            continue
        if path.name not in _TEXT_NAMES and path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            if not path.is_file() or path.is_symlink():
                continue
            if path.stat().st_size > _EOL_SCAN_MAX_BYTES:
                skipped.append(f"{relative_posix}:超过 {_EOL_SCAN_MAX_BYTES} 字节")
                continue
            with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
                worktree_text = handle.read()
        except (OSError, UnicodeError) as exc:
            skipped.append(f"{relative_posix}:{type(exc).__name__}: {exc}")
            continue

        blob_result = run_git(repo, "show", f"HEAD:{relative_posix}")
        if blob_result.returncode != 0:
            skipped.append(
                f"{relative_posix}:git show 失败: "
                + (blob_result.stderr.strip() or f"returncode={blob_result.returncode}")
            )
            continue
        checked += 1
        blob_text = blob_result.stdout
        if worktree_text == blob_text:
            continue
        worktree_normalised = _normalise_eol(worktree_text)
        blob_normalised = _normalise_eol(blob_text)
        if worktree_normalised == blob_normalised:
            eol_only.append(relative_posix)
            continue
        if relative_posix == ".gitignore":
            lines = worktree_normalised.splitlines()
            if lines and lines[-1] == "audit_out/":
                without_allowed_append = "\n".join(lines[:-1]).rstrip("\n")
                if without_allowed_append == blob_normalised.rstrip("\n"):
                    eol_plus_ignore.append(relative_posix)
                    continue
        semantic.append(relative_posix)

    return {
        "status": "WARN" if eol_only or eol_plus_ignore or semantic else "OK",
        "source": (
            f"{list_source}; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 "
            "读取工作树后比较 CRLF/LF 归一化结果"
        ),
        "note": "不读取 datasets/TextReID，不写 index，不修改工作树。",
        "checked": checked,
        "eol_only_paths": eol_only,
        "eol_plus_audit_ignore_paths": eol_plus_ignore,
        "semantic_paths": semantic,
        "skipped": skipped,
    }


def collect_repo_state(repo_argument: Path) -> dict[str, Any]:
    """只读采集 Git 状态；任何单项失败均保留 UNKNOWN 后继续。"""

    captured_at = datetime.now().astimezone().isoformat(timespec="seconds")
    repo = repo_argument.expanduser().resolve(strict=False)
    unknowns: list[AuditFinding] = []
    basics: list[AuditFinding] = [
        AuditFinding(
            item="采集时间",
            value=captured_at,
            source="Python datetime.now().astimezone()",
            status="OK",
            note="本地时区 ISO 8601 时间",
        ),
        AuditFinding(
            item="命令行仓库路径",
            value=str(repo),
            source="命令行参数 --repo",
            status="OK" if repo.exists() else "WARN",
            note="路径未找到，仍将尝试 Git 探测" if not repo.exists() else "",
        ),
    ]

    top_args = ("rev-parse", "--show-toplevel")
    top_source = _source(repo, *top_args)
    top_result = run_git(repo, *top_args)
    if top_result.returncode == 0 and top_result.stdout.strip():
        repo_root = Path(top_result.stdout.strip()).resolve(strict=False)
        basics.append(AuditFinding("Git 顶层目录", str(repo_root), top_source, "OK", ""))
    else:
        repo_root = repo
        finding = AuditFinding(
            "Git 顶层目录",
            "无法判定",
            top_source,
            "UNKNOWN",
            _error_note(top_result.returncode, top_result.stderr),
        )
        basics.append(finding)
        unknowns.append(finding)

    branch_args = ("branch", "--show-current")
    branch_source = _source(repo_root, *branch_args)
    branch_result = run_git(repo_root, *branch_args)
    if branch_result.returncode == 0:
        branch = branch_result.stdout.strip()
        basics.append(
            AuditFinding(
                "当前 branch",
                branch or "detached HEAD",
                branch_source,
                "OK" if branch else "WARN",
                "" if branch else "当前 HEAD 未附着到本地分支",
            )
        )
    else:
        finding = AuditFinding(
            "当前 branch",
            "无法判定",
            branch_source,
            "UNKNOWN",
            _error_note(branch_result.returncode, branch_result.stderr),
        )
        basics.append(finding)
        unknowns.append(finding)

    head_args = ("rev-parse", "HEAD")
    head_source = _source(repo_root, *head_args)
    head_result = run_git(repo_root, *head_args)
    if head_result.returncode == 0 and head_result.stdout.strip():
        basics.append(AuditFinding("HEAD commit", head_result.stdout.strip(), head_source, "OK", ""))
        has_head = True
    else:
        finding = AuditFinding(
            "HEAD commit",
            "无法判定",
            head_source,
            "UNKNOWN",
            _error_note(head_result.returncode, head_result.stderr),
        )
        basics.append(finding)
        unknowns.append(finding)
        has_head = False

    remote_args = ("remote", "-v")
    remote_source = _source(repo_root, *remote_args)
    remote_result = run_git(repo_root, *remote_args)
    remotes: list[dict[str, str]] = []
    if remote_result.returncode == 0:
        remotes = _parse_remote_lines(remote_result.stdout, remote_source)
        if not remotes:
            remotes.append(
                {
                    "name": "未找到",
                    "url": "未找到",
                    "kind": "无法判定",
                    "status": "WARN",
                    "source": remote_source,
                    "note": "仓库未配置 remote",
                }
            )
    else:
        finding = AuditFinding(
            "remote 列表",
            "无法判定",
            remote_source,
            "UNKNOWN",
            _error_note(remote_result.returncode, remote_result.stderr),
        )
        unknowns.append(finding)
        remotes.append(
            {
                "name": "无法判定",
                "url": "无法判定",
                "kind": "无法判定",
                "status": finding.status,
                "source": finding.source,
                "note": finding.note,
            }
        )

    branch_groups: dict[str, list[dict[str, str]]] = {}
    for label, arguments in (
        ("本地", ("branch", "--list", "--format=%(refname:short)")),
        ("远程", ("branch", "--remotes", "--format=%(refname:short)")),
    ):
        command_source = _source(repo_root, *arguments)
        result = run_git(repo_root, *arguments)
        if result.returncode == 0:
            values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            branch_groups[label] = [
                {
                    "scope": label,
                    "name": value,
                    "status": "OK",
                    "source": command_source,
                    "note": "",
                }
                for value in values
            ]
            if not values:
                branch_groups[label] = [
                    {
                        "scope": label,
                        "name": "未找到",
                        "status": "WARN",
                        "source": command_source,
                        "note": f"未找到{label}分支",
                    }
                ]
        else:
            finding = AuditFinding(
                f"{label}分支列表",
                "无法判定",
                command_source,
                "UNKNOWN",
                _error_note(result.returncode, result.stderr),
            )
            unknowns.append(finding)
            branch_groups[label] = [
                {
                    "scope": label,
                    "name": "无法判定",
                    "status": finding.status,
                    "source": finding.source,
                    "note": finding.note,
                }
            ]

    tag_args = ("tag", "--list")
    tag_source = _source(repo_root, *tag_args)
    tag_result = run_git(repo_root, *tag_args)
    if tag_result.returncode == 0:
        tag_values = [line.strip() for line in tag_result.stdout.splitlines() if line.strip()]
        tags = [
            {"name": value, "status": "OK", "source": tag_source, "note": ""}
            for value in tag_values
        ]
        if not tags:
            tags = [
                {
                    "name": "未找到",
                    "status": "WARN",
                    "source": tag_source,
                    "note": "仓库没有 tag",
                }
            ]
    else:
        finding = AuditFinding(
            "tag 列表",
            "无法判定",
            tag_source,
            "UNKNOWN",
            _error_note(tag_result.returncode, tag_result.stderr),
        )
        unknowns.append(finding)
        tags = [
            {
                "name": "无法判定",
                "status": finding.status,
                "source": finding.source,
                "note": finding.note,
            }
        ]

    worktree_args = ("worktree", "list", "--porcelain")
    worktree_source = _source(repo_root, *worktree_args)
    worktree_result = run_git(repo_root, *worktree_args)
    if worktree_result.returncode == 0:
        worktrees = _parse_worktrees(worktree_result.stdout, worktree_source)
        if not worktrees:
            finding = AuditFinding(
                "worktree 列表",
                "无法判定",
                worktree_source,
                "UNKNOWN",
                "Git 命令成功，但未返回 worktree",
            )
            unknowns.append(finding)
            worktrees = [
                {
                    "path": "无法判定",
                    "head": "无法判定",
                    "branch": "无法判定",
                    "locked": "无法判定",
                    "prunable": "无法判定",
                    "status": finding.status,
                    "source": finding.source,
                    "note": finding.note,
                }
            ]
    else:
        finding = AuditFinding(
            "worktree 列表",
            "无法判定",
            worktree_source,
            "UNKNOWN",
            _error_note(worktree_result.returncode, worktree_result.stderr),
        )
        unknowns.append(finding)
        worktrees = [
            {
                "path": "无法判定",
                "head": "无法判定",
                "branch": "无法判定",
                "locked": "无法判定",
                "prunable": "无法判定",
                "status": finding.status,
                "source": finding.source,
                "note": finding.note,
            }
        ]

    status_args = ("status", "--porcelain=v1", "-z", "--untracked-files=all")
    status_source = _source(repo_root, *status_args)
    status_result = run_git(repo_root, *status_args)
    changes: list[dict[str, str]] = []
    if status_result.returncode == 0:
        changes = _parse_status_z(status_result.stdout)
        basics.append(
            AuditFinding(
                "未提交改动数量",
                len(changes),
                status_source,
                "WARN" if changes else "OK",
                "数量按 porcelain -z 记录计数",
            )
        )
        for change in changes:
            code = change["code"]
            path = change["path"]
            original_path = change["original_path"]
            if code == "??":
                change.update(
                    {
                        "diff_stat": "不适用（未跟踪文件）",
                        "status": "WARN",
                        "source": status_source,
                        "note": "git diff 不显示未跟踪文件",
                    }
                )
                continue
            diff_arguments = ["diff", "--stat"]
            if has_head:
                diff_arguments.append("HEAD")
            diff_arguments.extend(("--", path))
            if original_path:
                diff_arguments.append(original_path)
            diff_source = _source(repo_root, *diff_arguments)
            diff_result = run_git(repo_root, *diff_arguments)
            if diff_result.returncode == 0:
                change.update(
                    {
                        "diff_stat": diff_result.stdout.strip() or "Git 未返回 diff --stat 摘要",
                        "status": "WARN",
                        "source": f"{status_source}; {diff_source}",
                        "note": "",
                    }
                )
            else:
                finding = AuditFinding(
                    f"改动文件 diff --stat：{path}",
                    "无法判定",
                    diff_source,
                    "UNKNOWN",
                    _error_note(diff_result.returncode, diff_result.stderr),
                )
                unknowns.append(finding)
                change.update(
                    {
                        "diff_stat": "无法判定",
                        "status": finding.status,
                        "source": f"{status_source}; {finding.source}",
                        "note": finding.note,
                    }
                )
    else:
        finding = AuditFinding(
            "未提交改动清单",
            "无法判定",
            status_source,
            "UNKNOWN",
            _error_note(status_result.returncode, status_result.stderr),
        )
        basics.append(finding)
        unknowns.append(finding)
        changes = [
            {
                "code": "无法判定",
                "path": "无法判定",
                "original_path": "",
                "diff_stat": "无法判定",
                "status": finding.status,
                "source": finding.source,
                "note": finding.note,
            }
        ]

    log_args = (
        "log",
        "-n",
        "50",
        "--date=iso-strict",
        "--no-decorate",
        "--pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e",
    )
    log_source = _source(repo_root, *log_args)
    log_result = run_git(repo_root, *log_args)
    if log_result.returncode == 0:
        commits = _parse_log(log_result.stdout, log_source)
        if not commits:
            finding = AuditFinding(
                "最近 50 条 commit",
                "未找到",
                log_source,
                "UNKNOWN",
                "Git 命令成功，但没有可解析的 commit",
            )
            unknowns.append(finding)
            commits = [
                {
                    "hash": "未找到",
                    "time": "无法判定",
                    "author": "无法判定",
                    "subject": "未找到",
                    "status": finding.status,
                    "source": finding.source,
                    "note": finding.note,
                }
            ]
    else:
        finding = AuditFinding(
            "最近 50 条 commit",
            "无法判定",
            log_source,
            "UNKNOWN",
            _error_note(log_result.returncode, log_result.stderr),
        )
        unknowns.append(finding)
        commits = [
            {
                "hash": "无法判定",
                "time": "无法判定",
                "author": "无法判定",
                "subject": "无法判定",
                "status": finding.status,
                "source": finding.source,
                "note": finding.note,
            }
        ]

    tracked_text_audit = _scan_tracked_text_differences(repo_root)
    if tracked_text_audit["status"] == "UNKNOWN":
        finding = AuditFinding(
            "tracked 文本 EOL 核验",
            "无法判定",
            tracked_text_audit["source"],
            "UNKNOWN",
            tracked_text_audit["note"],
        )
        unknowns.append(finding)

    return {
        "captured_at": captured_at,
        "repo_argument": str(repo),
        "repo_root": str(repo_root),
        "basics": [finding.to_dict() for finding in basics],
        "remotes": remotes,
        "branches": branch_groups,
        "tags": tags,
        "worktrees": worktrees,
        "changes": changes,
        "commits": commits,
        "tracked_text_audit": tracked_text_audit,
        "unknowns": [finding.to_dict() for finding in unknowns],
        "status_source": status_source,
    }


def _finding_from_dict(value: dict[str, Any]) -> AuditFinding:
    return AuditFinding(
        item=str(value.get("item", "无法判定")),
        value=value.get("value", "无法判定"),
        source=str(value.get("source", "无法判定")),
        status=str(value.get("status", "UNKNOWN")),
        note=str(value.get("note", "")),
    )


def render_repo_report(state: dict[str, Any]) -> list[tuple[str, str]]:
    basics = [_finding_from_dict(item) for item in state["basics"]]
    remotes = state["remotes"]
    branches = state["branches"]["本地"] + state["branches"]["远程"]
    tags = state["tags"]
    worktrees = state["worktrees"]
    changes = state["changes"]
    commits = state["commits"]
    unknowns = [_finding_from_dict(item) for item in state["unknowns"]]
    text_audit = state["tracked_text_audit"]

    change_rows: list[Sequence[Any]]
    if changes:
        change_rows = [
            (
                item["code"],
                item["path"],
                item.get("original_path", ""),
                item["diff_stat"],
                item["status"],
                item["source"],
                item["note"],
            )
            for item in changes
        ]
    else:
        change_rows = [
            (
                "--",
                "未找到未提交改动",
                "",
                "不适用",
                "OK",
                state["status_source"],
                "工作树在采集时为 clean",
            )
        ]

    unresolved = (
        _finding_table(unknowns)
        if unknowns
        else _table(
            ("检查项", "值", "状态", "来源", "备注"),
            [("UNKNOWN 检查项", "未找到", "OK", "本报告中 status 字段汇总", "没有 UNKNOWN 项")],
        )
    )

    return [
        (
            "# 仓库状态审计报告",
            "本报告仅记录 Git 只读命令在采集时返回的事实；状态含义不替代人工裁定。",
        ),
        ("一、基本状态", _finding_table(basics)),
        (
            "二、Remote 列表",
            _table(
                ("名称", "URL", "用途", "状态", "来源", "备注"),
                [
                    (item["name"], item["url"], item["kind"], item["status"], item["source"], item["note"])
                    for item in remotes
                ],
            ),
        ),
        (
            "三、本地与远程分支",
            _table(
                ("范围", "分支", "状态", "来源", "备注"),
                [(item["scope"], item["name"], item["status"], item["source"], item["note"]) for item in branches],
            ),
        ),
        (
            "四、Tags",
            _table(
                ("Tag", "状态", "来源", "备注"),
                [(item["name"], item["status"], item["source"], item["note"]) for item in tags],
            ),
        ),
        (
            "五、Worktree 列表",
            _table(
                ("路径", "HEAD", "分支", "Locked", "Prunable", "状态", "来源", "备注"),
                [
                    (
                        item["path"],
                        item["head"],
                        item["branch"],
                        item["locked"],
                        item["prunable"],
                        item["status"],
                        item["source"],
                        item["note"],
                    )
                    for item in worktrees
                ],
            ),
        ),
        (
            "六、未提交改动与逐文件 diff --stat",
            _table(
                ("XY", "路径", "原路径", "diff --stat", "状态", "来源", "备注"),
                change_rows,
            ),
        ),
        (
            "七、最近 50 条 Commit",
            _table(
                ("Hash", "时间", "作者", "标题", "状态", "来源", "备注"),
                [
                    (
                        item["hash"],
                        item["time"],
                        item["author"],
                        item["subject"],
                        item["status"],
                        item["source"],
                        item["note"],
                    )
                    for item in commits
                ],
            ),
        ),
        (
            "八、tracked 文本内容与 EOL 核验",
            _table(
                ("检查项", "数量/值", "状态", "来源", "备注"),
                (
                    ("检查的 tracked 文本文件", text_audit["checked"], text_audit["status"], text_audit["source"], text_audit["note"]),
                    ("仅 EOL 不同", len(text_audit["eol_only_paths"]), "WARN" if text_audit["eol_only_paths"] else "OK", text_audit["source"], text_audit["eol_only_paths"]),
                    ("EOL + 允许追加 audit_out/", len(text_audit["eol_plus_audit_ignore_paths"]), "WARN" if text_audit["eol_plus_audit_ignore_paths"] else "OK", text_audit["source"], text_audit["eol_plus_audit_ignore_paths"]),
                    ("其它语义差异", len(text_audit["semantic_paths"]), "WARN" if text_audit["semantic_paths"] else "OK", text_audit["source"], text_audit["semantic_paths"]),
                    ("跳过/无法核验", len(text_audit["skipped"]), "WARN" if text_audit["skipped"] else "OK", text_audit["source"], text_audit["skipped"]),
                ),
            ),
        ),
        ("九、未解决项", unresolved),
    ]


def render_dirty_snapshot(state: dict[str, Any]) -> list[tuple[str, str]]:
    changes = state["changes"]
    text_audit = state["tracked_text_audit"]
    if changes:
        rows = [
            (
                item["code"],
                item["path"],
                item.get("original_path", ""),
                item["diff_stat"],
                item["status"],
                item["source"],
                item["note"],
            )
            for item in changes
        ]
    else:
        rows = [
            (
                "--",
                "未找到未提交改动",
                "",
                "不适用",
                "OK",
                state["status_source"],
                "工作树在采集时为 clean",
            )
        ]
    metadata = [
        AuditFinding("采集时间", state["captured_at"], "Python datetime.now().astimezone()", "OK", ""),
        AuditFinding("Git 顶层目录", state["repo_root"], "git rev-parse --show-toplevel 的采集结果", "OK", ""),
        AuditFinding(
            "快照记录数",
            len(changes),
            state["status_source"],
            "WARN" if changes else "OK",
            "报告与原始 JSON 均在本次状态采集之后写入",
        ),
        AuditFinding(
            "Stage 0 前既有 tracked EOL 基线（合计）",
            len(text_audit["eol_only_paths"]) + len(text_audit["eol_plus_audit_ignore_paths"]),
            text_audit["source"],
            "WARN" if text_audit["eol_only_paths"] or text_audit["eol_plus_audit_ignore_paths"] else "OK",
            "仅 EOL 差异 + `.gitignore` 的 EOL 差异及本任务允许追加行；不会暂存这些换行改写",
        ),
        AuditFinding(
            "tracked 文本仅 EOL 差异",
            len(text_audit["eol_only_paths"]),
            text_audit["source"],
            "WARN" if text_audit["eol_only_paths"] else "OK",
            "；".join(text_audit["eol_only_paths"]),
        ),
        AuditFinding(
            "EOL 差异且仅追加 audit_out/",
            len(text_audit["eol_plus_audit_ignore_paths"]),
            text_audit["source"],
            "WARN" if text_audit["eol_plus_audit_ignore_paths"] else "OK",
            "；".join(text_audit["eol_plus_audit_ignore_paths"]),
        ),
        AuditFinding(
            "其它 tracked 文本语义差异",
            len(text_audit["semantic_paths"]),
            text_audit["source"],
            "WARN" if text_audit["semantic_paths"] else "OK",
            "；".join(text_audit["semantic_paths"]),
        ),
    ]
    return [
        (
            "# Worktree Dirty Snapshot",
            "该文件是一次性事实快照，仅描述采集时 Git porcelain 与逐文件 diff --stat 的输出。",
        ),
        ("一、快照元数据", _finding_table(metadata)),
        (
            "二、未提交改动",
            _table(
                ("XY", "路径", "原路径", "diff --stat", "状态", "来源", "备注"),
                rows,
            ),
        ),
        (
            "三、EOL 内容核验清单",
            _table(
                ("类别", "文件", "source", "status"),
                [
                    *(('仅 EOL', path, text_audit['source'], 'WARN') for path in text_audit['eol_only_paths']),
                    *(('EOL + audit_out/', path, text_audit['source'], 'WARN') for path in text_audit['eol_plus_audit_ignore_paths']),
                    *(('其它语义差异', path, text_audit['source'], 'WARN') for path in text_audit['semantic_paths']),
                ]
                or [("无", "未找到", text_audit["source"], "OK")],
            ),
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="只读采集 Git 仓库状态并生成中文审计报告。")
    parser.add_argument("--repo", type=Path, required=True, help="待审计仓库或其子目录路径。")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("audit_out"),
        help="原始 JSON 输出目录（默认：audit_out）。",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    state = collect_repo_state(args.repo)

    output_dir = resolve_audit_output_dir(PROJECT_ROOT, args.out)
    raw_path = output_dir / "repo_state.json"
    report_path = DOCS_AUDIT_DIR / "REPO_STATE.md"
    snapshot_path = DOCS_AUDIT_DIR / "WORKTREE_DIRTY_SNAPSHOT.md"

    output_dir.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_report(report_path, render_repo_report(state))
    write_report(snapshot_path, render_dirty_snapshot(state))

    print(f"已生成原始记录：{raw_path}")
    print(f"已生成仓库报告：{report_path}")
    print(f"已生成工作树快照：{snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
