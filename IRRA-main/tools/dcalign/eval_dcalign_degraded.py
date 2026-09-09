#!/usr/bin/env python3
"""Evaluate one DCAlign checkpoint under exactly one RGB-T degradation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping

import torch
import torch.nn.functional as F


COPY_HEADER_BYTES = (
    b"# DCALIGN_DEGRADATION_COPY_HEADER_V1\n"
    b"# This file carries a byte-identical payload copied from\n"
    b"# IRRA-main/utils/eval_degradation.py so IRRA and DCAlign use exactly the same\n"
    b"# evaluation degradations. Synchronize this copy whenever the upstream changes.\n"
)
METRIC_NAMES = ("R1", "R5", "R10", "mAP", "mINP")
METRICS_HEADER = (
    "group",
    "seed",
    "scenario",
    "mode",
    "strength",
    "kernel",
    "dseed",
    "split",
    *METRIC_NAMES,
)
EXPECTED_SCORE_SHAPE = (2490, 1473)
CORE_DEGRADE_MODES = frozenset(
    {
        "none",
        "rgb_missing",
        "ir_missing",
        "rgb_noise",
        "ir_noise",
        "both_noise",
        "rgb_blur",
        "ir_blur",
        "both_blur",
    }
)
COPIED_DEGRADE_MODES = frozenset(
    {
        "rgb_brightness",
        "ir_brightness",
        "rgb_lowres",
        "ir_lowres",
        "rgb_jpeg",
        "ir_jpeg",
    }
)
SUPPORTED_DEGRADE_MODES = tuple(sorted(CORE_DEGRADE_MODES | COPIED_DEGRADE_MODES))
ALLOWED_GROUPS = ("DC-add", "DC-dyn", "DC-add-best", "DC-dyn-best")
COLLIDING_PACKAGES = ("datasets", "model", "utils")
EXPECTED_DCALIGN_HEAD = "1ca105f668fb71352d888a6c1f37e0b382461d0d"
EXPECTED_DCALIGN_DIRTY_PATHS = frozenset(
    {
        "datasets/bases.py",
        "model/build.py",
        "processor/processor.py",
        "test.py",
        "train.py",
        "utils/metrics.py",
        "utils/options.py",
    }
)
IRRA_RUNTIME_KIT_FILES = (
    "tools/dcalign/run_dcalign_grid.sh",
    "tools/dcalign/eval_dcalign_degraded.py",
    "tools/dcalign/eval_degradation.py",
    "tools/dcalign/check_degradation_parity.py",
    "utils/metrics.py",
    "utils/eval_degradation.py",
)


class ImportProvenanceError(RuntimeError):
    """Raised when a generic Python import resolves outside its required tree."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_output(root: Path, *arguments: str) -> bytes:
    command = ["git", "-C", str(root), *arguments]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", b"")
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"DCAlign git provenance command failed: {command!r}; stderr={detail}"
        ) from error
    return completed.stdout


def audit_dcalign_source_state(dcalign_root: Path) -> dict[str, Any]:
    head = _git_output(dcalign_root, "rev-parse", "HEAD").decode().strip()
    if head != EXPECTED_DCALIGN_HEAD:
        raise RuntimeError(
            "DCAlign HEAD mismatch: "
            f"expected={EXPECTED_DCALIGN_HEAD}, actual={head}"
        )
    dirty_raw = _git_output(
        dcalign_root, "diff", "--name-only", "-z", "HEAD", "--"
    )
    dirty_paths = {
        item.decode("utf-8", errors="surrogateescape")
        for item in dirty_raw.split(b"\0")
        if item
    }
    if dirty_paths != EXPECTED_DCALIGN_DIRTY_PATHS:
        raise RuntimeError(
            "DCAlign tracked dirty path set mismatch: "
            f"expected={sorted(EXPECTED_DCALIGN_DIRTY_PATHS)}, "
            f"actual={sorted(dirty_paths)}"
        )
    untracked_python_raw = _git_output(
        dcalign_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "*.py",
    )
    untracked_python = sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in untracked_python_raw.split(b"\0")
        if item
    )
    if untracked_python:
        raise RuntimeError(
            "untracked Python files could alter DCAlign imports: "
            f"expected=none, actual={untracked_python}"
        )
    tracked_diff = _git_output(
        dcalign_root,
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "HEAD",
        "--",
    )
    state = {
        "head": head,
        "tracked_dirty_paths": sorted(dirty_paths),
        "tracked_diff_sha256": _sha256(tracked_diff),
    }
    print(
        "[DCALIGN_SOURCE_STATE] "
        + json.dumps(state, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return state


def _assert_pinned_source_state(
    source_state: Mapping[str, Any],
    expected_head: str,
    expected_diff_sha256: str,
) -> None:
    expected = {
        "head": expected_head,
        "tracked_diff_sha256": expected_diff_sha256,
    }
    for key, expected_value in expected.items():
        actual = source_state.get(key)
        if actual != expected_value:
            raise RuntimeError(
                "DCAlign source changed after the runner pinned it: "
                f"field={key}, expected={expected_value}, actual={actual}"
            )


def audit_irra_runtime_kit(project_root: Path) -> dict[str, Any]:
    file_hashes: dict[str, str] = {}
    manifest = bytearray()
    for relative in IRRA_RUNTIME_KIT_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"IRRA runtime-kit file is missing: {path}")
        digest = _sha256(path.read_bytes())
        file_hashes[relative] = digest
        manifest.extend(f"{relative}={digest}\n".encode("utf-8"))
    state = {
        "manifest_sha256": _sha256(bytes(manifest)),
        "files": file_hashes,
    }
    print(
        "[IRRA_RUNTIME_KIT_STATE] "
        + json.dumps(state, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return state


def _canonical_lf_bytes(label: str, path: Path, data: bytes) -> bytes:
    canonical = data.replace(b"\r\n", b"\n")
    if b"\r" in canonical:
        raise RuntimeError(
            f"{label} contains a standalone CR byte after CRLF normalization: {path}"
        )
    return canonical


def assert_degradation_copy_provenance(project_root: Path) -> dict[str, str]:
    """Fail closed unless payload equals the source under Git's LF convention.

    A Windows checkout may present the tracked LF source as CRLF even though the
    committed blob and every Linux checkout use LF.  Canonicalizing only CRLF
    here makes the byte assertion portable while still detecting every content
    change (and any unsupported standalone CR byte).
    """
    source_path = project_root / "utils" / "eval_degradation.py"
    copy_path = Path(__file__).resolve().with_name("eval_degradation.py")
    source_checkout_bytes = source_path.read_bytes()
    source_bytes = _canonical_lf_bytes(
        "IRRA degradation source", source_path, source_checkout_bytes
    )
    copy_checkout_bytes = copy_path.read_bytes()
    copy_bytes = _canonical_lf_bytes(
        "DCAlign degradation copy", copy_path, copy_checkout_bytes
    )
    if not copy_bytes.startswith(COPY_HEADER_BYTES):
        actual_prefix = copy_bytes[: len(COPY_HEADER_BYTES)]
        raise RuntimeError(
            "DCAlign degradation copy has an unexpected provenance header: "
            f"expected={COPY_HEADER_BYTES!r}, actual={actual_prefix!r}"
        )
    payload = copy_bytes[len(COPY_HEADER_BYTES) :]
    source_sha = _sha256(source_bytes)
    copy_sha = _sha256(copy_bytes)
    payload_sha = _sha256(payload)
    expected_copy_sha = _sha256(COPY_HEADER_BYTES + source_bytes)
    if copy_sha != expected_copy_sha:
        raise RuntimeError(
            "DCAlign degradation copy SHA mismatch: "
            f"expected={expected_copy_sha}, actual={copy_sha}"
        )
    if payload != source_bytes or payload_sha != source_sha:
        raise RuntimeError(
            "DCAlign degradation payload differs from IRRA source: "
            f"expected={source_path} sha256={source_sha}, "
            f"actual={copy_path} payload_sha256={payload_sha}"
        )
    return {
        "source_path": str(source_path.resolve()),
        "copy_path": str(copy_path),
        "source_sha256": source_sha,
        "source_checkout_sha256": _sha256(source_checkout_bytes),
        "source_checkout_eol_normalized": str(
            source_checkout_bytes != source_bytes
        ).lower(),
        "copy_sha256": copy_sha,
        "copy_checkout_sha256": _sha256(copy_checkout_bytes),
        "copy_checkout_eol_normalized": str(
            copy_checkout_bytes != copy_bytes
        ).lower(),
        "payload_sha256": payload_sha,
    }


def _module_path(module: ModuleType | None) -> Path | None:
    if module is None:
        return None
    raw_path = getattr(module, "__file__", None)
    return Path(raw_path).resolve() if raw_path else None


def _symbol_path(symbol: Any) -> Path | None:
    module_name = getattr(symbol, "__module__", None)
    module = sys.modules.get(module_name) if module_name else inspect.getmodule(symbol)
    return _module_path(module)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _assert_path_within(label: str, actual: Path | None, expected_root: Path) -> Path:
    expected = expected_root.resolve()
    if actual is None or not _is_within(actual, expected):
        raise ImportProvenanceError(
            f"import provenance mismatch for {label}: "
            f"expected={expected}, actual={actual}"
        )
    return actual.resolve()


def _collision_module_names() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in COLLIDING_PACKAGES)
    )


def _remove_sys_path(path: Path) -> None:
    expected = path.resolve()
    sys.path[:] = [
        item
        for item in sys.path
        if not item or Path(item).resolve() != expected
    ]


def _load_imports_fail_closed(project_root: Path, dcalign_root: Path) -> dict[str, Any]:
    """Load IRRA metrics and DCAlign APIs without allowing namespace leakage."""
    preexisting = _collision_module_names()
    if preexisting:
        actual_paths = {
            name: str(_module_path(sys.modules.get(name))) for name in preexisting
        }
        raise ImportProvenanceError(
            "generic packages were imported before provenance setup: "
            f"expected=none, actual={actual_paths}"
        )

    project_root = project_root.resolve()
    dcalign_root = dcalign_root.resolve()
    sys.path.insert(0, str(project_root))
    try:
        irra_metrics = importlib.import_module("utils.metrics")
        irra_rank = getattr(irra_metrics, "rank")
        irra_stable_hash = getattr(irra_metrics, "_stable_hash")
        rank_path = _assert_path_within(
            "IRRA rank", _symbol_path(irra_rank), project_root
        )
        stable_hash_path = _assert_path_within(
            "IRRA _stable_hash", _symbol_path(irra_stable_hash), project_root
        )
        irra_dependency_paths: dict[str, str] = {}
        for name in _collision_module_names():
            if not (name == "utils" or name.startswith("utils.")):
                continue
            actual = _assert_path_within(
                f"IRRA rank dependency {name}",
                _module_path(sys.modules.get(name)),
                project_root,
            )
            irra_dependency_paths[name] = str(actual)
    finally:
        _remove_sys_path(project_root)

    # The retained function objects keep their original globals. Clearing only
    # generic module names lets DCAlign import its own datasets/model/utils tree.
    for name in _collision_module_names():
        del sys.modules[name]

    before_external = set(sys.modules)
    sys.path.insert(0, str(dcalign_root))
    try:
        datasets_module = importlib.import_module("datasets")
        model_module = importlib.import_module("model")
        checkpoint_module = importlib.import_module("utils.checkpoint")
        iotools_module = importlib.import_module("utils.iotools")
        build_dataloader = getattr(datasets_module, "build_dataloader")
        build_model = getattr(model_module, "build_model")
        Checkpointer = getattr(checkpoint_module, "Checkpointer")
        load_train_configs = getattr(iotools_module, "load_train_configs")

        symbol_paths = {
            "build_dataloader": _assert_path_within(
                "DCAlign build_dataloader",
                _symbol_path(build_dataloader),
                dcalign_root,
            ),
            "build_model": _assert_path_within(
                "DCAlign build_model", _symbol_path(build_model), dcalign_root
            ),
            "Checkpointer": _assert_path_within(
                "DCAlign Checkpointer", _symbol_path(Checkpointer), dcalign_root
            ),
            "load_train_configs": _assert_path_within(
                "DCAlign load_train_configs",
                _symbol_path(load_train_configs),
                dcalign_root,
            ),
        }

        # Check every repository-style generic module imported directly or
        # indirectly by these APIs. Third-party modules are intentionally exempt.
        dcalign_dependency_paths: dict[str, str] = {}
        for name in _collision_module_names():
            actual = _assert_path_within(
                f"DCAlign indirect module {name}",
                _module_path(sys.modules.get(name)),
                dcalign_root,
            )
            dcalign_dependency_paths[name] = str(actual)
        for name in sorted(set(sys.modules) - before_external):
            module_path = _module_path(sys.modules.get(name))
            if module_path is not None and _is_within(module_path, project_root):
                raise ImportProvenanceError(
                    f"DCAlign indirect import leaked into IRRA for {name}: "
                    f"expected={dcalign_root}, actual={module_path}"
                )
    except Exception:
        _remove_sys_path(dcalign_root)
        raise

    provenance = {
        "irra_rank": str(rank_path),
        "irra_stable_hash": str(stable_hash_path),
        **{f"dcalign_{key}": str(value) for key, value in symbol_paths.items()},
        **{
            f"irra_dependency:{key}": value
            for key, value in sorted(irra_dependency_paths.items())
        },
        **{
            f"dcalign_dependency:{key}": value
            for key, value in sorted(dcalign_dependency_paths.items())
        },
    }
    print(
        "[IMPORT_PROVENANCE] "
        + json.dumps(provenance, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return {
        "rank": irra_rank,
        "stable_hash": irra_stable_hash,
        "build_dataloader": build_dataloader,
        "build_model": build_model,
        "Checkpointer": Checkpointer,
        "load_train_configs": load_train_configs,
        "provenance": provenance,
        "irra_dependency_paths": irra_dependency_paths,
    }


def _load_degradation_copy() -> ModuleType:
    module_path = Path(__file__).resolve().with_name("eval_degradation.py")
    spec = importlib.util.spec_from_file_location(
        "_irra_dcalign_eval_degradation_copy", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load degradation copy: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvalDegradationApplier:
    """Mirror utils.metrics.Evaluator's evaluation-only degradation semantics."""

    def __init__(self, copied_module: ModuleType):
        self.copied_module = copied_module
        self._generators: dict[str, torch.Generator] = {}

    def _generator(self, device: torch.device, seed: int) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            try:
                generator = torch.Generator(device=device)
            except (TypeError, RuntimeError):
                device_type = getattr(device, "type", str(device).split(":")[0])
                generator = torch.Generator(device=device_type)
            self._generators[key] = generator.manual_seed(int(seed))
        return self._generators[key]

    @staticmethod
    def _clamp_like(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if value.dim() <= 1:
            return value
        reduce_dims = tuple(range(1, reference.dim()))
        ref = reference.detach().float()
        min_value = ref.amin(dim=reduce_dims, keepdim=True).to(dtype=value.dtype)
        max_value = ref.amax(dim=reduce_dims, keepdim=True).to(dtype=value.dtype)
        return torch.max(torch.min(value, max_value), min_value)

    def _add_noise(
        self, tensor: torch.Tensor | None, strength: float, seed: int
    ) -> torch.Tensor | None:
        if tensor is None or float(strength) <= 0.0:
            return tensor
        noise = torch.randn(
            tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype,
            generator=self._generator(tensor.device, seed),
        )
        return self._clamp_like(tensor + float(strength) * noise, tensor)

    @staticmethod
    def _blur(
        tensor: torch.Tensor | None, kernel_size: int
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        kernel_size = int(kernel_size)
        if kernel_size <= 1:
            return tensor
        pad_total = kernel_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padded = F.pad(
            tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="replicate",
        )
        return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)

    def apply(
        self,
        mode: str,
        rgb_img: torch.Tensor,
        thermal_img: torch.Tensor | None,
        strength: float,
        blur_kernel: int,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mode = str(mode).lower()
        if mode == "none":
            return rgb_img, thermal_img
        if mode == "rgb_missing":
            rgb_img = torch.zeros_like(rgb_img)
        elif mode == "ir_missing":
            if thermal_img is not None:
                thermal_img = torch.zeros_like(thermal_img)
        elif mode == "rgb_noise":
            rgb_img = self._add_noise(rgb_img, strength, seed)
        elif mode == "ir_noise":
            thermal_img = self._add_noise(thermal_img, strength, seed)
        elif mode == "both_noise":
            # One shared generator and this RGB-then-T call order exactly mirror
            # IRRA's utils/metrics.py implementation.
            rgb_img = self._add_noise(rgb_img, strength, seed)
            thermal_img = self._add_noise(thermal_img, strength, seed)
        elif mode == "rgb_blur":
            rgb_img = self._blur(rgb_img, blur_kernel)
        elif mode == "ir_blur":
            thermal_img = self._blur(thermal_img, blur_kernel)
        elif mode == "both_blur":
            rgb_img = self._blur(rgb_img, blur_kernel)
            thermal_img = self._blur(thermal_img, blur_kernel)
        elif mode in self.copied_module.NEW_EVAL_DEGRADATION_MODES:
            rgb_img, thermal_img = self.copied_module.apply_new_eval_degradation(
                mode=mode,
                rgb_img=rgb_img,
                thermal_img=thermal_img,
                strength=strength,
            )
        else:
            raise ValueError(f"Unsupported eval degradation mode: {mode}")
        return rgb_img, thermal_img


def _torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def _validate_cli(args: argparse.Namespace) -> None:
    for label, value, expected_length in (
        ("--expected-dcalign-head", args.expected_dcalign_head, 40),
        (
            "--expected-dcalign-diff-sha256",
            args.expected_dcalign_diff_sha256,
            64,
        ),
        (
            "--expected-irra-kit-sha256",
            args.expected_irra_kit_sha256,
            64,
        ),
        (
            "--expected-base-config-sha256",
            args.expected_base_config_sha256,
            64,
        ),
    ):
        if len(value) != expected_length:
            raise ValueError(f"{label} has an invalid digest length: {value!r}")
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{label} must be lowercase hexadecimal: {value!r}")
    if args.dump_sim.suffix.lower() != ".pt":
        raise ValueError("--dump-sim must use a .pt path")
    if args.metrics_out.suffix.lower() != ".tsv":
        raise ValueError("--metrics-out must use a .tsv path")
    protected_inputs = {
        "--checkpoint": args.checkpoint,
        "--config-file": args.config_file,
        "--protocol-base-config": args.protocol_base_config,
    }
    for output_label, output_path in (
        ("--dump-sim", args.dump_sim),
        ("--metrics-out", args.metrics_out),
    ):
        for input_label, input_path in protected_inputs.items():
            if output_path == input_path:
                raise ValueError(
                    f"{output_label} must not overwrite {input_label}: {output_path}"
                )
        if output_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing {output_label} artifact: {output_path}"
            )
    if args.dump_sim == args.metrics_out:
        raise ValueError("--dump-sim and --metrics-out must be different paths")
    if not args.protocol_base_config.is_file():
        raise FileNotFoundError(
            f"protocol base config does not exist: {args.protocol_base_config}"
        )
    actual_base_sha = _sha256(args.protocol_base_config.read_bytes())
    if actual_base_sha != args.expected_base_config_sha256:
        raise RuntimeError(
            "protocol base config changed after the runner pinned it: "
            f"expected={args.expected_base_config_sha256}, actual={actual_base_sha}"
        )
    if not math.isfinite(args.degrade_strength) or args.degrade_strength < 0.0:
        raise ValueError("--degrade-strength must be finite and >= 0")
    if args.degrade_kernel <= 0:
        raise ValueError("--degrade-kernel must be > 0")
    if args.degrade_mode in {
        "rgb_brightness",
        "ir_brightness",
        "rgb_lowres",
        "ir_lowres",
    } and not (0.0 < args.degrade_strength <= 1.0):
        raise ValueError(
            f"{args.degrade_mode} requires 0 < --degrade-strength <= 1"
        )
    if args.degrade_mode in {"rgb_jpeg", "ir_jpeg"} and (
        not float(args.degrade_strength).is_integer()
        or not (1.0 <= args.degrade_strength <= 100.0)
    ):
        raise ValueError("JPEG degradation requires integer strength in [1, 100]")
    is_best_group = args.group_name.endswith("-best")
    if is_best_group:
        if args.checkpoint.name != "best.pth":
            raise ValueError("*-best groups must evaluate an explicitly named best.pth")
        if args.scenario_name != "clean" or args.degrade_mode != "none":
            raise ValueError("*-best groups may evaluate only the clean/none scenario")
    elif args.checkpoint.name != "last.pth":
        raise ValueError("degradation protocol groups must evaluate last.pth")


def _validate_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = _torch_load_cpu(path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"checkpoint top level must be a mapping: {path}; "
            f"got {type(checkpoint).__name__}"
        )
    if "model" not in checkpoint:
        raise KeyError(f"checkpoint top level lacks required 'model' key: {path}")
    if not isinstance(checkpoint["model"], Mapping):
        raise TypeError("checkpoint['model'] must be a state-dict mapping")
    if not checkpoint["model"]:
        raise ValueError("checkpoint['model'] must not be empty")
    return checkpoint["model"]


def _summarize_keys(keys: set[str], limit: int = 20) -> str:
    ordered = sorted(keys)
    shown = ordered[:limit]
    suffix = "" if len(ordered) <= limit else f" ... (+{len(ordered) - limit} more)"
    return repr(shown) + suffix


def _validate_checkpoint_model_compatibility(
    model: Any,
    loaded_state: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    """Reject the suffix-match fallback in DCAlign's upstream Checkpointer.

    Commit 1ca105f maps any loaded key that is a suffix of a model key and
    silently keeps the freshly initialized value for unmatched parameters.
    Exact key/shape coverage is required here before invoking that loader.
    """
    if any(not isinstance(key, str) for key in loaded_state):
        raise TypeError("checkpoint model state keys must all be strings")
    loaded = dict(loaded_state)
    if loaded and all(key.startswith("module.") for key in loaded):
        loaded = {key[len("module.") :]: value for key, value in loaded.items()}
    expected = model.state_dict()
    loaded_keys = set(loaded)
    expected_keys = set(expected)
    missing = expected_keys - loaded_keys
    unexpected = loaded_keys - expected_keys
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/model key mismatch before DCAlign suffix matching: "
            f"checkpoint={checkpoint_path}; missing_count={len(missing)} "
            f"missing={_summarize_keys(missing)}; "
            f"unexpected_count={len(unexpected)} "
            f"unexpected={_summarize_keys(unexpected)}"
        )
    bad_shapes: list[str] = []
    non_tensors: list[str] = []
    for key in sorted(expected_keys):
        value = loaded[key]
        if not torch.is_tensor(value):
            non_tensors.append(key)
            continue
        if tuple(value.shape) != tuple(expected[key].shape):
            bad_shapes.append(
                f"{key}:expected={tuple(expected[key].shape)},actual={tuple(value.shape)}"
            )
    if non_tensors or bad_shapes:
        raise RuntimeError(
            "checkpoint/model tensor mismatch: "
            f"checkpoint={checkpoint_path}; "
            f"non_tensor_keys={_summarize_keys(set(non_tensors))}; "
            f"shape_mismatches={bad_shapes[:20]!r}"
        )
    print(
        "[CHECKPOINT_MODEL_ASSERT] "
        f"path={checkpoint_path} exact_keys={len(expected_keys)} shapes=exact",
        flush=True,
    )


def _encode_all(
    model: Any,
    img_loader: Any,
    txt_loader: Any,
    device: torch.device,
    degradation: EvalDegradationApplier,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qids: list[torch.Tensor] = []
    gids: list[torch.Tensor] = []
    qfeats: list[torch.Tensor] = []
    gfeats: list[torch.Tensor] = []

    model.eval()
    for batch in txt_loader:
        if len(batch) != 2:
            raise RuntimeError(
                f"DCAlign text loader must return (pid, caption), got {len(batch)} items"
            )
        pid, caption = batch
        caption = caption.to(device)
        with torch.no_grad():
            text_feature = model.encode_text(caption)
        if not torch.is_tensor(text_feature) or text_feature.dim() != 2:
            raise RuntimeError(
                "model.encode_text must return a [B,D] tensor; "
                f"got {type(text_feature).__name__} "
                f"shape={getattr(text_feature, 'shape', None)}"
            )
        qids.append(torch.as_tensor(pid).view(-1).detach().cpu().long())
        qfeats.append(text_feature)

    for batch in img_loader:
        if len(batch) != 3:
            raise RuntimeError(
                "DCAlign image loader must return (pid, rgb_img, thermal_img), "
                f"got {len(batch)} items"
            )
        pid, rgb_img, thermal_img = batch
        rgb_img = rgb_img.to(device)
        thermal_img = thermal_img.to(device)

        # Injection-point parity with IRRA-main/utils/metrics.py lines 385-404:
        # tensors move to device first (lines 387-388), degradation is applied
        # next (lines 393-397), and image encoding starts only afterwards
        # (lines 398-404).
        rgb_img, thermal_img = degradation.apply(
            mode=args.degrade_mode,
            rgb_img=rgb_img,
            thermal_img=thermal_img,
            strength=args.degrade_strength,
            blur_kernel=args.degrade_kernel,
            seed=args.degrade_seed,
        )
        if thermal_img is None:
            raise RuntimeError("DCAlign evaluation requires a thermal image tensor")
        with torch.no_grad():
            rgb_tokens = model.encode_image(rgb_img)
            thermal_tokens = model.encode_t_image(thermal_img)
            fusion_output = model.fuse(rgb_tokens, thermal_tokens)
        if not isinstance(fusion_output, (tuple, list)) or len(fusion_output) != 2:
            raise RuntimeError(
                "model.fuse must return (token_features, global_feature)"
            )
        fused_global = fusion_output[1]
        if not torch.is_tensor(fused_global) or fused_global.dim() != 2:
            raise RuntimeError(
                "model.fuse global output must be a [B,D] tensor; "
                f"got {type(fused_global).__name__} "
                f"shape={getattr(fused_global, 'shape', None)}"
            )
        gids.append(torch.as_tensor(pid).view(-1).detach().cpu().long())
        gfeats.append(fused_global)

    if not qfeats or not gfeats:
        raise RuntimeError("DCAlign evaluation loaders must both be non-empty")
    return (
        torch.cat(qfeats, dim=0),
        torch.cat(gfeats, dim=0),
        torch.cat(qids, dim=0),
        torch.cat(gids, dim=0),
    )


def _compute_metrics(
    rank_function: Any,
    similarity: torch.Tensor,
    qids: torch.Tensor,
    gids: torch.Tensor,
) -> dict[str, float]:
    cmc, mean_ap, mean_inp, _ = rank_function(
        similarity=similarity,
        q_pids=qids,
        g_pids=gids,
        max_rank=10,
        get_mAP=True,
    )
    metrics = {
        "R1": float(cmc[0].item()),
        "R5": float(cmc[4].item()),
        "R10": float(cmc[9].item()),
        "mAP": float(mean_ap.item()),
        "mINP": float(mean_inp.item()),
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite retrieval metrics: {metrics}")
    return metrics


def _build_score_payload(
    similarity: torch.Tensor,
    qids: torch.Tensor,
    gids: torch.Tensor,
    metrics: Mapping[str, float],
    stable_hash: Any,
    cli_args: argparse.Namespace,
    config_args: Any,
    import_provenance: Mapping[str, str],
    degradation_provenance: Mapping[str, str],
    dcalign_source_state: Mapping[str, Any],
    irra_runtime_kit_state: Mapping[str, Any],
) -> dict[str, Any]:
    sim_cpu = similarity.detach().cpu().float()
    txt_pids = qids.detach().cpu().long().view(-1)
    img_pids = gids.detach().cpu().long().view(-1)
    order_hash = stable_hash(
        {
            "txt_pids": txt_pids.tolist(),
            "img_pids": img_pids.tolist(),
            "score_shape": list(sim_cpu.shape),
        }
    )
    metadata = {
        "dataset_name": str(getattr(config_args, "dataset_name", "")),
        "eval_dataset": "test",
        "score_shape": list(sim_cpu.shape),
        "score_semantics": "global_cosine_similarity",
        "use_qalr": False,
        "use_qalr_v2": False,
        "use_qalr_v3_gate": False,
        "use_qalr_v3_two_stage_gate": False,
        "use_qalr_v3_candidate_diag": False,
        "qalr_args": {},
        "use_transport_rerank": False,
        "use_uspc_slot_rerank": False,
        "use_modality_text_prompt": False,
        "config_file": str(cli_args.config_file.resolve()),
        "test_ckpt_file": str(cli_args.checkpoint.resolve()),
        "model_name": str(getattr(config_args, "name", "")),
        "output_dir": str(getattr(config_args, "output_dir", "")),
        "eval_source": "dcalign",
        "eval_degrade_mode": cli_args.degrade_mode,
        "eval_degrade_strength": float(cli_args.degrade_strength),
        "eval_degrade_blur_kernel": int(cli_args.degrade_kernel),
        "eval_degrade_seed": int(cli_args.degrade_seed),
        "order_hash": order_hash,
        "raw_similarity_saved": False,
        "scenario": cli_args.scenario_name,
        "mode": cli_args.degrade_mode,
        "strength": float(cli_args.degrade_strength),
        "kernel": int(cli_args.degrade_kernel),
        "dseed": int(cli_args.degrade_seed),
        "group": cli_args.group_name,
        "seed": int(cli_args.seed),
        "source": "dcalign",
        "import_provenance": dict(import_provenance),
        "degradation_provenance": dict(degradation_provenance),
        "dcalign_source_state": dict(dcalign_source_state),
        "irra_runtime_kit_state": dict(irra_runtime_kit_state),
        "protocol_base_config": str(cli_args.protocol_base_config.resolve()),
        "protocol_base_config_sha256": cli_args.expected_base_config_sha256,
    }
    payload = {
        "sim": sim_cpu,
        "txt_pids": txt_pids,
        "img_pids": img_pids,
        "txt_indices": torch.arange(sim_cpu.shape[0], dtype=torch.long),
        "img_indices": torch.arange(sim_cpu.shape[1], dtype=torch.long),
        "metrics": dict(metrics),
        "metadata": metadata,
        "order_hash": order_hash,
    }
    _validate_score_payload(payload, stable_hash)
    return payload


def _validate_score_payload(payload: Mapping[str, Any], stable_hash: Any) -> None:
    required = {
        "sim",
        "txt_pids",
        "img_pids",
        "txt_indices",
        "img_indices",
        "metrics",
        "metadata",
        "order_hash",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"score payload is missing keys: {missing}")
    sim = payload["sim"]
    if not torch.is_tensor(sim) or tuple(sim.shape) != EXPECTED_SCORE_SHAPE:
        raise RuntimeError(
            f"score payload shape mismatch: expected={EXPECTED_SCORE_SHAPE}, "
            f"actual={getattr(sim, 'shape', None)}"
        )
    if sim.device.type != "cpu" or sim.dtype != torch.float32:
        raise RuntimeError(
            f"score sim must be CPU float32, got device={sim.device}, dtype={sim.dtype}"
        )
    if not bool(torch.isfinite(sim).all().item()):
        raise RuntimeError("score sim contains NaN or infinity")
    if payload["txt_pids"].numel() != sim.shape[0]:
        raise RuntimeError("txt_pids length does not match score rows")
    if payload["img_pids"].numel() != sim.shape[1]:
        raise RuntimeError("img_pids length does not match score columns")
    txt_pids = payload["txt_pids"]
    img_pids = payload["img_pids"]
    for label, pids in (("txt_pids", txt_pids), ("img_pids", img_pids)):
        if not torch.is_tensor(pids) or pids.device.type != "cpu" \
                or pids.dtype != torch.long:
            raise RuntimeError(f"{label} must be a CPU int64 tensor")
    txt_unique = torch.unique(txt_pids)
    img_unique = torch.unique(img_pids)
    if txt_unique.numel() != 556 or img_unique.numel() != 556 \
            or not torch.equal(txt_unique, img_unique):
        raise RuntimeError(
            "query/gallery pid coverage mismatch: "
            f"txt_unique={txt_unique.numel()}, img_unique={img_unique.numel()}"
        )
    if not torch.equal(
        payload["txt_indices"], torch.arange(sim.shape[0], dtype=torch.long)
    ):
        raise RuntimeError("txt_indices are not the canonical query order")
    if not torch.equal(
        payload["img_indices"], torch.arange(sim.shape[1], dtype=torch.long)
    ):
        raise RuntimeError("img_indices are not the canonical gallery order")
    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping) or any(key not in metrics for key in METRIC_NAMES):
        raise RuntimeError(f"score payload lacks required metrics: {metrics!r}")
    if any(not math.isfinite(float(metrics[key])) for key in METRIC_NAMES):
        raise RuntimeError(f"score payload contains non-finite metrics: {metrics!r}")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise RuntimeError("score payload metadata must be a mapping")
    if metadata.get("score_shape") != list(EXPECTED_SCORE_SHAPE):
        raise RuntimeError("score metadata shape does not match payload shape")
    if metadata.get("source") != "dcalign":
        raise RuntimeError("score metadata source must equal 'dcalign'")
    source_state = metadata.get("dcalign_source_state")
    if not isinstance(source_state, Mapping):
        raise RuntimeError("score metadata lacks DCAlign source provenance")
    if source_state.get("head") != EXPECTED_DCALIGN_HEAD:
        raise RuntimeError("score metadata DCAlign HEAD is not the pinned commit")
    if set(source_state.get("tracked_dirty_paths", ())) \
            != EXPECTED_DCALIGN_DIRTY_PATHS:
        raise RuntimeError("score metadata DCAlign dirty path set is not pinned")
    diff_sha = source_state.get("tracked_diff_sha256")
    if not isinstance(diff_sha, str) or len(diff_sha) != 64:
        raise RuntimeError("score metadata DCAlign diff SHA256 is invalid")
    kit_state = metadata.get("irra_runtime_kit_state")
    if not isinstance(kit_state, Mapping):
        raise RuntimeError("score metadata lacks IRRA runtime-kit provenance")
    if set(kit_state.get("files", ())) != set(IRRA_RUNTIME_KIT_FILES):
        raise RuntimeError("score metadata IRRA runtime-kit file set is invalid")
    kit_sha = kit_state.get("manifest_sha256")
    if not isinstance(kit_sha, str) or len(kit_sha) != 64:
        raise RuntimeError("score metadata IRRA runtime-kit SHA256 is invalid")
    base_sha = metadata.get("protocol_base_config_sha256")
    if not isinstance(base_sha, str) or len(base_sha) != 64:
        raise RuntimeError("score metadata protocol base-config SHA256 is invalid")
    if metadata.get("order_hash") != payload["order_hash"]:
        raise RuntimeError("score metadata and payload order_hash differ")
    expected_order_hash = stable_hash(
        {
            "txt_pids": payload["txt_pids"].view(-1).tolist(),
            "img_pids": payload["img_pids"].view(-1).tolist(),
            "score_shape": list(sim.shape),
        }
    )
    if payload["order_hash"] != expected_order_hash:
        raise RuntimeError(
            "score payload order_hash mismatch: "
            f"expected={expected_order_hash}, actual={payload['order_hash']}"
        )


def _atomic_torch_save(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_metrics_write(
    output_path: Path,
    args: argparse.Namespace,
    metrics: Mapping[str, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    row: dict[str, Any] = {
        "group": args.group_name,
        "seed": args.seed,
        "scenario": args.scenario_name,
        "mode": args.degrade_mode,
        "strength": format(args.degrade_strength, ".10g"),
        "kernel": args.degrade_kernel,
        "dseed": args.degrade_seed,
        "split": "test",
    }
    row.update({key: format(float(metrics[key]), ".10g") for key in METRIC_NAMES})
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(METRICS_HEADER),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exactly one degradation scenario for a DCAlign checkpoint "
            "using IRRA's metric and score-dump contracts."
        )
    )
    parser.add_argument("--dcalign-root", type=Path, required=True)
    parser.add_argument(
        "--expected-dcalign-head",
        required=True,
        help="pinned 40-character DCAlign Git commit checked before evaluation",
    )
    parser.add_argument(
        "--expected-dcalign-diff-sha256",
        required=True,
        help="pinned SHA256 of the seven-file tracked DCAlign adaptation diff",
    )
    parser.add_argument(
        "--expected-irra-kit-sha256",
        required=True,
        help="pinned SHA256 manifest of the IRRA evaluator runtime files",
    )
    parser.add_argument("--protocol-base-config", type=Path, required=True)
    parser.add_argument(
        "--expected-base-config-sha256",
        required=True,
        help="pinned SHA256 of the DCAlign base config used for every arm/seed",
    )
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="explicit last.pth or clean-comparison best.pth; there is no fallback",
    )
    parser.add_argument(
        "--degrade-mode", choices=SUPPORTED_DEGRADE_MODES, required=True
    )
    parser.add_argument("--degrade-strength", type=float, default=1.0)
    parser.add_argument("--degrade-kernel", type=int, default=7)
    parser.add_argument("--degrade-seed", type=int, default=2026)
    parser.add_argument("--scenario-name", required=True)
    parser.add_argument("--group-name", choices=ALLOWED_GROUPS, required=True)
    parser.add_argument(
        "--seed", type=int, required=True, help="training seed recorded in metrics.tsv"
    )
    parser.add_argument("--dump-sim", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.dcalign_root = args.dcalign_root.expanduser().resolve()
    args.protocol_base_config = args.protocol_base_config.expanduser().resolve()
    args.config_file = args.config_file.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dump_sim = args.dump_sim.expanduser().resolve()
    args.metrics_out = args.metrics_out.expanduser().resolve()
    _validate_cli(args)

    project_root = Path(__file__).resolve().parents[2]
    degradation_provenance = assert_degradation_copy_provenance(project_root)
    print(
        "[DEGRADATION_PROVENANCE] "
        + json.dumps(degradation_provenance, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    irra_runtime_kit_state = audit_irra_runtime_kit(project_root)
    if (
        irra_runtime_kit_state["manifest_sha256"]
        != args.expected_irra_kit_sha256
    ):
        raise RuntimeError(
            "IRRA runtime kit changed after the runner pinned it: "
            f"expected={args.expected_irra_kit_sha256}, "
            f"actual={irra_runtime_kit_state['manifest_sha256']}"
        )
    if not args.dcalign_root.is_dir():
        raise FileNotFoundError(f"DCAlign root does not exist: {args.dcalign_root}")
    dcalign_source_state = audit_dcalign_source_state(args.dcalign_root)
    _assert_pinned_source_state(
        dcalign_source_state,
        args.expected_dcalign_head,
        args.expected_dcalign_diff_sha256,
    )
    for relative in (
        Path("model/build.py"),
        Path("model/feature_fusion.py"),
        Path("datasets/build.py"),
        Path("utils/checkpoint.py"),
        Path("utils/iotools.py"),
    ):
        required_path = args.dcalign_root / relative
        if not required_path.is_file():
            raise FileNotFoundError(f"required DCAlign source is missing: {required_path}")
    if not args.config_file.is_file():
        raise FileNotFoundError(f"config file does not exist: {args.config_file}")
    checkpoint_model_state = _validate_checkpoint(args.checkpoint)

    imported = _load_imports_fail_closed(project_root, args.dcalign_root)
    config_args = imported["load_train_configs"](str(args.config_file))
    config_seed = getattr(config_args, "seed", None)
    if config_seed is not None and int(config_seed) != args.seed:
        raise ValueError(
            f"--seed disagrees with config seed: expected={config_seed}, actual={args.seed}"
        )
    expected_dynamic = args.group_name.startswith("DC-dyn")
    actual_dynamic = bool(
        getattr(config_args, "dcalign_dynamic_fusion", False)
    )
    if actual_dynamic != expected_dynamic:
        raise ValueError(
            "dynamic-fusion config/group mismatch: "
            f"expected={expected_dynamic}, actual={actual_dynamic}"
        )
    config_args.training = False
    config_args.eval_dataset = "test"
    config_args.config_file = str(args.config_file)
    config_args.test_ckpt_file = str(args.checkpoint)

    loader_outputs = imported["build_dataloader"](config_args)
    if not isinstance(loader_outputs, (tuple, list)) or len(loader_outputs) != 3:
        raise RuntimeError(
            "DCAlign build_dataloader(training=False) must return exactly "
            "(test_img_loader, test_txt_loader, num_classes)"
        )
    img_loader, txt_loader, num_classes = loader_outputs
    model = imported["build_model"](config_args, num_classes=num_classes)
    _validate_checkpoint_model_compatibility(
        model, checkpoint_model_state, args.checkpoint
    )
    del checkpoint_model_state
    imported["Checkpointer"](model).load(f=str(args.checkpoint))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    model.to(device)
    copied_module = _load_degradation_copy()
    degradation = EvalDegradationApplier(copied_module)
    qfeats, gfeats, qids, gids = _encode_all(
        model, img_loader, txt_loader, device, degradation, args
    )
    qfeats = F.normalize(qfeats, p=2, dim=1)
    gfeats = F.normalize(gfeats, p=2, dim=1)
    similarity = qfeats @ gfeats.t()
    if tuple(similarity.shape) != EXPECTED_SCORE_SHAPE:
        raise RuntimeError(
            "unexpected DCAlign score shape: "
            f"expected={EXPECTED_SCORE_SHAPE}, actual={tuple(similarity.shape)}"
        )

    metrics = _compute_metrics(
        imported["rank"], similarity, qids=qids, gids=gids
    )
    payload = _build_score_payload(
        similarity,
        qids,
        gids,
        metrics,
        imported["stable_hash"],
        args,
        config_args,
        imported["provenance"],
        degradation_provenance,
        dcalign_source_state,
        irra_runtime_kit_state,
    )
    _atomic_torch_save(payload, args.dump_sim)
    persisted = _torch_load_cpu(args.dump_sim)
    if not isinstance(persisted, Mapping):
        raise RuntimeError("persisted score dump is not a mapping")
    _validate_score_payload(persisted, imported["stable_hash"])
    _atomic_metrics_write(args.metrics_out, args, metrics)
    print(
        "[DCALIGN_EVAL] "
        f"group={args.group_name} seed={args.seed} "
        f"scenario={args.scenario_name} shape={list(similarity.shape)} "
        + " ".join(f"{key}={metrics[key]:.6f}" for key in METRIC_NAMES),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
