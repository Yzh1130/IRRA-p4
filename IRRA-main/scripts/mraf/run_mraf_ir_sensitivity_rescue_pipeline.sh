#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run the MRAF IR Sensitivity Rescue V4 pipeline.

Usage:
  bash scripts/mraf/run_mraf_ir_sensitivity_rescue_pipeline.sh \
    --source-run-root <read-only-V3-run> \
    --data-root <TextReID-root> \
    --output-root <new-parent> \
    --base-config <historical-configs.yaml> \
    --reference-screen-root <old-S0-S5-screen-root> \
    --reference-run-root <old-seed1-root> \
    --device 0 --seed 1 --auto-continue

Resume:
  bash scripts/mraf/run_mraf_ir_sensitivity_rescue_pipeline.sh \
    --resume-run-root <existing-V4-run> --auto-continue

Options:
  --source-run-root PATH       read-only V3 CONFIRMATION_FAILED run
  --data-root PATH             TextReID dataset root
  --output-root PATH           parent for a new non-overwriting V4 run
  --base-config PATH           original formal MDRC+MRAF configs.yaml
  --reference-screen-root PATH historical S4 screen root; read only
  --reference-run-root PATH    historical seed1 root; read only
  --device INDEX               CUDA index or cuda:INDEX (default: 0)
  --seed N                     initial/formal gate seed; must be 1
  --auto-continue              allow fresh 12/60 epoch training after gates
  --resume-run-root PATH       safely continue one incomplete V4 run
  --dry-run                    print a machine-readable plan; write nothing
  --mock                       CPU/filesystem state-machine fixture only
  --shutdown-on-terminal       shutdown after any real terminal status
  -h, --help                   show this help

The source V3 run is never modified. Scale search is forward-only. Formal
training always starts from the original pretrained initialization.
EOF
}

die() {
  echo "Error: $*" >&2
  return 2
}

source_run_root=""
data_root=""
output_root=""
base_config=""
reference_screen_root=""
reference_run_root=""
device="0"
seed="1"
auto_continue=false
resume_run_root=""
dry_run=false
mock=false
shutdown_on_terminal=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-run-root) source_run_root=${2-}; shift 2 ;;
    --data-root) data_root=${2-}; shift 2 ;;
    --output-root) output_root=${2-}; shift 2 ;;
    --base-config) base_config=${2-}; shift 2 ;;
    --reference-screen-root) reference_screen_root=${2-}; shift 2 ;;
    --reference-run-root) reference_run_root=${2-}; shift 2 ;;
    --device) device=${2-}; shift 2 ;;
    --seed) seed=${2-}; shift 2 ;;
    --auto-continue) auto_continue=true; shift ;;
    --resume-run-root) resume_run_root=${2-}; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    --mock) mock=true; shift ;;
    --shutdown-on-terminal) shutdown_on_terminal=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$seed" =~ ^[0-9]+$ ]] || { echo "Error: --seed must be a non-negative integer" >&2; exit 2; }
[[ "$seed" == "1" ]] || { echo "Error: V4 protocol requires --seed 1" >&2; exit 2; }
[[ -n "$device" ]] || { echo "Error: --device cannot be empty" >&2; exit 2; }
device_index=${device#cuda:}
[[ "$device_index" =~ ^[0-9]+$ ]] || { echo "Error: --device must be an index or cuda:INDEX" >&2; exit 2; }

code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$code_root"
git_root="$(cd .. && pwd)"
python_bin="${PYTHON_BIN:-python}"
shutdown_command="${MRAF_SHUTDOWN_COMMAND:-shutdown}"
timestamp="$(date +%Y%m%d_%H%M%S)"

run_git() {
  local pointer="${git_root}/.git"
  if [[ -f "$pointer" ]]; then
    local raw git_dir
    raw="$(sed -n 's/^gitdir: //p' "$pointer")"
    if [[ "$raw" =~ ^[A-Za-z]:[\\/].* ]]; then
      local drive=${raw:0:1}
      drive=$(printf '%s' "$drive" | tr '[:upper:]' '[:lower:]')
      raw="/mnt/${drive}/${raw:3}"
      raw=${raw//\\//}
    elif [[ "$raw" != /* ]]; then
      raw="${git_root}/${raw}"
    fi
    git --git-dir="$raw" --work-tree="$git_root" "$@"
  else
    git -C "$git_root" "$@"
  fi
}

git_head="$(run_git rev-parse HEAD)"
git_branch="$(run_git branch --show-current)"
short_head=${git_head:0:8}
resume_mode=false
if [[ -n "$resume_run_root" ]]; then
  resume_mode=true
fi

manifest_value() {
  local key=$1
  sed -n "s/^${key}=//p" "${run_dir}/experiment_manifest.txt" | head -n1
}

if [[ "$resume_mode" == "true" ]]; then
  [[ -d "$resume_run_root" ]] || { echo "Error: --resume-run-root is not a directory" >&2; exit 2; }
  run_dir="$(cd "$resume_run_root" && pwd)"
  [[ -f "${run_dir}/experiment_manifest.txt" ]] || { echo "Error: resume manifest missing" >&2; exit 2; }
  [[ "$(manifest_value PIPELINE_ID)" == "mraf_ir_sensitivity_rescue_v4" ]] || {
    echo "Error: resume PIPELINE_ID mismatch" >&2; exit 2;
  }
  [[ "$(manifest_value seed)" == "$seed" ]] || { echo "Error: resume seed mismatch" >&2; exit 2; }
  resume_status="$(sed -n 's/^STATUS=//p' "${run_dir}/status.txt" | head -n1)"
  resume_failed_stage="$(sed -n 's/^FAILED_STAGE=//p' "${run_dir}/status.txt" | head -n1)"
  case "$resume_status" in
    RUNNING|FAILED|MULTISEED_FAILED) ;;
    *)
      echo "Error: resume refuses terminal method/completed status: ${resume_status:-missing}" >&2
      exit 2
      ;;
  esac
  [[ -n "$source_run_root" ]] || source_run_root="$(manifest_value source_run_root)"
  [[ -n "$data_root" ]] || data_root="$(manifest_value data_root)"
  [[ -n "$base_config" ]] || base_config="$(manifest_value base_config)"
  [[ -n "$reference_screen_root" ]] || reference_screen_root="$(manifest_value reference_screen_root)"
  [[ -n "$reference_run_root" ]] || reference_run_root="$(manifest_value reference_run_root)"
  output_root="$(dirname "$run_dir")"
else
  if [[ "$mock" == "true" && -z "$output_root" ]]; then
    output_root="${TMPDIR:-/tmp}"
  fi
  run_dir="${output_root%/}/${timestamp}_${short_head}_mraf_ir_sensitivity_rescue_v4_seed1"
fi

if [[ "$mock" != "true" ]]; then
  for pair in \
    "source-run-root:$source_run_root" \
    "data-root:$data_root" \
    "output-root:$output_root" \
    "base-config:$base_config" \
    "reference-screen-root:$reference_screen_root" \
    "reference-run-root:$reference_run_root"; do
    name=${pair%%:*}
    value=${pair#*:}
    [[ -n "$value" ]] || { echo "Error: --${name} is required" >&2; exit 2; }
  done
  [[ -d "$source_run_root" ]] || { echo "Error: --source-run-root is not a directory" >&2; exit 2; }
  [[ -d "$data_root" ]] || { echo "Error: --data-root is not a directory" >&2; exit 2; }
  [[ -d "$output_root" ]] || { echo "Error: --output-root is not a directory" >&2; exit 2; }
  [[ -f "$base_config" ]] || { echo "Error: --base-config is not a file" >&2; exit 2; }
  [[ -d "$reference_screen_root" ]] || { echo "Error: --reference-screen-root is not a directory" >&2; exit 2; }
  [[ -d "$reference_run_root" ]] || { echo "Error: --reference-run-root is not a directory" >&2; exit 2; }
  source_run_root="$(cd "$source_run_root" && pwd)"
  data_root="$(cd "$data_root" && pwd)"
  output_root="$(cd "$output_root" && pwd)"
  base_config="$(cd "$(dirname "$base_config")" && pwd)/$(basename "$base_config")"
  reference_screen_root="$(cd "$reference_screen_root" && pwd)"
  reference_run_root="$(cd "$reference_run_root" && pwd)"
  if [[ "$resume_mode" == "false" ]]; then
    run_dir="${output_root}/${timestamp}_${short_head}_mraf_ir_sensitivity_rescue_v4_seed1"
  fi
  "$python_bin" -B - "$source_run_root" "$run_dir" <<'PY'
import pathlib, sys
source = pathlib.Path(sys.argv[1]).resolve()
target = pathlib.Path(sys.argv[2]).resolve()
if source == target or source in target.parents or target in source.parents:
    raise SystemExit("V4 run-root and V3 source run-root must be independent")
PY
fi

source_plan=""
if [[ "$mock" == "true" ]]; then
  source_plan=$'PLAN_VALID=true\nSOURCE_CONFIRMATION_STATUS=CONFIRMATION_FAILED\nSOURCE_BEST_FOUND=true\nSOURCE_LAST_FOUND=true\nSOURCE_RUN_ROOT=MOCK'
else
  if ! source_plan="$("$python_bin" -B \
      tools/revision/summarize_mraf_ir_sensitivity_rescue.py \
      --mode source-plan --source-run-root "$source_run_root" 2>&1)"; then
    printf '%s\n' "$source_plan"
    exit 2
  fi
fi

stage_complete() {
  local stage=$1
  [[ -s "${run_dir}/stage_status.tsv" ]] || return 1
  "$python_bin" -B - "${run_dir}" "$stage" <<'PY'
import sys
from pathlib import Path
from tools.revision.summarize_mraf_ir_sensitivity_rescue import read_tsv_compatible
root = Path(sys.argv[1])
stage = sys.argv[2]
rows = read_tsv_compatible(root / "stage_status.tsv").rows
recorded = any(
    row.get("stage") == sys.argv[2]
    and row.get("status") == "COMPLETED"
    and row.get("exit_code") == "0"
    for row in rows
)

screen = (
    "clean", "rgb_missing", "ir_missing", "rgb_noise_0.10",
    "ir_noise_0.10", "rgb_blur_k7", "ir_blur_k7", "both_blur_k7",
)
formal = (
    "clean", "rgb_missing", "ir_missing", "rgb_noise_0.05",
    "rgb_noise_0.10", "ir_noise_0.05", "ir_noise_0.10",
    "rgb_blur_k7", "ir_blur_k7", "both_noise_0.10", "both_blur_k7",
)

def nonempty(path):
    return path.is_file() and path.stat().st_size > 0

def unit_complete(unit, scenarios):
    core = (
        "checkpoint.pth", "configs.yaml", "train.log",
        "mraf_epoch_metrics.tsv", "checkpoint_selection.tsv", "unit.json",
    )
    if not all(nonempty(unit / name) for name in core):
        return False
    if not all(nonempty(unit / "eval" / name / "scores.pt") for name in scenarios):
        return False
    diagnostic = unit / "diagnostics"
    active = unit / "active_diagnostics.txt"
    if active.is_file():
        diagnostic = unit / active.read_text(encoding="utf-8").strip()
    return all(
        nonempty(diagnostic / name)
        for name in (
            "diagnosis.json", "scenario_gate_statistics.csv",
            "gate_direction_statistics.csv", "gradient_diagnostics.csv",
            "fusion_path_check.json",
        )
    )

artifacts = False
if stage in {"full_rediagnosis", "full_rediagnosis_strong"}:
    artifacts = all(
        nonempty(root / name)
        for name in (
            "full_rediagnosis_metrics.tsv",
            "full_rediagnosis_mechanism.tsv",
            "full_rediagnosis_ranking.tsv",
        )
    ) and all(
        unit_complete(root / "full_rediagnosis" / "units" / kind, screen)
        for kind in ("best", "last")
    )
    if stage == "full_rediagnosis_strong":
        artifacts = artifacts and nonempty(root / "selected_ir_rescue_scale.json")
elif stage == "readonly_scale_screen":
    artifacts = all(
        nonempty(root / name)
        for name in (
            "readonly_scale_metrics.tsv", "readonly_scale_mechanism.tsv",
            "readonly_scale_ranking.tsv", "selected_ir_rescue_scale.json",
        )
    ) and all(
        unit_complete(
            root / "readonly_scale" / "units" / f"scale_{scale}" / kind,
            screen,
        )
        for scale in ("2.5", "3", "4", "5")
        for kind in ("best", "last")
    )
elif stage == "ir_rescue_confirmation":
    artifacts = all(
        nonempty(root / name)
        for name in (
            "ir_rescue_confirmation_metrics.tsv",
            "ir_rescue_confirmation_mechanism.tsv",
            "ir_rescue_confirmation_summary.tsv",
        )
    ) and all(
        unit_complete(
            root / "ir_rescue_confirmation" / "units" / kind, screen
        )
        for kind in ("best", "last")
    )
elif stage == "multiseed":
    artifacts = all(
        nonempty(root / name)
        for name in (
            "multiseed_retrieval_summary.tsv",
            "multiseed_robustness_summary.tsv",
            "multiseed_mechanism_summary.tsv",
            "multiseed_classification_summary.tsv",
            "PAPER_READY_RESULTS_ZH.md", "PAPER_READY_RESULTS_EN.md",
            "FINAL_REPORT.txt",
        )
    ) and all(
        nonempty(root / "paper_tables" / name)
        for name in (
            "main_retrieval_results.tsv", "robustness_results.tsv",
            "mechanism_results.tsv", "ablation_scale_results.tsv",
            "seed_level_results.tsv",
        )
    )
raise SystemExit(0 if recorded and artifacts else 1)
PY
}

seed_complete() {
  local formal_seed=$1
  [[ -s "${run_dir}/seed_status.tsv" ]] || return 1
  [[ -s "${run_dir}/formal_seed${formal_seed}_summary.tsv" ]] || return 1
  [[ -s "${run_dir}/selected_formal_seed${formal_seed}_checkpoint.json" ]] || return 1
  "$python_bin" -B - "${run_dir}" "$formal_seed" <<'PY'
import sys
from pathlib import Path
from tools.revision.summarize_mraf_ir_sensitivity_rescue import read_tsv_compatible
root = Path(sys.argv[1])
seed = sys.argv[2]
rows = read_tsv_compatible(root / "seed_status.tsv").rows
recorded = any(
    row.get("seed") == sys.argv[2]
    and row.get("status") == "COMPLETED"
    and row.get("exit_code") == "0"
    for row in rows
)
def nonempty(path):
    return path.is_file() and path.stat().st_size > 0
formal = (
    "clean", "rgb_missing", "ir_missing", "rgb_noise_0.05",
    "rgb_noise_0.10", "ir_noise_0.05", "ir_noise_0.10",
    "rgb_blur_k7", "ir_blur_k7", "both_noise_0.10", "both_blur_k7",
)
def unit_complete(unit):
    if not all(
        nonempty(unit / name)
        for name in (
            "checkpoint.pth", "configs.yaml", "train.log",
            "mraf_epoch_metrics.tsv", "checkpoint_selection.tsv", "unit.json",
        )
    ):
        return False
    if not all(nonempty(unit / "eval" / name / "scores.pt") for name in formal):
        return False
    diagnostic = unit / "diagnostics"
    active = unit / "active_diagnostics.txt"
    if active.is_file():
        diagnostic = unit / active.read_text(encoding="utf-8").strip()
    return all(
        nonempty(diagnostic / name)
        for name in (
            "diagnosis.json", "scenario_gate_statistics.csv",
            "gate_direction_statistics.csv", "gradient_diagnostics.csv",
            "fusion_path_check.json",
        )
    )
artifacts = (
    nonempty(root / f"formal_seed{seed}_metrics.tsv")
    and nonempty(root / f"formal_seed{seed}_mechanism.tsv")
    and nonempty(root / f"formal_seed{seed}_summary.tsv")
    and nonempty(root / f"selected_formal_seed{seed}_checkpoint.json")
    and all(
        unit_complete(root / "formal" / f"seed{seed}" / "units" / kind)
        for kind in ("best", "last")
    )
)
raise SystemExit(0 if recorded and artifacts else 1)
PY
}

next_stage="full_rediagnosis"
if [[ "$resume_mode" == "true" ]]; then
  if stage_complete multiseed; then
    next_stage="completed"
  elif seed_complete 5; then
    next_stage="multiseed"
  elif seed_complete 4; then
    next_stage="formal_seed5"
  elif seed_complete 3; then
    next_stage="formal_seed4"
  elif seed_complete 2; then
    next_stage="formal_seed3"
  elif seed_complete 1; then
    next_stage="formal_seed2"
  elif stage_complete ir_rescue_confirmation || stage_complete full_rediagnosis_strong; then
    next_stage="formal_seed1"
  elif stage_complete readonly_scale_screen; then
    next_stage="ir_rescue_confirmation"
  elif stage_complete full_rediagnosis; then
    next_stage="readonly_scale_screen"
  fi
fi

if [[ "$dry_run" == "true" ]]; then
  printf '%s\n' "$source_plan"
  echo "FULL_REDIAGNOSIS_BATCHES=24"
  echo "READONLY_SCALES=2.5,3,4,5"
  echo "NEXT_STAGE=$next_stage"
  echo "FORMAL_SEEDS=1,2,3,4,5"
  echo "SOURCE_RUN_ROOT=${source_run_root:-MOCK}"
  echo "RUN_ROOT=$run_dir"
  echo "SHUTDOWN_ON_TERMINAL=$shutdown_on_terminal"
  echo "DRY_RUN_READ_ONLY=true"
  echo "TRAINING_EXECUTED=false"
  echo "GPU_ACCESSED=false"
  echo "SHUTDOWN_EXECUTED=false"
  exit 0
fi

if [[ "$resume_mode" == "false" ]]; then
  [[ ! -e "$run_dir" ]] || { echo "Error: refusing to overwrite V4 run-root" >&2; exit 2; }
  if [[ "$mock" == "true" && "$output_root" == "${TMPDIR:-/tmp}" ]]; then
    mock_parent="$(mktemp -d "${TMPDIR:-/tmp}/mraf_ir_sensitivity_v4_mock_XXXXXX")"
    run_dir="${mock_parent}/${timestamp}_${short_head}_mraf_ir_sensitivity_rescue_v4_seed1"
  fi
  if [[ "$mock" != "true" ]]; then
    mkdir -p "$run_dir"
  fi
fi

if [[ "$mock" == "true" ]]; then
  "$python_bin" -B tools/revision/summarize_mraf_ir_sensitivity_rescue.py \
    --mode mock --output-dir "$run_dir"
  archive="${run_dir}/mraf_ir_sensitivity_rescue_reports.tar.gz"
  (
    cd "$run_dir"
    find . -type f \
      \( -name '*.tsv' -o -name '*.json' -o -name '*.md' \
         -o -name '*.txt' -o -name '*.log' -o -name '*.yaml' \) \
      ! -name '*.pth' ! -name '*.pt' -print | sort >reports_index.txt
    tar -czf "$(basename "$archive")" -T reports_index.txt
  )
  echo "STATUS=MOCK_COMPLETED"
  echo "RUN_ROOT=$run_dir"
  echo "TRAINING_EXECUTED=false"
  echo "GPU_ACCESSED=false"
  echo "SHUTDOWN_EXECUTED=false"
  exit 0
fi

append_stage_status() {
  local stage=$1 status=$2 code=$3 details=$4
  "$python_bin" -B - "${run_dir}/stage_status.tsv" \
    "$(date -Iseconds)" "$stage" "$status" "$code" "$details" <<'PY'
import sys
from pathlib import Path
from tools.revision.summarize_mraf_ir_sensitivity_rescue import (
    STAGE_STATUS_FIELDS, read_tsv_compatible, write_tsv_atomic,
)
path = Path(sys.argv[1])
rows = read_tsv_compatible(path).rows
rows.append(dict(zip(STAGE_STATUS_FIELDS, sys.argv[2:])))
write_tsv_atomic(path, STAGE_STATUS_FIELDS, rows)
PY
}

append_seed_status() {
  local formal_seed=$1 status=$2 code=$3 checkpoint=$4 classification=$5
  "$python_bin" -B - "${run_dir}/seed_status.tsv" \
    "$(date -Iseconds)" "$formal_seed" "$status" "$code" \
    "$checkpoint" "$classification" <<'PY'
import sys
from pathlib import Path
from tools.revision.summarize_mraf_ir_sensitivity_rescue import (
    SEED_STATUS_FIELDS, read_tsv_compatible, write_tsv_atomic,
)
path = Path(sys.argv[1])
rows = read_tsv_compatible(path).rows
rows.append(dict(zip(SEED_STATUS_FIELDS, sys.argv[2:])))
write_tsv_atomic(path, SEED_STATUS_FIELDS, rows)
PY
}

write_status() {
  local status=$1 code=$2 stage=$3
  local temporary
  temporary="$(mktemp "${run_dir}/.status.txt.XXXXXX")"
  {
    echo "STATUS=$status"
    echo "FAILED_STAGE=$stage"
    echo "EXIT_CODE=$code"
    echo "UPDATED_AT=$(date -Iseconds)"
  } >"$temporary"
  mv -f "$temporary" "${run_dir}/status.txt"
}

snapshot_source() {
  local destination=$1
  {
    for path in \
      "${source_run_root}/status.txt" \
      "${source_run_root}/experiment_manifest.txt" \
      "${source_run_root}/selected_amplitude_candidate.json" \
      "${source_run_root}/confirmation/units/best/checkpoint.pth" \
      "${source_run_root}/confirmation/units/last/checkpoint.pth"; do
      sha256sum "$path"
    done
  } >"$destination"
}

verify_source_unchanged() {
  local after="${run_dir}/source_snapshot_after.txt"
  snapshot_source "$after"
  cmp -s "${run_dir}/source_snapshot_before.txt" "$after" || {
    echo "Error: read-only V3 source changed during V4 execution" >&2
    return 1
  }
}

terminal_written=false
terminal_exit() {
  local status=$1 code=$2 stage=$3
  terminal_written=true
  append_stage_status "$stage" "$status" "$code" "terminal"
  verify_source_unchanged || true
  write_status "$status" "$code" "$stage"
  echo "STATUS=$status"
  echo "FAILED_STAGE=$stage"
  echo "EXIT_CODE=$code"
  if [[ "$shutdown_on_terminal" == "true" ]]; then
    "$shutdown_command" -h now || echo "shutdown_failed=true" >&2
  else
    echo "SHUTDOWN_EXECUTED=false"
  fi
  exit "$code"
}

on_exit() {
  local code=$?
  if [[ $code -ne 0 && "$terminal_written" != "true" ]]; then
    terminal_written=true
    append_stage_status "$current_stage" "FAILED" "$code" "unexpected_engineering_failure" || true
    verify_source_unchanged || true
    write_status "FAILED" "$code" "$current_stage" || true
    if [[ "$shutdown_on_terminal" == "true" ]]; then
      "$shutdown_command" -h now || true
    fi
  fi
}
trap on_exit EXIT

current_stage="preflight"
if [[ "$resume_mode" == "false" ]]; then
  {
    echo "PIPELINE_ID=mraf_ir_sensitivity_rescue_v4"
    echo "git_branch=$git_branch"
    echo "git_head=$git_head"
    echo "source_training_commit=c667946b2c51740bfa7892f5a2daee73d1279b44"
    echo "source_run_root=$source_run_root"
    echo "data_root=$data_root"
    echo "base_config=$base_config"
    echo "reference_screen_root=$reference_screen_root"
    echo "reference_run_root=$reference_run_root"
    echo "seed=1"
    echo "degrade_seed=2026"
    echo "readonly_scales=2.5,3,4,5"
    echo "formal_seeds=1,2,3,4,5"
  } >"${run_dir}/experiment_manifest.txt"
  "$python_bin" -B - "$run_dir" <<'PY'
from pathlib import Path
import sys
from tools.revision.summarize_mraf_ir_sensitivity_rescue import (
    RESUME_HISTORY_FIELDS, SEED_STATUS_FIELDS, STAGE_STATUS_FIELDS,
    write_tsv_atomic,
)
root = Path(sys.argv[1])
write_tsv_atomic(root / "stage_status.tsv", STAGE_STATUS_FIELDS, [])
write_tsv_atomic(root / "seed_status.tsv", SEED_STATUS_FIELDS, [])
write_tsv_atomic(root / "resume_history.tsv", RESUME_HISTORY_FIELDS, [])
PY
  snapshot_source "${run_dir}/source_snapshot_before.txt"
else
  [[ -s "${run_dir}/source_snapshot_before.txt" ]] || {
    echo "Error: resume source snapshot missing" >&2; exit 2;
  }
  verify_source_unchanged
  "$python_bin" -B - "${run_dir}/resume_history.tsv" \
    "$(date -Iseconds)" "$resume_status" "$resume_failed_stage" \
    "$git_head" "$next_stage" "validated_artifact_resume" <<'PY'
import sys
from pathlib import Path
from tools.revision.summarize_mraf_ir_sensitivity_rescue import (
    RESUME_HISTORY_FIELDS, read_tsv_compatible, write_tsv_atomic,
)
path = Path(sys.argv[1])
rows = read_tsv_compatible(path).rows
rows.append(dict(zip(RESUME_HISTORY_FIELDS, sys.argv[2:])))
write_tsv_atomic(path, RESUME_HISTORY_FIELDS, rows)
PY
fi
write_status "RUNNING" 0 "NONE"

[[ -z "$(run_git status --short)" ]] || {
  echo "Error: real V4 pipeline requires a clean Git worktree" >&2; exit 2;
}
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "Error: nvidia-smi not found" >&2; exit 2;
}
env "CUDA_VISIBLE_DEVICES=$device_index" "$python_bin" -B -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' \
  >"${run_dir}/gpu.txt"
if pgrep -af 'train.py.*mraf_ir_sensitivity' | grep -v "$$" >/dev/null 2>&1; then
  echo "Error: a V4 training process is already active" >&2
  exit 2
fi
if ps -eo args= | grep -F "$run_dir" \
    | grep -E '(train\.py|test\.py|diagnose_mraf_mechanism\.py)' \
    | grep -v -E '(grep -F|grep -E)' >/dev/null 2>&1; then
  echo "Error: a V4 train/eval/diagnosis process for this run-root is active" >&2
  exit 2
fi
"$python_bin" -B tools/revision/validate_mraf_rescue_config.py \
  --config "$base_config" --require-gate-rescue-protocol >/dev/null
"$python_bin" -B - "$base_config" <<'PY'
import sys
from utils.config import load_config_mapping
config = load_config_mapping(sys.argv[1])
for key in ("resume", "save_all_epoch_ckpt", "save_metric_best_ckpt",
            "save_full_eval_state_ckpt"):
    if bool(config.get(key, False)):
        raise SystemExit(f"base config must keep {key}=false")
for key in ("resume_ckpt_file", "load_ckpt_file"):
    if str(config.get(key, "") or "").strip():
        raise SystemExit(f"base config must keep {key} empty")
PY

common_train_args=(
  --config_file "$base_config"
  --dataset_name RGBT-PEDES --root_dir "$data_root"
  --img_aug --MLM --loss_names sdm+mlm+id
  --batch_size 64 --test_batch_size 512 --num_workers 8
  --lr 1e-5 --eval_use_fused_mu
  --log_period 100 --eval_period 1 --degrade_aug_mode none
  --use_rgbt_add_fusion --mraf_enabled --mraf_gate_fp32
  --mraf_hidden_dim 128 --mraf_dropout 0.1
  --mraf_gate_temperature 1.0 --mraf_target_temperature 0.1
  --mraf_balance_lambda 0.0
  --mdrc_enable --mdrc_prob 0.5 --mdrc_lambda 0.05
  --mdrc_loss_scale 100 --mdrc_warmup_epoch 3
  --mraf_scorer_lr_multiplier 50
  --mraf_rel_lambda 1
  --mraf_gate_base_grad_scale 1
  --save_last_checkpoint
)

run_logged() {
  local log=$1 pid_file=$2
  shift 2
  "$@" >"$log" 2>&1 &
  local child=$!
  printf '%s\n' "$child" >"$pid_file"
  wait "$child"
}

resolve_unique_artifact() {
  local root=$1 name=$2
  mapfile -t matches < <(find "$root" -type f -name "$name" -print)
  [[ ${#matches[@]} -eq 1 ]] || {
    echo "Expected one $name under $root; found ${#matches[@]}" >&2
    return 1
  }
  printf '%s\n' "${matches[0]}"
}

training_complete() {
  local root=$1
  [[ -d "$root/train_output" ]] || return 1
  for name in configs.yaml best.pth last.pth mraf_epoch_metrics.tsv \
      checkpoint_selection.tsv optimizer_param_groups.tsv; do
    resolve_unique_artifact "$root/train_output" "$name" >/dev/null || return 1
  done
  [[ -s "$root/train.log" ]]
}

run_training() {
  local attempt=$1 label=$2 epochs=$3 warmup=$4 formal_seed=$5 scale=$6
  [[ ! -e "$attempt" ]] || return 90
  mkdir -p "$attempt/train_output"
  local command=(
    env "CUDA_VISIBLE_DEVICES=$device_index" "TRAIN_SEED=$formal_seed"
    "$python_bin" -u train.py
    --name "$label" --output_dir "$attempt/train_output"
    --num_epoch "$epochs" --warmup_epochs "$warmup"
    --mraf_rel_warmup_epoch 1
    --mraf_gate_logit_scale "$scale"
    "${common_train_args[@]}"
  )
  printf '%q ' "${command[@]}" >"$attempt/training_command.txt"
  printf '\n' >>"$attempt/training_command.txt"
  run_logged "$attempt/train.log" "$attempt/train.pid" "${command[@]}"
  grep -Eiq 'CUDA out of memory|OutOfMemoryError|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
    "$attempt/train.log" && return 86
  training_complete "$attempt"
}

ensure_training() {
  local container=$1
  shift
  if training_complete "$container"; then
    effective_train_dir=$container
    return 0
  fi
  if [[ -e "$container" ]]; then
    local index=1 attempt
    while [[ -e "${container}/attempt_${index}" ]]; do index=$((index + 1)); done
    attempt="${container}/attempt_${index}"
    run_training "$attempt" "$@"
    printf '%s\n' "${attempt#${container}/}" >"${container}/active_attempt.txt"
    effective_train_dir=$attempt
  else
    run_training "$container" "$@"
    effective_train_dir=$container
  fi
}

scenario_values() {
  case "$1" in
    clean) printf '%s %s %s\n' none 1.0 7 ;;
    rgb_missing) printf '%s %s %s\n' rgb_missing 1.0 7 ;;
    ir_missing) printf '%s %s %s\n' ir_missing 1.0 7 ;;
    rgb_noise_0.05) printf '%s %s %s\n' rgb_noise 0.05 7 ;;
    rgb_noise_0.10) printf '%s %s %s\n' rgb_noise 0.10 7 ;;
    ir_noise_0.05) printf '%s %s %s\n' ir_noise 0.05 7 ;;
    ir_noise_0.10) printf '%s %s %s\n' ir_noise 0.10 7 ;;
    rgb_blur_k7) printf '%s %s %s\n' rgb_blur 1.0 7 ;;
    ir_blur_k7) printf '%s %s %s\n' ir_blur 1.0 7 ;;
    both_noise_0.10) printf '%s %s %s\n' both_noise 0.10 7 ;;
    both_blur_k7) printf '%s %s %s\n' both_blur 1.0 7 ;;
    *) echo "Error: unknown scenario $1" >&2; return 2 ;;
  esac
}

write_unit_json() {
  local path=$1 candidate=$2 checkpoint=$3 scale=$4 source_mode=$5
  "$python_bin" -B - "$path" "$candidate" "$checkpoint" "$scale" "$source_mode" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "candidate": sys.argv[2],
    "checkpoint": sys.argv[3],
    "mraf_scorer_lr_multiplier": 50,
    "mraf_gate_logit_scale": float(sys.argv[4]),
    "mraf_rel_lambda": 1,
    "mraf_gate_base_grad_scale": 1,
    "read_only_source": sys.argv[5] == "source",
    "has_nan": False,
    "has_inf": False,
    "has_oom": False,
}
fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".unit.", suffix=".tmp")
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
}

prepare_source_unit() {
  local unit=$1 checkpoint=$2 scale=$3
  [[ ! -e "$unit" ]] || return 0
  mkdir -p "$unit"
  local source="${source_run_root}/confirmation/units/${checkpoint}"
  for name in checkpoint.pth configs.yaml train.log mraf_epoch_metrics.tsv \
      checkpoint_selection.tsv; do
    ln -s "${source}/${name}" "${unit}/${name}"
  done
  write_unit_json "${unit}/unit.json" source_confirmation "$checkpoint" "$scale" source
}

prepare_trained_unit() {
  local unit=$1 train_dir=$2 candidate=$3 checkpoint=$4 scale=$5
  [[ ! -e "$unit" ]] || return 0
  mkdir -p "$unit"
  ln -s "$(resolve_unique_artifact "$train_dir/train_output" configs.yaml)" "${unit}/configs.yaml"
  ln -s "$(resolve_unique_artifact "$train_dir/train_output" "${checkpoint}.pth")" "${unit}/checkpoint.pth"
  ln -s "$train_dir/train.log" "${unit}/train.log"
  ln -s "$(resolve_unique_artifact "$train_dir/train_output" mraf_epoch_metrics.tsv)" "${unit}/mraf_epoch_metrics.tsv"
  ln -s "$(resolve_unique_artifact "$train_dir/train_output" checkpoint_selection.tsv)" "${unit}/checkpoint_selection.tsv"
  write_unit_json "${unit}/unit.json" "$candidate" "$checkpoint" "$scale" trained
}

run_evaluations() {
  local unit=$1 scenarios=$2 scale=$3
  local scenario mode strength kernel eval_dir log pid
  for scenario in $scenarios; do
    eval_dir="${unit}/eval/${scenario}"
    [[ -s "${eval_dir}/scores.pt" ]] && continue
    mkdir -p "$eval_dir"
    read -r mode strength kernel < <(scenario_values "$scenario")
    log="${eval_dir}/eval_${timestamp}.log"
    pid="${eval_dir}/pid_${timestamp}"
    run_logged "$log" "$pid" \
      env "CUDA_VISIBLE_DEVICES=$device_index" "$python_bin" -B test.py \
      --config_file "${unit}/configs.yaml" \
      --test_ckpt_file "${unit}/checkpoint.pth" \
      --mraf_gate_logit_scale "$scale" \
      --eval_degrade_mode "$mode" \
      --eval_degrade_strength "$strength" \
      --eval_degrade_blur_kernel "$kernel" \
      --eval_degrade_seed 2026 \
      --save_eval_scores --eval_scores_save_raw \
      --eval_scores_file "${eval_dir}/scores.pt"
  done
}

run_diagnosis() {
  local unit=$1 label=$2 batches=$3 scale=$4
  local directory="${unit}/diagnostics"
  [[ -s "${directory}/diagnosis.json" ]] && return 0
  if [[ -e "$directory" ]]; then
    directory="${unit}/diagnostics_resume_${timestamp}"
  fi
  run_logged "${unit}/diagnosis_${timestamp}.log" "${unit}/diagnosis_${timestamp}.pid" \
    env "CUDA_VISIBLE_DEVICES=$device_index" "$python_bin" -B \
    tools/revision/diagnose_mraf_mechanism.py \
    --config-file "${unit}/configs.yaml" \
    --checkpoint "${unit}/checkpoint.pth" \
    --case-label "$label" --data-root "$data_root" --device cuda:0 \
    --degrade-seed 2026 --max-batches "$batches" \
    --gradient-batches 3 --gradient-microbatch-size 4 \
    --mraf-gate-logit-scale "$scale" --output-dir "$directory"
  if [[ "$directory" != "${unit}/diagnostics" ]]; then
    basename "$directory" >"${unit}/active_diagnostics.txt"
  fi
}

evaluate_unit() {
  local unit=$1 label=$2 scenarios=$3 batches=$4 scale=$5
  run_evaluations "$unit" "$scenarios" "$scale"
  run_diagnosis "$unit" "$label" "$batches" "$scale"
}

screen_scenarios="clean rgb_missing ir_missing rgb_noise_0.10 ir_noise_0.10 rgb_blur_k7 ir_blur_k7 both_blur_k7"
formal_scenarios="clean rgb_missing ir_missing rgb_noise_0.05 rgb_noise_0.10 ir_noise_0.05 ir_noise_0.10 rgb_blur_k7 ir_blur_k7 both_noise_0.10 both_blur_k7"

full_units="${run_dir}/full_rediagnosis/units"
if ! stage_complete full_rediagnosis && ! stage_complete full_rediagnosis_strong; then
  current_stage="full_rediagnosis"
  append_stage_status "$current_stage" RUNNING 0 "scale=2,max_batches=24"
  for checkpoint in best last; do
    unit="${full_units}/${checkpoint}"
    prepare_source_unit "$unit" "$checkpoint" 2
    evaluate_unit "$unit" source_confirmation "$screen_scenarios" 24 2
  done
  full_result="${run_dir}/full_rediagnosis_result.json"
  "$python_bin" -B tools/revision/summarize_mraf_ir_sensitivity_rescue.py \
    --mode full-rediagnosis --candidate-root "$full_units" \
    --reference-screen-root "$reference_screen_root" --output-dir "$run_dir" \
    >"$full_result"
  full_status="$("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$full_result")"
  if [[ "$full_status" == "STRONG_SELECTED" ]]; then
    append_stage_status full_rediagnosis_strong COMPLETED 0 "selected_scale=2"
  else
    append_stage_status full_rediagnosis COMPLETED 0 "readonly_scale_required"
  fi
fi

selection_json="${run_dir}/selected_ir_rescue_scale.json"
if [[ ! -s "$selection_json" ]]; then
  current_stage="readonly_scale_screen"
  scale_root="${run_dir}/readonly_scale/units"
  append_stage_status "$current_stage" RUNNING 0 "scales=2.5,3,4,5"
  for scale in 2.5 3 4 5; do
    for checkpoint in best last; do
      unit="${scale_root}/scale_${scale}/${checkpoint}"
      prepare_source_unit "$unit" "$checkpoint" "$scale"
      evaluate_unit "$unit" "scale_${scale}" "$screen_scenarios" 4 "$scale"
    done
  done
  scale_result="${run_dir}/readonly_scale_result.json"
  "$python_bin" -B tools/revision/summarize_mraf_ir_sensitivity_rescue.py \
    --mode readonly-scale --candidate-root "$scale_root" \
    --reference-screen-root "$reference_screen_root" --output-dir "$run_dir" \
    >"$scale_result"
  scale_status="$("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$scale_result")"
  if [[ "$scale_status" != "STRONG_SELECTED" ]]; then
    terminal_exit IR_SENSITIVITY_RESCUE_FAILED 3 readonly_scale_screen
  fi
  append_stage_status readonly_scale_screen COMPLETED 0 "strong_scale_selected"
fi

selected_scale="$("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_scale"])' "$selection_json")"
selected_source_checkpoint="$("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "$selection_json")"

if [[ "$auto_continue" != "true" ]]; then
  append_stage_status awaiting_auto_continue RUNNING 0 "selected_scale=${selected_scale}"
  write_status RUNNING 0 awaiting_auto_continue
  terminal_written=true
  echo "STATUS=RUNNING"
  echo "NEXT_STAGE=ir_rescue_confirmation"
  echo "Selected scale is gated; rerun with --auto-continue."
  exit 0
fi

if "$python_bin" -B -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) > 2 else 1)' "$selected_scale"; then
  if ! stage_complete ir_rescue_confirmation; then
    current_stage="ir_rescue_confirmation"
    append_stage_status "$current_stage" RUNNING 0 "fresh_12_epoch"
    confirmation_train="${run_dir}/ir_rescue_confirmation/train"
    ensure_training "$confirmation_train" \
      "mraf_ir_sensitivity_confirmation_seed1" 12 1 1 "$selected_scale"
    confirmation_train=$effective_train_dir
    confirmation_units="${run_dir}/ir_rescue_confirmation/units"
    for checkpoint in best last; do
      unit="${confirmation_units}/${checkpoint}"
      prepare_trained_unit "$unit" "$confirmation_train" \
        ir_rescue_confirmation "$checkpoint" "$selected_scale"
      evaluate_unit "$unit" ir_rescue_confirmation \
        "$screen_scenarios" 24 "$selected_scale"
    done
    confirmation_result="${run_dir}/ir_rescue_confirmation_result.json"
    "$python_bin" -B tools/revision/summarize_mraf_ir_sensitivity_rescue.py \
      --mode confirmation --candidate-root "$confirmation_units" \
      --reference-screen-root "$reference_screen_root" --output-dir "$run_dir" \
      >"$confirmation_result"
    confirmation_status="$("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$confirmation_result")"
    if [[ "$confirmation_status" != "CONFIRMATION_PASSED" ]]; then
      terminal_exit IR_SENSITIVITY_CONFIRMATION_FAILED 4 ir_rescue_confirmation
    fi
    append_stage_status ir_rescue_confirmation COMPLETED 0 "strong_checkpoint_selected"
  fi
else
  if ! stage_complete full_rediagnosis_strong; then
    echo "Error: scale=2 may proceed only after STRONG full rediagnosis" >&2
    exit 2
  fi
fi

for formal_seed in 1 2 3 4 5; do
  if seed_complete "$formal_seed"; then
    echo "Skipping complete formal seed ${formal_seed}."
    continue
  fi
  current_stage="formal_seed${formal_seed}"
  append_stage_status "$current_stage" RUNNING 0 "fresh_60_epoch"
  formal_train="${run_dir}/formal/seed${formal_seed}/train"
  ensure_training "$formal_train" \
    "mraf_ir_sensitivity_formal_seed${formal_seed}" \
    60 3 "$formal_seed" "$selected_scale"
  formal_train=$effective_train_dir
  formal_units="${run_dir}/formal/seed${formal_seed}/units"
  for checkpoint in best last; do
    unit="${formal_units}/${checkpoint}"
    prepare_trained_unit "$unit" "$formal_train" \
      "formal_seed${formal_seed}" "$checkpoint" "$selected_scale"
    evaluate_unit "$unit" "formal_seed${formal_seed}" \
      "$formal_scenarios" 24 "$selected_scale"
  done
  formal_result="${run_dir}/formal_seed${formal_seed}_result.json"
  "$python_bin" -B tools/revision/summarize_mraf_ir_sensitivity_rescue.py \
    --mode formal-seed --candidate-root "$formal_units" \
    --reference-screen-root "$reference_screen_root" \
    --output-dir "$run_dir" --seed "$formal_seed" >"$formal_result"
  formal_status="$("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$formal_result")"
  selected_checkpoint="$("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "${run_dir}/selected_formal_seed${formal_seed}_checkpoint.json")"
  classification="$("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["classification"])' "${run_dir}/selected_formal_seed${formal_seed}_checkpoint.json")"
  if [[ "$formal_seed" == "1" && "$formal_status" != "STRONG_SELECTED" ]]; then
    append_seed_status 1 FAILED 5 "$selected_checkpoint" "$classification"
    terminal_exit FORMAL_SEED1_FAILED 5 formal_seed1_gate
  fi
  append_seed_status "$formal_seed" COMPLETED 0 "$selected_checkpoint" "$classification"
  append_stage_status "$current_stage" COMPLETED 0 "selected=${selected_checkpoint},${classification}"
done

current_stage="multiseed"
if ! stage_complete multiseed; then
  append_stage_status multiseed RUNNING 0 "seeds=1,2,3,4,5"
  if ! "$python_bin" -B tools/revision/summarize_mraf_ir_sensitivity_rescue.py \
      --mode multiseed --run-root "$run_dir" \
      --reference-screen-root "$reference_screen_root" \
      --reference-run-root "$reference_run_root" --output-dir "$run_dir" \
      >"${run_dir}/multiseed_result.json"; then
    terminal_exit MULTISEED_FAILED 6 multiseed
  fi
  append_stage_status multiseed COMPLETED 0 "paper_outputs_written"
fi

current_stage="archive"
archive="${run_dir}/mraf_ir_sensitivity_rescue_reports.tar.gz"
(
  cd "$run_dir"
  find . -type f \
    \( -name '*.tsv' -o -name '*.json' -o -name '*.md' \
       -o -name '*.txt' -o -name '*.log' -o -name '*.yaml' \) \
    ! -name '*.pth' ! -name '*.pt' ! -name 'scores.pt' -print \
    | sort >reports_index.txt
  grep -E '\.(pth|pt)$' reports_index.txt && {
    echo "Error: checkpoint entered report index" >&2
    exit 2
  }
  tar -czf "$(basename "$archive")" -T reports_index.txt
)
[[ -s "$archive" ]] || { echo "Error: final report archive is missing" >&2; exit 2; }
terminal_exit COMPLETED 0 NONE
