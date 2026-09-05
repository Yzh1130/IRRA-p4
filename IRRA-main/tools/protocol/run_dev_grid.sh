#!/usr/bin/env bash
# Run the frozen Stage 2 development grid on AutoDL.
#
# Checkpoint protocol: every scenario uses last.pth. 本文声明“统一采用训练结束时的
# 最终 checkpoint，不依据测试指标在多个 checkpoint 之间选择”。

set -uo pipefail
IFS=$'\n\t'

export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${OUT:-/root/autodl-tmp/DEV_GRID}"
BASE_CONFIG="${BASE_CONFIG:-}"
DEV_SPLIT="${DEV_SPLIT:-$PROJECT_ROOT/splits/dev_190.json}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"

usage() {
    printf '%s\n' \
        "Usage: $0 --base-config PATH [--out PATH] [--python PATH]" \
        "" \
        "Environment overrides: OUT, BASE_CONFIG, DEV_SPLIT, PYTHON_BIN," \
        "AUTO_SHUTDOWN (default 1), MIN_FREE_GB (default 20)."
}

while (($#)); do
    case "$1" in
        --base-config)
            (($# >= 2)) || { printf 'error: --base-config requires a value\n' >&2; exit 2; }
            BASE_CONFIG="$2"
            shift 2
            ;;
        --out)
            (($# >= 2)) || { printf 'error: --out requires a value\n' >&2; exit 2; }
            OUT="$2"
            shift 2
            ;;
        --python)
            (($# >= 2)) || { printf 'error: --python requires a value\n' >&2; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'error: unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

OUT="$(realpath -m "$OUT")"
DEV_SPLIT="$(realpath -m "$DEV_SPLIT")"
if [[ -n "$BASE_CONFIG" ]]; then
    BASE_CONFIG="$(realpath -m "$BASE_CONFIG")"
fi

mkdir -p "$OUT"/{configs,runs,scores,marks,logs}
CONFIG_DIR="$OUT/configs"
RUNS_DIR="$OUT/runs"
SCORES_DIR="$OUT/scores"
MARKS_DIR="$OUT/marks"
LOGS_DIR="$OUT/logs"
PROGRESS_TSV="$OUT/progress.tsv"
METRICS_TSV="$OUT/metrics.tsv"
FAILED_TSV="$OUT/failed.tsv"
DRIVER_LOG="$LOGS_DIR/driver.log"

exec > >(tee -a "$DRIVER_LOG") 2>&1

now() {
    date '+%Y-%m-%dT%H:%M:%S%z'
}

log() {
    printf '[%s] %s\n' "$(now)" "$*"
}

sanitize_tsv() {
    local value="$1"
    value="${value//$'\t'/ }"
    value="${value//$'\r'/ }"
    value="${value//$'\n'/ }"
    printf '%s' "$value"
}

ensure_tsv_header() {
    local path="$1"
    local header="$2"
    if [[ ! -s "$path" ]]; then
        printf '%b\n' "$header" > "$path"
    fi
}

ensure_tsv_header "$PROGRESS_TSV" 'timestamp\tgroup\tseed\tstatus\telapsed_seconds\tremaining_gb\trun_dir'
ensure_tsv_header "$METRICS_TSV" 'group\tseed\tscenario\tmode\tstrength\tkernel\tdseed\tsplit\tR1\tR5\tR10\tmAP\tmINP'
ensure_tsv_header "$FAILED_TSV" 'timestamp\tgroup\tseed\tstage\texit_code\treason'

append_failed() {
    local group="$1"
    local seed="$2"
    local stage="$3"
    local exit_code="$4"
    local reason
    reason="$(sanitize_tsv "$5")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(now)" "$group" "$seed" "$stage" "$exit_code" "$reason" \
        >> "$FAILED_TSV"
    log "[FAILED] group=$group seed=$seed stage=$stage exit=$exit_code reason=$reason"
}

remaining_kb() {
    df -Pk "$OUT" | awk 'NR == 2 {print $4}'
}

remaining_gb() {
    local available_kb
    available_kb="$(remaining_kb)" || return 1
    awk -v kb="$available_kb" 'BEGIN {printf "%.2f", kb / 1024 / 1024}'
}

has_minimum_disk() {
    local available_kb required_kb
    available_kb="$(remaining_kb)" || return 1
    required_kb=$((MIN_FREE_GB * 1024 * 1024))
    [[ "$available_kb" =~ ^[0-9]+$ ]] && ((available_kb >= required_kb))
}

on_signal() {
    append_failed "GLOBAL" "-" "signal" "130" "runner interrupted; automatic shutdown cancelled"
    log "Runner interrupted; leaving checkpoints and partial outputs in place."
    exit 130
}
trap on_signal INT TERM

GLOBAL_START="$(date +%s)"

if [[ -z "$BASE_CONFIG" ]]; then
    append_failed "GLOBAL" "-" "startup" "2" "--base-config or BASE_CONFIG is required"
    exit 2
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    append_failed "GLOBAL" "-" "startup" "2" "base config does not exist: $BASE_CONFIG"
    exit 2
fi
if [[ ! -f "$DEV_SPLIT" ]]; then
    append_failed "GLOBAL" "-" "startup" "2" "dev split does not exist: $DEV_SPLIT"
    exit 2
fi

cd "$PROJECT_ROOT"

log "Generating five development-grid configs from $BASE_CONFIG"
"$PYTHON_BIN" -m tools.protocol.gen_configs \
    --dev-grid \
    --base-config "$BASE_CONFIG" \
    --out-dir "$CONFIG_DIR"
rc=$?
if ((rc != 0)); then
    append_failed "GLOBAL" "-" "config_generation" "$rc" "dev-grid config generation failed"
    exit "$rc"
fi

"$PYTHON_BIN" - "$DEV_SPLIT" "$CONFIG_DIR" <<'PY'
import json
import sys
from pathlib import Path

from utils.config import load_config_mapping

split_path = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
with split_path.open("r", encoding="utf-8") as handle:
    split = json.load(handle)
dev_ids = split.get("dev_ids")
if not isinstance(dev_ids, list):
    raise SystemExit("dev split must contain a dev_ids list")
if split.get("n_dev") != 190 or len(dev_ids) != 190:
    raise SystemExit(
        f"dev split n_dev mismatch: metadata={split.get('n_dev')} ids={len(dev_ids)}"
    )
if split.get("n_train_reduced") != 1076:
    raise SystemExit(
        f"dev split n_train_reduced mismatch: {split.get('n_train_reduced')}"
    )
if len(set(dev_ids)) != len(dev_ids):
    raise SystemExit("dev split contains duplicate dev_ids")

expected = ("devA", "devT010", "devT020", "devT030", "devT050")
for name in expected:
    path = config_dir / f"{name}.yaml"
    if not path.is_file():
        raise SystemExit(f"missing generated config: {path}")
    config = load_config_mapping(path)
    if config.get("val_dataset") != "val":
        raise SystemExit(f"{name}: val_dataset must be val")
    if config.get("dev_split_file") != "splits/dev_190.json":
        raise SystemExit(f"{name}: unexpected dev_split_file={config.get('dev_split_file')!r}")
    if config.get("MLM") is not True:
        raise SystemExit(f"{name}: MLM must be true")
    if config.get("save_model_only") is not True:
        raise SystemExit(f"{name}: save_model_only must be true")
print("[STARTUP_ASSERT] dev split and five configs are valid")
PY
rc=$?
if ((rc != 0)); then
    append_failed "GLOBAL" "-" "startup_assert" "$rc" "dev split/config safety assertion failed"
    exit "$rc"
fi

GROUPS=(devA devT010 devT020 devT030 devT050)
SEEDS=(1 2 3)
SCENARIOS=(
    'clean|none|1.0|7|2026'
    'rgb_missing|rgb_missing|1.0|7|2026'
    'ir_missing|ir_missing|1.0|7|2026'
    'rgb_noise_005|rgb_noise|0.05|7|2026'
    'rgb_noise_010|rgb_noise|0.10|7|2026'
    'ir_noise_005|ir_noise|0.05|7|2026'
    'ir_noise_010|ir_noise|0.10|7|2026'
    'rgb_blur_k7|rgb_blur|1.0|7|2026'
    'ir_blur_k7|ir_blur|1.0|7|2026'
    'both_noise_005|both_noise|0.05|7|2026'
    'both_blur_k7|both_blur|1.0|7|2026'
)

assert_val_eval_command() {
    local -a command=("$@")
    local found=0 index argument value
    for ((index = 0; index < ${#command[@]}; index++)); do
        argument="${command[$index]}"
        if [[ "$argument" == "--eval_split" ]]; then
            ((index + 1 < ${#command[@]})) || return 1
            value="${command[$((index + 1))]}"
            [[ "$value" == "val" ]] || return 1
            found=$((found + 1))
        elif [[ "$argument" == --eval_split=* ]]; then
            value="${argument#--eval_split=}"
            [[ "$value" == "val" ]] || return 1
            found=$((found + 1))
        fi
    done
    ((found == 1))
}

extract_metric_row() {
    local score_file="$1"
    local group="$2"
    local seed="$3"
    local scenario="$4"
    local mode="$5"
    local strength="$6"
    local kernel="$7"
    local dseed="$8"
    "$PYTHON_BIN" - "$score_file" "$group" "$seed" "$scenario" \
        "$mode" "$strength" "$kernel" "$dseed" <<'PY'
import math
import sys

import torch

path, group, seed, scenario, mode, strength, kernel, dseed = sys.argv[1:]
try:
    payload = torch.load(path, map_location="cpu", weights_only=True)
except TypeError:
    payload = torch.load(path, map_location="cpu")
if not isinstance(payload, dict):
    raise SystemExit("score payload must be a mapping")
metadata = payload.get("metadata")
if not isinstance(metadata, dict) or metadata.get("eval_dataset") != "val":
    raise SystemExit(f"score metadata is not val: {metadata!r}")
metrics = payload.get("metrics")
required = ("R1", "R5", "R10", "mAP", "mINP")
if not isinstance(metrics, dict) or any(key not in metrics for key in required):
    raise SystemExit(f"score payload lacks metrics: {metrics!r}")
values = [float(metrics[key]) for key in required]
if not all(math.isfinite(value) for value in values):
    raise SystemExit(f"non-finite metrics: {values!r}")
fields = [group, seed, scenario, mode, strength, kernel, dseed, "val"]
fields.extend(format(value, ".10g") for value in values)
print("\t".join(fields))
PY
}

commit_unit_metrics() {
    local fragment="$1"
    local group="$2"
    local seed="$3"
    "$PYTHON_BIN" - "$METRICS_TSV" "$fragment" "$group" "$seed" <<'PY'
import csv
import os
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
fragment_path = Path(sys.argv[2])
group, seed = sys.argv[3:5]
header = [
    "group", "seed", "scenario", "mode", "strength", "kernel", "dseed",
    "split", "R1", "R5", "R10", "mAP", "mINP",
]
with metrics_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
rows = [row for row in rows if not (row["group"] == group and row["seed"] == seed)]
with fragment_path.open("r", encoding="utf-8", newline="") as handle:
    fragment_rows = list(csv.reader(handle, delimiter="\t"))
if len(fragment_rows) != 11 or any(len(row) != len(header) for row in fragment_rows):
    raise SystemExit(f"unit metrics fragment must contain 11 complete rows, got {len(fragment_rows)}")
rows.extend(dict(zip(header, row)) for row in fragment_rows)
temporary = metrics_path.with_name(f".{metrics_path.name}.tmp")
with temporary.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
os.replace(temporary, metrics_path)
PY
}

run_one() {
    local group="$1"
    local seed="$2"
    local run_key="${group}_seed${seed}"
    local mark="$MARKS_DIR/${run_key}.done"
    local config="$CONFIG_DIR/${group}.yaml"
    local run_start elapsed disk_gb timestamp attempt_root train_log train_rc
    local run_config actual_run_dir checkpoint score_dir unit_metrics_tmp

    if [[ -f "$mark" ]]; then
        log "[SKIP] $run_key already complete: $mark"
        return 0
    fi
    if ! has_minimum_disk; then
        disk_gb="$(remaining_gb 2>/dev/null || printf 'unknown')"
        append_failed "$group" "$seed" "disk_guard" "1" \
            "remaining disk ${disk_gb}GB is below ${MIN_FREE_GB}GB"
        return 1
    fi

    run_start="$(date +%s)"
    disk_gb="$(remaining_gb 2>/dev/null || printf 'unknown')"
    log "[RUN_START] group=$group seed=$seed remaining_disk_gb=$disk_gb"
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    attempt_root="$RUNS_DIR/$run_key/attempt_$timestamp"
    train_log="$LOGS_DIR/${run_key}_${timestamp}_train.log"
    score_dir="$SCORES_DIR/$run_key"
    unit_metrics_tmp="$OUT/.${run_key}.metrics.tmp"
    mkdir -p "$attempt_root" "$score_dir"
    : > "$unit_metrics_tmp"

    local -a train_command=(
        env "TRAIN_SEED=$seed"
        "$PYTHON_BIN" -u train.py
        --config_file "$config"
        --name "$run_key"
        --output_dir "$attempt_root"
        --dev_split_file "$DEV_SPLIT"
        --val_dataset val
        --MLM
    )
    log "[TRAIN_START] $run_key"
    "${train_command[@]}" 2>&1 | tee -a "$train_log"
    train_rc="${PIPESTATUS[0]}"
    if ((train_rc != 0)); then
        append_failed "$group" "$seed" "train" "$train_rc" "training process failed"
        return 1
    fi
    if ! grep -Eq '^\|[[:space:]]*train[[:space:]]*\|[[:space:]]*1076[[:space:]]*\|' "$train_log"; then
        append_failed "$group" "$seed" "train_identity_assert" "1" \
            "training log does not report exactly 1076 train identities"
        return 1
    fi
    log "[TRAIN_DONE] group=$group seed=$seed elapsed_seconds=$(($(date +%s) - run_start))"

    mapfile -t run_configs < <(find "$attempt_root" -type f -name configs.yaml -print)
    if ((${#run_configs[@]} != 1)); then
        append_failed "$group" "$seed" "locate_run" "1" \
            "expected exactly one configs.yaml under $attempt_root, found ${#run_configs[@]}"
        return 1
    fi
    run_config="${run_configs[0]}"
    actual_run_dir="$(dirname -- "$run_config")"
    checkpoint="$actual_run_dir/last.pth"
    if [[ ! -f "$checkpoint" ]]; then
        append_failed "$group" "$seed" "checkpoint" "1" \
            "required final checkpoint is missing: $checkpoint"
        return 1
    fi

    local scenario_spec scenario mode strength kernel dseed
    local score_file score_tmp eval_log eval_rc metric_row
    for scenario_spec in "${SCENARIOS[@]}"; do
        IFS='|' read -r scenario mode strength kernel dseed <<< "$scenario_spec"
        score_file="$score_dir/${scenario}.pt"
        score_tmp="$score_dir/.${scenario}.pt.tmp"
        eval_log="$LOGS_DIR/${run_key}_${timestamp}_${scenario}.log"
        local -a eval_command=(
            "$PYTHON_BIN" -u test.py
            --config_file "$run_config"
            --test_ckpt_file "$checkpoint"
            --dev_split_file "$DEV_SPLIT"
            --eval_split val
            --save_eval_scores
            --eval_scores_file "$score_tmp"
            --eval_degrade_mode "$mode"
            --eval_degrade_strength "$strength"
            --eval_degrade_blur_kernel "$kernel"
            --eval_degrade_seed "$dseed"
        )
        if ! assert_val_eval_command "${eval_command[@]}"; then
            append_failed "$group" "$seed" "eval_split_assert" "1" \
                "evaluation command must contain exactly one --eval_split val and never test/default"
            return 1
        fi
        log "[EVAL_START] group=$group seed=$seed scenario=$scenario mode=$mode"
        "${eval_command[@]}" 2>&1 | tee -a "$eval_log"
        eval_rc="${PIPESTATUS[0]}"
        if ((eval_rc != 0)); then
            append_failed "$group" "$seed" "eval:$scenario" "$eval_rc" \
                "evaluation process failed"
            return 1
        fi
        if ! grep -Fq '[EVAL_SPLIT] split=val num_images=478 num_captions=818 num_ids=190' "$eval_log"; then
            append_failed "$group" "$seed" "eval_dataset_assert:$scenario" "1" \
                "evaluation log does not prove the expected 190-identity dev split"
            return 1
        fi
        if [[ ! -f "$score_tmp" ]]; then
            append_failed "$group" "$seed" "score:$scenario" "1" \
                "score matrix was not written: $score_tmp"
            return 1
        fi
        if ! mv -f -- "$score_tmp" "$score_file"; then
            append_failed "$group" "$seed" "score_commit:$scenario" "1" \
                "failed to atomically publish score matrix: $score_file"
            return 1
        fi
        if ! metric_row="$(extract_metric_row "$score_file" "$group" "$seed" \
            "$scenario" "$mode" "$strength" "$kernel" "$dseed")"; then
            append_failed "$group" "$seed" "metric_extract:$scenario" "1" \
                "failed to validate score metadata or extract metrics"
            return 1
        fi
        printf '%s\n' "$metric_row" >> "$unit_metrics_tmp"
    done

    local fragment_rows
    fragment_rows="$(awk 'END {print NR + 0}' "$unit_metrics_tmp")"
    if [[ "$fragment_rows" != "11" ]]; then
        append_failed "$group" "$seed" "metrics_guard" "1" \
            "only $fragment_rows of 11 scenario metrics are complete; checkpoint retained"
        return 1
    fi
    if ! commit_unit_metrics "$unit_metrics_tmp" "$group" "$seed"; then
        append_failed "$group" "$seed" "metrics_commit" "1" \
            "failed to atomically update metrics.tsv; checkpoint retained"
        return 1
    fi
    local committed_rows
    committed_rows="$(awk -F '\t' -v g="$group" -v s="$seed" \
        'NR > 1 && $1 == g && $2 == s {count++} END {print count + 0}' \
        "$METRICS_TSV")"
    if [[ "$committed_rows" != "11" ]]; then
        append_failed "$group" "$seed" "metrics_guard" "1" \
            "metrics.tsv contains $committed_rows instead of 11 rows; checkpoint retained"
        return 1
    fi
    log "[METRICS_GUARD] group=$group seed=$seed rows=$committed_rows"

    local actual_real allowed_real checkpoint_count remaining_checkpoints
    elapsed="$(($(date +%s) - run_start))"
    disk_gb="$(remaining_gb 2>/dev/null || printf 'unknown')"
    if ! printf 'group=%s\nseed=%s\ncompleted_at=%s\nmetrics_rows=11\nrun_dir=%s\n' \
        "$group" "$seed" "$(now)" "$actual_run_dir" > "${mark}.tmp"; then
        append_failed "$group" "$seed" "mark_prepare" "1" \
            "could not prepare completion marker; checkpoint retained"
        return 1
    fi
    actual_real="$(realpath -m "$actual_run_dir")"
    allowed_real="$(realpath -m "$RUNS_DIR/$run_key")"
    case "$actual_real/" in
        "$allowed_real/"*) ;;
        *)
            append_failed "$group" "$seed" "checkpoint_path_guard" "1" \
                "refusing checkpoint deletion outside $allowed_real: $actual_real"
            return 1
            ;;
    esac
    checkpoint_count="$(find "$actual_run_dir" -type f \( -name '*.pth' -o -name '*.ckpt' \) | wc -l)"
    if ((checkpoint_count < 1)); then
        append_failed "$group" "$seed" "checkpoint_delete" "1" \
            "no checkpoint files found after successful evaluation"
        return 1
    fi
    if ! find "$actual_run_dir" -type f \( -name '*.pth' -o -name '*.ckpt' \) -delete; then
        append_failed "$group" "$seed" "checkpoint_delete" "1" \
            "checkpoint deletion command failed"
        return 1
    fi
    remaining_checkpoints="$(find "$actual_run_dir" -type f \( -name '*.pth' -o -name '*.ckpt' \) | wc -l)"
    if [[ "$remaining_checkpoints" != "0" ]]; then
        append_failed "$group" "$seed" "checkpoint_delete" "1" \
            "$remaining_checkpoints checkpoint files remain after deletion"
        return 1
    fi

    if ! mv -f -- "${mark}.tmp" "$mark"; then
        append_failed "$group" "$seed" "mark_commit" "1" \
            "checkpoint was deleted but completion marker could not be published"
        return 1
    fi
    if ! printf '%s\t%s\t%s\tcompleted\t%s\t%s\t%s\n' \
        "$(now)" "$group" "$seed" "$elapsed" "$disk_gb" "$actual_run_dir" \
        >> "$PROGRESS_TSV"; then
        append_failed "$group" "$seed" "progress" "1" \
            "completion marker exists, but progress.tsv append failed"
        return 1
    fi
    log "[EVAL_DONE] group=$group seed=$seed scenarios=11"
    log "[RUN_DONE] group=$group seed=$seed elapsed_seconds=$elapsed remaining_disk_gb=$disk_gb"
    return 0
}

for group in "${GROUPS[@]}"; do
    first_seed_failed=0
    for seed in "${SEEDS[@]}"; do
        if ((first_seed_failed)); then
            append_failed "$group" "$seed" "skipped_after_seed1" "1" \
                "seed 1 failed for this config; remaining seeds skipped by protocol"
            continue
        fi
        if ! run_one "$group" "$seed"; then
            if [[ "$seed" == "1" ]]; then
                first_seed_failed=1
            fi
        fi
    done
done

success_count=0
for group in "${GROUPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        [[ -f "$MARKS_DIR/${group}_seed${seed}.done" ]] && success_count=$((success_count + 1))
    done
done
total_runs=$((${#GROUPS[@]} * ${#SEEDS[@]}))
failed_count=$((total_runs - success_count))
total_elapsed="$(($(date +%s) - GLOBAL_START))"
disk_gb="$(remaining_gb 2>/dev/null || printf 'unknown')"
log "[GRID_SUMMARY] total_runs=$total_runs successful_runs=$success_count failed_runs=$failed_count total_elapsed_seconds=$total_elapsed remaining_disk_gb=$disk_gb"

if [[ "$AUTO_SHUTDOWN" != "1" ]]; then
    log "[SHUTDOWN] AUTO_SHUTDOWN=$AUTO_SHUTDOWN; automatic shutdown disabled"
    exit 0
fi

log "[SHUTDOWN] automatic shutdown scheduled after a 60-second grace period"
for remaining in $(seq 60 -1 1); do
    if [[ -e "$OUT/STOP" ]]; then
        log "[SHUTDOWN] cancelled because $OUT/STOP exists"
        exit 0
    fi
    if ((remaining == 60 || remaining % 10 == 0 || remaining <= 5)); then
        log "[SHUTDOWN] countdown=${remaining}s; create $OUT/STOP to cancel"
    fi
    sleep 1
done
if [[ -e "$OUT/STOP" ]]; then
    log "[SHUTDOWN] cancelled because $OUT/STOP exists"
    exit 0
fi
log "[SHUTDOWN] grace period complete; executing shutdown -h now"
shutdown -h now
