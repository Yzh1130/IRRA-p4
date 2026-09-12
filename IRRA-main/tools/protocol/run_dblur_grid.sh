#!/usr/bin/env bash
# Run a frozen supplementary control on the complete RGBT-PEDES train set.
#
# Checkpoint protocol: every evaluation uses last.pth. 本文声明“统一采用训练结束时的
# 最终 checkpoint，不依据测试指标在多个 checkpoint 之间选择”。

set -uo pipefail
IFS=$'\n\t'

export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${OUT:-/root/autodl-tmp/DBLUR_GRID}"
BASE_CONFIG="${BASE_CONFIG:-}"
FROZEN_HPARAMS="${FROZEN_HPARAMS:-$PROJECT_ROOT/frozen/FROZEN_HPARAMS.json}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
ARMS_SPEC=""
ARMS_SPEC_SET=0
OUT_SPEC_SET=0
SEEDS_SPEC="101-110"
VALIDATE_ONLY=0

usage() {
    printf '%s\n' \
        "Usage: $0 --base-config PATH [--arms Dblur|DnoNoise] [--seeds 101-105] [--out PATH]" \
        "          [--frozen-hparams PATH] [--python PATH] [--validate-only]" \
        "" \
        "Select exactly one of Dblur or DnoNoise; the default is Dblur." \
        "DnoNoise requires an explicit --out outside the default DBLUR_GRID tree." \
        "--seeds defaults to 101-110." \
        "--validate-only generates and validates configs, prints the run plan," \
        "and exits without training, evaluation, mechanism diagnostics, or shutdown." \
        "" \
        "Environment overrides: OUT, BASE_CONFIG, FROZEN_HPARAMS, PYTHON_BIN," \
        "AUTO_SHUTDOWN (default 1), MIN_FREE_GB (default 25)."
}

while (($#)); do
    case "$1" in
        --base-config)
            (($# >= 2)) || { printf 'error: --base-config requires a value\n' >&2; exit 2; }
            BASE_CONFIG="$2"
            shift 2
            ;;
        --arms)
            (($# >= 2)) || { printf 'error: --arms requires a value\n' >&2; exit 2; }
            ARMS_SPEC="$2"
            ARMS_SPEC_SET=1
            shift 2
            ;;
        --seeds)
            (($# >= 2)) || { printf 'error: --seeds requires a value\n' >&2; exit 2; }
            SEEDS_SPEC="$2"
            shift 2
            ;;
        --out)
            (($# >= 2)) || { printf 'error: --out requires a value\n' >&2; exit 2; }
            OUT="$2"
            OUT_SPEC_SET=1
            shift 2
            ;;
        --frozen-hparams)
            (($# >= 2)) || { printf 'error: --frozen-hparams requires a value\n' >&2; exit 2; }
            FROZEN_HPARAMS="$2"
            shift 2
            ;;
        --python)
            (($# >= 2)) || { printf 'error: --python requires a value\n' >&2; exit 2; }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --validate-only)
            VALIDATE_ONLY=1
            shift
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

if [[ "$ARMS_SPEC" == "DnoNoise" && "$OUT_SPEC_SET" != "1" ]]; then
    printf 'error: --arms DnoNoise requires an explicit independent --out\n' >&2
    exit 2
fi
if [[ -z "$OUT" ]]; then
    printf 'error: --out/OUT must not be empty\n' >&2
    exit 2
fi

OUT="$(realpath -m "$OUT")"
FROZEN_HPARAMS="$(realpath -m "$FROZEN_HPARAMS")"
if [[ -n "$BASE_CONFIG" ]]; then
    BASE_CONFIG="$(realpath -m "$BASE_CONFIG")"
fi

DEFAULT_DBLUR_GRID="$(realpath -m "/root/autodl-tmp/DBLUR_GRID")"
if [[ "$ARMS_SPEC" == "DnoNoise" \
    && ("$OUT" == "$DEFAULT_DBLUR_GRID" || "$OUT" == "$DEFAULT_DBLUR_GRID/"*) ]]; then
    printf 'error: DnoNoise --out must be independent of DBLUR_GRID: %s\n' "$OUT" >&2
    exit 2
fi

PROTECTED_MAIN_GRID="$(realpath -m "/root/autodl-tmp/MAIN_GRID")"
if [[ "$OUT" == "$PROTECTED_MAIN_GRID" || "$OUT" == "$PROTECTED_MAIN_GRID/"* ]]; then
    printf 'error: supplementary runner refuses to write into MAIN_GRID: %s\n' "$OUT" >&2
    exit 2
fi

if ! mkdir -p "$OUT"/{configs,runs,sim,mech,marks,logs}; then
    printf 'error: could not create supplementary output directories under %s\n' "$OUT" >&2
    exit 2
fi
CONFIG_DIR="$OUT/configs"
RUNS_DIR="$OUT/runs"
SIM_DIR="$OUT/sim"
MECH_DIR="$OUT/mech"
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
    append_failed "GLOBAL" "-" "signal" "130" \
        "runner interrupted; automatic shutdown cancelled"
    log "Runner interrupted; leaving last.pth and partial outputs in place."
    exit 130
}
trap on_signal INT TERM

GLOBAL_START="$(date +%s)"

ALL_ARM_NAMES=(Dblur DnoNoise)
ARM_NAMES=()
REQUESTED_ARM_NAMES=()
SEEDS=()
CORE_SCENARIOS=(
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
EXTENDED_SCENARIOS=(
    'rgb_noise_005_s2027|rgb_noise|0.05|7|2027'
    'rgb_noise_010_s2027|rgb_noise|0.10|7|2027'
    'ir_noise_005_s2027|ir_noise|0.05|7|2027'
    'ir_noise_010_s2027|ir_noise|0.10|7|2027'
    'both_noise_005_s2027|both_noise|0.05|7|2027'
    'rgb_noise_005_s2028|rgb_noise|0.05|7|2028'
    'rgb_noise_010_s2028|rgb_noise|0.10|7|2028'
    'ir_noise_005_s2028|ir_noise|0.05|7|2028'
    'ir_noise_010_s2028|ir_noise|0.10|7|2028'
    'both_noise_005_s2028|both_noise|0.05|7|2028'
    'rgb_brightness_03|rgb_brightness|0.3|7|2026'
    'ir_brightness_03|ir_brightness|0.3|7|2026'
    'rgb_lowres_025|rgb_lowres|0.25|7|2026'
    'ir_lowres_025|ir_lowres|0.25|7|2026'
    'rgb_jpeg_q20|rgb_jpeg|20|7|2026'
    'ir_jpeg_q20|ir_jpeg|20|7|2026'
    'rgb_noise_002|rgb_noise|0.02|7|2026'
    'rgb_noise_020|rgb_noise|0.20|7|2026'
    'ir_noise_002|ir_noise|0.02|7|2026'
    'ir_noise_020|ir_noise|0.20|7|2026'
    'both_noise_010|both_noise|0.10|7|2026'
    'rgb_blur_k3|rgb_blur|1.0|3|2026'
    'rgb_blur_k11|rgb_blur|1.0|11|2026'
    'ir_blur_k3|ir_blur|1.0|3|2026'
    'ir_blur_k11|ir_blur|1.0|11|2026'
    'both_blur_k3|both_blur|1.0|3|2026'
    'both_blur_k11|both_blur|1.0|11|2026'
)

# Bash array-name audit: script-owned arrays are ALL_ARM_NAMES, ARM_NAMES,
# REQUESTED_ARM_NAMES, SEEDS, CORE_SCENARIOS, EXTENDED_SCENARIOS, command,
# train_command, run_configs, checkpoint_files, scenario_specs, eval_command,
# pending_sim_sources, pending_sim_targets, mech_command, and
# missing_pilot_marks. None is a Bash special variable.
# BASH_SOURCE, BASH_REMATCH, and PIPESTATUS are shell-provided arrays that are
# read, never assigned. IFS is intentionally scoped for token parsing; GROUPS is
# never used.

if [[ "$SEEDS_SPEC" =~ ^(10[1-9]|110)-(10[1-9]|110)$ ]]; then
    seed_first="${BASH_REMATCH[1]}"
    seed_last="${BASH_REMATCH[2]}"
else
    append_failed "GLOBAL" "-" "seed_assert" "2" \
        "--seeds must be an ordered range within 101-110 (for example 101-105): $SEEDS_SPEC"
    exit 2
fi
if ((seed_first > seed_last)); then
    append_failed "GLOBAL" "-" "seed_assert" "2" \
        "--seeds range must be ordered from low to high: $SEEDS_SPEC"
    exit 2
fi
for ((seed_value = seed_first; seed_value <= seed_last; seed_value++)); do
    SEEDS+=("$seed_value")
done

if ((ARMS_SPEC_SET)); then
    if [[ -z "$ARMS_SPEC" ]]; then
        append_failed "GLOBAL" "-" "arms_assert" "2" "--arms must not be empty"
        exit 2
    fi
    if [[ ! "$ARMS_SPEC" =~ ^(Dblur|DnoNoise)$ ]]; then
        append_failed "GLOBAL" "-" "arms_assert" "2" \
            "--arms accepts exactly one of Dblur or DnoNoise: $ARMS_SPEC"
        exit 2
    fi
    IFS=',' read -r -a REQUESTED_ARM_NAMES <<< "$ARMS_SPEC"
    if ((${#REQUESTED_ARM_NAMES[@]} == 0)); then
        append_failed "GLOBAL" "-" "arms_assert" "2" "--arms selected no arms"
        exit 2
    fi
    for requested_arm in "${REQUESTED_ARM_NAMES[@]}"; do
        if [[ ! "$requested_arm" =~ ^(Dblur|DnoNoise)$ ]]; then
            append_failed "GLOBAL" "-" "arms_assert" "2" \
                "invalid supplementary arm: $requested_arm"
            exit 2
        fi
        duplicate_count=0
        for compared_arm in "${REQUESTED_ARM_NAMES[@]}"; do
            [[ "$compared_arm" == "$requested_arm" ]] \
                && duplicate_count=$((duplicate_count + 1))
        done
        if ((duplicate_count != 1)); then
            append_failed "GLOBAL" "-" "arms_assert" "2" \
                "duplicate supplementary arm: $requested_arm"
            exit 2
        fi
    done
    for canonical_arm in "${ALL_ARM_NAMES[@]}"; do
        for requested_arm in "${REQUESTED_ARM_NAMES[@]}"; do
            if [[ "$canonical_arm" == "$requested_arm" ]]; then
                ARM_NAMES+=("$canonical_arm")
                break
            fi
        done
    done
else
    ARM_NAMES=(Dblur)
fi

if ((${#ALL_ARM_NAMES[@]} != 2 || ${#ARM_NAMES[@]} != 1)); then
    append_failed "GLOBAL" "-" "arms_assert" "2" \
        "expected two canonical supplementary arms and exactly one selected arm"
    exit 2
fi
if ((${#SEEDS[@]} < 1 || ${#SEEDS[@]} > 10)); then
    append_failed "GLOBAL" "-" "seed_assert" "2" \
        "expected an ordered 101-110 seed range or subrange"
    exit 2
fi
if ((${#CORE_SCENARIOS[@]} != 11 || ${#EXTENDED_SCENARIOS[@]} != 27)); then
    append_failed "GLOBAL" "-" "scenario_assert" "2" \
        "expected 11 core and 27 extended scenarios"
    exit 2
fi

ARMS_CSV="$(IFS=,; printf '%s' "${ARM_NAMES[*]}")"
SEEDS_CSV="$(IFS=,; printf '%s' "${SEEDS[*]}")"
log "[ARMS] selected=$ARMS_CSV canonical_order=Dblur,DnoNoise"
log "[SEEDS] specification=$SEEDS_SPEC selected=$SEEDS_CSV"

if [[ ! "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || ((MIN_FREE_GB < 1)); then
    append_failed "GLOBAL" "-" "startup" "2" \
        "MIN_FREE_GB must be a positive integer, got $MIN_FREE_GB"
    exit 2
fi
if [[ "$AUTO_SHUTDOWN" != "0" && "$AUTO_SHUTDOWN" != "1" ]]; then
    append_failed "GLOBAL" "-" "startup" "2" \
        "AUTO_SHUTDOWN must be 0 or 1, got $AUTO_SHUTDOWN"
    exit 2
fi
if [[ -z "$BASE_CONFIG" ]]; then
    append_failed "GLOBAL" "-" "startup" "2" \
        "--base-config or BASE_CONFIG is required"
    exit 2
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    append_failed "GLOBAL" "-" "startup" "2" \
        "base config does not exist: $BASE_CONFIG"
    exit 2
fi
if [[ ! -f "$FROZEN_HPARAMS" ]]; then
    append_failed "GLOBAL" "-" "startup" "2" \
        "frozen hyperparameter file does not exist: $FROZEN_HPARAMS"
    exit 2
fi

cd "$PROJECT_ROOT"

"$PYTHON_BIN" - "$FROZEN_HPARAMS" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
if not isinstance(payload, dict):
    raise SystemExit("FROZEN_HPARAMS.json top level must be an object")
try:
    temperature = payload["E"]["mraf_target_temperature"]
    utility_prior = payload["U"]["mraf_utility_prior"]
except (KeyError, TypeError) as exc:
    raise SystemExit(f"FROZEN_HPARAMS.json lacks required E/U values: {exc}")
if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
    raise SystemExit("E.mraf_target_temperature must be numeric")
if not math.isfinite(float(temperature)) or float(temperature) != 0.2:
    raise SystemExit(
        f"E.mraf_target_temperature must equal 0.2, got {temperature!r}"
    )
if not isinstance(utility_prior, list) or len(utility_prior) != 2:
    raise SystemExit("U.mraf_utility_prior must be a two-element list")
values = []
for index, value in enumerate(utility_prior):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"U.mraf_utility_prior[{index}] must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(
            f"U.mraf_utility_prior[{index}] must be positive and finite"
        )
    values.append(value)
if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
    raise SystemExit(
        f"U.mraf_utility_prior must sum to 1 within 1e-6, got {sum(values)!r}"
    )
print(
    "[FROZEN_ASSERT] E.mraf_target_temperature=0.2 "
    f"U.mraf_utility_prior={values} sum={sum(values):.12f}"
)
PY
rc=$?
if ((rc != 0)); then
    append_failed "GLOBAL" "-" "frozen_assert" "$rc" \
        "frozen hyperparameter safety assertion failed"
    exit "$rc"
fi

log "Generating the Dblur/DnoNoise supplementary configs from $BASE_CONFIG"
"$PYTHON_BIN" -m tools.protocol.gen_configs \
    --base-config "$BASE_CONFIG" \
    --frozen-hparams "$FROZEN_HPARAMS" \
    --supplementary \
    --out-dir "$CONFIG_DIR"
rc=$?
if ((rc != 0)); then
    append_failed "GLOBAL" "-" "config_generation" "$rc" \
        "Dblur/DnoNoise supplementary config generation failed"
    exit "$rc"
fi

"$PYTHON_BIN" - "$CONFIG_DIR" <<'PY'
import sys
from pathlib import Path

from utils.config import load_config_mapping

config_dir = Path(sys.argv[1])
expected_arms = ("Dblur", "DnoNoise")
mraf_arms = {"Dblur", "DnoNoise"}
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
for arm in expected_arms:
    path = config_dir / f"{arm}.yaml"
    if not path.is_file():
        raise SystemExit(f"missing generated supplementary config: {path}")
    config = load_config_mapping(path)
    dev_split_file = config.get("dev_split_file", "")
    if dev_split_file is not None and str(dev_split_file).strip():
        raise SystemExit(
            f"{arm}.dev_split_file must be empty for full training, "
            f"got {dev_split_file!r}"
        )
    if str(config.get("val_dataset", "test")).lower() != "test":
        raise SystemExit(
            f"{arm}.val_dataset must be test, got {config.get('val_dataset')!r}"
        )
    if config.get("dataset_name") != "RGBT-PEDES":
        raise SystemExit(
            f"{arm}.dataset_name must be RGBT-PEDES, "
            f"got {config.get('dataset_name')!r}"
        )
    if config.get("MLM") is not True:
        raise SystemExit(f"{arm}.MLM must be true")
    if config.get("training") is not True:
        raise SystemExit(
            f"{arm}.training must be true, got {config.get('training')!r}"
        )
    expected_values = {
        "save_last_checkpoint": True,
        "save_model_only": True,
        "save_metric_best_ckpt": False,
        "save_all_epoch_ckpt": False,
    }
    for key, expected in expected_values.items():
        if config.get(key) is not expected:
            raise SystemExit(
                f"{arm}.{key} must be {expected!r}, got {config.get(key)!r}"
            )
    if bool(config.get("save_full_eval_state_ckpt", False)):
        raise SystemExit(f"{arm}.save_full_eval_state_ckpt must be false")
    if bool(config.get("verify_ckpt_immediately_after_save", False)):
        raise SystemExit(f"{arm}.verify_ckpt_immediately_after_save must be false")
    expected_mraf = arm in mraf_arms
    if config.get("mraf_enabled") is not expected_mraf:
        raise SystemExit(
            f"{arm}.mraf_enabled must be {expected_mraf!r}, "
            f"got {config.get('mraf_enabled')!r}"
        )
    expected_modes = expected_modes_by_arm[arm]
    modes = config.get("mdrc_degrade_modes")
    if not isinstance(modes, (list, tuple)) or list(modes) != expected_modes:
        raise SystemExit(
            f"{arm}.mdrc_degrade_modes must be {expected_modes!r}, got {modes!r}"
        )
print(
    "[CONFIG_ASSERT] Dblur/DnoNoise supplementary configs are present; full-train/test "
    "split, checkpoint policy, MRAF enablement, and MDRC modes are valid"
)
PY
rc=$?
if ((rc != 0)); then
    append_failed "GLOBAL" "-" "config_assert" "$rc" \
        "generated supplementary config safety assertion failed"
    exit "$rc"
fi

for arm_index in "${!ALL_ARM_NAMES[@]}"; do
    arm_name="${ALL_ARM_NAMES[$arm_index]}"
    if [[ ! "$arm_name" =~ ^(Dblur|DnoNoise)$ ]]; then
        append_failed "GLOBAL" "-" "config_assert" "2" \
            "invalid canonical arm at index $arm_index: $arm_name"
        exit 2
    fi
    if [[ ! -f "$CONFIG_DIR/${ALL_ARM_NAMES[$arm_index]}.yaml" ]]; then
        append_failed "GLOBAL" "-" "config_assert" "2" \
            "missing supplementary config: $CONFIG_DIR/${ALL_ARM_NAMES[$arm_index]}.yaml"
        exit 2
    fi
done

if ((VALIDATE_ONLY)); then
    planned_runs=$((${#ARM_NAMES[@]} * ${#SEEDS[@]}))
    log "[VALIDATE_ONLY] selected_arms=$ARMS_CSV planned_runs=$planned_runs"
    for arm_name in "${ARM_NAMES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            printf '[RUN_PLAN] %s_seed%s\n' "$arm_name" "$seed"
        done
    done
    log "[VALIDATE_ONLY_DONE] no training, evaluation, diagnostics, or shutdown executed"
    exit 0
fi

assert_no_option() {
    local forbidden="$1"
    shift
    local -a command=("$@")
    local argument
    for argument in "${command[@]}"; do
        if [[ "$argument" == "$forbidden" || "$argument" == "${forbidden}="* ]]; then
            return 1
        fi
    done
    return 0
}

assert_train_command() {
    local -a command=("$@")
    local argument mlm_count=0
    assert_no_option "--dev_split_file" "${command[@]}" || return 1
    for argument in "${command[@]}"; do
        [[ "$argument" == "--MLM" ]] && mlm_count=$((mlm_count + 1))
    done
    ((mlm_count == 1))
}

assert_test_command() {
    local -a command=("$@")
    assert_no_option "--eval_split" "${command[@]}" || return 1
    assert_no_option "--dev_split_file" "${command[@]}" || return 1
    return 0
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
if not isinstance(metadata, dict):
    raise SystemExit("score payload metadata must be a mapping")
if metadata.get("dataset_name") != "RGBT-PEDES":
    raise SystemExit(f"unexpected score dataset: {metadata.get('dataset_name')!r}")
if metadata.get("eval_dataset") != "test":
    raise SystemExit(f"score metadata is not test: {metadata!r}")
if metadata.get("score_shape") != [2490, 1473]:
    raise SystemExit(f"unexpected score shape: {metadata.get('score_shape')!r}")
if metadata.get("eval_degrade_mode") != mode:
    raise SystemExit(
        f"degrade mode mismatch: {metadata.get('eval_degrade_mode')!r} != {mode!r}"
    )
if not math.isclose(
    float(metadata.get("eval_degrade_strength")),
    float(strength),
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit("degrade strength mismatch")
if int(metadata.get("eval_degrade_blur_kernel")) != int(kernel):
    raise SystemExit("degrade blur kernel mismatch")
if int(metadata.get("eval_degrade_seed")) != int(dseed):
    raise SystemExit("degrade seed mismatch")
metrics = payload.get("metrics")
required = ("R1", "R5", "R10", "mAP", "mINP")
if not isinstance(metrics, dict) or any(key not in metrics for key in required):
    raise SystemExit(f"score payload lacks metrics: {metrics!r}")
values = [float(metrics[key]) for key in required]
if not all(math.isfinite(value) for value in values):
    raise SystemExit(f"non-finite metrics: {values!r}")
fields = [group, seed, scenario, mode, strength, kernel, dseed, "test"]
fields.extend(format(value, ".10g") for value in values)
print("\t".join(fields))
PY
}

commit_unit_metrics() {
    local fragment="$1"
    local group="$2"
    local seed="$3"
    local expected_rows="$4"
    "$PYTHON_BIN" - "$METRICS_TSV" "$fragment" "$group" "$seed" \
        "$expected_rows" <<'PY'
import csv
import os
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
fragment_path = Path(sys.argv[2])
group, seed = sys.argv[3:5]
expected_rows = int(sys.argv[5])
header = [
    "group", "seed", "scenario", "mode", "strength", "kernel", "dseed",
    "split", "R1", "R5", "R10", "mAP", "mINP",
]
with metrics_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames != header:
        raise SystemExit(
            f"metrics.tsv header mismatch: {reader.fieldnames!r} != {header!r}"
        )
    rows = list(reader)
rows = [row for row in rows if not (row["group"] == group and row["seed"] == seed)]
with fragment_path.open("r", encoding="utf-8", newline="") as handle:
    fragment_rows = list(csv.reader(handle, delimiter="\t"))
if len(fragment_rows) != expected_rows or any(
    len(row) != len(header) for row in fragment_rows
):
    raise SystemExit(
        f"unit metrics fragment must contain {expected_rows} complete rows, "
        f"got {len(fragment_rows)}"
    )
rows.extend(dict(zip(header, row)) for row in fragment_rows)
temporary = metrics_path.with_name(f".{metrics_path.name}.{os.getpid()}.tmp")
with temporary.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=header, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
os.replace(temporary, metrics_path)
PY
}

validate_model_only_checkpoint() {
    local checkpoint="$1"
    "$PYTHON_BIN" - "$checkpoint" <<'PY'
import sys

import torch

path = sys.argv[1]
try:
    payload = torch.load(path, map_location="cpu", weights_only=True)
except TypeError:
    payload = torch.load(path, map_location="cpu")
if not isinstance(payload, dict):
    raise SystemExit("last.pth must contain a mapping")
if set(payload) != {"model"}:
    raise SystemExit(
        f"last.pth must be model-only with keys {{'model'}}, got {sorted(payload)}"
    )
print("[CHECKPOINT_ASSERT] last.pth is model-only")
PY
}

read_run_data_root() {
    local run_config="$1"
    "$PYTHON_BIN" - "$run_config" <<'PY'
import sys
from pathlib import Path

from utils.config import load_config_mapping

config = load_config_mapping(Path(sys.argv[1]))
root = str(config.get("root_dir", "")).strip()
if not root:
    raise SystemExit("saved run config has no root_dir")
print(Path(root).expanduser().resolve())
PY
}

read_mdrc_degrade_modes() {
    local config="$1"
    "$PYTHON_BIN" - "$config" <<'PY'
import json
import sys
from pathlib import Path

from utils.config import load_config_mapping

config = load_config_mapping(Path(sys.argv[1]))
modes = config.get("mdrc_degrade_modes")
if not isinstance(modes, (list, tuple)):
    raise SystemExit("mdrc_degrade_modes must be a list or tuple")
print(json.dumps(list(modes), separators=(",", ":")))
PY
}

arm_has_mraf() {
    case "$1" in
        Dblur|DnoNoise) return 0 ;;
        *) return 2 ;;
    esac
}

run_one() {
    local group="$1"
    local seed="$2"
    local run_key="${group}_seed${seed}"
    local mark="$MARKS_DIR/${run_key}.done"
    local config="$CONFIG_DIR/${group}.yaml"
    local run_start elapsed disk_gb timestamp attempt_root train_log train_rc
    local mdrc_modes degraded_counts_lines mdrc_startup_lines
    local required_rgb_mode required_ir_mode forbidden_mode_pattern
    local run_config actual_run_dir checkpoint sim_run_dir unit_metrics_tmp
    local actual_real allowed_real checkpoint_real runs_root_real
    local sim_root_real sim_run_real metrics_parent_real out_real
    local -a run_configs=()
    local -a checkpoint_files=()
    local -a scenario_specs=("${CORE_SCENARIOS[@]}")
    local -a pending_sim_sources=()
    local -a pending_sim_targets=()

    case "$group" in
        Dblur)
            required_rgb_mode="rgb_blur"
            required_ir_mode="ir_blur"
            forbidden_mode_pattern="rgb_noise|ir_noise"
            ;;
        DnoNoise)
            required_rgb_mode="rgb_attenuate"
            required_ir_mode="ir_attenuate"
            forbidden_mode_pattern="rgb_noise|ir_noise|rgb_blur|ir_blur"
            ;;
        *)
            append_failed "$group" "$seed" "mode_assert_setup" "2" \
                "unsupported supplementary arm: $group"
            return 1
            ;;
    esac

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
    timestamp="$(date '+%Y%m%d_%H%M%S')_$$"
    attempt_root="$RUNS_DIR/$run_key/attempt_$timestamp"
    train_log="$LOGS_DIR/${run_key}_${timestamp}_train.log"
    sim_run_dir="$SIM_DIR/$run_key"
    unit_metrics_tmp="$OUT/.${run_key}.${timestamp}.metrics.tmp"
    mkdir -p "$attempt_root" "$sim_run_dir"
    : > "$unit_metrics_tmp"
    sim_root_real="$(realpath -m "$SIM_DIR")"
    sim_run_real="$(realpath -m "$sim_run_dir")"
    if [[ "$sim_run_real" != "$sim_root_real/$run_key" ]]; then
        append_failed "$group" "$seed" "sim_path_guard" "1" \
            "refusing sim writes outside $sim_root_real/$run_key: $sim_run_real"
        return 1
    fi

    if ! mdrc_modes="$(read_mdrc_degrade_modes "$config")"; then
        append_failed "$group" "$seed" "mdrc_modes_read" "1" \
            "could not read mdrc_degrade_modes from $config"
        return 1
    fi
    log "[MDRC_CONFIG] group=$group seed=$seed mdrc_degrade_modes=$mdrc_modes" \
        | tee -a "$train_log"

    local -a train_command=(
        env "TRAIN_SEED=$seed"
        "$PYTHON_BIN" -u train.py
        --config_file "$config"
        --name "$run_key"
        --output_dir "$attempt_root"
        --MLM
    )
    if ! assert_train_command "${train_command[@]}"; then
        append_failed "$group" "$seed" "train_command_assert" "1" \
            "training command must include --MLM exactly once and must never contain --dev_split_file"
        return 1
    fi
    log "[TRAIN_START] $run_key"
    "${train_command[@]}" 2>&1 | tee -a "$train_log"
    train_rc="${PIPESTATUS[0]}"
    if ((train_rc != 0)); then
        append_failed "$group" "$seed" "train" "$train_rc" \
            "training process failed"
        return 1
    fi
    mdrc_startup_lines="$(grep -F '[MDRC] enable=' "$train_log" || true)"
    if [[ -z "$mdrc_startup_lines" ]]; then
        append_failed "$group" "$seed" "mdrc_startup_assert" "1" \
            "training log has no [MDRC] enable=... modes=... startup entry"
        return 1
    fi
    if ! grep -Eq "modes=.*${required_rgb_mode}.*${required_ir_mode}" \
        <<< "$mdrc_startup_lines"; then
        append_failed "$group" "$seed" "mdrc_startup_assert" "1" \
            "[MDRC] startup entry does not include $required_rgb_mode and $required_ir_mode"
        return 1
    fi
    if grep -Eq "modes=.*(${forbidden_mode_pattern})" \
        <<< "$mdrc_startup_lines"; then
        append_failed "$group" "$seed" "mdrc_startup_assert" "1" \
            "[MDRC] startup entry includes a forbidden mode: $forbidden_mode_pattern"
        return 1
    fi
    log "[MDRC_STARTUP_ASSERT] group=$group seed=$seed required_modes=$required_rgb_mode,$required_ir_mode forbidden_modes=$forbidden_mode_pattern"
    if grep -Eiq 'nan' "$train_log"; then
        append_failed "$group" "$seed" "train_nan_assert" "1" \
            "training log contains a case-insensitive nan token"
        return 1
    fi
    degraded_counts_lines="$(grep -F '[DEGRADED_MODE_COUNTS]' "$train_log" || true)"
    if [[ -z "$degraded_counts_lines" ]]; then
        append_failed "$group" "$seed" "degraded_mode_counts_assert" "1" \
            "training log has no [DEGRADED_MODE_COUNTS] entries"
        return 1
    fi
    if ! grep -Eq "\"${required_rgb_mode}\"[[:space:]]*:[[:space:]]*[1-9][0-9]*" \
        <<< "$degraded_counts_lines"; then
        append_failed "$group" "$seed" "degraded_mode_counts_assert" "1" \
            "[DEGRADED_MODE_COUNTS] does not show a positive $required_rgb_mode count"
        return 1
    fi
    if ! grep -Eq "\"${required_ir_mode}\"[[:space:]]*:[[:space:]]*[1-9][0-9]*" \
        <<< "$degraded_counts_lines"; then
        append_failed "$group" "$seed" "degraded_mode_counts_assert" "1" \
            "[DEGRADED_MODE_COUNTS] does not show a positive $required_ir_mode count"
        return 1
    fi
    if grep -Eq "\"(${forbidden_mode_pattern})\"[[:space:]]*:" \
        <<< "$degraded_counts_lines"; then
        append_failed "$group" "$seed" "degraded_mode_counts_assert" "1" \
            "[DEGRADED_MODE_COUNTS] contains a forbidden mode: $forbidden_mode_pattern"
        return 1
    fi
    log "[DEGRADED_MODE_COUNTS_ASSERT] group=$group seed=$seed required_modes=$required_rgb_mode,$required_ir_mode forbidden_modes=$forbidden_mode_pattern"
    if ! grep -Eq '^\|[[:space:]]*train[[:space:]]*\|[[:space:]]*1266[[:space:]]*\|[[:space:]]*3250[[:space:]]*\|[[:space:]]*5497[[:space:]]*\|' "$train_log"; then
        append_failed "$group" "$seed" "train_identity_assert" "1" \
            "training log does not report train 1266/3250/5497"
        return 1
    fi
    if ! grep -Eq '^\|[[:space:]]*test[[:space:]]*\|[[:space:]]*556[[:space:]]*\|[[:space:]]*1473[[:space:]]*\|[[:space:]]*2490[[:space:]]*\|' "$train_log"; then
        append_failed "$group" "$seed" "train_test_split_assert" "1" \
            "training log does not report test 556/1473/2490"
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
    if [[ -e "$actual_run_dir/best.pth" ]]; then
        append_failed "$group" "$seed" "best_checkpoint_assert" "1" \
            "best.pth must not be generated when save_metric_best_ckpt=false"
        return 1
    fi
    if [[ -e "$actual_run_dir/checkpoint_selection.tsv" ]]; then
        append_failed "$group" "$seed" "checkpoint_selection_assert" "1" \
            "checkpoint_selection.tsv must not be generated when metric-best saving is disabled"
        return 1
    fi
    mapfile -t checkpoint_files < <(
        find "$actual_run_dir" -type f \( -name '*.pth' -o -name '*.ckpt' \) -print
    )
    if ((${#checkpoint_files[@]} != 1)); then
        append_failed "$group" "$seed" "checkpoint_count_assert" "1" \
            "expected only model-only last.pth, found ${#checkpoint_files[@]} checkpoint files"
        return 1
    fi
    checkpoint_real="$(realpath -m "$checkpoint")"
    if [[ "$(realpath -m "${checkpoint_files[0]}")" != "$checkpoint_real" ]]; then
        append_failed "$group" "$seed" "checkpoint_count_assert" "1" \
            "the sole checkpoint is not last.pth: ${checkpoint_files[0]}"
        return 1
    fi
    if ! validate_model_only_checkpoint "$checkpoint"; then
        append_failed "$group" "$seed" "checkpoint_model_only_assert" "1" \
            "last.pth is not a model-only checkpoint"
        return 1
    fi

    if ((seed <= 105)); then
        scenario_specs+=("${EXTENDED_SCENARIOS[@]}")
    fi
    local expected_metric_rows=11
    ((seed <= 105)) && expected_metric_rows=38
    local expected_sim_files=11
    ((seed <= 103)) && expected_sim_files=38

    local scenario_index scenario_spec scenario mode strength kernel dseed scope
    local score_file score_tmp eval_log eval_rc metric_row keep_sim
    for ((scenario_index = 0; scenario_index < ${#scenario_specs[@]}; scenario_index++)); do
        scenario_spec="${scenario_specs[$scenario_index]}"
        IFS='|' read -r scenario mode strength kernel dseed <<< "$scenario_spec"
        scope="extended"
        ((scenario_index < ${#CORE_SCENARIOS[@]})) && scope="core"
        score_file="$sim_run_dir/${scenario}.pt"
        score_tmp="$sim_run_dir/.${scenario}.${timestamp}.pt.tmp"
        eval_log="$LOGS_DIR/${run_key}_${timestamp}_${scenario}.log"
        if [[ -e "$score_tmp" ]]; then
            append_failed "$group" "$seed" "score_prepare:$scenario" "1" \
                "refusing to overwrite unexpected temporary score file: $score_tmp"
            return 1
        fi
        local -a eval_command=(
            "$PYTHON_BIN" -u test.py
            --config_file "$run_config"
            --test_ckpt_file "$checkpoint"
            --save_eval_scores
            --eval_scores_file "$score_tmp"
            --eval_degrade_mode "$mode"
            --eval_degrade_strength "$strength"
            --eval_degrade_blur_kernel "$kernel"
            --eval_degrade_seed "$dseed"
        )
        if ! assert_test_command "${eval_command[@]}"; then
            append_failed "$group" "$seed" "eval_command_assert" "1" \
                "evaluation command must use default test and never contain --eval_split or --dev_split_file"
            return 1
        fi
        log "[EVAL_START] group=$group seed=$seed scope=$scope scenario=$scenario mode=$mode"
        "${eval_command[@]}" 2>&1 | tee -a "$eval_log"
        eval_rc="${PIPESTATUS[0]}"
        if ((eval_rc != 0)); then
            append_failed "$group" "$seed" "eval:$scenario" "$eval_rc" \
                "evaluation process failed"
            return 1
        fi
        if ! grep -Fq '[EVAL_SPLIT] split=test num_images=1473 num_captions=2490 num_ids=556' "$eval_log"; then
            append_failed "$group" "$seed" "eval_dataset_assert:$scenario" "1" \
                "evaluation log does not prove the expected 556-identity test split"
            return 1
        fi
        if [[ ! -f "$score_tmp" ]]; then
            append_failed "$group" "$seed" "score:$scenario" "1" \
                "temporary score matrix was not written: $score_tmp"
            return 1
        fi
        if ! metric_row="$(extract_metric_row "$score_tmp" "$group" "$seed" \
            "$scenario" "$mode" "$strength" "$kernel" "$dseed")"; then
            append_failed "$group" "$seed" "metric_extract:$scenario" "1" \
                "failed to validate test score metadata or extract metrics"
            return 1
        fi
        printf '%s\n' "$metric_row" >> "$unit_metrics_tmp"

        keep_sim=0
        if [[ "$scope" == "core" ]] || ((seed <= 103)); then
            keep_sim=1
        fi
        if ((keep_sim)); then
            pending_sim_sources+=("$score_tmp")
            pending_sim_targets+=("$score_file")
        else
            if ! rm -f -- "$score_tmp"; then
                append_failed "$group" "$seed" "score_cleanup:$scenario" "1" \
                    "failed to remove non-retained extended score payload"
                return 1
            fi
        fi
    done

    local mech_run_dir="$MECH_DIR/$run_key"
    local mech_tmp=""
    local mech_log mech_rc data_root data_root_rc
    local native_train_log diagnostic_train_log
    if arm_has_mraf "$group"; then
        # diagnose_mraf_mechanism.py direct mode discovers an exact train.log,
        # while the repository logger names the same artifact train_log.txt.
        # A relative symlink satisfies that read-only discovery contract without
        # copying a potentially large log or altering its contents.
        native_train_log="$actual_run_dir/train_log.txt"
        diagnostic_train_log="$actual_run_dir/train.log"
        if [[ ! -f "$native_train_log" ]]; then
            append_failed "$group" "$seed" "mech_train_log" "1" \
                "native training log is missing: $native_train_log"
            return 1
        fi
        if [[ -e "$diagnostic_train_log" || -L "$diagnostic_train_log" ]]; then
            if [[ ! -f "$diagnostic_train_log" ]] || \
                [[ "$(realpath -m "$diagnostic_train_log")" != "$(realpath -m "$native_train_log")" ]]; then
                append_failed "$group" "$seed" "mech_train_log" "1" \
                    "existing train.log does not resolve to train_log.txt"
                return 1
            fi
        elif ! ln -s -- "train_log.txt" "$diagnostic_train_log"; then
            append_failed "$group" "$seed" "mech_train_log" "1" \
                "could not create the diagnostic train.log compatibility symlink"
            return 1
        fi
        data_root="$(read_run_data_root "$run_config")"
        data_root_rc=$?
        if ((data_root_rc != 0)) || [[ ! -d "$data_root" ]]; then
            append_failed "$group" "$seed" "mech_data_root" "1" \
                "could not resolve an existing data root from $run_config"
            return 1
        fi
        mech_tmp="$MECH_DIR/.${run_key}.${timestamp}.tmp"
        mech_log="$LOGS_DIR/${run_key}_${timestamp}_mech.log"
        if [[ -e "$mech_tmp" ]]; then
            append_failed "$group" "$seed" "mech_prepare" "1" \
                "diagnostic temporary output already exists: $mech_tmp"
            return 1
        fi
        local -a mech_command=(
            "$PYTHON_BIN" -u tools/revision/diagnose_mraf_mechanism.py
            --data-root "$data_root"
            --config-file "$run_config"
            --checkpoint "$checkpoint"
            --case-label "$run_key"
            --output-dir "$mech_tmp"
        )
        log "[MECH_START] group=$group seed=$seed output=$mech_run_dir"
        "${mech_command[@]}" 2>&1 | tee -a "$mech_log"
        mech_rc="${PIPESTATUS[0]}"
        if ((mech_rc != 0)); then
            append_failed "$group" "$seed" "mech" "$mech_rc" \
                "MRAF mechanism diagnostics failed"
            return 1
        fi
        if [[ ! -f "$mech_tmp/diagnosis.json" || ! -f "$mech_tmp/summary.txt" ]]; then
            append_failed "$group" "$seed" "mech_output_assert" "1" \
                "mechanism diagnostics did not produce diagnosis.json and summary.txt"
            return 1
        fi
        log "[MECH_DONE] group=$group seed=$seed"
    else
        log "[MECH_SKIP] group=$group seed=$seed reason=mraf_enabled_false"
    fi

    local fragment_rows committed_rows
    fragment_rows="$(awk 'END {print NR + 0}' "$unit_metrics_tmp")"
    if [[ "$fragment_rows" != "$expected_metric_rows" ]]; then
        append_failed "$group" "$seed" "metrics_guard" "1" \
            "only $fragment_rows of $expected_metric_rows scenario metrics are complete; last.pth retained"
        return 1
    fi
    if ! commit_unit_metrics "$unit_metrics_tmp" "$group" "$seed" \
        "$expected_metric_rows"; then
        append_failed "$group" "$seed" "metrics_commit" "1" \
            "failed to atomically update metrics.tsv; last.pth retained"
        return 1
    fi
    committed_rows="$(awk -F '\t' -v g="$group" -v s="$seed" \
        'NR > 1 && $1 == g && $2 == s {count++} END {print count + 0}' \
        "$METRICS_TSV")"
    if [[ "$committed_rows" != "$expected_metric_rows" ]]; then
        append_failed "$group" "$seed" "metrics_guard" "1" \
            "metrics.tsv contains $committed_rows instead of $expected_metric_rows rows; last.pth retained"
        return 1
    fi
    log "[METRICS_GUARD] group=$group seed=$seed rows=$committed_rows"

    local pending_index
    for ((pending_index = 0; pending_index < ${#pending_sim_sources[@]}; pending_index++)); do
        if ! mv -f -- "${pending_sim_sources[$pending_index]}" \
            "${pending_sim_targets[$pending_index]}"; then
            append_failed "$group" "$seed" "sim_publish" "1" \
                "failed to publish sim matrix ${pending_sim_targets[$pending_index]}"
            return 1
        fi
    done
    local published_sim_files
    published_sim_files="$(find "$sim_run_dir" -maxdepth 1 -type f -name '*.pt' | wc -l)"
    if [[ "$published_sim_files" != "$expected_sim_files" ]]; then
        append_failed "$group" "$seed" "sim_count_assert" "1" \
            "expected $expected_sim_files retained sim matrices, found $published_sim_files"
        return 1
    fi
    log "[SIM_GUARD] group=$group seed=$seed retained=$published_sim_files"

    if [[ -n "$mech_tmp" ]]; then
        if [[ -e "$mech_run_dir" ]]; then
            local stale_mech_dir="$MECH_DIR/.${run_key}.stale.${timestamp}"
            if [[ -e "$stale_mech_dir" ]] || ! mv -- "$mech_run_dir" "$stale_mech_dir"; then
                append_failed "$group" "$seed" "mech_publish" "1" \
                    "could not preserve the previous mechanism output before publishing"
                return 1
            fi
            log "[MECH_STALE] previous output preserved at $stale_mech_dir"
        fi
        if ! mv -- "$mech_tmp" "$mech_run_dir"; then
            append_failed "$group" "$seed" "mech_publish" "1" \
                "failed to publish mechanism diagnostics to $mech_run_dir"
            return 1
        fi
    fi

    actual_real="$(realpath -m "$actual_run_dir")"
    runs_root_real="$(realpath -m "$RUNS_DIR")"
    allowed_real="$runs_root_real/$run_key"
    case "$actual_real/" in
        "$allowed_real/"*) ;;
        *)
            append_failed "$group" "$seed" "cleanup_path_guard" "1" \
                "refusing cleanup outside $allowed_real: $actual_real"
            return 1
            ;;
    esac
    if [[ "$committed_rows" != "$expected_metric_rows" ]]; then
        append_failed "$group" "$seed" "tfevents_metrics_guard" "1" \
            "refusing tfevents cleanup before all metrics are committed"
        return 1
    fi
    if ! find "$actual_run_dir" -type f -name 'events.out.tfevents.*' -delete; then
        append_failed "$group" "$seed" "tfevents_cleanup" "1" \
            "failed to remove TensorBoard event files after metrics completion"
        return 1
    fi
    if [[ "$(find "$actual_run_dir" -type f -name 'events.out.tfevents.*' | wc -l)" != "0" ]]; then
        append_failed "$group" "$seed" "tfevents_cleanup" "1" \
            "TensorBoard event files remain after cleanup"
        return 1
    fi
    sim_run_real="$(realpath -m "$sim_run_dir")"
    if [[ "$sim_run_real" != "$sim_root_real/$run_key" ]]; then
        append_failed "$group" "$seed" "temporary_cleanup_path_guard" "1" \
            "refusing temporary score cleanup outside $sim_root_real/$run_key"
        return 1
    fi
    for scenario_spec in "${CORE_SCENARIOS[@]}" "${EXTENDED_SCENARIOS[@]}"; do
        IFS='|' read -r scenario mode strength kernel dseed <<< "$scenario_spec"
        if ! find "$sim_run_dir" -maxdepth 1 -type f \
            -name ".${scenario}.*.pt.tmp" -delete; then
            append_failed "$group" "$seed" "temporary_cleanup" "1" \
                "failed to remove runner-owned temporary score files for $scenario"
            return 1
        fi
    done
    out_real="$(realpath -m "$OUT")"
    metrics_parent_real="$(realpath -m "$(dirname -- "$unit_metrics_tmp")")"
    if [[ "$metrics_parent_real" != "$out_real" ]]; then
        append_failed "$group" "$seed" "temporary_cleanup_path_guard" "1" \
            "refusing metrics-fragment cleanup outside $out_real"
        return 1
    fi
    if ! find "$OUT" -maxdepth 1 -type f \
        -name ".${run_key}.*.metrics.tmp" -delete; then
        append_failed "$group" "$seed" "temporary_cleanup" "1" \
            "failed to remove runner-owned metrics fragments for $run_key"
        return 1
    fi
    if [[ ! -f "$checkpoint" ]]; then
        append_failed "$group" "$seed" "checkpoint_retention_assert" "1" \
            "model-only last.pth disappeared during cleanup"
        return 1
    fi

    elapsed="$(($(date +%s) - run_start))"
    disk_gb="$(remaining_gb 2>/dev/null || printf 'unknown')"
    if ! printf 'group=%s\nseed=%s\ncompleted_at=%s\nmetrics_rows=%s\nsim_files=%s\ncheckpoint=%s\nrun_dir=%s\n' \
        "$group" "$seed" "$(now)" "$expected_metric_rows" \
        "$expected_sim_files" "$checkpoint" "$actual_run_dir" > "${mark}.tmp"; then
        append_failed "$group" "$seed" "mark_prepare" "1" \
            "could not prepare completion marker; last.pth retained"
        return 1
    fi
    if ! mv -f -- "${mark}.tmp" "$mark"; then
        append_failed "$group" "$seed" "mark_commit" "1" \
            "could not publish completion marker; last.pth retained"
        return 1
    fi
    if ! printf '%s\t%s\t%s\tcompleted\t%s\t%s\t%s\n' \
        "$(now)" "$group" "$seed" "$elapsed" "$disk_gb" "$actual_run_dir" \
        >> "$PROGRESS_TSV"; then
        append_failed "$group" "$seed" "progress" "1" \
            "completion marker exists, but progress.tsv append failed"
        return 1
    fi
    log "[EVAL_DONE] group=$group seed=$seed scenarios=$expected_metric_rows"
    log "[RUN_DONE] group=$group seed=$seed elapsed_seconds=$elapsed remaining_disk_gb=$disk_gb checkpoint_retained=$checkpoint"
    return 0
}

engineering_gate_failed=0
for group in "${ARM_NAMES[@]}"; do
    first_seed_failed=0
    for seed in "${SEEDS[@]}"; do
        if ((seed >= 106 && (seed_first != 101 || seed_last != 110))); then
            missing_pilot_marks=()
            for pilot_seed in 101 102 103 104 105; do
                pilot_mark="$MARKS_DIR/${group}_seed${pilot_seed}.done"
                [[ -f "$pilot_mark" ]] || missing_pilot_marks+=("$pilot_mark")
            done
            if ((${#missing_pilot_marks[@]} != 0)); then
                append_failed "$group" "$seed" "engineering_gate" "1" \
                    "refusing seeds 106-110 until all seed 101-105 completion marks exist: ${missing_pilot_marks[*]}"
                engineering_gate_failed=1
                break
            fi
            log "[ENGINEERING_GATE] group=$group seed=$seed pilot_marks_101_105=complete"
        fi
        if ((first_seed_failed)); then
            append_failed "$group" "$seed" "skipped_after_seed101" "1" \
                "seed 101 failed for this config; remaining seeds skipped by protocol"
            continue
        fi
        if ! run_one "$group" "$seed"; then
            if [[ "$seed" == "101" ]]; then
                first_seed_failed=1
            fi
        fi
    done
done

success_count=0
for group in "${ARM_NAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        [[ -f "$MARKS_DIR/${group}_seed${seed}.done" ]] \
            && success_count=$((success_count + 1))
    done
done
total_runs=$((${#ARM_NAMES[@]} * ${#SEEDS[@]}))
failed_count=$((total_runs - success_count))
total_elapsed="$(($(date +%s) - GLOBAL_START))"
disk_gb="$(remaining_gb 2>/dev/null || printf 'unknown')"
log "[GRID_SUMMARY] total_runs=$total_runs successful_runs=$success_count failed_runs=$failed_count total_elapsed_seconds=$total_elapsed remaining_disk_gb=$disk_gb"

if ((engineering_gate_failed)); then
    log "[ENGINEERING_GATE] blocked=true; seeds 106-110 were not started"
    exit 1
fi

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
