#!/usr/bin/env bash
# Run the preregistered RGBNT201-PEDES external-validation protocol.
#
# Phase precondition trains A on train2 (141 identities) with seeds 1,2,3
# and evaluates dev2. Phase main trains A,D,E,U on all 171 train identities
# with seeds 201..210 and evaluates the untouched 30-identity test split.
# Every evaluation uses the final last.pth checkpoint; no metric-selected
# checkpoint is generated or used.

set -uo pipefail
IFS=$'\n\t'

export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT="${OUT:-/root/autodl-tmp/EXT_GRID}"
BASE_CONFIG="${BASE_CONFIG:-}"
FROZEN_HPARAMS="${FROZEN_HPARAMS:-$PROJECT_ROOT/frozen/FROZEN_HPARAMS.json}"
DEV2_SPLIT="${DEV2_SPLIT:-$PROJECT_ROOT/splits/rgbnt201_dev2_30.json}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-25}"
PHASE=""
VALIDATE_ONLY=0

PREDICTION_REL="docs/protocol/EXTERNAL_PREDICTION.md"
PREDICTION_FILE="$PROJECT_ROOT/$PREDICTION_REL"
REFERENCE_CLEAN_R1="37.92"
REFERENCE_WARNING_THRESHOLD="8"

usage() {
    printf '%s\n' \
        "Usage: $0 --phase {precondition|main} --base-config PATH [--out PATH]" \
        "          [--frozen-hparams PATH] [--dev2-split PATH] [--python PATH]" \
        "          [--validate-only]" \
        "" \
        "precondition: A x seeds 1,2,3 on train2; evaluate dev2 (val)." \
        "main: A,D,E,U x seeds 201..210 on full train; evaluate test." \
        "" \
        "Phase main is rejected until docs/protocol/EXTERNAL_PREDICTION.md" \
        "contains a non-empty prediction that is committed and unmodified." \
        "" \
        "Environment overrides: OUT, BASE_CONFIG, FROZEN_HPARAMS, DEV2_SPLIT," \
        "PYTHON_BIN, AUTO_SHUTDOWN (default 1), MIN_FREE_GB (default 25)."
}

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

append_failed() {
    local group="$1"
    local seed="$2"
    local stage="$3"
    local exit_code="$4"
    local reason
    reason="$(sanitize_tsv "$5")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(now)" "$PHASE" "$group" "$seed" "$stage" "$exit_code" "$reason" \
        >> "$FAILED_TSV"
    log "[FAILED] phase=$PHASE group=$group seed=$seed stage=$stage exit=$exit_code reason=$reason"
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

assert_exact_flag_count() {
    local flag="$1"
    local expected_count="$2"
    shift 2
    local -a command=("$@")
    local count=0 argument
    for argument in "${command[@]}"; do
        [[ "$argument" == "$flag" ]] && count=$((count + 1))
    done
    ((count == expected_count))
}

assert_exact_option_value() {
    local option="$1"
    local expected_value="$2"
    shift 2
    local -a command=("$@")
    local count=0 index argument value
    for ((index = 0; index < ${#command[@]}; index++)); do
        argument="${command[$index]}"
        if [[ "$argument" == "$option" ]]; then
            ((index + 1 < ${#command[@]})) || return 1
            value="${command[$((index + 1))]}"
            [[ "$value" == "$expected_value" ]] || return 1
            count=$((count + 1))
        elif [[ "$argument" == "$option="* ]]; then
            value="${argument#$option=}"
            [[ "$value" == "$expected_value" ]] || return 1
            count=$((count + 1))
        fi
    done
    ((count == 1))
}

assert_precondition_train_command() {
    local dev_split="$1"
    shift
    local -a command=("$@")
    assert_exact_flag_count "--MLM" 1 "${command[@]}" || return 1
    assert_exact_option_value "--dev_split_file" "$dev_split" \
        "${command[@]}" || return 1
    assert_exact_option_value "--val_dataset" "val" \
        "${command[@]}" || return 1
}

assert_precondition_test_command() {
    local dev_split="$1"
    shift
    local -a command=("$@")
    assert_exact_option_value "--dev_split_file" "$dev_split" \
        "${command[@]}" || return 1
    assert_exact_option_value "--eval_split" "val" \
        "${command[@]}" || return 1
}

assert_main_train_command() {
    local -a command=("$@")
    assert_exact_flag_count "--MLM" 1 "${command[@]}" || return 1
    assert_no_option "--dev_split_file" "${command[@]}" || return 1
}

assert_main_test_command() {
    local -a command=("$@")
    assert_no_option "--dev_split_file" "${command[@]}" || return 1
    # Default test is stronger and less ambiguous than accepting any explicit
    # eval split, so all --eval_split spellings are rejected in phase main.
    assert_no_option "--eval_split" "${command[@]}" || return 1
}

assert_phase_train_command() {
    local phase="$1"
    local dev_split="$2"
    shift 2
    case "$phase" in
        precondition)
            assert_precondition_train_command "$dev_split" "$@"
            ;;
        main)
            assert_main_train_command "$@"
            ;;
        *)
            return 2
            ;;
    esac
}

assert_phase_test_command() {
    local phase="$1"
    local dev_split="$2"
    shift 2
    case "$phase" in
        precondition)
            assert_precondition_test_command "$dev_split" "$@"
            ;;
        main)
            assert_main_test_command "$@"
            ;;
        *)
            return 2
            ;;
    esac
}

assert_external_prediction_preregistered() {
    local project_root="$1"
    local prediction_rel="$2"
    local prediction_file="$project_root/$prediction_rel"
    local prediction_commit prediction_status

    if [[ ! -f "$prediction_file" ]]; then
        printf 'prediction gate: required file does not exist: %s\n' \
            "$prediction_file" >&2
        return 21
    fi

    "$PYTHON_BIN" - "$prediction_file" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
heading = re.search(
    r"(?m)^##[ \t]+(?:2[.、][ \t]*)?依据前提写下的预测[ \t]*$",
    text,
)
if heading is None:
    raise SystemExit("prediction gate: missing heading '依据前提写下的预测'")
following = text[heading.end():]
next_heading = re.search(r"(?m)^##[ \t]+", following)
section = following[: next_heading.start()] if next_heading else following
section = re.sub(r"<!--.*?-->", "", section, flags=re.DOTALL).strip()
if not section:
    raise SystemExit(
        "prediction gate: section '依据前提写下的预测' is empty"
    )
print("[PREDICTION_CONTENT_ASSERT] non_empty=true")
PY
    local content_rc=$?
    if ((content_rc != 0)); then
        return 22
    fi

    prediction_commit="$(
        git -C "$project_root" log -1 --format=%H -- "$prediction_rel"
    )"
    if [[ -z "$prediction_commit" ]]; then
        printf 'prediction gate: file has no committed history: %s\n' \
            "$prediction_rel" >&2
        return 23
    fi

    prediction_status="$(
        git -C "$project_root" status --porcelain --untracked-files=all -- \
            "$prediction_rel"
    )"
    if [[ -n "$prediction_status" ]]; then
        printf 'prediction gate: file has uncommitted changes: %s\n%s\n' \
            "$prediction_rel" "$prediction_status" >&2
        return 24
    fi

    printf '[PREDICTION_GIT_ASSERT] commit=%s clean=true\n' \
        "$prediction_commit"
    return 0
}

validate_dev2_split() {
    local split_path="$1"
    "$PYTHON_BIN" - "$split_path" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"dev2 split file does not exist: {path}")
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"could not read dev2 split JSON: {exc}")
if not isinstance(payload, dict):
    raise SystemExit("dev2 split root must be an object")
required_ints = (
    "n_dev",
    "n_train_reduced",
    "dev_records",
    "dev_effective_captions",
    "train_reduced_records",
    "train_reduced_effective_captions",
)
for key in required_ints:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit(f"dev2 split {key} must be a non-negative integer")
if payload["n_dev"] != 30 or payload["n_train_reduced"] != 141:
    raise SystemExit(
        "dev2 split must have n_dev=30 and n_train_reduced=141, got "
        f"{payload['n_dev']} and {payload['n_train_reduced']}"
    )
if payload["n_dev"] + payload["n_train_reduced"] != 171:
    raise SystemExit("dev2 split identity conservation failed")
if payload.get("seed") != 20260907:
    raise SystemExit(
        f"dev2 split seed must be 20260907, got {payload.get('seed')!r}"
    )
if payload["dev_records"] + payload["train_reduced_records"] != 3951:
    raise SystemExit("dev2/train2 record counts must sum to 3951")
if (
    payload["dev_effective_captions"]
    + payload["train_reduced_effective_captions"]
    != 3948
):
    raise SystemExit("dev2/train2 effective-caption counts must sum to 3948")
dev_ids = payload.get("dev_ids")
train_ids = payload.get("train_reduced_ids")
if not isinstance(dev_ids, list) or len(dev_ids) != payload["n_dev"]:
    raise SystemExit("dev2 split dev_ids length does not match n_dev")
if not isinstance(train_ids, list) or len(train_ids) != payload["n_train_reduced"]:
    raise SystemExit(
        "dev2 split train_reduced_ids length does not match n_train_reduced"
    )
if any(isinstance(value, bool) or not isinstance(value, int) for value in dev_ids):
    raise SystemExit("dev2 split dev_ids must contain only integers")
if any(isinstance(value, bool) or not isinstance(value, int) for value in train_ids):
    raise SystemExit("dev2 split train_reduced_ids must contain only integers")
if len(set(dev_ids)) != len(dev_ids):
    raise SystemExit("dev2 split dev_ids must not contain duplicates")
if len(set(train_ids)) != len(train_ids):
    raise SystemExit("dev2 split train_reduced_ids must not contain duplicates")
if set(dev_ids) & set(train_ids):
    raise SystemExit("dev2 and train2 identities overlap")
if len(set(dev_ids) | set(train_ids)) != 171:
    raise SystemExit("dev2/train2 identity union must contain exactly 171 ids")
anno_md5 = payload.get("anno_md5")
if not isinstance(anno_md5, str) or re.fullmatch(r"[0-9a-fA-F]{32}", anno_md5) is None:
    raise SystemExit("dev2 split anno_md5 must be a 32-digit hexadecimal MD5")
artifact_md5 = hashlib.md5(path.read_bytes()).hexdigest()
print(
    "\t".join(
        str(value)
        for value in (
            payload["dev_records"],
            payload["dev_effective_captions"],
            payload["train_reduced_records"],
            payload["train_reduced_effective_captions"],
            anno_md5.lower(),
            artifact_md5,
        )
    )
)
PY
}

validate_frozen_hparams() {
    local frozen_path="$1"
    "$PYTHON_BIN" - "$frozen_path" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"frozen hyperparameter file does not exist: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
temperature = payload["E"]["mraf_target_temperature"]
prior = payload["U"]["mraf_utility_prior"]
if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
    raise SystemExit("E.mraf_target_temperature must be numeric")
if not math.isfinite(float(temperature)) or float(temperature) != 0.2:
    raise SystemExit(
        f"E.mraf_target_temperature must equal 0.2, got {temperature!r}"
    )
if not isinstance(prior, list) or len(prior) != 2:
    raise SystemExit("U.mraf_utility_prior must contain two elements")
values = [float(value) for value in prior]
if any(not math.isfinite(value) or value <= 0.0 for value in values):
    raise SystemExit("U.mraf_utility_prior values must be positive and finite")
if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
    raise SystemExit("U.mraf_utility_prior must sum to one within 1e-6")
print(
    "[FROZEN_ASSERT] E.mraf_target_temperature=0.2 "
    f"U.mraf_utility_prior={values} sum={sum(values):.12f}"
)
PY
}

# Script-owned arrays: ARM_NAMES, SEEDS, CORE_SCENARIOS, command,
# train_command, eval_command, run_configs, checkpoint_files,
# pending_score_sources, pending_score_targets, and mech_command. None is a
# Bash special variable. BASH_SOURCE and PIPESTATUS are shell-provided arrays
# that are read but never assigned; IFS is intentionally used for field
# parsing. GROUPS is never used.

validate_generated_configs() {
    local frozen_path="$1"
    local config_dir="$2"
    local phase="$3"
    "$PYTHON_BIN" - "$frozen_path" "$config_dir" "$phase" <<'PY'
import json
import math
import sys
from pathlib import Path

from utils.config import load_config_mapping

frozen_path = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
phase = sys.argv[3]
frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
frozen_temperature = float(frozen["E"]["mraf_target_temperature"])
frozen_prior = [float(value) for value in frozen["U"]["mraf_utility_prior"]]
selected = ("A",) if phase == "precondition" else ("A", "D", "E", "U")
mraf_arms = {"D", "E", "U"}

for arm in selected:
    path = config_dir / f"{arm}.yaml"
    if not path.is_file():
        raise SystemExit(f"missing external-grid config: {path}")
    config = load_config_mapping(path)
    required = {
        "dataset_name": "RGBNT201-PEDES",
        "training": True,
        "MLM": True,
        "save_last_checkpoint": True,
        "save_model_only": True,
        "save_metric_best_ckpt": False,
        "save_all_epoch_ckpt": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"{arm}.{key} must be {expected!r}, got {config.get(key)!r}"
            )
    if bool(config.get("save_full_eval_state_ckpt", False)):
        raise SystemExit(f"{arm}.save_full_eval_state_ckpt must be false")
    if bool(config.get("verify_ckpt_immediately_after_save", False)):
        raise SystemExit(
            f"{arm}.verify_ckpt_immediately_after_save must be false"
        )
    if not str(config.get("root_dir", "")).strip():
        raise SystemExit(f"{arm}.root_dir must be non-empty")
    dev_value = config.get("dev_split_file", "")
    if dev_value is not None and str(dev_value).strip():
        raise SystemExit(
            f"{arm}.dev_split_file in the base config must be empty; "
            f"phase-specific CLI arguments own dev2, got {dev_value!r}"
        )
    if str(config.get("val_dataset", "test")).lower() != "test":
        raise SystemExit(
            f"{arm}.val_dataset in the base config must be test, "
            f"got {config.get('val_dataset')!r}"
        )
    expected_mraf = arm in mraf_arms
    if bool(config.get("mraf_enabled")) is not expected_mraf:
        raise SystemExit(
            f"{arm}.mraf_enabled must be {expected_mraf}, "
            f"got {config.get('mraf_enabled')!r}"
        )

e_config = load_config_mapping(config_dir / "E.yaml")
if float(e_config.get("mraf_target_temperature")) != frozen_temperature:
    raise SystemExit("E.yaml temperature does not match frozen E value")
if frozen_temperature != 0.2:
    raise SystemExit("external protocol requires frozen E temperature 0.2")
u_config = load_config_mapping(config_dir / "U.yaml")
u_prior = u_config.get("mraf_utility_prior")
if not isinstance(u_prior, (list, tuple)) or len(u_prior) != 2:
    raise SystemExit("U.yaml utility prior must contain two values")
if any(
    not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    for actual, expected in zip(u_prior, frozen_prior)
):
    raise SystemExit("U.yaml utility prior does not match frozen U values")

print(
    f"[CONFIG_ASSERT] phase={phase} selected={','.join(selected)} "
    "dataset=RGBNT201-PEDES base_split=test"
)
PY
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
    local split="$9"
    local expected_queries="${10}"
    local expected_gallery="${11}"
    "$PYTHON_BIN" - "$score_file" "$group" "$seed" "$scenario" \
        "$mode" "$strength" "$kernel" "$dseed" "$split" \
        "$expected_queries" "$expected_gallery" <<'PY'
import math
import sys

import torch

(
    path,
    group,
    seed,
    scenario,
    mode,
    strength,
    kernel,
    dseed,
    split,
    expected_queries,
    expected_gallery,
) = sys.argv[1:]
try:
    payload = torch.load(path, map_location="cpu", weights_only=True)
except TypeError:
    payload = torch.load(path, map_location="cpu")
if not isinstance(payload, dict):
    raise SystemExit("score payload must be a mapping")
metadata = payload.get("metadata")
if not isinstance(metadata, dict):
    raise SystemExit("score payload metadata must be a mapping")
if metadata.get("dataset_name") != "RGBNT201-PEDES":
    raise SystemExit(f"unexpected score dataset: {metadata.get('dataset_name')!r}")
if metadata.get("eval_dataset") != split:
    raise SystemExit(
        f"unexpected score split: {metadata.get('eval_dataset')!r} != {split!r}"
    )
expected_shape = [int(expected_queries), int(expected_gallery)]
if metadata.get("score_shape") != expected_shape:
    raise SystemExit(
        f"unexpected score shape: {metadata.get('score_shape')!r} "
        f"!= {expected_shape!r}"
    )
if metadata.get("eval_degrade_mode") != mode:
    raise SystemExit("degrade mode mismatch")
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
fields = [group, seed, scenario, mode, strength, kernel, dseed, split]
fields.extend(format(value, ".10g") for value in values)
print("\t".join(fields))
PY
}

commit_unit_metrics() {
    local fragment="$1"
    local group="$2"
    local seed="$3"
    local split="$4"
    local expected_rows="$5"
    "$PYTHON_BIN" - "$METRICS_TSV" "$fragment" "$group" "$seed" \
        "$split" "$expected_rows" <<'PY'
import csv
import os
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
fragment_path = Path(sys.argv[2])
group, seed, split = sys.argv[3:6]
expected_rows = int(sys.argv[6])
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
rows = [
    row
    for row in rows
    if not (
        row["group"] == group
        and row["seed"] == seed
        and row["split"] == split
    )
]
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
if not isinstance(payload, dict) or set(payload) != {"model"}:
    keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
    raise SystemExit(
        f"last.pth must be model-only with keys {{'model'}}, got {keys!r}"
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

assert_dataset_table_row() {
    local log_path="$1"
    local subset="$2"
    local identities="$3"
    local images="$4"
    local captions="$5"
    grep -Eq \
        "^\|[[:space:]]*${subset}[[:space:]]*\|[[:space:]]*${identities}[[:space:]]*\|[[:space:]]*${images}[[:space:]]*\|[[:space:]]*${captions}[[:space:]]*\|" \
        "$log_path"
}

arm_has_mraf() {
    local phase="$1"
    local arm="$2"
    [[ "$phase" == "main" ]] || return 1
    case "$arm" in
        D|E|U) return 0 ;;
        A) return 1 ;;
        *) return 2 ;;
    esac
}

update_external_warning() {
    local group="$1"
    local seed="$2"
    local clean_r1="$3"
    "$PYTHON_BIN" - "$WARNINGS_TSV" "$group" "$seed" "$clean_r1" \
        "$REFERENCE_CLEAN_R1" "$REFERENCE_WARNING_THRESHOLD" "$(now)" <<'PY'
import csv
import math
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
group, seed = sys.argv[2:4]
observed = float(sys.argv[4])
reference = float(sys.argv[5])
threshold = float(sys.argv[6])
timestamp = sys.argv[7]
if not all(math.isfinite(value) for value in (observed, reference, threshold)):
    raise SystemExit("warning comparison received a non-finite value")
header = [
    "timestamp", "group", "seed", "observed_clean_R1",
    "reference_clean_R1", "absolute_difference", "threshold", "reason",
]
with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames != header:
        raise SystemExit("warnings.tsv header mismatch")
    rows = [
        row for row in reader
        if not (row["group"] == group and row["seed"] == seed)
    ]
difference = abs(observed - reference)
if difference > threshold:
    rows.append(
        {
            "timestamp": timestamp,
            "group": group,
            "seed": seed,
            "observed_clean_R1": format(observed, ".10g"),
            "reference_clean_R1": format(reference, ".10g"),
            "absolute_difference": format(difference, ".10g"),
            "threshold": format(threshold, ".10g"),
            "reason": "IRRA external-anchor deviation exceeds 8 percentage points",
        }
    )
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
with temporary.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=header, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
os.replace(temporary, path)
print(
    f"[BASELINE_WARNING_CHECK] group={group} seed={seed} "
    f"observed={observed:.4f} reference={reference:.2f} "
    f"absolute_difference={difference:.4f} warned={difference > threshold}"
)
PY
}

on_signal() {
    append_failed "GLOBAL" "-" "signal" "130" \
        "runner interrupted; automatic shutdown cancelled"
    log "Runner interrupted; retaining checkpoints and partial outputs."
    exit 130
}

run_one() {
    local group="$1"
    local seed="$2"
    local run_key="${PHASE}_${group}_seed${seed}"
    local mark="$MARKS_DIR/${run_key}.done"
    local config="$CONFIG_DIR/${group}.yaml"
    local run_start elapsed disk_gb timestamp attempt_root train_log train_rc
    local run_config actual_run_dir actual_real expected_run_root runs_root_real
    local checkpoint checkpoint_real checkpoint_status retain_checkpoint
    local score_run_dir score_root_real score_run_real unit_metrics_tmp
    local split expected_queries expected_gallery expected_train_records
    local expected_train_captions expected_val_records expected_val_captions
    local expected_log_line expected_rows=11 clean_r1=""
    local -a run_configs=()
    local -a checkpoint_files=()
    local -a pending_score_sources=()
    local -a pending_score_targets=()

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
    log "[RUN_START] phase=$PHASE group=$group seed=$seed remaining_disk_gb=$disk_gb"
    timestamp="$(date '+%Y%m%d_%H%M%S')_$$"
    attempt_root="$RUNS_DIR/$run_key/attempt_$timestamp"
    train_log="$LOGS_DIR/${run_key}_${timestamp}_train.log"
    score_run_dir="$SCORES_DIR/$run_key"
    unit_metrics_tmp="$OUT/.${run_key}.${timestamp}.metrics.tmp"
    mkdir -p "$attempt_root" "$score_run_dir"
    : > "$unit_metrics_tmp"

    score_root_real="$(realpath -m "$SCORES_DIR")"
    score_run_real="$(realpath -m "$score_run_dir")"
    if [[ "$score_run_real" != "$score_root_real/$run_key" ]]; then
        append_failed "$group" "$seed" "score_path_guard" "1" \
            "refusing score writes outside $score_root_real/$run_key"
        return 1
    fi

    local -a train_command=(
        env "TRAIN_SEED=$seed"
        "$PYTHON_BIN" -u train.py
        --config_file "$config"
        --name "$run_key"
        --output_dir "$attempt_root"
        --MLM
    )
    if [[ "$PHASE" == "precondition" ]]; then
        train_command+=(
            --dev_split_file "$DEV2_SPLIT"
            --val_dataset val
        )
    fi
    if ! assert_phase_train_command "$PHASE" "$DEV2_SPLIT" \
        "${train_command[@]}"; then
        append_failed "$group" "$seed" "train_command_assert" "1" \
            "phase-specific training command assertion failed"
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

    if [[ "$PHASE" == "precondition" ]]; then
        expected_train_records="$DEV2_TRAIN_RECORDS"
        expected_train_captions="$DEV2_TRAIN_CAPTIONS"
        expected_val_records="$DEV2_DEV_RECORDS"
        expected_val_captions="$DEV2_DEV_CAPTIONS"
        if ! assert_dataset_table_row "$train_log" train 141 \
            "$expected_train_records" "$expected_train_captions"; then
            append_failed "$group" "$seed" "train_split_assert" "1" \
                "training log does not report the exact 141-identity train2 split"
            return 1
        fi
        if ! assert_dataset_table_row "$train_log" val 30 \
            "$expected_val_records" "$expected_val_captions"; then
            append_failed "$group" "$seed" "val_split_assert" "1" \
                "training log does not report the exact 30-identity dev2 split"
            return 1
        fi
    else
        if ! assert_dataset_table_row "$train_log" train 171 3951 3948; then
            append_failed "$group" "$seed" "train_split_assert" "1" \
                "training log does not report train 171/3951/3948"
            return 1
        fi
    fi
    if ! assert_dataset_table_row "$train_log" test 30 836 836; then
        append_failed "$group" "$seed" "test_split_assert" "1" \
            "training log does not report test 30/836/836"
        return 1
    fi
    log "[TRAIN_DONE] phase=$PHASE group=$group seed=$seed elapsed_seconds=$(($(date +%s) - run_start))"

    mapfile -t run_configs < <(
        find "$attempt_root" -type f -name configs.yaml -print
    )
    if ((${#run_configs[@]} != 1)); then
        append_failed "$group" "$seed" "locate_run" "1" \
            "expected one configs.yaml under $attempt_root, found ${#run_configs[@]}"
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
            "best.pth must not exist under the external protocol"
        return 1
    fi
    if [[ -e "$actual_run_dir/checkpoint_selection.tsv" ]]; then
        append_failed "$group" "$seed" "selection_assert" "1" \
            "checkpoint_selection.tsv must not exist under the external protocol"
        return 1
    fi
    mapfile -t checkpoint_files < <(
        find "$actual_run_dir" -type f \( -name '*.pth' -o -name '*.ckpt' \) -print
    )
    if ((${#checkpoint_files[@]} != 1)); then
        append_failed "$group" "$seed" "checkpoint_count_assert" "1" \
            "expected only last.pth, found ${#checkpoint_files[@]} checkpoints"
        return 1
    fi
    checkpoint_real="$(realpath -m "$checkpoint")"
    if [[ "$(realpath -m "${checkpoint_files[0]}")" != "$checkpoint_real" ]]; then
        append_failed "$group" "$seed" "checkpoint_count_assert" "1" \
            "the sole checkpoint is not last.pth"
        return 1
    fi
    if ! validate_model_only_checkpoint "$checkpoint"; then
        append_failed "$group" "$seed" "checkpoint_model_only_assert" "1" \
            "last.pth is not model-only"
        return 1
    fi

    if [[ "$PHASE" == "precondition" ]]; then
        split="val"
        expected_queries="$DEV2_DEV_CAPTIONS"
        expected_gallery="$DEV2_DEV_RECORDS"
        expected_log_line="[EVAL_SPLIT] split=val num_images=$expected_gallery num_captions=$expected_queries num_ids=30"
    else
        split="test"
        expected_queries=836
        expected_gallery=836
        expected_log_line="[EVAL_SPLIT] split=test num_images=836 num_captions=836 num_ids=30"
    fi

    local scenario_spec scenario mode strength kernel dseed
    local score_file score_tmp eval_log eval_rc metric_row parsed_r1
    for scenario_spec in "${CORE_SCENARIOS[@]}"; do
        IFS='|' read -r scenario mode strength kernel dseed <<< "$scenario_spec"
        score_file="$score_run_dir/${scenario}.pt"
        score_tmp="$score_run_dir/.${scenario}.${timestamp}.pt.tmp"
        eval_log="$LOGS_DIR/${run_key}_${timestamp}_${scenario}.log"
        if [[ -e "$score_tmp" ]]; then
            append_failed "$group" "$seed" "score_prepare:$scenario" "1" \
                "unexpected temporary score already exists: $score_tmp"
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
        if [[ "$PHASE" == "precondition" ]]; then
            eval_command+=(
                --dev_split_file "$DEV2_SPLIT"
                --eval_split val
            )
        fi
        if ! assert_phase_test_command "$PHASE" "$DEV2_SPLIT" \
            "${eval_command[@]}"; then
            append_failed "$group" "$seed" "eval_command_assert" "1" \
                "phase-specific evaluation command assertion failed"
            return 1
        fi
        log "[EVAL_START] phase=$PHASE group=$group seed=$seed scenario=$scenario mode=$mode"
        "${eval_command[@]}" 2>&1 | tee -a "$eval_log"
        eval_rc="${PIPESTATUS[0]}"
        if ((eval_rc != 0)); then
            append_failed "$group" "$seed" "eval:$scenario" "$eval_rc" \
                "evaluation process failed"
            return 1
        fi
        if ! grep -Fq "$expected_log_line" "$eval_log"; then
            append_failed "$group" "$seed" "eval_split_assert:$scenario" "1" \
                "evaluation log does not prove the expected $split split"
            return 1
        fi
        if [[ ! -f "$score_tmp" ]]; then
            append_failed "$group" "$seed" "score:$scenario" "1" \
                "evaluation did not write $score_tmp"
            return 1
        fi
        if ! metric_row="$(extract_metric_row \
            "$score_tmp" "$group" "$seed" "$scenario" "$mode" "$strength" \
            "$kernel" "$dseed" "$split" "$expected_queries" "$expected_gallery"
        )"; then
            append_failed "$group" "$seed" "metric_extract:$scenario" "1" \
                "score metadata validation or metric extraction failed"
            return 1
        fi
        printf '%s\n' "$metric_row" >> "$unit_metrics_tmp"
        if [[ "$scenario" == "clean" ]]; then
            IFS=$'\t' read -r _ _ _ _ _ _ _ _ parsed_r1 _ <<< "$metric_row"
            clean_r1="$parsed_r1"
        fi
        pending_score_sources+=("$score_tmp")
        pending_score_targets+=("$score_file")
    done

    local mech_run_dir="$MECH_DIR/$run_key"
    local mech_tmp="" mech_log mech_rc data_root data_root_rc
    local native_train_log diagnostic_train_log
    if arm_has_mraf "$PHASE" "$group"; then
        native_train_log="$actual_run_dir/train_log.txt"
        diagnostic_train_log="$actual_run_dir/train.log"
        if [[ ! -f "$native_train_log" ]]; then
            append_failed "$group" "$seed" "mech_train_log" "1" \
                "native train_log.txt is missing"
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
                "could not create diagnostic train.log symlink"
            return 1
        fi
        data_root="$(read_run_data_root "$run_config")"
        data_root_rc=$?
        if ((data_root_rc != 0)) || [[ ! -d "$data_root" ]]; then
            append_failed "$group" "$seed" "mech_data_root" "1" \
                "could not resolve an existing dataset root"
            return 1
        fi
        mech_tmp="$MECH_DIR/.${run_key}.${timestamp}.tmp"
        mech_log="$LOGS_DIR/${run_key}_${timestamp}_mech.log"
        if [[ -e "$mech_tmp" ]]; then
            append_failed "$group" "$seed" "mech_prepare" "1" \
                "unexpected mechanism temporary directory exists"
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
        log "[MECH_START] phase=$PHASE group=$group seed=$seed"
        "${mech_command[@]}" 2>&1 | tee -a "$mech_log"
        mech_rc="${PIPESTATUS[0]}"
        if ((mech_rc != 0)); then
            append_failed "$group" "$seed" "mech" "$mech_rc" \
                "MRAF mechanism diagnostics failed"
            return 1
        fi
        if [[ ! -f "$mech_tmp/diagnosis.json" || ! -f "$mech_tmp/summary.txt" ]]; then
            append_failed "$group" "$seed" "mech_output_assert" "1" \
                "mechanism diagnostics lacks diagnosis.json or summary.txt"
            return 1
        fi
    else
        log "[MECH_SKIP] phase=$PHASE group=$group seed=$seed reason=mraf_not_required"
    fi

    local fragment_rows committed_rows
    fragment_rows="$(awk 'END {print NR + 0}' "$unit_metrics_tmp")"
    if [[ "$fragment_rows" != "$expected_rows" ]]; then
        append_failed "$group" "$seed" "metrics_guard" "1" \
            "only $fragment_rows of $expected_rows metrics complete; checkpoint retained"
        return 1
    fi
    if ! commit_unit_metrics "$unit_metrics_tmp" "$group" "$seed" \
        "$split" "$expected_rows"; then
        append_failed "$group" "$seed" "metrics_commit" "1" \
            "atomic metrics update failed; checkpoint retained"
        return 1
    fi
    committed_rows="$(awk -F '\t' -v g="$group" -v s="$seed" -v p="$split" \
        'NR > 1 && $1 == g && $2 == s && $8 == p {count++} END {print count + 0}' \
        "$METRICS_TSV")"
    if [[ "$committed_rows" != "$expected_rows" ]]; then
        append_failed "$group" "$seed" "metrics_guard" "1" \
            "metrics.tsv has $committed_rows instead of $expected_rows rows; checkpoint retained"
        return 1
    fi
    log "[METRICS_GUARD] phase=$PHASE group=$group seed=$seed rows=$committed_rows"

    local pending_index
    for ((pending_index = 0; pending_index < ${#pending_score_sources[@]}; pending_index++)); do
        if ! mv -f -- "${pending_score_sources[$pending_index]}" \
            "${pending_score_targets[$pending_index]}"; then
            append_failed "$group" "$seed" "score_publish" "1" \
                "failed to publish ${pending_score_targets[$pending_index]}; checkpoint retained"
            return 1
        fi
    done
    if [[ "$(find "$score_run_dir" -maxdepth 1 -type f -name '*.pt' | wc -l)" != "11" ]]; then
        append_failed "$group" "$seed" "score_count_assert" "1" \
            "expected 11 retained score matrices; checkpoint retained"
        return 1
    fi

    if [[ "$PHASE" == "main" && "$group" == "A" ]]; then
        if [[ -z "$clean_r1" ]] || \
            ! update_external_warning "$group" "$seed" "$clean_r1"; then
            append_failed "$group" "$seed" "baseline_warning" "1" \
                "could not evaluate the external baseline warning; checkpoint retained"
            return 1
        fi
    fi

    if [[ -n "$mech_tmp" ]]; then
        if [[ -e "$mech_run_dir" ]]; then
            local stale_mech_dir="$MECH_DIR/.${run_key}.stale.${timestamp}"
            if [[ -e "$stale_mech_dir" ]] || \
                ! mv -- "$mech_run_dir" "$stale_mech_dir"; then
                append_failed "$group" "$seed" "mech_publish" "1" \
                    "could not preserve previous mechanism output; checkpoint retained"
                return 1
            fi
        fi
        if ! mv -- "$mech_tmp" "$mech_run_dir"; then
            append_failed "$group" "$seed" "mech_publish" "1" \
                "could not publish mechanism output; checkpoint retained"
            return 1
        fi
    fi

    actual_real="$(realpath -m "$actual_run_dir")"
    runs_root_real="$(realpath -m "$RUNS_DIR")"
    expected_run_root="$runs_root_real/$run_key"
    case "$actual_real/" in
        "$expected_run_root/"*) ;;
        *)
            append_failed "$group" "$seed" "cleanup_path_guard" "1" \
                "refusing cleanup outside $expected_run_root: $actual_real"
            return 1
            ;;
    esac
    if [[ "$committed_rows" != "11" ]]; then
        append_failed "$group" "$seed" "cleanup_metrics_guard" "1" \
            "refusing cleanup before 11 committed metrics; checkpoint retained"
        return 1
    fi
    if ! find "$actual_run_dir" -type f -name 'events.out.tfevents.*' -delete; then
        append_failed "$group" "$seed" "tfevents_cleanup" "1" \
            "failed to remove tfevents after metrics completion; checkpoint retained"
        return 1
    fi
    if ! rm -f -- "$unit_metrics_tmp"; then
        append_failed "$group" "$seed" "metrics_fragment_cleanup" "1" \
            "failed to remove metrics fragment; checkpoint retained"
        return 1
    fi

    retain_checkpoint=0
    if [[ "$PHASE" == "main" ]] && ((seed >= 201 && seed <= 203)); then
        retain_checkpoint=1
    fi
    checkpoint_status="retained"
    if ((retain_checkpoint)); then
        if [[ ! -f "$checkpoint" ]]; then
            append_failed "$group" "$seed" "checkpoint_retention_assert" "1" \
                "required seed 201-203 checkpoint disappeared"
            return 1
        fi
    else
        checkpoint_real="$(realpath -m "$checkpoint")"
        if [[ "$checkpoint_real" != "$actual_real/last.pth" ]]; then
            append_failed "$group" "$seed" "checkpoint_delete_path_guard" "1" \
                "refusing to delete unexpected checkpoint path: $checkpoint_real"
            return 1
        fi
        checkpoint_status="deleted_after_11_metrics"
    fi

    elapsed=$(($(date +%s) - run_start))
    disk_gb="$(remaining_gb 2>/dev/null || printf 'unknown')"
    if ! printf 'phase=%s\ngroup=%s\nseed=%s\ncompleted_at=%s\nmetrics_rows=11\nscores=11\ncheckpoint_status=%s\nrun_dir=%s\n' \
        "$PHASE" "$group" "$seed" "$(now)" "$checkpoint_status" \
        "$actual_run_dir" > "${mark}.tmp"; then
        append_failed "$group" "$seed" "mark_prepare" "1" \
            "could not prepare completion marker; checkpoint retained"
        return 1
    fi
    if ! printf '%s\t%s\t%s\t%s\tcompleted\t%s\t%s\t%s\n' \
        "$(now)" "$PHASE" "$group" "$seed" "$elapsed" "$disk_gb" \
        "$actual_run_dir" >> "$PROGRESS_TSV"; then
        append_failed "$group" "$seed" "progress" "1" \
            "progress append failed; checkpoint retained"
        return 1
    fi
    if ! mv -f -- "${mark}.tmp" "$mark"; then
        append_failed "$group" "$seed" "mark_commit" "1" \
            "could not publish completion marker; checkpoint retained"
        return 1
    fi

    if (( ! retain_checkpoint )); then
        # This is deliberately the final fallible operation in a successful
        # run. It is reachable only after 11 metrics were atomically committed,
        # all score/mechanism outputs passed, the run realpath was validated,
        # and completion bookkeeping succeeded. Every earlier failure therefore
        # returns with last.pth untouched.
        if ! rm -f -- "$checkpoint"; then
            rm -f -- "$mark" || true
            append_failed "$group" "$seed" "checkpoint_cleanup" "1" \
                "failed to delete non-retained last.pth; checkpoint retained"
            return 1
        fi
        if [[ -e "$checkpoint" ]]; then
            rm -f -- "$mark" || true
            append_failed "$group" "$seed" "checkpoint_cleanup" "1" \
                "non-retained last.pth remains after deletion; checkpoint retained"
            return 1
        fi
    fi

    log "[RUN_DONE] phase=$PHASE group=$group seed=$seed elapsed_seconds=$elapsed remaining_disk_gb=$disk_gb checkpoint=$checkpoint_status"
    return 0
}

main() {
    while (($#)); do
        case "$1" in
            --phase)
                (($# >= 2)) || {
                    printf 'error: --phase requires a value\n' >&2
                    return 2
                }
                PHASE="$2"
                shift 2
                ;;
            --base-config)
                (($# >= 2)) || {
                    printf 'error: --base-config requires a value\n' >&2
                    return 2
                }
                BASE_CONFIG="$2"
                shift 2
                ;;
            --out)
                (($# >= 2)) || {
                    printf 'error: --out requires a value\n' >&2
                    return 2
                }
                OUT="$2"
                shift 2
                ;;
            --frozen-hparams)
                (($# >= 2)) || {
                    printf 'error: --frozen-hparams requires a value\n' >&2
                    return 2
                }
                FROZEN_HPARAMS="$2"
                shift 2
                ;;
            --dev2-split)
                (($# >= 2)) || {
                    printf 'error: --dev2-split requires a value\n' >&2
                    return 2
                }
                DEV2_SPLIT="$2"
                shift 2
                ;;
            --python)
                (($# >= 2)) || {
                    printf 'error: --python requires a value\n' >&2
                    return 2
                }
                PYTHON_BIN="$2"
                shift 2
                ;;
            --validate-only)
                VALIDATE_ONLY=1
                shift
                ;;
            -h|--help)
                usage
                return 0
                ;;
            *)
                printf 'error: unknown argument: %s\n' "$1" >&2
                usage >&2
                return 2
                ;;
        esac
    done

    if [[ "$PHASE" != "precondition" && "$PHASE" != "main" ]]; then
        printf 'error: --phase must be precondition or main\n' >&2
        return 2
    fi
    if [[ -z "$OUT" ]]; then
        printf 'error: --out/OUT must not be empty\n' >&2
        return 2
    fi
    if [[ -z "$BASE_CONFIG" ]]; then
        printf 'error: --base-config or BASE_CONFIG is required\n' >&2
        return 2
    fi
    if [[ ! "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || ((MIN_FREE_GB < 1)); then
        printf 'error: MIN_FREE_GB must be a positive integer\n' >&2
        return 2
    fi
    if [[ "$AUTO_SHUTDOWN" != "0" && "$AUTO_SHUTDOWN" != "1" ]]; then
        printf 'error: AUTO_SHUTDOWN must be 0 or 1\n' >&2
        return 2
    fi

    OUT="$(realpath -m "$OUT")"
    BASE_CONFIG="$(realpath -m "$BASE_CONFIG")"
    FROZEN_HPARAMS="$(realpath -m "$FROZEN_HPARAMS")"
    DEV2_SPLIT="$(realpath -m "$DEV2_SPLIT")"
    if [[ ! -f "$BASE_CONFIG" ]]; then
        printf 'error: base config does not exist: %s\n' "$BASE_CONFIG" >&2
        return 2
    fi

    if ! mkdir -p \
        "$OUT/configs/$PHASE" "$OUT/runs" "$OUT/scores" "$OUT/mech" \
        "$OUT/marks" "$OUT/logs"; then
        printf 'error: could not create output directories under %s\n' \
            "$OUT" >&2
        return 2
    fi
    CONFIG_DIR="$OUT/configs/$PHASE"
    RUNS_DIR="$OUT/runs"
    SCORES_DIR="$OUT/scores"
    MECH_DIR="$OUT/mech"
    MARKS_DIR="$OUT/marks"
    LOGS_DIR="$OUT/logs"
    PROGRESS_TSV="$OUT/progress.tsv"
    METRICS_TSV="$OUT/metrics.tsv"
    FAILED_TSV="$OUT/failed.tsv"
    WARNINGS_TSV="$OUT/warnings.tsv"
    DRIVER_LOG="$LOGS_DIR/${PHASE}_driver.log"

    exec > >(tee -a "$DRIVER_LOG") 2>&1

    ensure_tsv_header "$PROGRESS_TSV" \
        'timestamp\tphase\tgroup\tseed\tstatus\telapsed_seconds\tremaining_gb\trun_dir'
    ensure_tsv_header "$METRICS_TSV" \
        'group\tseed\tscenario\tmode\tstrength\tkernel\tdseed\tsplit\tR1\tR5\tR10\tmAP\tmINP'
    ensure_tsv_header "$FAILED_TSV" \
        'timestamp\tphase\tgroup\tseed\tstage\texit_code\treason'
    ensure_tsv_header "$WARNINGS_TSV" \
        'timestamp\tgroup\tseed\tobserved_clean_R1\treference_clean_R1\tabsolute_difference\tthreshold\treason'

    local dev2_stats dev2_rc
    dev2_stats="$(validate_dev2_split "$DEV2_SPLIT")"
    dev2_rc=$?
    if ((dev2_rc != 0)); then
        append_failed "GLOBAL" "-" "dev2_split_assert" "$dev2_rc" \
            "dev2 split is missing or invalid: $DEV2_SPLIT"
        return "$dev2_rc"
    fi
    IFS=$'\t' read -r \
        DEV2_DEV_RECORDS DEV2_DEV_CAPTIONS \
        DEV2_TRAIN_RECORDS DEV2_TRAIN_CAPTIONS \
        DEV2_ANNO_MD5 DEV2_ARTIFACT_MD5 <<< "$dev2_stats"
    log "[DEV2_ASSERT] n_dev=30 n_train_reduced=141 dev_records=$DEV2_DEV_RECORDS dev_effective_captions=$DEV2_DEV_CAPTIONS train2_records=$DEV2_TRAIN_RECORDS train2_effective_captions=$DEV2_TRAIN_CAPTIONS anno_md5=$DEV2_ANNO_MD5 artifact_md5=$DEV2_ARTIFACT_MD5"

    if [[ "$PHASE" == "main" ]]; then
        assert_external_prediction_preregistered \
            "$PROJECT_ROOT" "$PREDICTION_REL"
        local prediction_rc=$?
        if ((prediction_rc != 0)); then
            local prediction_reason
            case "$prediction_rc" in
                21) prediction_reason="EXTERNAL_PREDICTION.md does not exist" ;;
                22) prediction_reason="prediction section is empty" ;;
                23) prediction_reason="prediction file has no commit" ;;
                24) prediction_reason="prediction file has uncommitted changes" ;;
                *) prediction_reason="unknown prediction gate failure" ;;
            esac
            append_failed "GLOBAL" "-" "prediction_gate" \
                "$prediction_rc" "$prediction_reason"
            return "$prediction_rc"
        fi
    fi

    validate_frozen_hparams "$FROZEN_HPARAMS"
    local frozen_rc=$?
    if ((frozen_rc != 0)); then
        append_failed "GLOBAL" "-" "frozen_assert" "$frozen_rc" \
            "frozen hyperparameter safety assertion failed"
        return "$frozen_rc"
    fi

    if [[ "$PHASE" == "precondition" ]]; then
        ARM_NAMES=(A)
        SEEDS=(1 2 3)
        EXPECTED_SPLIT="val"
    else
        ARM_NAMES=(A D E U)
        SEEDS=(201 202 203 204 205 206 207 208 209 210)
        EXPECTED_SPLIT="test"
    fi
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
    if ((${#CORE_SCENARIOS[@]} != 11)); then
        append_failed "GLOBAL" "-" "scenario_assert" "2" \
            "external protocol requires exactly 11 core scenarios"
        return 2
    fi
    if [[ "$PHASE" == "precondition" ]] && \
        ((${#ARM_NAMES[@]} != 1 || ${#SEEDS[@]} != 3)); then
        append_failed "GLOBAL" "-" "plan_assert" "2" \
            "precondition plan must be A x seeds 1,2,3"
        return 2
    fi
    if [[ "$PHASE" == "main" ]] && \
        ((${#ARM_NAMES[@]} != 4 || ${#SEEDS[@]} != 10)); then
        append_failed "GLOBAL" "-" "plan_assert" "2" \
            "main plan must be A,D,E,U x seeds 201..210"
        return 2
    fi

    cd "$PROJECT_ROOT" || return 2
    log "Generating nine frozen formal configs from $BASE_CONFIG"
    "$PYTHON_BIN" -m tools.protocol.gen_configs \
        --base-config "$BASE_CONFIG" \
        --frozen-hparams "$FROZEN_HPARAMS" \
        --out-dir "$CONFIG_DIR"
    local config_rc=$?
    if ((config_rc != 0)); then
        append_failed "GLOBAL" "-" "config_generation" "$config_rc" \
            "formal config generation failed"
        return "$config_rc"
    fi
    validate_generated_configs \
        "$FROZEN_HPARAMS" "$CONFIG_DIR" "$PHASE"
    local config_assert_rc=$?
    if ((config_assert_rc != 0)); then
        append_failed "GLOBAL" "-" "config_assert" "$config_assert_rc" \
            "generated external config assertion failed"
        return "$config_assert_rc"
    fi

    local representative_seed="${SEEDS[0]}"
    local representative_arm="${ARM_NAMES[0]}"
    local representative_config="$CONFIG_DIR/${representative_arm}.yaml"
    local -a train_command=(
        env "TRAIN_SEED=$representative_seed"
        "$PYTHON_BIN" -u train.py
        --config_file "$representative_config"
        --name "${PHASE}_${representative_arm}_seed$representative_seed"
        --output_dir "$OUT/.validate_only"
        --MLM
    )
    local -a eval_command=(
        "$PYTHON_BIN" -u test.py
        --config_file "$representative_config"
        --test_ckpt_file "$OUT/.validate_only/last.pth"
        --save_eval_scores
        --eval_scores_file "$OUT/.validate_only/score.pt"
        --eval_degrade_mode none
        --eval_degrade_strength 1.0
        --eval_degrade_blur_kernel 7
        --eval_degrade_seed 2026
    )
    if [[ "$PHASE" == "precondition" ]]; then
        train_command+=(--dev_split_file "$DEV2_SPLIT" --val_dataset val)
        eval_command+=(--dev_split_file "$DEV2_SPLIT" --eval_split val)
    fi
    if ! assert_phase_train_command \
        "$PHASE" "$DEV2_SPLIT" "${train_command[@]}"; then
        append_failed "GLOBAL" "-" "train_command_assert" "1" \
            "representative training command failed safety assertions"
        return 1
    fi
    if ! assert_phase_test_command \
        "$PHASE" "$DEV2_SPLIT" "${eval_command[@]}"; then
        append_failed "GLOBAL" "-" "eval_command_assert" "1" \
            "representative evaluation command failed safety assertions"
        return 1
    fi
    log "[COMMAND_ASSERT] phase=$PHASE train=pass test=pass"

    local planned_runs=$((${#ARM_NAMES[@]} * ${#SEEDS[@]}))
    local arms_csv
    arms_csv="$(IFS=,; printf '%s' "${ARM_NAMES[*]}")"
    if ((VALIDATE_ONLY)); then
        log "[VALIDATE_ONLY] phase=$PHASE arms=$arms_csv planned_runs=$planned_runs"
        local plan_arm plan_seed
        for plan_arm in "${ARM_NAMES[@]}"; do
            for plan_seed in "${SEEDS[@]}"; do
                printf '[RUN_PLAN] phase=%s %s_seed%s split=%s\n' \
                    "$PHASE" "$plan_arm" "$plan_seed" "$EXPECTED_SPLIT"
            done
        done
        log "[VALIDATE_ONLY_DONE] no training, evaluation, diagnostics, cleanup, or shutdown executed"
        return 0
    fi

    trap on_signal INT TERM
    GLOBAL_START="$(date +%s)"
    local group seed first_seed_failed
    for group in "${ARM_NAMES[@]}"; do
        first_seed_failed=0
        for seed in "${SEEDS[@]}"; do
            if ((first_seed_failed)); then
                append_failed "$group" "$seed" "skipped_after_first_seed" "1" \
                    "first seed failed for this arm; later seeds skipped"
                continue
            fi
            if ! run_one "$group" "$seed"; then
                if [[ "$seed" == "${SEEDS[0]}" ]]; then
                    first_seed_failed=1
                fi
            fi
        done
    done

    local success_count=0 total_runs failed_count total_elapsed disk_gb
    for group in "${ARM_NAMES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            [[ -f "$MARKS_DIR/${PHASE}_${group}_seed${seed}.done" ]] \
                && success_count=$((success_count + 1))
        done
    done
    total_runs=$((${#ARM_NAMES[@]} * ${#SEEDS[@]}))
    failed_count=$((total_runs - success_count))
    total_elapsed=$(($(date +%s) - GLOBAL_START))
    disk_gb="$(remaining_gb 2>/dev/null || printf 'unknown')"
    log "[GRID_SUMMARY] phase=$PHASE total_runs=$total_runs successful_runs=$success_count failed_runs=$failed_count total_elapsed_seconds=$total_elapsed remaining_disk_gb=$disk_gb"

    if [[ "$AUTO_SHUTDOWN" != "1" ]]; then
        log "[SHUTDOWN] AUTO_SHUTDOWN=$AUTO_SHUTDOWN; automatic shutdown disabled"
        return 0
    fi
    log "[SHUTDOWN] automatic shutdown scheduled after a 60-second grace period"
    local remaining
    for remaining in $(seq 60 -1 1); do
        if [[ -e "$OUT/STOP" ]]; then
            log "[SHUTDOWN] cancelled because $OUT/STOP exists"
            return 0
        fi
        if ((remaining == 60 || remaining % 10 == 0 || remaining <= 5)); then
            log "[SHUTDOWN] countdown=${remaining}s; create $OUT/STOP to cancel"
        fi
        sleep 1
    done
    if [[ -e "$OUT/STOP" ]]; then
        log "[SHUTDOWN] cancelled because $OUT/STOP exists"
        return 0
    fi
    log "[SHUTDOWN] grace period complete; executing shutdown -h now"
    shutdown -h now
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
