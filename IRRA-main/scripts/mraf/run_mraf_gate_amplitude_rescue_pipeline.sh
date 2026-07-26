#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run the MRAF gate-amplitude rescue screen, calibration, confirmation, and formal gate.

Usage:
  bash scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh \
    --data-root <TextReID-root> \
    --output-root <new-parent> \
    --base-config <historical-configs.yaml> \
    --reference-screen-root <old-S0-S5-screen-root> \
    --reference-run-root <old-seed1-root> \
    --device 0 --seed 1 --auto-continue

Resume:
  bash scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh \
    --resume-run-root <existing-run> --auto-continue

Options:
  --data-root PATH             TextReID dataset root
  --output-root PATH           parent of a new non-overwriting run
  --base-config PATH           historical formal MDRC+MRAF configs.yaml
  --reference-screen-root PATH old S0-S5 result root; read only
  --reference-run-root PATH    old seed1 formal result root; read only
  --device INDEX               CUDA index or cuda:INDEX (default: 0)
  --seed N                     non-negative seed (default: 1)
  --auto-continue              allow confirmation and gated 60-epoch run
  --shutdown-on-success        shutdown only after complete formal artifacts
  --resume-run-root PATH       resume an incomplete run without overwriting it
  --dry-run                    print configuration/commands; write and run nothing
  --mock                       CPU/filesystem state-machine fixture; never train/eval
  -h, --help                   show this help

The script never uses screen. Use external nohup for unattended execution.
Calibration is read-only: it never trains or changes configs.yaml/checkpoints.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 2
}

data_root=""
output_root=""
base_config="${MRAF_BASE_CONFIG:-}"
reference_screen_root="${MRAF_REFERENCE_SCREEN_ROOT:-}"
reference_run_root="${MRAF_REFERENCE_RUN_ROOT:-}"
resume_run_root=""
device=0
seed=1
auto_continue=false
shutdown_on_success=false
dry_run=false
mock=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) [[ $# -ge 2 ]] || die "--data-root requires a value"; data_root=$2; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || die "--output-root requires a value"; output_root=$2; shift 2 ;;
    --base-config) [[ $# -ge 2 ]] || die "--base-config requires a value"; base_config=$2; shift 2 ;;
    --reference-screen-root) [[ $# -ge 2 ]] || die "--reference-screen-root requires a value"; reference_screen_root=$2; shift 2 ;;
    --reference-run-root) [[ $# -ge 2 ]] || die "--reference-run-root requires a value"; reference_run_root=$2; shift 2 ;;
    --resume-run-root) [[ $# -ge 2 ]] || die "--resume-run-root requires a value"; resume_run_root=$2; shift 2 ;;
    --device) [[ $# -ge 2 ]] || die "--device requires a value"; device=$2; shift 2 ;;
    --seed) [[ $# -ge 2 ]] || die "--seed requires a value"; seed=$2; shift 2 ;;
    --auto-continue) auto_continue=true; shift ;;
    --shutdown-on-success) shutdown_on_success=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --mock) mock=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ "$seed" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"
if [[ "$device" =~ ^cuda:([0-9]+)$ ]]; then
  device_index=${BASH_REMATCH[1]}
elif [[ "$device" =~ ^[0-9]+$ ]]; then
  device_index=$device
else
  die "--device must be a CUDA index or cuda:INDEX"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"
git_root="$(cd "${repo_dir}/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python}"
shutdown_command="${MRAF_SHUTDOWN_COMMAND:-shutdown}"
timestamp="$(date +%Y%m%d_%H%M%S)"

git_dir_override=""
if ! git -C "$git_root" rev-parse --git-dir >/dev/null 2>&1; then
  git_pointer_file="${git_root}/.git"
  if [[ -f "$git_pointer_file" ]] && command -v wslpath >/dev/null 2>&1; then
    git_dir_pointer="$(sed -n 's/^gitdir: //p' "$git_pointer_file")"
    if [[ "$git_dir_pointer" =~ ^[A-Za-z]:[/\\] ]]; then
      git_dir_override="$(wslpath -u "$git_dir_pointer")"
    fi
  fi
fi

run_git() {
  if [[ -n "$git_dir_override" ]]; then
    GIT_DIR="$git_dir_override" GIT_WORK_TREE="$git_root" git "$@"
  else
    git -C "$git_root" "$@"
  fi
}

git_head="$(run_git rev-parse HEAD)"
git_branch="$(run_git branch --show-current)"
git_short="${git_head:0:8}"

if [[ "$mock" == "true" && -z "$output_root" && -z "$resume_run_root" ]]; then
  if [[ "$dry_run" == "true" ]]; then
    output_root="/tmp/mraf_gate_amplitude_mock_DRY_RUN"
  else
    output_root="$(mktemp -d -t mraf_gate_amplitude_mock_XXXXXX)"
  fi
fi
if [[ -n "$resume_run_root" ]]; then
  [[ -d "$resume_run_root" ]] || die "--resume-run-root is not a directory"
  run_dir="$(cd "$resume_run_root" && pwd)"
  resume_mode=true
else
  [[ -n "$output_root" ]] || die "--output-root must not be empty"
  run_dir="${output_root%/}/${timestamp}_${git_short}_mraf_gate_amplitude_rescue_seed${seed}"
  resume_mode=false
fi
[[ "$resume_mode" != "true" || "$mock" != "true" ]] || die \
  "--mock cannot be combined with --resume-run-root"

status_file="${run_dir}/status.txt"
current_stage="preflight"
pipeline_completed=false
formal_completed=false

write_status() {
  local status=$1
  local exit_code=$2
  local failed_stage=${3:-NONE}
  {
    echo "STATUS=$status"
    echo "FAILED_STAGE=$failed_stage"
    echo "EXIT_CODE=$exit_code"
    echo "UPDATED_AT=$(date -Iseconds)"
  } >"$status_file"
}

on_error() {
  local exit_code=$?
  if [[ -d "$run_dir" && "$pipeline_completed" != "true" ]]; then
    write_status "FAILED" "$exit_code" "$current_stage"
  fi
  exit "$exit_code"
}
trap on_error ERR

print_matrix() {
  cat <<'EOF'
candidate fp32 rel_lambda base_grad_scale scorer_lr_multiplier logit_scale
A10 true 0.25 0.10 10  1
A50 true 0.25 0.10 50  1
A100 true 0.25 0.10 100 1
B50 true 1.00 1.00 50  1
EOF
}

candidate_values() {
  case "$1" in
    A10) printf '%s %s %s %s\n' 0.25 0.10 10 1 ;;
    A50) printf '%s %s %s %s\n' 0.25 0.10 50 1 ;;
    A100) printf '%s %s %s %s\n' 0.25 0.10 100 1 ;;
    B50) printf '%s %s %s %s\n' 1.00 1.00 50 1 ;;
    *) die "Unknown candidate: $1" ;;
  esac
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

manifest_value() {
  local key=$1
  sed -n "s/^${key}=//p" "${run_dir}/experiment_manifest.txt" | head -n1
}

if [[ "$resume_mode" == "true" ]]; then
  [[ -f "${run_dir}/experiment_manifest.txt" ]] || die \
    "Resume root lacks experiment_manifest.txt"
  [[ "$(manifest_value PIPELINE_ID)" == "mraf_gate_amplitude_rescue_v3" ]] || die \
    "Resume pipeline ID differs"
  [[ "$(manifest_value seed)" == "$seed" ]] || die "Resume seed differs"
  [[ -n "$data_root" ]] || data_root="$(manifest_value data_root)"
  [[ -n "$base_config" ]] || base_config="$(manifest_value base_config)"
  [[ -n "$reference_screen_root" ]] || reference_screen_root="$(manifest_value reference_screen_root)"
  [[ -n "$reference_run_root" ]] || reference_run_root="$(manifest_value reference_run_root)"
fi

if [[ "$mock" != "true" ]]; then
  [[ -d "$data_root" ]] || die "--data-root is not a directory: $data_root"
  [[ -d "$output_root" || "$resume_mode" == "true" ]] || die \
    "--output-root parent does not exist: $output_root"
  [[ -f "$base_config" ]] || die "--base-config is not a file: $base_config"
  [[ -d "$reference_screen_root" ]] || die \
    "--reference-screen-root is not a directory"
  [[ -d "$reference_run_root" ]] || die \
    "--reference-run-root is not a directory"
  data_root="$(cd "$data_root" && pwd)"
  base_config="$(cd "$(dirname "$base_config")" && pwd)/$(basename "$base_config")"
  reference_screen_root="$(cd "$reference_screen_root" && pwd)"
  reference_run_root="$(cd "$reference_run_root" && pwd)"
  "$python_bin" -B tools/revision/validate_mraf_rescue_config.py \
    --config "$base_config" --require-gate-rescue-protocol >/dev/null
  "$python_bin" -B - "$base_config" <<'PY'
import sys
from utils.config import load_config_mapping

path = sys.argv[1]
config = load_config_mapping(path)
for key in (
    "resume",
    "save_all_epoch_ckpt",
    "save_metric_best_ckpt",
    "save_full_eval_state_ckpt",
):
    if bool(config.get(key, False)):
        raise SystemExit(
            f"Historical base config must keep {key}=false for this pipeline: {path}"
        )
for key in ("resume_ckpt_file", "load_ckpt_file"):
    if str(config.get(key, "") or "").strip():
        raise SystemExit(
            f"Historical base config must keep {key} empty so every candidate "
            f"starts from the same pretrained initialization: {path}"
        )
PY
  if [[ "$resume_mode" == "true" ]]; then
    [[ "$(manifest_value git_head)" == "$git_head" ]] || die \
      "Resume Git HEAD differs from the original run"
    [[ "$(manifest_value data_root)" == "$data_root" ]] || die \
      "Resume data root differs from the original run"
    [[ "$(manifest_value base_config)" == "$base_config" ]] || die \
      "Resume base config differs from the original run"
    [[ "$(manifest_value reference_screen_root)" == "$reference_screen_root" ]] || die \
      "Resume reference screen root differs from the original run"
    [[ "$(manifest_value reference_run_root)" == "$reference_run_root" ]] || die \
      "Resume reference run root differs from the original run"
  fi
fi

if [[ "$dry_run" == "true" ]]; then
  echo "dry_run=true"
  echo "mock=$mock"
  echo "git_branch=$git_branch"
  echo "git_head=$git_head"
  echo "data_root=${data_root:-MOCK}"
  echo "output_root=$output_root"
  echo "base_config=${base_config:-MOCK}"
  echo "reference_screen_root=${reference_screen_root:-MOCK}"
  echo "reference_run_root=${reference_run_root:-MOCK}"
  echo "resume_run_root=${resume_run_root:-NONE}"
  echo "run_dir=$run_dir"
  echo "device=$device_index"
  echo "seed=$seed"
  echo "auto_continue=$auto_continue"
  echo "shutdown_on_success=$shutdown_on_success"
  print_matrix
  echo "stage1_epochs=10 confirmation_epochs=12 formal_epochs=60"
  echo "calibration_scales=2,5,10,20,50"
  echo "No directory, train.py, test.py, GPU, diagnosis, or shutdown was used."
  exit 0
fi

if [[ "$resume_mode" == "false" ]]; then
  [[ ! -e "$run_dir" ]] || die "Refusing to overwrite run directory: $run_dir"
  mkdir -p "$run_dir"
  write_status "RUNNING" 0 "NONE"
  {
    echo "PIPELINE_ID=mraf_gate_amplitude_rescue_v3"
    echo "git_branch=$git_branch"
    echo "git_head=$git_head"
    echo "seed=$seed"
    echo "degrade_seed=2026"
    echo "device=$device_index"
    echo "data_root=${data_root:-MOCK}"
    echo "base_config=${base_config:-MOCK}"
    echo "reference_screen_root=${reference_screen_root:-MOCK}"
    echo "reference_run_root=${reference_run_root:-MOCK}"
    echo "auto_continue=$auto_continue"
    echo "shutdown_on_success=$shutdown_on_success"
    echo "mock=$mock"
  } >"${run_dir}/experiment_manifest.txt"
else
  write_status "RUNNING" 0 "NONE"
  {
    echo "resume_at=$(date -Iseconds)"
    echo "resume_git_head=$git_head"
  } >>"${run_dir}/experiment_manifest.txt"
fi

if [[ "$mock" == "true" ]]; then
  current_stage="mock"
  mock_reports="${run_dir}/reports"
  "$python_bin" -B tools/revision/summarize_mraf_gate_amplitude_rescue.py \
    --mode mock --output-dir "$mock_reports"
  cp "${run_dir}/experiment_manifest.txt" "${mock_reports}/experiment_manifest.txt"
  cp "${mock_reports}"/* "${run_dir}/"
  archive="${run_dir}/mraf_gate_amplitude_rescue_reports.tar.gz"
  tar -czf "$archive" -C "$run_dir" reports experiment_manifest.txt
  [[ -s "$archive" ]] || die "Mock report archive is empty"
  cp "${mock_reports}/FINAL_REPORT.txt" "${run_dir}/FINAL_REPORT.txt"
  write_status "MOCK_COMPLETED" 0 "NONE"
  pipeline_completed=true
  echo "STATUS=MOCK_COMPLETED"
  echo "mock_output_dir=$run_dir"
  echo "shutdown_executed=false"
  exit 0
fi

current_stage="real_preflight"
[[ -z "$(run_git status --short)" ]] || die \
  "Real pipeline requires a clean Git worktree"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
env "CUDA_VISIBLE_DEVICES=$device_index" "$python_bin" -B -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' \
  >"${run_dir}/gpu.txt"
if pgrep -af 'train.py.*mraf_gate_amplitude' | grep -v "$$" >/dev/null 2>&1; then
  die "A matching gate-amplitude training process is already active"
fi

check_space_gb() {
  local path=$1
  local required=$2
  local available_kb
  available_kb="$(df -Pk "$path" | awk 'NR==2 {print $4}')"
  [[ "$available_kb" =~ ^[0-9]+$ ]] || die "Unable to read free disk space"
  local available_gb=$((available_kb / 1024 / 1024))
  {
    echo "stage=$current_stage"
    echo "required_gb=$required"
    echo "available_gb=$available_gb"
    echo "checked_at=$(date -Iseconds)"
  } >>"${run_dir}/disk_usage.log"
  (( available_gb >= required )) || die \
    "Insufficient disk space: ${available_gb}GB available, ${required}GB required"
}

check_space_gb "$run_dir" 18

common_train_args=(
  --config_file "$base_config"
  --dataset_name RGBT-PEDES --root_dir "$data_root"
  --img_aug --MLM --loss_names sdm+mlm+id
  --batch_size 64 --test_batch_size 512 --num_workers 8
  --warmup_epochs 5 --lr 1e-5 --eval_use_fused_mu
  --log_period 100 --eval_period 1 --degrade_aug_mode none
  --use_rgbt_add_fusion --mraf_enabled --mraf_gate_fp32
  --mraf_hidden_dim 128 --mraf_dropout 0.1
  --mraf_gate_temperature 1.0 --mraf_target_temperature 0.1
  --mraf_balance_lambda 0.0
  --mdrc_enable --mdrc_prob 0.5 --mdrc_lambda 0.05
  --mdrc_loss_scale 100 --mdrc_warmup_epoch 3
  --save_last_checkpoint
)

run_logged() {
  local log_file=$1
  local pid_file=$2
  shift 2
  "$@" >"$log_file" 2>&1 &
  local child_pid=$!
  printf '%s\n' "$child_pid" >"$pid_file"
  wait "$child_pid"
}

resolve_unique_artifact() {
  local root=$1
  local name=$2
  mapfile -t matches < <(find "$root" -type f -name "$name" -print)
  [[ ${#matches[@]} -eq 1 ]] || {
    echo "Expected one $name under $root; found ${#matches[@]}" >&2
    return 1
  }
  printf '%s\n' "${matches[0]}"
}

run_training() {
  local train_dir=$1
  local label=$2
  local epochs=$3
  local rel_warmup=$4
  local rel_lambda=$5
  local grad_scale=$6
  local lr_multiplier=$7
  local logit_scale=$8
  [[ ! -e "$train_dir" ]] || return 90
  mkdir -p "${train_dir}/train_output"
  local command=(
    env "CUDA_VISIBLE_DEVICES=$device_index" "TRAIN_SEED=$seed"
    "$python_bin" -u train.py
    --name "$label" --output_dir "${train_dir}/train_output"
    --num_epoch "$epochs" --mraf_rel_warmup_epoch "$rel_warmup"
    --mraf_rel_lambda "$rel_lambda"
    --mraf_gate_base_grad_scale "$grad_scale"
    --mraf_scorer_lr_multiplier "$lr_multiplier"
    --mraf_gate_logit_scale "$logit_scale"
    "${common_train_args[@]}"
  )
  print_command "${command[@]}" >"${train_dir}/training_command.txt"
  run_logged "${train_dir}/train.log" "${train_dir}/train.pid" "${command[@]}"
  if grep -Eiq 'CUDA out of memory|OutOfMemoryError|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
    "${train_dir}/train.log"; then
    return 86
  fi
  resolve_unique_artifact "${train_dir}/train_output" configs.yaml >/dev/null
  resolve_unique_artifact "${train_dir}/train_output" best.pth >/dev/null
  resolve_unique_artifact "${train_dir}/train_output" last.pth >/dev/null
}

active_training_dir() {
  local container=$1
  if [[ -s "${container}/active_attempt.txt" ]]; then
    local relative
    relative="$(<"${container}/active_attempt.txt")"
    [[ "$relative" != /* && "$relative" != *".."* ]] || return 1
    printf '%s\n' "${container}/${relative}"
  else
    printf '%s\n' "$container"
  fi
}

training_complete() {
  local path=$1
  [[ -s "${path}/train.log" ]] || return 1
  resolve_unique_artifact "${path}/train_output" configs.yaml >/dev/null 2>&1 \
    && resolve_unique_artifact "${path}/train_output" best.pth >/dev/null 2>&1 \
    && resolve_unique_artifact "${path}/train_output" last.pth >/dev/null 2>&1 \
    && resolve_unique_artifact "${path}/train_output" mraf_epoch_metrics.tsv >/dev/null 2>&1 \
    && resolve_unique_artifact "${path}/train_output" checkpoint_selection.tsv >/dev/null 2>&1
}

ensure_training() {
  local container=$1
  shift
  local active
  active="$(active_training_dir "$container")"
  if training_complete "$active"; then
    effective_train_dir=$active
    return 0
  fi
  if [[ -e "$container" ]]; then
    active="${container}/resume_attempt_${timestamp}"
    mkdir -p "$container"
    run_training "$active" "$@"
    printf '%s\n' "${active#${container}/}" >"${container}/active_attempt.txt"
  else
    active=$container
    run_training "$active" "$@"
  fi
  effective_train_dir=$active
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
    both_noise_0.05) printf '%s %s %s\n' both_noise 0.05 7 ;;
    both_blur_k7) printf '%s %s %s\n' both_blur 1.0 7 ;;
    *) die "Unknown scenario: $1" ;;
  esac
}

prepare_unit() {
  local unit_dir=$1
  local train_dir=$2
  local candidate=$3
  local checkpoint_kind=$4
  local lr_multiplier=$5
  local logit_scale=$6
  local rel_lambda=$7
  local grad_scale=$8
  [[ ! -e "$unit_dir" ]] || return 90
  mkdir -p "$unit_dir"
  local config checkpoint epoch_metrics checkpoint_selection
  config="$(resolve_unique_artifact "${train_dir}/train_output" configs.yaml)"
  checkpoint="$(resolve_unique_artifact "${train_dir}/train_output" "${checkpoint_kind}.pth")"
  epoch_metrics="$(resolve_unique_artifact "${train_dir}/train_output" mraf_epoch_metrics.tsv)"
  checkpoint_selection="$(resolve_unique_artifact "${train_dir}/train_output" checkpoint_selection.tsv)"
  ln -s "$config" "${unit_dir}/configs.yaml"
  ln -s "$checkpoint" "${unit_dir}/checkpoint.pth"
  ln -s "${train_dir}/train.log" "${unit_dir}/train.log"
  ln -s "$epoch_metrics" "${unit_dir}/mraf_epoch_metrics.tsv"
  ln -s "$checkpoint_selection" "${unit_dir}/checkpoint_selection.tsv"
  cat >"${unit_dir}/unit.json" <<EOF
{
  "candidate": "$candidate",
  "checkpoint": "$checkpoint_kind",
  "mraf_scorer_lr_multiplier": $lr_multiplier,
  "mraf_gate_logit_scale": $logit_scale,
  "mraf_rel_lambda": $rel_lambda,
  "mraf_gate_base_grad_scale": $grad_scale,
  "has_nan": false,
  "has_inf": false,
  "has_oom": false
}
EOF
}

run_evaluations() {
  local unit_dir=$1
  local scenarios=$2
  local logit_scale=$3
  local scenario mode strength kernel eval_dir eval_log eval_pid
  for scenario in $scenarios; do
    eval_dir="${unit_dir}/eval/${scenario}"
    [[ -s "${eval_dir}/scores.pt" ]] && continue
    mkdir -p "$eval_dir"
    read -r mode strength kernel < <(scenario_values "$scenario")
    eval_log="${eval_dir}/eval.log"
    eval_pid="${eval_dir}/pid"
    if [[ -e "$eval_log" || -e "$eval_pid" ]]; then
      eval_log="${eval_dir}/eval_resume_${timestamp}.log"
      eval_pid="${eval_dir}/pid_resume_${timestamp}"
    fi
    run_logged "$eval_log" "$eval_pid" \
      env "CUDA_VISIBLE_DEVICES=$device_index" "$python_bin" -B test.py \
      --config_file "${unit_dir}/configs.yaml" \
      --test_ckpt_file "${unit_dir}/checkpoint.pth" \
      --mraf_gate_logit_scale "$logit_scale" \
      --eval_degrade_mode "$mode" \
      --eval_degrade_strength "$strength" \
      --eval_degrade_blur_kernel "$kernel" \
      --eval_degrade_seed 2026 \
      --save_eval_scores --eval_scores_save_raw \
      --eval_scores_file "${eval_dir}/scores.pt"
  done
}

run_diagnosis() {
  local unit_dir=$1
  local label=$2
  local max_batches=$3
  local logit_scale=$4
  local diagnosis_dir="${unit_dir}/diagnostics"
  local diagnosis_log="${unit_dir}/diagnosis.log"
  local diagnosis_pid="${unit_dir}/diagnosis.pid"
  if [[ -s "${diagnosis_dir}/diagnosis.json" ]]; then
    return 0
  fi
  if [[ -e "$diagnosis_dir" || -e "$diagnosis_log" || -e "$diagnosis_pid" ]]; then
    diagnosis_dir="${unit_dir}/diagnostics_resume_${timestamp}"
    diagnosis_log="${unit_dir}/diagnosis_resume_${timestamp}.log"
    diagnosis_pid="${unit_dir}/diagnosis_resume_${timestamp}.pid"
  fi
  run_logged "$diagnosis_log" "$diagnosis_pid" \
    env "CUDA_VISIBLE_DEVICES=$device_index" "$python_bin" -B \
    tools/revision/diagnose_mraf_mechanism.py \
    --config-file "${unit_dir}/configs.yaml" \
    --checkpoint "${unit_dir}/checkpoint.pth" \
    --case-label "$label" --data-root "$data_root" --device cuda:0 \
    --degrade-seed 2026 --max-batches "$max_batches" \
    --gradient-batches 3 --gradient-microbatch-size 4 \
    --mraf-gate-logit-scale "$logit_scale" \
    --output-dir "$diagnosis_dir"
  if [[ "$diagnosis_dir" != "${unit_dir}/diagnostics" ]]; then
    printf '%s\n' "$(basename "$diagnosis_dir")" \
      >"${unit_dir}/active_diagnostics.txt"
  fi
}

evaluate_unit() {
  local unit_dir=$1
  local label=$2
  local scenarios=$3
  local max_batches=$4
  local logit_scale=$5
  run_evaluations "$unit_dir" "$scenarios" "$logit_scale"
  run_diagnosis "$unit_dir" "$label" "$max_batches" "$logit_scale"
}

screen_scenarios="clean rgb_missing ir_missing rgb_noise_0.10 ir_noise_0.10 rgb_blur_k7 ir_blur_k7 both_blur_k7"
formal_scenarios="clean rgb_missing ir_missing rgb_noise_0.05 rgb_noise_0.10 ir_noise_0.05 ir_noise_0.10 rgb_blur_k7 ir_blur_k7 both_noise_0.05 both_blur_k7"
stage1_train_root="${run_dir}/stage1/train"
stage1_unit_root="${run_dir}/stage1/units"
mkdir -p "$stage1_train_root" "$stage1_unit_root"
candidate_status_file="${run_dir}/candidate_status.tsv"
if [[ ! -e "$candidate_status_file" ]]; then
  printf 'candidate\tcheckpoint\tstatus\texit_code\tfailed_stage\n' \
    >"$candidate_status_file"
fi

current_stage="amplitude_screen"
for candidate in A10 A50 A100 B50; do
  read -r rel_lambda grad_scale lr_multiplier logit_scale < <(
    candidate_values "$candidate"
  )
  train_container="${stage1_train_root}/${candidate}"
  if ensure_training "$train_container" \
      "mraf_gate_amplitude_${candidate}_seed${seed}" \
      10 1 "$rel_lambda" "$grad_scale" "$lr_multiplier" "$logit_scale"; then
    :
  else
    candidate_exit=$?
    printf '%s\tALL\tFAILED\t%s\ttraining\n' \
      "$candidate" "$candidate_exit" >>"$candidate_status_file"
    continue
  fi
  train_dir=$effective_train_dir
  for checkpoint_kind in best last; do
    unit_dir="${stage1_unit_root}/${candidate}_${checkpoint_kind}"
    if {
      if [[ ! -d "$unit_dir" ]]; then
        prepare_unit "$unit_dir" "$train_dir" "$candidate" "$checkpoint_kind" \
          "$lr_multiplier" "$logit_scale" "$rel_lambda" "$grad_scale"
      fi
      evaluate_unit "$unit_dir" "$candidate" "$screen_scenarios" 4 "$logit_scale"
    }; then
      printf '%s\t%s\tCOMPLETED\t0\tNONE\n' \
        "$candidate" "$checkpoint_kind" >>"$candidate_status_file"
    else
      candidate_exit=$?
      printf '%s\t%s\tFAILED\t%s\tevaluation_or_diagnosis\n' \
        "$candidate" "$checkpoint_kind" "$candidate_exit" \
        >>"$candidate_status_file"
    fi
  done
done

"$python_bin" -B tools/revision/summarize_mraf_gate_amplitude_rescue.py \
  --mode screen --candidate-root "$stage1_unit_root" \
  --reference-screen-root "$reference_screen_root" --output-dir "$run_dir" \
  >"${run_dir}/screen_summary.log"
selection_json="${run_dir}/selected_amplitude_candidate.json"
selection_status="$("$python_bin" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$selection_json")"

if [[ "$selection_status" == "AMPLITUDE_SCREEN_FAILED" ]]; then
  write_status "AMPLITUDE_SCREEN_FAILED" 3 "amplitude_screen"
  pipeline_completed=true
  exit 3
fi

if [[ "$selection_status" == "VIABLE_SELECTED_FOR_CALIBRATION" ]]; then
  current_stage="calibration"
  calibration_root="${run_dir}/calibration"
  mkdir -p "$calibration_root"
  source_candidate="$("$python_bin" -B -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["calibration_candidate"])' \
    "$selection_json")"
  source_checkpoint="$("$python_bin" -B -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["calibration_checkpoint"])' \
    "$selection_json")"
  source_unit="${stage1_unit_root}/${source_candidate}_${source_checkpoint}"
  for scale in 2 5 10 20 50; do
    unit_dir="${calibration_root}/scale_${scale}"
    if [[ ! -d "$unit_dir" ]]; then
      mkdir -p "$unit_dir"
      ln -s "${source_unit}/configs.yaml" "${unit_dir}/configs.yaml"
      ln -s "${source_unit}/checkpoint.pth" "${unit_dir}/checkpoint.pth"
      ln -s "${source_unit}/train.log" "${unit_dir}/train.log"
      ln -s "${source_unit}/mraf_epoch_metrics.tsv" "${unit_dir}/mraf_epoch_metrics.tsv"
      ln -s "${source_unit}/checkpoint_selection.tsv" "${unit_dir}/checkpoint_selection.tsv"
      cat >"${unit_dir}/unit.json" <<EOF
{
  "candidate": "$source_candidate",
  "checkpoint": "$source_checkpoint",
  "mraf_scorer_lr_multiplier": $("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["mraf_scorer_lr_multiplier"])' "$selection_json"),
  "mraf_gate_logit_scale": $scale,
  "mraf_rel_lambda": $("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["mraf_rel_lambda"])' "$selection_json"),
  "mraf_gate_base_grad_scale": $("$python_bin" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["mraf_gate_base_grad_scale"])' "$selection_json"),
  "read_only_calibration": true,
  "has_nan": false,
  "has_inf": false,
  "has_oom": false
}
EOF
    fi
    if evaluate_unit "$unit_dir" "$source_candidate" \
        "$screen_scenarios" 4 "$scale"; then
      printf 'calibration_scale_%s\t%s\tCOMPLETED\t0\tNONE\n' \
        "$scale" "$source_checkpoint" >>"$candidate_status_file"
    else
      calibration_exit=$?
      printf 'calibration_scale_%s\t%s\tFAILED\t%s\tevaluation_or_diagnosis\n' \
        "$scale" "$source_checkpoint" "$calibration_exit" \
        >>"$candidate_status_file"
    fi
  done
  "$python_bin" -B tools/revision/summarize_mraf_gate_amplitude_rescue.py \
    --mode calibration --candidate-root "$calibration_root" \
    --reference-screen-root "$reference_screen_root" \
    --selected-json "$selection_json" --output-dir "$run_dir" \
    >"${run_dir}/calibration_summary.log"
  selection_status="$("$python_bin" -B -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$selection_json")"
  if [[ "$selection_status" != "STRONG_SELECTED" ]]; then
    write_status "CALIBRATION_FAILED" 4 "calibration"
    pipeline_completed=true
    exit 4
  fi
else
  _write_empty_calibration="$python_bin"
  "$_write_empty_calibration" -B - "$run_dir" <<'PY'
import csv, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "calibration_screen.tsv"
with p.open("w", newline="", encoding="utf-8") as f:
    csv.writer(f, delimiter="\t").writerow(
        ["rank", "source_candidate", "source_checkpoint",
         "mraf_gate_logit_scale", "classification", "strong_pass",
         "clean_R1", "degraded_mean_R1", "alpha_std",
         "alpha_target_pearson", "failed_strong_criteria"]
    )
PY
fi

if [[ "$auto_continue" != "true" ]]; then
  write_status "RUNNING" 0 "awaiting_auto_continue"
  pipeline_completed=true
  echo "STATUS=RUNNING"
  echo "selected_candidate_json=$selection_json"
  echo "Confirmation/formal skipped because --auto-continue was not supplied."
  exit 0
fi

selected_candidate="$("$python_bin" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate"])' "$selection_json")"
selected_lr="$("$python_bin" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["mraf_scorer_lr_multiplier"])' "$selection_json")"
selected_scale="$("$python_bin" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["mraf_gate_logit_scale"])' "$selection_json")"
selected_lambda="$("$python_bin" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["mraf_rel_lambda"])' "$selection_json")"
selected_grad="$("$python_bin" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["mraf_gate_base_grad_scale"])' "$selection_json")"

current_stage="confirmation"
confirmation_train="${run_dir}/confirmation/train"
confirmation_units="${run_dir}/confirmation/units"
if ensure_training "$confirmation_train" \
    "mraf_gate_amplitude_confirmation_${selected_candidate}_seed${seed}" \
    12 1 "$selected_lambda" "$selected_grad" "$selected_lr" "$selected_scale"; then
  :
else
  confirmation_exit=$?
  write_status "CONFIRMATION_FAILED" "$confirmation_exit" "confirmation_training"
  pipeline_completed=true
  exit "$confirmation_exit"
fi
confirmation_train=$effective_train_dir
mkdir -p "$confirmation_units"
for checkpoint_kind in best last; do
  unit_dir="${confirmation_units}/${checkpoint_kind}"
  if {
    if [[ ! -d "$unit_dir" ]]; then
      prepare_unit "$unit_dir" "$confirmation_train" confirmation \
        "$checkpoint_kind" "$selected_lr" "$selected_scale" \
        "$selected_lambda" "$selected_grad"
    fi
    evaluate_unit "$unit_dir" confirmation \
      "$screen_scenarios" 4 "$selected_scale"
  }; then
    printf 'confirmation\t%s\tCOMPLETED\t0\tNONE\n' \
      "$checkpoint_kind" >>"$candidate_status_file"
  else
    confirmation_exit=$?
    printf 'confirmation\t%s\tFAILED\t%s\tevaluation_or_diagnosis\n' \
      "$checkpoint_kind" "$confirmation_exit" >>"$candidate_status_file"
  fi
done
"$python_bin" -B tools/revision/summarize_mraf_gate_amplitude_rescue.py \
  --mode confirmation --candidate-root "$confirmation_units" \
  --reference-screen-root "$reference_screen_root" --output-dir "$run_dir" \
  >"${run_dir}/confirmation_result.json"
confirmation_status="$("$python_bin" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "${run_dir}/confirmation_result.json")"
if [[ "$confirmation_status" != "CONFIRMATION_PASSED" ]]; then
  write_status "CONFIRMATION_FAILED" 5 "confirmation"
  pipeline_completed=true
  exit 5
fi

current_stage="formal_preflight"
check_space_gb "$run_dir" 8
formal_train="${run_dir}/formal/train"
formal_units="${run_dir}/formal/units"
current_stage="formal_training"
ensure_training "$formal_train" \
  "mraf_gate_amplitude_formal_${selected_candidate}_seed${seed}" \
  60 3 "$selected_lambda" "$selected_grad" "$selected_lr" "$selected_scale"
formal_train=$effective_train_dir
mkdir -p "$formal_units"
current_stage="formal_evaluation_and_diagnosis"
for checkpoint_kind in best last; do
  unit_dir="${formal_units}/${checkpoint_kind}"
  if [[ ! -d "$unit_dir" ]]; then
    prepare_unit "$unit_dir" "$formal_train" formal \
      "$checkpoint_kind" "$selected_lr" "$selected_scale" \
      "$selected_lambda" "$selected_grad"
  fi
  evaluate_unit "$unit_dir" formal "$formal_scenarios" 24 "$selected_scale"
done

current_stage="final_reports"
"$python_bin" -B tools/revision/summarize_mraf_gate_amplitude_rescue.py \
  --mode final --candidate-root "$formal_units" \
  --reference-screen-root "$reference_screen_root" \
  --reference-run-root "$reference_run_root" --output-dir "$run_dir" \
  >"${run_dir}/final_summary.json"
final_status="$("$python_bin" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "${run_dir}/final_summary.json")"
[[ "$final_status" == "COMPLETED" ]] || die "Formal report did not pass strict gate"

{
  echo "selected_candidate=$selected_candidate"
  echo "selected_scorer_lr_multiplier=$selected_lr"
  echo "selected_gate_logit_scale=$selected_scale"
  echo "formal_best_sha256=$(sha256sum "${formal_units}/best/checkpoint.pth" | awk '{print $1}')"
  echo "formal_last_sha256=$(sha256sum "${formal_units}/last/checkpoint.pth" | awk '{print $1}')"
} >>"${run_dir}/experiment_manifest.txt"
archive="${run_dir}/mraf_gate_amplitude_rescue_reports.tar.gz"
tar -czf "$archive" -C "$run_dir" \
  amplitude_screen_metrics.tsv amplitude_screen_mechanism.tsv \
  checkpoint_comparison.tsv amplitude_candidate_ranking.tsv \
  calibration_screen.tsv selected_amplitude_candidate.json \
  confirmation_summary.tsv final_metrics.tsv \
  final_robustness_summary.tsv final_mechanism_summary.tsv \
  comparison_with_previous_screen.tsv comparison_with_old_seed1.tsv \
  FINAL_REPORT.txt experiment_manifest.txt candidate_status.tsv
[[ -s "$archive" ]] || die "Final report archive is missing or empty"

write_status "COMPLETED" 0 "NONE"
formal_completed=true
pipeline_completed=true
echo "STATUS=COMPLETED"
echo "output_dir=$run_dir"
if [[ "$shutdown_on_success" == "true" && "$formal_completed" == "true" \
      && -s "$archive" ]]; then
  "$shutdown_command" -h now
else
  echo "shutdown_executed=false"
fi
