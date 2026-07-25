#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run the fixed MRAF gate-rescue screen and optionally continue with one formal run.

Usage:
  bash scripts/mraf/run_mraf_gate_rescue_pipeline.sh \
    --data-root <TextReID-root> --output-root <new-parent> \
    --base-config <configs.yaml> \
    --reference-run-root <old-seed1-root> \
    --device 0 --seed 1 \
    [--auto-continue] [--shutdown-on-success]

Safe validation:
  bash scripts/mraf/run_mraf_gate_rescue_pipeline.sh --dry-run --mock
  bash scripts/mraf/run_mraf_gate_rescue_pipeline.sh --mock

Resume:
  bash scripts/mraf/run_mraf_gate_rescue_pipeline.sh \
    --resume-run-root <existing-pipeline-run> [--auto-continue]

Options:
  --data-root PATH          TextReID dataset root
  --output-root PATH        parent for a timestamped, non-overwriting run
  --base-config PATH        historical formal MDRC+MRAF configs.yaml
  --reference-run-root PATH old seed1 root used only for final comparison
  --resume-run-root PATH    safely resume this pipeline's incomplete run
  --device INDEX            CUDA device index (default: 0)
  --seed N                  non-negative training seed (default: 1)
  --auto-continue           train the selected configuration for 60 epochs
  --shutdown-on-success     shutdown only after a fully completed formal run
  --dry-run                 print preflight and commands; run nothing
  --mock                    CPU/filesystem state-machine test; never train/shutdown
  -h, --help                show this help

The script never uses screen. For a real unattended launch, wrap this command
with external nohup. A failed screen, incomplete report, or mock run never shuts
down the host.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 2
}

data_root=""
output_root=""
base_config="${MRAF_BASE_CONFIG:-}"
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
[[ "$device" =~ ^[0-9]+$ ]] || die "--device must be a non-negative integer"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"
git_root="$(cd "${repo_dir}/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python}"
shutdown_command="${MRAF_SHUTDOWN_COMMAND:-shutdown}"

# Windows-created worktrees contain a Windows-form gitdir pointer. Convert it
# when this Linux-only launcher is invoked through WSL.
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
git_short="${git_head:0:8}"
git_branch="$(run_git branch --show-current)"
timestamp="$(date +%Y%m%d_%H%M%S)"

if [[ "$mock" == "true" && -z "$output_root" && -z "$resume_run_root" ]]; then
  output_root="$(mktemp -d -t mraf_gate_rescue_mock_XXXXXX)"
fi
if [[ -n "$resume_run_root" ]]; then
  [[ -d "$resume_run_root" ]] || die "--resume-run-root is not a directory: $resume_run_root"
  run_dir="$(cd "$resume_run_root" && pwd)"
  output_root="$(dirname "$run_dir")"
  resume_mode=true
else
  [[ -n "$output_root" ]] || die "--output-root must not be empty"
  run_dir="${output_root%/}/${timestamp}_${git_short}_mraf_gate_rescue_seed${seed}"
  resume_mode=false
fi
[[ "$resume_mode" != "true" || "$mock" != "true" ]] || die \
  "--mock creates a fresh disposable run and cannot be combined with --resume-run-root"
candidate_root="${run_dir}/candidates"
screen_dir="${run_dir}/screen"
status_file="${run_dir}/status.txt"
candidate_status_file="${run_dir}/candidate_status.tsv"
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
candidate mraf_gate_fp32 mraf_rel_lambda mraf_gate_base_grad_scale
S0 false 0.01 1.00
S1 true  0.01 1.00
S2 true  0.25 1.00
S3 true  1.00 1.00
S4 true  0.25 0.10
S5 true  0.25 0.00
EOF
}

candidate_values() {
  case "$1" in
    S0) printf '%s %s %s\n' false 0.01 1.00 ;;
    S1) printf '%s %s %s\n' true  0.01 1.00 ;;
    S2) printf '%s %s %s\n' true  0.25 1.00 ;;
    S3) printf '%s %s %s\n' true  1.00 1.00 ;;
    S4) printf '%s %s %s\n' true  0.25 0.10 ;;
    S5) printf '%s %s %s\n' true  0.25 0.00 ;;
    *) die "Unknown candidate: $1" ;;
  esac
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

manifest_value() {
  local key=$1
  sed -n "s/^${key}=//p" "${run_dir}/manifest.txt" | tail -n 1
}

matrix_id="S0:false:0.01:1.00,S1:true:0.01:1.00,S2:true:0.25:1.00,S3:true:1.00:1.00,S4:true:0.25:0.10,S5:true:0.25:0.00"

if [[ "$resume_mode" == "true" ]]; then
  [[ -f "${run_dir}/manifest.txt" && -f "${run_dir}/status.txt" ]] || die \
    "--resume-run-root must contain this pipeline's manifest.txt and status.txt"
  [[ "$(manifest_value PIPELINE_ID)" == "mraf_gate_rescue_v2" ]] || die \
    "Resume manifest is not an MRAF gate-rescue v2 run"
  [[ "$(manifest_value git_head)" == "$git_head" ]] || die \
    "Resume commit differs from current Git HEAD"
  [[ "$(manifest_value seed)" == "$seed" ]] || die \
    "Resume seed differs from the original run"
  [[ "$(manifest_value candidate_matrix)" == "$matrix_id" ]] || die \
    "Resume candidate matrix differs from the fixed S0-S5 matrix"
  [[ "$(sed -n 's/^STATUS=//p' "$status_file" | tail -n 1)" != "COMPLETED" ]] || die \
    "STATUS=COMPLETED runs cannot be resumed"
  [[ -n "$data_root" ]] || data_root="$(manifest_value data_root)"
  [[ -n "$base_config" ]] || base_config="$(manifest_value base_config)"
  [[ -n "$reference_run_root" ]] || reference_run_root="$(manifest_value reference_run_root)"
fi

validate_protocol_paths() {
  [[ -n "$data_root" && -d "$data_root" ]] || die \
    "--data-root is not a directory: $data_root"
  [[ -d "$output_root" ]] || die "--output-root parent does not exist: $output_root"
  [[ -n "$reference_run_root" && -d "$reference_run_root" ]] || die \
    "--reference-run-root is not a directory: $reference_run_root"
  data_root="$(cd "$data_root" && pwd)"
  reference_run_root="$(cd "$reference_run_root" && pwd)"
  if [[ -z "$base_config" ]]; then
    mapfile -t base_matches < <(
      find "$reference_run_root" -type f \
        -path '*formal/*_mdrc_mraf_seed1*/configs.yaml' -print 2>/dev/null || true
    )
    [[ ${#base_matches[@]} -eq 1 ]] || die \
      "Provide --base-config; reference discovery found ${#base_matches[@]} matches"
    base_config=${base_matches[0]}
  fi
  [[ -f "$base_config" ]] || die "Base config not found: $base_config"
  base_config="$(cd "$(dirname "$base_config")" && pwd)/$(basename "$base_config")"
  caption_file=""
  for candidate_caption in \
    "${data_root%/}/RGBT-PEDES/RGBT-PEDES-Final/Caption.json" \
    "${data_root%/}/RGBT-PEDES/RGBT-PEDES/Caption.json" \
    "${data_root%/}/RGBT-PEDES/Caption.json"; do
    if [[ -f "$candidate_caption" ]]; then caption_file=$candidate_caption; break; fi
  done
  [[ -n "$caption_file" ]] || die "Caption.json not found under --data-root"
  "$python_bin" -B - "$base_config" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "batch_size": 64, "loss_names": "sdm+mlm+id", "MLM": True,
    "warmup_epochs": 5, "eval_use_fused_mu": True,
    "mdrc_loss_scale": 100, "mdrc_lambda": 0.05, "mdrc_prob": 0.5,
    "mdrc_warmup_epoch": 3,
}
bad = {key: (cfg.get(key), value) for key, value in expected.items()
       if cfg.get(key) != value}
if bad:
    raise SystemExit(f"Historical formal config mismatch: {bad}")
PY
}

if [[ "$mock" != "true" ]]; then
  validate_protocol_paths
  if [[ "$resume_mode" == "true" ]]; then
    [[ "$(manifest_value data_root)" == "$data_root" ]] || die \
      "Resume data root differs from the original run"
    [[ "$(manifest_value base_config)" == "$base_config" ]] || die \
      "Resume base config differs from the original run"
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
  echo "reference_run_root=${reference_run_root:-MOCK}"
  echo "resume_run_root=${resume_run_root:-NONE}"
  echo "run_dir=$run_dir"
  echo "device=$device"
  echo "seed=$seed"
  echo "auto_continue=$auto_continue"
  echo "shutdown_on_success=$shutdown_on_success"
  print_matrix
  echo "No directory, training, evaluation, diagnosis, or shutdown was executed."
  exit 0
fi

if [[ "$resume_mode" == "false" ]]; then
  [[ ! -e "$run_dir" ]] || die "Refusing to overwrite existing run directory: $run_dir"
  mkdir -p "$candidate_root" "$screen_dir"
  printf 'candidate\tstatus\texit_code\tfailed_stage\n' >"$candidate_status_file"
  write_status "RUNNING" 0 "NONE"
  {
    echo "PIPELINE_ID=mraf_gate_rescue_v2"
    echo "git_branch=$git_branch"
    echo "git_head=$git_head"
    echo "seed=$seed"
    echo "degrade_seed=2026"
    echo "device=$device"
    echo "data_root=${data_root:-MOCK}"
    echo "base_config=${base_config:-MOCK}"
    echo "reference_run_root=${reference_run_root:-MOCK}"
    echo "candidate_matrix=$matrix_id"
    echo "auto_continue=$auto_continue"
    echo "shutdown_on_success=$shutdown_on_success"
    echo "shutdown_command=$shutdown_command"
    echo "mock=$mock"
  } >"${run_dir}/manifest.txt"
  {
    echo "timestamp=$(date -Iseconds)"
    echo "python=$("$python_bin" --version 2>&1)"
    echo "git_head=$git_head"
    if [[ "$mock" == "true" ]]; then
      echo "git_status=SKIPPED_MOCK"
    else
      echo "git_status=$(run_git status --short | tr '\n' ';')"
    fi
  } >"${run_dir}/environment.txt"
else
  mkdir -p "$candidate_root" "$screen_dir"
  write_status "RUNNING" 0 "NONE"
  {
    echo "resume_at=$(date -Iseconds)"
    echo "resume_git_head=$git_head"
  } >>"${run_dir}/environment.txt"
fi

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

link_candidate_artifacts() {
  local candidate_dir=$1
  local config_path checkpoint_path train_log_path
  config_path="$(resolve_unique_artifact "${candidate_dir}/train_output" configs.yaml)"
  checkpoint_path="$(resolve_unique_artifact "${candidate_dir}/train_output" best.pth)"
  train_log_path="${candidate_dir}/train.log"
  [[ -f "$train_log_path" ]] || return 1
  ln -s "$config_path" "${candidate_dir}/configs.yaml"
  ln -s "$checkpoint_path" "${candidate_dir}/best.pth"
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
    *) die "Unknown evaluation scenario: $1" ;;
  esac
}

run_evaluations() {
  local run_path=$1
  local scenarios=$2
  local config_path="${run_path}/configs.yaml"
  local checkpoint_path="${run_path}/best.pth"
  local scenario degrade_mode strength blur_kernel eval_dir
  for scenario in $scenarios; do
    read -r degrade_mode strength blur_kernel < <(scenario_values "$scenario")
    eval_dir="${run_path}/eval/${scenario}"
    if [[ -s "${eval_dir}/scores.pt" ]]; then
      continue
    fi
    mkdir -p "$eval_dir"
    eval_log="${eval_dir}/eval.log"
    eval_pid="${eval_dir}/pid"
    if [[ -e "$eval_log" || -e "$eval_pid" ]]; then
      eval_log="${eval_dir}/eval_resume_${timestamp}.log"
      eval_pid="${eval_dir}/pid_resume_${timestamp}"
    fi
    run_logged \
      "$eval_log" "$eval_pid" \
      env "CUDA_VISIBLE_DEVICES=$device" "$python_bin" -B test.py \
      --config_file "$config_path" \
      --test_ckpt_file "$checkpoint_path" \
      --eval_degrade_mode "$degrade_mode" \
      --eval_degrade_strength "$strength" \
      --eval_degrade_blur_kernel "$blur_kernel" \
      --eval_degrade_seed 2026 \
      --save_eval_scores \
      --eval_scores_save_raw \
      --eval_scores_file "${eval_dir}/scores.pt"
  done
}

run_diagnosis() {
  local run_path=$1
  local label=$2
  local max_batches=$3
  local diagnosis_dir="${run_path}/diagnostics"
  local diagnosis_log="${run_path}/diagnosis.log"
  local diagnosis_pid="${run_path}/diagnosis.pid"
  if [[ -s "${diagnosis_dir}/diagnosis.json" \
        && -s "${diagnosis_dir}/gradient_diagnostics.csv" \
        && -s "${diagnosis_dir}/fusion_path_check.json" ]]; then
    return 0
  fi
  if [[ -e "$diagnosis_dir" ]]; then
    diagnosis_dir="${run_path}/diagnostics_resume_${timestamp}"
    diagnosis_log="${run_path}/diagnosis_resume_${timestamp}.log"
    diagnosis_pid="${run_path}/diagnosis_resume_${timestamp}.pid"
  fi
  run_logged \
    "$diagnosis_log" "$diagnosis_pid" \
    env "CUDA_VISIBLE_DEVICES=$device" "$python_bin" -B \
    tools/revision/diagnose_mraf_mechanism.py \
    --config-file "${run_path}/configs.yaml" \
    --checkpoint "${run_path}/best.pth" \
    --case-label "$label" \
    --data-root "$data_root" \
    --device cuda:0 \
    --degrade-seed 2026 \
    --max-batches "$max_batches" \
    --gradient-batches 3 \
    --gradient-microbatch-size 4 \
    --output-dir "$diagnosis_dir"
  if [[ "$diagnosis_dir" != "${run_path}/diagnostics" ]]; then
    printf '%s\n' "$(basename "$diagnosis_dir")" \
      >"${run_path}/active_diagnostics.txt"
  fi
}

make_mock_candidate() {
  local label=$1
  local candidate_dir="${candidate_root}/${label}"
  local gate_fp32 rel_lambda grad_scale
  read -r gate_fp32 rel_lambda grad_scale < <(candidate_values "$label")
  mkdir -p "$candidate_dir/eval" "$candidate_dir/diagnostics"
  printf 'mock\n' >"${candidate_dir}/configs.yaml"
  printf 'mock\n' >"${candidate_dir}/best.pth"
  printf 'mock\n' >"${candidate_dir}/train.log"
  "$python_bin" -B - "$candidate_dir" "$label" "$gate_fp32" "$rel_lambda" "$grad_scale" <<'PY'
import csv
import json
import sys
from pathlib import Path

root, label, fp32, rel_lambda, grad_scale = sys.argv[1:]
root = Path(root)
labels = ["S0", "S1", "S2", "S3", "S4", "S5"]
index = labels.index(label)
passes = label == "S4"
metadata = {
    "candidate": label,
    "mraf_gate_fp32": fp32 == "true",
    "mraf_rel_lambda": float(rel_lambda),
    "mraf_gate_base_grad_scale": float(grad_scale),
    "has_nan": False,
    "has_inf": False,
    "has_oom": label == "S2",
}
(root / "candidate.json").write_text(json.dumps(metadata, indent=2) + "\n")
scenarios = [
    "clean", "rgb_missing", "ir_missing", "rgb_noise_0.10",
    "ir_noise_0.10", "rgb_blur_k7", "ir_blur_k7", "both_blur_k7",
]
base = 70.0 + 0.03 * index
for offset, scenario in enumerate(scenarios):
    directory = root / "eval" / scenario
    directory.mkdir(parents=True)
    r1 = base if scenario == "clean" else 60.0 + 0.02 * index - 0.01 * offset
    payload = {"metrics": {"R1": r1, "R5": r1 + 10, "R10": r1 + 15,
                           "mAP": r1 - 5, "mINP": r1 - 10}}
    (directory / "scores.json").write_text(json.dumps(payload) + "\n")
alpha_std = 0.008 if passes else 0.001
pearson = 0.45 if passes else 0.10
case = {
    "alpha_rgb_std": alpha_std,
    "alpha_target_pearson_degraded_scenarios": pearson,
    "weighted_rel_base_ratio_median": 0.01 if passes else 0.0001,
    "rel_only_last_weight_grad_norm_median": 0.02,
    "fusion_path_sensitive": True,
    "alarms": {"PARAMETER_NOT_UPDATED": False},
}
(root / "diagnostics" / "diagnosis.json").write_text(
    json.dumps({"cases": {label: case}}, indent=2) + "\n"
)
with (root / "diagnostics" / "scenario_gate_statistics.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    fields = ["scenario", "alpha_rgb_mean", "alpha_ir_mean", "raw_logits_dtype"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for scenario in scenarios:
        writer.writerow({
            "scenario": scenario,
            "alpha_rgb_mean": 0.25 if scenario == "rgb_missing" and passes else 0.5,
            "alpha_ir_mean": 0.25 if scenario == "ir_missing" and passes else 0.5,
            "raw_logits_dtype": "torch.float32" if fp32 == "true" else "torch.float16",
        })
with (root / "diagnostics" / "gate_direction_statistics.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    fields = ["scenario", "degraded_alpha_delta_mean", "correct_direction_fraction"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for scenario in ["rgb_noise_0.10", "ir_noise_0.10", "rgb_blur_k7", "ir_blur_k7"]:
        writer.writerow({
            "scenario": scenario,
            "degraded_alpha_delta_mean": -0.03 if passes else -0.001,
            "correct_direction_fraction": 0.8 if passes else 0.5,
        })
with (root / "diagnostics" / "gradient_diagnostics.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    fields = ["weighted_mraf_rel_over_loss_base",
              "rel_only_scorer_last_weight_grad_norm"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerow({"weighted_mraf_rel_over_loss_base": 0.01,
                     "rel_only_scorer_last_weight_grad_norm": 0.02})
PY
}

if [[ "$mock" == "true" ]]; then
  current_stage="mock_candidates"
  for label in S0 S1 S2 S3 S4 S5; do
    make_mock_candidate "$label"
    if [[ "$label" == "S2" ]]; then
      printf '%s\tFAILED\t1\tMOCK_OOM\n' "$label" >>"$candidate_status_file"
    else
      printf '%s\tCOMPLETED\t0\tNONE\n' "$label" >>"$candidate_status_file"
    fi
  done
  current_stage="mock_summary"
  "$python_bin" -B tools/revision/summarize_mraf_gate_rescue.py \
    --candidate-root "$candidate_root" --output-dir "$screen_dir" \
    >"${run_dir}/summarize.log" 2>&1
  selected="$("$python_bin" -B -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate"] or "")' \
    "${screen_dir}/selected_candidate.json")"
  printf '%s\n' "$selected" >"${run_dir}/selected_candidate.txt"
  write_status "MOCK_COMPLETED" 0 "NONE"
  pipeline_completed=true
  echo "mock_output_dir=$run_dir"
  echo "selected_candidate=$selected"
  echo "shutdown_executed=false"
  exit 0
fi

current_stage="preflight"
[[ -z "$(run_git status --short)" ]] || die \
  "Real pipeline requires a clean Git worktree"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  >"${run_dir}/gpu.txt"
env "CUDA_VISIBLE_DEVICES=$device" "$python_bin" -B -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))' \
  >>"${run_dir}/gpu.txt"
if pgrep -af 'train.py.*mraf_gate_rescue' | grep -v "$$" >/dev/null 2>&1; then
  die "A matching MRAF gate-rescue training process is already active"
fi
if [[ ! -f "${run_dir}/historical_base_configs.yaml" ]]; then
  cp "$base_config" "${run_dir}/historical_base_configs.yaml"
fi

common_train_args=(
  --dataset_name RGBT-PEDES --root_dir "$data_root"
  --img_aug --MLM --loss_names sdm+mlm+id
  --batch_size 64 --test_batch_size 512 --num_workers 8
  --warmup_epochs 5 --lr 1e-5 --eval_use_fused_mu
  --log_period 100 --eval_period 1 --degrade_aug_mode none
  --use_rgbt_add_fusion --mraf_enabled
  --mraf_hidden_dim 128 --mraf_dropout 0.1
  --mraf_gate_temperature 1.0 --mraf_target_temperature 0.1
  --mraf_balance_lambda 0.0
  --mdrc_enable --mdrc_prob 0.5 --mdrc_lambda 0.05
  --mdrc_loss_scale 100 --mdrc_warmup_epoch 3
)

run_training() {
  local run_path=$1
  local label=$2
  local epochs=$3
  local rel_warmup=$4
  local gate_fp32=$5
  local rel_lambda=$6
  local grad_scale=$7
  local name=$8
  mkdir -p "${run_path}/train_output"
  local command=(
    env "CUDA_VISIBLE_DEVICES=$device" "TRAIN_SEED=$seed"
    "$python_bin" -u train.py
    --name "$name" --output_dir "${run_path}/train_output"
    --num_epoch "$epochs" --mraf_rel_warmup_epoch "$rel_warmup"
    --mraf_rel_lambda "$rel_lambda"
    --mraf_gate_base_grad_scale "$grad_scale"
    "${common_train_args[@]}"
  )
  if [[ "$gate_fp32" == "true" ]]; then
    command+=(--mraf_gate_fp32)
  fi
  print_command "${command[@]}" >"${run_path}/training_command.txt"
  run_logged "${run_path}/train.log" "${run_path}/train.pid" "${command[@]}"
  if grep -Eiq 'CUDA out of memory|OutOfMemoryError|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
    "${run_path}/train.log"; then
    return 86
  fi
  link_candidate_artifacts "$run_path"
}

run_candidate() {
  local label=$1
  local candidate_dir=$2
  local gate_fp32 rel_lambda grad_scale
  read -r gate_fp32 rel_lambda grad_scale < <(candidate_values "$label")
  [[ ! -e "$candidate_dir" ]] || return 90
  mkdir -p "$candidate_dir"
  cat >"${candidate_dir}/candidate.json" <<EOF
{
  "candidate": "$label",
  "mraf_gate_fp32": $gate_fp32,
  "mraf_rel_lambda": $rel_lambda,
  "mraf_gate_base_grad_scale": $grad_scale,
  "seed": $seed,
  "epochs": 8,
  "mraf_rel_warmup_epoch": 1,
  "has_nan": false,
  "has_inf": false,
  "has_oom": false
}
EOF
  run_training "$candidate_dir" "$label" 8 1 "$gate_fp32" \
    "$rel_lambda" "$grad_scale" "mraf_gate_rescue_${label}_seed${seed}"
  run_evaluations "$candidate_dir" \
    "clean rgb_missing ir_missing rgb_noise_0.10 ir_noise_0.10 rgb_blur_k7 ir_blur_k7 both_blur_k7"
  run_diagnosis "$candidate_dir" "$label" 4
}

active_candidate_path() {
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

candidate_complete() {
  local path=$1
  [[ -s "${path}/configs.yaml" && -s "${path}/best.pth" \
      && -s "${path}/train.log" ]] || return 1
  local scenario
  for scenario in clean rgb_missing ir_missing rgb_noise_0.10 ir_noise_0.10 \
    rgb_blur_k7 ir_blur_k7 both_blur_k7; do
    [[ -s "${path}/eval/${scenario}/scores.pt" ]] || return 1
  done
  local diagnostic_path="${path}/diagnostics"
  if [[ -s "${path}/active_diagnostics.txt" ]]; then
    diagnostic_path="${path}/$(<"${path}/active_diagnostics.txt")"
  fi
  [[ -s "${diagnostic_path}/diagnosis.json" \
      && -s "${diagnostic_path}/gradient_diagnostics.csv" \
      && -s "${diagnostic_path}/fusion_path_check.json" ]]
}

mark_candidate_failure_flags() {
  local path=$1
  local metadata="${path}/candidate.json"
  [[ -f "$metadata" ]] || return 0
  if grep -REiq --include='*.log' \
    'CUDA out of memory|OutOfMemoryError' "$path" 2>/dev/null; then
    sed -i 's/"has_oom": false/"has_oom": true/' "$metadata"
  fi
  if grep -REiq --include='*.log' \
    '(^|[^[:alpha:]])nan([^[:alpha:]]|$)' "$path" 2>/dev/null; then
    sed -i 's/"has_nan": false/"has_nan": true/' "$metadata"
  fi
  if grep -REiq --include='*.log' \
    '(^|[^[:alpha:]])inf([^[:alpha:]]|$)' "$path" 2>/dev/null; then
    sed -i 's/"has_inf": false/"has_inf": true/' "$metadata"
  fi
}

current_stage="candidate_screen"
for label in S0 S1 S2 S3 S4 S5; do
  candidate_stage="train_eval_diagnose_${label}"
  candidate_container="${candidate_root}/${label}"
  candidate_run_path="$(active_candidate_path "$candidate_container")"
  if [[ "$resume_mode" == "true" ]] && candidate_complete "$candidate_run_path"; then
    printf '%s\tCOMPLETED\t0\tRESUME_SKIPPED_COMPLETE\n' \
      "$label" >>"$candidate_status_file"
    continue
  fi
  if [[ "$resume_mode" == "true" && -e "$candidate_container" ]]; then
    candidate_run_path="${candidate_container}/resume_attempt_${timestamp}"
  else
    candidate_run_path="$candidate_container"
  fi
  if run_candidate "$label" "$candidate_run_path"; then
    if [[ "$candidate_run_path" != "$candidate_container" ]]; then
      printf '%s\n' "${candidate_run_path#${candidate_container}/}" \
        >"${candidate_container}/active_attempt.txt"
    fi
    printf '%s\tCOMPLETED\t0\tNONE\n' "$label" >>"$candidate_status_file"
  else
    candidate_exit=$?
    mark_candidate_failure_flags "$candidate_run_path"
    printf '%s\tFAILED\t%s\t%s\n' \
      "$label" "$candidate_exit" "$candidate_stage" >>"$candidate_status_file"
  fi
done

current_stage="screen_summary"
if [[ "$resume_mode" == "true" && -n "$(find "$screen_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  screen_dir="${run_dir}/screen_resume_${timestamp}"
  printf '%s\n' "$(basename "$screen_dir")" >"${run_dir}/active_screen.txt"
fi
"$python_bin" -B tools/revision/summarize_mraf_gate_rescue.py \
  --candidate-root "$candidate_root" --output-dir "$screen_dir" \
  >"${run_dir}/summarize.log" 2>&1
selection_status="$("$python_bin" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "${screen_dir}/selected_candidate.json")"
selected="$("$python_bin" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_candidate"] or "")' \
  "${screen_dir}/selected_candidate.json")"
if [[ -s "${run_dir}/selected_candidate.txt" ]]; then
  [[ "$(<"${run_dir}/selected_candidate.txt")" == "$selected" ]] || die \
    "Resume selection differs from the previously selected candidate"
else
  printf '%s\n' "$selected" >"${run_dir}/selected_candidate.txt"
fi
if [[ "$selection_status" != "SELECTED" ]]; then
  write_status "SCREEN_FAILED" 3 "screen_summary"
  pipeline_completed=true
  echo "No candidate passed; retained all screen results at $run_dir"
  exit 3
fi

if [[ "$auto_continue" != "true" ]]; then
  write_status "SCREEN_COMPLETED" 0 "NONE"
  pipeline_completed=true
  echo "selected_candidate=$selected"
  echo "Formal training skipped because --auto-continue was not supplied."
  exit 0
fi

read -r formal_fp32 formal_lambda formal_grad_scale < <(
  candidate_values "$selected"
)
formal_dir="${run_dir}/formal/${selected}_seed${seed}"
formal_container="$formal_dir"
formal_run_path="$(active_candidate_path "$formal_container")"
if [[ "$resume_mode" == "true" \
      && -s "${formal_run_path}/configs.yaml" \
      && -s "${formal_run_path}/best.pth" \
      && -s "${formal_run_path}/train.log" ]]; then
  :
elif [[ "$resume_mode" == "true" && -e "$formal_container" ]]; then
  formal_run_path="${formal_container}/resume_attempt_${timestamp}"
  mkdir -p "$formal_container"
  current_stage="formal_training"
  run_training "$formal_run_path" "$selected" 60 3 "$formal_fp32" \
    "$formal_lambda" "$formal_grad_scale" \
    "mraf_gate_rescue_formal_${selected}_seed${seed}"
  printf '%s\n' "${formal_run_path#${formal_container}/}" \
    >"${formal_container}/active_attempt.txt"
else
  [[ ! -e "$formal_container" ]] || die "Refusing to overwrite formal directory"
  mkdir -p "$formal_container"
  formal_run_path="$formal_container"
  current_stage="formal_training"
  run_training "$formal_run_path" "$selected" 60 3 "$formal_fp32" \
    "$formal_lambda" "$formal_grad_scale" \
    "mraf_gate_rescue_formal_${selected}_seed${seed}"
fi
current_stage="formal_training"
[[ -f "${formal_run_path}/best.pth" ]] || die "Formal best.pth missing"

current_stage="formal_evaluation"
run_evaluations "$formal_run_path" \
  "clean rgb_missing ir_missing rgb_noise_0.05 rgb_noise_0.10 ir_noise_0.05 ir_noise_0.10 rgb_blur_k7 ir_blur_k7 both_noise_0.05 both_blur_k7"
[[ "$(find "${formal_run_path}/eval" -name scores.pt -type f | wc -l)" -eq 11 ]] || \
  die "Formal evaluation did not produce 11 score files"

current_stage="formal_diagnosis"
run_diagnosis "$formal_run_path" "${selected}_formal" 24
formal_diagnostics="${formal_run_path}/diagnostics"
if [[ -s "${formal_run_path}/active_diagnostics.txt" ]]; then
  formal_diagnostics="${formal_run_path}/$(<"${formal_run_path}/active_diagnostics.txt")"
fi
for required in checkpoint_parameter_stats.csv scenario_gate_statistics.csv \
  scenario_target_statistics.csv gate_direction_statistics.csv \
  gradient_diagnostics.csv fusion_path_check.json diagnosis.json summary.txt \
  sample_level_diagnostics.pt; do
  [[ -f "${formal_diagnostics}/${required}" ]] || die \
    "Formal diagnosis missing ${required}"
done
if grep -Eiq 'CUDA out of memory|OutOfMemoryError|(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)' \
  "${formal_run_path}/train.log" "${formal_run_path}"/diagnosis*.log; then
  die "Formal outputs contain NaN, Inf, or OOM"
fi

current_stage="final_reports"
final_reports_dir="${run_dir}/final_reports"
if [[ "$resume_mode" == "true" \
      && -n "$(find "$final_reports_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  final_reports_dir="${run_dir}/final_reports_resume_${timestamp}"
  printf '%s\n' "$(basename "$final_reports_dir")" \
    >"${run_dir}/active_final_reports.txt"
fi
"$python_bin" -B tools/revision/summarize_mraf_gate_rescue.py \
  --final-run-dir "$formal_run_path" \
  --old-run-root "$reference_run_root" \
  --output-dir "$final_reports_dir" \
  >"${run_dir}/final_summary.log" 2>&1
{
  echo "base_config=$base_config"
  echo "reference_run_root=$reference_run_root"
} >>"${final_reports_dir}/FINAL_REPORT.txt"
if [[ ! -e "${formal_run_path}/metrics.tsv" ]]; then
  cp "${final_reports_dir}/final_metrics.tsv" "${formal_run_path}/metrics.tsv"
fi
if [[ ! -e "${formal_run_path}/robustness_summary.tsv" ]]; then
  cp "${final_reports_dir}/final_robustness_summary.tsv" \
    "${formal_run_path}/robustness_summary.tsv"
fi
{
  echo "git_head=$git_head"
  echo "git_branch=$git_branch"
  echo "selected_candidate=$selected"
  echo "base_config=$base_config"
  echo "reference_run_root=$reference_run_root"
  echo "formal_config=${formal_run_path}/configs.yaml"
  echo "formal_checkpoint=${formal_run_path}/best.pth"
  echo "formal_checkpoint_sha256=$(sha256sum "${formal_run_path}/best.pth" | awk '{print $1}')"
  echo "formal_config_sha256=$(sha256sum "${formal_run_path}/configs.yaml" | awk '{print $1}')"
  while IFS= read -r score_file; do
    echo "score_sha256=$(sha256sum "$score_file" | awk '{print $1}') $score_file"
  done < <(find "${formal_run_path}/eval" -name scores.pt -type f -print | sort)
  echo "degrade_seed=2026"
} >"${final_reports_dir}/experiment_manifest.txt"
archive_path="${run_dir}/mraf_gate_rescue_reports.tar.gz"
if [[ -e "$archive_path" ]]; then
  archive_path="${run_dir}/mraf_gate_rescue_reports_resume_${timestamp}.tar.gz"
  printf '%s\n' "$(basename "$archive_path")" \
    >"${run_dir}/active_report_archive.txt"
fi
tar -czf "$archive_path" \
  -C "$run_dir" "$(basename "$screen_dir")" "$(basename "$final_reports_dir")" candidate_status.tsv \
  selected_candidate.txt manifest.txt environment.txt
[[ -s "$archive_path" ]] || die \
  "Final report archive is missing or empty"

write_status "COMPLETED" 0 "NONE"
formal_completed=true
pipeline_completed=true
echo "STATUS=COMPLETED"
echo "output_dir=$run_dir"
if [[ "$shutdown_on_success" == "true" && "$formal_completed" == "true" \
      && -s "$archive_path" ]]; then
  "$shutdown_command" -h now
else
  echo "shutdown_executed=false"
fi
