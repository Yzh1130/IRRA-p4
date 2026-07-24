#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Prepare/run one MRAF experiment in the foreground.

Usage:
  run_mraf_experiments.sh --case mraf_only [options]
  MDRC_LOSS_SCALE=<verified-value> run_mraf_experiments.sh --case mdrc_mraf [options]
  MDRC_LOSS_SCALE=<verified-value> run_mraf_experiments.sh --case mdrc_mraf_norel [options]

Cases:
  mraf_only       MRAF-only with reliability supervision
  mdrc_mraf       MDRC+MRAF
  mdrc_mraf_norel MDRC+MRAF with mraf_rel_lambda=0

Options:
  --case NAME         Required case name from the list above
  --seed N            Non-negative training seed (default: 1)
  --num-epoch N       Positive epoch count (default: 60)
  --batch-size N      Positive per-step batch size
  --loss-names STRING '+'-separated tasks: sdm, id, mlm, cmpm, itc
  --output-root PATH  Parent output directory (default: $OUTPUT_ROOT or mraf_runs)
  --dry-run           Print the resolved command/configuration without writing or training
  -h, --help          Show this help

For num_epoch=1 only, omitted --batch-size and --loss-names use the unverified
smoke defaults batch_size=16 and loss_names=sdm+mlm. Formal/long-run training
(num_epoch>1), including dry-run, requires both options to be supplied after
verification against the historical Base/MDRC training artifacts.

The script runs in the foreground and is compatible with:
  nohup bash scripts/mraf/run_mraf_experiments.sh --case ... > driver.log 2>&1 &

It never invokes screen or nohup itself. Because the historical effective MDRC
loss scale is not proven by local evidence, MDRC cases refuse to start unless
MDRC_LOSS_SCALE is explicitly supplied.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 2
}

case_name=""
seed=1
num_epoch=60
batch_size=16
loss_names="sdm+mlm"
batch_size_explicit=false
loss_names_explicit=false
output_root="${OUTPUT_ROOT:-mraf_runs}"
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)
      [[ $# -ge 2 ]] || die "--case requires a value."
      case_name=$2
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || die "--seed requires a value."
      seed=$2
      shift 2
      ;;
    --num-epoch)
      [[ $# -ge 2 ]] || die "--num-epoch requires a value."
      num_epoch=$2
      shift 2
      ;;
    --batch-size)
      [[ $# -ge 2 ]] || die "--batch-size requires a value."
      batch_size=$2
      batch_size_explicit=true
      shift 2
      ;;
    --loss-names)
      [[ $# -ge 2 ]] || die "--loss-names requires a value."
      loss_names=$2
      loss_names_explicit=true
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || die "--output-root requires a value."
      output_root=$2
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

case "$case_name" in
  mraf_only|mdrc_mraf|mdrc_mraf_norel) ;;
  "") die "--case is required." ;;
  *) die "Unsupported --case: $case_name" ;;
esac

[[ "$seed" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer: $seed"
[[ "$num_epoch" =~ ^[1-9][0-9]*$ ]] || die \
  "--num-epoch must be a positive integer: $num_epoch"
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || die \
  "--batch-size must be a positive integer: $batch_size"
[[ -n "$loss_names" ]] || die "--loss-names must not be empty."
[[ -n "$output_root" ]] || die "--output-root must not be empty."

if (( num_epoch > 1 )); then
  if [[ "$batch_size_explicit" != "true" || "$loss_names_explicit" != "true" ]]; then
    die "Formal/long-run training requires verified --batch-size and --loss-names."
  fi
  training_protocol_verified=true
  protocol_source="operator_verified_explicit_cli"
elif [[ "$batch_size_explicit" == "true" && "$loss_names_explicit" == "true" ]]; then
  training_protocol_verified=true
  protocol_source="operator_verified_explicit_cli"
else
  training_protocol_verified=false
  protocol_source="unverified_smoke_defaults"
fi

[[ "$loss_names" != +* && "$loss_names" != *+ && "$loss_names" != *++* ]] || die \
  "--loss-names must be a non-empty '+'-separated task list: $loss_names"

IFS='+' read -r -a raw_loss_tasks <<<"$loss_names"
loss_tasks=()
id_loss_enabled=false
mlm_in_loss_names=false
for raw_task in "${raw_loss_tasks[@]}"; do
  task="${raw_task#"${raw_task%%[![:space:]]*}"}"
  task="${task%"${task##*[![:space:]]}"}"
  [[ -n "$task" ]] || die "--loss-names contains an empty task: $loss_names"
  case "$task" in
    sdm|id|mlm|cmpm|itc) ;;
    *) die "Unsupported task in --loss-names: $task (allowed: sdm, id, mlm, cmpm, itc)" ;;
  esac
  loss_tasks+=("$task")
  [[ "$task" == "id" ]] && id_loss_enabled=true
  [[ "$task" == "mlm" ]] && mlm_in_loss_names=true
done

loss_names="${loss_tasks[0]}"
for ((task_index = 1; task_index < ${#loss_tasks[@]}; task_index++)); do
  loss_names+="+${loss_tasks[$task_index]}"
done

mlm_cli_flag_enabled=true
mlm_loss_enabled=false
if [[ "$mlm_in_loss_names" == "true" && "$mlm_cli_flag_enabled" == "true" ]]; then
  mlm_loss_enabled=true
fi
mlm_alias_double_count_risk=false
if [[ "$mlm_loss_enabled" == "true" ]]; then
  mlm_alias_double_count_risk=true
fi

world_size="${WORLD_SIZE:-1}"
[[ "$world_size" =~ ^[1-9][0-9]*$ ]] || die \
  "WORLD_SIZE must be a positive integer when set: $world_size"
[[ "$world_size" == "1" ]] || die \
  "This launcher supports the audited single-process protocol only; got WORLD_SIZE=$world_size."

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"
cd "$repo_dir"

# A worktree created by Windows Git stores a Windows-form gitdir pointer.
# Convert that pointer to a Linux path when this script is checked from WSL.
git_worktree_root="$(cd "${repo_dir}/.." && pwd)"
git_dir_override=""
if ! git -C "$git_worktree_root" rev-parse --git-dir >/dev/null 2>&1; then
  git_pointer_file="${git_worktree_root}/.git"
  if [[ -f "$git_pointer_file" ]] && command -v wslpath >/dev/null 2>&1; then
    git_dir_pointer="$(sed -n 's/^gitdir: //p' "$git_pointer_file")"
    if [[ "$git_dir_pointer" =~ ^[A-Za-z]:[/\\] ]]; then
      git_dir_override="$(wslpath -u "$git_dir_pointer")"
    fi
  fi
fi

run_git() {
  if [[ -n "$git_dir_override" ]]; then
    GIT_DIR="$git_dir_override" GIT_WORK_TREE="$git_worktree_root" git "$@"
  else
    git -C "$git_worktree_root" "$@"
  fi
}

python_bin=${PYTHON_BIN:-python}
data_root=${DATA_ROOT:-datasets/TextReID}
degrade_aug_mode="${DEGRADE_AUG_MODE:-none}"
test_batch_size=512
gradient_accumulation_steps=1
learning_rate="1e-5"
warmup_epochs=5
eval_use_fused_mu=true
num_workers=8
[[ "$degrade_aug_mode" == "none" ]] || die \
  "MRAF reliability experiments require degrade_aug_mode=none; got $degrade_aug_mode."

timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="${case_name}_seed${seed}"
run_root="${output_root%/}/${timestamp}_${run_name}"

if [[ -e "$run_root" ]]; then
  die "Refusing to overwrite existing run directory: $run_root"
fi

git_head="$(run_git rev-parse HEAD)"
git_branch="$(run_git branch --show-current)"

common_args=(
  --name "$run_name"
  --output_dir "$run_root"
  --dataset_name RGBT-PEDES
  --root_dir "$data_root"
  --img_aug
  --MLM
  --loss_names "$loss_names"
  --batch_size "$batch_size"
  --test_batch_size "$test_batch_size"
  --num_workers "$num_workers"
  --num_epoch "$num_epoch"
  --warmup_epochs "$warmup_epochs"
  --lr "$learning_rate"
  --eval_use_fused_mu
  --log_period 100
  --eval_period 1
  --degrade_aug_mode "$degrade_aug_mode"
  --use_rgbt_add_fusion
  --mraf_enabled
  --mraf_hidden_dim 128
  --mraf_dropout 0.1
  --mraf_gate_temperature 1.0
  --mraf_target_temperature 0.1
  --mraf_rel_warmup_epoch 3
  --mraf_balance_lambda 0.0
)

case_args=()
effective_mdrc_loss_scale="NA"
mdrc_enabled=false
mraf_rel_lambda="0.01"
case "$case_name" in
  mraf_only)
    case_args+=(--mraf_rel_lambda "$mraf_rel_lambda")
    ;;
  mdrc_mraf)
    [[ -n "${MDRC_LOSS_SCALE:-}" ]] || die \
      "MDRC_LOSS_SCALE is required for case mdrc_mraf; verify it from historical formal MDRC evidence first."
    mdrc_enabled=true
    mraf_rel_lambda="0.01"
    effective_mdrc_loss_scale="$MDRC_LOSS_SCALE"
    case_args+=(
      --mdrc_enable
      --mdrc_prob 0.5
      --mdrc_lambda 0.05
      --mdrc_loss_scale "$MDRC_LOSS_SCALE"
      --mdrc_warmup_epoch 3
      --mraf_rel_lambda "$mraf_rel_lambda"
    )
    ;;
  mdrc_mraf_norel)
    [[ -n "${MDRC_LOSS_SCALE:-}" ]] || die \
      "MDRC_LOSS_SCALE is required for case mdrc_mraf_norel; verify it from historical formal MDRC evidence first."
    mdrc_enabled=true
    mraf_rel_lambda="0.0"
    effective_mdrc_loss_scale="$MDRC_LOSS_SCALE"
    case_args+=(
      --mdrc_enable
      --mdrc_prob 0.5
      --mdrc_lambda 0.05
      --mdrc_loss_scale "$MDRC_LOSS_SCALE"
      --mdrc_warmup_epoch 3
      --mraf_rel_lambda "$mraf_rel_lambda"
    )
    ;;
esac

command=(env "TRAIN_SEED=$seed" "$python_bin" -u train.py "${common_args[@]}" "${case_args[@]}")
printf -v command_text "%q " "${command[@]}"
command_text=${command_text% }

print_effective_config() {
  echo "case=$case_name"
  echo "seed=$seed"
  echo "num_epoch=$num_epoch"
  echo "batch_size=$batch_size"
  echo "effective_batch_size=$batch_size"
  echo "world_size=$world_size"
  echo "gradient_accumulation_steps=$gradient_accumulation_steps"
  echo "test_batch_size=$test_batch_size"
  echo "loss_names=$loss_names"
  echo "id_loss_enabled=$id_loss_enabled"
  echo "mlm_in_loss_names=$mlm_in_loss_names"
  echo "mlm_cli_flag_enabled=$mlm_cli_flag_enabled"
  echo "mlm_loss_enabled=$mlm_loss_enabled"
  echo "mlm_alias_double_count_risk=$mlm_alias_double_count_risk"
  echo "training_protocol_verified=$training_protocol_verified"
  echo "protocol_source=$protocol_source"
  echo "learning_rate=$learning_rate"
  echo "warmup_epochs=$warmup_epochs"
  echo "eval_use_fused_mu=$eval_use_fused_mu"
  echo "num_workers=$num_workers"
  echo "output_root=$output_root"
  echo "output_dir=$run_root"
  echo "git_branch=$git_branch"
  echo "git_head=$git_head"
  echo "data_root=$data_root"
  echo "mraf_enabled=true"
  echo "mdrc_enabled=$mdrc_enabled"
  echo "mraf_rel_lambda=$mraf_rel_lambda"
  echo "mdrc_loss_scale=$effective_mdrc_loss_scale"
  echo "effective_mdrc_loss_scale=$effective_mdrc_loss_scale"
  echo "degrade_aug_mode=$degrade_aug_mode"
}

print_protocol_warning() {
  if [[ "$mlm_alias_double_count_risk" == "true" ]]; then
    echo "WARNING: Current non-MUM aggregation may include both mlm_loss and loss_mlm aliases. Preserve historical behavior for comparability; do not change model/processor in this task." >&2
  fi
}

if [[ "$dry_run" == "true" ]]; then
  echo "dry_run=true"
  print_effective_config
  echo "training_command=$command_text"
  print_protocol_warning
  echo "Dry run complete: no output directory was created and Python training was not started."
  exit 0
fi

mkdir -p "$run_root"
printf "%s\n" "$command_text" >"${run_root}/command.txt"

{
  echo "dry_run=false"
  print_effective_config
  echo "started_at=$(date -Iseconds)"
  echo "training_command=$command_text"
} | tee "${run_root}/launch_metadata.txt"

print_protocol_warning
echo "effective_mdrc_loss_scale=$effective_mdrc_loss_scale"
"${command[@]}" 2>&1 | tee "${run_root}/train.log"
