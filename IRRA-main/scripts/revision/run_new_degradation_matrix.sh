#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Run the six evaluation-only degradation scenarios and save one score/log per case.

Usage:
  run_new_degradation_matrix.sh --repo PATH --result-root PATH \
    --config-file FILE --checkpoint FILE --label NAME [--degrade-seed N]

Required arguments:
  --repo PATH          IRRA-main worktree path
  --result-root PATH   Parent directory for new results
  --config-file FILE   Training configs.yaml used by test.py
  --checkpoint FILE    Checkpoint evaluated by test.py
  --label NAME         Unique Base/MDRC run label (no slash)

Optional arguments:
  --degrade-seed N     Evaluation degradation seed (default: 2026)
  -h, --help           Show this help

The script never uses nohup or screen. Existing score.pt files are skipped.
Logs are appended so interrupted cases can be rerun without erasing prior output.
EOF
}

repo=""
result_root=""
config_file=""
checkpoint=""
label=""
degrade_seed=2026

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo=${2:?"--repo requires a value"}; shift 2 ;;
    --result-root) result_root=${2:?"--result-root requires a value"}; shift 2 ;;
    --config-file) config_file=${2:?"--config-file requires a value"}; shift 2 ;;
    --checkpoint) checkpoint=${2:?"--checkpoint requires a value"}; shift 2 ;;
    --label) label=${2:?"--label requires a value"}; shift 2 ;;
    --degrade-seed) degrade_seed=${2:?"--degrade-seed requires a value"}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required_name in repo result_root config_file checkpoint label; do
  if [[ -z ${!required_name} ]]; then
    echo "Missing required argument: --${required_name//_/-}" >&2
    usage >&2
    exit 2
  fi
done

if [[ $label == */* || $label == *\\* || $label == "." || $label == ".." ]]; then
  echo "--label must be a single safe directory name: $label" >&2
  exit 2
fi
if [[ ! $degrade_seed =~ ^-?[0-9]+$ ]]; then
  echo "--degrade-seed must be an integer: $degrade_seed" >&2
  exit 2
fi

repo=$(cd "$repo" && pwd)
config_file=$(cd "$(dirname "$config_file")" && pwd)/$(basename "$config_file")
checkpoint=$(cd "$(dirname "$checkpoint")" && pwd)/$(basename "$checkpoint")
mkdir -p "$result_root"
result_root=$(cd "$result_root" && pwd)
run_root="$result_root/$label"
mkdir -p "$run_root"

[[ -f $repo/test.py ]] || { echo "test.py not found under --repo: $repo" >&2; exit 2; }
[[ -f $config_file ]] || { echo "Config file not found: $config_file" >&2; exit 2; }
[[ -f $checkpoint ]] || { echo "Checkpoint not found: $checkpoint" >&2; exit 2; }

python_bin=${PYTHON_BIN:-python}

run_case() {
  local mode=$1
  local strength=$2
  local case_dir="$run_root/${mode}_strength_${strength}"
  local score_file="$case_dir/score.pt"
  local log_file="$case_dir/eval.log"

  if [[ -f $score_file ]]; then
    echo "[SKIP] $mode strength=$strength score exists: $score_file"
    return 0
  fi

  mkdir -p "$case_dir"
  {
    echo
    echo "===== $(date -Iseconds) START mode=$mode strength=$strength seed=$degrade_seed ====="
  } >>"$log_file"

  echo "[RUN] $mode strength=$strength log=$log_file"
  if ! (
    cd "$repo"
    "$python_bin" -u test.py \
      --config_file "$config_file" \
      --test_ckpt_file "$checkpoint" \
      --eval_degrade_mode "$mode" \
      --eval_degrade_strength "$strength" \
      --eval_degrade_seed "$degrade_seed" \
      --save_eval_scores \
      --eval_scores_save_raw \
      --eval_scores_file "$score_file"
  ) >>"$log_file" 2>&1; then
    echo "[FAIL] $mode strength=$strength; tail of $log_file:" >&2
    tail -n 80 "$log_file" >&2
    exit 1
  fi
  echo "===== $(date -Iseconds) DONE mode=$mode strength=$strength =====" >>"$log_file"
  echo "[DONE] $mode strength=$strength score=$score_file"
}

run_case rgb_brightness 0.3
run_case ir_brightness 0.3
run_case rgb_lowres 0.25
run_case ir_lowres 0.25
run_case rgb_jpeg 20
run_case ir_jpeg 20
