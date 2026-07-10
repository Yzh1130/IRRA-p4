#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Re-run selected legacy evaluation degradations for later element-wise comparison.

Usage:
  run_old_degradation_regression.sh --repo PATH --result-root PATH \
    --config-file FILE --checkpoint FILE --label NAME

Required arguments:
  --repo PATH          IRRA-main worktree path
  --result-root PATH   Parent directory for new regression results
  --config-file FILE   Training configs.yaml used by test.py
  --checkpoint FILE    Checkpoint evaluated by test.py
  --label NAME         Unique baseline label (no slash)
  -h, --help           Show this help

The degradation seed is fixed to 2026. The script saves scores but does not
compare them. Existing score.pt files are skipped and logs are appended.
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
  local case_name=$1
  local mode=$2
  local strength=$3
  local blur_kernel=$4
  local case_dir="$run_root/$case_name"
  local score_file="$case_dir/score.pt"
  local log_file="$case_dir/eval.log"

  if [[ -f $score_file ]]; then
    echo "[SKIP] $case_name score exists: $score_file"
    return 0
  fi

  mkdir -p "$case_dir"
  {
    echo
    echo "===== $(date -Iseconds) START case=$case_name mode=$mode ====="
  } >>"$log_file"

  echo "[RUN] $case_name log=$log_file"
  if ! (
    cd "$repo"
    "$python_bin" -u test.py \
      --config_file "$config_file" \
      --test_ckpt_file "$checkpoint" \
      --eval_degrade_mode "$mode" \
      --eval_degrade_strength "$strength" \
      --eval_degrade_blur_kernel "$blur_kernel" \
      --eval_degrade_seed "$degrade_seed" \
      --save_eval_scores \
      --eval_scores_save_raw \
      --eval_scores_file "$score_file"
  ) >>"$log_file" 2>&1; then
    echo "[FAIL] $case_name; tail of $log_file:" >&2
    tail -n 80 "$log_file" >&2
    exit 1
  fi
  echo "===== $(date -Iseconds) DONE case=$case_name =====" >>"$log_file"
  echo "[DONE] $case_name score=$score_file"
}

run_case none none 1.0 7
run_case rgb_missing rgb_missing 1.0 7
run_case rgb_noise_0.05 rgb_noise 0.05 7
run_case rgb_blur_k7 rgb_blur 1.0 7
