#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ROOT_DIR=${ROOT_DIR:-/path/to/RGBT-PEDES}
BASE_CONFIG=${BASE_CONFIG:-""}
PSEUDO_JSON=${PSEUDO_JSON:-""}
CKPT=${CKPT:-""}
OUT_TAG=${OUT_TAG:-$(date +%Y%m%d_%H%M%S)}

TRAIN_ROOT_DIR="${ROOT_DIR}"
if [[ "$(basename "${ROOT_DIR}")" == "RGBT-PEDES" ]]; then
  if [[ -d "${ROOT_DIR}/RGB" || -d "${ROOT_DIR}/RGBT-PEDES-Final" || -d "${ROOT_DIR}/RGBT-PEDES" ]]; then
    TRAIN_ROOT_DIR="$(dirname "${ROOT_DIR}")"
  fi
fi

PACK_ROOT="pack"
PACK_DIR="${PACK_ROOT}/mga_rde_screening_${OUT_TAG}"
RUNS_DIR="${PACK_DIR}/runs"
LOGS_DIR="${PACK_DIR}/logs"
CMDS_DIR="${PACK_DIR}/cmds"
SCRIPTS_DIR="${PACK_DIR}/scripts"
MASTER_LOG="${PACK_DIR}/master.log"
SUMMARY_TSV="${PACK_DIR}/results_summary.tsv"
LATEST_FILE="${PACK_ROOT}/LATEST_MGA_RDE_SCREENING_PACK.txt"

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}" "${CMDS_DIR}" "${SCRIPTS_DIR}"
cp "${BASH_SOURCE[0]}" "${SCRIPTS_DIR}/run_mga_rde_screening_4exp_20ep.sh"
printf '%s\n' "${PACK_DIR}" > "${LATEST_FILE}"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[MGA_RDE_SCREENING_START] PACK_DIR=${PACK_DIR}"
echo "[MGA_RDE_SCREENING_START] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[MGA_RDE_SCREENING_START] pwd=$(pwd)"
echo "[MGA_RDE_SCREENING_CONFIG] ROOT_DIR=${ROOT_DIR}"
echo "[MGA_RDE_SCREENING_CONFIG] TRAIN_ROOT_DIR=${TRAIN_ROOT_DIR}"
echo "[MGA_RDE_SCREENING_CONFIG] BASE_CONFIG=${BASE_CONFIG}"
echo "[MGA_RDE_SCREENING_CONFIG] PSEUDO_JSON=${PSEUDO_JSON}"
echo "[MGA_RDE_SCREENING_CONFIG] CKPT=${CKPT}"
echo "[MGA_RDE_SCREENING_CONFIG] OUT_TAG=${OUT_TAG}"

if [[ -n "${BASE_CONFIG}" ]]; then
  if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "[MGA_RDE_SCREENING_ABORT] BASE_CONFIG does not exist: ${BASE_CONFIG}" >&2
    exit 2
  fi
  # BASE_CONFIG is a bash fragment. It may override COMMON_ARGS after the
  # default block below or define EXTRA_COMMON_ARGS / EXTRA_E0_ARGS arrays.
  # train.py currently has no --config_file parser, so a shell fragment is the
  # least invasive way to reuse a local best command without changing training.
fi

if [[ -z "${PSEUDO_JSON}" ]]; then
  echo "[MGA_RDE_SCREENING_WARN] PSEUDO_JSON is empty; E1/E2/E3 will auto-skip MGA pseudo fields."
elif [[ ! -f "${PSEUDO_JSON}" ]]; then
  echo "[MGA_RDE_SCREENING_WARN] PSEUDO_JSON does not exist: ${PSEUDO_JSON}; training will warn and continue."
fi

COMMON_ARGS=(
  --dataset_name RGBT-PEDES
  --root_dir "${TRAIN_ROOT_DIR}"
  --img_aug
  --MLM
  --loss_names "sdm+mlm"
  --batch_size 16
  --test_batch_size 512
  --num_workers 8
  --num_epoch 20
  --warmup_epochs 1
  --log_period 100
  --eval_period 1
  --output_dir "${RUNS_DIR}"
  --use_rgbt_add_fusion
  --use_global_uncertainty
  --uncertainty_dist_type diag_gaussian
  --use_precision_fusion
  --uncertainty_loss_type w2
  --lambda_unc 0.09
  --lambda_tr 1.10
  --lambda_rt 0.90
  --lambda_tt 0.0
  --lambda_var_reg 1e-4
  --eval_use_fused_mu
  --save_metric_best_ckpt
  --save_full_eval_state_ckpt
)

# The default COMMON_ARGS mirror the recent GU screening baseline used in this
# repo. If your current best includes prototype init, USPC, DCAlign, or a
# specific checkpoint schedule, put a bash fragment in BASE_CONFIG that overrides
# COMMON_ARGS or appends EXTRA_COMMON_ARGS / EXTRA_E0_ARGS.
EXTRA_COMMON_ARGS=()
EXTRA_E0_ARGS=()
EXTRA_E1_ARGS=()
EXTRA_E2_ARGS=()
EXTRA_E3_ARGS=()

if [[ -n "${BASE_CONFIG}" ]]; then
  # shellcheck source=/dev/null
  source "${BASE_CONFIG}"
fi

if [[ -n "${CKPT}" ]]; then
  if [[ ! -f "${CKPT}" ]]; then
    echo "[MGA_RDE_SCREENING_WARN] CKPT does not exist: ${CKPT}; train.py will receive it only if the path is valid."
  else
    EXTRA_COMMON_ARGS+=(--load_ckpt_file "${CKPT}")
  fi
fi

printf 'exp_name\tstatus\tlog_file\tcmd_file\toutput_dir\tbest_R1\tbest_epoch\tR1\tR5\tR10\tmAP\tmINP\tlast_metric_line\n' > "${SUMMARY_TSV}"

quote_cmd() {
  printf '%q ' "$@"
}

find_output_dir() {
  local exp_name="$1"
  local dataset_dir="${RUNS_DIR}/RGBT-PEDES"
  if [[ ! -d "${dataset_dir}" ]]; then
    return 0
  fi
  find "${dataset_dir}" -maxdepth 1 -type d -name "*_${exp_name}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

extract_field() {
  local line="$1"
  local key="$2"
  sed -nE "s/.*${key}=([0-9.]+).*/\\1/p" <<< "${line}"
}

append_summary() {
  local exp_name="$1"
  local status="$2"
  local log_file="$3"
  local cmd_file="$4"

  local output_dir best_line best_r1 best_epoch metric_line r1 r5 r10 map minp
  output_dir="$(find_output_dir "${exp_name}" || true)"
  [[ -n "${output_dir}" ]] || output_dir="NA"

  best_line="$(grep -E 'best R1:' "${log_file}" | tail -n 1 || true)"
  best_r1="$(sed -nE 's/.*best R1:[[:space:]]*([0-9.]+)[[:space:]]*at epoch[[:space:]]*([0-9]+).*/\1/p' <<< "${best_line}")"
  best_epoch="$(sed -nE 's/.*best R1:[[:space:]]*([0-9.]+)[[:space:]]*at epoch[[:space:]]*([0-9]+).*/\2/p' <<< "${best_line}")"
  [[ -n "${best_r1}" ]] || best_r1="NA"
  [[ -n "${best_epoch}" ]] || best_epoch="NA"

  metric_line="$(
    grep -E '\[METRIC_CKPT_SAVE\]|\[FULL_STATE_CKPT_SAVE\]|\[IMMEDIATE_REEVAL|R1=|R5=|R10=|mAP=|mINP=|best R1:|Validation Results' "${log_file}" \
      | tail -n 1 \
      | tr '\t' ' ' || true
  )"
  r1="$(extract_field "${metric_line}" "R1")"
  r5="$(extract_field "${metric_line}" "R5")"
  r10="$(extract_field "${metric_line}" "R10")"
  map="$(extract_field "${metric_line}" "mAP")"
  minp="$(extract_field "${metric_line}" "mINP")"
  [[ -n "${r1}" ]] || r1="${best_r1}"
  [[ -n "${r5}" ]] || r5="NA"
  [[ -n "${r10}" ]] || r10="NA"
  [[ -n "${map}" ]] || map="NA"
  [[ -n "${minp}" ]] || minp="NA"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${exp_name}" "${status}" "${log_file}" "${cmd_file}" "${output_dir}" \
    "${best_r1}" "${best_epoch}" "${r1}" "${r5}" "${r10}" "${map}" "${minp}" \
    "${metric_line}" >> "${SUMMARY_TSV}"

  echo "[MGA_RDE_SCREENING_SUMMARY] exp=${exp_name} status=${status} best_R1=${best_r1} best_epoch=${best_epoch} line=${metric_line}"
}

run_exp() {
  local exp_name="$1"
  shift

  local log_file="${LOGS_DIR}/${exp_name}.log"
  local cmd_file="${CMDS_DIR}/${exp_name}.cmd"
  local cmd=(
    python train.py
    --name "${exp_name}"
    "${COMMON_ARGS[@]}"
    "${EXTRA_COMMON_ARGS[@]}"
    "$@"
  )

  echo
  echo "[MGA_RDE_SCREENING_RUN_START] exp=${exp_name} started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  quote_cmd "${cmd[@]}" > "${cmd_file}"
  printf '\n' >> "${cmd_file}"
  echo "[MGA_RDE_SCREENING_RUN_CMD] $(cat "${cmd_file}")"

  set +e
  "${cmd[@]}" 2>&1 | tee "${log_file}"
  local train_status="${PIPESTATUS[0]}"
  set -e

  local status="OK"
  if [[ "${train_status}" -ne 0 ]]; then
    status="TRAIN_FAILED_${train_status}"
    echo "[MGA_RDE_SCREENING_RUN_WARN] exp=${exp_name} failed status=${train_status}; continuing."
  else
    echo "[MGA_RDE_SCREENING_RUN_DONE] exp=${exp_name} finished."
  fi

  append_summary "${exp_name}" "${status}" "${log_file}" "${cmd_file}"
  return "${train_status}"
}

FAIL_COUNT=0

run_exp "E0_control_current_best_20ep" \
  "${EXTRA_E0_ARGS[@]}" \
  || FAIL_COUNT=$((FAIL_COUNT + 1))

run_exp "E1_mga_pseudo_only_20ep" \
  --use_mga_pseudo \
  --mga_pseudo_json "${PSEUDO_JSON}" \
  --lambda_mga_pseudo_sdm 0.30 \
  "${EXTRA_E1_ARGS[@]}" \
  || FAIL_COUNT=$((FAIL_COUNT + 1))

run_exp "E2_mga_attr_align_20ep" \
  --use_mga_pseudo \
  --mga_pseudo_json "${PSEUDO_JSON}" \
  --use_mga_attr_align \
  --lambda_mga_pseudo_sdm 0.30 \
  --lambda_mga_rgb_color 0.10 \
  --lambda_mga_t_shape 0.10 \
  "${EXTRA_E2_ARGS[@]}" \
  || FAIL_COUNT=$((FAIL_COUNT + 1))

run_exp "E3_mga_attr_rde_20ep" \
  --use_mga_pseudo \
  --mga_pseudo_json "${PSEUDO_JSON}" \
  --use_mga_attr_align \
  --use_rde_robust \
  --rde_warmup_epochs 10 \
  --rde_conf_source hybrid \
  --lambda_mga_pseudo_sdm 0.30 \
  --lambda_mga_rgb_color 0.10 \
  --lambda_mga_t_shape 0.10 \
  --lambda_rde_consensus 0.20 \
  --use_confidence_hard_negative \
  --conf_hn_min_weight 0.50 \
  "${EXTRA_E3_ARGS[@]}" \
  || FAIL_COUNT=$((FAIL_COUNT + 1))

echo
echo "[MGA_RDE_SCREENING_RESULTS]"
cat "${SUMMARY_TSV}"
echo
echo "[MGA_RDE_SCREENING_METRIC_GREP]"
grep -E '\[METRIC_CKPT_SAVE\]|\[FULL_STATE_CKPT_SAVE\]|R1=|R5=|R10=|mAP=|mINP=|best R1:' "${LOGS_DIR}"/*.log || true
echo
echo "[MGA_RDE_SCREENING_SUMMARY_TSV] ${SUMMARY_TSV}"
echo "[MGA_RDE_SCREENING_MASTER_LOG] ${MASTER_LOG}"
echo "[MGA_RDE_SCREENING_DONE] PACK_DIR=${PACK_DIR}"

if [[ "${FAIL_COUNT}" -ne 0 ]]; then
  echo "[MGA_RDE_SCREENING_WARN] FAIL_COUNT=${FAIL_COUNT}"
  exit 1
fi
