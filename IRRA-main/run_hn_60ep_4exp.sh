#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
PACK_ROOT="pack"
PACK_DIR="${PACK_ROOT}/hn_60ep_${TS}"
RUNS_DIR="${PACK_DIR}/runs"
LOGS_DIR="${PACK_DIR}/logs"
CMDS_DIR="${PACK_DIR}/cmds"
MASTER_LOG="${PACK_DIR}/master.log"
SUMMARY_FILE="${PACK_DIR}/results_summary.tsv"
LATEST_FILE="${PACK_ROOT}/LATEST_HN_60EP_PACK.txt"

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}" "${CMDS_DIR}"
printf "%s\n" "${PACK_DIR}" > "${LATEST_FILE}"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[HN_60EP_START] pack_dir=${PACK_DIR}"
echo "[HN_60EP_START] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[HN_60EP_START] pwd=$(pwd)"

printf "exp_name\tstatus\tlog_file\tcmd_file\toutput_dir\tbest_R1\tbest_epoch\thn_loss_found\tlast_metric_line\n" > "${SUMMARY_FILE}"

COMMON_ARGS=(
  --dataset_name RGBT-PEDES
  --root_dir datasets/TextReID
  --img_aug
  --MLM
  --loss_names "sdm+mlm"
  --batch_size 16
  --test_batch_size 512
  --num_workers 8
  --num_epoch 60
  --warmup_epochs 1
  --log_period 100
  --eval_period 1
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
  --output_dir "${RUNS_DIR}"
)

quote_cmd() {
  printf "%q " "$@"
}

find_output_dir() {
  local run_name="$1"
  local dataset_dir="${RUNS_DIR}/RGBT-PEDES"
  if [[ ! -d "${dataset_dir}" ]]; then
    return 0
  fi
  find "${dataset_dir}" -maxdepth 1 -type d -name "*_${run_name}" -printf "%T@ %p\n" 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d" " -f2-
}

append_summary() {
  local exp_name="$1"
  local status="$2"
  local log_file="$3"
  local cmd_file="$4"

  local output_dir best_line best_r1 best_epoch hn_loss_found last_metric_line
  output_dir="$(find_output_dir "${exp_name}" || true)"
  [[ -n "${output_dir}" ]] || output_dir="NA"

  best_line="$(grep -E "best R1:" "${log_file}" | tail -n 1 || true)"
  best_r1="$(sed -nE "s/.*best R1:[[:space:]]*([0-9.]+)[[:space:]]*at epoch[[:space:]]*([0-9]+).*/\\1/p" <<< "${best_line}")"
  best_epoch="$(sed -nE "s/.*best R1:[[:space:]]*([0-9.]+)[[:space:]]*at epoch[[:space:]]*([0-9]+).*/\\2/p" <<< "${best_line}")"
  [[ -n "${best_r1}" ]] || best_r1="NA"
  [[ -n "${best_epoch}" ]] || best_epoch="NA"

  hn_loss_found="no"
  if grep -Eq "loss_hn|hn_active|hn_num_valid_queries|hn_mean_pos_sim|hn_margin_violation_rate" "${log_file}"; then
    hn_loss_found="yes"
  fi

  last_metric_line="$(grep -E "R1|Rank-1|mAP|best|loss_hn|hn_active" "${log_file}" | tail -n 1 | tr '\t' ' ' || true)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${exp_name}" "${status}" "${log_file}" "${cmd_file}" "${output_dir}" \
    "${best_r1}" "${best_epoch}" "${hn_loss_found}" "${last_metric_line}" \
    >> "${SUMMARY_FILE}"
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
    "$@"
  )

  echo
  echo "[HN_60EP_RUN_START] exp=${exp_name} started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  quote_cmd "${cmd[@]}" > "${cmd_file}"
  printf "\n" >> "${cmd_file}"
  echo "[HN_60EP_RUN_CMD] $(cat "${cmd_file}")"

  "${cmd[@]}" 2>&1 | tee "${log_file}"
  local train_status="${PIPESTATUS[0]}"
  local status="OK"
  if [[ "${train_status}" -ne 0 ]]; then
    status="TRAIN_FAILED_${train_status}"
    echo "[HN_60EP_RUN_WARN] exp=${exp_name} failed with status=${train_status}; continuing."
  else
    echo "[HN_60EP_RUN_DONE] exp=${exp_name} finished."
  fi

  append_summary "${exp_name}" "${status}" "${log_file}" "${cmd_file}"
  return "${train_status}"
}

FAIL_COUNT=0

run_exp "hn_t2i_l002_m005_k1_w5_60ep" \
  --use_hard_negative_loss \
  --lambda_hn 0.02 \
  --hn_margin 0.05 \
  --hn_topk 1 \
  --hn_warmup_epochs 5 \
  --hn_direction t2i || FAIL_COUNT=$((FAIL_COUNT + 1))

run_exp "hn_t2i_l003_m005_k3_w5_60ep" \
  --use_hard_negative_loss \
  --lambda_hn 0.03 \
  --hn_margin 0.05 \
  --hn_topk 3 \
  --hn_warmup_epochs 5 \
  --hn_direction t2i || FAIL_COUNT=$((FAIL_COUNT + 1))

run_exp "hn_both_l002_m003_k1_w5_60ep" \
  --use_hard_negative_loss \
  --lambda_hn 0.02 \
  --hn_margin 0.03 \
  --hn_topk 1 \
  --hn_warmup_epochs 5 \
  --hn_direction both || FAIL_COUNT=$((FAIL_COUNT + 1))

run_exp "hn_t2i_l001_m005_k1_w10_60ep" \
  --use_hard_negative_loss \
  --lambda_hn 0.01 \
  --hn_margin 0.05 \
  --hn_topk 1 \
  --hn_warmup_epochs 10 \
  --hn_direction t2i || FAIL_COUNT=$((FAIL_COUNT + 1))

echo
echo "[HN_60EP_SUMMARY] ${SUMMARY_FILE}"
echo "[HN_60EP_MASTER_LOG] ${MASTER_LOG}"
echo "[HN_60EP_DONE] pack_dir=${PACK_DIR}"

if [[ "${FAIL_COUNT}" -ne 0 ]]; then
  echo "[HN_60EP_WARN] fail_count=${FAIL_COUNT}"
  exit 1
fi
