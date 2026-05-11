#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PACK_ROOT="pack"
PACK_DIR="${PACK_ROOT}/uspc_v2a_attr_smoke_${TIMESTAMP}"
RUNS_DIR="${PACK_DIR}/runs"
LOGS_DIR="${PACK_DIR}/logs"
CMDS_DIR="${PACK_DIR}/cmds"
MASTER_LOG="${PACK_DIR}/master.log"
SUMMARY_TSV="${PACK_DIR}/results_summary.tsv"

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}" "${CMDS_DIR}"
printf '%s\n' "${PACK_DIR}" > "${PACK_ROOT}/LATEST_USPC_V2A_ATTR_SMOKE_PACK.txt"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[USPC_V2A_ATTR_SMOKE_START] PACK_DIR=${PACK_DIR}"
echo "[USPC_V2A_ATTR_SMOKE_START] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[USPC_V2A_ATTR_SMOKE_START] pwd=$(pwd)"

COMMON_ARGS=(
  --dataset_name RGBT-PEDES
  --root_dir datasets/TextReID
  --img_aug
  --MLM
  --loss_names "sdm+mlm"
  --batch_size 16
  --test_batch_size 512
  --num_workers 8
  --num_epoch 2
  --warmup_epochs 0
  --eval_period 1
  --log_period 20
  --output_dir "${RUNS_DIR}"
  --use_rgbt_add_fusion
  --use_global_uncertainty
  --use_batch_uncertainty
  --use_id_uncertainty
  --uncertainty_dist_type diag_gaussian
  --use_precision_fusion
  --uncertainty_loss_type w2
  --lambda_unc 0.09
  --lambda_tr 1.10
  --lambda_rt 0.90
  --lambda_tt 0.0
  --lambda_var_reg 0.0001
  --eval_use_fused_mu
  --use_proto_calib
  --proto_version v2
  --lambda_proto_self_v 0.05
  --lambda_proto_self_t 0.05
  --lambda_proto_cross_t2v 0.0
  --lambda_proto_cross_v2t 0.0
  --lambda_proto_align 0.0
  --proto_temp 0.07
  --proto_momentum 0.9
  --proto_warmup_epochs 0
  --save_metric_best_ckpt
)

printf 'run_name\tstatus\tattr_expected\tattr_loss_found\ttrain_exit\toutput_dir\tlog_file\tcmd_file\n' > "${SUMMARY_TSV}"

quote_cmd() {
  printf '%q ' "$@"
}

find_output_dir() {
  local run_name="$1"
  local dataset_dir="${RUNS_DIR}/RGBT-PEDES"
  if [[ ! -d "${dataset_dir}" ]]; then
    return 0
  fi
  find "${dataset_dir}" -maxdepth 1 -type d -name "*_${run_name}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

summarize_run() {
  local run_name="$1"
  local attr_expected="$2"
  local train_exit="$3"
  local run_log="$4"
  local cmd_file="$5"

  local output_dir
  output_dir="$(find_output_dir "${run_name}")"
  if [[ -z "${output_dir}" ]]; then
    output_dir="NA"
  fi

  local attr_loss_found="no"
  if grep -Eq 'loss_attr_aux|\[ATTR_AUX\]' "${run_log}"; then
    attr_loss_found="yes"
  fi

  local status="OK"
  if [[ "${train_exit}" -ne 0 ]]; then
    status="TRAIN_FAILED"
  elif [[ "${attr_expected}" == "yes" && "${attr_loss_found}" != "yes" ]]; then
    status="ATTR_LOSS_MISSING"
  elif [[ "${attr_expected}" == "no" && "${attr_loss_found}" == "yes" ]]; then
    status="ATTR_LOSS_UNEXPECTED"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_name}" "${status}" "${attr_expected}" "${attr_loss_found}" \
    "${train_exit}" "${output_dir}" "${run_log}" "${cmd_file}" >> "${SUMMARY_TSV}"

  echo "[USPC_V2A_ATTR_SMOKE_SUMMARY] run=${run_name} status=${status} attr_loss_found=${attr_loss_found} output_dir=${output_dir}"
  [[ "${status}" == "OK" ]]
}

run_experiment() {
  local run_name="$1"
  local attr_expected="$2"
  shift 2

  local run_log="${LOGS_DIR}/${run_name}.out"
  local cmd_file="${CMDS_DIR}/${run_name}.cmd"
  local cmd=(python train.py --name "${run_name}" "${COMMON_ARGS[@]}" "$@")

  quote_cmd "${cmd[@]}" > "${cmd_file}"
  echo
  echo "[USPC_V2A_ATTR_SMOKE_RUN_START] run=${run_name} attr_expected=${attr_expected} started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[USPC_V2A_ATTR_SMOKE_RUN_CMD] $(cat "${cmd_file}")"

  set +e
  "${cmd[@]}" 2>&1 | tee "${run_log}"
  local train_exit="${PIPESTATUS[0]}"
  set -e

  echo "[USPC_V2A_ATTR_SMOKE_RUN_END] run=${run_name} exit=${train_exit} ended_at=$(date '+%Y-%m-%d %H:%M:%S')"
  summarize_run "${run_name}" "${attr_expected}" "${train_exit}" "${run_log}" "${cmd_file}"
}

FAIL_COUNT=0

run_experiment "uspc_v2a_attr_default_off_2ep" "no" \
  || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "uspc_v2a_attr_on_l005_2ep" "yes" \
  --use_attr_aux \
  --lambda_attr_aux 0.05 \
  --attr_feature_source fused_det \
  --attr_warmup_epochs 0 \
  --debug_attr_aux \
  || FAIL_COUNT=$((FAIL_COUNT + 1))

echo
echo "[USPC_V2A_ATTR_SMOKE_SUMMARY_TSV] ${SUMMARY_TSV}"
echo "[USPC_V2A_ATTR_SMOKE_DONE] PACK_DIR=${PACK_DIR}"

if [[ "${FAIL_COUNT}" -ne 0 ]]; then
  echo "[USPC_V2A_ATTR_SMOKE_WARN] FAIL_COUNT=${FAIL_COUNT}"
  exit 1
fi
