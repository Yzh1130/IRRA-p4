#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PACK_ROOT="pack"
PACK_DIR="${PACK_ROOT}/uspc_v2b0_slot_local_smoke_${TIMESTAMP}"
RUNS_DIR="${PACK_DIR}/runs"
LOGS_DIR="${PACK_DIR}/logs"
CMDS_DIR="${PACK_DIR}/cmds"
MASTER_LOG="${PACK_DIR}/master.log"
SUMMARY_TSV="${PACK_DIR}/results_summary.tsv"

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}" "${CMDS_DIR}"
printf '%s\n' "${PACK_DIR}" > "${PACK_ROOT}/LATEST_USPC_V2B0_SLOT_LOCAL_SMOKE_PACK.txt"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[USPC_V2B0_SLOT_LOCAL_SMOKE_START] PACK_DIR=${PACK_DIR}"
echo "[USPC_V2B0_SLOT_LOCAL_SMOKE_START] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[USPC_V2B0_SLOT_LOCAL_SMOKE_START] pwd=$(pwd)"

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

printf 'run_name\tstatus\tslot_expected\tslot_loss_found\ttrain_exit\tbest_r1\toutput_dir\tlog_file\tcmd_file\n' > "${SUMMARY_TSV}"

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
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

extract_best_r1() {
  local run_log="$1"
  local best_r1
  best_r1="$(grep -Eio 'best R1:?[[:space:]]*[0-9.]+' "${run_log}" 2>/dev/null \
    | tail -n 1 \
    | grep -Eo '[0-9.]+$' || true)"
  if [[ -z "${best_r1}" ]]; then
    best_r1="NA"
  fi
  printf '%s\n' "${best_r1}"
}

summarize_run() {
  local run_name="$1"
  local slot_expected="$2"
  local train_exit="$3"
  local run_log="$4"
  local cmd_file="$5"

  local output_dir
  output_dir="$(find_output_dir "${run_name}")"
  if [[ -z "${output_dir}" ]]; then
    output_dir="NA"
  fi

  local slot_loss_found="no"
  if grep -Eq 'loss_uspc_slot_local|\[USPC_SLOT_LOCAL\]' "${run_log}"; then
    slot_loss_found="yes"
  fi

  local best_r1
  best_r1="$(extract_best_r1 "${run_log}")"

  local status="OK"
  if [[ "${train_exit}" -ne 0 ]]; then
    status="TRAIN_FAILED"
  elif [[ "${slot_expected}" == "yes" && "${slot_loss_found}" != "yes" ]]; then
    status="SLOT_LOSS_MISSING"
  elif [[ "${slot_expected}" == "no" && "${slot_loss_found}" == "yes" ]]; then
    status="SLOT_LOSS_UNEXPECTED"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_name}" "${status}" "${slot_expected}" "${slot_loss_found}" \
    "${train_exit}" "${best_r1}" "${output_dir}" "${run_log}" "${cmd_file}" >> "${SUMMARY_TSV}"

  echo "[USPC_V2B0_SLOT_LOCAL_SMOKE_SUMMARY] run=${run_name} status=${status} slot_loss_found=${slot_loss_found} best_r1=${best_r1} output_dir=${output_dir}"
  [[ "${status}" == "OK" ]]
}

run_experiment() {
  local run_name="$1"
  local slot_expected="$2"
  shift 2

  local run_log="${LOGS_DIR}/${run_name}.out"
  local cmd_file="${CMDS_DIR}/${run_name}.cmd"
  local cmd=(python train.py --name "${run_name}" "${COMMON_ARGS[@]}" "$@")

  quote_cmd "${cmd[@]}" > "${cmd_file}"
  echo
  echo "[USPC_V2B0_SLOT_LOCAL_SMOKE_RUN_START] run=${run_name} slot_expected=${slot_expected} started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[USPC_V2B0_SLOT_LOCAL_SMOKE_RUN_CMD] $(cat "${cmd_file}")"

  set +e
  "${cmd[@]}" 2>&1 | tee "${run_log}"
  local train_exit="${PIPESTATUS[0]}"
  set -e

  echo "[USPC_V2B0_SLOT_LOCAL_SMOKE_RUN_END] run=${run_name} exit=${train_exit} ended_at=$(date '+%Y-%m-%d %H:%M:%S')"
  summarize_run "${run_name}" "${slot_expected}" "${train_exit}" "${run_log}" "${cmd_file}"
}

FAIL_COUNT=0

run_experiment "uspc_v2b0_default_off_2ep" "no" \
  || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "uspc_v2b0_rgb_l0005_2ep" "yes" \
  --use_uspc_slot_local \
  --lambda_uspc_slot_local 0.005 \
  --uspc_slot_patch_source rgb \
  --uspc_slot_region_mode fixed_stripe \
  --uspc_slot_loss_type ce \
  --uspc_slot_warmup_epochs 0 \
  --uspc_slot_active_until_epoch -1 \
  --debug_uspc_slot_local \
  || FAIL_COUNT=$((FAIL_COUNT + 1))

echo
echo "[USPC_V2B0_SLOT_LOCAL_SMOKE_SUMMARY_TSV] ${SUMMARY_TSV}"
cat "${SUMMARY_TSV}"
echo "[USPC_V2B0_SLOT_LOCAL_SMOKE_DONE] PACK_DIR=${PACK_DIR}"

if [[ "${FAIL_COUNT}" -ne 0 ]]; then
  echo "[USPC_V2B0_SLOT_LOCAL_SMOKE_WARN] FAIL_COUNT=${FAIL_COUNT}"
  exit 1
fi
