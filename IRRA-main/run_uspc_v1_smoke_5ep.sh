#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PACK_ROOT="pack"
PACK_DIR="${PACK_ROOT}/03_USPC_V1_SMOKE_${TIMESTAMP}"
RUNS_DIR="${PACK_DIR}/runs"
TABLES_DIR="${PACK_DIR}/tables"
LOGS_DIR="${PACK_DIR}/logs"
SCRIPTS_DIR="${PACK_DIR}/scripts"
MASTER_LOG="${PACK_DIR}/master.log"
SUMMARY_MD="${TABLES_DIR}/uspc_v1_smoke_summary.md"
SUMMARY_TSV="${TABLES_DIR}/uspc_v1_smoke_summary.tsv"

mkdir -p "${RUNS_DIR}" "${TABLES_DIR}" "${LOGS_DIR}" "${SCRIPTS_DIR}"
cp "${BASH_SOURCE[0]}" "${SCRIPTS_DIR}/run_uspc_v1_smoke_5ep.sh"
printf '%s\n' "${PACK_DIR}" > "${PACK_ROOT}/LATEST_USPC_V1_SMOKE_PACK.txt"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[USPC_SMOKE_START] PACK_DIR=${PACK_DIR}"
echo "[USPC_SMOKE_START] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[USPC_SMOKE_START] pwd=$(pwd)"

COMMON_ARGS=(
  --dataset_name RGBT-PEDES
  --root_dir datasets/TextReID
  --img_aug
  --MLM
  --loss_names "sdm+mlm"
  --batch_size 16
  --test_batch_size 512
  --num_workers 8
  --num_epoch 5
  --eval_period 1
  --log_period 200
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
  --save_metric_best_ckpt
  --save_full_eval_state_ckpt
)

printf '| run_name | status | uspc_loss_expected | uspc_loss_found | loss_numeric_check | best_R1 | best_epoch | t2i_R1 | t2i_R5 | t2i_R10 | t2i_mAP | t2i_mINP | output_dir | config | best_ckpt | t2i_table | loss_samples |\n' > "${SUMMARY_MD}"
printf '|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|\n' >> "${SUMMARY_MD}"
printf 'run_name\tstatus\tuspc_loss_expected\tuspc_loss_found\tloss_numeric_check\tbest_R1\tbest_epoch\tt2i_R1\tt2i_R5\tt2i_R10\tt2i_mAP\tt2i_mINP\toutput_dir\tconfig\tbest_ckpt\tt2i_table\tloss_samples\n' > "${SUMMARY_TSV}"

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

extract_table_field() {
  local line="$1"
  local field_index="$2"
  if [[ -z "${line}" ]]; then
    printf 'NA'
    return 0
  fi
  awk -F'|' -v idx="${field_index}" '{
    gsub(/^[ \t]+|[ \t]+$/, "", $idx);
    print $idx;
  }' <<< "${line}"
}

append_summary() {
  local run_name="$1"
  local status="$2"
  local expect_uspc_loss="$3"
  local uspc_loss_found="$4"
  local loss_numeric_check="$5"
  local best_r1="$6"
  local best_epoch="$7"
  local t2i_r1="$8"
  local t2i_r5="$9"
  local t2i_r10="${10}"
  local t2i_map="${11}"
  local t2i_minp="${12}"
  local output_dir="${13}"
  local config_path="${14}"
  local best_ckpt="${15}"
  local t2i_table_file="${16}"
  local loss_samples_file="${17}"

  printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | `%s` | `%s` | `%s` | `%s` | `%s` |\n' \
    "${run_name}" "${status}" "${expect_uspc_loss}" "${uspc_loss_found}" "${loss_numeric_check}" \
    "${best_r1}" "${best_epoch}" "${t2i_r1}" "${t2i_r5}" "${t2i_r10}" "${t2i_map}" "${t2i_minp}" \
    "${output_dir}" "${config_path}" "${best_ckpt}" "${t2i_table_file}" "${loss_samples_file}" >> "${SUMMARY_MD}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_name}" "${status}" "${expect_uspc_loss}" "${uspc_loss_found}" "${loss_numeric_check}" \
    "${best_r1}" "${best_epoch}" "${t2i_r1}" "${t2i_r5}" "${t2i_r10}" "${t2i_map}" "${t2i_minp}" \
    "${output_dir}" "${config_path}" "${best_ckpt}" "${t2i_table_file}" "${loss_samples_file}" >> "${SUMMARY_TSV}"
}

summarize_run() {
  local run_name="$1"
  local expect_uspc_loss="$2"
  local train_status="$3"
  local run_log="$4"

  local output_dir
  output_dir="$(find_output_dir "${run_name}" || true)"
  local config_path="NA"
  local best_ckpt="NA"
  if [[ -n "${output_dir}" && -f "${output_dir}/configs.yaml" ]]; then
    config_path="${output_dir}/configs.yaml"
  fi
  if [[ -n "${output_dir}" && -f "${output_dir}/best.pth" ]]; then
    best_ckpt="${output_dir}/best.pth"
  fi
  if [[ -n "${output_dir}" && -d "${output_dir}/metric_ckpts" ]]; then
    local metric_best
    metric_best="$(find "${output_dir}/metric_ckpts" -type f -name '*best*.pth' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2- || true)"
    if [[ -n "${metric_best}" ]]; then
      best_ckpt="${metric_best}"
    fi
  fi

  local best_line best_r1 best_epoch
  best_line="$(grep -E 'best R1:' "${run_log}" | tail -n 1 || true)"
  best_r1="$(sed -nE 's/.*best R1:[[:space:]]*([0-9.]+)[[:space:]]*at epoch[[:space:]]*([0-9]+).*/\1/p' <<< "${best_line}")"
  best_epoch="$(sed -nE 's/.*best R1:[[:space:]]*([0-9.]+)[[:space:]]*at epoch[[:space:]]*([0-9]+).*/\2/p' <<< "${best_line}")"
  [[ -n "${best_r1}" ]] || best_r1="NA"
  [[ -n "${best_epoch}" ]] || best_epoch="NA"

  local t2i_line t2i_table_file
  t2i_table_file="${TABLES_DIR}/${run_name}_last_t2i.txt"
  t2i_line="$(grep -E '\|[[:space:]]*t2i[[:space:]]*\|' "${run_log}" | tail -n 1 || true)"
  if [[ -n "${t2i_line}" ]]; then
    local t2i_line_no t2i_start_line t2i_end_line
    t2i_line_no="$(grep -nE '\|[[:space:]]*t2i[[:space:]]*\|' "${run_log}" | tail -n 1 | cut -d: -f1 || true)"
    t2i_start_line=$((t2i_line_no > 3 ? t2i_line_no - 3 : 1))
    t2i_end_line=$((t2i_line_no + 1))
    sed -n "${t2i_start_line},${t2i_end_line}p" "${run_log}" > "${t2i_table_file}"
  else
    printf '[NO_T2I_TABLE_FOUND]\n' > "${t2i_table_file}"
  fi

  local t2i_r1 t2i_r5 t2i_r10 t2i_map t2i_minp
  t2i_r1="$(extract_table_field "${t2i_line}" 3)"
  t2i_r5="$(extract_table_field "${t2i_line}" 4)"
  t2i_r10="$(extract_table_field "${t2i_line}" 5)"
  t2i_map="$(extract_table_field "${t2i_line}" 6)"
  t2i_minp="$(extract_table_field "${t2i_line}" 7)"

  local loss_samples_file
  loss_samples_file="${LOGS_DIR}/${run_name}.loss_uspc_samples.txt"
  if awk '/loss_uspc|uspc_active|\[USPC\]\[LOSS_STATUS\]/ {
      print;
      count += 1;
      if (count >= 12) {
        exit 0;
      }
    }
    END {
      exit(count > 0 ? 0 : 1);
    }' "${run_log}" > "${loss_samples_file}"; then
    :
  else
    printf '[NO_USPC_LOSS_LOG_LINES]\n' > "${loss_samples_file}"
  fi

  local uspc_loss_found="no"
  if grep -Eq 'loss_uspc|uspc_active|\[USPC\]\[LOSS_STATUS\]' "${run_log}"; then
    uspc_loss_found="yes"
  fi

  local loss_numeric_check="NA"
  if [[ "${uspc_loss_found}" == "yes" ]]; then
    if grep -Eiq 'loss_uspc[^,]*:[[:space:]]*(-?nan|nan|-?inf|inf)' "${run_log}"; then
      loss_numeric_check="FAIL_NAN_OR_INF"
    else
      loss_numeric_check="OK"
    fi
  fi

  local status="OK"
  if [[ "${train_status}" -ne 0 ]]; then
    status="TRAIN_FAILED_${train_status}"
  fi
  if [[ "${expect_uspc_loss}" == "yes" && "${uspc_loss_found}" != "yes" ]]; then
    status="${status}_USPC_LOSS_MISSING"
  fi
  if [[ "${expect_uspc_loss}" == "no" && "${uspc_loss_found}" == "yes" ]]; then
    status="${status}_USPC_LOSS_UNEXPECTED"
  fi
  if [[ "${loss_numeric_check}" == "FAIL_NAN_OR_INF" ]]; then
    status="${status}_USPC_LOSS_NAN_INF"
  fi

  append_summary \
    "${run_name}" "${status}" "${expect_uspc_loss}" "${uspc_loss_found}" "${loss_numeric_check}" \
    "${best_r1}" "${best_epoch}" "${t2i_r1}" "${t2i_r5}" "${t2i_r10}" "${t2i_map}" "${t2i_minp}" \
    "${output_dir:-NA}" "${config_path}" "${best_ckpt}" "${t2i_table_file}" "${loss_samples_file}"

  echo "[USPC_SMOKE_SUMMARY] run=${run_name} status=${status} best_R1=${best_r1} best_epoch=${best_epoch} loss_found=${uspc_loss_found} output_dir=${output_dir:-NA}"
  if [[ "${status}" == "OK" ]]; then
    return 0
  fi
  return 1
}

run_experiment() {
  local run_name="$1"
  local expect_uspc_loss="$2"
  shift 2

  local run_log="${LOGS_DIR}/${run_name}.out"
  echo
  echo "[USPC_SMOKE_RUN_START] run=${run_name} expect_uspc_loss=${expect_uspc_loss} started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[USPC_SMOKE_RUN_CMD] python train.py --name ${run_name} ${COMMON_ARGS[*]} $*"

  set +e
  python train.py \
    --name "${run_name}" \
    "${COMMON_ARGS[@]}" \
    "$@" 2>&1 | tee "${run_log}"
  local train_status="${PIPESTATUS[0]}"
  set -e

  echo "[USPC_SMOKE_RUN_END] run=${run_name} status=${train_status} ended_at=$(date '+%Y-%m-%d %H:%M:%S')"
  local summary_status=0
  summarize_run "${run_name}" "${expect_uspc_loss}" "${train_status}" "${run_log}" || summary_status=$?
  return "${summary_status}"
}

FAIL_COUNT=0

run_experiment "uspc_off_control_5ep" "no" || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "uspc_l002_5ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.02 \
  --uspc_uncertainty_weighting \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "uspc_l005_5ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.05 \
  --uspc_uncertainty_weighting \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "uspc_l008_5ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.08 \
  --uspc_uncertainty_weighting \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "uspc_l010_5ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.10 \
  --uspc_uncertainty_weighting \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

echo
echo "[USPC_SMOKE_TABLE_MD] ${SUMMARY_MD}"
echo "[USPC_SMOKE_TABLE_TSV] ${SUMMARY_TSV}"
echo "[USPC_SMOKE_DONE] PACK_DIR=${PACK_DIR}"

if [[ "${FAIL_COUNT}" -ne 0 ]]; then
  echo "[USPC_SMOKE_WARN] FAIL_COUNT=${FAIL_COUNT}"
  exit 1
fi
