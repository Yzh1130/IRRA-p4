#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PACK_ROOT="pack"
PACK_DIR="${PACK_ROOT}/04_USPC_V1_ABLATION_60EP_${TIMESTAMP}"
RUNS_DIR="${PACK_DIR}/runs"
TABLES_DIR="${PACK_DIR}/tables"
LOGS_DIR="${PACK_DIR}/logs"
SCRIPTS_DIR="${PACK_DIR}/scripts"
MASTER_LOG="${PACK_DIR}/master.log"
SUMMARY_MD="${TABLES_DIR}/uspc_v1_60ep_summary.md"
SUMMARY_TSV="${TABLES_DIR}/uspc_v1_60ep_summary.tsv"

mkdir -p "${RUNS_DIR}" "${TABLES_DIR}" "${LOGS_DIR}" "${SCRIPTS_DIR}"
cp "${BASH_SOURCE[0]}" "${SCRIPTS_DIR}/run_uspc_v1_ablation_60ep.sh"
printf '%s\n' "${PACK_DIR}" > "${PACK_ROOT}/LATEST_USPC_V1_ABLATION_60EP_PACK.txt"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[USPC_60EP_START] PACK_DIR=${PACK_DIR}"
echo "[USPC_60EP_START] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[USPC_60EP_START] pwd=$(pwd)"

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

printf '| run_name | status | uspc_loss_expected | uspc_loss_found | loss_numeric_check | error_found | best_R1 | best_epoch | t2i_R1 | t2i_R5 | t2i_R10 | t2i_mAP | t2i_mINP | output_dir | config | best_ckpt | full_best_ckpt | t2i_table | loss_samples | error_grep |\n' > "${SUMMARY_MD}"
printf '|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|\n' >> "${SUMMARY_MD}"
printf 'run_name\tstatus\tuspc_loss_expected\tuspc_loss_found\tloss_numeric_check\terror_found\tbest_R1\tbest_epoch\tt2i_R1\tt2i_R5\tt2i_R10\tt2i_mAP\tt2i_mINP\toutput_dir\tconfig\tbest_ckpt\tfull_best_ckpt\tt2i_table\tloss_samples\terror_grep\n' > "${SUMMARY_TSV}"

quote_cmd() {
  printf '%q ' "$@"
}

command_has_warmup_epochs_1() {
  local args=("$@")
  local i
  for ((i = 0; i < ${#args[@]}; i += 1)); do
    if [[ "${args[$i]}" == "--warmup_epochs=1" ]]; then
      return 0
    fi
    if [[ "${args[$i]}" == "--warmup_epochs" ]]; then
      if (( i + 1 < ${#args[@]} )) && [[ "${args[$((i + 1))]}" == "1" ]]; then
        return 0
      fi
    fi
  done
  return 1
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

newest_file_or_na() {
  local search_dir="$1"
  local name_glob="$2"
  if [[ ! -d "${search_dir}" ]]; then
    printf 'NA'
    return 0
  fi
  local found
  found="$(find "${search_dir}" -type f -name "${name_glob}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2- || true)"
  if [[ -n "${found}" ]]; then
    printf '%s' "${found}"
  else
    printf 'NA'
  fi
}

append_summary() {
  local run_name="$1"
  local status="$2"
  local expect_uspc_loss="$3"
  local uspc_loss_found="$4"
  local loss_numeric_check="$5"
  local error_found="$6"
  local best_r1="$7"
  local best_epoch="$8"
  local t2i_r1="$9"
  local t2i_r5="${10}"
  local t2i_r10="${11}"
  local t2i_map="${12}"
  local t2i_minp="${13}"
  local output_dir="${14}"
  local config_path="${15}"
  local best_ckpt="${16}"
  local full_best_ckpt="${17}"
  local t2i_table_file="${18}"
  local loss_samples_file="${19}"
  local error_grep_file="${20}"

  printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` | `%s` |\n' \
    "${run_name}" "${status}" "${expect_uspc_loss}" "${uspc_loss_found}" "${loss_numeric_check}" "${error_found}" \
    "${best_r1}" "${best_epoch}" "${t2i_r1}" "${t2i_r5}" "${t2i_r10}" "${t2i_map}" "${t2i_minp}" \
    "${output_dir}" "${config_path}" "${best_ckpt}" "${full_best_ckpt}" "${t2i_table_file}" "${loss_samples_file}" "${error_grep_file}" >> "${SUMMARY_MD}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_name}" "${status}" "${expect_uspc_loss}" "${uspc_loss_found}" "${loss_numeric_check}" "${error_found}" \
    "${best_r1}" "${best_epoch}" "${t2i_r1}" "${t2i_r5}" "${t2i_r10}" "${t2i_map}" "${t2i_minp}" \
    "${output_dir}" "${config_path}" "${best_ckpt}" "${full_best_ckpt}" "${t2i_table_file}" "${loss_samples_file}" "${error_grep_file}" >> "${SUMMARY_TSV}"
}

summarize_run() {
  local run_name="$1"
  local expect_uspc_loss="$2"
  local train_status="$3"
  local run_log="$4"

  local output_dir
  output_dir="$(find_output_dir "${run_name}" || true)"
  local config_path="NA"
  if [[ -n "${output_dir}" && -f "${output_dir}/configs.yaml" ]]; then
    config_path="${output_dir}/configs.yaml"
  fi

  local best_ckpt="NA"
  local full_best_ckpt="NA"
  if [[ -n "${output_dir}" ]]; then
    if [[ -f "${output_dir}/best.pth" ]]; then
      best_ckpt="${output_dir}/best.pth"
    fi
    if [[ -d "${output_dir}/metric_ckpts" ]]; then
      local metric_best full_metric_best
      metric_best="$(newest_file_or_na "${output_dir}/metric_ckpts" 'best_epoch*.pth')"
      full_metric_best="$(newest_file_or_na "${output_dir}/metric_ckpts" 'full_best_epoch*.pth')"
      if [[ "${metric_best}" != "NA" ]]; then
        best_ckpt="${metric_best}"
      fi
      if [[ "${full_metric_best}" != "NA" ]]; then
        full_best_ckpt="${full_metric_best}"
      fi
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

  local error_grep_file error_found
  error_grep_file="${LOGS_DIR}/${run_name}.error_grep.txt"
  if grep -En 'Traceback|ZeroDivisionError|TRAIN_FAILED' "${run_log}" > "${error_grep_file}"; then
    error_found="yes"
  else
    error_found="no"
    printf '[NO_TRACEBACK_ZERODIVISION_OR_TRAIN_FAILED]\n' > "${error_grep_file}"
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
  if [[ "${error_found}" == "yes" ]]; then
    status="${status}_ERROR_LOG_FOUND"
  fi

  append_summary \
    "${run_name}" "${status}" "${expect_uspc_loss}" "${uspc_loss_found}" "${loss_numeric_check}" "${error_found}" \
    "${best_r1}" "${best_epoch}" "${t2i_r1}" "${t2i_r5}" "${t2i_r10}" "${t2i_map}" "${t2i_minp}" \
    "${output_dir:-NA}" "${config_path}" "${best_ckpt}" "${full_best_ckpt}" "${t2i_table_file}" "${loss_samples_file}" "${error_grep_file}"

  echo "[USPC_60EP_SUMMARY] run=${run_name} status=${status} best_R1=${best_r1} best_epoch=${best_epoch} loss_found=${uspc_loss_found} error_found=${error_found} output_dir=${output_dir:-NA}"
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
  local cmd=(python train.py --name "${run_name}" "${COMMON_ARGS[@]}" "$@")

  echo
  echo "[USPC_60EP_RUN_START] run=${run_name} expect_uspc_loss=${expect_uspc_loss} started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[USPC_60EP_RUN_CMD] $(quote_cmd "${cmd[@]}")"

  if ! command_has_warmup_epochs_1 "${cmd[@]}"; then
    echo "[USPC_60EP_ABORT] run=${run_name} command missing required --warmup_epochs 1"
    echo "[USPC_60EP_ABORT_CMD] $(quote_cmd "${cmd[@]}")"
    exit 2
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "${run_log}"
  local train_status="${PIPESTATUS[0]}"
  set -e

  echo "[USPC_60EP_RUN_END] run=${run_name} status=${train_status} ended_at=$(date '+%Y-%m-%d %H:%M:%S')"
  local summary_status=0
  summarize_run "${run_name}" "${expect_uspc_loss}" "${train_status}" "${run_log}" || summary_status=$?
  return "${summary_status}"
}

FAIL_COUNT=0

run_experiment "uspc_off_control_60ep" "no" || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "uspc_l005_60ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.05 \
  --uspc_uncertainty_weighting \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "uspc_l008_60ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.08 \
  --uspc_uncertainty_weighting \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "uspc_l010_60ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.10 \
  --uspc_uncertainty_weighting \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "uspc_l008_no_unc_weight_60ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.08 \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

echo
echo "[USPC_60EP_TABLE_MD] ${SUMMARY_MD}"
echo "[USPC_60EP_TABLE_TSV] ${SUMMARY_TSV}"
echo "[USPC_60EP_DONE] PACK_DIR=${PACK_DIR}"

if [[ "${FAIL_COUNT}" -ne 0 ]]; then
  echo "[USPC_60EP_WARN] FAIL_COUNT=${FAIL_COUNT}"
  exit 1
fi
