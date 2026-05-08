#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PACK_ROOT="pack"
PREFERRED_STRICT_ROOT="${PACK_ROOT}/00_PAPER_ABLATION_RESULTS_ORDERED/07_STRICT_E2E_REPRO_PROTO2_PROTO3_20260508_011825"
STRICT_E2E_ROOT="${STRICT_E2E_ROOT:-${PREFERRED_STRICT_ROOT}}"
PACK_DIR="${PACK_ROOT}/05_STRICT_PROTO3_USPC_20EP_${TIMESTAMP}"
RUNS_DIR="${PACK_DIR}/runs"
LOGS_DIR="${PACK_DIR}/logs"
TABLES_DIR="${PACK_DIR}/tables"
SCRIPTS_DIR="${PACK_DIR}/scripts"
MASTER_LOG="${PACK_DIR}/master.log"
SUMMARY_MD="${TABLES_DIR}/strict_proto3_uspc_20ep_summary.md"
SUMMARY_TSV="${TABLES_DIR}/strict_proto3_uspc_20ep_summary.tsv"

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}" "${TABLES_DIR}" "${SCRIPTS_DIR}"
cp "${BASH_SOURCE[0]}" "${SCRIPTS_DIR}/run_strict_proto3_uspc_20ep.sh"
printf '%s\n' "${PACK_DIR}" > "${PACK_ROOT}/LATEST_STRICT_PROTO3_USPC_20EP_PACK.txt"

exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[STRICT_PROTO3_USPC_START] PACK_DIR=${PACK_DIR}"
echo "[STRICT_PROTO3_USPC_START] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[STRICT_PROTO3_USPC_START] pwd=$(pwd)"

print_candidates() {
  local title="$1"
  local root="$2"
  local pattern="$3"
  echo "[STRICT_PROTO3_USPC_CANDIDATES] ${title}" >&2
  if [[ -d "${root}" ]]; then
    find "${root}" -type f -iname "${pattern}" | sort | sed 's/^/  /' >&2 || true
  else
    echo "  [ROOT_NOT_FOUND] ${root}" >&2
    echo "  [AVAILABLE_STRICT_ROOTS]" >&2
    find "${PACK_ROOT}" -maxdepth 4 -type d -iname '*STRICT*PROTO2*PROTO3*' 2>/dev/null | sort | sed 's/^/    /' >&2 || true
  fi
}

resolve_strict_root() {
  if [[ -d "${STRICT_E2E_ROOT}" ]]; then
    printf '%s' "${STRICT_E2E_ROOT}"
    return 0
  fi
  local roots=()
  while IFS= read -r item; do
    roots+=("${item}")
  done < <(find "${PACK_ROOT}" -maxdepth 4 -type d -iname '07_STRICT_E2E_REPRO_PROTO2_PROTO3_*' 2>/dev/null | sort)
  if [[ "${#roots[@]}" -eq 1 ]]; then
    printf '%s' "${roots[0]}"
    return 0
  fi
  echo "[STRICT_PROTO3_USPC_ABORT] strict E2E root is not uniquely located." >&2
  echo "[STRICT_PROTO3_USPC_ABORT] preferred=${PREFERRED_STRICT_ROOT}" >&2
  echo "[STRICT_PROTO3_USPC_ABORT] override with STRICT_E2E_ROOT=/path/to/07_STRICT_E2E..." >&2
  print_candidates "strict-root candidates" "${PACK_ROOT}" '*'
  return 1
}

resolve_unique_from_candidates() {
  local label="$1"
  shift
  local candidates=("$@")
  local existing=()
  local item
  for item in "${candidates[@]}"; do
    if [[ -f "${item}" ]]; then
      existing+=("${item}")
    fi
  done
  if [[ "${#existing[@]}" -eq 1 ]]; then
    printf '%s' "${existing[0]}"
    return 0
  fi
  echo "[STRICT_PROTO3_USPC_ABORT] ${label} is not uniquely located. count=${#existing[@]}" >&2
  for item in "${existing[@]}"; do
    echo "  ${item}" >&2
  done
  return 1
}

resolve_proto2_ckpt() {
  if [[ -n "${PROTO2_CKPT_FILE:-}" ]]; then
    if [[ -f "${PROTO2_CKPT_FILE}" ]]; then
      printf '%s' "${PROTO2_CKPT_FILE}"
      return 0
    fi
    echo "[STRICT_PROTO3_USPC_ABORT] PROTO2_CKPT_FILE does not exist: ${PROTO2_CKPT_FILE}" >&2
    return 1
  fi

  local exact=()
  while IFS= read -r item; do
    exact+=("${item}")
  done < <(find "${STRICT_ROOT}" -type f -name 'checkpoint_strict_e2e_proto2_best_or_full_best.pth' 2>/dev/null | sort)
  if [[ "${#exact[@]}" -eq 1 ]]; then
    printf '%s' "${exact[0]}"
    return 0
  fi

  local candidates=()
  local item base dir
  while IFS= read -r item; do
    base="$(basename "${item}")"
    dir="$(dirname "${item}")"
    if ! grep -Eiq 'proto2' <<< "${base} ${dir}"; then
      continue
    fi
    if ! grep -Eiq 'best|full_best' <<< "${base}"; then
      continue
    fi
    if grep -Eiq 'proto3|final|frozen' <<< "${base}"; then
      continue
    fi
    candidates+=("${item}")
  done < <(find "${STRICT_ROOT}" -type f -iname '*.pth' 2>/dev/null | sort)
  if [[ "${#candidates[@]}" -eq 1 ]]; then
    printf '%s' "${candidates[0]}"
    return 0
  fi
  echo "[STRICT_PROTO3_USPC_ABORT] Proto2 best checkpoint could not be uniquely located." >&2
  echo "[STRICT_PROTO3_USPC_HINT] set PROTO2_CKPT_FILE=/abs/path/to/proto2_best.pth" >&2
  print_candidates "Proto2 checkpoint candidates" "${STRICT_ROOT}" '*.pth'
  return 1
}

resolve_proto_init() {
  if [[ -n "${OFFLINE_PROTO_INIT_FILE:-}" ]]; then
    if [[ -f "${OFFLINE_PROTO_INIT_FILE}" ]]; then
      printf '%s' "${OFFLINE_PROTO_INIT_FILE}"
      return 0
    fi
    echo "[STRICT_PROTO3_USPC_ABORT] OFFLINE_PROTO_INIT_FILE does not exist: ${OFFLINE_PROTO_INIT_FILE}" >&2
    return 1
  fi

  local candidates=()
  while IFS= read -r item; do
    candidates+=("${item}")
  done < <(find "${STRICT_ROOT}" -type f -name 'offline_proto_from_proto2_best.pt' 2>/dev/null | sort)
  if [[ "${#candidates[@]}" -eq 1 ]]; then
    printf '%s' "${candidates[0]}"
    return 0
  fi
  echo "[STRICT_PROTO3_USPC_ABORT] offline prototype init could not be uniquely located." >&2
  echo "[STRICT_PROTO3_USPC_HINT] set OFFLINE_PROTO_INIT_FILE=/abs/path/to/offline_proto_from_proto2_best.pt" >&2
  print_candidates "offline prototype candidates" "${STRICT_ROOT}" '*.pt'
  return 1
}

resolve_optional_config() {
  if [[ -n "${STRICT_PROTO3_CONFIG_FILE:-}" ]]; then
    if [[ -f "${STRICT_PROTO3_CONFIG_FILE}" ]]; then
      printf '%s' "${STRICT_PROTO3_CONFIG_FILE}"
      return 0
    fi
    echo "[STRICT_PROTO3_USPC_WARN] STRICT_PROTO3_CONFIG_FILE does not exist: ${STRICT_PROTO3_CONFIG_FILE}" >&2
    printf 'NA'
    return 0
  fi
  local candidates=()
  while IFS= read -r item; do
    candidates+=("${item}")
  done < <(find "${STRICT_ROOT}" -type f -name 'config_strict_e2e_proto3_self_calibration.yaml' 2>/dev/null | sort)
  if [[ "${#candidates[@]}" -eq 1 ]]; then
    printf '%s' "${candidates[0]}"
  else
    printf 'NA'
  fi
}

resolve_optional_file() {
  local pattern="$1"
  local candidates=()
  while IFS= read -r item; do
    candidates+=("${item}")
  done < <(find "${STRICT_ROOT}" -type f -name "${pattern}" 2>/dev/null | sort)
  if [[ "${#candidates[@]}" -eq 1 ]]; then
    printf '%s' "${candidates[0]}"
  else
    printf 'NA'
  fi
}

STRICT_ROOT="$(resolve_strict_root)"
PROTO2_CKPT="$(resolve_proto2_ckpt)"
PROTO_INIT_FILE="$(resolve_proto_init)"
STRICT_PROTO3_CONFIG="$(resolve_optional_config)"
STRICT_PROTO3_BEST_RECORD="$(resolve_optional_file 'checkpoint_strict_e2e_proto3_best_or_full_best.pth')"
STRICT_PROTO3_FULL_RECORD="$(resolve_optional_file 'fullstate_R1_57550.pth')"

echo "[STRICT_PROTO3_USPC_PATH] STRICT_ROOT=${STRICT_ROOT}"
echo "[STRICT_PROTO3_USPC_PATH] PROTO2_CKPT=${PROTO2_CKPT}"
echo "[STRICT_PROTO3_USPC_PATH] PROTO_INIT_FILE=${PROTO_INIT_FILE}"
echo "[STRICT_PROTO3_USPC_PATH] STRICT_PROTO3_CONFIG=${STRICT_PROTO3_CONFIG}"
echo "[STRICT_PROTO3_USPC_PATH] STRICT_PROTO3_BEST_RECORD=${STRICT_PROTO3_BEST_RECORD}"
echo "[STRICT_PROTO3_USPC_PATH] STRICT_PROTO3_FULL_RECORD=${STRICT_PROTO3_FULL_RECORD}"

proto2_ckpt_basename="$(basename "${PROTO2_CKPT}")"
if grep -Eqi 'proto3|fullstate_R1_57550|frozen' <<< "${proto2_ckpt_basename}"; then
  echo "[STRICT_PROTO3_USPC_ABORT] Refusing to train from a Proto3/frozen checkpoint: ${PROTO2_CKPT}" >&2
  exit 2
fi

COMMON_ARGS=(
  --dataset_name RGBT-PEDES
  --root_dir datasets/TextReID
  --img_aug
  --MLM
  --loss_names "sdm+mlm"
  --batch_size 16
  --test_batch_size 512
  --num_workers 8
  --num_epoch 20
  --warmup_epochs 0
  --lr 1e-6
  --eval_period 1
  --log_period 200
  --output_dir "${RUNS_DIR}"
  --load_ckpt_file "${PROTO2_CKPT}"
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
  --proto_init_file "${PROTO_INIT_FILE}"
  --proto_init_strict
  --save_metric_best_ckpt
  --save_full_eval_state_ckpt
)

printf '| run_name | status | uspc_loss_expected | uspc_loss_found | loss_numeric_check | error_found | best_R1 | best_epoch | t2i_R1 | t2i_R5 | t2i_R10 | t2i_mAP | t2i_mINP | output_dir | config | best_ckpt | full_best_ckpt | t2i_table | loss_samples | error_grep |\n' > "${SUMMARY_MD}"
printf '|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|\n' >> "${SUMMARY_MD}"
printf 'run_name\tstatus\tuspc_loss_expected\tuspc_loss_found\tloss_numeric_check\terror_found\tbest_R1\tbest_epoch\tt2i_R1\tt2i_R5\tt2i_R10\tt2i_mAP\tt2i_mINP\toutput_dir\tconfig\tbest_ckpt\tfull_best_ckpt\tt2i_table\tloss_samples\terror_grep\n' > "${SUMMARY_TSV}"

quote_cmd() {
  printf '%q ' "$@"
}

command_has_required_inputs() {
  local args=("$@")
  local has_load="no"
  local has_proto="no"
  local i
  for ((i = 0; i < ${#args[@]}; i += 1)); do
    if [[ "${args[$i]}" == "--load_ckpt_file" ]] && (( i + 1 < ${#args[@]} )) && [[ "${args[$((i + 1))]}" == "${PROTO2_CKPT}" ]]; then
      has_load="yes"
    fi
    if [[ "${args[$i]}" == "--proto_init_file" ]] && (( i + 1 < ${#args[@]} )) && [[ "${args[$((i + 1))]}" == "${PROTO_INIT_FILE}" ]]; then
      has_proto="yes"
    fi
  done
  [[ "${has_load}" == "yes" && "${has_proto}" == "yes" ]]
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
  if grep -En 'Traceback|RuntimeError|ZeroDivisionError|TRAIN_FAILED' "${run_log}" > "${error_grep_file}"; then
    error_found="yes"
  else
    error_found="no"
    printf '[NO_TRACEBACK_RUNTIME_ZERODIVISION_OR_TRAIN_FAILED]\n' > "${error_grep_file}"
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

  echo "[STRICT_PROTO3_USPC_SUMMARY] run=${run_name} status=${status} best_R1=${best_r1} best_epoch=${best_epoch} loss_found=${uspc_loss_found} error_found=${error_found} output_dir=${output_dir:-NA}"
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
  echo "[STRICT_PROTO3_USPC_RUN_START] run=${run_name} expect_uspc_loss=${expect_uspc_loss} started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[STRICT_PROTO3_USPC_RUN_CMD] $(quote_cmd "${cmd[@]}")"

  if ! command_has_required_inputs "${cmd[@]}"; then
    echo "[STRICT_PROTO3_USPC_ABORT] run=${run_name} command missing required Proto2 checkpoint or offline prototype init"
    echo "[STRICT_PROTO3_USPC_ABORT_CMD] $(quote_cmd "${cmd[@]}")"
    exit 3
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "${run_log}"
  local train_status="${PIPESTATUS[0]}"
  set -e

  echo "[STRICT_PROTO3_USPC_RUN_END] run=${run_name} status=${train_status} ended_at=$(date '+%Y-%m-%d %H:%M:%S')"
  local summary_status=0
  summarize_run "${run_name}" "${expect_uspc_loss}" "${train_status}" "${run_log}" || summary_status=$?
  return "${summary_status}"
}

FAIL_COUNT=0

run_experiment "strict_proto3_control_20ep" "no" || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "strict_proto3_uspc_l005_20ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.05 \
  --uspc_uncertainty_weighting \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "strict_proto3_uspc_l010_20ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.10 \
  --uspc_uncertainty_weighting \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "strict_proto3_uspc_l005_no_unc_20ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.05 \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

run_experiment "strict_proto3_uspc_l010_no_unc_20ep" "yes" \
  --use_uspc \
  --lambda_uspc 0.10 \
  --uspc_weight_detach || FAIL_COUNT=$((FAIL_COUNT + 1))

echo
echo "[STRICT_PROTO3_USPC_TABLE_MD] ${SUMMARY_MD}"
echo "[STRICT_PROTO3_USPC_TABLE_TSV] ${SUMMARY_TSV}"
echo "[STRICT_PROTO3_USPC_DONE] PACK_DIR=${PACK_DIR}"

if [[ "${FAIL_COUNT}" -ne 0 ]]; then
  echo "[STRICT_PROTO3_USPC_WARN] FAIL_COUNT=${FAIL_COUNT}"
  exit 1
fi
