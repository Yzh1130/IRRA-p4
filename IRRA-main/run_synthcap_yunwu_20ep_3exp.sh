#!/usr/bin/env bash
set -uo pipefail

BANK_FILE="pack/synthcap_bank/synthcap_train_yunwu_gpt55_v5.json"
TS="$(date +%Y%m%d_%H%M%S)"
PACK_DIR="pack/synthcap_yunwu_${TS}"
MASTER_LOG="${PACK_DIR}/master.log"
SUMMARY_FILE="${PACK_DIR}/results_summary.tsv"
LATEST_FILE="pack/LATEST_SYNTHCAP_YUNWU_PACK.txt"

mkdir -p "${PACK_DIR}"
printf "%s\n" "${PACK_DIR}" > "${LATEST_FILE}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[SYNTH-20EP] pack_dir=${PACK_DIR}"
echo "[SYNTH-20EP] bank_file=${BANK_FILE}"
echo "[SYNTH-20EP] latest_file=${LATEST_FILE}"

if [[ ! -f "${BANK_FILE}" ]]; then
  echo "[SYNTH-20EP] missing bank file: ${BANK_FILE}"
  exit 2
fi

printf "exp_name\tkeep_per_image\tstatus\tlog_file\tlast_metric_line\n" > "${SUMMARY_FILE}"

BASE_ARGS=(
  --dataset_name RGBT-PEDES
  --root_dir datasets/TextReID
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
  --use_synth_caption_bank
  --synth_caption_bank_file "${BANK_FILE}"
  --output_dir "${PACK_DIR}/train_outputs"
)

run_exp() {
  local keep="$1"
  local exp_name="synthcap_yunwu_keep${keep}_20ep"
  local log_file="${PACK_DIR}/${exp_name}.log"

  echo "[SYNTH-20EP] starting ${exp_name}"
  python train.py \
    --name "${exp_name}" \
    "${BASE_ARGS[@]}" \
    --synth_caption_keep_per_image "${keep}" \
    2>&1 | tee "${log_file}"
  local status="${PIPESTATUS[0]}"

  local metric_line
  metric_line="$(grep -E "R1|Rank-1|mAP|best" "${log_file}" | tail -n 1 | tr '\t' ' ' || true)"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${exp_name}" "${keep}" "${status}" "${log_file}" "${metric_line}" \
    >> "${SUMMARY_FILE}"

  if [[ "${status}" -ne 0 ]]; then
    echo "[SYNTH-20EP] ${exp_name} failed with status=${status}; continuing."
  else
    echo "[SYNTH-20EP] ${exp_name} finished."
  fi
}

run_exp 1
run_exp 2
run_exp 3

echo "[SYNTH-20EP] summary=${SUMMARY_FILE}"
echo "[SYNTH-20EP] master_log=${MASTER_LOG}"
