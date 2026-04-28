#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PACK_DIR}/../.." && pwd)"
MASTER_LOG="${PACK_DIR}/master.log"
RUN_ROOT="${PACK_DIR}/runs"
WARMSTART_CKPT="/home/IRRA-main/IRRA-main/logs/RGBT-PEDES/2026年4月20日105430/LOCAL60_MULTIPACK_20260419_113804/baseline_60ep/RGBT-PEDES/20260419_113823_baseline/best.pth"

mkdir -p "${RUN_ROOT}"
: > "${MASTER_LOG}"
exec >> "${MASTER_LOG}" 2>&1

cd "${REPO_ROOT}"

echo "[MASTER] hardneg v2 sweep started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[MASTER] repo_root=${REPO_ROOT}"
echo "[MASTER] pack_dir=${PACK_DIR}"
echo "[MASTER] run_root=${RUN_ROOT}"
echo "[MASTER] warmstart_ckpt=${WARMSTART_CKPT}"

COMMON_ARGS=(
  --dataset_name RGBT-PEDES
  --root_dir datasets/TextReID
  --img_aug
  --MLM
  --loss_names sdm+mlm
  --batch_size 16
  --test_batch_size 512
  --num_workers 8
  --warmup_epochs 1
  --log_period 200
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
  --lambda_mlm 1.0
)

HARDNEG_V2_ARGS=(
  --use_hardneg_aux
  --hardneg_version v2
  --hn_loss_type softplus
  --hn_direction both
  --debug_hardneg_stats
)

run_train() {
  local name="$1"
  shift
  local output_dir="${RUN_ROOT}/${name}"
  mkdir -p "${output_dir}"

  echo
  echo "[RUN] ${name} started at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "[RUN] output_dir=${output_dir}"
  echo "[RUN] command: python train.py ${COMMON_ARGS[*]} --output_dir ${output_dir} --name ${name} $*"

  python train.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "${output_dir}" \
    --name "${name}" \
    "$@"

  echo "[RUN] ${name} finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

echo
echo "[MASTER] Step 1/2: running hardneg v2 2-epoch smoke."
run_train "v2_smoke_l005_m005_top3_temp007_2" \
  --num_epoch 2 \
  "${HARDNEG_V2_ARGS[@]}" \
  --lambda_hn 0.05 \
  --hn_margin 0.05 \
  --hn_topn 3 \
  --hn_start_epoch 1 \
  --hn_temperature 0.07
echo "[MASTER] Smoke succeeded. Starting formal experiments."

echo
echo "[MASTER] Step 2/2A: from-scratch 60-epoch experiments."
run_train "v2_scratch_l005_m005_top3_temp007_60" \
  --num_epoch 60 \
  "${HARDNEG_V2_ARGS[@]}" \
  --lambda_hn 0.05 \
  --hn_margin 0.05 \
  --hn_topn 3 \
  --hn_start_epoch 5 \
  --hn_temperature 0.07

run_train "v2_scratch_l010_m005_top3_temp007_60" \
  --num_epoch 60 \
  "${HARDNEG_V2_ARGS[@]}" \
  --lambda_hn 0.10 \
  --hn_margin 0.05 \
  --hn_topn 3 \
  --hn_start_epoch 5 \
  --hn_temperature 0.07

run_train "v2_scratch_l005_m010_top3_temp007_60" \
  --num_epoch 60 \
  "${HARDNEG_V2_ARGS[@]}" \
  --lambda_hn 0.05 \
  --hn_margin 0.10 \
  --hn_topn 3 \
  --hn_start_epoch 5 \
  --hn_temperature 0.07

echo
echo "[MASTER] Step 2/2B: warmstart 10-epoch experiments."
run_train "v2_warm_l005_m005_top3_temp007_10_lr1e6" \
  --num_epoch 10 \
  --load_ckpt_file "${WARMSTART_CKPT}" \
  --lr 1e-6 \
  "${HARDNEG_V2_ARGS[@]}" \
  --lambda_hn 0.05 \
  --hn_margin 0.05 \
  --hn_topn 3 \
  --hn_start_epoch 1 \
  --hn_temperature 0.07

run_train "v2_warm_l010_m005_top3_temp007_10_lr1e6" \
  --num_epoch 10 \
  --load_ckpt_file "${WARMSTART_CKPT}" \
  --lr 1e-6 \
  "${HARDNEG_V2_ARGS[@]}" \
  --lambda_hn 0.10 \
  --hn_margin 0.05 \
  --hn_topn 3 \
  --hn_start_epoch 1 \
  --hn_temperature 0.07

run_train "v2_warm_l005_m010_top3_temp007_10_lr1e6" \
  --num_epoch 10 \
  --load_ckpt_file "${WARMSTART_CKPT}" \
  --lr 1e-6 \
  "${HARDNEG_V2_ARGS[@]}" \
  --lambda_hn 0.05 \
  --hn_margin 0.10 \
  --hn_topn 3 \
  --hn_start_epoch 1 \
  --hn_temperature 0.07

echo
echo "[MASTER] All experiments finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[MASTER] Writing summary."
python "${PACK_DIR}/summarize_hardneg_v2.py" "${PACK_DIR}"
