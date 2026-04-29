#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNS_ROOT="${PROJECT_ROOT}/pack/runs"
MASTER_LOG="${SCRIPT_DIR}/master.log"
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_proto_calib_v2.py"
SUMMARY_CSV="${SCRIPT_DIR}/summary.csv"
WARMSTART_CKPT="${WARMSTART_CKPT:-/home/IRRA-main/IRRA-main/logs/RGBT-PEDES/2026年4月20日105430/LOCAL60_MULTIPACK_20260419_113804/baseline_60ep/RGBT-PEDES/20260419_113823_baseline/best.pth}"

mkdir -p "${RUNS_ROOT}"
echo "[launcher] master log: ${MASTER_LOG}"
exec >> "${MASTER_LOG}" 2>&1

cd "${PROJECT_ROOT}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

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
  --use_proto_calib
  --proto_version v2
  --proto_temp 0.07
  --proto_momentum 0.9
  --proto_warmup_epochs 1
)

run_train() {
  local name="$1"
  shift
  local output_dir="${RUNS_ROOT}/${name}"
  mkdir -p "${output_dir}"
  log "START ${name}"
  log "output_dir=${output_dir}"
  log "command: python -u train.py --name ${name} --output_dir ${output_dir} $*"
  python -u train.py \
    --name "${name}" \
    --output_dir "${output_dir}" \
    "$@"
  log "END ${name}"
}

run_proto2() {
  local name="$1"
  local mode="$2"
  local epochs="$3"
  local lr="$4"
  local self_v="$5"
  local self_t="$6"
  local cross_t2v="$7"
  local cross_v2t="$8"
  local align="$9"
  shift 9

  local extra_args=("$@")
  run_train "${name}" \
    "${COMMON_ARGS[@]}" \
    --num_epoch "${epochs}" \
    --lr "${lr}" \
    --lambda_proto_self_v "${self_v}" \
    --lambda_proto_self_t "${self_t}" \
    --lambda_proto_cross_t2v "${cross_t2v}" \
    --lambda_proto_cross_v2t "${cross_v2t}" \
    --lambda_proto_align "${align}" \
    "${extra_args[@]}"
}

log "============================================================"
log "Prototype Calibration v2 sweep"
log "project_root=${PROJECT_ROOT}"
log "runs_root=${RUNS_ROOT}"
log "warmstart_ckpt=${WARMSTART_CKPT}"
log "No synth, no hardneg, no offline hardneg flags are enabled."
log "============================================================"

log "Running 2 epoch smoke before formal experiments."
run_proto2 \
  proto2_smoke_self005_cross005_align001_2 \
  smoke \
  2 \
  1e-5 \
  0.05 \
  0.05 \
  0.05 \
  0.05 \
  0.01 \
  --log_period 20 \
  --debug_proto_stats
log "Smoke succeeded; starting formal experiments."

run_proto2 \
  proto2_scratch_self005_cross000_align000_60 \
  scratch \
  60 \
  1e-5 \
  0.05 \
  0.05 \
  0 \
  0 \
  0

run_proto2 \
  proto2_scratch_self005_cross005_align000_60 \
  scratch \
  60 \
  1e-5 \
  0.05 \
  0.05 \
  0.05 \
  0.05 \
  0

run_proto2 \
  proto2_scratch_self005_cross005_align001_60 \
  scratch \
  60 \
  1e-5 \
  0.05 \
  0.05 \
  0.05 \
  0.05 \
  0.01

run_proto2 \
  proto2_scratch_self010_cross005_align001_60 \
  scratch \
  60 \
  1e-5 \
  0.10 \
  0.10 \
  0.05 \
  0.05 \
  0.01

if [[ ! -f "${WARMSTART_CKPT}" ]]; then
  log "ERROR: warmstart checkpoint not found: ${WARMSTART_CKPT}"
  log "Set WARMSTART_CKPT=/path/to/best.pth before launching if this path differs."
  exit 2
fi

run_proto2 \
  proto2_warm_self005_cross000_align000_10_lr1e6 \
  warm \
  10 \
  1e-6 \
  0.05 \
  0.05 \
  0 \
  0 \
  0 \
  --load_ckpt_file "${WARMSTART_CKPT}"

run_proto2 \
  proto2_warm_self005_cross005_align000_10_lr1e6 \
  warm \
  10 \
  1e-6 \
  0.05 \
  0.05 \
  0.05 \
  0.05 \
  0 \
  --load_ckpt_file "${WARMSTART_CKPT}"

run_proto2 \
  proto2_warm_self005_cross005_align001_10_lr1e6 \
  warm \
  10 \
  1e-6 \
  0.05 \
  0.05 \
  0.05 \
  0.05 \
  0.01 \
  --load_ckpt_file "${WARMSTART_CKPT}"

run_proto2 \
  proto2_warm_self010_cross005_align001_10_lr1e6 \
  warm \
  10 \
  1e-6 \
  0.10 \
  0.10 \
  0.05 \
  0.05 \
  0.01 \
  --load_ckpt_file "${WARMSTART_CKPT}"

log "All formal experiments finished. Building summary."
python "${SUMMARY_SCRIPT}" --runs_root "${RUNS_ROOT}" --output "${SUMMARY_CSV}"
log "summary_csv=${SUMMARY_CSV}"
log "DONE"
