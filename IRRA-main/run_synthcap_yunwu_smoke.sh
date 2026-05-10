#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_BASE="${OPENAI_API_BASE:-https://yunwu.ai/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.5}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[SYNTH-SMOKE] OPENAI_API_KEY is not set. Export it in the shell before running."
  exit 2
fi

PACK_DIR="pack/synthcap_yunwu_smoke_$(date +%Y%m%d_%H%M%S)"
BANK_DIR="pack/synthcap_bank"
BANK_FILE="${BANK_DIR}/synthcap_yunwu_smoke20_v5.json"
mkdir -p "${PACK_DIR}" "${BANK_DIR}"

echo "[SYNTH-SMOKE] api_base=${OPENAI_API_BASE}"
echo "[SYNTH-SMOKE] model=${OPENAI_MODEL}"
echo "[SYNTH-SMOKE] key_source=OPENAI_API_KEY environment variable"
echo "[SYNTH-SMOKE] pack_dir=${PACK_DIR}"
echo "[SYNTH-SMOKE] bank_file=${BANK_FILE}"

python tools/build_large_synth_bank_v5_llm.py \
  --provider openai_compatible \
  --test_api \
  --temperature 0.2 \
  --max_tokens 64

python tools/build_large_synth_bank_v5_llm.py \
  --provider openai_compatible \
  --dataset_name RGBT-PEDES \
  --root_dir datasets/TextReID \
  --output_file "${BANK_FILE}" \
  --limit 20 \
  --resume \
  --debug \
  --temperature 0.2 \
  --max_tokens 256 \
  --timeout 60 \
  --max_retries 3 \
  --retry_sleep 2.0

python tools/check_synth_caption_bank.py \
  --dataset_name RGBT-PEDES \
  --root_dir datasets/TextReID \
  --synth_caption_bank_file "${BANK_FILE}"

python - "${BANK_FILE}" <<'PY'
import json
import sys

bank_file = sys.argv[1]
with open(bank_file, "r", encoding="utf-8") as f:
    bank = json.load(f)
print("[SYNTH-SMOKE] sample_bank_items:")
for idx, (key, captions) in enumerate(bank.items()):
    if idx >= 5:
        break
    print(f"  - {key}:")
    for caption in captions[:3]:
        print(f"      {caption}")
PY

if [[ "${RUN_TRAIN_SMOKE:-0}" == "1" ]]; then
  TRAIN_LOG="${PACK_DIR}/train_smoke_1ep.log"
  echo "[SYNTH-SMOKE] RUN_TRAIN_SMOKE=1, starting one-epoch train smoke."
  python train.py \
    --name synthcap_yunwu_smoke_1ep \
    --output_dir "${PACK_DIR}/train_outputs" \
    --dataset_name RGBT-PEDES \
    --root_dir datasets/TextReID \
    --img_aug \
    --MLM \
    --loss_names "sdm+mlm" \
    --batch_size 16 \
    --test_batch_size 512 \
    --num_workers 8 \
    --num_epoch 1 \
    --warmup_epochs 1 \
    --log_period 20 \
    --eval_period 1 \
    --use_rgbt_add_fusion \
    --use_global_uncertainty \
    --uncertainty_dist_type diag_gaussian \
    --use_precision_fusion \
    --uncertainty_loss_type w2 \
    --lambda_unc 0.09 \
    --lambda_tr 1.10 \
    --lambda_rt 0.90 \
    --lambda_tt 0.0 \
    --lambda_var_reg 1e-4 \
    --eval_use_fused_mu \
    --use_synth_caption_bank \
    --synth_caption_bank_file "${BANK_FILE}" \
    --synth_caption_keep_per_image 2 \
    2>&1 | tee "${TRAIN_LOG}"
  echo "[SYNTH-SMOKE] train_log=${TRAIN_LOG}"
else
  echo "[SYNTH-SMOKE] one-epoch train smoke skipped; set RUN_TRAIN_SMOKE=1 to enable it."
fi
