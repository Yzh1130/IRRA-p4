#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

CONFIG_FILE="${CONFIG_FILE:-${QALR_SAFE_CONFIG_FILE:-}}"
CKPT_FILE="${CKPT_FILE:-${QALR_SAFE_CKPT_FILE:-}}"
if [[ -z "${CONFIG_FILE}" || ! -f "${CONFIG_FILE}" ]]; then
  echo "[QALR_V2_GRID_ABORT] Set CONFIG_FILE=/path/to/current_qalr_safe_configs.yaml" >&2
  exit 1
fi
if [[ -z "${CKPT_FILE}" || ! -f "${CKPT_FILE}" ]]; then
  echo "[QALR_V2_GRID_ABORT] Set CKPT_FILE=/path/to/current_proto3_best.pth" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PACK_DIR="pack/qalr_v2_safe_grid_${TIMESTAMP}"
LOGS_DIR="${PACK_DIR}/logs"
DIAG_DIR="${PACK_DIR}/diag"
TABLES_DIR="${PACK_DIR}/tables"
SUMMARY_TSV="${TABLES_DIR}/summary.tsv"
MASTER_LOG="${PACK_DIR}/master.log"
mkdir -p "${LOGS_DIR}" "${DIAG_DIR}" "${TABLES_DIR}"
cp "${BASH_SOURCE[0]}" "${PACK_DIR}/run_qalr_v2_safe_grid.sh"
printf '%s\n' "${PACK_DIR}" > pack/LATEST_QALR_V2_SAFE_GRID_PACK.txt

exec > >(tee -a "${MASTER_LOG}") 2>&1

SAFE_RERANK_TOPK="${SAFE_RERANK_TOPK:-128}"
SAFE_RERANK_ALPHA="${SAFE_RERANK_ALPHA:-0.01}"
SAFE_TOKEN_TOPM="${SAFE_TOKEN_TOPM:-8}"
SAFE_PATCH_TOPR="${SAFE_PATCH_TOPR:-4}"
SAFE_GATE_MARGIN="${SAFE_GATE_MARGIN:-0.05}"
SAFE_FOCUS_TOPN="${SAFE_FOCUS_TOPN:-16}"

printf 'run_name\tstatus\tR1\tR5\tR10\tmAP\tmINP\tchanged\timproved\thurt\tneutral\tnet\tdiag_file\tlog_file\n' > "${SUMMARY_TSV}"

append_summary() {
  local run_name="$1"
  local status="$2"
  local log_file="$3"
  local diag_file="$4"
  python - "$SUMMARY_TSV" "$run_name" "$status" "$log_file" "$diag_file" <<'PY'
import json
import re
import sys

summary_tsv, run_name, status, log_file, diag_file = sys.argv[1:6]
metrics = {"R1": "NA", "R5": "NA", "R10": "NA", "mAP": "NA", "mINP": "NA"}
try:
    text = open(log_file, "r", encoding="utf-8", errors="ignore").read()
    rows = re.findall(
        r"\|\s*t2i\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
        text,
    )
    if rows:
        r1, r5, r10, m_ap, m_inp = rows[-1]
        metrics.update({"R1": r1, "R5": r5, "R10": r10, "mAP": m_ap, "mINP": m_inp})
except OSError:
    pass

diag = {
    "changed_top1_count": "NA",
    "improved_top1_count": "NA",
    "hurt_top1_count": "NA",
    "neutral_changed_top1_count": "NA",
    "net_top1_gain": "NA",
}
try:
    with open(diag_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("summary", {})
    for key in diag:
        if key in summary:
            diag[key] = str(summary[key])
except OSError:
    pass

row = [
    run_name,
    status,
    metrics["R1"],
    metrics["R5"],
    metrics["R10"],
    metrics["mAP"],
    metrics["mINP"],
    diag["changed_top1_count"],
    diag["improved_top1_count"],
    diag["hurt_top1_count"],
    diag["neutral_changed_top1_count"],
    diag["net_top1_gain"],
    diag_file,
    log_file,
]
with open(summary_tsv, "a", encoding="utf-8") as handle:
    handle.write("\t".join(row) + "\n")
PY
}

run_eval() {
  local run_name="$1"
  shift
  local log_file="${LOGS_DIR}/${run_name}.log"
  local diag_file="${DIAG_DIR}/${run_name}.json"
  local cmd=(
    python test.py
    --config_file "${CONFIG_FILE}"
    --test_ckpt_file "${CKPT_FILE}"
    --use_qalr
    --rerank_topk "${SAFE_RERANK_TOPK}"
    --rerank_alpha "${SAFE_RERANK_ALPHA}"
    --rerank_token_topm "${SAFE_TOKEN_TOPM}"
    --rerank_patch_topr "${SAFE_PATCH_TOPR}"
    --use_qalr_score_norm
    --use_qalr_query_gate
    --qalr_query_gate_type top12_margin
    --qalr_query_gate_margin "${SAFE_GATE_MARGIN}"
    --use_qalr_focus_topn
    --qalr_focus_topn "${SAFE_FOCUS_TOPN}"
    --qalr_v2_save_diag
    --qalr_v2_diag_file "${diag_file}"
    "$@"
  )

  echo "[QALR_V2_GRID_RUN] ${run_name}"
  printf '[QALR_V2_GRID_CMD]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  set +e
  "${cmd[@]}" 2>&1 | tee "${log_file}"
  local status_code="${PIPESTATUS[0]}"
  set -e
  local status="ok"
  if [[ "${status_code}" -ne 0 ]]; then
    status="fail_${status_code}"
  fi
  append_summary "${run_name}" "${status}" "${log_file}" "${diag_file}"
}

echo "[QALR_V2_GRID_START] PACK_DIR=${PACK_DIR}"
echo "[QALR_V2_GRID_START] CONFIG_FILE=${CONFIG_FILE}"
echo "[QALR_V2_GRID_START] CKPT_FILE=${CKPT_FILE}"

run_eval old_qalr_safe

run_eval v2_diag_only_concat \
  --use_qalr_v2 --qalr_v2_diag_only --qalr_v2_modality_fuse concat
run_eval v2_concat_guard \
  --use_qalr_v2 --qalr_v2_top1_guard --qalr_v2_modality_fuse concat
run_eval v2_concat_guard_promote \
  --use_qalr_v2 --qalr_v2_top1_guard --qalr_v2_promote_only --qalr_v2_modality_fuse concat
run_eval v2_avg_guard \
  --use_qalr_v2 --qalr_v2_top1_guard --qalr_v2_modality_fuse avg
run_eval v2_precision_guard \
  --use_qalr_v2 --qalr_v2_top1_guard --qalr_v2_modality_fuse precision
run_eval v2_rgb_guard \
  --use_qalr_v2 --qalr_v2_top1_guard --qalr_v2_modality_fuse rgb
run_eval v2_thermal_guard \
  --use_qalr_v2 --qalr_v2_top1_guard --qalr_v2_modality_fuse thermal
run_eval v2_concat_alpha005 \
  --use_qalr_v2 --qalr_v2_top1_guard --qalr_v2_alpha_max 0.005 --qalr_v2_modality_fuse concat
run_eval v2_concat_alpha015 \
  --use_qalr_v2 --qalr_v2_top1_guard --qalr_v2_alpha_max 0.015 --qalr_v2_modality_fuse concat
run_eval v2_avg_promote002 \
  --use_qalr_v2 --qalr_v2_top1_guard --qalr_v2_promote_only --qalr_v2_promote_margin 0.002 --qalr_v2_modality_fuse avg
run_eval v2_precision_promote002 \
  --use_qalr_v2 --qalr_v2_top1_guard --qalr_v2_promote_only --qalr_v2_promote_margin 0.002 --qalr_v2_modality_fuse precision
run_eval v2_concat_focus8 \
  --use_qalr_v2 --qalr_focus_topn 8 --qalr_v2_top1_guard --qalr_v2_modality_fuse concat

echo "[QALR_V2_GRID_DONE] summary=${SUMMARY_TSV}"
