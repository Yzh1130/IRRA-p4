#!/usr/bin/env bash
# R19 post-hoc validation. All stage output and resume state stay in p4_r19_out.
set -euo pipefail
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ROOT=${P4_ROOT:-/root/autodl-tmp}
REPO=${P4_REPO:-$ROOT/IRRA-p4}
PY=${PYTHON:-python}
STAGE=${STAGE:-S0}
case "$STAGE" in S0|S1|S2|S3|S4) ;; *) echo 'STAGE must be S0, S1, S2, S3 or S4'; exit 2;; esac
if [ "${SKIP_CONDA:-0}" != 1 ]; then
  source "${CONDA_BASE:-/root/miniconda3}/etc/profile.d/conda.sh"
  conda activate rgbt5090
fi
ARGS=(--stage "$STAGE" --root "$ROOT" --repo "$REPO")
[ -z "${R19_SPEC_JSON:-}" ] || ARGS+=(--spec "$R19_SPEC_JSON")
[ "${R19_SYNTHETIC:-0}" != 1 ] || ARGS+=(--synthetic)
exec "$PY" -B "$REPO/IRRA-main/tools/p4/r19/driver.py" "${ARGS[@]}"
