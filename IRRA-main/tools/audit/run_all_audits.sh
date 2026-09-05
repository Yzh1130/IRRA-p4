#!/usr/bin/env bash

# Stage 0 只读审计总入口。任一步失败都不会阻止后续审计。
set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
worktree_root="$(cd "${project_root}/.." && pwd)"

data_root="${DATA_ROOT:-${project_root}/datasets/TextReID}"
repo="${REPO:-${worktree_root}}"
python_bin="${PYTHON:-python}"
out_dir="${project_root}/audit_out"

asset_roots=()
if [[ -n "${ASSET_ROOTS:-}" ]]; then
  while IFS= read -r asset_root; do
    [[ -n "${asset_root}" ]] && asset_roots+=("${asset_root}")
  done < <(
    AUDIT_ASSET_ROOTS_TEXT="${ASSET_ROOTS}" "${python_bin}" -B -c \
      'import os, shlex; print("\n".join(shlex.split(os.environ["AUDIT_ASSET_ROOTS_TEXT"])))'
  )
else
  asset_roots=("${project_root}/mraf_runs" "${project_root}/logs")
fi

run_step() {
  local label="$1"
  shift
  printf '\n[%s]\n' "${label}"
  if "$@"; then
    printf '[OK] %s\n' "${label}"
  else
    local exit_code=$?
    printf '[WARN] %s 失败（exit=%s），继续下一项。\n' "${label}" "${exit_code}" >&2
  fi
}

cd "${project_root}" || {
  printf '无法进入 Python 项目根：%s\n' "${project_root}" >&2
  exit 1
}

run_step "数据审计" \
  "${python_bin}" -B -m tools.audit.probe_dataset \
  --data-root "${data_root}" --repo "${project_root}" --out "${out_dir}"

run_step "实验资产审计" \
  "${python_bin}" -B -m tools.audit.audit_assets \
  --roots "${asset_roots[@]}" --repo "${repo}" --out "${out_dir}"

run_step "超参数历史审计" \
  "${python_bin}" -B -m tools.audit.audit_hparams \
  --roots "${asset_roots[@]}" --repo "${repo}" --out "${out_dir}"

run_step "仓库状态审计" \
  "${python_bin}" -B -m tools.audit.audit_repo_state \
  --repo "${repo}" --out "${out_dir}"

printf '\n生成的报告文件：\n'
if [[ -d "${project_root}/docs/audit" ]]; then
  find "${project_root}/docs/audit" -maxdepth 1 -type f -name '*.md' -print | sort
else
  printf '未找到：%s\n' "${project_root}/docs/audit"
fi
