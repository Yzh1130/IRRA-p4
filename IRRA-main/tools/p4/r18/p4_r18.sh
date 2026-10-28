#!/usr/bin/env bash
# R18: L1 training/development, one-shot L2, report-only L2R. No downloads.
set -uo pipefail
set -f
ROOT=${P4_ROOT:-/root/autodl-tmp}
REPO=$ROOT/IRRA-p4; CODE=$REPO/IRRA-main; T=$CODE/tools/p4/transfer
R=$CODE/tools/p4/r18; DATA_ROOT=$CODE/datasets/TextReID
OUT=$ROOT/p4_r18_out; UPLOAD=$ROOT/upload; P4_OUT=$ROOT/p4_preexp; R13_IN=$ROOT/r13_inputs
DEV_SPLIT=$CODE/splits/rgbnt201_dev30.json; OVERLAP=$CODE/splits/rgbnt201_overlap.json
STAGE=${STAGE:-L1}; AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-1}; PY=${PYTHON:-python}
SEEDS='101 102 103'; FIT_SEEDS='7001 7002 7003'; ASSIGN_SEEDS='7031 7032 7033 7034 7035'
MIXES='M0 M4 M6c M1g M4c60 M9 M5e M5f M4id M10 M4c25 M4c50 M4c75'
METHODS='B* CAET A-SRC-2f A-2f-EM A-2f-ctxn A-2f-ctxn-g GP-same KR RVN QE-DBA KRir RVNir QEir FeatImp FeatImp-g KR+M RVN+M QE+M KRir+M RVNir+M QEir+M GP+M QE+CAET QE+SRC2f QE+EM QE+Orc DB Raw Oracle-A'
METHODS17='B* QE-DBA QE+M QE+CAET QE+SRC2f QE+EM QE+Const QE+NoGate'
GRID=v3; CONST_GRID='0 0.05 0.1 0.2 0.25 0.5 0.75 1'
case $STAGE in L1|L2|L2R) ;; *) echo 'STAGE must be L1, L2 or L2R'; exit 2;; esac
PHASE=$STAGE; [ "$STAGE" != L2R ] || PHASE=L2
mkdir -p "$OUT/logs" "$OUT/checks" "$OUT/.manifests"
mkdir "$OUT/.lock" 2>/dev/null || { echo "上次运行可能被中断；确认没有 p4_r18 进程在运行后，执行 rmdir $OUT/.lock 再重试"; exit 2; }
TS=$(date +%Y%m%d_%H%M%S); LOG=$OUT/logs/${STAGE}_${TS}.log
exec > >(tee -a "$LOG") 2>&1
export PYTHONUTF8=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SYNTHETIC=${R18_SYNTHETIC:-0}; VERIFY=(); PAR=1
if [ "$SYNTHETIC" = 1 ]; then
  [ -f "$ROOT/.r18_synthetic" ] || { echo 'Synthetic root marker missing'; rmdir "$OUT/.lock"; exit 2; }
  [ "$AUTO_SHUTDOWN" = 0 ] || { echo 'Synthetic mode requires AUTO_SHUTDOWN=0'; rmdir "$OUT/.lock"; exit 2; }
  SEEDS=${R18_SYNTHETIC_SEEDS:-101}; FIT_SEEDS=${R18_SYNTHETIC_FITS:-7001}
  ASSIGN_SEEDS=${R18_SYNTHETIC_ASSIGN:-7031}; GRID=dev
  # Methods and mixtures stay at the formal values even for synthetic fixtures.
fi
[ "$STAGE" != L2R ] || VERIFY=(--verify_only)
STATUS=$OUT/STATUS_$PHASE
FINAL=STOPPED
guard() {
  "$PY" -B "$R/workflow.py" guard --manifest "$OUT/checks/protected.json" \
    "$ROOT/p4_r15_out" "$ROOT/p4_r17_out" "$R13_IN" \
    "$P4_OUT/runs/base_clean_s101" "$P4_OUT/runs/base_clean_s102" "$P4_OUT/runs/base_clean_s103"
}
finish() {
  local rc=$?
  trap - EXIT INT TERM
  while read -r pid; do wait "$pid" || rc=1; done < <(jobs -rp)
  guard || { rc=1; FINAL=PROTECTED_TREE_CHANGED; }
  if [ "$rc" -ne 0 ]; then FINAL="$FINAL (exit=$rc)"; fi
  printf '%s\n' "$FINAL" > "$STATUS"
  "$PY" -B "$R/driver_support.py" finish --out "$OUT" --root "$ROOT" --phase "$PHASE" --status "$FINAL" || rc=1
  rmdir "$OUT/.lock"
  if [ "$AUTO_SHUTDOWN" = 1 ]; then
    if [ ! -e /usr/bin/shutdown ]; then
      echo '!!! /usr/bin/shutdown 不存在：无法自动关机；请人工确认实例状态。不会尝试其他路径。 !!!'
    else
      echo "Shutdown in 180 seconds; touch $ROOT/p4_r18_no_shutdown to cancel"
      (sleep 180; [ -e "$ROOT/p4_r18_no_shutdown" ] || { sync; /usr/bin/shutdown; }) </dev/null >/dev/null 2>&1 &
    fi
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT TERM
stop() { FINAL="STOPPED: $*"; echo "$FINAL"; exit 1; }
must() { "$@" || stop "command failed: $*"; }
capture() { local file=$1; shift; "$@" | tee "$file" || stop "report failed: $*"; }
if [ "${SKIP_CONDA:-0}" != 1 ]; then
  CONDA_BASE=${CONDA_BASE:-/root/miniconda3}
  source "$CONDA_BASE/etc/profile.d/conda.sh" || stop 'conda setup'
  conda activate rgbt5090 || stop 'conda rgbt5090'
fi
cd "$CODE" || stop 'CODE missing'
echo "STAGE=$STAGE ROOT=$ROOT synthetic=$SYNTHETIC"
if [ "${SKIP_GIT:-0}" != 1 ]; then
  must git rev-parse HEAD
  [ -z "$(git status --porcelain --untracked-files=no)" ] || stop 'Tracked working tree is dirty'
fi
# BEGIN EXPECTED_SHA256 (workspace values must equal committed blob values before push)
R17_SHA=0871f83d715f64fe1099969d0918f9e7348739c8b7a51f71e4c74d34f8163d95
R15_SHA=670a89bed5c5a7a367382bc3d9484c1f38455f828826e6f41711db4674fdb82c
XCAM_SHA=ea2951d63034244ac407418a43b8b8237d2a7e36c1a2eb6052b1800e82b0e61d
while read -r expected relative; do
  actual=$(sha256sum "$CODE/$relative") || stop "Code file missing: $relative"
  [ "${actual%% *}" = "$expected" ] || stop "Code SHA256 differs: $relative"
  echo "CODE_OK $relative $expected"
done <<'R18_HASHES'
16990fc569ba987d41bf9243adc7624961e13d589e7309c0e27d7c64f948a02c  .gitattributes
c50c78a0215f43ad63b5f14a25fe1db0e2f229b34d97d648b36b91d6b55f7bfe  datasets/build.py
21a35ce04ad56b6ba64ebb611b8a732b02f75b97fcf78443fe214a6005127a8c  datasets/rgbnt201pedes.py
3d94e6622fb2dd62f5f305f353308d0cf6ec97649d197130bac8477acd7cdea0  splits/rgbnt201_dev30.json
767580b5acf6ef749596d5cb76a1a63695f360e4387f88c2f6b2194ad5e8711d  splits/rgbnt201_overlap.json
7a2c4aac36ad549fcff30baf8f9c0460a6c4101d5f184efed78c3cfeb7f79540  tests/p4/test_r18_core.py
fef3ad5ee2907a47657380237d97248bda4fbd458db83363934f8f5cb1a13114  tests/p4/test_r18_data.py
152655fe2d8b3665367c4720b0118184020acfe70bd900ab45af88cf2a68b8f4  tests/p4/test_r18_eval.py
e6ce3b30fb0f71fb9760d24eb412404ab2e4175d1fe51d1771e06c0df66d6866  tests/p4/test_r18_execution.py
5bd0069fc2d8f712f1a43e79ab4d71e910c23def5f641ac3dc2c2328f3d8e275  tests/p4/test_r18_runner.py
e387c70c55582e9bacbd8b78f6edb64bc9dcef9b859e91f11ccf33f1392c0a1c  tools/p4/export_dev_records.py
8dc919479ab8922694feea324b020f4665b6e22fbb65b05c195f1a1a27cfd6f0  tools/p4/export_eval_split.py
9b8b64765927ff34ec6dfb26b42d4e08c1a64378a58074210387d8468fb58801  tools/p4/export_text_bank.py
84e9725540f62ba4fdf8896c5e5c40771b49f8d1c09b9b825b76d88d4dacb2fb  tools/p4/export_train_ir_feats.py
15abd8312bafa628cf2192ff1c64b5bc33e167e1f7d8bf5fd0dcbd545b7a8cff  tools/p4/pack_feats.py
8acb3386908d634cb5fc06b2a585e65a67c81a0c32a5ad633d6e69b9f13e5cbe  tools/p4/r18/common.py
d8d8e3fc984329270565fceb60dc6a5cdf8f139481b3a767a45b8700065c3da8  tools/p4/r18/driver_support.py
d12473117d954766d6f3dbe50c33647b2f39d367a20d3fc2199f996e07ca8b7b  tools/p4/r18/local_smoke.py
71c154d51378a0bd6ab23f0ffe28acb001dfc24a64523f3a75e628fe5d1915ba  tools/p4/r18/make_rgbnt201_dev_split.py
60ee079e376124894d61c5577dfb6afdf4feee344fa9e5aaf662b94f2c3c6f4a  tools/p4/r18/overlap_map.py
dded39d3a611e2981c734638811b6fae88a24d813d1746f8ec1f08686db32fb5  tools/p4/r18/package_data.py
b9d0a6d661fd276af410cf715b0f58ead92fb1c0f161a7c5fa51c72c493c493a  tools/p4/r18/r18_core.py
13cf243728da62975339b79fe18e55fba27650f214ccc5f31a5716c1881e7bb7  tools/p4/r18/r18_devsmoke_report.py
7ee81a07e3a6e9777457ba8d2761b213c0d5f19c4b066d9bb43a60fd1e9ac2af  tools/p4/r18/r18_exact.py
89e508c7fcdf3af68a80a2243c60ea7e0739d04704fa032e46c74a5b23dba7fd  tools/p4/r18/r18_subset.py
86326a969104dfa5029aee31dcdea5eed5398fd0bb1677361589265b08a32399  tools/p4/r18/read_e1_core.py
bbb245ff9a0f142eb529c2ecf9328eae517d7f4931fe6cccddabfc5e4ccc0f99  tools/p4/r18/run_frozen.py
7829a0f03266421ad7a657b78dc9c634cf731984cf387748f35e0cc862a22ede  tools/p4/r18/workflow.py
d62102d63a544726821a759b646625b34fbca8a73c0912e920fa81b9f07c1fdf  tools/p4/r18/xcam_core.py
f12ed883cbc0b3639f3578a5cbc505bc5f3a7bfb5d15b14ad43c0fe7cce04d3e  tools/p4/train_acd.py
1375f6add366f72e2ff9677d16b6a84fab950a2ece4d99a94494dd5c123e126a  tools/p4/transfer/judge_test.py
1ba9cd4759b7a1f84b48853bdd127f67f9d6e6b5e179997ac009f447e5ebbf65  tools/p4/transfer/p4_r17.sh
670a89bed5c5a7a367382bc3d9484c1f38455f828826e6f41711db4674fdb82c  tools/p4/transfer/r15_test.py
be788b73ed04d535db8c6f25cfd09f826a1ca3683d3c81cb9f23d97ef2f072af  tools/p4/transfer/r17_check_archives.py
0871f83d715f64fe1099969d0918f9e7348739c8b7a51f71e4c74d34f8163d95  tools/p4/transfer/r17_const.py
3a5d73fcec8cb448cd5bbe2a345a3f35c3c4bddf67c61e78cf69fd41bbc74c57  tools/p4/transfer/r17_judge.py
ad8b01daaa179ecead4b661d0dd99c5d9f5606dd0cf1cf66e8db71285a4f1b52  tools/p4/transfer/r17_reagg_r15.py
ea2951d63034244ac407418a43b8b8237d2a7e36c1a2eb6052b1800e82b0e61d  tools/p4/transfer/r18_xcam.py
e434efe0e44f1fff0dcd613b0195afc268ee5fd6be0e653b54403abf281f7a12  tools/p4/transfer/r18_xcam_judge.py
R18_HASHES
# END EXPECTED_SHA256
echo 'Driver SHA256 (printed only):'; must sha256sum "$R/p4_r18.sh"
must guard
if [ "$SYNTHETIC" = 1 ]; then
  must "$PY" -B "$R/driver_support.py" environment --synthetic --parallel_file "$OUT/checks/parallel.txt" --parallel_json "$OUT/checks/parallel.json"
else
  must "$PY" -B "$R/driver_support.py" environment --parallel_file "$OUT/checks/parallel.txt" --parallel_json "$OUT/checks/parallel.json"
  must nvidia-smi -L
  must df -h "$ROOT"
  must "$PY" -B "$R/driver_support.py" data --root "$DATA_ROOT" --upload "$UPLOAD"
  (cd "$DATA_ROOT" && sha256sum -c --quiet "$UPLOAD/RGBNT201_files.sha256") || stop 'Dataset file verification'
  [ "$(wc -l < "$UPLOAD/RGBNT201_files.sha256")" -eq 9575 ] || stop 'Data count'
  must git check-ignore "$DATA_ROOT/RGBNT201/captionform0_8B.json"
fi
read -r PAR < "$OUT/checks/parallel.txt"
echo "Memory-limited parallel fits: $PAR"

# unit NAME OUTPUT INPUT... -- COMMAND...; every output remains read-only on reuse.
unit() {
  local name=$1 output=$2; shift 2
  local inputs=() cmd=() make=() kind=l1
  case "$name" in test_constrained_*|const_*|xcam_*|subset_test_constrained_*|subset_const_*|devsmoke) make=(--mkdir);; esac
  case "$name" in test_constrained_*|const_*|xcam_*|subset_*|devsmoke) kind=eval;; esac
  while [ "$#" -gt 0 ] && [ "$1" != -- ]; do inputs+=("$1"); shift; done
  if [ "$SYNTHETIC" != 1 ]; then
    inputs+=("$UPLOAD/RGBNT201_files.sha256" "$DATA_ROOT/RGBNT201/captionform0_8B.json")
  fi
  [ "$#" -gt 0 ] || return 2
  shift; cmd=("$@")
  if [ "$STAGE" = L1 ]; then
    local i
    for ((i=0;i<${#cmd[@]}-1;i++)); do
      [ "${cmd[i]} ${cmd[i+1]}" != '--split test' ] || { echo 'L1 test export forbidden'; return 2; }
    done
  fi
  "$PY" -B "$R/workflow.py" stage --manifest "$OUT/.manifests/$name.json" --done "$OUT/.manifests/$name.done" \
    "${VERIFY[@]}" "${make[@]}" --signature_kind "$kind" --inputs "${inputs[@]}" --outputs "$output" -- "${cmd[@]}"
}
find_run() { "$PY" -B "$R/workflow.py" find-run --parent "$1" --name "$2" "${@:3}"; }
export_split() {
  local seed=$1 split=$2 run=$3 name=r201_${2}_s${1}
  must unit "export_$name" "$OUT/exports/$name" "$run/last.pth" "$run/configs.yaml" "$DEV_SPLIT" -- \
    "$PY" -B tools/p4/export_eval_split.py --split "$split" --checkpoint "$run/last.pth" --config_file "$run/configs.yaml" \
    --root_dir "$DATA_ROOT" --dev_split_file "$DEV_SPLIT" --eval_degrade_seed 2026 --device cuda \
    --save_modality_features --save_item_meta --output_dir "$OUT/exports/$name"
  must unit "pack_$name" "$OUT/feats_full/$name" "$OUT/exports/$name" -- \
    "$PY" -B tools/p4/pack_feats.py --export_dirs "$OUT/exports/$name" --names "$name" --out_dir "$OUT/feats_full/$name"
  must unit "pack12_$name" "$OUT/feats/$name.npz" "$OUT/feats_full/$name/$name.npz" -- \
    "$PY" -B "$R/workflow.py" pack12 --input "$OUT/feats_full/$name/$name.npz" --output "$OUT/feats/$name.npz"
}
check_archives() {
  local suffix=$1 pairs=() seed
  for seed in $SEEDS; do pairs+=("s$seed=$OUT/feats/r201_dev_s$seed.npz,$OUT/feats/r201_${suffix}_s$seed.npz"); done
  must "$PY" -B "$T/r17_check_archives.py" --pairs "${pairs[@]}" --assign_seeds $ASSIGN_SEEDS --json "$OUT/checks/archives_$suffix.json"
}

if [ "$STAGE" = L1 ]; then
  [ ! -e "$OUT/.eval_started" ] || stop 'L1 refused after test evaluation started'
  [ "$SYNTHETIC" = 0 ] || stop 'Synthetic mode supports held-out-fixture L2/L2R only'
  S101=$(find_run "$P4_OUT/runs/base_clean_s101/RGBT-PEDES" base_clean_s101 --sha 544971a2777016321a7ff854026ee5916de1aab17513c919238c8976e9cddb49) || stop 's101 base'
  S102=$(find_run "$P4_OUT/runs/base_clean_s102/RGBT-PEDES" base_clean_s102 --sha 799d5cf52d1e2fd1a51fe53dca32937ecf930a57ea8eb696e2d9651143baca44) || stop 's102 base'
  BASE_YAML=$("$PY" -B "$R/driver_support.py" base-config --config "$S102/configs.yaml") || stop 'BASE_YAML'
  must cat "$BASE_YAML"; must sha256sum "$BASE_YAML"
  must "$PY" -B "$R/workflow.py" config-paths --config "$S102/configs.yaml" --out "$OUT/checks/recipe_paths.json"
  must unit regression_export "$OUT/exports/regression" "$S101/last.pth" "$S101/configs.yaml" "$CODE/splits/dev_190.json" -- \
    "$PY" -B tools/p4/export_eval_split.py --split dev --checkpoint "$S101/last.pth" --config_file "$S101/configs.yaml" \
    --root_dir "$DATA_ROOT" --dev_split_file "$CODE/splits/dev_190.json" --eval_degrade_seed 2026 --device cuda --save_modality_features --output_dir "$OUT/exports/regression"
  must unit regression_pack "$OUT/feats_full/regression" "$OUT/exports/regression" -- \
    "$PY" -B tools/p4/pack_feats.py --export_dirs "$OUT/exports/regression" --names regression --out_dir "$OUT/feats_full/regression"
  must unit regression12 "$OUT/feats/regression.npz" "$OUT/feats_full/regression/regression.npz" -- \
    "$PY" -B "$R/workflow.py" pack12 --input "$OUT/feats_full/regression/regression.npz" --output "$OUT/feats/regression.npz"
  must "$PY" -B "$R/workflow.py" compare-npz --first "$OUT/feats/regression.npz" --second "$R13_IN/base_ref_s101.npz" --out "$OUT/checks/regression.json"
  must guard
  must "$PY" -B "$R/driver_support.py" construct --root "$DATA_ROOT" --split "$DEV_SPLIT"
  must unit smoke "$OUT/smoke" "$BASE_YAML" "$DEV_SPLIT" -- env -u OPENBLAS_NUM_THREADS -u OMP_NUM_THREADS -u MKL_NUM_THREADS TRAIN_SEED=101 "$PY" -B train.py \
    --config_file "$BASE_YAML" --dataset_name RGBNT201-PEDES --root_dir "$DATA_ROOT" --dev_split_file "$DEV_SPLIT" \
    --name r201_smoke --output_dir "$OUT/smoke" --smoke_steps 30
  must "$PY" -B "$R/driver_support.py" smoke --directory "$OUT/smoke" --out "$OUT/checks/smoke.json"
  L1FILES=()
  for s in $SEEDS; do
    must unit "train_s$s" "$OUT/runs/r201_base_s$s" "$BASE_YAML" "$DEV_SPLIT" -- \
      timeout 5400 env -u OPENBLAS_NUM_THREADS -u OMP_NUM_THREADS -u MKL_NUM_THREADS TRAIN_SEED="$s" "$PY" -B train.py \
      --config_file "$BASE_YAML" --root_dir "$DATA_ROOT" --dataset_name RGBNT201-PEDES --dev_split_file "$DEV_SPLIT" \
      --name "r201_base_s$s" --output_dir "$OUT/runs/r201_base_s$s"
    RUN=$(find_run "$OUT/runs/r201_base_s$s/RGBNT201-PEDES" "r201_base_s$s") || stop "train run s$s"
    must "$PY" -B "$R/workflow.py" health --run "$RUN" --base "$S102/configs.yaml" --out "$OUT/checks/health_s$s.json"
    must guard
    export_split "$s" dev "$RUN"
    must unit "ir_s$s" "$OUT/train_ir/r201_s${s}_train_ir.npz" "$RUN/last.pth" "$RUN/configs.yaml" "$DEV_SPLIT" "$OUT/feats/r201_dev_s$s.npz" -- \
      "$PY" -B tools/p4/export_train_ir_feats.py --checkpoint "$RUN/last.pth" --config_file "$RUN/configs.yaml" --root_dir "$DATA_ROOT" \
      --dev_split_file "$DEV_SPLIT" --output "$OUT/train_ir/r201_s${s}_train_ir.npz" --verify_dev_npz "$OUT/feats/r201_dev_s$s.npz" --device cuda
    L1FILES+=("$RUN/last.pth" "$OUT/feats/r201_dev_s$s.npz" "$OUT/train_ir/r201_s${s}_train_ir.npz" "$OUT/exports/r201_dev_s$s/item_meta.json")
    must guard
  done
  check_archives dev
  must unit devsmoke "$OUT/devsmoke/s101" "$OUT/feats/r201_dev_s101.npz" "$OUT/train_ir/r201_s101_train_ir.npz" -- \
    "$PY" -B "$R/run_frozen.py" --expect_sha "$R17_SHA" --log "$OUT/devsmoke/s101/degenerate_fits.json" -- "$T/r17_const.py" --b12 . --feats "$OUT/feats" --base r201_dev_s101 --adapter r201_dev_s101 --assign none \
    --out "$OUT/devsmoke/s101" --tag d_smoke_s101 --folds 2 --split_seeds 2026 --assign_seeds 2026 2027 2028 \
    --mixes M0 M4 M6c M1g M4c60 M9 M5e M5f M4id --skip_joint --methods 'B*' QE-DBA QE+M QE+CAET QE+SRC2f QE+EM FeatImp-g A-2f-ctxn \
    --grid_preset v3 --select_rule constrained --r_max 0.999999 --tie_break bstar --link_map train_lda --train_ir_npz "$OUT/train_ir/r201_s101_train_ir.npz" --timing
  must "$PY" -B "$R/r18_devsmoke_report.py" --run "$OUT/devsmoke/s101" --archive "$OUT/feats/r201_dev_s101.npz" --out "$OUT/checks/devsmoke.json"
  must guard
  must "$PY" -B "$R/workflow.py" manifest --out "$OUT/L1_manifest.json" "${L1FILES[@]}" "$UPLOAD/RGBNT201_files.sha256" "$DATA_ROOT/RGBNT201/captionform0_8B.json"
  FINAL=COMPLETE; exit 0
fi

[ -f "$OUT/STATUS_L1" ] && [ "$(cat "$OUT/STATUS_L1")" = COMPLETE ] || stop 'L1 not COMPLETE'
must "$PY" -B "$R/workflow.py" manifest --out "$OUT/L1_manifest.json" --verify
if [ "$STAGE" = L2 ]; then
  [ ! -e "$OUT/.eval_started" ] || stop 'One-shot L2 already started; use L2R for reports'
  if [ "$SYNTHETIC" != 1 ]; then
    for s in $SEEDS; do
      RUN=$(find_run "$OUT/runs/r201_base_s$s/RGBNT201-PEDES" "r201_base_s$s") || stop "run s$s"
      export_split "$s" test "$RUN"
      must unit "cams_s$s" "$OUT/feats/cams_s$s.npz" "$OUT/exports/r201_dev_s$s/item_meta.json" "$OUT/exports/r201_test_s$s/item_meta.json" "$OUT/feats/r201_dev_s$s.npz" "$OUT/feats/r201_test_s$s.npz" "$OUT/feats_full/r201_dev_s$s/r201_dev_s$s.npz" "$OUT/feats_full/r201_test_s$s/r201_test_s$s.npz" -- \
        "$PY" -B "$R/workflow.py" cams --dev_meta "$OUT/exports/r201_dev_s$s/item_meta.json" --test_meta "$OUT/exports/r201_test_s$s/item_meta.json" \
        --dev_archive "$OUT/feats/r201_dev_s$s.npz" --test_archive "$OUT/feats/r201_test_s$s.npz" --output "$OUT/feats/cams_s$s.npz" \
        --dev_full "$OUT/feats_full/r201_dev_s$s/r201_dev_s$s.npz" --test_full "$OUT/feats_full/r201_test_s$s/r201_test_s$s.npz"
    done
  fi
  must guard
  check_archives test
  must touch "$OUT/.eval_started"
else
  [ -e "$OUT/.eval_started" ] || stop 'L2R requires .eval_started'
fi

evaluate_units() {
  local branch=$1 prefix=$2 suffix=$3 s fs n=0 pid failed=0
  local pids=()
  for s in $SEEDS; do for fs in $FIT_SEEDS; do
    local target=$OUT/${prefix}${branch}/s${s}_f${fs} tag script methods expected extra=()
    tag=t_constrained_s${s}_f${fs}; script=$T/r15_test.py; methods=$METHODS; expected=$R15_SHA
    if [ "$branch" = const ]; then
      tag=c_s${s}_f${fs}; script=$T/r17_const.py; methods=$METHODS17; expected=$R17_SHA
      extra=(--fixed_selection "$OUT/${prefix}test/constrained/s${s}_f${fs}/r12_t_constrained_s${s}_f${fs}_selection.json" --const_grid $CONST_GRID --const_curve --perq_ext --timing)
    elif [ "$branch" = xcam ]; then
      tag=x_s${s}_f${fs}; script=$T/r18_xcam.py; methods='B* QE-DBA QE+M QE+M-xcam'; expected=$XCAM_SHA
      extra=(--fixed_selection "$OUT/test/constrained/s${s}_f${fs}/r12_t_constrained_s${s}_f${fs}_selection.json" --item_cams "$OUT/feats/cams_s$s.npz" --perq_ext)
    fi
    local ins=("$OUT/feats/r201_dev_s$s.npz" "$OUT/feats/r201_${suffix}_s$s.npz" "$OUT/train_ir/r201_s${s}_train_ir.npz")
    [ "$branch" = test/constrained ] || ins+=("${extra[1]}")
    [ "$branch" != xcam ] || ins+=("$OUT/feats/cams_s$s.npz")
    local command=("$PY" -B "$R/run_frozen.py" --expect_sha "$expected" --log "$target/degenerate_fits.json" -- "$script" --b12 . --feats "$OUT/feats" --assign none --out "$target" --base "r201_dev_s$s" --adapter "r201_dev_s$s" --tag "$tag" \
      --assign_seeds $ASSIGN_SEEDS --mixes $MIXES --skip_joint --methods $methods --grid_preset "$GRID" --select_rule constrained --r_max 0.999999 --tie_break bstar \
      --test_npz "$OUT/feats/r201_${suffix}_s$s.npz" --link_map train_lda --train_ir_npz "$OUT/train_ir/r201_s${s}_train_ir.npz" --test_seed "$fs" "${extra[@]}")
    local key=${prefix//\//_}${branch//\//_}_s${s}_f${fs}
    if [ "$STAGE" = L2R ]; then
      must unit "$key" "$target" "${ins[@]}" -- "${command[@]}"
    else
      unit "$key" "$target" "${ins[@]}" -- "${command[@]}" > "$OUT/logs/${key}_${TS}.log" 2>&1 &
      pid=$!; pids+=("$pid"); n=$((n+1))
      if [ "$n" -ge "$PAR" ]; then wait -n || { failed=1; break 2; }; n=$((n-1)); fi
    fi
  done; done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [ "$failed" = 0 ] || stop "Evaluation unit failed: $prefix$branch (see unit logs)"
  must guard
}
reports() {
  local scope=$1 prefix=$2 suffix=$3 runs=() qraw=() s fs paths
  for s in $SEEDS; do
    paths=''; for fs in $FIT_SEEDS; do paths=${paths:+$paths,}$OUT/${prefix}test/constrained/s${s}_f${fs}; done
    runs+=("s$s=$paths"); qraw+=("s$s=$OUT/feats/r201_dev_s$s.npz,$OUT/feats/r201_${suffix}_s$s.npz")
  done
  capture "$OUT/verdict_$scope.txt" "$PY" -B "$T/judge_test.py" --runs "${runs[@]}" --qraw "${qraw[@]}" --evalset test --out "$OUT/verdict_$scope.json"
  capture "$OUT/report_const_$scope.txt" "$PY" -B "$T/r17_judge.py" --r17_root "$OUT/${prefix}const" --r15_root "$OUT/${prefix}test/constrained" \
    --seeds $SEEDS --fits $FIT_SEEDS --expect_script_sha "$R17_SHA" --out "$OUT/report_const_$scope.json"
  must "$PY" -B "$R/driver_support.py" reports "$OUT/verdict_$scope.json" "$OUT/report_const_$scope.json"
  must guard
}
evaluate_units test/constrained '' test
evaluate_units const '' test
evaluate_units xcam '' test
reports full '' test
must "$PY" -B "$T/r18_xcam_judge.py" --xcam_root "$OUT/xcam" --r15_root "$OUT/test/constrained" \
  --seeds $SEEDS --fits $FIT_SEEDS --expect_script_sha "$XCAM_SHA" --out "$OUT/report_xcam.json"
must "$PY" -B "$R/driver_support.py" reports "$OUT/report_xcam.json"
must guard
JUDGES=("full=$OUT/verdict_full.json"); R17S=("full=$OUT/report_const_full.json"); CELLS=("full=$OUT/test/constrained")
RUNNERS=("full=$OUT/test/constrained" "full=$OUT/const" "full=$OUT/xcam")
SELECTIONS=(); JUDGE_EXITS=(full=0)
NOVERLAP=$("$PY" -B -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["overlap_test_ids"]))' "$OVERLAP") || stop 'Overlap JSON'
if [ "$NOVERLAP" -gt 0 ]; then
  COMPLETE_SUBSET=1
  if [ "$STAGE" = L2 ]; then
    for s in $SEEDS; do
      must unit "subset_s$s" "$OUT/feats/r201_testnov_s$s.npz" "$OUT/feats/r201_test_s$s.npz" "$OVERLAP" -- \
        "$PY" -B "$R/r18_subset.py" --input "$OUT/feats/r201_test_s$s.npz" --overlap "$OVERLAP" --output "$OUT/feats/r201_testnov_s$s.npz"
    done
    check_archives testnov
  else
    for s in $SEEDS; do for fs in $FIT_SEEDS; do for branch in test_constrained const; do
      [ -e "$OUT/.manifests/subset_${branch}_s${s}_f${fs}.done" ] || COMPLETE_SUBSET=0
    done; done; done
    # Any completed subset unit is still checked, even if the subset stopped partway.
    if [ "$COMPLETE_SUBSET" = 0 ]; then
      while IFS= read -r manifest; do
        must "$PY" -B "$R/workflow.py" verify-manifest --manifest "$manifest"
      done < <(find "$OUT/.manifests" -name 'subset_*_f*.json' -type f)
      echo 'Subset incomplete: existing units verified; no subset rescoring or new subset report'
    fi
  fi
  if [ "$COMPLETE_SUBSET" = 1 ]; then
    evaluate_units test/constrained subset/ testnov
    evaluate_units const subset/ testnov
    reports subset subset/ testnov
    must "$PY" -B "$R/workflow.py" selection --full "$OUT/test/constrained" --subset "$OUT/subset/test/constrained" --out "$OUT/checks/selection_equal.json"
    JUDGES+=("subset=$OUT/verdict_subset.json"); R17S+=("subset=$OUT/report_const_subset.json"); CELLS+=("subset=$OUT/subset/test/constrained")
    RUNNERS+=("subset=$OUT/subset/test/constrained" "subset=$OUT/subset/const")
    SELECTIONS+=("subset=$OUT/checks/selection_equal.json"); JUDGE_EXITS+=(subset=0)
  fi
else
  echo 'No overlap test IDs; no subset generated'
fi
must "$PY" -B "$R/r18_exact.py" --judge "${JUDGES[@]}" --r17 "${R17S[@]}" --xcam "$OUT/report_xcam.json" --cells "${CELLS[@]}" --runner_roots "${RUNNERS[@]}" --seeds $SEEDS --out "$OUT/R18J.txt" --core_out "$OUT/R18J_core.txt" --selection_equal "${SELECTIONS[@]}" --judge_exit_status "${JUDGE_EXITS[@]}"
must guard
FINAL=COMPLETE
