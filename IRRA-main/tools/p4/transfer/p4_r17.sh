# 论文4 第十七轮：只读核查 + 常数迁移对照（事后补充对照，同一测试集，非盲）+ 效率计时 + 开发集敏感性，一次性驱动
# 规格：项目文档《论文4_第十七轮_独立裁决与收尾计划》§3（跑前写定）。第十五轮的 C1–C5 判定不变；p4_r15_out 只读，任何文件都不改。
# 顺序：0 环境与代码核对 -> 1 只读核查（特征档案键名与退化数组、第十五轮记录的显式对齐汇总）-> 2 常数迁移对照（3 个 base × 3 次拟合）
#       -> 3 报告 -> 4 开发集敏感性（关联温度、WCCN 收缩）-> 5 报告（含敏感性）-> 6 核对 p4_r15_out 未被改动、打包。
# 断点续跑：已完成的单元（目录里有 .done）跳过；失败的单元下次运行时移到 _failed_<时间> 后重跑。失败即停：一旦有单元失败，不再启动新单元。
# 只用 CPU。默认跑完不关机；要自动关机，运行前 export AUTO_SHUTDOWN=1。
ROOT=${P4_ROOT:-/root/autodl-tmp}
exec 9>$ROOT/.p4_r17.lock
flock -n 9 || { echo "已有一个 p4_r17 任务在运行，本次退出"; exit 3; }
if [ "${SKIP_CONDA:-0}" != 1 ]; then
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate rgbt5090 || { echo "conda 环境 rgbt5090 激活失败"; exit 1; }
fi
set -uo pipefail
set -f   # 关闭通配符展开：方法名 B* 原样传给脚本
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GIT_TERMINAL_PROMPT=0
PY=${PYTHON:-python}
REPO=${P4_REPO:-$ROOT/IRRA-p4}
CODE=$REPO/IRRA-main
R13_IN=$ROOT/r13_inputs
TIR13=$ROOT/p4_r13_out/train_ir
R15=$ROOT/p4_r15_out
OUT=$ROOT/p4_r17_out
T=tools/p4/transfer
SHA_R15=670a89bed5c5a7a367382bc3d9484c1f38455f828826e6f41711db4674fdb82c
SHA_JUDGE15=1375f6add366f72e2ff9677d16b6a84fab950a2ece4d99a94494dd5c123e126a
SHA_R17=4a6b59f69479ee80b137549ba47c3d9c1deff394996ce7599ba7afb6f0d2fed7
SHA_R17_JUDGE=19b06b133c939de73977dff66123ae7e86cc02dadf94a664df89d29fd725fcfa
SHA_R17_REAGG=ad8b01daaa179ecead4b661d0dd99c5d9f5606dd0cf1cf66e8db71285a4f1b52
SHA_R17_ARCH=53220b00f197a7bb890d2a3b9258d64833b0844b1846bd228f024d1b60011429
SEEDS=${SEEDS:-'101 102 103'}
FIT_SEEDS=${FIT_SEEDS:-'7001 7002 7003'}
ASSIGN_SEEDS=${ASSIGN_SEEDS:-'7031 7032 7033 7034 7035'}
MIXES=${MIXES:-'M0 M4 M6c M1g M4c60 M9 M5e M5f M4id M10 M4c25 M4c50 M4c75'}
GRID=${GRID:-v3}
CONST_GRID='0 0.05 0.1 0.2 0.25 0.5 0.75 1'
METHODS17='B* QE-DBA QE+M QE+CAET QE+SRC2f QE+EM QE+Const QE+NoGate'
DEV_SPLITS=${DEV_SPLITS:-'2026 2027 2028 2029 2030'}
DEV_ASEEDS=${DEV_ASEEDS:-'2026 2027 2028'}
DEV_MIXES='M4 M6c M1g M4c60 M9 M5e M5f M4id'
DEVSETS="ref: ka10:--kappa_a_fixed=10 ka20:--kappa_a_fixed=20 ka45:--kappa_a_fixed=45 ka80:--kappa_a_fixed=80 shr025fit:--wccn_shrink=0.25 shr075fit:--wccn_shrink=0.75 shr025inf:--wccn_shrink_eval=0.25 shr075inf:--wccn_shrink_eval=0.75"
JOB_GB=${JOB_GB:-8}
DEV_JOB_GB=${DEV_JOB_GB:-4}
RUN_DEVSENS=${RUN_DEVSENS:-1}
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}
SHUTDOWN_GRACE_MIN=3
NO_SHUTDOWN=$ROOT/p4_r17_no_shutdown
TS=$(date +%m%d_%H%M%S)
mkdir -p $OUT/logs $OUT/checks $OUT/reagg $OUT/const $OUT/devsens
LOG=$OUT/logs/driver.log
SUM=$OUT/logs/summary.txt
DIGEST=$OUT/DIGEST17.txt
[ -e $SUM ] && mv $SUM $SUM.prev_$TS
touch $OUT/logs/.t0_$TS
rm -f $NO_SHUTDOWN
mask() { sed -E 's#https://[^@/ ]+@#https://***@#g'; }
say()  { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
note() { echo "$*" >> $SUM; say "$*"; }
run()  { local name=$1; shift; say "START $name"; "$@" > $OUT/logs/$name.log 2>&1; local rc=$?; say "END   $name exit=$rc"; echo "$name exit=$rc" >> $SUM; return $rc; }
devnpz() { [ $1 = 103 ] && echo $R15/feats/base_ref_s103.npz || echo $R13_IN/base_ref_s$1.npz; }
devdir() { [ $1 = 103 ] && echo $R15/feats || echo $R13_IN; }
trainir() { [ $1 = 103 ] && echo $R15/train_ir/base_ref_s103_train_ir.npz || echo $TIR13/base_ref_s$1_train_ir.npz; }
r15dir() { echo $R15/test/constrained/s$1_f$2; }
r15file() { local p; p=$(find $(r15dir $1 $2) -maxdepth 1 -name 'r12_*_summary.csv' 2>/dev/null | sort | head -n 1); [ -n "$p" ] && echo ${p%summary.csv}$3; }   # set -f：用 find，不用通配符
write_digest() {
  {
    echo "################ 论文4 第十七轮 摘要（$(date '+%F %T')）################"
    echo "（常数迁移对照是看过测试结果之后的事后补充对照：同一测试集、非盲；第十五轮 C1–C5 原判定不变）"
    echo; echo "======== 运行记录与核对 ========"; cat $SUM 2>/dev/null
    [ -f $OUT/checks/archives.txt ] && { echo; echo "======== 1a 特征档案只读核查 ========"; cat $OUT/checks/archives.txt; }
    if [ -f $OUT/reagg/reagg_r15.txt ]; then
      echo; echo "======== 1b 第十五轮已保存记录的显式对齐汇总（结构、对齐与配对差；完整表见 reagg/reagg_r15.txt）========"
      grep -E '^######## |\[structure\]|\[status\]|^    QE\+M - [A-Za-z+-]+ +(A|B) |\[flips\]|^    (A|B): pairs' $OUT/reagg/reagg_r15.txt
    fi
    [ -f $OUT/report_const.txt ] && { echo; echo "======== 2-5 常数迁移对照与敏感性报告 ========"; cat $OUT/report_const.txt; }
    for f in $(find $OUT/const $OUT/devsens -mindepth 2 -maxdepth 2 -name run.log 2>/dev/null | grep -v _failed_ | sort); do
      [ -e $(dirname $f)/.done ] || { echo; echo "--- 未正常结束的单元：$f"; tail -n 25 $f | mask; }
    done
    if grep -q '^STOPPED' $SUM 2>/dev/null; then
      echo; echo "======== 停止处的日志末尾 ========"
      for f in $(find $OUT/logs $OUT/const $OUT/devsens -name '*.log' -newer $OUT/logs/.t0_$TS -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -3 | cut -d' ' -f2); do echo "--- $f"; tail -n 30 $f | mask; done
    fi
  } > $DIGEST 2>&1
}
power_off() {
  [ "$AUTO_SHUTDOWN" = 1 ] || { say "按要求不关机。结果：cat $DIGEST ；打包：$ROOT/p4_r17_results.tar.gz"; return 0; }
  say "将在 $SHUTDOWN_GRACE_MIN 分钟后自动关机；要取消，执行 touch $NO_SHUTDOWN"
  sleep $(( SHUTDOWN_GRACE_MIN * 60 ))
  if [ -e $NO_SHUTDOWN ]; then say "检测到 $NO_SHUTDOWN：取消关机"; return 0; fi
  sync; /usr/bin/shutdown
}
package() {
  local extra=""; for x in report_const.txt report_const.json; do [ -f $OUT/$x ] && extra="$extra $x"; done
  tar -czf $ROOT/p4_r17_results.tar.gz -C $OUT --exclude='*perq*.pkl' DIGEST17.txt logs checks reagg const devsens $extra 2>> $LOG \
    && say "打包：$ROOT/p4_r17_results.tar.gz（$(du -h $ROOT/p4_r17_results.tar.gz | cut -f1)，不含每查询记录 pkl；pkl 留在 $OUT）"
}
finish() { write_digest; package; say "摘要：$DIGEST"; sync; power_off; }
stop()   { note "STOPPED: $*"; finish; exit 1; }
memcpu() {
$PY -B -c "
import math, os
def rd(p):
    try:
        return open(p).read().strip()
    except Exception:
        return ''
av = 0
for l in open('/proc/meminfo'):
    if l.startswith('MemAvailable:'):
        av = int(l.split()[1]) * 1024
lim = rd('/sys/fs/cgroup/memory.max') or rd('/sys/fs/cgroup/memory/memory.limit_in_bytes')
use = rd('/sys/fs/cgroup/memory.current') or rd('/sys/fs/cgroup/memory/memory.usage_in_bytes')
if lim.isdigit() and int(lim) < 2 ** 60 and use.isdigit():
    av = min(av, int(lim) - int(use))
cpus = len(os.sched_getaffinity(0))
q = rd('/sys/fs/cgroup/cpu.max').split()
qa, pa = (q if len(q) == 2 else [rd('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'), rd('/sys/fs/cgroup/cpu/cpu.cfs_period_us')])
if qa.isdigit() and pa.isdigit() and int(qa) > 0:
    cpus = min(cpus, max(1, math.ceil(int(qa) / int(pa))))
print(max(0, av) // 2 ** 30, cpus)
"
}
r15sha() { { find $R15/test -type f \( -name '*.csv' -o -name '*.pkl' -o -name '*.json' -o -name 'run.log' \) -print0; find $R15 -maxdepth 1 -type f \( -name 'DIGEST*.txt' -o -name 'verdict*' \) -print0; } 2>/dev/null | sort -z | xargs -0 sha256sum; }

say "== 0 环境与代码核对 =="
{ df -h $ROOT; free -g; $PY -V; } >> $LOG 2>&1
read -r MEMGB NCPU <<< "$(memcpu)"; [ -n "$MEMGB" ] || MEMGB=0; [ -n "$NCPU" ] || NCPU=1
[ "$MEMGB" -ge "$JOB_GB" ] || stop "可用内存约 ${MEMGB} GB，不足一个评估进程所需的 ${JOB_GB} GB：请用有卡模式（正常模式）开机后再运行；无卡模式只有约 2 GB 内存"
FREEGB=$(df -BG --output=avail $ROOT | tail -n 1 | tr -dc 0-9)
[ "${FREEGB:-0}" -ge 5 ] || stop "数据盘剩余 ${FREEGB} GB，不足 5 GB"
cd $CODE || stop "找不到代码目录 $CODE"
if [ "${SKIP_GIT:-0}" != 1 ]; then
  HEAD=$(git -C $REPO rev-parse HEAD); BR=$(git -C $REPO rev-parse --abbrev-ref HEAD)
  [ -z "$(git -C $REPO status --porcelain --untracked-files=no)" ] || stop "仓库已跟踪文件有改动"
  note "代码：$BR @ $HEAD，工作区干净"
fi
for pair in r15_test.py:$SHA_R15 judge_test.py:$SHA_JUDGE15 r17_const.py:$SHA_R17 r17_judge.py:$SHA_R17_JUDGE r17_reagg_r15.py:$SHA_R17_REAGG r17_check_archives.py:$SHA_R17_ARCH; do
  IFS=: read -r f h <<< "$pair"
  [ "$(sha256sum $T/$f | cut -d' ' -f1)" = "$h" ] || stop "$f 的 sha256 不符（期望 ${h:0:12}）"
done
note "脚本哈希核对通过：r15_test.py ${SHA_R15:0:12}（冻结，未改）| judge_test.py ${SHA_JUDGE15:0:12}（冻结，未改）| r17_const.py ${SHA_R17:0:12} | r17_judge.py ${SHA_R17_JUDGE:0:12} | r17_reagg_r15.py ${SHA_R17_REAGG:0:12} | r17_check_archives.py ${SHA_R17_ARCH:0:12}"
MISS=""
for s in $SEEDS; do
  for f in $(devnpz $s) $R15/feats/test_base_s$s.npz $(trainir $s); do [ -f $f ] || MISS="$MISS $f"; done
  for fs in $FIT_SEEDS; do for x in summary.csv cells.csv selection.json perq.pkl run_record.json; do
    p=$(r15file $s $fs $x); [ -n "$p" ] && [ -f "$p" ] || MISS="$MISS $(r15dir $s $fs)/r12_*_$x"
  done; done
done
[ -z "$MISS" ] || stop "缺少输入：$MISS"
[ -f $OUT/checks/r15_sha_before.txt ] || r15sha > $OUT/checks/r15_sha_before.txt
note "输入齐全：3 个 base 的开发档案、测试档案、训练身份档案，以及第十五轮 9 个拟合目录；第十五轮输出的哈希清单已记录（$(grep -c '' $OUT/checks/r15_sha_before.txt) 个文件）"
note "资源：可用内存约 ${MEMGB} GB，CPU ${NCPU} 核，数据盘剩余 ${FREEGB} GB"

say "== 1 只读核查 =="
PAIRS=""; QR=""
for s in $SEEDS; do PAIRS="$PAIRS s$s=$(devnpz $s),$R15/feats/test_base_s$s.npz"; QR="$QR s$s=$(devnpz $s),$R15/feats/test_base_s$s.npz"; done
$PY -B $T/r17_check_archives.py --pairs $PAIRS --assign_seeds $ASSIGN_SEEDS --json $OUT/checks/archives.json > $OUT/checks/archives.txt 2>&1; RCA=$?
echo "check_archives exit=$RCA" >> $SUM
if [ $RCA = 0 ]; then
  note "1a 特征档案：三个种子的开发与测试档案中，6 种退化各自改变的数组都存在、且与干净数组不同；合并后的键集合完整（直接读到原始档案）"
elif [ $RCA = 2 ] && [ "${ALLOW_ARCHIVE_PROBLEM:-0}" != 1 ]; then
  stop "1a 特征档案核查发现缺失或异常的退化数组（见 checks/archives.txt）。按计划先报告影响范围，不继续算新对照；确认处理方式后用 ALLOW_ARCHIVE_PROBLEM=1 续跑"
else
  note "WARNING: 1a 特征档案核查 exit=$RCA（ALLOW_ARCHIVE_PROBLEM=1 时继续）"
fi
$PY -B $T/r17_reagg_r15.py --r15_root $R15/test/constrained --qraw $QR --fits $FIT_SEEDS --json $OUT/reagg/reagg_r15.json > $OUT/reagg/reagg_r15.txt 2>&1; RCR=$?
echo "reagg_r15 exit=$RCR" >> $SUM
[ $RCR = 0 ] && note "1b 第十五轮已保存记录：结构与显式对齐核对通过（从保存的逐查询记录与 cells.csv 汇总，不打新分）" || note "WARNING: 1b 第十五轮记录汇总 exit=$RCR，见 reagg/reagg_r15.txt"

say "== 2 常数迁移对照（3 个 base × 3 次拟合；前端配置取第十五轮 selection.json，不重新择优）=="
read -r MEMGB NCPU <<< "$(memcpu)"; [ -n "$MEMGB" ] || MEMGB=0; [ -n "$NCPU" ] || NCPU=1
PAR=$(( MEMGB / JOB_GB )); NC=$(( NCPU > 1 ? NCPU - 1 : 1 )); [ $PAR -gt $NC ] && PAR=$NC; [ $PAR -lt 1 ] && PAR=1
FAILFLAG=$OUT/logs/.fail_$TS; rm -f $FAILFLAG
runconst() {   # 参数：种子 拟合种子
  local s=$1 fs=$2 d=$OUT/const/s$1_f$2
  mkdir -p $d
  $PY -B $T/r17_const.py --b12 . --feats $(devdir $s) --assign none --out $d --base base_ref_s$s --adapter base_ref_s$s --tag c_s${s}_f$fs \
    --assign_seeds $ASSIGN_SEEDS --mixes $MIXES --skip_joint --methods $METHODS17 --grid_preset $GRID --select_rule constrained \
    --r_max 0.999999 --tie_break bstar --test_npz $R15/feats/test_base_s$s.npz --link_map train_lda --train_ir_npz $(trainir $s) \
    --test_seed $fs --fixed_selection $(r15file $s $fs selection.json) --const_grid $CONST_GRID --const_curve --perq_ext --timing > $d/run.log 2>&1
  local rc=$?
  if [ $rc = 0 ] && grep -q '^total time' $d/run.log; then touch $d/.done; else touch $FAILFLAG; fi
  echo "const s$s f$fs exit=$rc" >> $OUT/logs/const_exit.txt
}
NJ=0; NSKIP=0
for s in $SEEDS; do for fs in $FIT_SEEDS; do
  d=$OUT/const/s${s}_f$fs
  if [ -e $d/.done ]; then NSKIP=$(( NSKIP + 1 )); continue; fi
  [ -d $d ] && mv $d $OUT/const/_failed_${TS}_s${s}_f$fs
  [ -e $FAILFLAG ] && break 2
  while [ "$(jobs -rp | wc -l)" -ge $PAR ]; do sleep 10; [ -e $FAILFLAG ] && break; done
  [ -e $FAILFLAG ] && break 2
  say "START const s$s f$fs"; runconst $s $fs & NJ=$(( NJ + 1 ))
done; done
wait
NDONE=$(find $OUT/const -mindepth 2 -maxdepth 2 -name .done | grep -v _failed_ | wc -l)
note "2 常数迁移对照：本次启动 $NJ 个进程（并行 $PAR），跳过已完成 $NSKIP 个；已完成 $NDONE / 9"
[ -e $FAILFLAG ] && stop "常数迁移对照有进程失败（见 const/*/run.log），未启动的单元不再启动；修正后重新运行本脚本即可续跑"

say "== 3 报告（常数迁移对照）=="
$PY -B $T/r17_judge.py --r17_root $OUT/const --r15_root $R15/test/constrained --seeds $SEEDS --fits $FIT_SEEDS --out $OUT/report_const.json > $OUT/report_const.txt 2>&1; RCJ=$?
echo "judge exit=$RCJ" >> $SUM
[ $RCJ = 0 ] && note "3 报告：结构核对与第十五轮复现核对通过（新算的 B★、QE-DBA、QE+M、QE+CAET、QE+SRC2f、QE+EM 与冻结结果一致）" || note "WARNING: 报告 exit=$RCJ：结构或复现核对未通过，新对照不能与冻结结果配对（见 report_const.txt）"
write_digest

if [ "$RUN_DEVSENS" = 1 ]; then
  say "== 4 开发集敏感性（开发集半图库，B★ 前端上的迁移层；不回改最终方法）=="
  read -r MEMGB NCPU <<< "$(memcpu)"; [ -n "$MEMGB" ] || MEMGB=0; [ -n "$NCPU" ] || NCPU=1
  PAR=$(( MEMGB / DEV_JOB_GB )); NC=$(( NCPU > 1 ? NCPU - 1 : 1 )); [ $PAR -gt $NC ] && PAR=$NC; [ $PAR -lt 1 ] && PAR=1
  FAILDEV=$OUT/logs/.faildev_$TS; rm -f $FAILDEV
  rundev() {   # 参数：设置名 额外参数 种子
    local name=$1 extra=$2 s=$3 d=$OUT/devsens/$1_s$3
    mkdir -p $d
    $PY -B $T/r17_const.py --b12 . --feats $(devdir $s) --assign none --out $d --base base_ref_s$s --adapter base_ref_s$s --tag d_${name}_s$s \
      --folds 2 --split_seeds $DEV_SPLITS --assign_seeds $DEV_ASEEDS --mixes $DEV_MIXES --skip_joint --methods 'B*' A-2f-ctxn --grid_preset $GRID \
      --select_rule constrained --r_max 0.999999 --tie_break bstar --link_map train_lda --train_ir_npz $(trainir $s) ${extra//=/ } > $d/run.log 2>&1
    local rc=$?
    if [ $rc = 0 ] && grep -q '^total time' $d/run.log; then touch $d/.done; else touch $FAILDEV; fi
    echo "devsens $name s$s exit=$rc" >> $OUT/logs/devsens_exit.txt
  }
  NJ=0; NSKIP=0
  for spec in $DEVSETS; do
    name=${spec%%:*}; extra=${spec#*:}
    for s in $SEEDS; do
      d=$OUT/devsens/${name}_s$s
      if [ -e $d/.done ]; then NSKIP=$(( NSKIP + 1 )); continue; fi
      [ -d $d ] && mv $d $OUT/devsens/_failed_${TS}_${name}_s$s
      [ -e $FAILDEV ] && break 2
      while [ "$(jobs -rp | wc -l)" -ge $PAR ]; do sleep 10; [ -e $FAILDEV ] && break; done
      [ -e $FAILDEV ] && break 2
      say "START devsens $name s$s"; rundev $name "$extra" $s & NJ=$(( NJ + 1 ))
    done
  done
  wait
  note "4 开发集敏感性：本次启动 $NJ 个进程（并行 $PAR），跳过已完成 $NSKIP 个"
  [ -e $FAILDEV ] && note "WARNING: 开发集敏感性有进程失败（见 devsens/*/run.log）；主对照结果不受影响，重新运行本脚本可续跑"
  say "== 5 报告（含敏感性）=="
  $PY -B $T/r17_judge.py --r17_root $OUT/const --r15_root $R15/test/constrained --seeds $SEEDS --fits $FIT_SEEDS --out $OUT/report_const.json \
    --devsens_root $OUT/devsens > $OUT/report_const.txt 2>&1; echo "judge+devsens exit=$?" >> $SUM
fi

say "== 6 第十五轮输出未被改动的核对 =="
r15sha > $OUT/checks/r15_sha_after.txt
if cmp -s $OUT/checks/r15_sha_before.txt $OUT/checks/r15_sha_after.txt; then note "6 p4_r15_out：前后哈希清单逐行相同（未被改动）"; else note "WARNING: p4_r15_out 的哈希清单前后不同（见 checks/r15_sha_*.txt）"; fi
note "ALL DONE"
finish
