# 论文4 第十七轮：只读核查 + 常数迁移对照（事后补充对照，同一测试集，非盲）+ 效率计时 + 开发集敏感性，一次性驱动（v3）
# 规格：项目文档《论文4_第十七轮_独立裁决与收尾计划》§3（跑前写定）。第十五轮的 C1–C5 判定不变。
# 顺序：0 环境、代码与输入核对 -> 1a 特征档案只读核查 -> 1b 第十五轮记录的显式对齐汇总 -> 2 常数迁移对照（3 个 base × 3 次拟合）
#       -> 3 结构与复现核对、报告 -> 4 开发集敏感性 -> 5 敏感性报告（单独文件）-> 6 终检第十五轮文件、打包。
# 失败即停：0–3 任一步失败（1a 档案有问题、1b 旧记录对不齐、2 有单元失败、3 结构或复现不通过），或任何一次核对发现第十五轮文件的
#   哈希清单变了，立即停止：不再启动后续步骤，日志全部保留，STATUS 写 STOPPED，退出码 1。
#   开发集敏感性（4、5）失败不改变 0–3 的结果，但不算全部完成：STATUS 写 INCOMPLETE: DEV SENSITIVITY FAILED，退出码 4。
#   全部步骤通过才写 COMPLETE（退出码 0）；RUN_DEVSENS=0 时写 INCOMPLETE: DEV SENSITIVITY NOT RUN（退出码 5）；
#   已有一个本任务在运行时直接退出（退出码 3，不改动 STATUS）。conda 激活失败也写 STATUS（STOPPED）。
# 断点续跑：每个计算单元（2 的 9 个、4 的 27 个）开算前写 manifest.json：r17_const.py 的 sha256、全部输入文件的 sha256 与大小、
#   完整参数、Python 与库版本、线程设置；成功结束才写 .done（内容为 manifest 的 sha256）。重跑时，只有 .done 存在且目录里的
#   manifest 与本次应有的逐字节相同才复用；否则把旧目录整体移到 _stale/<单元>_<时间>（保留，不删除）再重算。
#   1a、1b、3、5 只读且快，每次都重算；上一次的报告先移到 _history/<时间>/。
# 只写 p4_r17_out（它与 p4_r15_out 互不包含，否则拒绝运行）；p4_r15_out 只读：开跑前记录其哈希清单（test/、feats/、train_ir/
#   的全部文件与顶层文件，不含 JupyterLab 打开文件时自动生成的 .ipynb_checkpoints），1b、2、3、4–5 之后和结束时逐行比较，
#   有任何文件改动、缺失或新增即停。清单原子写入（先写临时文件，完成后改名）。
# 只用 CPU。默认跑完不关机；要自动关机，运行前 export AUTO_SHUTDOWN=1。
ROOT=$(realpath -ms "${P4_ROOT:-/root/autodl-tmp}")   # 绝对路径（步骤 0 之后工作目录是代码目录）
R15=$ROOT/p4_r15_out
OUT=$ROOT/p4_r17_out
SELF_SHA=$(sha256sum "$0" 2>/dev/null | cut -d' ' -f1)
ROUT=$(realpath -m "$OUT"); RR15=$(realpath -m "$R15")
case "$ROUT/" in "$RR15/"*) echo "输出目录 $OUT（$ROUT）位于 $R15 之内：拒绝运行"; exit 1;; esac
case "$RR15/" in "$ROUT/"*) echo "$R15（$RR15）位于输出目录 $OUT 之内：拒绝运行"; exit 1;; esac
exec 9>$ROOT/.p4_r17.lock || { echo "无法在 $ROOT 创建锁文件"; exit 1; }
flock -n 9 || { echo "已有一个 p4_r17 任务在运行，本次退出（退出码 3）"; exit 3; }
PY=${PYTHON:-python}
REPO=$(realpath -ms "${P4_REPO:-$ROOT/IRRA-p4}")
CODE=$REPO/IRRA-main
R13_IN=$ROOT/r13_inputs
TIR13=$ROOT/p4_r13_out/train_ir
T=tools/p4/transfer
SHA_R15=670a89bed5c5a7a367382bc3d9484c1f38455f828826e6f41711db4674fdb82c
SHA_JUDGE15=1375f6add366f72e2ff9677d16b6a84fab950a2ece4d99a94494dd5c123e126a
SHA_R17=0871f83d715f64fe1099969d0918f9e7348739c8b7a51f71e4c74d34f8163d95
SHA_R17_JUDGE=3a5d73fcec8cb448cd5bbe2a345a3f35c3c4bddf67c61e78cf69fd41bbc74c57
SHA_R17_REAGG=ad8b01daaa179ecead4b661d0dd99c5d9f5606dd0cf1cf66e8db71285a4f1b52
SHA_R17_ARCH=be788b73ed04d535db8c6f25cfd09f826a1ca3683d3c81cb9f23d97ef2f072af
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
mkdir -p $OUT/logs $OUT/checks $OUT/reagg $OUT/const $OUT/devsens $OUT/_stale $OUT/_history || { echo "无法创建 $OUT"; exit 1; }
LOG=$OUT/logs/driver.log
SUM=$OUT/logs/summary.txt
DIGEST=$OUT/DIGEST17.txt
STATUSF=$OUT/STATUS
GUARD0=$OUT/checks/r15_guard_before.sha256
GUARDN=$OUT/checks/r15_guard_now.sha256
MFDIR=$OUT/logs/manifests_$TS
for f in STATUS DIGEST17.txt logs/summary.txt checks/archives.txt checks/archives.json reagg/reagg_r15.txt reagg/reagg_r15.json \
         report_const.txt report_const.json report_devsens.txt report_devsens.json; do   # 上一次运行的报告移到 _history，不覆盖
  [ -e $OUT/$f ] && { mkdir -p $(dirname $OUT/_history/$TS/$f); mv $OUT/$f $OUT/_history/$TS/$f; }
done
mkdir -p $MFDIR; touch $OUT/logs/.t0_$TS; rm -f $NO_SHUTDOWN
mask() { sed -E 's#https://[^@/ ]+@#https://***@#g'; }
say()  { echo "[$(date '+%F %T')] $*" | tee -a $LOG; }
note() { echo "$*" >> $SUM; say "$*"; }
setstatus() { echo "$*" > $STATUSF; echo "[$(date '+%F %T')] $*" >> $OUT/logs/status_history.txt; }
devnpz() { [ $1 = 103 ] && echo $R15/feats/base_ref_s103.npz || echo $R13_IN/base_ref_s$1.npz; }
devdir() { [ $1 = 103 ] && echo $R15/feats || echo $R13_IN; }
trainir() { [ $1 = 103 ] && echo $R15/train_ir/base_ref_s103_train_ir.npz || echo $TIR13/base_ref_s$1_train_ir.npz; }
r15dir() { echo $R15/test/constrained/s$1_f$2; }
r15file() { local p; p=$(find $(r15dir $1 $2) -maxdepth 1 -name 'r12_*_summary.csv' 2>/dev/null | sort | head -n 1); [ -n "$p" ] && echo ${p%summary.csv}$3; }   # set -f：用 find，不用通配符
write_digest() {
  {
    echo "################ 论文4 第十七轮 摘要（$(date '+%F %T')，运行 $TS）################"
    echo "状态：$(cat $STATUSF 2>/dev/null)"
    echo "退出码：0 = COMPLETE 全部完成；1 = STOPPED 已停止（原因见下）；3 = 另有任务在运行；4 = INCOMPLETE 步骤 0–3 通过、开发集敏感性失败；5 = INCOMPLETE 步骤 0–3 通过、未跑开发集敏感性。只有 COMPLETE 才算全部完成"
    echo "（常数迁移对照是看过测试结果之后的事后补充对照：同一测试集、非盲；第十五轮 C1–C5 原判定不变）"
    echo; echo "======== 运行记录与核对 ========"; cat $SUM 2>/dev/null
    [ -f $OUT/checks/r15_guard_diff_$TS.txt ] && { echo; echo "======== 第十五轮文件哈希清单的差异（前 40 行）========"; head -n 40 $OUT/checks/r15_guard_diff_$TS.txt; }
    [ -f $OUT/checks/archives.txt ] && { echo; echo "======== 1a 特征档案只读核查 ========"; cat $OUT/checks/archives.txt; }
    if [ -f $OUT/reagg/reagg_r15.txt ]; then
      echo; echo "======== 1b 第十五轮已保存记录的显式对齐汇总（结构、对齐与配对差；完整表见 reagg/reagg_r15.txt）========"
      grep -E '^######## |\[structure\]|\[status\]|^    QE\+M - [A-Za-z+-]+ +(A|B) |\[flips\]|^    (A|B): pairs' $OUT/reagg/reagg_r15.txt
    fi
    [ -f $OUT/report_const.txt ] && { echo; echo "======== 2–3 常数迁移对照：结构与复现核对、报告 ========"; cat $OUT/report_const.txt; }
    [ -f $OUT/report_devsens.txt ] && { echo; echo "======== 4–5 开发集敏感性报告 ========"; cat $OUT/report_devsens.txt; }
    for f in $(find $OUT/const $OUT/devsens -mindepth 2 -maxdepth 2 -name run.log 2>/dev/null | sort); do
      [ -e $(dirname $f)/.done ] || { echo; echo "--- 未正常结束的单元：$f"; tail -n 25 $f | mask; }
    done
    if grep -q '^STOPPED' $SUM 2>/dev/null; then
      echo; echo "======== 停止处的日志末尾 ========"
      for f in $(find $OUT/logs $OUT/checks $OUT/reagg $OUT/const $OUT/devsens -maxdepth 2 -type f \( -name '*.log' -o -name '*.txt' \) ! -name driver.log \
                 ! -name summary.txt ! -name status_history.txt -newer $OUT/logs/.t0_$TS -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -3 | cut -d' ' -f2); do
        echo "--- $f"; tail -n 30 $f | mask
      done
    fi
  } > $DIGEST 2>&1
}
power_off() {
  [ "$AUTO_SHUTDOWN" = 1 ] || { say "按要求不关机。状态：cat $STATUSF ；摘要：cat $DIGEST ；打包：$ROOT/p4_r17_results.tar.gz"; return 0; }
  say "将在 $SHUTDOWN_GRACE_MIN 分钟后自动关机；要取消，执行 touch $NO_SHUTDOWN"
  sleep $(( SHUTDOWN_GRACE_MIN * 60 ))
  if [ -e $NO_SHUTDOWN ]; then say "检测到 $NO_SHUTDOWN：取消关机"; return 0; fi
  sync; /usr/bin/shutdown
}
package() {
  local items="DIGEST17.txt STATUS logs checks reagg const devsens _history" x
  for x in report_const.txt report_const.json report_devsens.txt report_devsens.json; do [ -f $OUT/$x ] && items="$items $x"; done
  tar -czf $ROOT/p4_r17_results.tar.gz -C $OUT --exclude='*perq*.pkl' $items 2>> $LOG \
    && say "打包：$ROOT/p4_r17_results.tar.gz（$(du -h $ROOT/p4_r17_results.tar.gz | cut -f1)，不含每查询记录 pkl 与 _stale；它们留在 $OUT）"
}
finish() {   # finish 退出码 状态：先等已在运行的单元结束（不再启动新单元），再写 STATUS、摘要、打包，按设置关机，以该退出码结束
  local rc=$1 nj; shift
  nj=$(jobs -rp | wc -l)
  if [ "$nj" -gt 0 ]; then say "等待已在运行的 $nj 个计算单元结束后再收尾（不再启动新单元）"; wait; fi
  setstatus "$* | exit $rc | run $TS"
  echo "STATUS: $* (exit $rc)" >> $SUM
  write_digest; package
  say "状态：$(cat $STATUSF)；摘要：$DIGEST"
  sync; power_off
  exit $rc
}
stop() { note "STOPPED: $*"; finish 1 "STOPPED: $*"; }
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
r15list() {   # 第十七轮读取或引用的全部第十五轮文件：test/、feats/、train_ir/ 下所有文件与 p4_r15_out 顶层文件（不含 .ipynb_checkpoints）
  { find $R15/test $R15/feats $R15/train_ir -name .ipynb_checkpoints -prune -o -type f -print0; find $R15/ -maxdepth 1 -type f -print0; } 2>/dev/null \
    | LC_ALL=C sort -z | xargs -0 -r sha256sum
}
r15guard() {   # r15guard 时点：与开跑前（首次运行时）记录的哈希清单逐行比较；不同即停
  r15list > $GUARDN
  if [ -s $GUARDN ] && cmp -s $GUARD0 $GUARDN; then
    echo "第十五轮文件核对（$1）：$(grep -c '' $GUARDN) 个文件的哈希与开跑前相同" >> $SUM; say "第十五轮文件核对（$1）：未改动"; return 0
  fi
  diff $GUARD0 $GUARDN > $OUT/checks/r15_guard_diff_$TS.txt 2>&1
  stop "第十五轮文件在「$1」时与开跑前的哈希清单不同（见 checks/r15_guard_diff_$TS.txt）。p4_r15_out 必须只读：先查明原因，后续步骤不进行"
}
mkmanifest() {   # mkmanifest 输出文件 类别 单元 "输入文件..." 命令参数...：决定该单元结果的全部内容（规范化 JSON）
  local outf=$1 kind=$2 unit=$3 ins=$4; shift 4
  $PY -B - "$ENVJSON" "$kind" "$unit" "$ins" "$@" > $outf 2>> $LOG <<'PYEOF'
import hashlib, json, os, sys
envf, kind, unit, ins = sys.argv[1:5]; cmd = sys.argv[5:]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
m = dict(kind=kind, unit=unit, script=cmd[0], script_sha256=sha(cmd[0]), args=cmd[1:],
         inputs=[dict(path=p, size=os.path.getsize(p), sha256=sha(p)) for p in ins.split()], env=json.load(open(envf)))
print(json.dumps(m, indent=1, sort_keys=True))
PYEOF
}
reusable() {   # reusable 目录 本次manifest：.done 存在、目录里的 manifest 与本次逐字节相同、.done 记的正是它的 sha256
  [ -f $1/.done ] && [ -f $1/manifest.json ] && cmp -s $1/manifest.json $2 && [ "$(cat $1/.done)" = "$(sha256sum < $2 | cut -d' ' -f1)" ]
}
freshdir() {   # freshdir 目录 本次manifest：旧目录（未完成、无 manifest 或 manifest 不同）整体移到 _stale 保留，再新建并写入 manifest
  local d=$1 why
  if [ -e $d ]; then
    if [ ! -f $d/.done ]; then why="上次未完成或失败"; elif [ ! -f $d/manifest.json ]; then why="只有 .done、没有 manifest（不能证明与本次兼容）"
    else why="manifest 与本次不同（代码、输入、参数或环境变了）"; fi
    mv $d $OUT/_stale/$(basename $d)_$TS || stop "无法把 $d 移到 _stale"
    note "  $(basename $d)：$why，旧目录移到 _stale/$(basename $d)_$TS 后重算"
  fi
  mkdir -p $d && cp $2 $d/manifest.json || stop "无法新建 $d"
}
runjob() {   # runjob 目录 失败标记 退出记录 标签 命令参数...（后台运行；成功才写 .done）
  local d=$1 ff=$2 xl=$3 lab=$4; shift 4
  $PY -B "$@" > $d/run.log 2>&1; local rc=$?
  if [ $rc = 0 ] && grep -q '^total time' $d/run.log; then sha256sum < $d/manifest.json | cut -d' ' -f1 > $d/.done.tmp && mv $d/.done.tmp $d/.done
  else echo "$lab exit=$rc" >> $ff; fi
  echo "$lab exit=$rc" >> $xl
}
slots() {   # slots 每进程所需内存GB：按可用内存与 CPU 定并行数（至少 1，至多 CPU-1）
  local mem ncpu par nc
  read -r mem ncpu <<< "$(memcpu)"; [ -n "$mem" ] || mem=0; [ -n "$ncpu" ] || ncpu=1
  par=$(( mem / $1 )); nc=$(( ncpu > 1 ? ncpu - 1 : 1 )); [ $par -gt $nc ] && par=$nc; [ $par -lt 1 ] && par=1; echo $par
}

setstatus "RUNNING | run $TS"
if [ "${SKIP_CONDA:-0}" != 1 ]; then   # 在 set -u、set -f 之前激活（conda 的激活脚本可能引用未定义变量、用到通配符）；失败也写 STATUS
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate rgbt5090 || stop "conda 环境 rgbt5090 激活失败（没有运行任何步骤）"
fi
set -uo pipefail
set -f   # 关闭通配符展开：方法名 B* 原样传给脚本
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GIT_TERMINAL_PROMPT=0
say "== 0 环境、代码与输入核对（运行 $TS）=="
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
note "本驱动 p4_r17.sh sha256 ${SELF_SHA:0:12}（v3：失败即停、按 manifest 续跑）"
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
ENVJSON=$OUT/logs/env_$TS.json
$PY -B - > $ENVJSON 2>> $LOG <<'PYEOF' || stop "无法读取 Python 与库版本（见 logs/driver.log）"
import json, os, sys
import numpy, scipy, sklearn, pandas, torch
print(json.dumps(dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                      pandas=pandas.__version__, torch=torch.__version__,
                      threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}), sort_keys=True))
PYEOF
note "环境（写入每个单元的 manifest）：$(cat $ENVJSON)"
if [ -f $GUARD0 ]; then
  r15guard "本次开跑时，与首次运行记录的清单比较"
else
  { r15list > $GUARD0.tmp && [ -s $GUARD0.tmp ] && mv $GUARD0.tmp $GUARD0; } || stop "无法记录 p4_r15_out 的哈希清单（为空或计算失败；路径不对？）"
  note "第十五轮文件哈希清单已记录：$(grep -c '' $GUARD0) 个文件（test/、feats/、train_ir/ 的全部文件与顶层文件）；之后每一步都与它比较"
fi
note "输入齐全：$(echo $SEEDS | wc -w) 个 base 的开发档案、测试档案、训练身份档案，以及第十五轮 $(( $(echo $SEEDS | wc -w) * $(echo $FIT_SEEDS | wc -w) )) 个拟合目录"
note "资源：可用内存约 ${MEMGB} GB，CPU ${NCPU} 核，数据盘剩余 ${FREEGB} GB"

say "== 1a 特征档案只读核查 =="
PAIRS=""
for s in $SEEDS; do PAIRS="$PAIRS s$s=$(devnpz $s),$R15/feats/test_base_s$s.npz"; done
$PY -B $T/r17_check_archives.py --pairs $PAIRS --assign_seeds $ASSIGN_SEEDS --json $OUT/checks/archives.json > $OUT/checks/archives.txt 2>&1; RC=$?
echo "check_archives exit=$RC" >> $SUM
[ $RC = 0 ] || stop "1a 特征档案核查未通过（exit=$RC：2 = 发现问题，其他 = 脚本出错；见 checks/archives.txt 的 [status] 行）。按计划先报告影响范围，不计算新对照"
note "1a 特征档案：每个种子的开发、测试两部分分别核对通过——用到的 5 种退化只改变各自的目标模态（缺失、噪声逐行都变，模糊至少一行变），非目标模态逐行不变；形状一致、数值有限"

say "== 1b 第十五轮已保存记录的显式对齐汇总 =="
$PY -B $T/r17_reagg_r15.py --r15_root $R15/test/constrained --qraw $PAIRS --fits $FIT_SEEDS --json $OUT/reagg/reagg_r15.json > $OUT/reagg/reagg_r15.txt 2>&1; RC=$?
echo "reagg_r15 exit=$RC" >> $SUM
[ $RC = 0 ] || stop "1b 第十五轮记录的显式对齐核对未通过（exit=$RC：3 = 结构或对齐问题，其他 = 脚本出错；见 reagg/reagg_r15.txt）：旧记录对不齐，不计算新对照"
note "1b 第十五轮已保存记录：结构与显式对齐核对通过（从保存的逐查询记录与 cells.csv 汇总，不打新分）"
r15guard "1b 之后"

say "== 2 常数迁移对照（$(echo $SEEDS | wc -w) 个 base × $(echo $FIT_SEEDS | wc -w) 次拟合；前端配置取第十五轮 selection.json，不重新择优）=="
PAR=$(slots $JOB_GB); FAILFLAG=$OUT/logs/.fail_const_$TS; rm -f $FAILFLAG
NJ=0; NSKIP=0; NTOT=$(( $(echo $SEEDS | wc -w) * $(echo $FIT_SEEDS | wc -w) ))
for s in $SEEDS; do for fs in $FIT_SEEDS; do
  u=s${s}_f$fs; d=$OUT/const/$u; m=$MFDIR/const_$u.json
  ARGS=($T/r17_const.py --b12 . --feats $(devdir $s) --assign none --out $d --base base_ref_s$s --adapter base_ref_s$s --tag c_s${s}_f$fs
        --assign_seeds $ASSIGN_SEEDS --mixes $MIXES --skip_joint --methods $METHODS17 --grid_preset $GRID --select_rule constrained
        --r_max 0.999999 --tie_break bstar --test_npz $R15/feats/test_base_s$s.npz --link_map train_lda --train_ir_npz $(trainir $s)
        --test_seed $fs --fixed_selection $(r15file $s $fs selection.json) --const_grid $CONST_GRID --const_curve --perq_ext --timing)
  mkmanifest $m const $u "$(devnpz $s) $R15/feats/test_base_s$s.npz $(trainir $s) $(r15file $s $fs selection.json)" "${ARGS[@]}" || stop "无法写 $u 的 manifest"
  if reusable $d $m; then NSKIP=$(( NSKIP + 1 )); echo "  $u：已完成且 manifest 与本次逐字节相同，复用" >> $SUM; continue; fi
  [ -e $FAILFLAG ] && break 2
  while [ "$(jobs -rp | wc -l)" -ge $PAR ]; do sleep 10; [ -e $FAILFLAG ] && break; done
  [ -e $FAILFLAG ] && break 2
  freshdir $d $m
  say "START const $u"; runjob $d $FAILFLAG $OUT/logs/const_exit.txt "const $u" "${ARGS[@]}" & NJ=$(( NJ + 1 ))
done; done
wait
NDONE=0
for s in $SEEDS; do for fs in $FIT_SEEDS; do d=$OUT/const/s${s}_f$fs; reusable $d $MFDIR/const_s${s}_f$fs.json && NDONE=$(( NDONE + 1 )); done; done
note "2 常数迁移对照：本次启动 $NJ 个进程（并行 $PAR），复用 $NSKIP 个；与本次 manifest 一致的已完成单元 $NDONE / $NTOT"
[ -e $FAILFLAG ] && stop "2 常数迁移对照有单元失败（$(paste -sd ';' $FAILFLAG)；见 const/<单元>/run.log）：未启动的单元不再启动，后续步骤不进行；修正原因后重新运行本脚本，manifest 一致的已完成单元会复用"
[ $NDONE = $NTOT ] || stop "2 常数迁移对照只有 $NDONE / $NTOT 个单元完成"
r15guard "2 之后"

say "== 3 结构与复现核对、报告 =="
$PY -B $T/r17_judge.py --r17_root $OUT/const --r15_root $R15/test/constrained --seeds $SEEDS --fits $FIT_SEEDS --expect_script_sha $SHA_R17 \
  --out $OUT/report_const.json > $OUT/report_const.txt 2>&1; RC=$?
echo "judge exit=$RC" >> $SUM
[ $RC = 0 ] || stop "3 结构或复现核对未通过（exit=$RC：3 = INVALID，其他 = 脚本出错；见 report_const.txt）：新对照不与冻结结果配对、不报告，开发集敏感性不运行"
note "3 报告：结构核对与第十五轮复现核对通过（新算的 B★、QE-DBA、QE+M、QE+CAET、QE+SRC2f、QE+EM 与冻结结果逐格一致，容差 1e-6；全部单元由 r17_const.py ${SHA_R17:0:12} 产生）"
r15guard "3 之后"
setstatus "RUNNING | steps 0-3 passed; development sensitivity next | run $TS"
write_digest

DEVFAIL=0
if [ "$RUN_DEVSENS" = 1 ]; then
  say "== 4 开发集敏感性（开发集半图库，B★ 前端上的迁移层；不回改最终方法，不进入测试集结论）=="
  PAR=$(slots $DEV_JOB_GB); FAILDEV=$OUT/logs/.fail_dev_$TS; rm -f $FAILDEV
  NJ=0; NSKIP=0; EXPECT=""
  for spec in $DEVSETS; do for s in $SEEDS; do EXPECT="$EXPECT ${spec%%:*}_s$s"; done; done
  for spec in $DEVSETS; do
    name=${spec%%:*}; extra=${spec#*:}
    for s in $SEEDS; do
      u=${name}_s$s; d=$OUT/devsens/$u; m=$MFDIR/dev_$u.json
      ARGS=($T/r17_const.py --b12 . --feats $(devdir $s) --assign none --out $d --base base_ref_s$s --adapter base_ref_s$s --tag d_${name}_s$s
            --folds 2 --split_seeds $DEV_SPLITS --assign_seeds $DEV_ASEEDS --mixes $DEV_MIXES --skip_joint --methods 'B*' A-2f-ctxn --grid_preset $GRID
            --select_rule constrained --r_max 0.999999 --tie_break bstar --link_map train_lda --train_ir_npz $(trainir $s) ${extra//=/ })
      mkmanifest $m devsens $u "$(devnpz $s) $(trainir $s)" "${ARGS[@]}" || { echo "devsens $u: manifest not written" >> $FAILDEV; break 2; }
      if reusable $d $m; then NSKIP=$(( NSKIP + 1 )); echo "  $u：已完成且 manifest 与本次逐字节相同，复用" >> $SUM; continue; fi
      [ -e $FAILDEV ] && break 2
      while [ "$(jobs -rp | wc -l)" -ge $PAR ]; do sleep 10; [ -e $FAILDEV ] && break; done
      [ -e $FAILDEV ] && break 2
      freshdir $d $m
      say "START devsens $u"; runjob $d $FAILDEV $OUT/logs/devsens_exit.txt "devsens $u" "${ARGS[@]}" & NJ=$(( NJ + 1 ))
    done
  done
  wait
  NDONE=0; NTOT=0
  for u in $EXPECT; do NTOT=$(( NTOT + 1 )); reusable $OUT/devsens/$u $MFDIR/dev_$u.json && NDONE=$(( NDONE + 1 )); done
  note "4 开发集敏感性：本次启动 $NJ 个进程（并行 $PAR），复用 $NSKIP 个；与本次 manifest 一致的已完成单元 $NDONE / $NTOT"
  if [ -e $FAILDEV ] || [ $NDONE != $NTOT ]; then
    DEVFAIL=1; note "DEV SENSITIVITY FAILED: 开发集敏感性未全部完成（$( [ -e $FAILDEV ] && paste -sd ';' $FAILDEV)；见 devsens/<单元>/run.log）；未启动的单元不再启动。步骤 0–3 的结果不受影响，但本次不算全部完成"
  else
    say "== 5 开发集敏感性报告（单独文件，不改动 report_const）=="
    $PY -B $T/r17_judge.py --devsens_only --devsens_root $OUT/devsens --devsens_expect $EXPECT --out $OUT/report_devsens.json > $OUT/report_devsens.txt 2>&1; RC=$?
    echo "judge devsens exit=$RC" >> $SUM
    if [ $RC = 0 ]; then note "5 开发集敏感性报告：$NTOT 个单元齐全（report_devsens.txt）"
    else DEVFAIL=1; note "DEV SENSITIVITY FAILED: 敏感性报告未通过（exit=$RC，见 report_devsens.txt）。步骤 0–3 的结果不受影响，但本次不算全部完成"; fi
  fi
  r15guard "4–5 之后"
fi

say "== 6 终检 =="
r15guard "结束时"
[ "$RUN_DEVSENS" = 1 ] || finish 5 "INCOMPLETE: DEV SENSITIVITY NOT RUN (RUN_DEVSENS=0); steps 0-3 passed"
[ $DEVFAIL = 0 ] || finish 4 "INCOMPLETE: DEV SENSITIVITY FAILED; steps 0-3 passed"
finish 0 "COMPLETE"
