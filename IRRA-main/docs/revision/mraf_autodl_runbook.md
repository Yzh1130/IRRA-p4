# MRAF AutoDL 运行手册

本文仅适用于分支 `feature/mraf-reliability-fusion`。命令中的
`<REMOTE_URL>`、提交 SHA 和绝对目录需要替换为实际值；所有 CLI 参数名均
来自当前代码的真实接口。正式训练协议已由 AutoDL 历史正式训练脚本及对应
`configs.yaml` 确认。

严禁使用 `screen`。两组 60 epoch 正式训练必须串行执行，不要同时占用同一
GPU。MDRC 的 scale 虽已确认为 100，仍必须通过 `MDRC_LOSS_SCALE=100`
显式传入，脚本不会设置隐藏默认值。

## 1. 本地提交、推送和远端提交确认

以下操作应在本地审核通过后由人工执行，本次代码审计不会代为 commit 或
push：

```bash
cd /path/to/IRRA-main-mraf
git status --short
git add .gitattributes \
  IRRA-main/scripts/mraf/run_mraf_experiments.sh \
  IRRA-main/docs/revision/mraf_autodl_runbook.md \
  IRRA-main/docs/revision/mraf_training_protocol_audit.md \
  IRRA-main/docs/revision/mraf_test_report.md \
  IRRA-main/docs/revision/mraf_changed_files.md
git update-index --chmod=+x IRRA-main/scripts/mraf/run_mraf_experiments.sh
git diff --cached --check
git commit -m "feat: add reliability-adaptive RGB-T fusion for MDRC"
git push -u origin feature/mraf-reliability-fusion

export MRAF_BRANCH=feature/mraf-reliability-fusion
export MRAF_COMMIT="$(git rev-parse HEAD)"
git ls-remote --exit-code origin "refs/heads/${MRAF_BRANCH}"
printf 'branch=%s\ncommit=%s\n' "$MRAF_BRANCH" "$MRAF_COMMIT"
```

确认 `git ls-remote` 显示的 SHA 与 `$MRAF_COMMIT` 完全一致后，再登录
AutoDL。

## 2. AutoDL 磁盘空间检查

```bash
df -h
du -sh /root/autodl-tmp 2>/dev/null || true
nvidia-smi
```

训练前需要为两个独立 60 epoch 输出、checkpoint、TensorBoard 事件、
11 场景评估分数及日志预留空间。

## 3. 获取分支并创建独立 worktree

若 AutoDL 尚未有仓库：

```bash
cd /root/autodl-tmp
git clone <REMOTE_URL> IRRA
```

随后在已有 Git 根目录中获取经过确认的分支和提交：

```bash
export SOURCE_REPO=/root/autodl-tmp/IRRA
export MRAF_WORKTREE=/root/autodl-tmp/IRRA-mraf-worktree
export MRAF_BRANCH=feature/mraf-reliability-fusion
export MRAF_COMMIT=<本地推送后确认的完整SHA>

cd "$SOURCE_REPO"
git fetch origin "$MRAF_BRANCH"
test "$(git rev-parse "origin/${MRAF_BRANCH}")" = "$MRAF_COMMIT"
test ! -e "$MRAF_WORKTREE"
git worktree add --detach "$MRAF_WORKTREE" "$MRAF_COMMIT"
git -C "$MRAF_WORKTREE" rev-parse HEAD
git -C "$MRAF_WORKTREE" status --short --branch
```

进入实际代码目录：

```bash
cd "$MRAF_WORKTREE/IRRA-main"
```

## 4. 激活既有 Python 环境

不要新建或安装依赖；激活该项目已经验证过的现有环境：

```bash
export IRRA_CONDA_ENV=rgbt5090
eval "$(conda shell.bash hook)"
conda activate "$IRRA_CONDA_ENV"
which python
```

打印 Python、PyTorch、CUDA 和 GPU 信息：

```bash
python -V
python - <<'PY'
import torch

print("torch_version=", torch.__version__)
print("torch_cuda_version=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0=", torch.cuda.get_device_name(0))
PY
nvidia-smi
```

## 5. 提交前测试和静态检查

```bash
cd "$MRAF_WORKTREE/IRRA-main"
export PYTHONDONTWRITEBYTECODE=1

python -B -m pytest -q tests/test_mraf.py
python -B -m pytest -q

python -B tools/revision/analyze_mraf_weights.py --help
python -B tools/revision/summarize_mraf_experiment.py --help

bash -n scripts/mraf/run_mraf_experiments.sh

cd "$MRAF_WORKTREE"
git diff --check
test -z "$(git status --porcelain)"
cd "$MRAF_WORKTREE/IRRA-main"
```

任何一步失败都应停止，不要启动训练。

## 6. 核实 Base/MDRC 正式训练协议

正式协议已经从下列 AutoDL 历史资产交叉核实：

1. 正式实验目录中的 `configs.yaml`；
2. 训练日志开头打印的完整 args；
3. 当时实际使用的启动脚本或命令；
4. checkpoint 同目录或旁边保存的配置。

确认结果如下：

- `batch_size=64`；
- `loss_names=sdm+mlm+id`；
- `MLM=true`；
- `warmup_epochs=5`；
- `eval_use_fused_mu=true`；
- `mdrc_loss_scale=100`；
- `mdrc_lambda=0.05`；
- `mdrc_prob=0.5`；
- `mdrc_warmup_epoch=3`；
- conda 环境为 `rgbt5090`。

```bash
export HISTORICAL_BASE_DIR=<历史正式Base实验目录>
export HISTORICAL_MDRC_DIR=<历史正式MDRC实验目录>

find "$HISTORICAL_BASE_DIR" "$HISTORICAL_MDRC_DIR" \
  -type f \( -name configs.yaml -o -name '*.log' -o -name '*.sh' \) -print
rg -n --hidden -S \
  "batch_size|loss_names|MLM|mdrc_loss_scale|--mdrc_loss_scale" \
  "$HISTORICAL_BASE_DIR" "$HISTORICAL_MDRC_DIR"
```

为避免命令中出现未经核实的临时值，统一导出已确认值：

```bash
export VERIFIED_BATCH_SIZE=64
export VERIFIED_LOSS_NAMES='sdm+mlm+id'
export VERIFIED_MLM_ENABLED=true
export VERIFIED_WARMUP_EPOCHS=5
export VERIFIED_EVAL_USE_FUSED_MU=true
export VERIFIED_MDRC_LOSS_SCALE=100

test -n "$VERIFIED_BATCH_SIZE"
test -n "$VERIFIED_LOSS_NAMES"
test "$VERIFIED_MLM_ENABLED" = true
printf 'batch_size=%s\nloss_names=%s\nMLM=%s\nwarmup_epochs=%s\neval_use_fused_mu=%s\nmdrc_loss_scale=%s\n' \
  "$VERIFIED_BATCH_SIZE" "$VERIFIED_LOSS_NAMES" "$VERIFIED_MLM_ENABLED" \
  "$VERIFIED_WARMUP_EPOCHS" "$VERIFIED_EVAL_USE_FUSED_MU" \
  "$VERIFIED_MDRC_LOSS_SCALE"
```

当前包装脚本固定传入真实项目参数 `--MLM`、`--warmup_epochs 5` 和
`--eval_use_fused_mu`。非 MUM 聚合可能同时计入 `mlm_loss` 与
`loss_mlm` 别名；本次为了保持既有代码路径可比而不修改该行为。未来若
修复，Base、MDRC 和新方法必须全部重新训练。

## 7. MRAF-only dry-run

`--dry-run` 不创建输出目录、不启动 Python，也不要求
`MDRC_LOSS_SCALE`：

```bash
export DRY_RUN_ROOT="$PWD/mraf_dry_run_$(date +%Y%m%d_%H%M%S)"
test ! -e "$DRY_RUN_ROOT"
bash scripts/mraf/run_mraf_experiments.sh \
  --case mraf_only \
  --seed 1 \
  --num-epoch 1 \
  --batch-size "$VERIFIED_BATCH_SIZE" \
  --loss-names "$VERIFIED_LOSS_NAMES" \
  --output-root "$DRY_RUN_ROOT" \
  --dry-run
test ! -e "$DRY_RUN_ROOT"
```

输出中必须包含已核实的 `batch_size`、`effective_batch_size`、
`loss_names`、ID/MLM 状态、`training_protocol_verified=true`、
`warmup_epochs=5`、`eval_use_fused_mu=true`、
`mraf_enabled=true`、`mdrc_enabled=false`、`mraf_rel_lambda=0.01`、
`mdrc_loss_scale=NA`、`degrade_aug_mode=none` 和完整
`training_command`。若 MLM 启用，还必须看到 alias 双计数风险警告。

## 8. 核实并设置正式 MDRC loss scale

AutoDL 历史正式训练脚本及对应 `configs.yaml` 已确认
`mdrc_loss_scale=100`。保留以下证据检查命令用于服务器复核：

```bash
rg -n --hidden -S \
  "mdrc_loss_scale|--mdrc_loss_scale|paper_e3_mdrc_seed1|mdrc_seed1" \
  "$HISTORICAL_MDRC_DIR"
```

显式设置已确认值；不得依赖 `utils/options.py` 的代码默认值：

```bash
export VERIFIED_MDRC_LOSS_SCALE=100
export MDRC_LOSS_SCALE="$VERIFIED_MDRC_LOSS_SCALE"
test -n "$MDRC_LOSS_SCALE"
printf 'verified MDRC_LOSS_SCALE=%s\n' "$MDRC_LOSS_SCALE"
```

脚本不为 `MDRC_LOSS_SCALE` 设置隐藏默认值；MDRC 相关 case 在变量缺失时
仍会主动退出。

## 9. MDRC+MRAF dry-run

```bash
: "${MDRC_LOSS_SCALE:?先根据历史正式MDRC证据设置MDRC_LOSS_SCALE}"
export MDRC_DRY_RUN_ROOT="$PWD/mdrc_mraf_dry_run_$(date +%Y%m%d_%H%M%S)"
test ! -e "$MDRC_DRY_RUN_ROOT"
bash scripts/mraf/run_mraf_experiments.sh \
  --case mdrc_mraf \
  --seed 1 \
  --num-epoch 1 \
  --batch-size "$VERIFIED_BATCH_SIZE" \
  --loss-names "$VERIFIED_LOSS_NAMES" \
  --output-root "$MDRC_DRY_RUN_ROOT" \
  --dry-run
test ! -e "$MDRC_DRY_RUN_ROOT"
```

输出必须显示 `mdrc_enabled=true`，且 `mdrc_loss_scale` 等于人工核实值。

## 10. 两组 seed1 单 epoch GPU 冒烟测试

AutoDL 冒烟也必须使用核实后的 batch size 和 loss names，不能依赖脚本的
本地未核实默认值。先运行 MRAF-only：

```bash
export CUDA_VISIBLE_DEVICES=0
export SMOKE_ROOT="$PWD/mraf_smoke_runs"
bash scripts/mraf/run_mraf_experiments.sh \
  --case mraf_only \
  --seed 1 \
  --num-epoch 1 \
  --batch-size "$VERIFIED_BATCH_SIZE" \
  --loss-names "$VERIFIED_LOSS_NAMES" \
  --output-root "$SMOKE_ROOT"
```

完成后再运行 MDRC+MRAF，不要并行：

```bash
: "${MDRC_LOSS_SCALE:?请显式设置已确认值MDRC_LOSS_SCALE=100}"
MDRC_LOSS_SCALE="$MDRC_LOSS_SCALE" \
  bash scripts/mraf/run_mraf_experiments.sh \
    --case mdrc_mraf \
    --seed 1 \
    --num-epoch 1 \
    --batch-size "$VERIFIED_BATCH_SIZE" \
    --loss-names "$VERIFIED_LOSS_NAMES" \
    --output-root "$SMOKE_ROOT"
```

检查显存、loss、门控权重、checkpoint 保存和重新加载：

```bash
nvidia-smi
export MRAF_SMOKE_DRIVER_DIR="$(
  ls -dt "$SMOKE_ROOT"/*_mraf_only_seed1 | head -n 1
)"
rg -n \
  "loss_mraf_rel|lambda_mraf_rel_cur|alpha_rgb|\[MRAF_EPOCH\]|NaN|Inf" \
  "$MRAF_SMOKE_DRIVER_DIR/train.log"
find "$MRAF_SMOKE_DRIVER_DIR" -type f \
  \( -name 'configs.yaml' -o -name '*.pth' -o -name 'events.out.tfevents.*' \) \
  -print

export MRAF_SMOKE_CONFIG="$(
  find "$MRAF_SMOKE_DRIVER_DIR" -type f -name configs.yaml -print -quit
)"
export MRAF_SMOKE_CKPT="$(
  find "$MRAF_SMOKE_DRIVER_DIR" -type f -name best.pth -print -quit
)"
test -n "$MRAF_SMOKE_CONFIG"
test -n "$MRAF_SMOKE_CKPT"
python -B test.py \
  --config_file "$MRAF_SMOKE_CONFIG" \
  --test_ckpt_file "$MRAF_SMOKE_CKPT" \
  --eval_degrade_mode none
```

还要检查 MDRC+MRAF 的 `train.log` 中没有 NaN/Inf，实际
`effective_mdrc_loss_scale` 与人工核实值一致，且
`alpha_rgb + alpha_ir` 没有异常漂移或单模态塌缩。

## 11. 两组 seed1 60 epoch 正式训练

确保单 epoch 冒烟测试和 checkpoint 重载均通过。先单独运行
MRAF-only：

```bash
export CUDA_VISIBLE_DEVICES=0
export FORMAL_ROOT="$PWD/mraf_formal_runs"
bash scripts/mraf/run_mraf_experiments.sh \
  --case mraf_only \
  --seed 1 \
  --num-epoch 60 \
  --batch-size "$VERIFIED_BATCH_SIZE" \
  --loss-names "$VERIFIED_LOSS_NAMES" \
  --output-root "$FORMAL_ROOT"
```

该进程完全结束并检查结果后，才允许启动 MDRC+MRAF：

```bash
: "${MDRC_LOSS_SCALE:?未显式设置MDRC_LOSS_SCALE=100，禁止启动MDRC+MRAF}"
MDRC_LOSS_SCALE="$MDRC_LOSS_SCALE" \
  bash scripts/mraf/run_mraf_experiments.sh \
    --case mdrc_mraf \
    --seed 1 \
    --num-epoch 60 \
    --batch-size "$VERIFIED_BATCH_SIZE" \
    --loss-names "$VERIFIED_LOSS_NAMES" \
    --output-root "$FORMAL_ROOT"
```

不要同时并行运行这两组 60 epoch 正式训练。

## 12. 外部 nohup 后台运行

脚本自身不调用 `nohup`。后台运行 MRAF-only：

```bash
mkdir -p "$PWD/mraf_driver_logs"
nohup env CUDA_VISIBLE_DEVICES=0 \
  bash scripts/mraf/run_mraf_experiments.sh \
    --case mraf_only \
    --seed 1 \
    --num-epoch 60 \
    --batch-size "$VERIFIED_BATCH_SIZE" \
    --loss-names "$VERIFIED_LOSS_NAMES" \
    --output-root "$FORMAL_ROOT" \
  >"$PWD/mraf_driver_logs/mraf_only_seed1.driver.log" 2>&1 &
echo $! | tee "$PWD/mraf_driver_logs/mraf_only_seed1.pid"
```

只有上述 PID 已结束并确认结果后，才可后台运行 MDRC+MRAF：

```bash
: "${MDRC_LOSS_SCALE:?未显式设置MDRC_LOSS_SCALE=100，禁止启动MDRC+MRAF}"
nohup env CUDA_VISIBLE_DEVICES=0 MDRC_LOSS_SCALE="$MDRC_LOSS_SCALE" \
  bash scripts/mraf/run_mraf_experiments.sh \
    --case mdrc_mraf \
    --seed 1 \
    --num-epoch 60 \
    --batch-size "$VERIFIED_BATCH_SIZE" \
    --loss-names "$VERIFIED_LOSS_NAMES" \
    --output-root "$FORMAL_ROOT" \
  >"$PWD/mraf_driver_logs/mdrc_mraf_seed1.driver.log" 2>&1 &
echo $! | tee "$PWD/mraf_driver_logs/mdrc_mraf_seed1.pid"
```

严禁改用 `screen`。

## 13. 查看日志、进程和正常停止错误实验

```bash
tail -f "$PWD/mraf_driver_logs/mraf_only_seed1.driver.log"
ps -fp "$(cat "$PWD/mraf_driver_logs/mraf_only_seed1.pid")"
pgrep -af "train.py|run_mraf_experiments.sh"
nvidia-smi
```

若确认启动参数错误，先发送普通终止信号，不要直接 `kill -9`：

```bash
kill "$(cat "$PWD/mraf_driver_logs/mraf_only_seed1.pid")"
ps -fp "$(cat "$PWD/mraf_driver_logs/mraf_only_seed1.pid")" || true
```

停止前后保留 `driver.log`、`train.log`、`command.txt` 和
`launch_metadata.txt` 供审计。

## 14. 标准 10 退化评估

以下循环运行 clean 加 10 个标准退化场景。`CONFIG_FILE` 和
`CHECKPOINT` 必须来自同一完成训练：

```bash
export CONFIG_FILE=<完成训练目录中的configs.yaml>
export CHECKPOINT=<该训练的最佳checkpoint.pth>
export ROBUST_ROOT="$PWD/mraf_eval_$(date +%Y%m%d_%H%M%S)"
test -f "$CONFIG_FILE"
test -f "$CHECKPOINT"
test ! -e "$ROBUST_ROOT"
mkdir -p "$ROBUST_ROOT"

cases=(
  "clean|none|1.0|7"
  "rgb_missing|rgb_missing|1.0|7"
  "ir_missing|ir_missing|1.0|7"
  "rgb_noise_005|rgb_noise|0.05|7"
  "rgb_noise_010|rgb_noise|0.10|7"
  "ir_noise_005|ir_noise|0.05|7"
  "ir_noise_010|ir_noise|0.10|7"
  "rgb_blur_k7|rgb_blur|1.0|7"
  "ir_blur_k7|ir_blur|1.0|7"
  "both_noise_005|both_noise|0.05|7"
  "both_blur_k7|both_blur|1.0|7"
)

for spec in "${cases[@]}"; do
  IFS='|' read -r name mode strength kernel <<<"$spec"
  case_dir="$ROBUST_ROOT/$name"
  test ! -e "$case_dir"
  mkdir -p "$case_dir"
  python -B -u test.py \
    --config_file "$CONFIG_FILE" \
    --test_ckpt_file "$CHECKPOINT" \
    --eval_degrade_mode "$mode" \
    --eval_degrade_strength "$strength" \
    --eval_degrade_blur_kernel "$kernel" \
    --eval_degrade_seed 2026 \
    --save_eval_scores \
    --eval_scores_save_raw \
    --eval_scores_file "$case_dir/score.pt" \
    >"$case_dir/eval.log" 2>&1
done
```

逐个确认 11 个 `score.pt` 与 `eval.log` 均存在，失败场景不得用其他场景
结果代填。

## 15. MRAF 门控权重统计

```bash
export ALPHA_ROOT="$ROBUST_ROOT/alpha_statistics"
python -B tools/revision/analyze_mraf_weights.py \
  --config-file "$CONFIG_FILE" \
  --checkpoint "$CHECKPOINT" \
  --output-prefix "$ALPHA_ROOT/mraf_weights" \
  --device cuda \
  --degrade-seed 2026
```

输出应为 `mraf_weights.csv` 和 `mraf_weights.json`。若配置未启用 MRAF
或 checkpoint 不含 MRAF 参数，工具会明确报错。

## 16. 实验结果汇总

根据真实 clean/退化评估、训练日志和 checkpoint 信息创建 manifest。
未测值必须写 `null`，不能写 0 或估计值：

```bash
export SUMMARY_MANIFEST="$ROBUST_ROOT/completed_runs.json"
cat >"$SUMMARY_MANIFEST" <<EOF
{
  "runs": [
    {
      "method": "mraf_only",
      "train_seed": 1,
      "checkpoint": "$CHECKPOINT",
      "alpha_statistics_file": "$ALPHA_ROOT/mraf_weights.json",
      "git_head": "$MRAF_COMMIT",
      "best_epoch": null,
      "clean_r1": null,
      "clean_r5": null,
      "clean_r10": null,
      "clean_map": null,
      "clean_minp": null,
      "mean_abs_delta_r1": null,
      "mean_degraded_r1": null,
      "rgb_missing_r1": null,
      "ir_missing_r1": null,
      "rgb_noise_010_r1": null,
      "ir_noise_010_r1": null,
      "both_blur_k7_r1": null,
      "parameter_count": null,
      "inference_time": null,
      "test_elapsed_time": null,
      "config": {"config_file": "$CONFIG_FILE"}
    }
  ]
}
EOF
```

用真实结果替换对应 `null` 后运行：

```bash
python -B tools/revision/summarize_mraf_experiment.py \
  --manifest "$SUMMARY_MANIFEST" \
  --output-dir "$ROBUST_ROOT/summary" \
  --basename mraf_seed1_summary
```

工具会生成 JSON、CSV 和 Markdown；仍缺失的字段保持 `null/NA`。

## 17. 每组必须归档和下载的文件

每个 `mraf_only_seed1` 或 `mdrc_mraf_seed1` 实验至少下载：

- `command.txt`、`launch_metadata.txt`、`train.log` 和外部 driver log；
- 实际训练目录中的 `configs.yaml`；
- `best.pth`、需要保留的 metric checkpoint 和完整评估状态 checkpoint；
- TensorBoard `events.out.tfevents.*`；
- clean 加 10 退化场景的全部 `score.pt` 与 `eval.log`；
- `mraf_weights.csv`、`mraf_weights.json`；
- `completed_runs.json`；
- `mraf_seed1_summary.json`、`.csv` 和 `.md`；
- 分支名和实际 Git HEAD；
- 经历史证据确认的完整训练协议及证据文件，包括 batch/loss、MLM、
  warmup、fused-mu、MDRC 参数和 conda 环境。

下载完成前不要删除、覆盖或复用实验目录。`MDRC_LOSS_SCALE` 未显式设置为
已确认值 100 时，不能启动 MDRC+MRAF。
