# MRAF IR Sensitivity Rescue V4

## 用途与边界

该流水线只解决 V3 confirmation 已具备相关性和总体门控幅度、但
`IR_NOISE`/`IR_BLUR` 方向幅度仍未达到原 STRONG 门槛的问题。它先对 V3
confirmation checkpoint 做只读 24-batch 复诊，必要时只读搜索更高的
`mraf_gate_logit_scale`，通过原门槛后才允许重新训练。

固定安全边界：

- V3 source run-root 始终只读，不修改、覆盖、移动或删除任何文件；
- 所有新产物写入独立 V4 run-root；
- STRONG/VIABLE 分类直接复用 V3 的 `classify_candidate()`，不复制或降低阈值；
- scale 搜索只改变评估和诊断前向中的 `mraf_gate_logit_scale`；
- 12 epoch confirmation 和所有 60 epoch 正式训练均从原始 pretrained
  初始化开始，禁止从 V3、scale 搜索或短周期 checkpoint 续训；
- 不改变数据划分、退化定义、optimizer、基础学习率、损失或输入尺寸；
- dry-run 和 mock 不训练、不访问 GPU、不关机；
- 不使用 `screen`。

## V3 只读源

预期 source run-root：

```text
/root/autodl-tmp/MRAF_GATE_AMPLITUDE_RUNS_c667946b/20260727_092554_c667946b_mraf_gate_amplitude_rescue_seed1
```

严格要求：

```text
STATUS=CONFIRMATION_FAILED
FAILED_STAGE=confirmation
EXIT_CODE=5
git_head=c667946b2c51740bfa7892f5a2daee73d1279b44
selected_candidate=B50
selected_checkpoint=best
mraf_scorer_lr_multiplier=50
mraf_gate_logit_scale=2
mraf_rel_lambda=1
mraf_gate_base_grad_scale=1
```

`confirmation/units/best` 和 `confirmation/units/last` 必须分别包含非空的：

- `checkpoint.pth`
- `configs.yaml`
- `train.log`
- `mraf_epoch_metrics.tsv`
- `checkpoint_selection.tsv`

源 `configs.yaml` 还必须保持 V3 真实协议：
`num_epoch=12`、`warmup_epochs=5`、`mraf_rel_warmup_epoch=1`。这里的
`warmup_epochs=5` 是对既有 V3 checkpoint 的来源校验；V4 在 scale 大于 2 时
新启动的 12 epoch confirmation 按本任务新协议显式使用 `warmup_epochs=1`，
两者不能混淆。

流水线启动时记录这些核心文件的 SHA256；每个终态前再次核对。V4 只在自己的
run-root 内创建指向源文件的只读单元链接，评估、诊断和汇总输出仍全部位于 V4。

## 阶段

### 1. scale=2 完整复诊

对 V3 confirmation 的 best/last 分别执行：

- 8 个筛选场景；
- `degrade_seed=2026`；
- `mraf_gate_logit_scale=2`；
- 24-batch 完整机制诊断；
- 原 STRONG/VIABLE 和 S4 性能门禁。

输出：

- `full_rediagnosis_metrics.tsv`
- `full_rediagnosis_mechanism.tsv`
- `full_rediagnosis_ranking.tsv`

如果任一 checkpoint 为 `STRONG_PASS`，固定 `selected_scale=2`，不再搜索
scale，也不重新运行 12 epoch confirmation；V3 的 12 epoch checkpoint 仅用于
证明门禁通过，正式 60 epoch 仍从 pretrained 初始化。

### 2. 只读 scale 搜索

仅当 scale=2 的 24-batch 复诊没有 STRONG 时运行。固定组合：

```text
2.5, 3.0, 4.0, 5.0 × best, last
```

每个单元运行相同 8 场景和 4-batch 机制诊断，检查 finite/OOM、
`fusion_path_sensitive`、`scorer_parameter_updated`，并继续使用原 S4
clean/degraded 门禁。

选择顺序在运行前固定：

1. 只允许 `STRONG_PASS`；
2. 最小 scale；
3. 同 scale 下更高的 `degraded_mean_R1`；
4. 再比较 `clean_R1`；
5. 完全相同时优先 best。

不允许选择 VIABLE，不允许人工覆盖 JSON。无 STRONG 时终态为：

```text
STATUS=IR_SENSITIVITY_RESCUE_FAILED
FAILED_STAGE=readonly_scale_screen
```

输出：

- `readonly_scale_metrics.tsv`
- `readonly_scale_mechanism.tsv`
- `readonly_scale_ranking.tsv`
- `selected_ir_rescue_scale.json`

### 3. 新 12 epoch confirmation

仅在自动选择的 scale 大于 2 时运行。配置：

```text
seed=1
epochs=12
warmup_epochs=1
mraf_scorer_lr_multiplier=50
mraf_gate_logit_scale=<selected_scale>
mraf_rel_lambda=1
mraf_gate_base_grad_scale=1
mraf_gate_fp32=true
```

best/last 均执行 8 场景和 24-batch 机制诊断。至少一个 checkpoint 必须
`STRONG_PASS`；否则终态：

```text
STATUS=IR_SENSITIVITY_CONFIRMATION_FAILED
FAILED_STAGE=ir_rescue_confirmation
```

输出：

- `ir_rescue_confirmation_metrics.tsv`
- `ir_rescue_confirmation_mechanism.tsv`
- `ir_rescue_confirmation_summary.tsv`

### 4. 正式 seed1

从 pretrained 初始化重新训练：

```text
seed=1
epochs=60
warmup_epochs=3
mraf_scorer_lr_multiplier=50
mraf_gate_logit_scale=<selected_scale>
mraf_rel_lambda=1
mraf_gate_base_grad_scale=1
mraf_gate_fp32=true
```

best/last 都评估 11 场景并运行 24-batch 机制诊断。选择优先级为
STRONG、degraded mean R1、clean R1、best。seed1 没有 STRONG 时停止：

```text
STATUS=FORMAL_SEED1_FAILED
FAILED_STAGE=formal_seed1_gate
```

### 5. 正式 seed2–5

只有正式 seed1 通过 STRONG 后才依次运行 seed2、3、4、5。各 seed：

- 从同一 pretrained 初始化；
- 60 epoch，warmup 3；
- 使用 seed1 已固定的 scale 和全部超参数；
- 不再搜索、不调阈值；
- best/last 均保留并评估 11 场景；
- 24-batch 机制诊断；
- 完整记录 STRONG、VIABLE 或 FAIL。

工程异常停止并保留已完成 seed。resume 只在完整验证该 seed 的训练核心文件、
11 个 `scores.pt` 和诊断文件后才跳过。

## 场景

筛选/确认的 8 场景：

```text
clean
rgb_missing
ir_missing
rgb_noise_0.10
ir_noise_0.10
rgb_blur_k7
ir_blur_k7
both_blur_k7
```

正式实验的 11 场景：

```text
clean
rgb_missing
ir_missing
rgb_noise_0.05
rgb_noise_0.10
ir_noise_0.05
ir_noise_0.10
rgb_blur_k7
ir_blur_k7
both_noise_0.10
both_blur_k7
```

## TSV 与 legacy 兼容

所有 V4 TSV 都使用：

- 固定显式 schema；
- `csv.DictWriter`；
- `delimiter="\t"`；
- `lineterminator="\n"`；
- `extrasaction="raise"`；
- 唯一、非空字段名；
- 临时文件校验后 `os.replace` 原子替换。

只读 legacy 解析只允许四个白名单合并表头：

- `target_rgb_meantarget_rgb_std`
- `scorer_first_weight_normscorer_last_weight_norm`
- `mraf_rel_lambdamraf_gate_base_grad_scale`
- `scorer_parameter_deltaraw_logit_gap_std`

未知 schema 或行宽不一致会失败，不猜测、不回写 V3。

## 状态、resume 与关机

V4 run-root 包含：

- `status.txt`
- `experiment_manifest.txt`
- `stage_status.tsv`
- `seed_status.tsv`
- `resume_history.tsv`

resume 接受工程中断的 `RUNNING`、`FAILED` 或 `MULTISEED_FAILED`，拒绝
`COMPLETED` 以及三个方法门禁失败状态。完成记录本身不足以跳过阶段；核心文件、
场景分数和诊断必须全部完整。训练中断时创建新的 attempt 子目录，不覆盖旧
checkpoint。

`--shutdown-on-terminal` 对真实运行的下列终态生效：

- `COMPLETED`
- `IR_SENSITIVITY_RESCUE_FAILED`
- `IR_SENSITIVITY_CONFIRMATION_FAILED`
- `FORMAL_SEED1_FAILED`
- `MULTISEED_FAILED`
- `FAILED`

流水线先写终态、stage 快照并核对 source SHA256，再调用 shutdown。shutdown
失败不会抹掉已经写入的终态。mock、dry-run 和单元测试永不关机；外部 supervisor
可作为第二层保护。

## 多种子与论文输出

五个 seed 工程完成后生成：

- `multiseed_retrieval_summary.tsv`
- `multiseed_robustness_summary.tsv`
- `multiseed_mechanism_summary.tsv`
- `multiseed_classification_summary.tsv`
- `comparison_with_s4_and_old_seed1.tsv`

统计包含每个正式场景及机制指标的 mean、sample std、min、max、n，以及
STRONG/VIABLE/FAIL seed 数。
单 seed 与五 seed 标签严格区分。

论文表：

- `paper_tables/main_retrieval_results.tsv`
- `paper_tables/robustness_results.tsv`
- `paper_tables/mechanism_results.tsv`
- `paper_tables/ablation_scale_results.tsv`
- `paper_tables/seed_level_results.tsv`

同时生成：

- `PAPER_READY_RESULTS_ZH.md`
- `PAPER_READY_RESULTS_EN.md`
- `FINAL_REPORT.txt`
- `mraf_ir_sensitivity_rescue_reports.tar.gz`

压缩包只包含报告、TSV、JSON、Markdown、日志索引和配置，不包含
checkpoint 或 `scores.pt`。

## AutoDL 部署与只读检查

```bash
cd /root/autodl-tmp/IRRA-main-mraf
git fetch origin
git switch experiment/mraf-ir-sensitivity-rescue-v4
git pull --ff-only origin experiment/mraf-ir-sensitivity-rescue-v4
git status --short
git rev-parse HEAD

conda activate rgbt5090
cd IRRA-main

python -B -m pytest -q tests/test_mraf_ir_sensitivity_rescue.py
bash -n scripts/mraf/run_mraf_ir_sensitivity_rescue_pipeline.sh

SOURCE_RUN=/root/autodl-tmp/MRAF_GATE_AMPLITUDE_RUNS_c667946b/20260727_092554_c667946b_mraf_gate_amplitude_rescue_seed1
OUTPUT_ROOT=/root/autodl-tmp/MRAF_IR_SENSITIVITY_RESCUE_RUNS_$(git rev-parse --short=8 HEAD)
DATA_ROOT=/root/autodl-tmp/TextReID
BASE_CONFIG="$SOURCE_RUN/confirmation/units/best/configs.yaml"
REFERENCE_SCREEN_ROOT=/root/autodl-tmp/<V3_REFERENCE_SCREEN_ROOT>
REFERENCE_RUN_ROOT=/root/autodl-tmp/<OLD_SEED1_REFERENCE_ROOT>

mkdir -p "$OUTPUT_ROOT"

bash scripts/mraf/run_mraf_ir_sensitivity_rescue_pipeline.sh \
  --source-run-root "$SOURCE_RUN" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --base-config "$BASE_CONFIG" \
  --reference-screen-root "$REFERENCE_SCREEN_ROOT" \
  --reference-run-root "$REFERENCE_RUN_ROOT" \
  --device 0 \
  --seed 1 \
  --auto-continue \
  --shutdown-on-terminal \
  --dry-run
```

dry-run 必须显示：

```text
PLAN_VALID=true
SOURCE_CONFIRMATION_STATUS=CONFIRMATION_FAILED
SOURCE_BEST_FOUND=true
SOURCE_LAST_FOUND=true
FULL_REDIAGNOSIS_BATCHES=24
READONLY_SCALES=2.5,3,4,5
NEXT_STAGE=full_rediagnosis
FORMAL_SEEDS=1,2,3,4,5
DRY_RUN_READ_ONLY=true
```

## AutoDL 一键正式运行

确认上面的占位路径已替换、dry-run 通过且磁盘/GPU 空闲后：

```bash
nohup bash scripts/mraf/run_mraf_ir_sensitivity_rescue_pipeline.sh \
  --source-run-root "$SOURCE_RUN" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --base-config "$BASE_CONFIG" \
  --reference-screen-root "$REFERENCE_SCREEN_ROOT" \
  --reference-run-root "$REFERENCE_RUN_ROOT" \
  --device 0 \
  --seed 1 \
  --auto-continue \
  --shutdown-on-terminal \
  > /root/autodl-tmp/mraf_ir_sensitivity_rescue_v4.log 2>&1 &

tail -f /root/autodl-tmp/mraf_ir_sensitivity_rescue_v4.log
```

不要同时启动第二个 V4 Pipeline。

## 安全 resume

机器未关机或工程中断后：

```bash
nohup bash scripts/mraf/run_mraf_ir_sensitivity_rescue_pipeline.sh \
  --resume-run-root <V4_RUN_ROOT> \
  --device 0 \
  --seed 1 \
  --auto-continue \
  --shutdown-on-terminal \
  > /root/autodl-tmp/mraf_ir_sensitivity_rescue_v4_resume.log 2>&1 &

tail -f /root/autodl-tmp/mraf_ir_sensitivity_rescue_v4_resume.log
```

resume 不会重跑已经通过完整产物验证的 seed，也不会覆盖任何 checkpoint。
