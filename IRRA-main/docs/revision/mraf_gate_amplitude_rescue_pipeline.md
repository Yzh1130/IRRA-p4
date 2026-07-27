# MRAF 门控幅度修复流水线

## 背景与边界

上一轮 S0–S5 均正常完成，但最终为 `SCREEN_FAILED`。S3、S4、S5 的
alpha-target Pearson 分别可达到约 0.232、0.489、0.771，alpha 标准差却仍只有
约 `5.9e-5` 到 `9.5e-5`，噪声/模糊退化引起的权重变化也仅约 `1e-5`。
这说明 scorer 可以学到排序方向，但没有形成有实际意义的 logit 幅度；高
Pearson 本身不能证明门控有效。

本流水线只增加两个默认保持旧行为的控制量：

- `mraf_scorer_lr_multiplier`：只提高 `IRRA.mraf_fusion.scorer` 的学习率；
  默认 `1.0`。
- `mraf_gate_logit_scale`：在 softmax 前执行
  `scaled_logits = raw_logits * scale`；默认 `1.0`。

默认值下 optimizer、scheduler、前向、checkpoint 维度、target、融合和评估
语义不变。流水线不修改历史 checkpoint/config，也不从筛选 checkpoint
续训。

## 流程

Stage 1 从相同预训练初始化分别训练 10 epoch，均使用 FP32 gate、
`mraf_rel_warmup_epoch=1`、seed 1：

| 候选 | rel lambda | base gate grad scale | scorer LR multiplier | logit scale |
| --- | ---: | ---: | ---: | ---: |
| A10 | 0.25 | 0.10 | 10 | 1 |
| A50 | 0.25 | 0.10 | 50 | 1 |
| A100 | 0.25 | 0.10 | 100 | 1 |
| B50 | 1.00 | 1.00 | 50 | 1 |

每个候选仅保存 `best.pth` 和 `last.pth`，两个 checkpoint 均独立完成 8 场景
评估和 `max-batches=4`、`gradient-batches=3`、
`gradient-microbatch-size=4` 的诊断。旧 S3/S4 只作为只读 reference，
绝不重训或修改。

训练命令先用 `train.py --config_file <historical-configs.yaml>` 把历史正式
MDRC+MRAF 配置作为 parser defaults，再由 CLI 显式覆盖输出目录、数据根、
epoch、候选 MRAF 参数以及已确认的正式协议。这样其余受支持参数来自历史
配置，同时 CLI 候选矩阵优先；未传 `--config_file` 时 `train.py` 的旧默认
行为完全不变。历史配置中当前 parser 已不再支持的字段不会注入运行参数。

筛选分两级：

- `STRONG_PASS`：alpha std ≥ 0.005、Pearson ≥ 0.30、缺失模态自身权重
  ≤ 0.35、四类 noise/blur 自身权重下降均 ≥ 0.02、
  weighted-rel/base 位于 `[1e-3, 5e-2]`，clean 相对旧 S4 降幅不超过
  0.50，退化均值降幅不超过 0.20，并要求 scorer 已更新、融合路径敏感且
  结果完整、无 NaN/Inf/OOM。
- `VIABLE_PASS`：只用于决定是否进入校准，不代表方法成功。要求 alpha std
  ≥ 0.001、Pearson ≥ 0.30、两个缺失模态自身权重 ≤ 0.45，至少三类
  noise/blur 下降 ≥ 0.002，并满足同样的性能、参数更新和融合路径门禁。

若 Stage 1 已有 `STRONG_PASS`，直接进入确认训练；若只有 `VIABLE_PASS`，
只对排名最高的候选 checkpoint 依次用 scale `2,5,10,20,50` 做只读评估和
诊断。校准不会训练，不写回 `configs.yaml`，也不写回 checkpoint。只有
校准后达到 `STRONG_PASS` 才能继续。

随后从同一预训练初始化重新训练 12 epoch 确认实验，不从 Stage 1 续训。
确认实验的 best/last 仍须达到 `STRONG_PASS`。只有确认通过，才从相同初始
化重新启动 60 epoch，恢复 `mraf_rel_warmup_epoch=3`。正式 best/last 均完成
11 场景评估及 24-batch 机制诊断，最终依据机制门禁和 clean/degraded 综合
排名选择 checkpoint。

## 训练期记录与 checkpoint

`mraf_epoch_metrics.tsv` 每 epoch 记录 base/scorer LR、倍率、raw/scaled
logit 差统计、alpha/target、Pearson、损失比、首末 scorer 层权重/梯度范数、
相对初始化参数变化和 dtype。`optimizer_param_groups.tsv` 记录各参数组名称、
参数数量、初始/当前 LR、weight decay、倍率和参数名。

`--save_last_checkpoint` 默认关闭。流水线显式开启后，训练结束额外保存
`last.pth`；现有 clean R1 选出的 `best.pth` 语义保持不变。
`checkpoint_selection.tsv` 记录 best/last 对应 epoch、clean R1 和门控指标。
每个候选最多保留这两个完整 checkpoint。

## AutoDL 一键命令

正式启动前至少为 Stage 1 预留 18 GB；60 epoch 前流水线会再次要求至少
8 GB。空间不足只报错，不清理任何历史结果。禁止使用 `screen`，后台运行
请使用外部 `nohup`：

```bash
conda activate rgbt5090
cd /root/autodl-tmp/IRRA-main-mraf/IRRA-main

nohup bash scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh \
  --data-root /root/autodl-tmp/TextReID \
  --output-root /root/autodl-tmp/MRAF_GATE_AMPLITUDE_RUNS \
  --base-config /root/autodl-tmp/history/mdrc_mraf/configs.yaml \
  --reference-screen-root /root/autodl-tmp/MRAF_GATE_RESCUE_RUNS_da8c836e/20260725_224709_da8c836e_mraf_gate_rescue_seed1 \
  --reference-run-root /root/autodl-tmp/MRAF_OLD_SEED1 \
  --device 0 \
  --seed 1 \
  --auto-continue \
  --shutdown-on-success \
  > /root/autodl-tmp/mraf_gate_amplitude_driver.log 2>&1 &
```

`--reference-screen-root` 必须显式传入，脚本没有硬编码 AutoDL 绝对路径。
只有 `STATUS=COMPLETED`、正式 checkpoint、11 场景、24-batch 诊断和压缩包
全部存在时才允许关机。任意失败、mock、dry-run 或筛选未通过均不关机。

查询：

```bash
tail -f /root/autodl-tmp/mraf_gate_amplitude_driver.log
cat <pipeline-run>/status.txt
column -t -s $'\t' <pipeline-run>/amplitude_candidate_ranking.tsv
column -t -s $'\t' <pipeline-run>/calibration_screen.tsv
cat <pipeline-run>/selected_amplitude_candidate.json
cat <pipeline-run>/FINAL_REPORT.txt
ps -ef | grep -E 'run_mraf_gate_amplitude|train.py|test.py'
nvidia-smi
```

恢复：

```bash
bash scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh \
  --resume-run-root <pipeline-run> \
  --seed 1 \
  --auto-continue \
  --shutdown-on-success
```

安全校验：

```bash
bash scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh --help
bash scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh --mock
bash scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh --mock --dry-run
```

mock/dry-run 不调用真实 `train.py`、`test.py`，不访问 GPU，不启动后台进程，
不 sleep，不关机；mock 写 `STATUS=MOCK_COMPLETED`，不会伪装成正式完成。
失败状态包括 `AMPLITUDE_SCREEN_FAILED`、
`CALIBRATION_FAILED`、`CONFIRMATION_FAILED` 和 `FAILED`；状态文件同时记录
`FAILED_STAGE`、`EXIT_CODE`、`UPDATED_AT`。

## 输出

流水线生成：

- `amplitude_screen_metrics.tsv`
- `amplitude_screen_mechanism.tsv`
- `checkpoint_comparison.tsv`
- `amplitude_candidate_ranking.tsv`
- `calibration_screen.tsv`
- `selected_amplitude_candidate.json`
- `confirmation_summary.tsv`
- `final_metrics.tsv`
- `final_robustness_summary.tsv`
- `final_mechanism_summary.tsv`
- `comparison_with_previous_screen.tsv`
- `comparison_with_old_seed1.tsv`
- `FINAL_REPORT.txt`
- `experiment_manifest.txt`
- `mraf_gate_amplitude_rescue_reports.tar.gz`

## Legacy 输出兼容与 amplitude_screen 恢复

2026-07-27 的真实 AutoDL 失败不是 Stage 1 数据缺失：A10、A50、A100、B50
的 best/last 共 8 个单元均已完成训练、8 场景评估和快速机制诊断。失败发生在
`amplitude_screen` 汇总器只搜索新版文件名，未找到历史 S4 的真实文件
`<reference-root>/screen/rescue_screen_summary.tsv`。旧文件的退化指标字段为
`key_degraded_mean_R1`。

汇总器现在按固定优先级只读搜索：

1. `screen/rescue_screen_summary.tsv`
2. `rescue_screen_summary.tsv`
3. `screen/rescue_screen_metrics.tsv`
4. 其他已知的旧版 summary、metrics、ranking 和 comparison 位置

`clean_R1`、`clean_r1`、`clean_rank1`、`R1_clean` 会统一为
`clean_R1`；`key_degraded_mean_R1`、`degraded_mean_R1`、
`mean_degraded_R1`、`key_degraded_R1_mean` 会统一为
`key_degraded_mean_R1`。候选必须在去除首尾空白后精确等于 `S4`，不会把
`S40` 当作 S4；NaN、Inf 和无法转换的数值会被明确拒绝。历史目录、文件内容
和 mtime 均不会被修改。

历史 `mraf_epoch_metrics.tsv` 若包含
`target_rgb_meantarget_rgb_std` 或
`scorer_first_weight_normscorer_last_weight_norm`，只读解析器仅按这两个白名单
拆分表头，并要求拆分后的列数严格等于每一行的数据列数。未知合并表头或行宽不一致
会失败，不会猜测 schema，也不会回写历史 TSV。报告中的
`legacy_header_repaired` 和 `legacy_header_path` 仅表示读取时做了兼容映射，
不表示原文件已修复。

未来训练写出的 `mraf_epoch_metrics.tsv` 使用唯一固定字段表、
`csv.DictWriter(delimiter="\t", lineterminator="\n",
extrasaction="raise")`，写入前校验字段唯一、非空和关键字段齐全。每次追加都会在
同目录临时文件中生成并校验完整 header/row 列数，再用 `os.replace` 原子替换。
汇总 TSV/JSON 和 `status.txt` 同样采用临时文件后原子替换，避免失败时留下半行
或半个文件。

### 恢复前只读 dry-run

先部署 hotfix，再执行：

```bash
conda activate rgbt5090
cd /root/autodl-tmp/IRRA-main-mraf/IRRA-main

bash scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh \
  --resume-run-root /root/autodl-tmp/MRAF_GATE_AMPLITUDE_RUNS_c667946b/20260727_092554_c667946b_mraf_gate_amplitude_rescue_seed1 \
  --device 0 \
  --auto-continue \
  --dry-run
```

有效计划必须输出 `RESUME_PLAN_VALID=true`、原训练 commit、当前聚合 commit、
四个训练候选和 8 个 checkpoint/evaluation/diagnosis 单元的跳过计数，以及
`NEXT_STAGE=amplitude_screen`。dry-run 不写 `status.txt`、
`resume_history.tsv` 或任何报告，不调用训练、评估、诊断、GPU 或 shutdown。

恢复只接受原始 commit 精确为
`c667946b2c51740bfa7892f5a2daee73d1279b44` 的 run，且当前 commit 必须是其
Git 后代；跨 commit 的差异只能包含本次兼容/resume 热修文件。还会校验：

- 原状态为 `FAILED/amplitude_screen`，seed、base config 和 reference roots 不变；
- 固定 A10/A50/A100/B50 矩阵与 Stage 1 协议未变；
- `candidate_status.tsv` 的 8 个单元均为 `COMPLETED`；
- 每个单元的 checkpoint、config、训练日志、两份训练期 TSV 均非空；
- 8 场景评分和快速诊断必需文件均非空；
- `unit.json` 没有 NaN、Inf 或 OOM 标记。

任何一项不满足都会输出 `RESUME_PLAN_VALID=false` 和具体缺失项并退出，绝不会
静默重跑整个 Stage 1。

### 正式 nohup 恢复

dry-run 通过后：

```bash
nohup bash scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh \
  --resume-run-root /root/autodl-tmp/MRAF_GATE_AMPLITUDE_RUNS_c667946b/20260727_092554_c667946b_mraf_gate_amplitude_rescue_seed1 \
  --device 0 \
  --auto-continue \
  --shutdown-on-success \
  > /root/autodl-tmp/mraf_gate_amplitude_resume.log 2>&1 &

tail -f /root/autodl-tmp/mraf_gate_amplitude_resume.log
```

正式恢复复用原 run-root，先把旧状态写入 `resume_history.tsv`，再原子更新
`status.txt` 为 RUNNING；不改 `experiment_manifest.txt` 中的训练 commit，
不追加重复的 candidate status，也不调用 Stage 1 的 `train.py`、`test.py`
或诊断工具，而是直接重新执行 screen 汇总。随后仍按原门禁进入 STRONG 的确认训练、
VIABLE 的只读校准或无 VIABLE 的 `AMPLITUDE_SCREEN_FAILED`。失败绝不关机，只有
最终正式产物完整且状态为 COMPLETED 才允许 shutdown。

检查没有重跑 Stage 1 及恢复结果：

```bash
cat <run-root>/resume_history.tsv
cat <run-root>/status.txt
cat <run-root>/screen_summary.log
cat <run-root>/selected_amplitude_candidate.json
column -t -s $'\t' <run-root>/amplitude_candidate_ranking.tsv
grep -E 'STAGE1_.*_SKIPPED|Skipping completed Stage-1' \
  /root/autodl-tmp/mraf_gate_amplitude_resume.log
```

## calibration 完成后的安全恢复

2026-07-27 的真实 AutoDL run 已完成 Stage 1 的 8 个单元和 calibration 的
5 个 scale（`2/5/10/20/50`），随后在进入确认训练前因磁盘预检失败。历史状态为
`STATUS=FAILED`、`FAILED_STAGE=calibration`、`EXIT_CODE=2`；该失败不表示 calibration
产物无效。兼容未来版本时，同一恢复入口也接受
`FAILED_STAGE=confirmation_preflight`，但不会放宽其他校验。

恢复器只在以下条件全部满足时跳过 Stage 1 和 calibration：

- 原训练 commit 为 `c667946b2c51740bfa7892f5a2daee73d1279b44`，当前 commit 是
  `ed217ac5875c31a2d818fb643554b8fd8dcbc732` 的 Git 后代，跨 commit
  变更仅包含允许的 Pipeline、汇总器、测试和文档热修文件；
- seed 为 1，Stage 1 的 8 个单元和 calibration 的 5 个单元均为
  `COMPLETED/EXIT_CODE=0`，每个单元都有完整 checkpoint、配置、日志、8 场景评估和诊断；
- `calibration_screen.tsv` 中唯一通过强门槛的是
  `scale=2, candidate=B50, checkpoint=best`；
- `selected_amplitude_candidate.json` 精确记录
  `selection_status=STRONG_SELECTED`、`source=read_only_calibration`、
  `mraf_scorer_lr_mult=50`、`mraf_logit_scale=2`、
  `mraf_rel_lambda=1`、`mraf_base_grad_scale=1`；
- 不存在 confirmation 汇总、非空 confirmation 目录或已完成 confirmation 状态，
  且 run-root 下没有活跃的训练、评估或诊断进程。

任一条件不满足都会输出 `RESUME_PLAN_VALID=false` 并退出；不会猜测、覆盖或重跑已有
Stage 1/calibration。正式恢复会在 `resume_history.tsv` 追加兼容性说明，并保留原
manifest 和历史状态行。恢复后的下一阶段固定为 `confirmation`：确认训练仍从原始
pretrained 初始化开始跑 12 epoch，不从 screening/calibration checkpoint 续训；
通过后正式 60 epoch 同样从原始 pretrained 初始化开始，实验协议不变。

磁盘检查发生在确认训练前时，失败阶段写为 `confirmation_preflight`；正式训练前的
磁盘检查仍写为 `formal_preflight`。失败时不会关机，只有最终正式产物完整且状态为
`COMPLETED` 时，`--shutdown-on-success` 才允许关机。

### AutoDL 恢复前只读检查

```bash
conda activate rgbt5090
cd /root/autodl-tmp/IRRA-main-mraf/IRRA-main

bash scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh \
  --resume-run-root /root/autodl-tmp/MRAF_GATE_AMPLITUDE_RUNS_c667946b/20260727_092554_c667946b_mraf_gate_amplitude_rescue_seed1 \
  --device 0 \
  --auto-continue \
  --dry-run
```

有效 dry-run 必须至少显示：

```text
RESUME_PLAN_VALID=true
CALIBRATION_UNITS_SKIPPED=5
CALIBRATION_EVALUATIONS_SKIPPED=5
CALIBRATION_DIAGNOSTICS_SKIPPED=5
SELECTED_CANDIDATE=B50
SELECTED_CHECKPOINT=best
SELECTED_LOGIT_SCALE=2
NEXT_STAGE=confirmation
```

dry-run 是只读的：不写 `status.txt`/`resume_history.tsv`，不创建 confirmation
目录，不运行训练、评估、诊断、GPU 命令或关机命令。

### AutoDL 正式恢复

磁盘空间满足要求且 dry-run 校验通过后：

```bash
nohup bash scripts/mraf/run_mraf_gate_amplitude_rescue_pipeline.sh \
  --resume-run-root /root/autodl-tmp/MRAF_GATE_AMPLITUDE_RUNS_c667946b/20260727_092554_c667946b_mraf_gate_amplitude_rescue_seed1 \
  --device 0 \
  --auto-continue \
  --shutdown-on-success \
  > /root/autodl-tmp/mraf_gate_amplitude_resume_after_calibration.log 2>&1 &

tail -f /root/autodl-tmp/mraf_gate_amplitude_resume_after_calibration.log
```

该命令不会重跑 Stage 1 或 calibration。失败时保留现场且不关机；成功时仅在全部正式
产物校验完成后关机。

最终 `FINAL_REPORT.txt` 会分别记录
`training_commit=c667946b...` 与
`resume_aggregation_commit=<hotfix SHA>`；后者只是兼容汇总代码，不能写成训练
权重所用的 commit。失败的汇总目录或部分日志不能作为最终实验结论。

所有结果仍只属于 seed 1。短周期筛选只用于选择实验配置，不能直接作为论文
最终结论；只有通过后才具备继续 seed 2–5 的条件。
