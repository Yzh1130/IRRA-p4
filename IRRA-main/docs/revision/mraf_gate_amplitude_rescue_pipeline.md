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

所有结果仍只属于 seed 1。短周期筛选只用于选择实验配置，不能直接作为论文
最终结论；只有通过后才具备继续 seed 2–5 的条件。
