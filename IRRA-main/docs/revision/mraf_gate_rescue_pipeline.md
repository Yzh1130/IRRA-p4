# MRAF 门控修复与自动筛选流水线

## 目的与诊断依据

正式机制诊断显示：reliability target 的退化场景标准差约为 `0.284`，
并未塌缩；但 learned alpha 的标准差约为 `0.00036`，alpha-target
Pearson 约为 `-0.0066`，`weighted_mraf_rel/loss_base` 中位数约为
`4.03e-5`。同时原始 gate logits 是 FP16，alpha 落在约
`0.000488` 的量化台阶上；主任务传给 scorer 的梯度约为 reliability
梯度的 11 倍。checkpoint 参数、target 公式、梯度链和
`eval_use_fused_mu` 融合评估路径均正常，强制 RGB-only/IR-only 会改变检索
结果。

因此本轮不修改 target、MDRC、退化、CLIP 或基础损失，而只提供两个默认关闭
的可选机制，并通过固定候选筛选判断它们是否改善门控：

- `--mraf_gate_fp32`：仅让 MRAF scorer、logits、softmax alpha 和融合乘法
  保持 FP32；默认 `False`。
- `--mraf_gate_base_grad_scale`：只缩放基础检索损失经 alpha 传入 scorer 的
  梯度，范围 `[0,1]`；默认 `1.0`。reliability loss 始终使用原始 alpha，
  不受该值缩放。

默认 `False/1.0` 保持旧 checkpoint、前向数值和梯度语义。新旧 checkpoint
仍使用现有严格加载规则。

## 正式输入

真实运行必须显式提供：

- `--data-root`：TextReID 数据根目录，流水线会检查 `Caption.json`。
- `--output-root`：仅用于创建新的时间戳结果目录，不用于发现旧实验。
- `--base-config`：AutoDL 历史正式 MDRC+MRAF 的 `configs.yaml`。流水线会
  核对 batch size 64、`sdm+mlm+id`、MLM、warmup 5、
  `eval_use_fused_mu`、MDRC scale/lambda/prob/warmup 等已确认协议。
- `--reference-run-root`：旧 `mraf_only_seed1`、`mdrc_mraf_seed1` 和
  `mdrc_mraf_norel_seed1` 的结果根目录，只用于最终对照汇总。

若省略 `--base-config`，可在 `--reference-run-root` 中唯一发现时回退自动
发现；正式运行绝不依赖在新的 `--output-root` 中寻找旧配置。

## 候选矩阵与筛选流程

| 候选 | gate FP32 | reliability lambda | base gate grad scale |
| --- | --- | ---: | ---: |
| S0 | false | 0.01 | 1.00 |
| S1 | true | 0.01 | 1.00 |
| S2 | true | 0.25 | 1.00 |
| S3 | true | 1.00 | 1.00 |
| S4 | true | 0.25 | 0.10 |
| S5 | true | 0.25 | 0.00 |

六组候选均从相同预训练初始化独立训练 MDRC+MRAF 8 epoch，
`mraf_rel_warmup_epoch=1`，seed、数据顺序和其余正式协议相同；候选之间不会
续训。每组完成 clean、缺失、0.10 noise、k7 blur 和 both blur 共 8 场景
评估，并以 `max-batches=4`、`gradient-batches=3`、
`gradient-microbatch-size=4` 运行机制诊断。

硬筛选要求完整且无 NaN/Inf/OOM，并同时满足 alpha std、Pearson、缺失模态
权重、四个单模态 noise/blur 方向、reliability/base loss 比例、S0 clean 与
退化容差、scorer 已更新和融合路径敏感等条件。只在通过者中依次按关键退化
平均 R1、clean R1、Pearson、缺失模态 alpha、较小 reliability lambda、
较大 base gradient scale 和候选名稳定排序。

无通过者时写 `STATUS=SCREEN_FAILED`，保留全部短周期资产，不启动正式训练，
也不关机。短周期筛选只用于选配置，不能直接作为论文最终结论。

## 正式训练、评估、诊断与输出

只有显式传入 `--auto-continue` 且存在通过者时，流水线才从相同预训练初始化
重新训练所选配置 60 epoch；不会从 8 epoch checkpoint 续训，正式
`mraf_rel_warmup_epoch` 恢复为 3。随后独立加载 `best.pth`，完成 11 个标准
场景评估，以及 `max-batches=24` 的完整机制诊断。

主要输出包括：

- `status.txt`、`manifest.txt`、`environment.txt`、PID 和逐阶段日志；
- `candidate_status.tsv`、`selected_candidate.txt`；
- 六组候选的 config、`best.pth`、训练日志、score 和诊断；
- `rescue_screen_metrics.tsv`、`rescue_screen_mechanism.tsv`、
  `rescue_screen_summary.tsv`、`rescue_candidate_ranking.tsv`；
- 正式 `metrics.tsv`、`robustness_summary.tsv`、11 个 raw score；
- `final_metrics.tsv`、`final_robustness_summary.tsv`、
  `final_mechanism_summary.tsv`、`comparison_with_old_seed1.tsv`、
  `experiment_manifest.txt`、`FINAL_REPORT.txt`；
- `mraf_gate_rescue_reports.tar.gz`。

只有 60 epoch、11 场景、24-batch 诊断、无 NaN/Inf/OOM、报告和压缩包全部
成功时，才写 `STATUS=COMPLETED` 和 `EXIT_CODE=0`。

## 启动与监控

```bash
nohup bash scripts/mraf/run_mraf_gate_rescue_pipeline.sh \
  --data-root /root/autodl-tmp/TextReID \
  --output-root /root/autodl-tmp/mraf_gate_rescue \
  --base-config /root/autodl-tmp/reference/formal/mdrc_mraf/configs.yaml \
  --reference-run-root /root/autodl-tmp/reference \
  --device 0 \
  --seed 1 \
  --auto-continue \
  --shutdown-on-success \
  > /root/autodl-tmp/mraf_gate_rescue_driver.log 2>&1 &
```

禁止使用 `screen`。监控与正常停止：

```bash
tail -f /root/autodl-tmp/mraf_gate_rescue_driver.log
cat <pipeline-run>/status.txt
column -t -s $'\t' <pipeline-run>/candidate_status.tsv
ps -ef | grep -E 'run_mraf_gate_rescue_pipeline|train.py|test.py'
nvidia-smi
kill <错误实验PID>
```

## 恢复

恢复目录必须由本流水线创建并含 `manifest.txt`、`status.txt`；已
`STATUS=COMPLETED` 的目录拒绝重跑。恢复会核对 commit、seed、base config、
reference root 和固定候选矩阵，跳过已完整成功的候选及 score/诊断，不覆盖
checkpoint、score 或诊断。失败候选会写到新的 `resume_attempt_*` 子目录。

```bash
bash scripts/mraf/run_mraf_gate_rescue_pipeline.sh \
  --resume-run-root /root/autodl-tmp/mraf_gate_rescue/<pipeline-run> \
  --auto-continue \
  --shutdown-on-success
```

## dry-run、mock、失败和关机保护

`--dry-run` 只验证数据、输出父目录、base config、reference root 和正式协议，
然后打印候选与路径；不创建 run 目录、不训练。`--dry-run --mock` 可在无
数据/GPU时只检查 CLI 和矩阵。

`--mock` 使用临时 JSON/CSV 假结果验证候选失败继续、筛选、排序和状态机，
不调用 `train.py`、`test.py`、CUDA、nohup 或 shutdown。关机命令可通过
`MRAF_SHUTDOWN_COMMAND` 注入测试替身；真实关机仍只有在显式
`--shutdown-on-success`、正式任务完整成功且压缩包存在时才可能执行。

失败会写 `STATUS=FAILED`、`FAILED_STAGE` 和 `EXIT_CODE`；筛选失败和 mock
完成均不会触发关机。
