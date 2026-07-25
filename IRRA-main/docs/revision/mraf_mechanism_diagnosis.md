# MRAF 门控机制只读诊断

## 用途与边界

`tools/revision/diagnose_mraf_mechanism.py` 用于解释 MRAF 门控权重接近
`0.5/0.5` 的可能原因。工具复用现有配置加载、数据加载、模型构建、
checkpoint 加载、评估退化和 reliability target 公式，覆盖 checkpoint
参数、gate logits、target、门控方向、梯度量级和融合路径六类检查。

该工具不会执行 `optimizer.step()`，不会保存或修改 checkpoint，也不会改变
训练默认参数。梯度诊断仅在内存中执行前向和 `torch.autograd.grad`；强制门控
仅在上下文管理器内临时替换当前 MRAF 实例的 `forward`，退出时恢复原状态。
阈值只用于产生诊断警报，不等于论文结论，也不会反馈到训练行为。

## 输入目录与命令

工具在 `<run-root>/formal/` 下分别查找唯一的：

- `*_mraf_only_seed1`
- `*_mdrc_mraf_seed1`
- `*_mdrc_mraf_norel_seed1`

每个 case 目录中必须能唯一找到 `configs.yaml`、`best.pth` 和 `train.log`。
正式运行示例：

```bash
python -B tools/revision/diagnose_mraf_mechanism.py \
  --run-root /root/autodl-tmp/mraf_runs \
  --data-root /root/autodl-tmp/TextReID \
  --cases mraf_only mdrc_mraf mdrc_mraf_norel \
  --device cuda:0 \
  --degrade-seed 2026 \
  --max-batches 24 \
  --gradient-batches 3 \
  --output-dir /root/autodl-tmp/mraf_diagnosis
```

先检查目录发现结果、但不加载模型和不创建输出目录：

```bash
python -B tools/revision/diagnose_mraf_mechanism.py \
  --run-root /root/autodl-tmp/mraf_runs \
  --data-root /root/autodl-tmp/TextReID \
  --cases mraf_only mdrc_mraf mdrc_mraf_norel \
  --device cuda:0 \
  --degrade-seed 2026 \
  --max-batches 24 \
  --gradient-batches 3 \
  --output-dir /root/autodl-tmp/mraf_diagnosis \
  --dry-run
```

为避免覆盖诊断资产，非空的 `--output-dir` 会被拒绝。诊断过程不会写入训练
结果目录。

## 退化场景

场景定义直接复用 `utils.metrics.Evaluator._apply_eval_degradation`，随机种子通过
既有 `eval_degrade_seed` 接口传入：

| 场景 | 定义 |
| --- | --- |
| `clean` | 原始 RGB 与 thermal |
| `rgb_missing` | RGB 缺失 |
| `ir_missing` | thermal 缺失 |
| `rgb_noise_0.05` / `rgb_noise_0.10` | 仅 RGB 加对应标准差噪声 |
| `ir_noise_0.05` / `ir_noise_0.10` | 仅 thermal 加对应标准差噪声 |
| `rgb_blur_k7` / `ir_blur_k7` | 仅指定模态使用 `k=7` 模糊 |
| `both_noise_0.05` | 两模态同时加 `0.05` 噪声 |
| `both_blur_k7` | 两模态同时使用 `k=7` 模糊 |

相同场景和相同 batch 使用确定性 seed。样本顺序会写入 SHA256，便于比较复跑。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `checkpoint_parameter_stats.csv` | 所有 `mraf_fusion.*` 参数的 shape、numel、mean、std、abs mean/max、L2、零占比、角色和哈希；包含 scorer 首末线性层识别 |
| `scenario_gate_statistics.csv` | 每 case/场景的 `logit_rgb`、`logit_ir`、gap、`alpha_rgb`、`alpha_ir` 的均值、标准差和分位数 |
| `scenario_target_statistics.csv` | drift、target 分布及 alpha-target MAE/MSE、Pearson、Spearman、方向一致率 |
| `sample_level_diagnostics.pt` | 每 case/场景的逐样本 FP32 logits、alpha、drift 和 target |
| `gate_direction_statistics.csv` | 同一样本从 clean 到退化后的门控变化方向、均值、中位数、均值方向布尔值及正确方向比例 |
| `gradient_diagnostics.csv` | base、reliability、加权比例、total 及 scorer 首末层的 total/rel-only 梯度范数 |
| `fusion_path_check.json` | learned、0.5/0.5、RGB-only、IR-only 四种门控的特征距离、余弦、hash、检索 score 与 top-1 变化 |
| `summary.txt` | 人可读的 case 级警报和关键数值 |
| `diagnosis.json` | 机器可读结论、阈值和可重复性元数据 |

所有 gate logits、alpha、target 和相关统计都会先转换为 `torch.float32`。即便模型
主体处于混合精度，也不会只用 FP16 值计算统计量；若捕获到原始 gate 输出为
FP16/BF16，会产生 `FP16_QUANTIZATION_WARNING`。

## 指标含义

- `drift_rgb` / `drift_ir`：训练公式中的
  `1 - cosine(degraded_feature, clean_feature.detach())`。
- `target_rgb` / `target_ir`：复用
  `model.mraf.compute_mraf_reliability_targets`，即对负 drift 按训练温度做
  softmax 后 detach。
- `logit_gap`：`logit_rgb - logit_ir`；其绝对值接近零通常对应 alpha 接近
  `0.5/0.5`。
- `alpha-target` Pearson/Spearman：使用独立的 RGB 权重通道，分别描述线性相关
  和排序相关（IR 权重是其互补项）；任一输入为常数时相关系数记为未定义，
  而不是伪造为零。
- `direction_agreement`：以 `0.5` 为中心，alpha 与 target 偏向同一模态的样本比例。
- `weighted_mraf_rel / loss_base`：reliability 辅助项相对于既有 base loss 的量级。
- 梯度范数：用 `torch.autograd.grad` 分别检查 total loss 和仅
  `weighted_mraf_rel` 对 scorer 的作用；不写入 `.grad`，不更新参数。
- 融合路径：在同一图像 batch 与文本特征上比较四种门控，确认
  `eval_use_fused_mu` 当前评估特征确实对门控敏感。

## 自动警报阈值

| 警报 | 默认诊断条件 |
| --- | --- |
| `PARAMETER_NOT_UPDATED` | scorer 最后一层 L2 norm `<= 1e-8`，或三组 MRAF 参数完全相同 |
| `TARGET_COLLAPSED_TO_HALF` | target 标准差 `< 0.02` |
| `ALPHA_COLLAPSED_TO_HALF` | alpha 标准差 `< 0.005` |
| `ALPHA_TARGET_LOW_CORRELATION` | Pearson 未定义或 `< 0.3` |
| `RELIABILITY_LOSS_TOO_SMALL` | `weighted_mraf_rel/loss_base < 1e-4` |
| `GATE_GRADIENT_TOO_SMALL` | scorer 最后一层 total gradient norm `<= 1e-8` |
| `FUSION_PATH_NOT_SENSITIVE` | 强制门控后 fused feature 未发生可测变化 |
| `FP16_QUANTIZATION_WARNING` | 捕获到原始 gate logits/alpha 为 FP16/BF16 |

阈值用于快速定位，不替代训练曲线、检索指标或论文实验结论。

## 如何理解常见结果

- **alpha 塌缩、target 不塌缩**：门控没有学到 target，优先查看相关性、
  reliability loss 比例、末层参数和梯度。
- **target 也塌缩**：现有退化在表征空间产生的两模态 drift 差异不足，alpha
  接近一半可能是在拟合 target，而不一定是优化失败。
- **梯度过小**：若 loss 比例也很小，优先检查调度后的
  `lambda_mraf_rel_cur`；若 loss 不小但末层梯度仍小，再检查计算图和输入特征。
- **参数未更新**：末层仍接近初始化零值说明 scorer 可能未获得有效更新；三组
  checkpoint 完全相同也会被单独标记。
- **融合路径不敏感**：强制门控未改变 fused feature、检索 score 或 top-1 时，
  才应怀疑评估路径绕开门控。单独 top-1 不变并不足以得出该结论。

## 限制

梯度诊断以训练配置重建训练 dataloader，并调用现有
`processor.processor` 的 base、MDRC 与 MRAF reliability 组装函数。它面向当前
三种正式 case，不模拟 optimizer、scheduler 或 epoch 间参数变化。
`gradient-batches` 只限制诊断 batch 数，不能代表完整 epoch。融合路径使用当前
batch 内文本作为小型检索候选集，适合验证路径敏感性，不等同于完整验证集指标。
