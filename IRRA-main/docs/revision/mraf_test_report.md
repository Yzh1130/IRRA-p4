# MRAF 提交前最终测试报告

测试日期：2026-07-24。测试目录：
`C:\Users\12823\Desktop\IRRA\IRRA-main-mraf\IRRA-main`。

## 执行命令

```text
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_mraf.py
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q
bash -n scripts/mraf/run_mraf_experiments.sh
bash scripts/mraf/run_mraf_experiments.sh --case mraf_only --seed 1 --num-epoch 60 --batch-size 64 --loss-names sdm+mlm+id --output-root ./tmp_mraf_formal_check --dry-run
```

## 最终结果

- MRAF 定向测试：**14 passed，0 failed，2 warnings，8.83s**。
- 全量 pytest：**56 passed，1 skipped，0 failed，2 warnings，27.01s**。
  跳过项为 CUDA 退化模式测试，原因是当前本地测试进程检测到
  `CUDA is not available`。
- 两个分析 CLI 的上一轮最终审计结果保持为 `--help` 退出码 0；本轮聚焦协议
  核查未重复执行。
- Shell：
  - `bash -n scripts/mraf/run_mraf_experiments.sh`：退出码 0。
  - 文件保持纯 LF（`CR_COUNT=0`），合法 shebang 为
    `#!/usr/bin/env bash`。
- 60 epoch 正式参数 MRAF-only dry-run：
  - 显式传入 `--batch-size 64 --loss-names sdm+mlm+id`，退出码 0。
  - 输出 `batch_size=64`、`effective_batch_size=64`、
    `loss_names=sdm+mlm+id`、`id_loss_enabled=true`、
    `mlm_loss_enabled=true`。
  - 输出 `warmup_epochs=5`、`eval_use_fused_mu=true`、
    `training_protocol_verified=true`。
  - 最终 `training_command` 同时包含 `--batch_size 64`、
    `--loss_names sdm+mlm+id`、`--MLM`、`--warmup_epochs 5`、
    `--eval_use_fused_mu` 和 `--mraf_enabled`。
  - 不要求 `MDRC_LOSS_SCALE`。
  - 仅打印命令，未创建输出目录、未启动 Python、未生成 checkpoint。
- 正式训练的 `--batch-size` 和 `--loss-names` 显式门禁保持不变。
- MDRC 的正式 `mdrc_loss_scale` 已确认为 100，但
  `MDRC_LOSS_SCALE` 环境变量门禁保持不变，脚本没有隐藏默认值。
- `git diff --check`：退出码 0。

## Warning

两次 pytest 均出现相同的两个非失败 warning：

1. protobuf 使用已弃用的 `datetime.datetime.utcfromtimestamp()`。
2. 既有 checkpoint loader 调用 `torch.load(weights_only=False)`，PyTorch
   提示未来默认值会改变。

本轮限定范围不允许修改 protobuf、PyTorch 或 checkpoint 算法代码，因此仅
如实记录。

## 已确认正式协议

AutoDL 历史正式训练脚本及对应 `configs.yaml` 已确认：

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

`run_mraf_experiments.sh` 不为 `MDRC_LOSS_SCALE` 设置隐式默认值。MRAF-only
不依赖该变量；`mdrc_mraf` 和 `mdrc_mraf_norel` 强制要求显式设置。当前
非 MUM 聚合可能同时计入 `mlm_loss` 与 `loss_mlm`，本轮仅记录风险，不
修改既有行为。

## 配置边界

包装脚本显式传入 `--degrade_aug_mode none`。若环境试图为计划实验设置
  非 `none` 的 `DEGRADE_AUG_MODE`，脚本主动报错，避免 MRAF reliability
  与既有单分支退化增强形成双重退化语义。

## 训练声明

本次最终审计没有运行正式训练，也没有运行任何单 epoch GPU 训练；只执行了
pytest、Shell 静态检查和不启动 Python 的 dry-run。未修改算法代码，未
commit，未 push。
