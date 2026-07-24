# MRAF changed files

## Modified

- `model/build.py`
  - Conditional MRAF construction and centralized train/eval fusion.
  - Clean/degraded/evaluation gate auxiliary outputs.
- `processor/processor.py`
  - Shared degraded branch, reliability target/loss, independent warmup,
    loss composition, metrics, TensorBoard values, and epoch logs.
- `utils/options.py`
  - MRAF CLI arguments, validation, and precision-fusion mutual exclusion.
- `utils/mdrc.py`
  - Optional no-extra-RNG degradation mode metadata.
- `utils/checkpoint.py`
  - Strict MRAF checkpoint restoration and explicit old-checkpoint handling.
- `scripts/mraf/run_mraf_experiments.sh`
  - 保留 `--case` 及原有三个 case 名称。
  - 支持 `--seed`、`--num-epoch`、`--output-root`、`--dry-run`，新增
    `--batch-size` 和 `--loss-names`。
  - 单 epoch 未显式提供 batch/loss 时，仅使用
    `batch_size=16`、`loss_names=sdm+mlm` 作为
    `unverified_smoke_defaults`；`num_epoch>1` 时两项必须由人工显式提供。
  - `--loss-names` 按 `+` 拆分，只接受现有任务
    `sdm`、`id`、`mlm`、`cmpm`、`itc`；ID 判定使用独立 token，不使用子串。
  - 输出 effective batch、梯度累积步数、ID/MLM 状态、MLM alias 风险、协议
    来源以及完整训练协议；MLM 启用时提示非 MUM 聚合的 alias 双计数风险。
  - 根据 AutoDL 历史正式训练脚本和对应 `configs.yaml`，将
    `warmup_epochs` 修正为 5，并在训练命令中显式加入
    `--eval_use_fused_mu`；dry-run 输出 `warmup_epochs=5` 和
    `eval_use_fused_mu=true`。
  - dry-run 只打印解析后的命令、输出目录、Git HEAD 和有效配置，不创建
    实验目录、不启动 Python。
  - 固定 `degrade_aug_mode=none`；检测到非 `none` 配置时主动退出。
  - MRAF-only 不依赖 `MDRC_LOSS_SCALE`；MDRC 相关 case 仍强制要求人工
    显式设置该变量。AutoDL 历史正式值已确认是 100，但脚本不设置隐藏
    默认值。

## Added

- `.gitattributes`
  - 仅设置 `IRRA-main/scripts/mraf/*.sh text eol=lf`。
- `model/mraf.py`
- `tests/test_mraf.py`
- `tools/revision/analyze_mraf_weights.py`
- `tools/revision/summarize_mraf_experiment.py`
- `scripts/mraf/run_mraf_experiments.sh`
- `docs/revision/mraf_autodl_runbook.md`
- `docs/revision/mraf_training_protocol_audit.md`
  - 记录 batch、有效 batch、梯度累积、基础任务、MLM alias 风险及 MDRC
    关键参数的代码/历史证据。
  - 记录 AutoDL 已确认的正式协议：
    `batch_size=64`、`loss_names=sdm+mlm+id`、`MLM=true`、
    `warmup_epochs=5`、`eval_use_fused_mu=true`、
    `mdrc_loss_scale=100`、`mdrc_lambda=0.05`、`mdrc_prob=0.5`、
    `mdrc_warmup_epoch=3`，conda 环境为 `rgbt5090`。
- `docs/revision/mraf_code_audit.md`
- `docs/revision/mraf_mdrc_scale_audit.md`
- `docs/revision/mraf_implementation.md`
- `docs/revision/mraf_experiment_plan.md`
- `docs/revision/mraf_test_report.md`
- `docs/revision/mraf_changed_files.md`

## Intentionally unchanged

- `mdrc_loss_scale` default remains `1.0`.
- CLIP backbone, SDM, MLM, ID, datasets, splits, and paper files.
- Existing logs, checkpoints, and experiment outputs.
- Original worktree and its modified/untracked files.

## Shell 行尾和可执行位

- `scripts/mraf/run_mraf_experiments.sh` 当前保持纯 LF，shebang 为
  `#!/usr/bin/env bash`。
- `.gitattributes` 仅保护 MRAF 脚本目录，不改变其他 Shell 或 Python 文件
  的行尾策略。
- 当前文件尚未暂存，因此 Git 索引中还没有可执行位记录。暂存阶段需要执行：

  ```bash
  git update-index --chmod=+x IRRA-main/scripts/mraf/run_mraf_experiments.sh
  ```

- 本次审计不执行 commit 或 push。
