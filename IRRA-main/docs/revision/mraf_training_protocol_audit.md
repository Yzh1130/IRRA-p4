# MRAF 训练协议聚焦审计

审计范围仅包括训练任务选择、batch/梯度语义及历史启动证据；未运行训练，
未修改 `model/`、`processor/`、`utils/`、checkpoint 或论文。2026-07-24
已由人工从 AutoDL 历史正式训练脚本及对应 `configs.yaml` 核实正式协议。

| 参数 | 当前脚本 | 代码/历史证据 | 结论 | AutoDL 是否需要核实 |
| --- | --- | --- | --- | --- |
| `batch_size` | 单 epoch 未显式传参时仍可临时使用 16；长程必须显式传 `--batch-size 64` | `scripts/mraf/run_mraf_experiments.sh:47-53,105-125,217-231`；AutoDL 历史正式训练脚本及对应 `configs.yaml` 已确认 64 | 正式 Base/MDRC/MRAF 对比协议为 64；16 只保留为本地未核实冒烟默认值 | 已确认，无需再次推断 |
| effective batch size | 当前单进程包装路径等于 `batch_size` | `scripts/mraf/run_mraf_experiments.sh:162-167` 拒绝 `WORLD_SIZE!=1`；`datasets/build.py:553-560` 的 random sampler 直接将 `args.batch_size` 交给 DataLoader；`train.py:55-66` 仅在外部 `WORLD_SIZE>1` 时进入 DDP | 正式协议的有效 batch size 为 64；默认本地冒烟则为 16 | 否；若改变启动器或多卡协议则重新核实 |
| gradient accumulation | 1 step | `processor/processor.py:4645` 的 `do_train()`；每个 batch 在 `processor/processor.py:5742-5744` 依次执行 `zero_grad()`、`backward()`、`step()` | 没有梯度累积，`gradient_accumulation_steps=1` | 否 |
| test batch size | 512 | `scripts/mraf/run_mraf_experiments.sh:194-202,217-231`；`datasets/build.py:618-623` 的独立 test loader 使用 `args.test_batch_size` | 独立测试 batch 为 512；训练过程中的 val loader实际使用训练 `batch_size`，见 `datasets/build.py:577-585` | 否 |
| `loss_names` | 单 epoch 未显式传参时可临时使用 `sdm+mlm`；正式命令必须显式传 `sdm+mlm+id` | `utils/options.py:73-77` 支持 `mlm/cmpm/id/itc/sdm`；AutoDL 历史正式训练脚本及对应 `configs.yaml` 已确认 `sdm+mlm+id` | 正式 Base/MDRC/MRAF 对比协议为 `sdm+mlm+id` | 已确认 |
| ID loss | 正式协议启用 | `model/build.py:795-798` 的 `IRRA._set_task()` 按 `+` 拆分任务；`IRRA.forward()` 仅在 `model/build.py:2413-2425` 的 `"id" in self.current_task` 条件下生成 `id_loss`；AutoDL 已确认 `loss_names=sdm+mlm+id` | 正式命令包含独立 `id` token，因此 ID loss 会计算并进入总损失 | 已确认 |
| MLM | 命令固定传 `--MLM`，正式任务集合包含 `mlm` | `utils/options.py:71` 定义 MLM 数据开关；`IRRA.forward()` 在 `model/build.py:2435-2480` 仅为 `mlm` 任务计算并返回 MLM loss；AutoDL 历史配置确认 `MLM=true` | 正式协议的 MLM 数据开关和 MLM loss 均启用 | 已确认 |
| MLM alias 双计数风险 | MLM 启用时输出 `true` 和警告 | `IRRA.forward()` 同时返回 `mlm_loss` 与 `loss_mlm`（`model/build.py:2469-2480`）；非 MUM 的 `_build_total_loss()` 遍历所有含 `loss` 的 key，但只显式排除 SDM alias（`processor/processor.py:1968-2037`） | 当前非 MUM 路径可能同时计入两个 MLM alias；本次不修改，以保持与既有实验代码路径可比。未来若修复，Base、MDRC 和新方法都必须重新训练 | 是，正式启动前人工知悉并接受 |
| learning rate | `1e-5` | `scripts/mraf/run_mraf_experiments.sh:194-202,217-231`；项目默认见 `utils/options.py:463-466` | 当前脚本与项目默认一致 | 本轮 AutoDL 证据未新增该项 |
| epoch | CLI 默认 60；1 为冒烟；大于 1 触发协议门禁 | `scripts/mraf/run_mraf_experiments.sh:47-53,105-125`；项目默认见 `utils/options.py:474-479` | 正式计划为 60，但必须显式提供已核实 batch/loss | 否 |
| warmup epoch | 5 | `scripts/mraf/run_mraf_experiments.sh:194-202,217-231`；AutoDL 历史正式脚本和 `configs.yaml` 已确认 5 | 正式协议为 5，脚本已修正 | 已确认 |
| `eval_use_fused_mu` | 显式启用 | `scripts/mraf/run_mraf_experiments.sh:194-202,217-231`；真实 CLI 定义见 `utils/options.py:244-245`；AutoDL 历史正式配置已确认 true | 正式训练命令显式传 `--eval_use_fused_mu`，dry-run 输出 true | 已确认 |
| `num_workers` | 8 | `scripts/mraf/run_mraf_experiments.sh:194-202,217-231`；项目默认见 `utils/options.py:485-493` | 当前脚本与项目默认一致 | 否 |
| image augmentation | 启用 `--img_aug` | `scripts/mraf/run_mraf_experiments.sh:217-231`；开关定义见 `utils/options.py:60-64` | 当前命令启用图像增强 | 本轮 AutoDL 证据未新增该项 |
| `mdrc_prob` | MDRC case 为 0.5 | `scripts/mraf/run_mraf_experiments.sh:253-279`；参数定义见 `utils/options.py:86-93`；AutoDL 历史正式配置已确认 0.5 | 正式协议为 0.5 | 已确认 |
| `mdrc_lambda` | MDRC case 为 0.05 | `scripts/mraf/run_mraf_experiments.sh:253-279`；参数定义见 `utils/options.py:86-93`；AutoDL 历史正式配置已确认 0.05 | 正式协议为 0.05 | 已确认 |
| `mdrc_loss_scale` | MRAF-only 不使用；两个 MDRC case 强制要求环境变量且无脚本默认值 | `scripts/mraf/run_mraf_experiments.sh:245-283`；AutoDL 历史正式脚本和 `configs.yaml` 已确认 100 | 正式 MDRC 值为 100，但必须由 `MDRC_LOSS_SCALE=100` 显式传入；脚本不设置隐藏默认值 | 已确认 |
| `mdrc_warmup_epoch` | MDRC case 为 3 | `scripts/mraf/run_mraf_experiments.sh:253-279`；参数定义见 `utils/options.py:86-93`；AutoDL 历史正式配置已确认 3 | 正式协议为 3 | 已确认 |
| `degrade_aug_mode` | 固定 `none`，非 `none` 主动退出 | `scripts/mraf/run_mraf_experiments.sh:194-204,217-234`；选择定义见 `utils/options.py:103-105` | 避免 MRAF reliability 与另一条训练退化增强路径叠加 | 否 |
| conda 环境 | 脚本不负责激活环境 | AutoDL 历史正式训练环境已确认是 `rgbt5090` | AutoDL 运行前激活 `rgbt5090` | 已确认 |

## 核心结论

1. 正式协议已确认 `batch_size=64`，当前单进程且无梯度累积，因此有效 batch
   size 为 64。
2. 正式协议已确认 `loss_names=sdm+mlm+id` 和 `MLM=true`；独立 `id`
   token 会启用 ID loss。
3. `warmup_epochs=5`、`eval_use_fused_mu=true` 已由 AutoDL 历史正式配置
   确认并写入脚本。
4. MDRC 正式协议已确认 `mdrc_loss_scale=100`、`mdrc_lambda=0.05`、
   `mdrc_prob=0.5`、`mdrc_warmup_epoch=3`；其中 scale 仍必须通过
   `MDRC_LOSS_SCALE=100` 显式传入。
5. 训练环境已确认是 conda 环境 `rgbt5090`。
6. 当前非 MUM MLM alias 双计数行为暂不修改，以保持可比性；未来若修复，
   Base、MDRC 和新方法必须全部重新训练。
