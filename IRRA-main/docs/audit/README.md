# Stage 0 只读审计工具

本目录保存论文实验重投前的可复核审计报告。工具只读取数据、实验资产、源码和 Git 元数据；不会联网，不会启动训练、优化器、反向传播、DataLoader 迭代或模型 forward，也不会加载 checkpoint/score matrix 的内容。

允许写入的范围只有 Python 项目根下的 `audit_out/` 与 `docs/audit/`。其中 `audit_out/` 保存机器可读的 JSON/CSV，`docs/audit/` 保存中文 Markdown 报告。

## 运行位置

以下命令均从 MRAF Python 项目根运行：

```bash
cd /root/autodl-tmp/IRRA-mraf-ir-v4-6b469e2b/IRRA-main
```

`probe_dataset.py` 的 `--data-root` 应传入与训练参数 `--root_dir` 相同的目录，即其下包含 `RGBT-PEDES/` 或 `RGBNT201-PEDES/` 的父目录，而不是直接传入图像子目录。

代码路径、超参数、可靠性目标、退化注入、checkpoint、数据集扩展点、随机种子、优化器参数组与评测口径的静态清点见 [`CODE_INVENTORY.md`](CODE_INVENTORY.md)。

## 四个审计模块

数据与 loader 口径审计：

```bash
python -B -m tools.audit.probe_dataset \
  --dataset-name RGBT-PEDES \
  --data-root /path/to/TextReID \
  --repo . \
  --out audit_out
```

只输出候选标注的结构探测结果：

```bash
python -B -m tools.audit.probe_dataset \
  --dataset-name RGBNT201-PEDES \
  --data-root /path/to/TextReID \
  --repo . \
  --dump-schema \
  --out audit_out
```

实验资产审计：

```bash
python -B -m tools.audit.audit_assets \
  --roots /path/to/mraf_runs /path/to/legacy_logs \
  --repo .. \
  --out audit_out
```

超参数探索历史审计：

```bash
python -B -m tools.audit.audit_hparams \
  --roots /path/to/mraf_runs /path/to/legacy_logs \
  --repo .. \
  --out audit_out
```

仓库状态审计：

```bash
python -B -m tools.audit.audit_repo_state \
  --repo .. \
  --out audit_out
```

## 总入口

`run_all_audits.sh` 依次运行四个模块。单步失败不会中断后续步骤：

```bash
DATA_ROOT=/path/to/TextReID \
ASSET_ROOTS='"/path/to/mraf runs" "/path/to/legacy logs"' \
REPO=/root/autodl-tmp/IRRA-mraf-ir-v4-6b469e2b \
bash tools/audit/run_all_audits.sh
```

未设置环境变量时，默认值分别为项目内 `datasets/TextReID`、`mraf_runs logs`，以及 Python 项目根的父目录（Git worktree 根）。若路径不存在，脚本会在报告中标记“未找到”或“无法判定”，并继续其它检查。

`ASSET_ROOTS` 使用 shell 引号规则拆分，因此包含空格的单个路径应像上例一样在变量值内再加一层引号。

## 输出

| 模块 | 机器可读输出 | Markdown 报告 |
| --- | --- | --- |
| `probe_dataset` | `audit_out/dataset_probe.json` | `docs/audit/DATA_AUDIT_<dataset-name>.md` |
| `audit_assets` | `audit_out/assets.csv` | `docs/audit/EXPERIMENT_ASSET_AUDIT.md` |
| `audit_hparams` | `audit_out/hparam_history.json` | `docs/audit/HPARAM_HISTORY.md` |
| `audit_repo_state` | `audit_out/repo_state.json` | `docs/audit/REPO_STATE.md`、`WORKTREE_DIRTY_SNAPSHOT.md` |

所有报告中的事实项均带 `source`。`status=UNKNOWN` 表示证据缺失或解析失败，不等于数值为零；人工结论不能由脚本自动替代。
