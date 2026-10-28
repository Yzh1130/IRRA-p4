# 第十七轮：只读核查 + 常数迁移对照 + 效率与开发集敏感性（v3）

规格见项目文档《论文4_第十七轮_独立裁决与收尾计划》§3（跑前写定）。第十五轮的 C1–C5 判定不变；`p4_r15_out` 只读；冻结脚本 `r15_test.py`、`judge_test.py` 不改。

## 脚本

| 文件 | 作用 | 是否计算新分数 |
|---|---|---|
| r17_check_archives.py | 三个 base 的开发/测试档案：键名、合并后的键集合（测试模式取交集，缺键时两部分一起回退到干净数组）。对每个状态与模态，**开发、测试两部分分别**判定实际读到的数组：目标模态必须变化（缺失、噪声逐行都变，模糊至少一行变），非目标模态必须逐行不变；另查形状、有限值、维度。另报告 M1g 每个分配种子的状态计数 | 否 |
| r17_reagg_r15.py | 用第十五轮保存的 cells.csv 与逐查询记录，先逐位置核对对齐（每个位置的查询均值必须等于对应分配种子的单元值），再按显式键汇总逐混合 × 种子的 mAP/R1/R5/R10/mINP 与配对区间 | 否 |
| r17_const.py | `r15_test.py` 加：QE+Const（开发集按冻结主规则选一个全局常数 g）、QE+NoGate（g = 1，经 r_max 截断）、`--fixed_selection`（沿用第十五轮前端配置，不重新择优）、显式键逐查询记录、带边界说明的计时、开发集敏感性开关 | 是：同一测试集上的事后补充对照（非盲） |
| r17_judge.py | 结构核对、与第十五轮单元逐格复现核对（不通过即停在这里：INVALID、退出码 3、不出任何配对表）、所选 g 与开发表、绝对指标、配对差（AP/R1/R5/R10/mINP，身份簇 bootstrap）、常数曲线（只报告）、局部性、harm/help、top-5 组成、计时；`--devsens_only` 单独出开发集敏感性报告 | 否 |
| p4_r17.sh | AutoDL 一次性驱动：失败即停、按 manifest 断点续跑、第十五轮文件哈希守护、结果打包 | — |

常数网格在任何第十七轮测试运行之前锁定为 `0 0.05 0.1 0.2 0.25 0.5 0.75 1`；并列取网格中较小的值（与 `pooled_select` 相同的比较方式）；无可行值时取最差类 ΔR1 最大者（原回退规则）。

## 驱动的状态与退出码

`p4_r17_out/STATUS` 与 `DIGEST17.txt` 第二行写本次状态：

| 退出码 | STATUS | 含义 |
|---|---|---|
| 0 | `COMPLETE` | 0–6 全部通过 |
| 1 | `STOPPED: <原因>` | conda 激活、0 环境/代码/输入、1a 档案、1b 旧记录对齐、2 常数对照单元、3 结构或复现任一不通过，或任一次核对发现第十五轮文件改动、缺失或新增；之后的步骤都不运行，已在运行的单元等其结束，日志全部保留 |
| 3 | （不改动） | 另有一个本任务在运行 |
| 4 | `INCOMPLETE: DEV SENSITIVITY FAILED; steps 0-3 passed` | 0–3 通过；开发集敏感性有单元失败或报告不全。不算全部完成 |
| 5 | `INCOMPLETE: DEV SENSITIVITY NOT RUN (RUN_DEVSENS=0); steps 0-3 passed` | 只在设 `RUN_DEVSENS=0` 时 |

只有 `COMPLETE` 才算全部完成。输出目录与 `p4_r15_out` 相互包含时拒绝运行，且不写任何文件（这一种情形不改动 STATUS，只在驱动输出里说明）。

## 断点续跑

- 每个计算单元（常数对照 9 个、开发集敏感性 27 个）开算前写 `manifest.json`：`r17_const.py` 的 sha256、全部输入文件（开发档案、测试档案、训练身份档案、第十五轮 selection.json）的 sha256 与大小、完整参数、Python 与 numpy/scipy/sklearn/pandas/torch 版本、线程设置。成功结束才写 `.done`（内容为 manifest 的 sha256）。
- 重跑时只有 `.done` 存在、且目录里的 manifest 与本次应有的逐字节相同才复用。未完成、只有 `.done` 没有 manifest、或 manifest 不同的旧目录，整体移到 `_stale/<单元>_<时间>`（保留，不删除）后重算。
- 1a、1b、3、5 只读且快，每次都重算；上一次的报告先移到 `_history/<时间>/`。
- 报告阶段再核一次：每个常数对照单元的 run_record 里记录的脚本哈希必须等于固定的 `r17_const.py` 哈希。
- 复现核对：6 个共有方法在两边都必须存在；逐格最大差不超过 1e-6；只有一边是 NaN 也算不一致。

## 第十五轮文件

- `p4_r17_out` 与 `p4_r15_out` 必须互不包含（按真实路径判断），否则拒绝运行。
- 首次运行时记录 `p4_r15_out` 的哈希清单（`test/`、`feats/`、`train_ir/` 下全部文件与顶层文件；不含 JupyterLab 打开文件时自动生成的 `.ipynb_checkpoints`），原子写入 `checks/r15_guard_before.sha256`。之后每次开跑时、1b、2、3、4–5 之后和结束时逐行比较，有任何文件改动、缺失或新增即停，差异写到 `checks/r15_guard_diff_<时间>.txt`。
- 若真的被改动，先查明原因并恢复原文件；不要删除清单来绕过。

## 计时与内存的口径

- 输入都是缓存的特征数组：图像/文本编码器与数据读取从不计入。
- `online_bstar_scoring`：完整的 B★ 前端（drop-blank 嵌入、L2 归一化、查询–图库余弦、按可用性 Platt、sigmoid）。`online_bstar_sigmoid_only` 只测最后的 sigmoid，**不能**称作 B★ 前端耗时。`online_frontend_full` = B★ 打分 + QE-DBA。
- `module_memory`：每个阶段一次调用的输入大小与新分配峰值（tracemalloc；不含已存在的数组和 BLAS 工作缓冲）。
- `process_peak_rss_mb`：整个 Python 进程到计时点为止的峰值常驻内存（两份档案、全部拟合模型、所有方法的矩阵、解释器与库），**不是**迁移模块的推理内存。Windows 没有 `resource` 模块，不记录这一项，其余计时照常。

## 本机验收（无真实数据）

```
python -B IRRA-main/tests/p4/test_r17.py
python -B IRRA-main/tests/p4/test_r17_archives.py
```

- `test_r17.py`（8 项）：不带新选项时与 `r15_test.py` 逐位相同；带 `--fixed_selection` 时共有方法逐位相同；QE+Const@0 = QE-DBA、QE+Const@1 = QE+NoGate；局部性；计时边界与两项一致性核对；报告脚本正常退出；共有单元被改动、一边出现 NaN 或脚本哈希不符时 judge 停在复现核对、不出配对表；开发集敏感性单独报告。
- `test_r17_archives.py`（13 项）：合格档案通过；仅开发档案或仅测试档案出现非目标变化、目标在一部分未变、缺键回退、部分行变化、非有限值、形状或维度不符、基础键缺失、多种子中一个有问题——均判为问题（退出码 2）；模糊部分行不变、未用状态缺失不判为问题。

## AutoDL

```
cd /root/autodl-tmp/IRRA-p4 && git fetch origin p4/r17-const && git checkout -B p4/r17-const origin/p4/r17-const && git log --oneline -1 && (nohup bash IRRA-main/tools/p4/transfer/p4_r17.sh > /root/autodl-tmp/p4_r17_driver.out 2>&1 &)
```

状态：`cat /root/autodl-tmp/p4_r17_out/STATUS`；结果：`cat /root/autodl-tmp/p4_r17_out/DIGEST17.txt`；打包：`/root/autodl-tmp/p4_r17_results.tar.gz`（不含逐查询 pkl 与 `_stale`）。需要正常模式开机（无卡模式内存不足）；只用 CPU；默认跑完不关机（`AUTO_SHUTDOWN=1` 可自动关机）。
