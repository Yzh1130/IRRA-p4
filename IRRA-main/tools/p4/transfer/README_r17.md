# 第十七轮：只读核查 + 常数迁移对照 + 效率与开发集敏感性

规格见项目文档《论文4_第十七轮_独立裁决与收尾计划》§3（跑前写定）。第十五轮的 C1–C5 判定不变；`p4_r15_out` 只读；冻结脚本 `r15_test.py`、`judge_test.py` 不改。

## 脚本

| 文件 | 作用 | 是否计算新分数 |
|---|---|---|
| r17_check_archives.py | 列出三个 base 的开发/测试档案键名、合并后的键集合（测试模式取交集）、每个状态实际读到的数组是否与干净数组不同，以及 M1g 每个分配种子的状态计数 | 否 |
| r17_reagg_r15.py | 用第十五轮保存的 cells.csv 与逐查询记录，先逐位置核对对齐（每个位置的查询均值必须等于对应分配种子的单元值），再按显式键汇总逐混合 × 种子的 mAP/R1/R5/R10/mINP 与配对区间 | 否 |
| r17_const.py | `r15_test.py` 加：QE+Const（开发集按冻结主规则选一个全局常数 g）、QE+NoGate（g = 1，经 r_max 截断）、`--fixed_selection`（沿用第十五轮前端配置，不重新择优）、显式键逐查询记录、计时、开发集敏感性开关 | 是：同一测试集上的事后补充对照（非盲） |
| r17_judge.py | 结构核对、与第十五轮单元逐格复现核对、所选 g 与开发表、绝对指标、配对差（AP/R1/R5/R10/mINP，身份簇 bootstrap）、常数全曲线（只报告）、局部性、harm/help、top-5 组成、计时、开发集敏感性 | 否 |
| p4_r17.sh | AutoDL 一次性驱动：断点续跑、失败即停、结果打包、核对 `p4_r15_out` 前后哈希不变 | — |

常数网格在任何第十七轮测试运行之前锁定为 `0 0.05 0.1 0.2 0.25 0.5 0.75 1`；并列取网格中较小的值（与 `pooled_select` 相同的比较方式）；无可行值时取最差类 ΔR1 最大者（原回退规则）。

## 本机验收（无真实数据）

```
python -B IRRA-main/tests/p4/test_r17.py
```

六项检查：不带新选项、用第十五轮方法列表时与 `r15_test.py` 输出逐位相同；带 `--fixed_selection` 时共有方法逐位相同；QE+Const@0 = QE-DBA、QE+Const@1 = QE+NoGate、QE+Const = 所选 g 的曲线点；前端 top-1 不是仅红外项时 top-1 不变；逐查询记录与单元一致；三个报告脚本正常退出。

## AutoDL

```
cd /root/autodl-tmp/IRRA-p4 && git fetch origin p4/r17-const && git checkout -B p4/r17-const origin/p4/r17-const && git status --short | head
nohup bash IRRA-main/tools/p4/transfer/p4_r17.sh > /root/autodl-tmp/p4_r17_driver.out 2>&1 &
```

结果：`cat /root/autodl-tmp/p4_r17_out/DIGEST17.txt`；打包：`/root/autodl-tmp/p4_r17_results.tar.gz`（不含逐查询 pkl）。需要正常模式开机（无卡模式内存不足）；只用 CPU；默认跑完不关机（`export AUTO_SHUTDOWN=1` 可自动关机）。
