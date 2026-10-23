# 第十三轮离线证据迁移：1a 开发复现与 1b 导出

## 来源与文件边界

`r13_protocol.py` 是第十三轮 `r13_scripts/r13_simcf.py` 的逐字复制，仅添加两行来源注释。原文件 SHA256：`0af9a18c9c1480af0d58b9c31e39b247dc50439d60673d240aef2a844e58ca00`；当前脚本 SHA256：`b657ff06ea0172ca2c9cf473e7816dbdbb012ae7c85b6525ab17c887f55e97b8`。去掉头两行后的字节必须与原文件完全一致。

`judge_v2.py`、`paired_pairs.py`、`verify_1c_assignments.py`、`cmp_tables.py` 均按字节复制；judge 锚点为 `2b65c728216894cd3a13f804408943c2e8810194fe5a871918d2a140883a3e62`。输入须通过 `inputs_r13.sha256` 的八项校验。依赖文件仅列出 scikit-learn、pandas、pytest；numpy、torch 使用环境中已有版本。

本次只评估开发结果；不运行 `--stage1c_final`，不评估冻结的 1c 混合和种子。`verify_1c_assignments.py` 仅用 AST 取得分配函数并生成分配表，不计算检索指标。`judge_v2.py` 在本轮仅原样入库。四口径为 base s101/s102、A1 s101、Pull s102，不访问官方测试集。

## 公式与协议

含红外项 H、红外-only 项 J、完整项 F；映射由干净红外特征拟合，链接矩阵 C 使用冻结 base 在图库当前状态下的红外特征经收缩 WCCN 后的余弦，红外噪声/模糊状态不替换为干净特征。最近邻排除自身，图库统计为

```
n(j) = argmax_{k∈H\{j}} C_jk
h_I = mean_{j∈J} 1[n(j)∈J]
p_I = (|J|−1)/(|H|−1)
rho_hat = h_I − p_I
kappa_hat = rho_hat / max(1−p_I, 0.05)
```

主规则 A-2f-ctxn 用标准化的 f1F、f1F−f1I、kappa_hat、p_I 拟合 L2 逻辑回归来源头，得到迁移总量 r_j；完整项权重 a_jk=softmax_k(kappa_a·C_jk)。更新仅作用于红外-only 列：

```
P'_qj = (1−r_j) P_qj + r_j sum_{k∈F} a_jk P_qk
```

其他列不变。r=0 时为原分数；r<1 且原首位不是红外-only 时保留首位。r_max=0.999999 限制总迁移量，tie_break=bstar 在并列时保留原 B* 次序。`lda_map` 名称保持参考实现：它实际是按身份计算类内散度、收缩并求逆平方根的 WCCN，先中心化再映射归一化，不应表述为 Fisher LDA。

默认 2 折在身份隔离的拟合折/评估折之间工作；链接统计使用拟合折内部双向交叉拟合。3 折同时给出 small 和 large；full 保留拟合折图库作为干扰项，只计另一折查询，是压力测试。来源头、Platt 校准和标量选择均限于拟合身份；不更换 sklearn 优化器。

参考程序交叉拟合时仍会在冻结特征上拟合小型 torch 链接头，以便为 GP-same-head 选择参数；该选择不进入本轮比较的方法，平台间选择记录允许不同。`train_lda` 模式从单独训练身份特征拟合 WCCN，不拟合该链接头，并拒绝训练/开发身份交集。

## 运行方法

从 IRRA-main 运行。每个 `--out` 必须事先创建为全新的目录，脚本不会自动创建。建议进程环境 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`；本地 CPU 环境另设 `CUDA_VISIBLE_DEVICES` 为空。`-I` 下应确认当前解释器能直接导入 numpy、scikit-learn、torch。

```powershell
$env:R13_INPUTS = '<R13>/inputs'
python -B -m pytest tests/p4/test_r13_protocol.py tests/p4/test_export_train_ir_feats.py -v
python -I tools/p4/transfer/verify_1c_assignments.py --script tools/p4/transfer/r13_protocol.py --inputs_dir $env:R13_INPUTS
New-Item -ItemType Directory '<new-output>'
python -I tools/p4/transfer/r13_protocol.py --b12 . --feats $env:R13_INPUTS --assign "$env:R13_INPUTS/assignments_s102.csv" --out '<new-output>' --base base_ref_s102 --adapter acd_Pull_s102 --tag pull_s102 --mixes M1 M3 M4 M4c25 M4c50 M4c75 M4id --skip_joint --methods 'B*' CAET CAET-g A-SRC-2f A-2f-ctxn A-2f-EM A-ctxonly A-ctxonlyN CAET-recal CAET-recalN CAET-hardN Oracle-A
python -I tools/p4/transfer/cmp_tables.py '<new-output>' '<R13>/results/out_r1' pull_s102
```

其余主复现只更换以下参数，方法与混合保持相同：

| tag | base | adapter | assignments | 参考目录 |
|---|---|---|---|---|
| pull_s102 | base_ref_s102 | acd_Pull_s102 | assignments_s102.csv | out_r1 |
| a1_s101 | base_ref_s101 | acd_A1_s101 | assignments_s101.csv | out_r1 |
| base_s102 | base_ref_s102 | base_ref_s102 | assignments_s102.csv | out_r2 |
| base_s101 | base_ref_s101 | base_ref_s101 | assignments_s101.csv | out_r2 |

小筛查使用 Pull s102，增加 `--split_seeds 2026 --assign_seeds 2026 2027 2028`，方法仅为 `B* CAET A-SRC-2f A-2f-ctxn A-ctxonly A-ctxonlyN CAET-recalN CAET-hardN`，与 out_r0 比较。数值保护分别在 Pull s102、base s102 主命令增加 `--r_max 0.999999 --tie_break bstar`，与 out_safeguard、out_safeguard_base 比较。

验收阈值：summary/cells 最大绝对差≤1e−6 为通过；(1e−6,0.01] 报告平台数值差；>0.01 停下排查，不修改协议凑数。同时检查形状、文本列、NaN 掩码。run_record 保持参考的 `r12_<tag>_run_record.json` 命名，记录脚本/输入 SHA256、参数、版本、线程、耗时。仅来源注释使脚本哈希与锚点不同，正文未变。

## 1b：训练身份红外特征导出

该命令只供后续 GPU 推理使用，本机仅测不含真实模型的部分。训练身份为 dev_split 的 1076 个 train_reduced_ids，使用无训练增强的评估变换，每张 RGB/IR 图像对一项；完整开发集 IR 特征须先与对应 base NPZ 的 clean/f_ir、clean/img_pids 按顺序核验（最大绝对差≤1e−5），失败不写输出。

```powershell
python -B tools/p4/export_train_ir_feats.py --checkpoint '<base-run>/last.pth' --config_file '<base-run>/configs.yaml' --root_dir '<data-root>' --dev_split_file '<split.json>' --output '<new-train-ir.npz>' --batch_size 64 --num_workers 4 --device cuda --verify_dev_npz '<R13>/inputs/base_ref_s102.npz'
# 冒烟时可增加 --limit 16；开发集一致性校验仍使用完整开发图库。
```

输出键为 f_ir、f_rgb（FP32）、raw_pids（原始整数身份）、file_paths、provenance。provenance 保留检查点哈希、身份划分哈希、git HEAD 和工作区状态。已有输出拒绝覆盖。不会启动训练或重新编码另一遍以获取两路特征。

## 本机复现验收表

2026-10-23 联合 CPU 单测：`31 passed, 15 subtests passed in 147.99s`，无跳过。`R13_INPUTS` 未设置时按要求直接报错。

| 命令 | 进程耗时 s | run_record wall_seconds | summary 最大差 | cells 最大差 | 结论 |
|---|---:|---:|---:|---:|---|
| pull_s102 | 189.819 | 168.7 | 0.000206274563913 | 0.0 | 允许的平台数值差（AUC）；性能与CI精确相同 |
| a1_s101 | 203.850 | 175.3 | 0.000346721066697 | 0.0 | 允许的平台数值差（AUC）；性能与CI精确相同 |
| base_s102 | 190.480 | 171.5 | 0.000206274563913 | 0.0 | 允许的平台数值差（AUC）；性能与CI精确相同 |
| base_s101 | 184.064 | 164.0 | 0.000346721066697 | 0.0 | 允许的平台数值差（AUC）；性能与CI精确相同 |
| screen | 66.337 | 45.5 | 0.000561167227834 | 0.0 | 允许的平台数值差（AUC）；性能与CI精确相同 |
| safe_pull | 214.388 | 189.6 | 0.000206274563913 | 0.0 | 允许的平台数值差（AUC）；性能与CI精确相同 |
| safe_base | 207.319 | 183.3 | 0.000206274563913 | 0.0 | 允许的平台数值差（AUC）；性能与CI精确相同 |

所有表的形状、文本列、NaN 掩码一致；所有 cells、neardup 数值差为 0，summary 的 mAP/dmAP/dR1 与四个 bootstrap CI 列精确一致。显著数值差仅为 AUC：summary 最大 0.000561167228，diag_cells 最大 0.006097560976，均≤0.01。未修改原协议以拟合数字。

A4 表格保护检查：Pull 和 base s102 的 summary/neardup 均只 3 行 Oracle-A 变化，cells 均只 179 行 Oracle-A 变化，diag_cells 完全不变。selection 仅 THR 条目变化；**提示中“仅 τ 变化”的字面条件不成立**：THR 存储的是 `(tau, dl)`，Pull 的9条变化中 tau变7条、dl变7条，base的10条变化中tau变9条、dl变5条。原始参考包自身也有完全相同的变化计数。源码 `tau, dl = chosen['THR']` 和联合网格见 r13_protocol.py:629、770、776。这里不能宣称所有字面验收项均通过；保留参考定义并报告差异。

本机仅测试导出脚本的合成数据/无真实模型部分；GPU导出和真实开发特征一致性校验留待1b执行。本次未自动推送。

原始输出、完整命令、run_record脚本/输入哈希与比较 JSON 保存在仓库外 `E:/论文4/2过程文档/r13_1a_cpu_20261023_verified/`。不把输入 NPZ 或结果包提交到 Git。

## 1c 规格 v2：以下逐字摘录第十三轮报告 §6

## 6. 1c 规格 v2（v1 保留在第十二轮报告 §7）

### 6.1 相对 v1 的修订及原因

| 项目 | v1 | v2 | 原因 |
|---|---|---|---|
| 主口径 | Pull s102、A1 s101 | base s102、base s101；Pull、A1 降为次要 | §4.4：规则不依赖适配器 |
| 图库 | 239 半图库；478 全图库只对另一折查询计分 | 239 半图库（严格）；3 折约 319 项（严格）；478 全图库（压力测试） | 意见 4.3。注意：v1 要求 478 上也满足 S1–S3（第十二轮 §7 的①）；v2 不再把 478 上的 S1–S3 作为门槛，只报告 |
| 方法 | B★、CAET、主规则、EM、Oracle | 增加 CAET-g、A-SRC-2f、N1、N2、N3 | 意见 4.6 |
| S2 | 比 CAET 至少好 +0.5（固定值） | CAET 损害 ≥0.3 时，改善至少为其损害的 50%；v1 版本同时报告 | base 口径下 CAET 的损害本身较小（M4id 为 −0.63 / −0.68）。M5f 的红外-only 项少，固定 +0.5 可能根本无法达到，会把"判据不合理"误判成"方法失败"。这推翻了第十二轮跑前写的"M5f 上 +0.5 的门槛仍有意义"：那一判断依据的是适配器口径下确认 1 的 CAET 损害（约 −0.9） |
| S3 | 每个部分来源混合都 ≥ CAET，无容差 | 部分来源混合上（主规则 − CAET）的平均 ≥ −0.1；v1 版本同时报告 | base 开发集上的余量只有 +0.08～+0.28，单个配对差的标准误为 0.09～0.17。8 次逐个零容差检查，即使主规则真的略好，也很可能因噪声"失败" |
| 规模判据 | 478 对 239 | 3 折内 319 对 159（同一组拟合模型）；478 照常报告 | 消除"拟合身份数不同"的混杂 |
| 泄漏 | 478 上 M4id ≤ 0.08 | 319（严格）与 478（压力）上都 ≤ 0.08 | 意见 4.3 |
| 必要性判据 | 无 | 新增 Na–Nd（下表） | 意见 4.6 |
| 数值保护 | 无 | 迁移上限 1−10⁻⁶，并列保持 B★ 原序 | 意见 2.5；§4.6 在 Pull、base s102、base s101 上验证：主规则与 N1–N3 结果不变 |
| 混合、种子 | 冻结 | 不变（9 个混合；split 5031–5035，assign 5031–5033） | — |

S2、S3 放宽前，我已经看过 base 开发集上的数字。v2 是在 1c 运行之前，按上述原因写定的；v1 的版本照常报告，供你判断。

### 6.2 口径、方法与混合

- **链接映射**：在 1076 个训练身份的 base 干净红外特征上拟合收缩 WCCN（--link_map train_lda），两个 base 种子各一个。所有开发项对映射都是样本外，不再需要交叉拟合。
- **拟合**：B★ 的 Platt、来源头、κ_a、CAET 的 κ/θ，以及 CAET-g、N3 的标量，都只在开发集拟合折上拟合。
- **图库**：half（2 折，239，严格）、three（3 折：small 约 159、large 约 319，严格）、full（2 折，478，压力测试）。
- **方法**：B★、CAET、CAET-g、A-SRC-2f、A-2f-EM、N1 A-ctxonlyN、N2 CAET-recalN、N3 CAET-hardN、主规则 A-2f-ctxn、Oracle-A。
- **混合（冻结）**：
  - 有来源：M4、M6c、M1g；
  - 部分来源：M4c60、M9；
  - 无来源：M5e、M5f；
  - 泄漏：M4id；
  - 边界探针：M10。

### 6.3 判据（点估计；区间另行报告，不进入判定）【待检】

**门槛**：两个主口径都要满足，分别在 half 和 large 上检查。
- S1：每个有来源混合，ΔmAP(主) ≥ ΔmAP(CAET) − 0.3。
- S2：每个无来源混合，ΔmAP(主) ≥ −0.3 且 ΔR1(主) ≥ −0.3；若 ΔmAP(CAET) ≤ −0.3，还要求 ΔmAP(主) − ΔmAP(CAET) ≥ 0.5·|ΔmAP(CAET)|。
- S3：部分来源混合上，ΔmAP(主) − ΔmAP(CAET) 的平均 ≥ −0.1。
- L：M4id 上主规则的泄漏，在 large 与 full 上都 ≤ 0.08。
- R：3 折内 large 对 small，无来源优势比 ≥ 0.7，有来源收益比 ≥ 0.8。

**必要性**：两个主口径都要满足，只在 half 上检查。
- Na：无来源混合上，主 − A-SRC-2f 的平均 ≥ +0.2。
- Nb：有来源与部分来源混合上，主 − N1 的平均 ≥ +0.3。
- Nc：同上范围，主 − N3 的平均 ≥ +0.3。
- Nd：全部非探针混合上，N2 − 主 的平均 ≤ +0.1。
- Ne（只报告，不判定）：主 − EM，分别看多状态混合 M1g 和全部混合。

**决策**（judge_v2.py 自动给出）：
- 任何一项必需检查缺失（例如某次运行失败），或比值无定义，都按不满足处理。
- **全部满足** → 进入第二阶段：与标准公开重排和已发表近邻（k-reciprocal、DBA/α-QE、EM 先验更新、特征层邻居插补）正式比较，核实第二数据集，补第三个 base 种子，冻结后在官方测试集上测一次。
- **门槛满足、必要性部分不满足** → 按不满足的那一条收窄主张：
  - Na 不满足：图库项不必要，方法退化为已有的逐项门控，停止作为主方法（即 C）；
  - Nd 不满足：只能写"局部证据形式可替换"，不得事后改用 N2 作主规则。
- **门槛不满足** → 降级为"只在开发协议下成立"，停止作为第四篇主方法（即 C）。不再追加修正轮。
- **次要口径**（Pull、A1）只报告；不满足时，只收窄"可与更强的红外-only 通路组合"这一说法。
- **M10** 只报告；主规则低于 CAET 或 A-SRC-2f 时，论文中限定适用范围。

**开发集上的干跑**：仅用于检验判定脚本能跑通；开发数据不算证据，判定结果也不是证据。干跑只能算 half 上的 S1–S3 和 Na–Nd（开发运行没有 3 折和全图库，L、R 算不了），这些在 4 个口径上都满足。base 两个口径的 Na 为 +0.27 / +0.28，离 +0.2 的门槛余量很小。

### 6.4 跑之前写明的风险【待检】

- **S2 的 R1 条件**：第十二轮 3 折 large（319 项）上，Pull 的 M4id ΔR1 已到 −0.39；base 口径在 319 上尚无数据。
- **S3 的余量**：base 开发集上（主 − CAET）的余量只有 +0.08～+0.28。
- **Na 的余量**：base 开发集上为 +0.27～+0.28，离门槛 +0.2 不远。
- **训练身份映射**：base 见过这些身份，其类内散度偏小；链接准确率 a 和 κ̂ 的尺度可能变化（1c 运行输出的 wccn_diag、maplog、ctxlog 会给出）。
- **478 的泄漏**：可能接近 0.08。第十二轮 3 折 large 已是 0.062 / 0.060（Pull / A1），若按每翻倍 +47% 外推，478 上会接近这一上限。
- **M10**：外观相近的不同身份会一起缺失，κ̂ 会把这部分当成"身份相关"，规则可能拒绝过多。

---

