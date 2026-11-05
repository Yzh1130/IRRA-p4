# R19 追加验证驱动（post-hoc）

本目录只新增工具；r15_test.py、r17_const.py、r18_xcam.py、旧判读脚本和 run_frozen.py 保持冻结。默认不运行任何阶段。正式运行时输出统一在 `/root/autodl-tmp/p4_r19_out/`；每阶段需要显式指定 STAGE。

## 启动

```bash
cd /root/autodl-tmp/IRRA-p4/IRRA-main
STAGE=S0 bash tools/p4/r19/p4_r19.sh
# S0 通过后由操作者依次单独启动；S3 需要 GPU。
STAGE=S1 bash tools/p4/r19/p4_r19.sh
STAGE=S2 bash tools/p4/r19/p4_r19.sh
STAGE=S3 bash tools/p4/r19/p4_r19.sh
STAGE=S4 bash tools/p4/r19/p4_r19.sh
```

默认激活 `/root/miniconda3` 下的 `rgbt5090`；已有合适环境时 `SKIP_CONDA=1 PYTHON=/absolute/path/python`。`P4_ROOT`、`P4_REPO` 仅用于在独立合成目录演练；正式运行使用上面的固定路径。驱动不开关机。

| STAGE | 训练种子 × 拟合种子单元 | 输出（均相对 p4_r19_out） | 摘要 / 状态 |
|---|---|---|---|
| S0 | RGBT、RGBNT 各 1 个 s101_f7001；另跑旧相机权重无身份输入测试 | `s0/{rgbt,rgbnt}/s101_f7001/`、`s0/no_identity_test/` | `DIGEST19_S0.txt`、`STATUS_S0` |
| S1 | RGBT 9、RGBNT 全量 9、去重子集 9，共 27 个 | `s1/{rgbt,rgbnt,rgbnt_subset}/s*_f*/`、`reports/S1/` | `DIGEST19_S1.txt`、`STATUS_S1` |
| S2 | RGBNT 全量 9、去重 9、嵌套去重 9、k3 9，共 36 个；另 3 个相机切片单位 | `s2/{rgbnt,rgbnt_subset,rgbnt_nested,rgbnt_sparse}/s*_f*/`、`reports/S2/` | `DIGEST19_S2.txt`、`STATUS_S2` |
| S3 | 两数据集各 1 次两轮 smoke，再各 3 个正式训练种子，共 8 次训练 | `mlm1/{configs,runtime_configs,smoke,runs,health,exports,packed,feats,train_ir,subset}/` | `DIGEST19_S3.txt`、`STATUS_S3` |
| S4 | MLM 单计数的 RGBT 9、RGBNT 9、去重 9，共 27 个 | `mlm1/eval/{rgbt,rgbnt,rgbnt_subset}/s*_f*/`、`reports/S4/` | `DIGEST19_S4.txt`、`STATUS_S4` |

S1 每单元为 13 种混合 × 5 个分配种子。S2 的全量与去重部分为 15 × 5（增加 Mcam1/Mcam2），嵌套和 k3 为 13 × 5。S0 仅 M0/M4、分配种子 7031。所有评价都显式传 `--perq_ext`；曲线的逐查询记录按 r19_variants 的规则不增加。S4 使用 r15 完整 METHODS 加 CtxOnly、SRC2f+kap、SRC2f+pI、QE+Const、QE+NoGate，两数据集均如此；v3/constrained 重新选择，不传 fixed_selection、不开 const_curve。Const 用旧 γ 网格在 MLM1 开发集上按原主规则选择；仅 S4 显式传 `--nogate_rmax`，将 NoGate 设为 γ=r_max（默认及 S1/S2 不改变旧 NoGate 诊断口径）。V5 与表4式分解标记 post-hoc。

## 输入路径（正式服务器）

下列路径是运行前必须具备的输入，不由驱动下载。`s` 展开为 101、102、103，`f` 展开为 7001、7002、7003；包含通配符的路径必须恰好命中一个文件，否则失败即停。

RGBT-PEDES：

- `/root/autodl-tmp/r13_inputs/base_ref_s101.npz`
- `/root/autodl-tmp/r13_inputs/base_ref_s102.npz`
- `/root/autodl-tmp/p4_r15_out/feats/base_ref_s103.npz`
- `/root/autodl-tmp/p4_r15_out/feats/test_base_s{s}.npz`
- `/root/autodl-tmp/p4_r13_out/train_ir/base_ref_s101_train_ir.npz`
- `/root/autodl-tmp/p4_r13_out/train_ir/base_ref_s102_train_ir.npz`
- `/root/autodl-tmp/p4_r15_out/train_ir/base_ref_s103_train_ir.npz`
- `/root/autodl-tmp/p4_r15_out/test/constrained/s{s}_f{f}/*_selection.json`
- S0 额外回归：`/root/autodl-tmp/p4_r17_out/const/s101_f7001/*_cells.csv`（排除 diag_cells）。

RGBNT201 全量、去重：

- `/root/autodl-tmp/p4_r18_out/feats/r201_dev_s{s}.npz`
- `/root/autodl-tmp/p4_r18_out/feats/r201_test_s{s}.npz`
- `/root/autodl-tmp/p4_r18_out/feats/r201_testnov_s{s}.npz`
- `/root/autodl-tmp/p4_r18_out/train_ir/r201_s{s}_train_ir.npz`
- `/root/autodl-tmp/p4_r18_out/test/constrained/s{s}_f{f}/*_selection.json`
- `/root/autodl-tmp/p4_r18_out/subset/test/constrained/s{s}_f{f}/*_selection.json`
- S2 相机：`/root/autodl-tmp/p4_r18_out/feats/cams_s{s}.npz`，必须有严格对齐的 `dev_cams`、`test_cams`。子集相机文件由驱动按全量原始身份及身份内序号切片，仅写到 `p4_r19_out/s2/cameras/s{s}/subset.npz`。
- S0 额外回归：`/root/autodl-tmp/p4_r18_out/const/s101_f7001/*_cells.csv`（排除 diag_cells）。

S3 另需数据与配方、模型预训练缓存：

- RGBT 配方：`/root/autodl-tmp/p4_preexp/configs/p4/base_clean.yaml`，保持其 root_dir/dev_split_file 值。
- RGBNT 配方来源与 R18 L1 相同：读取 `/root/autodl-tmp/p4_preexp/runs/base_clean_s102/RGBT-PEDES/*_base_clean_s102/configs.yaml` 的 `config_file` 所指原始配方；覆盖为 `dataset_name=RGBNT201-PEDES`、`root_dir=/root/autodl-tmp/IRRA-p4/IRRA-main/datasets/TextReID`、`dev_split_file=/root/autodl-tmp/IRRA-p4/IRRA-main/splits/rgbnt201_dev30.json`。
- `/root/autodl-tmp/IRRA-p4/IRRA-main/splits/rgbnt201_overlap.json`（去重导出）。
- 数据集实际标注和图像、CLIP 预训练权重必须预先按既有训练入口的路径准备。驱动不下载数据或权重。

MLM1 配置由 training.py 在 `p4_r19_out/mlm1/configs/` 生成并记录来源、生成后哈希和逐项白名单差异；仓库不提交整份服务器 YAML。smoke 仅把 num_epoch 改为 2，使用单独 name/output_dir，不参加评价。两次 smoke 健康检查通过后才运行各自的正式训练。正式健康检查复用 workflow.health：先对通过来源白名单校验的配方调用 training.py 的 `runtime-health-base`（只解析既有参数，不训练、不构造 loader/model），在 `runtime_configs/` 保存完整默认值、CLI config_file、单进程元数据的预期配置，再以该配置为 base。这样既不扩大配方差异白名单，也不把既有 parser 补默认值或 config_file 来源字段误报为配方变化。

## 续跑与只读保护

每个单位输出目录的 `manifest.json` 包含完整命令、方法/参数/随机种子、Git 提交、代码 SHA-256、所有输入文件 SHA-256/字节数及线程设置。JSON 编码固定为 UTF-8/LF、排序键；`.done` 存其 SHA-256。只有本次预期 manifest 字节一致、完成标记一致、已有产物哈希也一致时复用。否则整个旧单位目录移到 `p4_r19_out/_stale/`，不删除。

开跑前把 `/root/autodl-tmp/p4_r15_out`、`p4_r17_out`、`p4_r18_out` 的所有文件路径/大小/哈希记录到 `p4_r19_out/checks/protected.json`；每条子命令后、复用后及结束时重算。任何新增、缺失或变化立即停止。旧目录只读，所有缓存、临时目录、日志、manifest、摘要均在新 OUT。代码用 `-B`/PYTHONDONTWRITEBYTECODE，不往仓库写 pyc。

S0 实际调用历史 `test_r18_eval.CameraTests`：其 restricted_weights 的输入只有 J/Fi/CF/X，没有 PID 或 SF，覆盖“推理权重不读身份”的既有路径；没有把身份交集检查冒称成推理测试。R19 自身另有无 SF 的 refit 权重合成测试。

## 本机 S0 合成演练

```python
# 在 IRRA-main 下，用已有 CPU 环境，仅生成合成档案。
import sys
from pathlib import Path
sys.path.insert(0, str(Path('tests/p4').resolve()))
from test_r19_driver import make_s0_fixture
spec = make_s0_fixture(Path('/absolute/independent/synthetic_root'))
print(spec)
```

```bash
P4_ROOT=/absolute/independent/synthetic_root P4_REPO=/absolute/repository \
R19_SYNTHETIC=1 R19_SPEC_JSON=/absolute/independent/synthetic_root/s0_spec.json \
SKIP_CONDA=1 PYTHON=/absolute/cpu/python STAGE=S0 bash tools/p4/r19/p4_r19.sh
```

生成器只使用 test_r17 的合成夹具，先由冻结 r18_xcam.py 写两份参考，再由新脚本回归；不会读真实特征。第二次完全相同的启动应复用；改变 spec 中参数或输入后，manifest 单元应移入 `_stale` 后重算。旧三目录任何改动会先被 guard 拒绝，不能作为 stale 演练的修改对象。

## 耗时与体积依据（估计，未运行 R19 真实数据）

只读参考本机归档 `p4_archive/20261030/records` 中 9 单元的 run_record.json：R17 RGBT const 每单元 1415.4–1423.1 秒（中位 1419.9），约 183.7 MB；R18 RGBNT const 276.4–280.8 秒（中位 277.1），约 62.1 MB；R18 constrained 2108.9–2817.5 秒（中位 2772.4），约 64.8 MB；R18 xcam 85.4–89.4 秒（中位 87.8），约 23.6 MB。这些是旧方法/旧硬件历史值，不是 R19 测量。

旧正式训练记录：RGBT 三种子约 1073–1198 秒，RGBNT 三种子约 857–866 秒。R19 增加曲线、稀疏/嵌套拟合及 refit，而且逐步全目录哈希检查有 I/O 成本；按顺序单线程启动，预算建议 S0 10–40 分钟，S1 6–24 小时，S2 4–20 小时，S3（GPU）2–6 小时，S4 12–48 小时。为保持旧硬件配置与本轮结果区分，实际运行以新 run_record 的 wall_seconds 为准。

结果盘预算估计：S1 4–10 GB、S2 2–8 GB、S3（包含权重与导出）5–20 GB、S4 5–14 GB（已计两数据集新增 Const/NoGate 的 perq_ext）；每次 `_stale` 保留会额外占同量空间。此为依据旧产物与方法数量的规划范围，未测量新正式输出，不能当作完成承诺。S1/S2/S4 为 CPU 路径；S3 必须有 GPU，启动前人工确认磁盘与预训练缓存。


## R19 补充口径与资源明细

V2 固定按 A 组、相对 QE-DBA、逐基础种子计算：gain=ΔmAP，loss=max(0,−ΔR10)。三种子均须满足选中 λ 的 gain≥0.7×主方法 gain 且 loss≤0.5×主方法 loss。主方法 loss=0 时，λ 的 loss 也必须为0；输出明确标注。lam2 同口径独立报告，不参与 V2。

V4a 比较三种子平均增益的绝对差，|nested−full|≤0.5；V4b 仅在 full gain>0 时，以 sparse gain<0.5×full gain 标记依赖重复观测；full≤0 时只报告数值和区间，不作该标记。

S4 追加 QE+Const、QE+NoGate（两个数据集都运行），不开 const_curve。Const γ 仍使用原 8 点网格与主选择规则，在新 MLM1 开发档案上选择。为同时保持默认旧路径和本轮显式 NoGate 口径，仅 S4 传 `--nogate_rmax`：诊断权重使用 r_max×A；不传时仍是冻结脚本的 γ=1 后在 post_update 内截断。S4 输出前端−B★、Const−前端、主方法−Const 的分解，V5 的 C1–C5 / E1–E4 与旧判定分开保存，全部 post-hoc。

旧逐查询扩展档案实测（仅读取本机归档文件大小）：RGBT const 的 9 个单元均为 75,955,631 字节；RGBNT const 的 9 个单元均为 25,636,568 字节。按逐查询数组字段、方法数和单元数估算，新 `perq_ext.pkl` 每单元约：

| 阶段 | RGBT | RGBNT 全量 | RGBNT 去重 / 附加集 |
|---|---|---|---|
| S1 | 120–140 MB | 40–50 MB | 35–45 MB |
| S2 相机 refit | 不运行 | 25–35 MB | 20–30 MB |
| S2 嵌套 / k3 | 不运行 | k3 单元同时含 test+k3，约45–60 MB | 嵌套约20–30 MB |
| S4 | 230–270 MB | 75–95 MB | 65–85 MB |

这些是扩展逐查询文件估计，不含另存的旧 perq.pkl、CSV、权重和 `_stale`。λ/γ曲线不进入 perq_ext；TOP10_METHODS 仍仅 QE-DBA、QE+M、QE+Const，新方法不保存前10名。

按旧 R17/R18 记录规划每个评估单元（不是 R19 实测）：S1 RGBT 25–60分钟、RGBNT全量/去重 8–25分钟；S2 约5–25分钟（k3 单元另含完整 test，按两套评价预算）；S4 RGBT 40–120分钟、RGBNT 35–90分钟。代码/输入哈希守护有额外磁盘 I/O，0.5核实例和不同内存配置可能超出上述范围。磁盘按前表合计预留15–50GB，另为每次 `_stale` 保留增加同量空间；实际以运行记录为准。本机只运行合成测试及合成 S0，未运行任何真实评估或训练。

ViT-L/14 仅核查：配置入口 utils/options.py:108，模型 URL 在 model/clip_model.py:32；已有构建支持，但 stride 需另行明确、显存未测，本轮未改相关配置。按目录名与 README 搜索未找到 DCAlign/MGANet 或 DIKDNet 的独立代码/命名权重，建议6暂缓。


S3 启动前只读核对基准配方的 pretrain_choice 对应权重：已知 CLIP 名称按原代码的 `~/.cache/clip/<文件名>` 和 URL 钉住的 SHA-256 检查，本地权重路径则记录其哈希。缺失或不符立即停止，不下载、不替换缓存；合格权重也进入训练单元的输入 manifest。该检查不切换骨干。

服务器输入尚未实地核实：已归档的 RGBT 原 base_clean.yaml 的 root_dir 为 `/root/autodl-tmp/MRAF_DATA_ROOT`，驱动按要求原样继承；不能仅因 s102 的保存配置使用 TextReID 就覆盖原配方。启动 S3 前该基准目录及其中的 RGBT-PEDES 标注/图像必须存在。本机没有连接 AutoDL 验证此路径。
