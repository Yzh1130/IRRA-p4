# DCAlign 统一退化评测集成说明

本目录用于把 DCAlign（服务器仓库 commit `1ca105f`）纳入本文的统一退化评测协议。DCAlign 源码仍保留在独立目录中；这里的脚本和补丁不会把该仓库复制进 IRRA，也不会修改它原有的 `test.py` 评测入口。

## 应用动态融合补丁

服务器 DCAlign 仓库已有 7 个文件的未提交环境适配改动：`datasets/bases.py`、`model/build.py`、`processor/processor.py`、`test.py`、`train.py`、`utils/metrics.py`、`utils/options.py`。这些改动必须保留，补丁必须直接应用在这个已修改的工作树上，不能先 reset、checkout 或 stash 后在干净版本上应用。

在服务器上依次执行：

```bash
cd /root/autodl-tmp/code/RGBT-PEDE
git stash list # 确认现有 7 个文件的环境适配改动状态
git apply --check /path/to/enable_dynamic_fusion.patch
git apply /path/to/enable_dynamic_fusion.patch
grep -n dcalign_dynamic_fusion model/build.py utils/options.py
```

`git apply --check` 不通过时必须停止，报告冲突文件、hunk 和行号，再基于服务器当前文件人工合并同等的两处改动。禁止使用 `git apply --reject`、`git apply --3way`、`--ignore-whitespace` 或其他强制应用方式绕过冲突，也禁止为了套用补丁而覆盖、丢弃或还原既有环境适配改动。应用前后都应保存 `git diff -- model/build.py utils/options.py` 作为溯源记录。

补丁只改变两点：

1. `model/build.py` 在 `DCAlign.fuse` 中增加默认关闭的动态融合分支；
2. `utils/options.py` 增加布尔命令行参数 `--dcalign_dynamic_fusion`，`action="store_true"`、默认值为 `False`。

省略该参数时，`getattr(self.args, 'dcalign_dynamic_fusion', False)` 为假，执行的仍是原来的 `rgb + t` 两行代码。因此 DC-add 的默认行为与补丁前逐位一致，旧 `configs.yaml` 中没有新字段时也会安全回落到加法融合。DC-dyn 训练时必须显式传入 `--dcalign_dynamic_fusion`。

## 动态融合模块为何原本没有参与计算

`DCAlign.__init__` 执行：

```python
self.fuse = FusionModule(self.embed_dim)
```

因为右值是 `nn.Module`，`nn.Module.__setattr__` 会把它登记到 `self._modules['fuse']`，而不是把一个普通的 `fuse` 值放入实例 `__dict__`。类体随后又定义了同名方法：

```python
def fuse(self, rgb, t):
    ...
```

Python 的常规属性查找发现实例 `__dict__` 中没有 `fuse` 后，会命中类上的这个方法（非数据描述符）。查找已经成功，因此不会再进入 `nn.Module.__getattr__` 从 `_modules` 取回注册模块。结果是 `self.fuse(...)` 始终调用加法方法；`FusionModule` 虽然仍存在于 `_modules`、`state_dict`、`named_parameters()` 和优化器参数组中，却没有进入 forward 计算图，其参数没有来自任务损失的梯度。

补丁使用：

```python
return self._modules['fuse'](rgb, t)
```

直接字典访问绕过同名类方法，不会再次调用 `self.fuse`，所以不会递归。`FusionModule.forward` 与加法方法都返回 `(fuser_image_feats, fuser_i_feats)`：第一项是 `[B, L, D]` 的融合 token，第二项是 `[B, D]` 的首 token 浮点特征。因此现有 forward 调用点：

```python
fuser_image_feats, fuser_i_feats = self.fuse(
    rgb_image_feats, t_image_features
)
```

不需要修改。`solver/build.py` 已按参数名中包含 `fuse` 的规则使用 `args.fuse_lr`，开启动态分支后，原有的 `fuse.*` 参数名及独立学习率组保持不变。

DC-dyn 必须作为独立配置从头训练。不能把 DC-add checkpoint 在评测时临时打开开关并视为 DC-dyn，因为 DC-add 训练期间注册的 `FusionModule` 没有参与任务损失，其权重没有得到对应训练。

## 其它源码事实

`model/build.py` 中 `encode_image` 被连续定义了两次。类创建后只有第二个定义生效；实际方法返回 `base_model.encode_image(image)` 的完整 token 序列，而不是第一个定义中的首 token。集成套件只记录这一事实，不在本补丁中改动它。

RGB 和热成像共用 `base_model.encode_image`。DCAlign 使用的 `caption_2_no_color.json` 与本文 `Caption.json` 的身份编号不同，但服务器已逐组确认其 train/test 身份分区完全一致，因此统一检索指标可比较；集成套件不尝试重写标注身份编号。

## checkpoint 与评测协议

DCAlign 原 `test.py` 硬编码加载 `output_dir/best.pth`。该权重由 test 指标择优产生，与本文“统一使用训练结束时的最终 checkpoint，不依据测试指标在多个 checkpoint 之间选择”的协议冲突。因此本套件遵循以下固定规则：

- `eval_dcalign_degraded.py` 的 `--checkpoint` 是必需参数，没有默认值，也绝不回退到 `best.pth`；
- 每个 DCAlign 训练运行同时保留 `last.pth` 与 `best.pth`，不得在退化评测后删除任意一个；
- 11 个统一退化场景全部只在 `last.pth` 上执行，每个场景使用独立评测进程；
- `best.pth` 只额外评测一次 clean 场景，用于与文献报告的 R1 `58.47` 做可比性核对，不进入退化主结果；
- last-checkpoint 结果的 group 为 `DC-add` 或 `DC-dyn`；best-checkpoint clean 核对的 group 为 `DC-add-best` 或 `DC-dyn-best`。

DC-add 表示默认的 RGB 与热成像逐元素相加；DC-dyn 表示显式启用已注册的 `FusionModule`。两者必须使用各自训练得到的 checkpoint，不能交换权重或在评测时切换融合语义。

DCAlign commit `1ca105f` 的上游 `Checkpointer` 会按 key 后缀匹配，并让未命中的参数保留模型刚初始化的值。独立评测脚本因此会先去除统一的 `module.` 前缀，再断言 checkpoint 与当前模型的 key 集及每个 tensor shape 完全一致；只有通过后才调用上游 loader，禁止截断或异构权重静默进入评测。

运行器还会把实验输入锁定为三部分：base config 的原始字节 SHA256；本仓库运行时套件（runner、独立 evaluator、退化副本、parity 脚本、`utils/metrics.py`、`utils/eval_degradation.py`）的 manifest SHA256；以及 DCAlign 源码状态。DCAlign 状态由完整 commit
`1ca105f668fb71352d888a6c1f37e0b382461d0d` 与上述 7 个已跟踪环境适配文件的完整二进制 diff SHA256 共同确定。除这 7 个路径外出现其它已跟踪改动，或出现未跟踪 Python 文件，都会 fail-closed。启动时锁定的 HEAD 与 diff SHA 会传给每个独立评测进程、写入每个 score payload 和完成标记，并在断点续跑时逐一复核；因此不同源码状态的 seed 或场景不能静默混入同一网格。

同样地，base config SHA 与 IRRA runtime-kit SHA 也会写入每个 score 和完成标记。使用同一 `OUT` 换配置或换评测实现后重启时，旧完成标记会被隔离而不会静默跳过。base config 必须明确保持 `resume: false`（缺省也视为 false）；运行器拒绝从旧 checkpoint 续训，以保证 DC-add 与 DC-dyn 的每个 seed 都是独立从头训练。

## 使用评测脚本与运行器

先在 IRRA 项目根运行一致性检查：

副本文件头之后的 payload 固定采用 Git/Linux 的 LF 字节形态。校验器只把 Windows checkout 自动呈现的 CRLF 规范化为 LF，并拒绝独立 CR；因此 `source_sha256` 与 `payload_sha256` 在 Windows 和 Linux 上都代表同一份已提交源码，且必须相等。输出还单独保留当前物理 checkout 的 `source_checkout_sha256` 与 `copy_checkout_sha256`，避免把平台换行转换误当成源码差异。

```bash
PYTHONDONTWRITEBYTECODE=1 \
python tools/dcalign/check_degradation_parity.py
```

单场景评测必须显式给出训练种子和 checkpoint；以下命令只评估
`last.pth` 上的一个 `rgb_noise_005` 场景：

`--dump-sim` 必须是尚不存在的 `.pt`，`--metrics-out` 必须是尚不存在的 `.tsv`，两者必须互异且不能与 config/checkpoint 重合。脚本会拒绝覆盖既有文件，避免参数误填时损坏 checkpoint、配置或上一轮评测产物。

```bash
PYTHONDONTWRITEBYTECODE=1 \
python tools/dcalign/eval_dcalign_degraded.py \
  --dcalign-root /root/autodl-tmp/code/RGBT-PEDE \
  --expected-dcalign-head 1ca105f668fb71352d888a6c1f37e0b382461d0d \
  --expected-dcalign-diff-sha256 '<运行器启动时打印的64位SHA256>' \
  --expected-irra-kit-sha256 '<运行器启动时打印的64位SHA256>' \
  --protocol-base-config /path/to/dcalign_base.yaml \
  --expected-base-config-sha256 '<运行器启动时打印的64位SHA256>' \
  --config-file /path/to/configs.yaml \
  --checkpoint /path/to/last.pth \
  --degrade-mode rgb_noise \
  --degrade-strength 0.05 \
  --degrade-kernel 7 \
  --degrade-seed 2026 \
  --scenario-name rgb_noise_005 \
  --group-name DC-add \
  --seed 101 \
  --dump-sim /path/to/rgb_noise_005.pt \
  --metrics-out /path/to/metrics_fragment.tsv
```

完整网格固定为 `DC-add`、`DC-dyn` 两组，各使用训练种子
`101,102,103,104,105`。先运行只验证模式；它会检查外部源码适配、
补丁标记、CLI、命令数组和退化 parity，但不会启动训练、评估或关机：

```bash
bash tools/dcalign/run_dcalign_grid.sh \
  --dcalign-root /root/autodl-tmp/code/RGBT-PEDE \
  --base-config /path/to/dcalign_base.yaml \
  --out /root/autodl-tmp/DCALIGN_GRID \
  --validate-only
```

去掉 `--validate-only` 才会执行 10 次训练和每次 12 个独立评测进程。
运行器要求服务器适配后的 `train.py` 明确支持 `--config_file`、
`--name`、`--output_dir`、`TRAIN_SEED`，且不再硬编码覆盖
`CUDA_VISIBLE_DEVICES`。启动审计会按词法作用域和实际语句顺序追踪
`TRAIN_SEED` 到每个 `set_seed(...)` 调用；例如先读取环境变量、随后又把
同一变量覆盖为常量会被拒绝。任一前提不成立都会在训练开始前失败退出。

真实运行的磁盘下限 `MIN_FREE_GB` 默认为 `25`。`AUTO_SHUTDOWN` 默认为 `1`：网格结束后（包括存在失败运行时）先打印总运行数、成功数、失败数、总耗时和剩余磁盘，再进入 60 秒关机宽限期。若本次不应自动关机，请在启动命令前设 `AUTO_SHUTDOWN=0`；倒计时期间也可创建 `$OUT/STOP` 取消关机。

## 服务器侧必要核验

本地仓库没有 DCAlign 代码、数据与 checkpoint，因此这里只能验证补丁语法、退化实现一致性和运行器命令结构，不能在本地启动真实训练或评测。服务器正式运行前至少应核验：

1. `git apply --check` 在保留 7 项环境适配的工作树上通过；
2. `grep` 同时命中 `model/build.py` 的分支和 `utils/options.py` 的参数定义；
3. DC-add 不传新参数，DC-dyn 明确传入新参数；
4. 首次加载 checkpoint 时检查 missing/unexpected keys，确认 `fuse.*` 参数存在且结构匹配；
5. 统一退化调用全部显式传 `last.pth`，只有 clean 可比性核对显式传 `best.pth`。
