# Local validation report

## Environment

- Date: 2026-07-10
- Branch: `paper_revision_20260710`
- Baseline HEAD: `87610d503a8a`
- Python: 3.12.4
- PyTorch: 2.5.1+cu121
- torchvision: 0.20.1+cu121
- Pillow: 10.3.0
- NumPy: 1.26.4
- CUDA available: yes
- Bash: GNU bash 5.2.21 through WSL

No real RGBT-PEDES dataloader, model checkpoint or AutoDL environment was used.

## Commands and results

### Syntax compilation

Initial command:

```powershell
python -m py_compile utils/eval_degradation.py test.py utils/metrics.py
```

Result: passed. Because this repository unexpectedly tracks `__pycache__/test.cpython-312.pyc`, the generated binary change was immediately restored to baseline. Final syntax validation was therefore performed without writing bytecode:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -c "from pathlib import Path; files=['utils/eval_degradation.py','utils/metrics.py','test.py','tests/test_evaluation_degradation.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]"
```

Result: passed for all four modified/new Python source files. Final Git status contains no `*.pyc` change.

### unittest

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_evaluation_degradation -v
```

Result: `Ran 29 tests ... OK`. The optional CUDA test ran and passed rather than being skipped.

### pytest

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/test_evaluation_degradation.py -q -p no:cacheprovider
```

Result: `29 passed in 1.87s`.

### CLI validation probe

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python test.py --config_file __missing__.yaml --eval_degrade_mode rgb_jpeg --eval_degrade_strength 20.5
```

Result: failed before config loading as intended with `ValueError` stating that JPEG quality must be an integer in `[1, 100]`.

### Shell syntax

```powershell
bash -n scripts/revision/run_new_degradation_matrix.sh
bash -n scripts/revision/run_old_degradation_regression.sh
```

Result: both exited 0. WSL printed a host NAT/network warning, but Bash reported no script syntax error.

### Git checks

```powershell
git diff --check
git status --short
```

Result: `git diff --check` passed. Final outputs are captured again at handoff. No checkpoint, score, result, log or tracked bytecode file is changed.

## Coverage summary

- CLIP normalize/denormalize inverse
- brightness shape/device/dtype, physical brightness reduction, factor 1 and validation
- low-resolution shape, small dimensions, scale 1 and validation
- JPEG qualities 1/100, invalid 20.5, pixel range and same-environment determinism
- CPU and available CUDA paths
- all six mode shapes, modality isolation with `torch.equal`, and no in-place input mutation
- `none` no-op
- exact legacy `utils/metrics.py` degradation source block presence

The legacy degradation block SHA-256 calculated from its normalized source lines was `8c11d0d1548588e80f9d81002bba99c0e7bc326e05d94b5be475216130533004` both before and after the edit. Git diff shows no textual modification inside that block.

## Not locally validated

- Full RGBT-PEDES evaluation with a real Base or MDRC checkpoint
- Runtime score generation on AutoDL
- Element-wise old-versus-new comparison of `sim` and `raw_sim`
- Equality of PID/index tensors and `order_hash` against old saved scores
- Cross-version Pillow/libjpeg JPEG equality; this is not guaranteed by design
- Throughput and GPU-memory impact at the production evaluation batch size

---

# 第二轮离线分析工具验证（追加）

## 实施边界

- 未修改 `test.py`、`utils/metrics.py`、`utils/eval_degradation.py` 或 `tests/test_evaluation_degradation.py` 的第一轮行为。
- 未修改 `model/`、`datasets/`、`train.py`、`processor/processor.py`、`utils/mdrc.py` 或 checkpoint/score 格式。
- Feature norm 复用现有 `model.encode_rgbt_eval(..., return_local_tokens=True)`，该接口已经返回 `rgb_global_feat`、`thermal_global_feat` 和最终 `global_feat`，无需模型源码改动。
- 第二轮仅新增 `tools/revision/` 工具与 `tests/test_revision_analysis_tools.py`，并在本报告和 `scripts/revision/README_AutoDL.md` 末尾追加说明。

## Python 语法检查

为避免再次写入仓库追踪的字节码，使用内存 `compile()`：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -c "from pathlib import Path; files=sorted(Path('tools/revision').glob('*.py'))+[Path('tests/test_revision_analysis_tools.py')]; [compile(p.read_text(encoding='utf-8'), str(p), 'exec') for p in files]"
```

结果：`in-memory compile OK 11 files`。

## 第一轮回归测试

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_evaluation_degradation -v
```

结果：`Ran 29 tests ... OK`。本机 CUDA 可用，CUDA 测试实际运行并通过。

## 第二轮单元测试

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest tests.test_revision_analysis_tools -v
```

结果：`Ran 14 tests ... OK`。

覆盖内容：

- shape/order_hash 不一致时拒绝融合；
- PID 相同但 indices 不同时拒绝融合；
- NaN/Inf 检测；
- row-wise z-score 每行均值接近 0；
- 等权融合 shape；
- 相同/不同 score 逐元素比较；
- Oracle 默认关闭且非 7 模型 Oracle 被拒绝；
- 颜色词单词边界；
- Wilcoxon 由输入动态计算；
- 缺失鲁棒性场景报错；
- 少于 3 个噪声 seed 标记 incomplete；
- 强度按数值排序；
- retrieval 指标和 feature norm 纯函数。

## pytest 合并运行

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest tests/test_evaluation_degradation.py tests/test_revision_analysis_tools.py -q -p no:cacheprovider
```

结果：`43 passed, 1 warning in 9.33s`。唯一 warning 来自本机 `google.protobuf` 的 `datetime.utcfromtimestamp()` 弃用提示，与本轮代码无关。

## 所有命令行入口 --help

以下 9 个入口均以退出码 0 通过：

```text
score_io.py --help OK
compare_score_files.py --help OK
ensemble_scores.py --help OK
summarize_robustness.py --help OK
paired_seed_statistics.py --help OK
noise_repeat_summary.py --help OK
intensity_curve_summary.py --help OK
color_group_evaluation.py --help OK
feature_norm_analysis.py --help OK
```

## 本地只读事实检查

- 使用项目正式 `RGBTPEDES` 解析器读取到测试集 1473 张图库图像、2490 条查询。
- 构造与正式顺序一致的内存 score metadata 后，`restore_test_order` 成功通过 2490/1473、唯一 indices 和 PID 对齐检查。
- 用非完美随机小矩阵对比第二轮 `compute_retrieval_metrics` 与现有 `utils.metrics.rank`，R1/R5/R10/mAP/mINP 五项 float 结果完全相同。
- SciPy、matplotlib 和 pytest 在本机均可导入。
- `git diff --check` 通过，仅有现有 Git LF/CRLF 提示。

## 尚未本地或 AutoDL 验证

- 未用真实 Base/MDRC score 执行逐元素旧模式回归比较。
- 未用 7 个真实模型 score 生成正式融合或 Oracle upper bound。
- 未运行完整 11 场景 robustness manifest、5 个真实 seed、3 个真实噪声 seed 或 14 点强度曲线。
- 颜色分组已验证真实标注恢复，但未对真实 score 计算两组最终指标。
- Feature norm 接口静态可用、纯统计函数已测试，但未加载真实 checkpoint 执行 1473 图像前向。
- 未在 AutoDL 的 Pillow/PyTorch/SciPy 环境复核版本差异、吞吐或显存。
