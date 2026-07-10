# AutoDL evaluation-only degradation runs

## Isolation and scope

These changes are local to the revision worktree and do not modify the older AutoDL worktree that is running seed 4/5 or existing degradation experiments. No commit, push, server command, checkpoint, dataset, annotation or historical result is changed by this implementation.

The six new modes are evaluation-only:

- `rgb_brightness`, `ir_brightness`
- `rgb_lowres`, `ir_lowres`
- `rgb_jpeg`, `ir_jpeg`

They are applied by `Evaluator` immediately before image model forward. Training degradation and MDRC training code are untouched.

## Local unit tests

From `IRRA-main`:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_evaluation_degradation -v
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_evaluation_degradation.py -q -p no:cacheprovider
```

## Prepare an isolated AutoDL worktree

Do not switch the branch or edit files inside the worktree currently running experiments. Create or use a separate worktree based on the revision branch/commit, then run all commands from that new worktree. A generic example is:

```bash
git worktree add /root/autodl-tmp/IRRA-paper-revision <revision-branch-or-commit>
cd /root/autodl-tmp/IRRA-paper-revision/IRRA-main
```

Substitute paths and revision according to the server repository layout. Confirm `git status --short --branch` and the checkpoint/config paths before evaluation.

## Foreground run: six new scenarios

The same script works for Base or MDRC; use a distinct `--label`, config and checkpoint for each model.

```bash
bash scripts/revision/run_new_degradation_matrix.sh \
  --repo /root/autodl-tmp/IRRA-paper-revision/IRRA-main \
  --result-root /root/autodl-tmp/revision_degradation_results \
  --config-file /path/to/model/configs.yaml \
  --checkpoint /path/to/model/checkpoint.pth \
  --label mdrc_seed1 \
  --degrade-seed 2026
```

Defaults run brightness `0.3`, low-resolution scale `0.25` and JPEG quality `20` for RGB and IR separately. Each case has its own directory, `eval.log` and `score.pt`. Both `--save_eval_scores` and `--eval_scores_save_raw` are enabled.

## Foreground run: selected legacy regression scenarios

```bash
bash scripts/revision/run_old_degradation_regression.sh \
  --repo /root/autodl-tmp/IRRA-paper-revision/IRRA-main \
  --result-root /root/autodl-tmp/revision_old_regression \
  --config-file /path/to/model/configs.yaml \
  --checkpoint /path/to/model/checkpoint.pth \
  --label old_code_reference_or_new_code
```

This runs `none`, `rgb_missing`, `rgb_noise` at `0.05`, and `rgb_blur` with kernel 7 using fixed seed 2026. It saves scores only; comparison belongs to the later analysis-tool stage.

## Background run with nohup

The scripts themselves do not invoke `nohup`. If a background process is needed, wrap the script externally:

```bash
mkdir -p /root/autodl-tmp/revision_driver_logs
nohup bash scripts/revision/run_new_degradation_matrix.sh \
  --repo /root/autodl-tmp/IRRA-paper-revision/IRRA-main \
  --result-root /root/autodl-tmp/revision_degradation_results \
  --config-file /path/to/model/configs.yaml \
  --checkpoint /path/to/model/checkpoint.pth \
  --label base_seed1 \
  --degrade-seed 2026 \
  > /root/autodl-tmp/revision_driver_logs/base_seed1.log 2>&1 &
echo $!
```

Do not use `screen`.

## Watch, interrupt and resume

Watch the driver or per-case log:

```bash
tail -f /root/autodl-tmp/revision_driver_logs/base_seed1.log
tail -f /root/autodl-tmp/revision_degradation_results/base_seed1/rgb_jpeg_strength_20/eval.log
```

Interrupt the foreground process with `Ctrl-C`, or send `TERM` to the recorded background PID. Rerun the identical command to resume. A case with an existing `score.pt` is skipped; an incomplete case appends to its existing log and reruns. No script deletes results, overwrites an existing score, calls `screen`, or powers off the machine. Use a new `--label` rather than reusing a historical result directory for a different model/checkpoint.

## Legacy score comparison contract

Do not compare entire `.pt` bytes or whole-file SHA-256 because the revised score metadata has four additive degradation fields. The later comparison tool must compare these fields independently:

- `sim`
- `raw_sim`
- `txt_pids`
- `img_pids`
- `txt_indices`
- `img_indices`
- `order_hash`

For every legacy mode, `sim` and `raw_sim` must be element-wise identical between old and revised code. PID/index tensors must match, and `order_hash` strings must match. The current `order_hash` covers only text PID order, image PID order and score shape; it does not cover indices or score values.

## JPEG reproducibility note

JPEG uses Pillow and in-memory `BytesIO` with explicit quality, 4:4:4 subsampling (`subsampling=0`), `optimize=False` and `progressive=False`. Repeated runs are deterministic in the same Pillow/libjpeg environment. Different Pillow or linked libjpeg versions may not decode to pixel-identical tensors, so record the environment version when comparing machines.

---

# 第二轮：离线分析、融合、统计与诊断工具

以下内容为追加说明。第二轮工具位于 `tools/revision/`，只读取已有 score、配置、checkpoint 和标注，不修改训练、模型源码、数据集或原评估流程。所有产生文件的命令默认拒绝覆盖已有路径；为每次 Base、MDRC、seed 或分析使用新的 `--output-prefix`/结果目录。

## AutoDL 隔离 worktree

不要在仍执行旧实验的 worktree 中切换分支。继续使用独立 worktree：

```bash
git worktree add /root/autodl-tmp/IRRA-paper-revision <revision-branch-or-commit>
cd /root/autodl-tmp/IRRA-paper-revision/IRRA-main
git status --short --branch
```

旧模式回归和第一轮新退化评估仍分别使用前文的：

```bash
bash scripts/revision/run_old_degradation_regression.sh --help
bash scripts/revision/run_new_degradation_matrix.sh --help
```

## score 逐元素回归比较

旧代码和新代码的 `.pt` metadata 可以不同，因此不得比较整个文件 SHA-256。分别比较 `sim` 与 `raw_sim`，工具还会同时校验 shape、PID、indices 和 `order_hash`：

```bash
python tools/revision/compare_score_files.py \
  --file-a /path/to/old/score.pt \
  --file-b /path/to/new/score.pt \
  --matrix-key sim \
  --rtol 0 \
  --atol 0 \
  --output-json /new/result/path/compare_sim.json

python tools/revision/compare_score_files.py \
  --file-a /path/to/old/score.pt \
  --file-b /path/to/new/score.pt \
  --matrix-key raw_sim \
  --rtol 0 \
  --atol 0 \
  --output-json /new/result/path/compare_raw_sim.json
```

完全满足顺序校验且矩阵 `allclose` 时退出码为 0；否则为非 0。工具不重排或修复 score。

## 7 模型正式融合：row_zscore_equal

正式结果固定为每个模型逐查询行 `(row-mean)/(row_std+epsilon)` 后等权平均，不搜索测试集权重：

```bash
python tools/revision/ensemble_scores.py \
  --score-files \
    /scores/model1.pt /scores/model2.pt /scores/model3.pt \
    /scores/model4.pt /scores/model5.pt /scores/model6.pt /scores/model7.pt \
  --matrix-key sim \
  --fusion-method row_zscore_equal \
  --epsilon 1e-6 \
  --output-prefix /new/results/formal_7model_row_zscore_equal
```

输出 `.pt`、`.json`、`.tsv`、`.md`。融合 `.pt` 保留输入的 PID、indices 和 `order_hash`，并重新计算 R1/R5/R10/mAP/mINP。对 Base 和 MDRC 使用不同输入列表和不同输出前缀。

### Oracle upper bound

Oracle 默认关闭。只有显式传入恰好 7 个权重时启用，且输出 basename 必须包含 `oracle_upper_bound`：

```bash
python tools/revision/ensemble_scores.py \
  --score-files /scores/m1.pt /scores/m2.pt /scores/m3.pt /scores/m4.pt \
                /scores/m5.pt /scores/m6.pt /scores/m7.pt \
  --fusion-method row_zscore_equal \
  --oracle-weights 0.10 0.15 0.10 0.20 0.15 0.10 0.20 \
  --output-prefix /new/results/oracle_upper_bound_7model
```

Oracle metadata 固定包含 `oracle_only=true` 和 `test_set_tuned_or_preselected_weights=true`，不得作为正式结果。

## 鲁棒性总体汇总

manifest 必须显式给出全部 11 个场景，并对每个退化写明训练是否见过。不能根据场景名猜测 seen/unseen。示例：

```json
{
  "model_name": "MDRC_seed1",
  "scenarios": {
    "clean": {"score_file": "scores/clean.pt"},
    "rgb_missing": {"score_file": "scores/rgb_missing.pt", "seen_in_training": true},
    "ir_missing": {"score_file": "scores/ir_missing.pt", "seen_in_training": true},
    "rgb_noise_005": {"score_file": "scores/rgb_noise_005.pt", "seen_in_training": true},
    "rgb_noise_010": {"score_file": "scores/rgb_noise_010.pt", "seen_in_training": false},
    "ir_noise_005": {"score_file": "scores/ir_noise_005.pt", "seen_in_training": true},
    "ir_noise_010": {"score_file": "scores/ir_noise_010.pt", "seen_in_training": false},
    "rgb_blur_k7": {"score_file": "scores/rgb_blur_k7.pt", "seen_in_training": false},
    "ir_blur_k7": {"score_file": "scores/ir_blur_k7.pt", "seen_in_training": false},
    "both_noise_005": {"score_file": "scores/both_noise_005.pt", "seen_in_training": true},
    "both_blur_k7": {"score_file": "scores/both_blur_k7.pt", "seen_in_training": false}
  }
}
```

相对路径以 manifest 所在目录为基准：

```bash
python tools/revision/summarize_robustness.py \
  --manifest /manifests/mdrc_seed1_robustness.json \
  --matrix-key sim \
  --output-prefix /new/results/mdrc_seed1_robustness
```

Base 和 MDRC 分别运行，输出 JSON/TSV/Markdown 及 Pareto 的 clean R1、退化平均绝对 R1 坐标。

## 五种子 Base/MDRC 配对统计

输入为上述鲁棒性 JSON：

```json
{
  "pairs": [
    {"seed": 1, "base_summary": "base_seed1.json", "mdrc_summary": "mdrc_seed1.json"},
    {"seed": 2, "base_summary": "base_seed2.json", "mdrc_summary": "mdrc_seed2.json"},
    {"seed": 3, "base_summary": "base_seed3.json", "mdrc_summary": "mdrc_seed3.json"},
    {"seed": 4, "base_summary": "base_seed4.json", "mdrc_summary": "mdrc_seed4.json"},
    {"seed": 5, "base_summary": "base_seed5.json", "mdrc_summary": "mdrc_seed5.json"}
  ]
}
```

```bash
python tools/revision/paired_seed_statistics.py \
  --manifest /manifests/five_seed_pairs.json \
  --output-prefix /new/results/base_vs_mdrc_five_seed
```

工具使用 SciPy 计算 Student t 95% CI、单边和双边 Wilcoxon，明确记录 `n=5` 与小样本局限；SciPy 不可用时直接报错，不提供近似 p 值。

## 噪声随机种子重复统计

manifest 每条记录显式包含模型、场景、退化 seed 和 score：

```json
{
  "expected_seeds": [2026, 2027, 2028],
  "records": [
    {"model": "Base", "scenario": "rgb_noise_005", "degrade_seed": 2026, "score_file": "base_rgb_noise_005_s2026.pt"},
    {"model": "Base", "scenario": "rgb_noise_005", "degrade_seed": 2027, "score_file": "base_rgb_noise_005_s2027.pt"},
    {"model": "Base", "scenario": "rgb_noise_005", "degrade_seed": 2028, "score_file": "base_rgb_noise_005_s2028.pt"}
  ]
}
```

```bash
python tools/revision/noise_repeat_summary.py \
  --manifest /manifests/noise_repeats.json \
  --output-prefix /new/results/noise_repeats
```

缺少 2026/2027/2028 任一次时保留现有统计并标记 `incomplete=true`。

## 强度曲线

manifest 的 `series` 必须为 `rgb_noise`、`ir_noise`、`rgb_blur`、`ir_blur`，并完整提供固定强度。noise 使用数值 `0.02/0.05/0.10/0.20`，blur 使用 kernel 数值 `3/7/11`：

```json
{
  "model_name": "MDRC_seed1",
  "records": [
    {"series": "rgb_noise", "intensity": 0.02, "score_file": "rgb_noise_002.pt"},
    {"series": "rgb_noise", "intensity": 0.05, "score_file": "rgb_noise_005.pt"}
  ]
}
```

上例仅展示格式；正式 manifest 必须补齐 14 条记录，否则工具报错：

```bash
python tools/revision/intensity_curve_summary.py \
  --manifest /manifests/mdrc_intensity.json \
  --output-prefix /new/results/mdrc_intensity \
  --plot-dir /new/results/mdrc_intensity_plots
```

图由 matplotlib 默认颜色生成，不使用 seaborn；每个 series/metric 单独一张图，无 subplot。

## 颜色词分组

颜色词来自 `tools/revision/color_terms_en.txt`。工具通过正式 `RGBTPEDES` 解析器读取测试集，再用 score 中 `txt_indices`/`img_indices` 恢复顺序并核对 PID；不会使用 `caption_2_no_color.json` 代替正式分组：

```bash
python tools/revision/color_group_evaluation.py \
  --score-file /scores/mdrc_clean.pt \
  --matrix-key sim \
  --root-dir datasets/TextReID \
  --output-prefix /new/results/mdrc_clean_color_groups
```

替换 `--score-file` 即可分析任意退化 score。工具强制检查 2490 条查询、1473 张图库图像、唯一 indices 和 PID 对齐。

## 特征范数诊断

现有模型接口已经直接返回融合前 `rgb_global_feat`、`thermal_global_feat` 和最终 `global_feat`，不需要改模型：

```bash
python tools/revision/feature_norm_analysis.py \
  --config-file /path/to/model/configs.yaml \
  --checkpoint /path/to/model/checkpoint.pth \
  --root-dir /root/autodl-tmp/IRRA-paper-revision/IRRA-main/datasets/TextReID \
  --split test \
  --device cuda \
  --output-prefix /new/results/mdrc_feature_norm
```

工具使用 `model.eval()` 与 `torch.inference_mode()`，不保存或修改 checkpoint，不更新模型参数，不给出无法严格定义的“RGB贡献百分比”。输出逐样本 CSV、总体 JSON 和 Markdown。

## 后台运行、查看与停止

工具内部不调用 `nohup`。需要后台运行时从外部包装：

```bash
mkdir -p /new/results/driver_logs
nohup python tools/revision/summarize_robustness.py \
  --manifest /manifests/mdrc_seed1_robustness.json \
  --output-prefix /new/results/mdrc_seed1_robustness \
  > /new/results/driver_logs/mdrc_seed1_robustness.log 2>&1 &
echo $!

tail -f /new/results/driver_logs/mdrc_seed1_robustness.log
ps -fp <PID>
kill -TERM <PID>
```

不要使用 `screen`。若任务中断，先确认目标文件是否已经完整生成；工具不会覆盖任何已存在输出，应换用新前缀或在确认无研究资产后由用户自行处理不完整文件。
