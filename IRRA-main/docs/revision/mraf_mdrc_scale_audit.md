# MRAF 前置审计：MDRC loss scale

## 审计结论

结论 **C：当前本地证据不足，无法确认正式 MDRC 实验实际使用的是
`mdrc_loss_scale=1.0` 还是 `mdrc_loss_scale=100.0`。**

后续代码继续保留现有 CLI 默认值 `1.0`。MDRC 复核及 MDRC+MRAF
正式实验脚本不得替用户选择数值，必须要求显式设置环境变量
`MDRC_LOSS_SCALE`，未设置时立即退出。

## 已检查的证据

- 当前 worktree：`C:\Users\12823\Desktop\IRRA\IRRA-main-mraf`
- 分支：`feature/mraf-reliability-fusion`
- 基准 HEAD：`5f787e67ca32404fee0256b78829456aed5a9484`
- 当前代码：
  - `utils/options.py` 将 `--mdrc_loss_scale` 默认值定义为 `1.0`。
  - `processor/processor.py` 的 `_mdrc_loss_scale()` 后备值为 `1.0`。
- Git 历史：
  - 提交 `a37d4e7a0237ede5044106910a8050b7d5e4a90c`
    （`add MDRC loss scale`）首次引入该参数。
  - 该提交将默认值设为 `1.0`，并把 scale 仅乘到 MDRC loss 上。
  - `git log -S`、`git log -G`、`git blame` 均未提供正式训练时的
    命令行实参。
- 当前 worktree 中可访问的脚本、README、revision 文档、YAML、JSON、
  日志和命令文件：
  - 未找到 `paper_e3_mdrc_seed1` 对应的启动记录。
  - 未找到包含 `--mdrc_loss_scale 100` 或
    `--mdrc_loss_scale 1` 的正式 MDRC 启动命令。
  - `scripts/revision/README_AutoDL.md` 出现 `mdrc_seed1` 评估标签，
    但没有训练期 `mdrc_loss_scale` 证据。
- 本机 PowerShell/Bash 命令历史：
  - 未找到 `mdrc_loss_scale`、`paper_e3_mdrc_seed1` 或正式 MDRC
    启动命令。
- Git 对象与全部可见 refs：
  - 未发现被跟踪的正式 MDRC `configs.yaml`、nohup 命令或相邻
    checkpoint 配置。

## 没有采用的推断

以下信息不能证明正式实验的实际有效参数，因此未被用于选择 scale：

- 论文或任务说明中出现的 `100.0`；
- 当前代码默认值 `1.0`；
- 指标、loss 数值或 checkpoint 表现；
- 提交时间与实验时间的间接关系。

## 范围限制

本轮明确禁止访问或处理原工作树
`C:\Users\12823\Desktop\IRRA\IRRA-main`。因此没有执行会遍历该目录的
父目录递归搜索，也没有读取原工作树中的未跟踪日志、配置或其他资产。
这项限制不会被当作 `1.0` 或 `100.0` 的反向证据。

## 后续执行规则

1. 保留 `mdrc_loss_scale` 默认值 `1.0`，不改变历史 CLI 行为。
2. MRAF-only 模式不使用 `mdrc_loss_scale`。
3. MDRC 复核、MDRC+MRAF 和 MDRC+MRAF 无可靠性监督实验必须显式传入：

   ```bash
   MDRC_LOSS_SCALE="${MDRC_LOSS_SCALE:?请先根据历史正式MDRC日志设置MDRC_LOSS_SCALE}"
   ```

4. 启动日志必须打印 `effective_mdrc_loss_scale`。
5. 在取得历史正式训练命令或训练目录中的 `configs.yaml` 后，应由人工
   将本结论更新为 A 或 B，再运行正式实验。
6. 本次实现与测试不启动任何正式训练。
