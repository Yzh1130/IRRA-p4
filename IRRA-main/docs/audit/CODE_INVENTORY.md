# 代码静态清点

本文档对应当前工作树基线 `d07171be5c2e12f9439c45568c9047b52ef8596d`。清点对象为源码中已经存在的路径、参数、数据流和计算式；代码片段均摘自所标文件与行号。

## 4.1 MRAF 模块完整清点

MRAF 主实现位于 `model/mraf.py`；模型接线位于 `model/build.py`；训练期可靠性监督及与 MDRC 共用的退化前向位于 `processor/processor.py`；参数定义位于 `utils/options.py`；评分器学习率参数组位于 `solver/build.py` 与 `solver/lr_scheduler.py`。

### 4.1(a) MRAF 超参数

命令行默认值定义于 `utils/options.py:122-136`：

```python
mraf_group = parser.add_argument_group("MRAF")
mraf_group.add_argument("--mraf_enabled", default=False, action='store_true', help="enable modality reliability-adaptive RGB-T fusion")
mraf_group.add_argument("--mraf_hidden_dim", type=int, default=128, help="hidden dimension of the shared MRAF reliability scorer")
mraf_group.add_argument("--mraf_dropout", type=float, default=0.1, help="dropout probability in the MRAF reliability scorer")
mraf_group.add_argument("--mraf_gate_temperature", type=float, default=1.0, help="softmax temperature for MRAF gate weights")
mraf_group.add_argument("--mraf_gate_logit_scale", type=float, default=1.0, help="positive multiplier applied to raw MRAF logits before temperature and softmax")
mraf_group.add_argument("--mraf_gate_fp32", default=False, action='store_true', help="keep only the MRAF scorer, logits, alpha, and fusion arithmetic in FP32")
mraf_group.add_argument("--mraf_gate_base_grad_scale", type=float, default=1.0, help="scale only base-task gate gradients into the MRAF scorer; valid range [0,1]")
mraf_group.add_argument("--mraf_equalize_modality_norm", default=False, action='store_true', help="equalize RGB/IR feature norms to their per-sample mean before MRAF weighted fusion; the gate and scorer path is unchanged")
mraf_group.add_argument("--mraf_freeze_gate", default=False, action='store_true', help="freeze MRAF gate weights at 0.5/0.5 for ablation; the scorer is still constructed but produces no effect on fusion")
mraf_group.add_argument("--mraf_scorer_lr_multiplier", type=float, default=1.0, help="positive learning-rate multiplier applied only to MRAF scorer parameters")
mraf_group.add_argument("--mraf_rel_lambda", type=float, default=0.01, help="maximum MRAF reliability supervision weight")
mraf_group.add_argument("--mraf_target_temperature", type=float, default=0.1, help="temperature for drift-derived MRAF reliability targets")
mraf_group.add_argument("--mraf_rel_warmup_epoch", type=int, default=3, help="number of epochs used to linearly warm up MRAF reliability supervision")
mraf_group.add_argument("--mraf_balance_lambda", type=float, default=0.0, help="reserved MRAF balance-loss weight; inactive in the first implementation")
```

| 参数 | 默认值 | 定义位置 | 读取与使用位置 |
| --- | ---: | --- | --- |
| `mraf_enabled` | `False` | `utils/options.py:123` | `model/build.py:110-112` 决定是否构造 `mraf_fusion`；`model/build.py:1075-1080` 执行融合；`processor/processor.py:4550-4551,5565-5579` 控制可靠性监督 |
| `mraf_target_temperature` | `0.1` | `utils/options.py:134` | `processor/processor.py:4778-4786` 传入目标函数；`model/mraf.py:307-308` 作负漂移 logit 的除数 |
| `mraf_gate_logit_scale` | `1.0` | `utils/options.py:127` | `model/build.py:119-121` 传入；`model/mraf.py:189,228` 乘 raw logits |
| `mraf_scorer_lr_multiplier` | `1.0` | `utils/options.py:132` | `solver/build.py:151-195` 形成评分器专用参数组；`solver/lr_scheduler.py:81-140` 保留倍率 |
| `mraf_rel_lambda` | `0.01` | `utils/options.py:133` | `processor/processor.py:4554-4562` 解析 warmup；`processor/processor.py:4828-4835` 加入总损失 |
| `mraf_freeze_gate` | `False` | `utils/options.py:131` | `model/build.py:129-131` 传入；`model/mraf.py:177-184,220-223` 固定 logits 与 alpha |
| `mraf_gate_temperature` | `1.0` | `utils/options.py:126` | `model/build.py:116-118` 传入；`model/mraf.py:190-192,229-231` 作 softmax 前除数 |
| `mraf_gate_base_grad_scale` | `1.0` | `utils/options.py:129` | `model/build.py:123-125` 传入；`model/mraf.py:193-199,232-237` 缩放基础任务流向 gate 的梯度 |
| `mraf_hidden_dim` | `128` | `utils/options.py:124` | `model/build.py:114` 传入；`model/mraf.py:65,98-111` 定义评分器隐藏维度 |
| `mraf_dropout` | `0.1` | `utils/options.py:125` | `model/build.py:115` 传入；`model/mraf.py:66,100,110` 构造 Dropout |
| `mraf_rel_warmup_epoch` | `3` | `utils/options.py:135` | `processor/processor.py:4554-4562` 将训练 epoch 转为零基；`model/mraf.py:338-355` 线性调制 |
| `mraf_balance_lambda` | `0.0` | `utils/options.py:136` | `utils/options.py:37-38` 校验非负；`processor/processor.py:4943` 写日志；生产损失路径没有读取该值 |
| `mraf_equalize_modality_norm` | `False` | `utils/options.py:130` | `model/build.py:126-128` 传入；`model/mraf.py:127-136,203-212,244-253` 调整参与融合的特征范数 |
| `mraf_gate_fp32` | `False` | `utils/options.py:128` | `model/build.py:122,2519-2523`；`model/mraf.py:114-115,167-213` 在禁用 autocast 的 FP32 路径计算评分器、权重与融合 |

模型构造处为 `model/build.py:109-132`：

```python
self.embed_dim = base_cfg["embed_dim"]
self.mraf_enabled = bool(getattr(args, "mraf_enabled", False))
if self.mraf_enabled:
    self.mraf_fusion = ModalityReliabilityAdaptiveFusion(
        feature_dim=self.embed_dim,
        hidden_dim=int(getattr(args, "mraf_hidden_dim", 128)),
        dropout=float(getattr(args, "mraf_dropout", 0.1)),
        gate_temperature=float(
            getattr(args, "mraf_gate_temperature", 1.0)
        ),
        gate_logit_scale=float(
            getattr(args, "mraf_gate_logit_scale", 1.0)
        ),
        gate_fp32=bool(getattr(args, "mraf_gate_fp32", False)),
        gate_base_grad_scale=float(
            getattr(args, "mraf_gate_base_grad_scale", 1.0)
        ),
        equalize_modality_norm=bool(
            getattr(args, "mraf_equalize_modality_norm", False)
        ),
        freeze_gate=bool(
            getattr(args, "mraf_freeze_gate", False)
        ),
    )
```

参数有效性校验位于 `utils/options.py:13-54`：`hidden_dim > 0`，`dropout` 位于 `[0,1)`，两个 temperature 大于 0，`gate_logit_scale` 与 `scorer_lr_multiplier` 为有限正数，两个 lambda 非负，`gate_base_grad_scale` 位于 `[0,1]`，warmup 非负；`mraf_enabled` 与 effective precision fusion 同时开启时抛出 `ValueError`。

### 4.1(b) 可靠性目标的完整计算链

#### 干净特征来源与前向次数

训练主路径的普通前向位于 `processor/processor.py:5341-5358`：

```python
if _degrade_aug_enabled(args):
    # Ablation single-branch degradation augmentation: replace the
    # image inputs before the forward so the degraded tensors flow
    # into all baseline losses. No second branch, no consistency loss.
    degrade_applied, degrade_mode_name = _apply_degrade_aug_to_batch(
        args, batch
    )
    if not degrade_aug_logged_this_epoch:
        logger.info(
            "[DEGRADE_AUG] epoch=%d iter=%d applied=%s mode=%s",
            epoch,
            n_iter + 1,
            bool(degrade_applied),
            degrade_mode_name,
        )
        degrade_aug_logged_this_epoch = True

ret = model(batch)
```

后续传入 `_compute_mraf_reliability_terms` 的 `clean_ret` 就是这里的 `ret`。当 `degrade_aug_mode` 启用时，该分支前的 batch 已由上述代码替换；否则它来自原训练 batch。

主前向形成逐模态均值特征的位置为 `model/build.py:2184-2214`：

```python
# 3) Optional diagonal-Gaussian heads over global pooled features.
# Returned mu/logvar are used by later uncertainty-aware losses/fusion.
rgb_mu, rgb_logvar, thermal_mu, thermal_logvar = self._encode_visual_global_gaussian(
    v_rgb_c_raw,
    v_t_c_raw,
    thermal_available=thermal_images is not None,
)
if self.mraf_enabled:
    fused_mu, fused_logvar, mraf_aux = self._fuse_rgbt_global(
        rgb_mu=rgb_mu,
        rgb_logvar=rgb_logvar,
        thermal_mu=thermal_mu,
        thermal_logvar=thermal_logvar,
        thermal_available=thermal_images is not None,
        return_mraf_aux=True,
    )
else:
    fused_mu, fused_logvar = self._fuse_rgbt_global(
        rgb_mu=rgb_mu,
        rgb_logvar=rgb_logvar,
        thermal_mu=thermal_mu,
        thermal_logvar=thermal_logvar,
        thermal_available=thermal_images is not None,
    )
    mraf_aux = None

# Keep existing training/eval flow compatible: deterministic embedding is mu.
v_rgb_c = rgb_mu
v_t_c = thermal_mu
l_c = text_mu
f_c = fused_mu
```

当对角高斯全局表示关闭时，`_encode_visual_global_gaussian` 原样返回两个模态的全局特征；开启时返回投影后的均值与 log variance。原文位于 `model/build.py:960-977`：

```python
def _encode_visual_global_gaussian(
    self, rgb_global_feat, thermal_global_feat, thermal_available
):
    if not self._use_diag_gaussian_global():
        return rgb_global_feat, None, thermal_global_feat, None

    rgb_mu, rgb_logvar = self._project_global_gaussian(
        rgb_global_feat, self.rgb_mu_head, self.rgb_logvar_head
    )
    if thermal_available:
        thermal_mu, thermal_logvar = self._project_global_gaussian(
            thermal_global_feat, self.thermal_mu_head, self.thermal_logvar_head
        )
    else:
        thermal_mu = torch.zeros_like(rgb_mu)
        thermal_logvar = torch.zeros_like(rgb_logvar)
    return rgb_mu, rgb_logvar, thermal_mu, thermal_logvar
```

可靠性监督与 MDRC 共用一次额外的退化图像前向。`processor/processor.py:4616-4647`：

```python
def _build_degraded_branch_context(args, model, batch):
    """Sample exactly one shared degraded RGB-T view and encode it once."""
    degraded_rgb, degraded_ir, degrade_info = degrade_rgbt_batch(
        batch["images"],
        batch.get("thermal_images", None),
        prob=float(getattr(args, "mdrc_prob", 0.5)),
        modes=getattr(args, "mdrc_degrade_modes", None),
        return_mode=True,
    )
    delta_rgb = (
        degraded_rgb.detach().float() - batch["images"].detach().float()
    ).abs().mean()
    if torch.is_tensor(degraded_ir) and "thermal_images" in batch:
        delta_ir = (
            degraded_ir.detach().float()
            - batch["thermal_images"].detach().float()
        ).abs().mean()
    else:
        delta_ir = delta_rgb.new_zeros(())

    degraded_batch = dict(batch)
    degraded_batch["images"] = degraded_rgb
    if "thermal_images" in batch:
        degraded_batch["thermal_images"] = degraded_ir
    degraded_ret = model(degraded_batch, mdrc_feature_only=True)
    return {
        "degraded_rgb": degraded_rgb,
        "degraded_ir": degraded_ir,
        "degraded_ret": degraded_ret,
        "delta_rgb": delta_rgb,
        "delta_ir": delta_ir,
        "degrade_info": degrade_info,
    }
```

`model/build.py:2027-2038,2046-2079` 的 `mdrc_feature_only` 分支只编码图像，不编码文本，也不计算基础损失；它返回 `v_rgb_c=rgb_mu`、`v_t_c=thermal_mu`、`fused_mu` 和 MRAF 辅助输出。执行接线位于 `processor/processor.py:5537-5579`：先执行主前向，再在 `_should_execute_degraded_branch` 为真时只执行一次上述 degraded image-only forward；没有第二次 clean forward。

#### 漂移量与目标分布

主/退化两组特征在 `processor/processor.py:4764-4791` 接入目标函数：

```python
degraded_ret = degraded_context["degraded_ret"]
required_clean_keys = ("v_rgb_c", "v_t_c", "mraf_alpha")
required_degraded_keys = ("v_rgb_c", "v_t_c", "mraf_alpha")
missing_clean = [key for key in required_clean_keys if key not in clean_ret]
missing_degraded = [
    key for key in required_degraded_keys if key not in degraded_ret
]
if missing_clean or missing_degraded:
    raise KeyError(
        "MRAF reliability supervision is missing model outputs. "
        f"Clean missing: {missing_clean}; degraded missing: "
        f"{missing_degraded}."
    )

target_outputs = compute_mraf_reliability_targets(
    clean_rgb=clean_ret["v_rgb_c"],
    clean_ir=clean_ret["v_t_c"],
    degraded_rgb=degraded_ret["v_rgb_c"],
    degraded_ir=degraded_ret["v_t_c"],
    target_temperature=float(
        getattr(args, "mraf_target_temperature", 0.1)
    ),
)
alpha_degraded = degraded_ret["mraf_alpha"]
loss_mraf_rel = compute_mraf_reliability_loss(
    alpha_degraded,
    target_outputs["target"],
)
```

目标的完整实现位于 `model/mraf.py:275-317`：

```python
def compute_mraf_reliability_targets(
    clean_rgb: torch.Tensor,
    clean_ir: torch.Tensor,
    degraded_rgb: torch.Tensor,
    degraded_ir: torch.Tensor,
    target_temperature: float = 0.1,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Build detached reliability targets from clean/degraded feature drift."""
    if float(target_temperature) <= 0.0:
        raise ValueError(
            f"target_temperature must be > 0, got {target_temperature}."
        )
    _validate_feature_pair("clean_rgb", clean_rgb, "clean_ir", clean_ir)
    _validate_feature_pair(
        "degraded_rgb", degraded_rgb, "degraded_ir", degraded_ir
    )
    _validate_feature_pair("clean_rgb", clean_rgb, "degraded_rgb", degraded_rgb)
    _validate_feature_pair("clean_ir", clean_ir, "degraded_ir", degraded_ir)

    drift_rgb = 1.0 - F.cosine_similarity(
        degraded_rgb.float(),
        clean_rgb.detach().float(),
        dim=-1,
        eps=float(eps),
    )
    drift_ir = 1.0 - F.cosine_similarity(
        degraded_ir.float(),
        clean_ir.detach().float(),
        dim=-1,
        eps=float(eps),
    )
    target_logits = torch.stack([-drift_rgb, -drift_ir], dim=-1)
    target = F.softmax(target_logits / float(target_temperature), dim=-1).detach()
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("MRAF reliability targets contain NaN or Inf values.")
    return {
        "target": target,
        "target_rgb": target[:, 0:1],
        "target_ir": target[:, 1:2],
        "drift_rgb": drift_rgb,
        "drift_ir": drift_ir,
    }
```

漂移的定义为：

\[
d_{rgb}=1-\operatorname{cosine}(v^{degraded}_{rgb},\operatorname{detach}(v^{clean}_{rgb})),
\quad
d_{ir}=1-\operatorname{cosine}(v^{degraded}_{ir},\operatorname{detach}(v^{clean}_{ir})).
\]

它是余弦距离；输入没有在函数外显式单位化，归一化发生在 `F.cosine_similarity` 内。四个输入为 `[B,D]`，两个漂移各为 `[B]`，stack 后 `target_logits` 为 `[B,2]`。目标是

\[
q=\operatorname{softmax}([-d_{rgb},-d_{ir}]/T_r).
\]

温度形式是 `d / T_r`，不是 `d * T_r`。softmax 的 `dim=-1` 是长度为 2 的模态维。目标在 `model/mraf.py:308` 的 softmax 结果上立即 `detach()`；`model/mraf.py:333-334` 还会拒绝 `requires_grad=True` 的 target。

### 4.1(c) 门控权重的完整计算链

#### 评分器结构

`model/mraf.py:98-124`：

```python
self.feature_dim = int(feature_dim)
self.hidden_dim = int(hidden_dim)
self.dropout = float(dropout)
self.gate_temperature = float(gate_temperature)
self.gate_logit_scale = float(gate_logit_scale)
self.gate_fp32 = bool(gate_fp32)
self.gate_base_grad_scale = float(gate_base_grad_scale)
self.equalize_modality_norm = bool(equalize_modality_norm)
self.freeze_gate = bool(freeze_gate)
self.scorer = nn.Sequential(
    nn.Linear(self.feature_dim, self.hidden_dim),
    nn.GELU(),
    nn.Dropout(self.dropout),
    nn.Linear(self.hidden_dim, 1),
)
self._zero_initialize_output_layer()

@property
def output_layer(self) -> nn.Linear:
    """Return the zero-initialized final scorer layer."""
    return self.scorer[-1]

def _zero_initialize_output_layer(self) -> None:
    nn.init.zeros_(self.output_layer.weight)
    nn.init.zeros_(self.output_layer.bias)
```

默认网络为 `[B,D] -> Linear(D,128) -> GELU -> Dropout(0.1) -> Linear(128,1)`；最后一层的 weight 与 bias 都初始化为 0。RGB 与 IR 复用同一个 `self.scorer`。

#### 输入、logit 与 alpha

MRAF 输入是 `_fuse_rgbt_global` 收到的 `rgb_mu`、`thermal_mu`。`model/build.py:1074-1080`：

```python
mraf_aux = None
if self.mraf_enabled and thermal_available:
    fused_mu, mraf_aux = self.mraf_fusion(
        rgb_mu,
        thermal_mu,
        return_aux=True,
    )
```

非 `gate_fp32` 路径位于 `model/mraf.py:214-231`：

```python
rgb_normalized = F.normalize(f_rgb, p=2, dim=-1).to(
    dtype=scorer_dtype
)
ir_normalized = F.normalize(f_ir, p=2, dim=-1).to(
    dtype=scorer_dtype
)
if self.freeze_gate:
    raw_logits = rgb_normalized.new_zeros((f_rgb.shape[0], 2))
    scaled_logits = raw_logits.clone()
    alpha = rgb_normalized.new_full((f_rgb.shape[0], 2), 0.5)
else:
    rgb_logit = self.scorer(rgb_normalized)
    ir_logit = self.scorer(ir_normalized)
    raw_logits = torch.cat([rgb_logit, ir_logit], dim=-1)
    scaled_logits = raw_logits * self.gate_logit_scale
    alpha = F.softmax(
        scaled_logits / self.gate_temperature, dim=-1
    )
```

两个 `[B,D]` 模态特征分别 L2 归一化；共享评分器各生成 `[B,1]`；拼接得到 `[B,2]` raw logits；`mraf_gate_logit_scale` 直接乘 raw logits；所得值再除 `mraf_gate_temperature`；最后沿模态维 `dim=-1` softmax 得 `[B,2]` 的 alpha。`gate_fp32=True` 时顺序不变，`model/mraf.py:166-192` 在 `torch.autocast(..., enabled=False)` 中把融合输入转为 float 后执行整条链。`freeze_gate=True` 时构造全零 logits 与固定 `[0.5,0.5]` alpha，不调用评分器。

#### 基础损失梯度与融合

`model/mraf.py:193-202` 的 detach 恒等式不改变前向 alpha，缩放基础任务流入 gate 的梯度：

```python
if self.gate_base_grad_scale == 1.0:
    alpha_for_base = alpha
else:
    alpha_for_base = (
        alpha.detach()
        + self.gate_base_grad_scale * (alpha - alpha.detach())
    )
alpha_rgb = alpha_for_base[:, 0:1]
alpha_ir = alpha_for_base[:, 1:2]
alpha_aux = alpha
```

非 FP32 分支的对应代码位于 `model/mraf.py:232-243`，并把未缩放梯度图的 `alpha_aux` 返回给可靠性监督。

融合公式位于 `model/mraf.py:244-253`：

```python
if self.equalize_modality_norm:
    rgb_for_fusion, ir_for_fusion = (
        self._equalize_feature_norms(f_rgb, f_ir)
    )
    fused = 2.0 * (
        alpha_rgb * rgb_for_fusion
        + alpha_ir * ir_for_fusion
    )
else:
    fused = 2.0 * (alpha_rgb * f_rgb + alpha_ir * f_ir)
```

即 `f_fused = 2 * (alpha_rgb * f_rgb + alpha_ir * f_ir)`。范数均衡开启时，`model/mraf.py:127-136` 先把两侧范数缩放到逐样本算术平均范数 `0.5*(n_rgb+n_ir)`，评分器的输入和 alpha 链不变。若使用对角高斯全局表示，`model/build.py:1081-1090` 按 `4*(alpha_rgb^2*var_rgb + alpha_ir^2*var_ir)` 传播融合方差。

### 4.1(d) 可靠性监督损失

损失实现位于 `model/mraf.py:320-335`：

```python
def compute_mraf_reliability_loss(
    alpha_degraded: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return MSE supervision for attached degraded-branch gate weights."""
    _validate_feature_pair(
        "alpha_degraded", alpha_degraded, "target", target
    )
    if alpha_degraded.shape[1] != 2:
        raise ValueError(
            "alpha_degraded and target must have shape [B, 2], got "
            f"{tuple(alpha_degraded.shape)}."
        )
    if target.requires_grad:
        raise ValueError("MRAF reliability target must be detached.")
    return F.mse_loss(alpha_degraded.float(), target.float())
```

它是退化分支 alpha 与 detached target 之间的 MSE。`processor/processor.py:4554-4562` 把训练使用的 1-based epoch 转为 0-based：

```python
def _resolve_mraf_rel_lambda(args, epoch):
    if not _mraf_enabled(args):
        return 0.0
    return resolve_mraf_rel_lambda(
        base_lambda=float(getattr(args, "mraf_rel_lambda", 0.01)),
        warmup_epoch=int(getattr(args, "mraf_rel_warmup_epoch", 3)),
        # do_train uses 1-based epochs; the mathematical schedule is 0-based.
        zero_based_epoch=max(0, int(epoch) - 1),
    )
```

`model/mraf.py:338-355` 返回 `base_lambda*(zero_based_epoch+1)/warmup_epoch`，直至达到 base lambda。默认 warmup=3 时第 1/2/3 epoch 分别为 `1/3`、`2/3`、`1` 倍。总损失接线 `processor/processor.py:4828-4835` 为：

```python
def _add_mraf_to_total_loss(args, total_loss, mraf_terms):
    if not _mraf_enabled(args) or not mraf_terms:
        return total_loss
    return (
        total_loss
        + mraf_terms["lambda_mraf_rel_cur"]
        * mraf_terms["loss_mraf_rel"]
    )
```

所以该项为 `lambda_rel(epoch) * MSE(alpha_degraded, target)`。`mraf_balance_lambda` 不进入该式。

### 4.1(e) 在目标 softmax logit 上加入逐模态常数偏置的位置

当前唯一的目标 softmax 数值位置是 `model/mraf.py:307-308`：

```python
target_logits = torch.stack([-drift_rgb, -drift_ir], dim=-1)
target = F.softmax(target_logits / float(target_temperature), dim=-1).detach()
```

目标形式 `softmax(-d_m/T_r + log u_m)` 的最少数值修改位置是 `model/mraf.py:308`：在 `target_logits / target_temperature` 之后、传给 `F.softmax` 之前，加入可广播为 `[1,2]` 的 `log(u)`。若先在第 307 行加入 `log(u)` 再把整体除以温度，得到的是 `-d/T_r + log(u)/T_r`，不是题设形式。

传递 `u` 的位置清单为：

1. `model/mraf.py:275-281`：给 `compute_mraf_reliability_targets` 增加两元素参数，顺序为 `[u_rgb,u_ir]`；数值在取 log 前为正。
2. `processor/processor.py:4778-4786`：在生产调用点传入训练期固定的两元素向量。
3. 若向量来自 CLI/config，则定义入口是 `utils/options.py:122-136` 的 MRAF 参数组，随后由第 2 项调用点向下传递。

`u=[1.0,1.0]` 时 `log(u)=[0,0]`。本清点未修改上述源码。

### 4.1(f) MDRC 参数与损失

MDRC 参数定义于 `utils/options.py:106-121`：

```python
mdrc_degrade_modes = (
    "rgb_drop",
    "ir_drop",
    "rgb_noise",
    "ir_noise",
    "rgb_attenuate",
    "ir_attenuate",
)
mdrc_group = parser.add_argument_group("MDRC")
mdrc_group.add_argument("--mdrc_enable", default=False, action='store_true', help="enable MDRC clean-degraded consistency training")
mdrc_group.add_argument("--mdrc_lambda", type=float, default=0.05, help="maximum MDRC consistency loss weight")
mdrc_group.add_argument("--mdrc_loss_scale", type=float, default=1.0, help="extra scale applied to MDRC loss before adding it to total loss")
mdrc_group.add_argument("--mdrc_prob", type=float, default=0.5, help="per-sample probability of applying one MDRC degradation")
mdrc_group.add_argument("--mdrc_warmup_epoch", type=int, default=3, help="1-based epoch at which MDRC starts; weight then linearly ramps")
mdrc_group.add_argument("--mdrc_temp", type=float, default=0.07, help="temperature for MDRC image-text similarity distribution consistency")
mdrc_group.add_argument("--mdrc_degrade_modes", type=str, nargs='+', default=list(mdrc_degrade_modes), help="MDRC degradation modes; supports space or comma separated values")
```

| 参数 | 默认值 | 实际读取与作用 |
| --- | ---: | --- |
| `mdrc_enable` | `False` | `processor/processor.py:4522-4523`，控制退化分支、损失与总损失接线 |
| `mdrc_lambda` | `0.05` | `processor/processor.py:4532-4547`，作为 warmup 后目标权重 |
| `mdrc_loss_scale` | `1.0` | `processor/processor.py:4526-4529,4707-4708,4741-4745`，在 lambda 外再乘一次 |
| `mdrc_prob` | `0.5` | `processor/processor.py:4618-4623`，逐样本退化激活概率 |
| `mdrc_warmup_epoch` | `3` | `processor/processor.py:4538-4547`，到指定 epoch 后线性爬升 |
| `mdrc_temp` | `0.07` | `utils/options.py:589-590` 校验，`processor/processor.py:4915` 写日志；当前 v1.1 损失没有读取 |
| `mdrc_degrade_modes` | 六种模式 | `processor/processor.py:4622` 传入 `degrade_rgbt_batch`；`utils/options.py:564-580` 规范化并校验 |
| noise 标准差 | `0.1` | `utils/mdrc.py:44-51` 的函数默认值 |
| attenuation 区间 | `[0.2,0.6]` | `utils/mdrc.py:54-63` 的函数默认值 |

MDRC 从 `processor/processor.py:4515-4521` 的候选键顺序选择二维图像特征：

```python
MDRC_IMAGE_FEATURE_KEYS = (
    "fused_det_feat",
    "fused_prob_feat",
    "fused_mu",
    "fused_feat_det",
    "fused_feat_sampled",
)
```

当前普通前向与 feature-only 前向均提供 `fused_det_feat`。`processor/processor.py:4675-4708` 取 clean/degraded 特征、检查 degraded 特征仍在 autograd 图中，然后计算：

```python
clean_img_feat, image_feature_key = _extract_mdrc_feature(
    clean_ret,
    MDRC_IMAGE_FEATURE_KEYS,
    "image",
    required=True,
)
degraded_ret = degraded_context["degraded_ret"]
degraded_img_feat, degraded_image_feature_key = _extract_mdrc_feature(
    degraded_ret,
    (image_feature_key,) + tuple(
        key for key in MDRC_IMAGE_FEATURE_KEYS if key != image_feature_key
    ),
    "degraded image",
    required=True,
)
degraded_requires_grad = bool(degraded_img_feat.requires_grad)
if not degraded_requires_grad:
    raise RuntimeError(
        "MDRC degraded image feature does not require grad. "
        "The degraded feature-only branch must remain in the autograd graph."
    )

loss_feature = mdrc_feature_consistency(clean_img_feat.detach(), degraded_img_feat)
loss_similarity = loss_feature.new_zeros(())
loss_mdrc = loss_feature
lambda_tensor = loss_mdrc.new_tensor(float(lambda_effective))
scale_tensor = loss_mdrc.new_tensor(float(_mdrc_loss_scale(args)))
loss_weighted = lambda_tensor * scale_tensor * loss_mdrc
```

`utils/mdrc.py:172-195` 的实际 v1.1 特征损失为归一化后的 MSE：

```python
def mdrc_feature_consistency(clean_feat, degraded_feat, eps=1e-12):
    """
    L2-normalized feature consistency with detached clean teacher features.
    """
    if not torch.is_tensor(clean_feat) or not torch.is_tensor(degraded_feat):
        raise TypeError("clean_feat and degraded_feat must be torch.Tensor values.")
    if clean_feat.shape != degraded_feat.shape:
        raise ValueError(
            "MDRC feature consistency expects matching feature shapes, got "
            f"{tuple(clean_feat.shape)} and {tuple(degraded_feat.shape)}."
        )
    clean_norm = F.normalize(
        torch.nan_to_num(clean_feat.detach().float(), nan=0.0, posinf=0.0, neginf=0.0),
        p=2,
        dim=-1,
        eps=eps,
    )
    degraded_norm = F.normalize(
        torch.nan_to_num(degraded_feat.float(), nan=0.0, posinf=0.0, neginf=0.0),
        p=2,
        dim=-1,
        eps=eps,
    )
    return F.mse_loss(degraded_norm, clean_norm)
```

`processor/processor.py:4736-4746` 将其以 `mdrc_lambda * mdrc_loss_scale` 加入总损失。`processor/processor.py:4532-4547` 在默认 warmup=3 时令 epoch 1/2 为 0、epoch 3/4/5 为目标 lambda 的 `1/3`、`2/3`、`1`，之后保持目标值。

`utils/mdrc.py:198-252` 还定义了双向 image-text similarity distribution KL：clean 分布为 detached teacher，degraded 分布为 student，i2t/t2i 取均值。当前 `processor/processor.py:37-41` 没有导入该函数，且 `processor/processor.py:4704` 把 `loss_similarity` 置零，故 `mdrc_temp` 当前不进入 v1.1 总损失。

## 4.2 退化注入点

### 4.2.1 评测入口、类型与参数

完整命令行定义位于 `test.py:294-336`：

```python
parser.add_argument(
    "--eval_degrade_mode",
    type=str,
    default="none",
    choices=[
        "none",
        "rgb_missing",
        "ir_missing",
        "rgb_noise",
        "ir_noise",
        "both_noise",
        "rgb_blur",
        "ir_blur",
        "both_blur",
        "rgb_brightness",
        "ir_brightness",
        "rgb_lowres",
        "ir_lowres",
        "rgb_jpeg",
        "ir_jpeg",
    ],
    help="evaluation-only temporary RGB/T degradation mode",
)
parser.add_argument(
    "--eval_degrade_strength",
    type=float,
    default=1.0,
    help=(
        "degradation strength: Gaussian std for *_noise, brightness factor "
        "for *_brightness, linear resize scale for *_lowres, or integer JPEG "
        "quality for *_jpeg"
    ),
)
parser.add_argument(
    "--eval_degrade_blur_kernel",
    type=int,
    default=7,
    help="evaluation-only blur kernel size for *_blur degrade modes",
)
parser.add_argument(
    "--eval_degrade_seed",
    type=int,
    default=2026,
    help="evaluation-only random seed for degradation noise",
)
```

`test.py:873-909` 校验并写入运行参数：strength 非负；brightness/lowres 的 strength 位于 `(0,1]`；JPEG quality 是 `[1,100]` 的整数；blur kernel 大于 0。

调度函数为 `utils/metrics.py:274-312`：

```python
def _apply_eval_degradation(self, model, img, thermal_img):
    model_args = getattr(model, "args", None)
    mode = str(getattr(model_args, "eval_degrade_mode", "none")).lower()
    if mode == "none":
        return img, thermal_img

    strength = float(getattr(model_args, "eval_degrade_strength", 1.0))
    blur_kernel = int(getattr(model_args, "eval_degrade_blur_kernel", 7))
    seed = int(getattr(model_args, "eval_degrade_seed", 2026))

    if mode == "rgb_missing":
        img = torch.zeros_like(img)
    elif mode == "ir_missing":
        if thermal_img is not None:
            thermal_img = torch.zeros_like(thermal_img)
    elif mode == "rgb_noise":
        img = self._add_eval_noise(img, strength, seed)
    elif mode == "ir_noise":
        thermal_img = self._add_eval_noise(thermal_img, strength, seed)
    elif mode == "both_noise":
        img = self._add_eval_noise(img, strength, seed)
        thermal_img = self._add_eval_noise(thermal_img, strength, seed)
    elif mode == "rgb_blur":
        img = self._blur_eval_image(img, blur_kernel)
    elif mode == "ir_blur":
        thermal_img = self._blur_eval_image(thermal_img, blur_kernel)
    elif mode == "both_blur":
        img = self._blur_eval_image(img, blur_kernel)
        thermal_img = self._blur_eval_image(thermal_img, blur_kernel)
    elif mode in NEW_EVAL_DEGRADATION_MODES:
        img, thermal_img = apply_new_eval_degradation(
            mode=mode,
            rgb_img=img,
            thermal_img=thermal_img,
            strength=strength,
        )
    else:
        raise ValueError(f"Unsupported eval_degrade_mode: {mode}.")
    return img, thermal_img
```

missing 把相应归一化张量置零；noise 的 strength 是归一化空间高斯标准差；blur 使用 replicate padding 后的 `avg_pool2d` 核尺寸；brightness 的 strength 是像素亮度乘数；lowres 是高宽线性缩放比例；JPEG 是 quality。

### 4.2.2 数据流中的位置

图像读取首先产生 PIL RGB，`utils/iotools.py:19-33`：

```python
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
```

标准评测变换在 `datasets/build.py:36-51` 为 `Resize -> ToTensor -> Normalize`：

```python
def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    # Build image preprocessing pipeline for train/eval.
    height, width = img_size

    # CLIP default normalization statistics.
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform
```

RGB 与 thermal 在 `datasets/bases.py:598-609` 都先应用该 transform。之后 evaluator 才移动到 device、施加退化并编码，`utils/metrics.py:380-404`：

```python
for data in self.img_loader:
    # Compatible with:
    #   (pid, img) for RGB-only gallery
    #   (pid, rgb_img, thermal_img) for RGB-T gallery
    if len(data) == 3:
        pid, img, thermal_img = data
        img = img.to(device)
        thermal_img = thermal_img.to(device)
    else:
        pid, img = data
        img = img.to(device)
        thermal_img = None
    img, thermal_img = self._apply_eval_degradation(
        model,
        img,
        thermal_img,
    )
    with torch.no_grad():
        image_outputs = self._encode_gallery_feature(
            model,
            img,
            thermal_img,
            return_local_tokens=need_rich_gallery_outputs,
        )
```

所以评测退化入口接收归一化后的 tensor，不接收 PIL 或 Normalize 前 tensor。旧 missing/noise/blur 直接在归一化空间处理。六个新模式 `rgb/ir_brightness`、`rgb/ir_lowres`、`rgb/ir_jpeg` 定义于 `utils/eval_degradation.py:17-26`；它们在 `utils/eval_degradation.py:119-134` 先做 CLIP 反归一化，在 `[0,1]` 像素空间处理，再重新 CLIP 归一化。亮度实现见 `utils/eval_degradation.py:137-146`；低分辨率见 `149-176`；Pillow BytesIO JPEG 见 `179-217`。

### 4.2.3 随机源与复现条件

`--eval_degrade_seed` 仅用于 noise。`utils/metrics.py:219-254`：

```python
def _get_eval_degrade_generator(self, device, seed):
    key = str(device)
    if key not in self._eval_degrade_generators:
        try:
            generator = torch.Generator(device=device)
        except (TypeError, RuntimeError):
            device_type = getattr(device, "type", str(device).split(":")[0])
            generator = torch.Generator(device=device_type)
        self._eval_degrade_generators[key] = generator.manual_seed(int(seed))
    return self._eval_degrade_generators[key]

def _add_eval_noise(self, tensor, strength, seed):
    if tensor is None:
        return None
    if float(strength) <= 0.0:
        return tensor
    generator = self._get_eval_degrade_generator(tensor.device, seed)
    noise = torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype,
        generator=generator,
    )
    noisy = tensor + float(strength) * noise
    return self._clamp_like_eval_image(noisy, tensor)
```

每个新 `Evaluator` 的 cache 为空；首次按 device 建 generator 并 seed。相同 device、loader 顺序、batch 划分、mode 与调用顺序时，同一 seed 产生相同抽样流。`both_noise` 让 RGB、IR 连续消耗同一个 generator。cache key 只有 device，不含 seed；同一 evaluator 内改变参数 seed 不会重建已有 generator。其余评测退化不调用随机数。

### 4.2.4 训练路径

训练 MDRC 的六种类型定义于 `utils/mdrc.py:5-12`：`rgb_drop`、`ir_drop`、`rgb_noise`、`ir_noise`、`rgb_attenuate`、`ir_attenuate`。`degrade_rgbt_batch` 在 `utils/mdrc.py:76-169` 克隆输入，用全局 `torch.rand` 决定每样本 active、全局 `torch.randint` 均匀选择 mode。处理对象是 dataset transform 后、已归一化的 tensor。

独立单分支开关定义于 `utils/options.py:137-139`：

```python
# Ablation-only degradation data augmentation (single-branch; no consistency loss).
# Active only when --mdrc_enable is off; "none" keeps the baseline path unchanged.
parser.add_argument("--degrade_aug_mode", type=str, default="none", choices=["none", "mdrc_style", "moddrop"], help="ablation degradation augmentation applied to image inputs before the forward pass; 'mdrc_style' samples one of the 6 MDRC modes, 'moddrop' zeros a whole modality; disabled when --mdrc_enable is set")
```

`processor/processor.py:4947-5009` 以 batch-level 概率 0.5 用 Python `random` 选是否处理及 mode，再用 `degrade_rgbt_batch(..., prob=1.0)` 替换当前 batch 图像。`mdrc_enable=True` 时 `_degrade_aug_enabled` 为 false。训练退化没有独立 seed 参数；`train.py:21-28,55-57` 设置的全局 Python/torch 随机流控制它。

评测 `--eval_degrade_*` 只由 `test.py` 写入 args；训练参数解析器不定义这些参数，训练验证中的 `Evaluator` 经 `getattr(...,"none")` 走无评测退化路径。

## 4.3 Checkpoint 保存与加载

### 4.3.1 `Checkpointer` 的键

`utils/checkpoint.py:29-50`：

```python
def save(self, name, **kwargs):
    # Skip silently when save directory is unavailable or this rank should not write.
    if not self.save_dir:
        return

    if not self.save_to_disk:
        return

    data = {}
    # Always save model weights.
    data["model"] = self.model.state_dict()
    if self.optimizer is not None:
        # Save optimizer state for exact resume.
        data["optimizer"] = self.optimizer.state_dict()
    if self.scheduler is not None:
        # Save scheduler state to keep LR progression consistent after resume.
        data["scheduler"] = self.scheduler.state_dict()
    data.update(kwargs)

    save_file = os.path.join(self.save_dir, "{}.pth".format(name))
    self.logger.info("Saving checkpoint to {}".format(save_file))
    torch.save(data, save_file)
```

固定/条件来源为：`model <- model.state_dict()`；非空时 `optimizer <- optimizer.state_dict()`、`scheduler <- scheduler.state_dict()`；其余键来自调用方 `kwargs`。训练入口 `train.py:179-186` 把 model、optimizer、scheduler 一同传给 Checkpointer。训练附加字典在 `processor/processor.py:5023-5028` 定义为：

```python
log_period = args.log_period
eval_period = args.eval_period
device = "cuda"
num_epoch = args.num_epoch

arguments = {"num_epoch": num_epoch, "iteration": 0, "epoch": 0}
```

普通 best、last 与 metric checkpoint 因而包含 model、optimizer、scheduler、num_epoch、iteration、epoch；源码中该 `arguments["iteration"]` 没有后续赋值。

### 4.3.2 三个保存开关与调用位置

默认值在 `utils/options.py:79-85`：

```python
parser.add_argument("--save_all_epoch_ckpt", default=False, action='store_true', help="save an extra metric-named checkpoint after every validation")
parser.add_argument("--save_metric_best_ckpt", default=False, action='store_true', help="save an extra non-overwritten metric-named checkpoint when R1 improves")
parser.add_argument("--metric_ckpt_dir", type=str, default="", help="directory for extra metric checkpoints; defaults to output_dir/metric_ckpts")
parser.add_argument("--save_full_eval_state_ckpt", default=False, action='store_true', help="save full eval-state checkpoints after validation for exact reproduction diagnostics")
parser.add_argument("--verify_ckpt_immediately_after_save", default=False, action='store_true', help="immediately re-evaluate in-memory and reloaded full-state checkpoints after validation for exact reproduction diagnostics")
parser.add_argument("--verify_ckpt_epochs", type=str, default="", help="comma-separated 1-based epochs for immediate checkpoint verification, e.g. '9,13'; empty means all validation epochs")
parser.add_argument("--save_last_checkpoint", default=False, action='store_true', help="save one additional last.pth after the final training epoch")
```

- `save_all_epoch_ckpt`：`processor/processor.py:6558-6568` 在每个实际验证 epoch 调 `_save_metric_checkpoint(..., reason="epoch")`。
- `save_metric_best_ckpt`：`processor/processor.py:6580-6611` 只在 `best_top1 < top1` 时额外保存 metric-named best checkpoint。
- `save_last_checkpoint`：`processor/processor.py:355-369,6633-6639` 在所有 epoch 完成后写 `last.pth`；若文件已存在则拒绝覆盖。该开关还控制 R1 改善时是否向 checkpoint-selection TSV 追加 best 记录。

`best.pth` 自身不受这三个开关控制。`processor/processor.py:6580-6583`：

```python
if best_top1 < top1:
    best_top1 = top1
    arguments["epoch"] = epoch
    checkpointer.save("best", **arguments)
```

训练恢复/仅权重载入在 `train.py:189-203`：training resume 调 `resume()` 并读取 epoch，测试模式 resume 与 `load_ckpt_file` 调 `load()`。独立测试默认读 `<output_dir>/best.pth`，也可由 `test.py:959-982` 的 `--test_ckpt_file` 指定。

### 4.3.3 只含 `model` 段的 checkpoint

`utils/checkpoint.py:52-61,81-86`：

```python
def load(self, f=None):
    # Load model-only checkpoint for inference/evaluation.
    if not f:
        # no checkpoint could be found
        self.logger.info("No checkpoint found.")
        return {}
    self.logger.info("Loading checkpoint from {}".format(f))
    checkpoint = self._load_file(f)
    self._load_model(checkpoint)
    return checkpoint

def _load_file(self, f):
    # Always load checkpoint to CPU first to avoid GPU memory spikes.
    return torch.load(f, map_location=torch.device("cpu"))

def _load_model(self, checkpoint, except_keys=None):
    load_state_dict(self.model, checkpoint.pop("model"), except_keys)
```

因此 `{"model": model.state_dict()}` 可以由 `test.py` 现有路径加载；optimizer、scheduler、epoch 不是 `load()` 必需键。测试端由 `Checkpointer(model)` 构造，不持有 optimizer/scheduler；没有 epoch 时，`test.py:982` 以 `1` 为 `best_epoch` 默认值。若文件是直接保存的裸 `state_dict`、没有顶层 `model` 键，则 `utils/checkpoint.py:86` 的 `pop("model")` 会失败；该格式与“保留 model 段的 checkpoint 包装”不同。

## 4.4 数据集注册与扩展点

### 4.4.1 注册映射

`datasets/build.py:22-33`：

```python
from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid
from .rgbtpedes import RGBTPEDES
from .mga_pseudo import collect_mga_pseudo_stats, load_mga_pseudo_json

__factory = {
    'CUHK-PEDES': CUHKPEDES,
    'ICFG-PEDES': ICFGPEDES,
    'RSTPReid': RSTPReid,
    'RGBT-PEDES': RGBTPEDES,
}
```

`datasets/build.py:445-451` 以直接字典索引实例化：

```python
def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    # Instantiate dataset by name from factory.
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)
```

`utils/options.py:519-527` 的 `--dataset_name` 没有 `choices=` 限制。

### 4.4.2 `RGBTPEDES` 路径、split 与产出

初始化顺序见 `datasets/rgbtpedes.py:17-37`：拼接 `<root>/RGBT-PEDES`，解析 data dir、RGB/T dir 与标注路径，检查存在性，拆分 train/test/val，然后分别调用 `_process_anno`。

数据目录与标注候选顺序见 `datasets/rgbtpedes.py:43-65`：

```python
def _resolve_data_dir(self, dataset_dir: str) -> str:
    # Support both layouts:
    #   <root>/RGBT-PEDES/RGBT-PEDES/...
    #   <root>/RGBT-PEDES/RGBT-PEDES-Final/...
    # and the flat one:
    #   <root>/RGBT-PEDES/...
    candidates = [
        op.join(dataset_dir, "RGBT-PEDES-Final"),
        op.join(dataset_dir, "RGBT-PEDES"),
        dataset_dir,
    ]
    for cand in candidates:
        if op.isdir(op.join(cand, "RGB")) and op.isdir(op.join(cand, "T")):
            return cand
    return dataset_dir

def _resolve_anno_path(self, data_dir: str) -> str:
    for fname in ("Caption.json", "caption.json"):
        path = op.join(data_dir, fname)
        if op.exists(path):
            return path
    # Return default path for clearer downstream error message.
    return op.join(data_dir, "Caption.json")
```

顺序为 `RGBT-PEDES-Final`、嵌套 `RGBT-PEDES`、平铺目录；标注先 `Caption.json` 后 `caption.json`。

split 解析见 `datasets/rgbtpedes.py:67-82`：

```python
def _split_anno(self, anno_path: str):
    train_annos, test_annos, val_annos = [], [], []
    annos = read_json(anno_path)
    for anno in annos:
        split = anno.get("split", "train")
        if split == "train":
            train_annos.append(anno)
        elif split == "test":
            test_annos.append(anno)
        else:
            val_annos.append(anno)

    # This dataset usually has train/test only.
    if len(val_annos) == 0:
        val_annos = test_annos
    return train_annos, test_annos, val_annos
```

缺失 split 默认 train；精确为 train/test 时进入对应集合；其他值进入 val；空 val 复用 test。

路径配对见 `datasets/rgbtpedes.py:84-88`：

```python
def _pair_paths(self, rel_path: str) -> Tuple[str, str]:
    name = op.basename(rel_path)
    rgb_path = op.join(self.rgb_dir, name)
    t_path = op.join(self.t_dir, name)
    return rgb_path, t_path
```

只取 file_path basename。

身份重编号与训练产出见 `datasets/rgbtpedes.py:90-109`：

```python
def _process_anno(self, annos: List[dict], training=False):
    # Re-index pids to contiguous labels inside each split.
    raw_pids = sorted({int(anno["id"]) for anno in annos})
    pid2label = {pid: idx for idx, pid in enumerate(raw_pids)}

    pid_container = set()
    if training:
        dataset = []
        image_id = 0
        for anno in annos:
            pid = pid2label[int(anno["id"])]
            pid_container.add(pid)
            rgb_path, t_path = self._pair_paths(anno["file_path"])

            for caption in anno["captions"]:
                # (pid, image_id, rgb_path, thermal_path, caption)
                dataset.append((pid, image_id, rgb_path, t_path, caption))
            image_id += 1

        return dataset, pid_container
```

每个 split 独立对原始 id 排序并重编号。训练每条 annotation 递增一次 image_id，每个 caption 生成一个五元组。

评测产出见 `datasets/rgbtpedes.py:111-136`：

```python
dataset = {}
img_paths = []
img_paths_t = []
captions = []
image_pids = []
caption_pids = []
for anno in annos:
    pid = pid2label[int(anno["id"])]
    pid_container.add(pid)
    rgb_path, t_path = self._pair_paths(anno["file_path"])

    image_pids.append(pid)
    img_paths.append(rgb_path)
    img_paths_t.append(t_path)
    for caption in anno["captions"]:
        captions.append(caption)
        caption_pids.append(pid)

dataset = {
    "image_pids": image_pids,
    "img_paths": img_paths,
    "img_paths_t": img_paths_t,
    "caption_pids": caption_pids,
    "captions": captions,
}
return dataset, pid_container
```

每条 annotation 在 gallery 加一个 RGB/T 对；每个 caption 在 query 加一行。`datasets/build.py:564-585,597-625` 分别为训练验证和纯测试建立独立 image gallery 与 text query loader，均不 shuffle。

### 4.4.3 RGBNT201-PEDES 的最小位置清单

给定 `<root>/RGBNT201-PEDES/RGB/`、`T/`、`captionform0_8B.json`，以及严格包含 `split/captions/file_path/id/processed_tokens` 且另含 `rgb_captions/tir_captions` 的 schema，功能性最少位置为：

1. `datasets/rgbtpedes.py`（现有类结束于第 148 行）：增加继承 `RGBTPEDES` 的数据集类，令 `dataset_dir="RGBNT201-PEDES"`，覆盖 `_resolve_anno_path()` 以选择 `captionform0_8B.json`。继承的 `_resolve_data_dir()` 会选平铺候选；继承的 `_pair_paths()` 会把 `RGBT/000220_cam1_0_18.jpg` 的 basename 映射到 `RGB/` 与 `T/`；继承的处理逻辑只读取前述公共字段，多余字段不参与输出。
2. `datasets/build.py:25,28-33`：导入该类，并在 `__factory` 增加 `dataset_name` 映射。实际实例化点仍是 `datasets/build.py:450`。

`utils/options.py:521` 没有 choices，因而参数解析本身不拒绝该新字符串。

### 4.4.4 空 caption

`datasets/rgbtpedes.py:104-107,125-127` 对 caption 无条件 append，不 strip、不跳过，所以 3 条空 caption 会保留在训练样本中。

`datasets/bases.py:46-62`：

```python
def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    # Add special start/end tokens and pad/truncate to fixed length.
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result
```

`SimpleTokenizer.encode("")` 返回空列表，故主 caption token 为 SOT、EOT、其余补零。若启用 MLM，`datasets/bases.py:793-825` 的 all-labels-zero 分支会把索引 1 的 EOT 写为 label，再把索引 1 替换为 mask。

## 4.5 随机种子与优化器参数组

### 4.5.1 训练随机种子

训练不读取 argparse 的 `args.seed`；主程序读取环境变量 `TRAIN_SEED`，默认字符串为 `"1"`。`train.py:55-65`：

```python
if __name__ == "__main__":
    args = get_args()
    set_seed(int(os.environ.get("TRAIN_SEED", "1")) + get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
```

`train.py:21-28`：

```python
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
```

它覆盖 PyTorch、当前 CUDA device、全部 CUDA devices、NumPy、Python `random`，并设置两个 cuDNN 标志。`set_seed(...+get_rank())` 在 `init_process_group()` 之前调用；`utils/comm.py:21-27` 在 distributed 尚未初始化时返回 rank 0，所以该处各启动进程得到相同 `TRAIN_SEED` 值。

### 4.5.2 普通参数组规则

`solver/build.py:50-60`：

```python
def _legacy_parameter_options(args, key):
    lr = args.lr
    weight_decay = args.weight_decay
    if "cross" in key:
        lr = args.lr * args.lr_factor
    if "bias" in key:
        lr = args.lr * args.bias_lr_factor
        weight_decay = args.weight_decay_bias
    if "classifier" in key or "mlm_head" in key:
        lr = args.lr * args.lr_factor
    return float(lr), float(weight_decay)
```

执行顺序为 cross、bias、classifier/mlm_head，后续赋值可覆盖先前的 lr。`requires_grad=False` 的参数不进入优化器。

### 4.5.3 MRAF scorer 分组与异常条件

评分器按模块对象身份定位。`solver/build.py:25-47`：

```python
def _model_core(model):
    return model.module if hasattr(model, "module") else model


def get_mraf_scorer_named_parameters(model):
    """Locate scorer parameters by module identity, including DP/DDP wrappers."""
    core = _model_core(model)
    fusion = getattr(core, "mraf_fusion", None)
    scorer = getattr(fusion, "scorer", None)
    if scorer is None:
        return []
    names_by_id = {
        id(parameter): name for name, parameter in core.named_parameters()
    }
    result = []
    for parameter in scorer.parameters():
        name = names_by_id.get(id(parameter))
        if name is None:
            raise RuntimeError(
                "An MRAF scorer parameter is not registered on the model."
            )
        result.append((name, parameter))
    return result
```

`solver/build.py:148-196` 对非有限或 `<=0` 的 multiplier 抛 `ValueError`；multiplier 不为 1 但模型无 MRAF scorer 时也抛 `ValueError`。multiplier 不为 1 时，scorer 依据 legacy weight decay 分为 `mraf_scorer_bias` / `mraf_scorer_weight`，二者 lr 都是 `args.lr * multiplier`；其他参数仍各自形成 legacy group。`solver/build.py:63-77` 对重复参数及未恰好覆盖所有 requires-grad 参数分别抛 `RuntimeError`。

优化器选择位于 `solver/build.py:198-218`：SGD 使用 momentum；Adam 使用 `(alpha,beta)` 与 `eps=1e-3`；AdamW 使用 `(alpha,beta)` 与 `eps=1e-8`；其他字符串抛 `NotImplementedError`。

## 4.6 评测口径

### 4.6.1 query、gallery 与相似度矩阵

纯测试在 `datasets/build.py:597-625` 按 `eval_dataset` 选择 val/test；gallery 为 `ImageDatasetRGBT` 或 `ImageDataset`，query 为 `TextDataset`，两侧 loader 都 `shuffle=False`。`utils/metrics.py:80-83` 将 `img_loader` 标为 gallery、`txt_loader` 标为 query。

`Evaluator._compute_embedding` 按 loader 顺序拼接特征与 pid。最终相似度见 `utils/metrics.py:1735-1753`：

```python
# 2) L2 normalize before cosine similarity.
qfeats = F.normalize(qfeats, p=2, dim=1) # text features
gfeats = F.normalize(gfeats, p=2, dim=1) # image features

# 3) Text-to-image similarity matrix.
similarity = qfeats @ gfeats.t()
if use_mtp_eval_fusion:
    alpha = float(getattr(model.args, "mtp_eval_fusion_alpha", 0.10))
    beta = float(getattr(model.args, "mtp_eval_fusion_beta", 0.00))
    qfeats_rgb = F.normalize(embedding_outputs["qfeats_rgb"], p=2, dim=1)
    qfeats_th = F.normalize(embedding_outputs["qfeats_th"], p=2, dim=1)
    gfeats_rgb = F.normalize(embedding_outputs["gfeats_rgb"], p=2, dim=1)
    gfeats_th = F.normalize(embedding_outputs["gfeats_th"], p=2, dim=1)
    similarity = (
        similarity
        + alpha * (qfeats_rgb @ gfeats_rgb.t())
        + beta * (qfeats_th @ gfeats_th.t())
    )
baseline_similarity = similarity.clone()
```

矩阵形状是 `[文本 query 数, 图像或 RGB/T 图像对 gallery 数]`，值越大越靠前。正样本判据是 caption pid 与 image pid 相等。可选 i2t 使用转置矩阵。

### 4.6.2 指标

R1/R5/R10、mAP、mINP 统一由 `utils/metrics.py:40-77` 计算：

```python
def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    # similarity: [num_query, num_gallery], larger means more similar.
    if get_mAP:
        # Full sort is needed when mAP/mINP are required.
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # For CMC only, top-k is faster than full sort.
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    # Predicted gallery pid for each query rank position.
    pred_labels = g_pids[indices.cpu()]  # q * k
    # Binary match matrix: True if predicted pid equals query pid.
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    # Compute CMC curve.
    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    # mINP considers the rank position of the last relevant match.
    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    # mAP is mean of per-query average precision.
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices
```

`utils/metrics.py:2358-2381` 以 `max_rank=10` 调用；R1/R5/R10 分别是 `cmc[0]`、`cmc[4]`、`cmc[9]`。

### 4.6.3 相似度 dump

CLI 位于 `test.py:275-291`：`--save_eval_scores` 默认 false，`--eval_scores_file` 默认空，`--eval_scores_save_raw` 默认 false。保存函数为 `utils/metrics.py:748-860`，训练验证来源会跳过；独立测试保存以下 payload，见 `utils/metrics.py:780-846`：

```python
sim_cpu = similarity.detach().cpu().float()
txt_pids = qids.detach().cpu().long().view(-1)
img_pids = gids.detach().cpu().long().view(-1)
order_hash = self._build_eval_order_hash(txt_pids, img_pids, sim_cpu)
score_semantics = self._score_semantics(
    use_qalr=use_qalr,
    use_qalr_v2=use_qalr_v2,
    use_qalr_v3_gate=use_qalr_v3_gate,
    use_qalr_v3_two_stage_gate=use_qalr_v3_two_stage_gate,
    use_transport_rerank=use_transport_rerank,
    use_uspc_slot_rerank=use_uspc_slot_rerank,
    use_mtp_eval_fusion=use_mtp_eval_fusion,
)
payload = {
    "sim": sim_cpu,
    "txt_pids": txt_pids,
    "img_pids": img_pids,
    "txt_indices": torch.arange(sim_cpu.shape[0], dtype=torch.long),
    "img_indices": torch.arange(sim_cpu.shape[1], dtype=torch.long),
    "metrics": dict(metrics),
    "metadata": metadata,
    "order_hash": order_hash,
}
if bool(getattr(model_args, "eval_scores_save_raw", False)):
    if raw_similarity is None:
        raise RuntimeError(
            "--eval_scores_save_raw was set, but raw similarity is unavailable."
        )
    payload["raw_sim"] = raw_similarity.detach().cpu().float()
```

`sim` 是所有已启用 rerank 后实际计算指标的矩阵；可选 `raw_sim` 是第 4.6.1 节 clone 的 `baseline_similarity`。metadata 在 `utils/metrics.py:793-830` 记录数据集、split、shape、score semantics、各 rerank 开关与参数、配置/checkpoint/模型/output、评测来源、退化参数、order hash、raw 是否保存。

### 4.6.4 fp16 路径

`model/build.py:2516-2523` 构建模型时无条件 `convert_weights(model)`；`model/clip_model.py:673-697` 把 Conv、Linear、MultiheadAttention 投影及若干 projection 参数转 half。编码器权重与 activation 因此存在 fp16 路径。

进入相似度的全局特征显式转 float32：视觉 CLS 在 `model/build.py:1469-1475,1529-1540` 使用 `.float()`，文字全局在 `model/build.py:1542-1547,1610-1617` 使用 `.float()`；视觉 local-token 路径在 `model/build.py:1355-1358` 转 float。`utils/metrics.py:1735-1740` 随后在这些 float32 特征上做归一化与矩阵乘，`rank` 内没有 `.half()` 或 autocast。故编码器前向含 fp16，最终相似度、排序及五项指标使用 float32。
