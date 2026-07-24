# MRAF implementation

## Module location

The core module is `model/mraf.py`:

- `ModalityReliabilityAdaptiveFusion`
- `compute_mraf_reliability_targets`
- `compute_mraf_reliability_loss`
- `resolve_mraf_rel_lambda`

`model.build.IRRA` constructs `self.mraf_fusion` only when
`mraf_enabled=true`. Consequently, an MRAF-disabled model has no MRAF
parameters or MRAF keys in its `state_dict`.

## Reliability gate and fusion

For RGB and IR global features `f_rgb, f_ir ∈ R^(B×D)`, the same scorer is
applied to both L2-normalized inputs:

```text
scorer(x) = Linear(D, H) → GELU → Dropout → Linear(H, 1)
logits = concat(scorer(normalize(f_rgb)), scorer(normalize(f_ir)))
alpha = softmax(logits / gate_temperature)
```

The fused feature is:

```text
f_fused = 2 * (alpha_rgb * f_rgb + alpha_ir * f_ir)
```

The factor 2 preserves the original addition scale. The last scorer linear
layer has zero weight and zero bias, so initialization gives:

```text
alpha_rgb = alpha_ir = 0.5
f_fused = f_rgb + f_ir
```

The feature dimension comes from the CLIP model configuration and is never
hard-coded to 512. Inputs are checked for rank, batch/feature dimensions,
device, dtype, and finite values. Both gate and target temperatures reject
non-positive values.

## MRAF versus precision fusion

- `mraf_enabled=false`: the old addition/precision selection is unchanged.
- `mraf_enabled=true`: MRAF supplies the global fused feature and the
  deterministic addition/precision-fusion selection is bypassed.
- Enabling MRAF together with effective uncertainty precision fusion raises:

  ```text
  MRAF and uncertainty precision fusion are mutually exclusive in the current implementation.
  ```

Global Gaussian heads may only coexist with MRAF when precision fusion is
explicitly disabled. This does not combine the two fusion rules.

## Reliability target

The shared clean/degraded branch exposes single-modality features. Drift is:

```text
d_rgb = 1 - cosine(degraded_rgb, stop_gradient(clean_rgb))
d_ir  = 1 - cosine(degraded_ir,  stop_gradient(clean_ir))
q = stop_gradient(softmax(stack(-d_rgb, -d_ir) / target_temperature))
```

The degraded gate remains attached:

```text
loss_mraf_rel = mean((alpha_degraded - q)^2)
```

Thus `q.requires_grad` is false, the target path cannot update clean features,
and scorer parameters receive gradients through `alpha_degraded`.

## Loss and warmup

MRAF has an independent linear warmup. For zero-based epoch `e`:

```text
lambda_rel(e) =
    base_lambda * (e + 1) / warmup_epoch, e < warmup_epoch
    base_lambda,                         otherwise
```

`do_train` converts its one-based epoch to this zero-based schedule.

The combined loss is:

```text
L = L_base
  + lambda_mdrc_cur * mdrc_loss_scale * L_mdrc
  + lambda_mraf_rel_cur * L_mraf_rel
```

`mdrc_loss_scale` never multiplies MRAF reliability loss.

## Shared degraded branch

`processor.processor._build_degraded_branch_context` calls the existing MDRC
sampler once and the image-only model forward once. The context is shared by
MDRC and MRAF. `utils.mdrc.degrade_rgbt_batch(return_mode=True)` derives mode
metadata from the already sampled tensors and performs no extra random draw.
The default `return_mode=False` two-value API is unchanged.

The logical branch condition is:

```text
mdrc_enable or (mraf_enabled and mraf_rel_lambda > 0)
```

Execution also respects the existing MDRC effective warmup so MDRC-only does
not acquire a new early second forward or changed RNG state.

## Supported modes

| Mode | Clean fusion | Degraded branch | MDRC loss | MRAF reliability |
|---|---|---:|---:|---:|
| Base | addition | no | no | no |
| MDRC-only | existing fusion | existing MDRC schedule | yes | no |
| MRAF-only | MRAF | yes | no | yes |
| MDRC+MRAF | MRAF | one shared branch | yes | yes |
| MRAF no-rel | MRAF | only if MDRC needs it | optional | no |

SDM, MLM, and ID remain clean-branch losses.

## Inference path

Evaluation performs one RGB/IR visual encoding and applies MRAF to the current
global features. It does not construct a degraded branch, target, or
reliability loss. The existing text encoding, feature normalization, similarity
matrix, metrics, and reranking logic are unchanged.

Rich evaluation output includes MRAF RGB/IR weights for the analysis tool.

## Checkpoint compatibility

- MRAF disabled: no MRAF module/state keys; historical Base/MDRC behavior uses
  the old loader path.
- New MRAF checkpoint: all MRAF and non-MRAF keys must match strictly.
- Old checkpoint loaded explicitly into an MRAF model: only a complete absence
  of MRAF keys is accepted. Missing MRAF keys are logged and retain neutral
  initialization.
- Any non-MRAF missing key, unexpected key, or partially present MRAF state
  raises an error.
- No global `strict=False` is used.
