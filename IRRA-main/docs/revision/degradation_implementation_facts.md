# Evaluation degradation implementation facts

## Audit scope

- Audited branch: `paper_revision_20260710`
- Audited baseline: `87610d503a8a`
- Audit type: static code inspection plus synthetic-tensor unit tests. No real RGBT-PEDES batch or checkpoint was loaded locally.
- Line numbers below refer to the modified working tree in this revision.

## Source locations and implementation facts

| File and exact line(s) | Function/class | Implementation fact |
| --- | --- | --- |
| `utils/iotools.py:18`, `utils/iotools.py:27` | `read_image` | Every image is opened with Pillow and converted with `convert('RGB')`. This applies to both RGB and thermal paths, so thermal input is replicated/converted to three channels at read time. |
| `datasets/bases.py:588`, `datasets/bases.py:598`, `datasets/bases.py:604-609` | `ImageDatasetRGBT.__getitem__` | Paired RGB and thermal paths are read independently, passed through the same evaluation transform, and returned as `(pid, rgb_img, t_img)`. |
| `datasets/build.py:36-51` | `build_transforms` | The default evaluation pipeline is exactly `Resize((height,width)) -> ToTensor() -> Normalize(mean,std)`. CLIP mean is `[0.48145466, 0.4578275, 0.40821073]`; std is `[0.26862954, 0.26130258, 0.27577711]`. |
| `datasets/build.py:594`, `datasets/build.py:608-609` | `build_dataloader` | The normal test path builds the default evaluation transform and supplies it to `ImageDatasetRGBT`. A caller-provided custom `tranforms` argument can replace this pipeline; the facts in this document concern the normal `test.py` path. |
| `utils/metrics.py:381-399` | `Evaluator._compute_embedding` | A standard RGBT batch contains `(pid, img, thermal_img)`. Both image tensors are moved to the model device at lines 387-388, degraded at line 393 before model encoding, and forwarded at line 399. |
| `utils/metrics.py:219-228` | `Evaluator._get_eval_degrade_generator` | One `torch.Generator` is cached per device and seeded with `eval_degrade_seed` on first use. Reproducibility therefore assumes a fresh `Evaluator`, identical device, dataloader order, batching, mode and call sequence. In `both_noise`, RGB and IR consume successive draws from the same generator rather than sharing identical noise. |
| `utils/metrics.py:230-254` | `_clamp_like_eval_image`, `_add_eval_noise` | Legacy noise is sampled with the input tensor's shape/device/dtype in CLIP-normalized space. `strength` is the Gaussian standard deviation in that normalized space. The result is clamped per sample to that sample's original minimum and maximum across all non-batch dimensions; it is not clamped to pixel `[0,1]`. |
| `utils/metrics.py:256-272` | `Evaluator._blur_eval_image` | Legacy blur is not Gaussian blur. It uses asymmetric-when-needed replicate padding followed by `F.avg_pool2d(kernel_size=kernel_size, stride=1)`. There is no Gaussian sigma. Padding totals `kernel_size-1` on each axis and uses `mode="replicate"`, preserving spatial size; `kernel_size <= 1` returns the input. |
| `utils/metrics.py:274-312` | `Evaluator._apply_eval_degradation` | Legacy `none`, missing, noise and blur remain in their original normalized-space branch. Only the six new modes are delegated to the new helper at lines 303-309. |
| `utils/eval_degradation.py:14-15`, `utils/eval_degradation.py:119-135` | constants, `clip_denormalize`, `clip_normalize` | The helper uses the same official CLIP mean/std as `datasets/build.py`. Conversion is out-of-place and preserves the input-compatible dtype/device. |
| `utils/eval_degradation.py:137-147` | `apply_brightness_degradation` | The tensor is CLIP-denormalized, clamped to `[0,1]`, multiplied by a validated `0 < factor <= 1`, clamped again, and CLIP-normalized. |
| `utils/eval_degradation.py:149-177` | `apply_lowres_degradation` | The tensor is denormalized and clamped in pixel space. Downsample and upsample both use bilinear interpolation with `align_corners=False`. Downsample explicitly uses `antialias=True`; upsample explicitly uses `antialias=False`. Low size is `max(1, round(size*scale))`, and the restored output has the original height/width. |
| `utils/eval_degradation.py:179-218` | `apply_jpeg_degradation` | Pixel values are clamped, quantized to uint8, encoded and decoded per batch item using in-memory `io.BytesIO`, then converted back to three-channel RGB, float pixel values and CLIP-normalized tensors on the original device/dtype. No temporary image is written. |
| `utils/eval_degradation.py:31-35`, `utils/eval_degradation.py:194-201` | JPEG constants and save call | Backend is Pillow `BytesIO`; current local Pillow version is recorded at import time. JPEG settings are explicit: `quality` from strength, `subsampling=0` (4:4:4), `optimize=False`, `progressive=False`. |
| `utils/eval_degradation.py:220-260` | `apply_new_eval_degradation` | Dispatch covers exactly six new modes plus a no-op `none` used by tests. Only the selected modality is replaced; the other modality is returned unchanged. |
| `model/build.py:1023-1075`, `model/build.py:1688-1744` | `_fuse_rgbt_global`, `encode_rgbt_global` | There is no uniform per-modality L2-normalization step immediately before RGB/IR addition or precision fusion. Some optional local-filter paths can normalize their own outputs, but this is not a universal pre-fusion operation. |
| `utils/metrics.py:1735-1737` | `Evaluator.eval` | The final text and already-fused image embeddings are L2-normalized before cosine similarity. Separate modality features are normalized only in the optional MTP evaluation-fusion scoring path. |
| `utils/metrics.py:552-559` | `_build_eval_order_hash` | `order_hash` currently covers only `txt_pids`, `img_pids` and score shape. Its algorithm was not changed. It does not cover score values, `txt_indices` or `img_indices`. |
| `utils/metrics.py:793-840` | `_maybe_save_eval_scores` | Existing metadata fields remain. Four additive fields now record mode, strength, blur kernel and seed. Payload keys and raw-score behavior remain unchanged. |

## Required factual answers

1. **How RGB and IR are read:** `ImageDatasetRGBT.__getitem__` reads paired paths with the same `read_image` helper and applies the same transform to each.
2. **How IR becomes three-channel:** `read_image` always calls Pillow `convert('RGB')`; there is no later one-channel thermal tensor on the standard path.
3. **Preprocessing order:** default evaluation is `Resize -> ToTensor -> CLIP Normalize`.
4. **Where legacy degradation occurs:** after Normalize and after transfer to the model device, immediately before image model forward.
5. **Physical meaning of legacy missing:** `torch.zeros_like` creates zero in normalized coordinates. Denormalizing it produces the CLIP channel means, not raw pixel black (`[0,0,0]`), so legacy missing is not physically equivalent to a black source image.
6. **Legacy noise strength:** Gaussian standard deviation in CLIP-normalized coordinates, not pixel intensity units.
7. **Legacy noise clamp:** yes, but to each original normalized tensor's per-sample min/max, not to `[0,1]`.
8. **Legacy blur:** replicate padding plus average pooling; kernel is `eval_degrade_blur_kernel`, stride is 1, sigma does not exist, and replicate boundary handling is used.
9. **Noise reproducibility:** a device-specific generator is seeded once from `eval_degrade_seed`; identical fresh evaluation setup and call order reproduce the same stream.
10. **Fusion normalization:** RGB and IR are not uniformly L2-normalized separately immediately before standard fusion; the fused gallery embedding is normalized later before cosine similarity.

## New degradation space and limitations

All brightness, low-resolution and JPEG operations first invert CLIP normalization, clamp to pixel `[0,1]`, apply the degradation, clamp as appropriate, and reapply the same CLIP mean/std. They never call the legacy noise generator and never alter training code.

JPEG is deterministic for identical input/settings in the same Pillow/libjpeg environment. Different Pillow or linked libjpeg versions can produce different encoded bytes and decoded pixels even with the same explicit settings; cross-environment pixel identity is therefore not guaranteed.

For old-score regression, whole-file SHA-256 or byte identity is not a valid criterion because metadata is intentionally additive. Compare `sim`, `raw_sim`, `txt_pids`, `img_pids`, `txt_indices`, `img_indices` and `order_hash` independently. Legacy `sim` and `raw_sim` are expected to be element-wise identical. The PID/index comparisons compensate for the limited coverage of the current `order_hash`.
