"""Evaluation-only pixel-space degradations for RGB-T retrieval."""

from io import BytesIO
import math

import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn.functional as F


# Source: datasets/build.py::build_transforms at baseline commit 87610d50.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

NEW_EVAL_DEGRADATION_MODES = frozenset(
    {
        "rgb_brightness",
        "ir_brightness",
        "rgb_lowres",
        "ir_lowres",
        "rgb_jpeg",
        "ir_jpeg",
    }
)

LOWRES_DOWNSAMPLE_ANTIALIAS = True
LOWRES_UPSAMPLE_ANTIALIAS = False

JPEG_BACKEND = "Pillow BytesIO"
JPEG_PILLOW_VERSION = PIL.__version__
JPEG_SUBSAMPLING = 0  # 4:4:4 chroma sampling.
JPEG_OPTIMIZE = False
JPEG_PROGRESSIVE = False


def _validate_image_tensor(tensor, mode, tensor_name="image"):
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"{mode}: invalid {tensor_name}={type(tensor).__name__}; "
            "expected a torch.Tensor with shape [B,3,H,W]."
        )
    if tensor.dim() != 4:
        raise ValueError(
            f"{mode}: invalid {tensor_name}.shape={tuple(tensor.shape)}; "
            "expected [B,3,H,W]."
        )
    if tensor.shape[1] != len(CLIP_MEAN):
        raise ValueError(
            f"{mode}: invalid {tensor_name}.shape={tuple(tensor.shape)}; "
            f"expected {len(CLIP_MEAN)} channels for CLIP normalization."
        )
    if tensor.shape[0] <= 0 or tensor.shape[2] <= 0 or tensor.shape[3] <= 0:
        raise ValueError(
            f"{mode}: invalid {tensor_name}.shape={tuple(tensor.shape)}; "
            "batch, height and width must be positive."
        )
    if not tensor.is_floating_point():
        raise TypeError(
            f"{mode}: invalid {tensor_name}.dtype={tensor.dtype}; "
            "expected a floating-point CLIP-normalized tensor."
        )


def _working_dtype(tensor):
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return tensor.dtype


def _clip_stats(reference):
    mean = reference.new_tensor(CLIP_MEAN).view(1, -1, 1, 1)
    std = reference.new_tensor(CLIP_STD).view(1, -1, 1, 1)
    return mean, std


def _validate_unit_strength(value, mode, parameter_name):
    if isinstance(value, bool):
        raise ValueError(
            f"{mode}: invalid {parameter_name}={value!r}; expected 0 < value <= 1."
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{mode}: invalid {parameter_name}={value!r}; expected 0 < value <= 1."
        ) from exc
    if not math.isfinite(parsed) or not (0.0 < parsed <= 1.0):
        raise ValueError(
            f"{mode}: invalid {parameter_name}={value!r}; expected 0 < value <= 1."
        )
    return parsed


def _validate_jpeg_quality(value, mode):
    if isinstance(value, bool):
        raise ValueError(
            f"{mode}: invalid quality={value!r}; expected an integer in [1, 100]."
        )
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{mode}: invalid quality={value!r}; expected an integer in [1, 100]."
        ) from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(
            f"{mode}: invalid quality={value!r}; expected an integer in [1, 100]."
        )
    quality = int(parsed)
    if not (1 <= quality <= 100):
        raise ValueError(
            f"{mode}: invalid quality={value!r}; expected an integer in [1, 100]."
        )
    return quality


def clip_denormalize(tensor, mode="clip_denormalize"):
    """Convert a CLIP-normalized [B,3,H,W] tensor back to pixel values."""
    _validate_image_tensor(tensor, mode)
    original_dtype = tensor.dtype
    working = tensor.to(dtype=_working_dtype(tensor))
    mean, std = _clip_stats(working)
    return (working * std + mean).to(dtype=original_dtype)


def clip_normalize(tensor, mode="clip_normalize"):
    """Apply the project's official CLIP mean/std to a pixel-space tensor."""
    _validate_image_tensor(tensor, mode)
    original_dtype = tensor.dtype
    working = tensor.to(dtype=_working_dtype(tensor))
    mean, std = _clip_stats(working)
    return ((working - mean) / std).to(dtype=original_dtype)


def apply_brightness_degradation(tensor, factor, mode="brightness"):
    """Reduce brightness in clamped [0,1] pixel space and re-normalize."""
    _validate_image_tensor(tensor, mode)
    factor = _validate_unit_strength(factor, mode, "factor")
    original_dtype = tensor.dtype
    pixels = clip_denormalize(tensor, mode=mode).to(dtype=_working_dtype(tensor))
    degraded = (pixels.clamp(0.0, 1.0) * factor).clamp(0.0, 1.0)
    return clip_normalize(degraded, mode=mode).to(
        device=tensor.device, dtype=original_dtype
    )


def apply_lowres_degradation(tensor, scale, mode="lowres"):
    """Downsample and restore an image in pixel space using bilinear resize."""
    _validate_image_tensor(tensor, mode)
    scale = _validate_unit_strength(scale, mode, "scale")
    original_dtype = tensor.dtype
    height, width = tensor.shape[-2:]
    low_height = max(1, round(height * scale))
    low_width = max(1, round(width * scale))

    pixels = clip_denormalize(tensor, mode=mode).to(dtype=_working_dtype(tensor))
    pixels = pixels.clamp(0.0, 1.0)
    lowres = F.interpolate(
        pixels,
        size=(low_height, low_width),
        mode="bilinear",
        align_corners=False,
        antialias=LOWRES_DOWNSAMPLE_ANTIALIAS,
    )
    restored = F.interpolate(
        lowres,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=LOWRES_UPSAMPLE_ANTIALIAS,
    ).clamp(0.0, 1.0)
    return clip_normalize(restored, mode=mode).to(
        device=tensor.device, dtype=original_dtype
    )


def apply_jpeg_degradation(tensor, quality, mode="jpeg"):
    """Round-trip each batch item through deterministic in-memory Pillow JPEG."""
    _validate_image_tensor(tensor, mode)
    quality = _validate_jpeg_quality(quality, mode)
    original_device = tensor.device
    original_dtype = tensor.dtype

    pixels = clip_denormalize(tensor, mode=mode).to(dtype=torch.float32)
    uint8_batch = (
        pixels.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).detach().cpu()
    )
    decoded_batch = []
    for image_tensor in uint8_batch:
        image_array = image_tensor.permute(1, 2, 0).contiguous().numpy()
        source_image = Image.fromarray(image_array, mode="RGB")
        with BytesIO() as buffer:
            source_image.save(
                buffer,
                format="JPEG",
                quality=quality,
                subsampling=JPEG_SUBSAMPLING,
                optimize=JPEG_OPTIMIZE,
                progressive=JPEG_PROGRESSIVE,
            )
            buffer.seek(0)
            with Image.open(buffer) as decoded_image:
                decoded_rgb = decoded_image.convert("RGB")
                decoded_array = np.array(decoded_rgb, dtype=np.uint8, copy=True)
        decoded_batch.append(
            torch.from_numpy(decoded_array).permute(2, 0, 1).contiguous()
        )

    decoded_pixels = torch.stack(decoded_batch, dim=0).to(
        device=original_device, dtype=torch.float32
    )
    decoded_pixels = decoded_pixels.div(255.0).clamp(0.0, 1.0)
    return clip_normalize(decoded_pixels, mode=mode).to(
        device=original_device, dtype=original_dtype
    )


def apply_new_eval_degradation(mode, rgb_img, thermal_img, strength):
    """Apply one of the six new degradations to only its selected modality."""
    if not isinstance(mode, str):
        raise ValueError(
            f"invalid eval degradation mode={mode!r}; expected a mode string."
        )
    normalized_mode = mode.lower()
    if normalized_mode == "none":
        return rgb_img, thermal_img
    if normalized_mode not in NEW_EVAL_DEGRADATION_MODES:
        raise ValueError(
            f"{normalized_mode}: invalid eval degradation mode={mode!r}; "
            f"expected one of {sorted(NEW_EVAL_DEGRADATION_MODES)} or 'none'."
        )

    target_is_rgb = normalized_mode.startswith("rgb_")
    target = rgb_img if target_is_rgb else thermal_img
    target_name = "rgb_img" if target_is_rgb else "thermal_img"
    if target is None:
        raise ValueError(
            f"{normalized_mode}: invalid {target_name}=None; "
            "the selected modality tensor is required."
        )

    if normalized_mode.endswith("_brightness"):
        degraded = apply_brightness_degradation(
            target, factor=strength, mode=normalized_mode
        )
    elif normalized_mode.endswith("_lowres"):
        degraded = apply_lowres_degradation(
            target, scale=strength, mode=normalized_mode
        )
    else:
        degraded = apply_jpeg_degradation(
            target, quality=strength, mode=normalized_mode
        )

    if target_is_rgb:
        return degraded, thermal_img
    return rgb_img, degraded
