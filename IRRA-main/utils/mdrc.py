import torch
import torch.nn.functional as F

from utils.blur_ops import blur_tensor


DEFAULT_DEGRADE_MODES = (
    "rgb_drop",
    "ir_drop",
    "rgb_noise",
    "ir_noise",
    "rgb_attenuate",
    "ir_attenuate",
)
ALLOWED_DEGRADE_MODES = DEFAULT_DEGRADE_MODES + (
    "rgb_blur",
    "ir_blur",
)


def _normalize_modes(modes):
    if modes is None:
        return list(DEFAULT_DEGRADE_MODES)
    if isinstance(modes, str):
        modes = [item.strip() for item in modes.split(",")]
    normalized = []
    for mode in modes:
        mode = str(mode).strip()
        if not mode:
            continue
        if mode not in ALLOWED_DEGRADE_MODES:
            raise ValueError(
                f"Unsupported MDRC degrade mode: {mode}. "
                f"Expected one of {list(ALLOWED_DEGRADE_MODES)}."
            )
        normalized.append(mode)
    return normalized


def _batch_mask(mask, target):
    shape = [mask.shape[0]] + [1] * (target.dim() - 1)
    return mask.to(device=target.device, dtype=target.dtype).reshape(shape)


def _apply_drop(target, sample_mask):
    mask = _batch_mask(sample_mask, target)
    return (target * (1.0 - mask)).to(dtype=target.dtype)


def _apply_noise(target, sample_mask, noise_std=0.1, generator=None):
    mask = _batch_mask(sample_mask, target)
    if target.is_floating_point():
        noise = (torch.randn_like(target) if generator is None else torch.randn(
            target.shape, device=target.device, dtype=target.dtype, generator=generator
        )) * float(noise_std)
        return target + mask * noise
    work = target.float()
    noise = (torch.randn_like(work) if generator is None else torch.randn(
        work.shape, device=work.device, dtype=work.dtype, generator=generator
    )) * float(noise_std)
    return (work + mask.float() * noise).to(dtype=target.dtype)


def _apply_attenuate(target, sample_mask, min_factor=0.2, max_factor=0.6,
                     generator=None, factors=None):
    mask = _batch_mask(sample_mask, target)
    if factors is None:
        factor = torch.empty(
            target.shape[0],
            device=target.device,
            dtype=torch.float32,
        ).uniform_(float(min_factor), float(max_factor), generator=generator)
    else:
        factor = factors.to(device=target.device, dtype=torch.float32)
    factor = factor.reshape([target.shape[0]] + [1] * (target.dim() - 1))
    factor = factor.to(dtype=target.dtype)
    return (target * (1.0 - mask + mask * factor)).to(dtype=target.dtype)


def _apply_blur(target, sample_mask, kernel_size=7):
    mask = _batch_mask(sample_mask, target)
    work = target if target.is_floating_point() else target.float()
    work_mask = mask if target.is_floating_point() else mask.float()
    blurred = blur_tensor(work, kernel_size)
    mixed = (work * (1.0 - work_mask) + blurred * work_mask).to(
        dtype=target.dtype
    )
    return torch.where(mask.bool(), mixed, target)


def _apply_mode(target, sample_mask, mode, generator=None, factors=None):
    if mode.endswith("_drop"):
        return _apply_drop(target, sample_mask)
    if mode.endswith("_noise"):
        return _apply_noise(target, sample_mask, generator=generator)
    if mode.endswith("_blur"):
        return _apply_blur(target, sample_mask)
    if mode.endswith("_attenuate"):
        return _apply_attenuate(target, sample_mask, generator=generator, factors=factors)
    raise ValueError(f"Unsupported MDRC degrade mode: {mode}.")


def degrade_rgbt_batch(
    rgb_imgs,
    ir_imgs,
    prob=0.5,
    modes=None,
    return_mode=False,
):
    """
    Build a degraded RGB/T view for training-stage clean-degraded consistency.

    The input tensors are never modified in place. For each sample, one selected
    degradation mode is drawn uniformly and applied with probability ``prob``.
    Shapes, devices, and dtypes are preserved. ``return_mode=False`` keeps the
    historical two-value return exactly. Optional mode metadata is derived from
    the already sampled tensors and never performs an additional random draw.
    """
    if not torch.is_tensor(rgb_imgs):
        raise TypeError("rgb_imgs must be a torch.Tensor.")
    if rgb_imgs.dim() < 1:
        raise ValueError(f"rgb_imgs must have a batch dimension, got {rgb_imgs.shape}.")

    degraded_rgb = rgb_imgs.clone()
    degraded_ir = ir_imgs.clone() if torch.is_tensor(ir_imgs) else ir_imgs

    has_ir = torch.is_tensor(ir_imgs)
    if has_ir and ir_imgs.shape[0] != rgb_imgs.shape[0]:
        raise ValueError(
            "rgb_imgs and ir_imgs must share batch size, got "
            f"{rgb_imgs.shape[0]} and {ir_imgs.shape[0]}."
        )

    mode_list = _normalize_modes(modes)
    valid_modes = [
        mode
        for mode in mode_list
        if mode.startswith("rgb_") or (has_ir and mode.startswith("ir_"))
    ]
    prob = max(0.0, min(1.0, float(prob)))
    if prob <= 0.0 or not valid_modes:
        if return_mode:
            batch_size = int(rgb_imgs.shape[0])
            return degraded_rgb, degraded_ir, {
                "valid_modes": tuple(valid_modes),
                "active_mask": torch.zeros(
                    batch_size, device=rgb_imgs.device, dtype=torch.bool
                ),
                "mode_indices": torch.full(
                    (batch_size,), -1, device=rgb_imgs.device, dtype=torch.long
                ),
                "mode_counts": {"none": batch_size},
            }
        return degraded_rgb, degraded_ir

    batch_size = rgb_imgs.shape[0]
    active = torch.rand(batch_size, device=rgb_imgs.device).lt(prob)
    if not bool(active.any().item()):
        if return_mode:
            return degraded_rgb, degraded_ir, {
                "valid_modes": tuple(valid_modes),
                "active_mask": active,
                "mode_indices": torch.full_like(active, -1, dtype=torch.long),
                "mode_counts": {"none": int(batch_size)},
            }
        return degraded_rgb, degraded_ir

    choice = torch.randint(
        low=0,
        high=len(valid_modes),
        size=(batch_size,),
        device=rgb_imgs.device,
    )
    for mode_index, mode in enumerate(valid_modes):
        sample_mask = active & choice.eq(mode_index)
        if not bool(sample_mask.any().item()):
            continue
        if mode.startswith("rgb_"):
            degraded_rgb = _apply_mode(degraded_rgb, sample_mask, mode)
        elif has_ir:
            degraded_ir = _apply_mode(degraded_ir, sample_mask, mode)

    if return_mode:
        mode_indices = torch.where(active, choice, torch.full_like(choice, -1))
        mode_counts = {
            mode: int((active & choice.eq(mode_index)).sum().item())
            for mode_index, mode in enumerate(valid_modes)
        }
        mode_counts["none"] = int((~active).sum().item())
        return degraded_rgb, degraded_ir, {
            "valid_modes": tuple(valid_modes),
            "active_mask": active,
            "mode_indices": mode_indices,
            "mode_counts": mode_counts,
        }
    return degraded_rgb, degraded_ir


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


def mdrc_similarity_consistency(clean_img_feat, degraded_img_feat, text_feat, temp=0.07, eps=1e-12):
    """
    Match clean/degraded image-text similarity distributions.

    Clean image and text features are treated as the detached teacher. The
    degraded image feature remains attached and receives the MDRC gradient.
    """
    if float(temp) <= 0.0:
        raise ValueError(f"MDRC temperature must be > 0, got {temp}.")
    if clean_img_feat.shape != degraded_img_feat.shape:
        raise ValueError(
            "MDRC similarity consistency expects matching image feature shapes, got "
            f"{tuple(clean_img_feat.shape)} and {tuple(degraded_img_feat.shape)}."
        )
    if clean_img_feat.dim() != 2 or text_feat.dim() != 2:
        raise ValueError(
            "MDRC similarity consistency expects 2D image/text features, got "
            f"{tuple(clean_img_feat.shape)} and {tuple(text_feat.shape)}."
        )
    if clean_img_feat.shape[1] != text_feat.shape[1]:
        raise ValueError(
            "MDRC image/text feature dimensions must match, got "
            f"{clean_img_feat.shape[1]} and {text_feat.shape[1]}."
        )

    clean_img = F.normalize(
        torch.nan_to_num(clean_img_feat.detach().float(), nan=0.0, posinf=0.0, neginf=0.0),
        p=2,
        dim=-1,
        eps=eps,
    )
    degraded_img = F.normalize(
        torch.nan_to_num(degraded_img_feat.float(), nan=0.0, posinf=0.0, neginf=0.0),
        p=2,
        dim=-1,
        eps=eps,
    )
    text = F.normalize(
        torch.nan_to_num(text_feat.detach().float(), nan=0.0, posinf=0.0, neginf=0.0),
        p=2,
        dim=-1,
        eps=eps,
    )

    clean_sim = (clean_img @ text.t()) / float(temp)
    degraded_sim = (degraded_img @ text.t()) / float(temp)

    teacher_i2t = F.softmax(clean_sim.detach(), dim=1)
    student_i2t = F.log_softmax(degraded_sim, dim=1)
    loss_i2t = F.kl_div(student_i2t, teacher_i2t, reduction="batchmean")

    teacher_t2i = F.softmax(clean_sim.detach().t(), dim=1)
    student_t2i = F.log_softmax(degraded_sim.t(), dim=1)
    loss_t2i = F.kl_div(student_t2i, teacher_t2i, reduction="batchmean")
    return 0.5 * (loss_i2t + loss_t2i)
