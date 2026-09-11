import torch
import torch.nn.functional as F


def blur_tensor(tensor, kernel_size):
    """Apply replicate-padded average blur while preserving shape and dtype."""
    if tensor is None:
        return None
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return tensor

    original_dtype = tensor.dtype
    needs_float_work = not tensor.is_floating_point() or (
        tensor.device.type == "cpu"
        and tensor.dtype in (torch.float16, torch.bfloat16)
    )
    work = tensor.float() if needs_float_work else tensor
    pad_total = kernel_size - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    padded = F.pad(
        work,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="replicate",
    )
    blurred = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)
    return blurred.to(dtype=original_dtype) if needs_float_work else blurred
