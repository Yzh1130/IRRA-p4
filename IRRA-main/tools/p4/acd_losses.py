"""Uniform multi-positive ACD objectives; FP32 in training, FP64 for algebra tests."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


MECHANISM_VARIANTS = ("Pull", "A1const", "A1shuf", "DCLbal")


def draw_teacher_permutation(batch_size, generator):
    """Draw one row permutation from an explicit CPU generator only."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("An independent CPU torch.Generator is required")
    return torch.randperm(batch_size, generator=generator, device="cpu")


def _mechanism_inputs(z_f, z_i, pids):
    if (z_f.shape != z_i.shape or z_i.ndim != 2
            or z_i.shape[0] != z_i.shape[1] or z_i.shape[0] < 1
            or z_i.shape[0] != pids.numel()):
        raise ValueError("Expected equally sized square logits and one PID per row")
    if not z_i.is_floating_point() or not z_f.is_floating_point():
        raise ValueError("Teacher and student logits must be floating point")
    labels = pids.reshape(-1).to(z_i.device)
    positive = labels[:, None].eq(labels[None, :])
    return positive, positive_targets(labels, z_i.dtype)


def _teacher_row_reference(z_f, z_i, variant, ref_mean, permutation):
    if variant == "A1const":
        if ref_mean is None or not math.isfinite(float(ref_mean)):
            raise ValueError("A1const requires a finite ref_mean")
        return z_i.new_tensor(float(ref_mean))
    reference = torch.logsumexp(z_f.detach(), dim=1).to(z_i.device)
    if variant == "A1shuf":
        if (not isinstance(permutation, torch.Tensor)
                or permutation.device.type != "cpu"
                or permutation.dtype != torch.int64
                or permutation.ndim != 1 or permutation.numel() != z_i.shape[0]
                or not torch.equal(torch.sort(permutation).values,
                                   torch.arange(z_i.shape[0], device="cpu"))):
            raise ValueError("A1shuf requires an explicit CPU int64 row permutation")
        reference = reference.index_select(0, permutation.to(reference.device))
    return reference


def repulsion_diagnostics(z_f, z_i, pids, variant, *, ref_mean=None,
                          permutation=None):
    """Detached row-mean coefficients, available without changing legacy logs."""
    positive, _ = _mechanism_inputs(z_f, z_i, pids)
    with torch.no_grad():
        if variant == "A0":
            coefficient = z_i.new_tensor(1.0)
        elif variant in {"A0bal", "DCLbal"}:
            # DCLbal's nominal coefficient applies only to negative columns.
            coefficient = z_i.new_tensor(0.5)
        elif variant == "Pull":
            coefficient = z_i.new_tensor(0.0)
        elif variant in {"A1", "A1const", "A1shuf"}:
            reference = _teacher_row_reference(z_f, z_i, variant, ref_mean, permutation)
            coefficient = torch.sigmoid(torch.logsumexp(z_i.detach(), dim=1)
                                        - reference).mean()
        else:
            raise ValueError("Unknown ACD variant: " + str(variant))
        diagnostics = {"t2i_repulsion_coeff_mean": coefficient}
        if variant == "DCLbal":
            diagnostics["t2i_no_negative_rows"] = positive.all(dim=1).sum()
    return diagnostics


def mechanism_t2i(z_f, z_i, pids, variant, *, ref_mean=None, permutation=None):
    """R10 t2i objectives; only student logits participate in autograd."""
    if variant not in MECHANISM_VARIANTS:
        raise ValueError("Unknown mechanism variant: " + str(variant))
    positive, target = _mechanism_inputs(z_f, z_i, pids)
    pull = -0.5 * (target * z_i).sum(dim=1)
    if variant == "Pull":
        loss = pull.mean()
    elif variant == "DCLbal":
        has_negative = ~positive.all(dim=1)
        # Never evaluate an all -inf logsumexp, even on a masked-out row:
        # its backward would otherwise produce NaNs for an all-positive batch.
        denominator = z_i.new_zeros(z_i.shape[0])
        if has_negative.any():
            denominator[has_negative] = torch.logsumexp(
                z_i[has_negative].masked_fill(positive[has_negative], -torch.inf), dim=1)
        loss = (pull + 0.5 * denominator).mean()
    else:
        reference = _teacher_row_reference(z_f, z_i, variant, ref_mean, permutation)
        gap = torch.logsumexp(z_i, dim=1) - reference
        # Exact stable softplus, without F.softplus's linear threshold shortcut
        # (the latter changes FP64 gradients around gap=20).
        denominator = torch.logaddexp(torch.zeros_like(gap), gap)
        loss = (pull + denominator).mean()
    diagnostics = repulsion_diagnostics(z_f, z_i, pids, variant,
                                        ref_mean=ref_mean, permutation=permutation)
    return loss, diagnostics


def _mechanism_acd_loss(z_f, z_i, pids, variant, a1_impl, ref_mean, permutation):
    t2i, extra = mechanism_t2i(z_f, z_i, pids, variant,
                               ref_mean=ref_mean, permutation=permutation)
    l_i_t2i = uniform_ce(z_i, pids)
    l_i_i2t = uniform_ce(z_i.t(), pids)
    l_f = uniform_ce(z_f.detach(), pids)
    _, correction = joint_t2i(z_f, z_i, pids, a1_impl)
    total = t2i + l_i_i2t
    positive, _ = _mechanism_inputs(z_f, z_i, pids)
    negative = z_i.detach().masked_fill(positive, -torch.inf).max(dim=1).values
    finite = torch.isfinite(negative)
    diagnostics = {
        "loss": total.detach(), "l_I_t2i": l_i_t2i.detach(),
        "l_I_i2t": l_i_i2t.detach(), "l_F": l_f.detach(),
        "log2cosh": correction.detach(),
        "pi_I": torch.sigmoid(torch.logsumexp(z_i.detach(), dim=1)
                              - torch.logsumexp(z_f.detach(), dim=1)).mean(),
        "student_positive_logit_mean": z_i.detach()[positive].mean(),
        "student_strongest_negative_logit_mean": negative[finite].mean()
        if finite.any() else z_i.new_tensor(float("nan")),
    }
    diagnostics.update(extra)
    return total, diagnostics


def positive_targets(pids, dtype):
    pids = pids.reshape(-1)
    target = pids[:, None].eq(pids[None, :]).to(dtype=dtype)
    return target / target.sum(dim=1, keepdim=True)


def uniform_ce(logits, pids):
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or logits.shape[0] != pids.numel():
        raise ValueError("Expected square batch logits and one PID per row")
    target = positive_targets(pids.to(logits.device), logits.dtype)
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def joint_t2i(z_f, z_i, pids, implementation="joint"):
    if z_f.shape != z_i.shape or z_f.ndim != 2 or z_f.shape[0] != z_f.shape[1]:
        raise ValueError("Teacher and student logits must be equally sized square matrices")
    # Teacher is fixed even if a caller mistakenly supplies an attached tensor.
    z_f = z_f.detach()
    target = positive_targets(pids.to(z_i.device), z_i.dtype)
    gap = torch.logsumexp(z_i, dim=1) - torch.logsumexp(z_f, dim=1)
    correction = (torch.logaddexp(gap / 2, -gap / 2)).mean()
    if implementation == "joint":
        labels = torch.cat((target, target), dim=1) * 0.5
        loss = -(labels * F.log_softmax(torch.cat((z_f, z_i), dim=1), dim=1)).sum(dim=1).mean()
    elif implementation == "decomposed":
        loss = 0.5 * uniform_ce(z_f, pids) + 0.5 * uniform_ce(z_i, pids) + correction
    else:
        raise ValueError("a1_impl must be joint or decomposed")
    return loss, correction


def acd_loss(z_f, z_i, pids, variant, a1_impl="joint", *, ref_mean=None,
             permutation=None):
    if variant in MECHANISM_VARIANTS:
        return _mechanism_acd_loss(z_f, z_i, pids, variant, a1_impl,
                                   ref_mean, permutation)
    if variant not in {"A0", "A0bal", "A1"}:
        raise ValueError("variant must be A0, A0bal or A1")
    l_i_t2i = uniform_ce(z_i, pids)
    l_i_i2t = uniform_ce(z_i.t(), pids)
    l_f = uniform_ce(z_f.detach(), pids)
    joint, correction = joint_t2i(z_f, z_i, pids, a1_impl)
    if variant == "A0":
        total = l_i_t2i + l_i_i2t
    elif variant == "A0bal":
        total = 0.5 * l_i_t2i + l_i_i2t
    else:
        total = joint + l_i_i2t
    positive = pids.reshape(-1, 1).eq(pids.reshape(1, -1))
    negative = z_i.detach().masked_fill(positive, -torch.inf).max(dim=1).values
    finite = torch.isfinite(negative)
    diagnostics = {
        "loss": total.detach(), "l_I_t2i": l_i_t2i.detach(),
        "l_I_i2t": l_i_i2t.detach(), "l_F": l_f.detach(),
        "log2cosh": correction.detach(),
        "pi_I": torch.sigmoid(torch.logsumexp(z_i.detach(), dim=1)
                              - torch.logsumexp(z_f.detach(), dim=1)).mean(),
        "student_positive_logit_mean": z_i.detach()[positive].mean(),
        "student_strongest_negative_logit_mean": negative[finite].mean()
        if finite.any() else z_i.new_tensor(float("nan")),
    }
    return total, diagnostics
