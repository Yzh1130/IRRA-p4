"""Uniform multi-positive ACD objectives; FP32 in training, FP64 for algebra tests."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


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


def acd_loss(z_f, z_i, pids, variant, a1_impl="joint"):
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
