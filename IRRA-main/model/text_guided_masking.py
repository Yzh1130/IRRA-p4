import torch
import torch.nn.functional as F


# Guided masking contract:
# A/noise masking samples low-score normal tokens and perturbs the main training
# text input. B/informative masking samples high-score normal tokens and builds
# MLM labels. Both return None unless their train-time switches are enabled.


def _build_candidate_token_mask(input_ids, attention_mask=None, special_token_ids=None):
    if input_ids is None:
        return None

    candidate_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask is not None:
        candidate_mask = candidate_mask & attention_mask.to(
            device=input_ids.device, dtype=torch.bool
        )
    else:
        # Token id 0 is padding in this project.
        candidate_mask = candidate_mask & input_ids.ne(0)

    special_ids = set(special_token_ids or [])
    special_ids.add(0)
    for token_id in special_ids:
        candidate_mask = candidate_mask & input_ids.ne(int(token_id))
    return candidate_mask


def compute_token_importance_scores(
    input_ids=None,
    token_embeddings=None,
    global_embedding=None,
    visual_embeddings=None,
    attention_maps=None,
    attention_mask=None,
    special_token_ids=None,
    score_source="none",
    detach_score=True,
    eps=1e-6,
    **kwargs,
):
    """
    Lightweight token-importance scoring entry for guided text masking.

    The score direction is fixed: lower score means lower estimated importance
    and therefore a more likely noise-token candidate.
    """
    score_source = str(score_source).lower()
    if score_source == "none" or input_ids is None:
        return None

    candidate_mask = _build_candidate_token_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_token_ids=special_token_ids,
    )
    if token_embeddings is None:
        scores = torch.zeros_like(input_ids, dtype=torch.float32)
    else:
        token_embeddings = token_embeddings.float()
        if global_embedding is not None:
            scores = F.cosine_similarity(
                token_embeddings,
                global_embedding.float().unsqueeze(1),
                dim=-1,
                eps=eps,
            )
        else:
            scores = token_embeddings.norm(dim=-1)

    if candidate_mask is not None:
        scores = scores.masked_fill(~candidate_mask, 0.0)
    if detach_score:
        scores = scores.detach()
    return scores


def build_noise_token_mask(
    input_ids,
    attention_mask=None,
    token_scores=None,
    enabled=False,
    is_training=True,
    mask_ratio=0.15,
    min_keep=1,
    special_token_ids=None,
    ablate_random=False,
    eps=1e-6,
    **kwargs,
):
    """
    Build a bool noise-token mask, shape-aligned with input_ids.

    Low token_scores are sampled with higher probability. When token_scores are
    unavailable, the function falls back to uniform random masking so the A switch
    can still run without a score source.
    """
    if (not enabled) or (not is_training):
        return None

    if input_ids is None:
        return None
    if input_ids.dim() != 2:
        raise ValueError(
            "input_ids for noise token masking must have shape [B, L], "
            f"got {tuple(input_ids.shape)}."
        )

    ratio = float(mask_ratio)
    if ratio <= 0.0:
        return torch.zeros_like(input_ids, dtype=torch.bool)
    if ratio > 1.0:
        raise ValueError(f"mask_ratio must be <= 1, got {ratio}.")

    candidate_mask = _build_candidate_token_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_token_ids=special_token_ids,
    )
    output_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    # Always keep at least one normal token when such a token exists.
    min_keep = max(1, int(min_keep))

    if token_scores is not None:
        if token_scores.shape != input_ids.shape:
            raise ValueError(
                "token_scores must match input_ids shape, got "
                f"{tuple(token_scores.shape)} and {tuple(input_ids.shape)}."
            )
        token_scores = token_scores.to(device=input_ids.device).float()

    valid_counts = candidate_mask.sum(dim=1)
    target_counts = torch.ceil(valid_counts.float() * ratio).long()
    max_counts = (valid_counts - min_keep).clamp_min(0)
    target_counts = torch.minimum(target_counts, max_counts)

    for row_idx in range(input_ids.shape[0]):
        k = int(target_counts[row_idx].item())
        if k <= 0:
            continue

        valid_positions = torch.nonzero(
            candidate_mask[row_idx], as_tuple=False
        ).flatten()
        if valid_positions.numel() == 0:
            continue

        if ablate_random or token_scores is None:
            weights = torch.ones(
                valid_positions.numel(),
                device=input_ids.device,
                dtype=torch.float32,
            )
        else:
            row_scores = token_scores[row_idx, valid_positions]
            finite_mask = torch.isfinite(row_scores)
            if not bool(finite_mask.all()):
                row_scores = torch.where(
                    finite_mask, row_scores, torch.zeros_like(row_scores)
                )
            centered = row_scores - row_scores.mean()
            scale = row_scores.std(unbiased=False).clamp_min(eps)
            weights = torch.softmax(-(centered / scale), dim=0)
            if not bool(torch.isfinite(weights).all()) or float(weights.sum().item()) <= 0.0:
                weights = torch.ones_like(row_scores, dtype=torch.float32)

        selected = torch.multinomial(weights, num_samples=k, replacement=False)
        output_mask[row_idx, valid_positions[selected]] = True

    return output_mask


def _sample_token_mask_from_scores(
    input_ids,
    candidate_mask,
    token_scores=None,
    mask_ratio=0.15,
    min_keep=1,
    prefer_high_score=True,
    ablate_random=False,
    eps=1e-6,
):
    output_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    ratio = float(mask_ratio)
    if ratio <= 0.0:
        return output_mask
    if ratio > 1.0:
        raise ValueError(f"mask_ratio must be <= 1, got {ratio}.")

    min_keep = max(1, int(min_keep))
    valid_counts = candidate_mask.sum(dim=1)
    target_counts = torch.ceil(valid_counts.float() * ratio).long()
    max_counts = (valid_counts - min_keep).clamp_min(0)
    target_counts = torch.minimum(target_counts, max_counts)

    if token_scores is not None:
        if token_scores.shape != input_ids.shape:
            raise ValueError(
                "token_scores must match input_ids shape, got "
                f"{tuple(token_scores.shape)} and {tuple(input_ids.shape)}."
            )
        token_scores = token_scores.to(device=input_ids.device).float()

    for row_idx in range(input_ids.shape[0]):
        k = int(target_counts[row_idx].item())
        if k <= 0:
            continue

        valid_positions = torch.nonzero(
            candidate_mask[row_idx], as_tuple=False
        ).flatten()
        if valid_positions.numel() == 0:
            continue

        if ablate_random or token_scores is None:
            weights = torch.ones(
                valid_positions.numel(),
                device=input_ids.device,
                dtype=torch.float32,
            )
        else:
            row_scores = token_scores[row_idx, valid_positions]
            finite_mask = torch.isfinite(row_scores)
            if not bool(finite_mask.all()):
                row_scores = torch.where(
                    finite_mask, row_scores, torch.zeros_like(row_scores)
                )
            centered = row_scores - row_scores.mean()
            scale = row_scores.std(unbiased=False).clamp_min(eps)
            direction = 1.0 if prefer_high_score else -1.0
            weights = torch.softmax(direction * (centered / scale), dim=0)
            if not bool(torch.isfinite(weights).all()) or float(weights.sum().item()) <= 0.0:
                weights = torch.ones_like(row_scores, dtype=torch.float32)

        selected = torch.multinomial(weights, num_samples=k, replacement=False)
        output_mask[row_idx, valid_positions[selected]] = True

    return output_mask


def build_informative_token_mask(
    input_ids,
    attention_mask=None,
    token_scores=None,
    enabled=False,
    is_training=True,
    mask_ratio=0.15,
    min_keep=1,
    special_token_ids=None,
    ablate_random=False,
    eps=1e-6,
    **kwargs,
):
    """
    Build a bool informative-token mask, shape-aligned with input_ids.

    High token_scores are sampled with higher probability. When token_scores are
    unavailable, the function falls back to uniform random masking.
    """
    if (not enabled) or (not is_training):
        return None
    if input_ids is None:
        return None
    if input_ids.dim() != 2:
        raise ValueError(
            "input_ids for informative token masking must have shape [B, L], "
            f"got {tuple(input_ids.shape)}."
        )

    candidate_mask = _build_candidate_token_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_token_ids=special_token_ids,
    )
    return _sample_token_mask_from_scores(
        input_ids=input_ids,
        candidate_mask=candidate_mask,
        token_scores=token_scores,
        mask_ratio=mask_ratio,
        min_keep=min_keep,
        prefer_high_score=True,
        ablate_random=ablate_random,
        eps=eps,
    )


def build_guided_mlm_inputs_and_labels(
    input_ids,
    informative_mask=None,
    mask_token_id=None,
    ignore_index=0,
    enabled=False,
    **kwargs,
):
    """
    Build MLM inputs/labels from an informative token mask.

    Selected informative tokens become prediction targets; all other positions
    are set to ignore_index, matching the existing compute_mlm(ignore_index=0).
    """
    if (not enabled) or informative_mask is None:
        return None, None
    if mask_token_id is None:
        raise ValueError("mask_token_id is required for guided MLM inputs.")
    if informative_mask.shape != input_ids.shape:
        raise ValueError(
            "informative_mask must match input_ids shape, got "
            f"{tuple(informative_mask.shape)} and {tuple(input_ids.shape)}."
        )

    informative_mask = informative_mask.to(device=input_ids.device, dtype=torch.bool)
    labels = torch.full_like(input_ids, fill_value=int(ignore_index))
    labels[informative_mask] = input_ids[informative_mask]
    mlm_input_ids = apply_guided_mask_to_input_ids(
        input_ids=input_ids,
        token_mask=informative_mask,
        mask_token_id=mask_token_id,
        enabled=True,
    )
    return mlm_input_ids, labels


def apply_guided_mask_to_input_ids(
    input_ids,
    token_mask=None,
    mask_token_id=None,
    enabled=False,
    **kwargs,
):
    """
    Apply a bool guided token mask by replacing selected ids with mask_token_id.

    The input tensor is returned unchanged when disabled or when the mask is empty.
    """
    if (not enabled) or token_mask is None:
        return input_ids
    if mask_token_id is None:
        raise ValueError("mask_token_id is required when guided masking is enabled.")
    if token_mask.shape != input_ids.shape:
        raise ValueError(
            "token_mask must match input_ids shape, got "
            f"{tuple(token_mask.shape)} and {tuple(input_ids.shape)}."
        )

    token_mask = token_mask.to(device=input_ids.device, dtype=torch.bool)
    if not bool(token_mask.any()):
        return input_ids

    masked_input_ids = input_ids.clone()
    masked_input_ids[token_mask] = int(mask_token_id)
    return masked_input_ids


def _count_mask_tokens(mask):
    if not torch.is_tensor(mask):
        return 0, 0
    mask_bool = mask.detach().bool()
    return int(mask_bool.sum().item()), int(mask_bool.numel())


def summarize_mask_stats(
    token_scores=None,
    noise_mask=None,
    info_mask=None,
    candidate_mask=None,
    enabled_noise=False,
    enabled_info=False,
    noise_mask_mode="none",
    info_mask_mode="none",
    guided_mtp_labels=None,
    **kwargs,
):
    """
    Return lightweight debug statistics for reserved guided masking outputs.
    """
    stats = {
        "enabled_noise": bool(enabled_noise),
        "enabled_info": bool(enabled_info),
        "has_token_scores": torch.is_tensor(token_scores),
        "has_noise_mask": torch.is_tensor(noise_mask),
        "has_info_mask": torch.is_tensor(info_mask),
        "noise_mask_mode": str(noise_mask_mode),
        "info_mask_mode": str(info_mask_mode),
    }

    noise_count, noise_total = _count_mask_tokens(noise_mask)
    info_count, info_total = _count_mask_tokens(info_mask)
    stats["noise_mask_tokens"] = noise_count
    stats["noise_mask_ratio"] = (
        float(noise_count) / float(noise_total) if noise_total > 0 else 0.0
    )
    if torch.is_tensor(candidate_mask):
        valid_total = int(candidate_mask.detach().bool().sum().item())
    else:
        valid_total = noise_total
    stats["noise_mask_valid_ratio"] = (
        float(noise_count) / float(valid_total) if valid_total > 0 else 0.0
    )
    stats["info_mask_tokens"] = info_count
    stats["info_mask_ratio"] = (
        float(info_count) / float(info_total) if info_total > 0 else 0.0
    )
    stats["info_mask_valid_ratio"] = (
        float(info_count) / float(valid_total) if valid_total > 0 else 0.0
    )
    if torch.is_tensor(noise_mask):
        row_counts = noise_mask.detach().bool().sum(dim=1).float()
        stats["avg_noise_mask_tokens"] = float(row_counts.mean().item())
    else:
        stats["avg_noise_mask_tokens"] = 0.0
    if torch.is_tensor(info_mask):
        row_counts = info_mask.detach().bool().sum(dim=1).float()
        stats["avg_info_mask_tokens"] = float(row_counts.mean().item())
    else:
        stats["avg_info_mask_tokens"] = 0.0
    if torch.is_tensor(noise_mask) and torch.is_tensor(info_mask):
        noise_bool = noise_mask.detach().bool()
        info_bool = info_mask.detach().bool()
        overlap = int((noise_bool & info_bool).sum().item())
        union = int((noise_bool | info_bool).sum().item())
        stats["noise_info_overlap_ratio"] = (
            float(overlap) / float(union) if union > 0 else 0.0
        )
    else:
        stats["noise_info_overlap_ratio"] = 0.0
    if torch.is_tensor(guided_mtp_labels):
        stats["guided_mtp_valid_tokens"] = int(
            guided_mtp_labels.detach().ne(0).sum().item()
        )
        stats["avg_guided_mtp_valid_tokens"] = float(
            guided_mtp_labels.detach().ne(0).sum(dim=1).float().mean().item()
        )
    else:
        stats["guided_mtp_valid_tokens"] = 0
        stats["avg_guided_mtp_valid_tokens"] = 0.0

    if torch.is_tensor(token_scores) and token_scores.numel() > 0:
        safe_scores = token_scores.detach().float()
        if torch.is_tensor(candidate_mask) and candidate_mask.shape == token_scores.shape:
            score_values = safe_scores[candidate_mask.detach().bool()]
        else:
            score_values = safe_scores.reshape(-1)
        if score_values.numel() == 0:
            score_values = safe_scores.reshape(-1)
        stats["token_score_mean"] = float(safe_scores.mean().item())
        stats["token_score_valid_mean"] = float(score_values.mean().item())
        stats["token_score_var"] = float(score_values.var(unbiased=False).item())
        stats["token_score_max"] = float(score_values.max().item())
        stats["token_score_min"] = float(score_values.min().item())
    else:
        stats["token_score_mean"] = 0.0
        stats["token_score_valid_mean"] = 0.0
        stats["token_score_var"] = 0.0
        stats["token_score_max"] = 0.0
        stats["token_score_min"] = 0.0

    return stats
