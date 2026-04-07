import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def build_transport_rerank_config(args) -> Dict[str, Any]:
    """
    Build a transport-rerank config dict from argparse/yacs-like args.

    This helper is intentionally side-effect free so later evaluation code can
    opt in explicitly without changing current baseline behavior.
    """
    return {
        "use_transport_rerank": bool(
            getattr(args, "use_transport_rerank", False)
        ),
        "transport_topk": int(getattr(args, "transport_topk", 128)),
        "transport_lambda": float(getattr(args, "transport_lambda", 0.1)),
        "transport_patch_topr": int(
            getattr(args, "transport_patch_topr", 4)
        ),
        "transport_token_topm": int(
            getattr(args, "transport_token_topm", 8)
        ),
        "transport_eps": float(getattr(args, "transport_eps", 1e-6)),
        "transport_iters": int(getattr(args, "transport_iters", 20)),
        "transport_fuse_mode": str(
            getattr(args, "transport_fuse_mode", "avg")
        ),
        "transport_score_norm": str(
            getattr(args, "transport_score_norm", "none")
        ),
        "transport_use_patch_filter": bool(
            getattr(args, "transport_use_patch_filter", False)
        ),
        "transport_use_token_filter": bool(
            getattr(args, "transport_use_token_filter", False)
        ),
    }


def sinkhorn_transport(cost, eps, iters):
    """
    Compute an entropic OT transport plan with uniform marginals.

    Args:
        cost: [M, N] or [B, M, N], lower is better.
        eps: positive entropic regularization strength.
        iters: number of Sinkhorn normalization iterations.
    Returns:
        transport plan with the same leading shape as cost.
    """
    if cost is None:
        raise ValueError("cost must not be None.")

    cost = cost.float()
    squeeze_batch = False
    if cost.dim() == 2:
        cost = cost.unsqueeze(0)
        squeeze_batch = True
    elif cost.dim() != 3:
        raise ValueError(
            f"cost must have shape [M, N] or [B, M, N], got {tuple(cost.shape)}."
        )

    batch_size, num_src, num_tgt = cost.shape
    if num_src == 0 or num_tgt == 0:
        plan = cost.new_zeros(cost.shape)
        return plan[0] if squeeze_batch else plan

    eps = float(eps)
    iters = int(iters)
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}.")
    if iters <= 0:
        raise ValueError(f"iters must be > 0, got {iters}.")

    log_kernel = -cost / eps
    log_a = cost.new_full((batch_size, num_src), -math.log(float(num_src)))
    log_b = cost.new_full((batch_size, num_tgt), -math.log(float(num_tgt)))
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(iters):
        log_u = log_a - torch.logsumexp(
            log_kernel + log_v.unsqueeze(1), dim=2
        )
        log_v = log_b - torch.logsumexp(
            log_kernel + log_u.unsqueeze(2), dim=1
        )

    log_plan = log_u.unsqueeze(2) + log_kernel + log_v.unsqueeze(1)
    plan = torch.exp(log_plan)
    return plan[0] if squeeze_batch else plan


def _build_valid_text_mask(query_token_ids, vocab_size, num_special_tokens):
    if query_token_ids is None:
        return None
    special_token_start = int(vocab_size) - int(num_special_tokens)
    return (query_token_ids > 0) & (query_token_ids < special_token_start)


def select_text_tokens(
    query_token_embeddings,
    query_token_ids=None,
    query_global_embedding=None,
    token_topm=8,
    use_token_filter=False,
    vocab_size=49408,
    num_special_tokens=3,
):
    """
    Select text tokens for local reranking.

    Default behavior keeps all valid non-special tokens.
    When use_token_filter=True, keep the top-M tokens most aligned with the
    query-level text embedding.
    """
    if query_token_embeddings.dim() != 2:
        raise ValueError(
            "query_token_embeddings must have shape [L_t, C], "
            f"got {tuple(query_token_embeddings.shape)}."
        )

    token_embeddings = query_token_embeddings.float()
    if token_embeddings.shape[0] == 0:
        return token_embeddings

    valid_mask = _build_valid_text_mask(
        query_token_ids=query_token_ids,
        vocab_size=vocab_size,
        num_special_tokens=num_special_tokens,
    )
    if valid_mask is None:
        valid_tokens = token_embeddings
    else:
        valid_tokens = token_embeddings[valid_mask]

    if valid_tokens.shape[0] == 0:
        return token_embeddings.new_zeros((0, token_embeddings.shape[-1]))

    if (not use_token_filter) or query_global_embedding is None:
        return valid_tokens

    effective_topm = min(int(token_topm), valid_tokens.shape[0])
    if effective_topm <= 0 or effective_topm >= valid_tokens.shape[0]:
        return valid_tokens

    norm_tokens = F.normalize(valid_tokens, p=2, dim=-1)
    norm_query = F.normalize(
        query_global_embedding.float().unsqueeze(0), p=2, dim=-1
    ).squeeze(0)
    saliency = norm_tokens @ norm_query
    topm_indices = torch.topk(
        saliency, k=effective_topm, largest=True, sorted=False
    ).indices
    return valid_tokens[topm_indices]


def select_patch_tokens(
    patch_tokens,
    query_global_embedding=None,
    patch_topr=4,
    use_patch_filter=False,
):
    """
    Select image patch tokens for a single gallery candidate.

    Default behavior keeps all patch tokens.
    When use_patch_filter=True, keep the top-R patches most aligned with the
    query-level text embedding.
    """
    if patch_tokens.dim() != 2:
        raise ValueError(
            "patch_tokens must have shape [N_p, C], "
            f"got {tuple(patch_tokens.shape)}."
        )

    patch_tokens = patch_tokens.float()
    if patch_tokens.shape[0] == 0:
        return patch_tokens

    if (not use_patch_filter) or query_global_embedding is None:
        return patch_tokens

    effective_topr = min(int(patch_topr), patch_tokens.shape[0])
    if effective_topr <= 0 or effective_topr >= patch_tokens.shape[0]:
        return patch_tokens

    norm_patches = F.normalize(patch_tokens, p=2, dim=-1)
    norm_query = F.normalize(
        query_global_embedding.float().unsqueeze(0), p=2, dim=-1
    ).squeeze(0)
    saliency = norm_patches @ norm_query
    topr_indices = torch.topk(
        saliency, k=effective_topr, largest=True, sorted=False
    ).indices
    return patch_tokens[topr_indices]


def _single_modality_transport_score(
    text_tokens,
    patch_tokens,
    query_global_embedding=None,
    patch_topr=4,
    eps=1e-6,
    iters=20,
    use_patch_filter=False,
):
    if text_tokens.shape[0] == 0 or patch_tokens is None or patch_tokens.shape[0] == 0:
        return text_tokens.new_zeros(())

    selected_patches = select_patch_tokens(
        patch_tokens=patch_tokens,
        query_global_embedding=query_global_embedding,
        patch_topr=patch_topr,
        use_patch_filter=use_patch_filter,
    )
    if selected_patches.shape[0] == 0:
        return text_tokens.new_zeros(())

    norm_text = F.normalize(text_tokens.float(), p=2, dim=-1)
    norm_patches = F.normalize(selected_patches.float(), p=2, dim=-1)
    similarity = norm_text @ norm_patches.t()
    cost = 1.0 - similarity
    transport_plan = sinkhorn_transport(cost=cost, eps=eps, iters=iters)
    return torch.sum(transport_plan * similarity)


def pair_transport_score(
    text_tokens,
    rgb_patch_tokens,
    thermal_patch_tokens=None,
    query_global_embedding=None,
    patch_topr=4,
    eps=1e-6,
    iters=20,
    fuse_mode="avg",
    use_patch_filter=False,
    rgb_precision_weight=None,
    thermal_precision_weight=None,
    precision_weight_valid=None,
):
    """
    Compute a transport-based local score for one query-gallery pair.

    Returns:
        {
            "local_score": scalar fused score,
            "rgb_local_score": scalar score from text-vs-RGB patches,
            "thermal_local_score": scalar score from text-vs-Thermal patches,
            "precision_fusion_used": bool, whether precision fusion was actually used
        }
    """
    rgb_local_score = _single_modality_transport_score(
        text_tokens=text_tokens,
        patch_tokens=rgb_patch_tokens,
        query_global_embedding=query_global_embedding,
        patch_topr=patch_topr,
        eps=eps,
        iters=iters,
        use_patch_filter=use_patch_filter,
    )

    thermal_available = (
        thermal_patch_tokens is not None
        and thermal_patch_tokens.dim() == 2
        and thermal_patch_tokens.shape[0] > 0
    )
    if thermal_available:
        thermal_local_score = _single_modality_transport_score(
            text_tokens=text_tokens,
            patch_tokens=thermal_patch_tokens,
            query_global_embedding=query_global_embedding,
            patch_topr=patch_topr,
            eps=eps,
            iters=iters,
            use_patch_filter=use_patch_filter,
        )
    else:
        thermal_local_score = rgb_local_score.new_zeros(())

    fuse_mode = str(fuse_mode).lower()
    precision_fusion_used = False
    if not thermal_available:
        local_score = rgb_local_score
    elif fuse_mode == "avg":
        local_score = 0.5 * rgb_local_score + 0.5 * thermal_local_score
    elif fuse_mode == "precision":
        weights_available = (
            rgb_precision_weight is not None
            and thermal_precision_weight is not None
        )
        if precision_weight_valid is not None:
            if torch.is_tensor(precision_weight_valid):
                weights_available = (
                    weights_available and bool(precision_weight_valid.item())
                )
            else:
                weights_available = weights_available and bool(
                    precision_weight_valid
                )

        if weights_available:
            rgb_weight = float(
                rgb_precision_weight.item()
                if torch.is_tensor(rgb_precision_weight)
                else rgb_precision_weight
            )
            thermal_weight = float(
                thermal_precision_weight.item()
                if torch.is_tensor(thermal_precision_weight)
                else thermal_precision_weight
            )
            if (
                math.isfinite(rgb_weight)
                and math.isfinite(thermal_weight)
                and rgb_weight >= 0.0
                and thermal_weight >= 0.0
            ):
                weight_sum = max(rgb_weight + thermal_weight, float(eps))
                rgb_weight = rgb_weight / weight_sum
                thermal_weight = thermal_weight / weight_sum
                local_score = (
                    rgb_local_score * rgb_weight
                    + thermal_local_score * thermal_weight
                )
                precision_fusion_used = True
            else:
                local_score = 0.5 * rgb_local_score + 0.5 * thermal_local_score
        else:
            local_score = 0.5 * rgb_local_score + 0.5 * thermal_local_score
    elif fuse_mode in {"add", "sum"}:
        local_score = rgb_local_score + thermal_local_score
    elif fuse_mode == "rgb":
        local_score = rgb_local_score
    elif fuse_mode in {"ir", "thermal"}:
        local_score = thermal_local_score
    else:
        raise ValueError(
            "Unsupported transport_fuse_mode="
            f"{fuse_mode}. Supported: avg, precision, add, rgb, thermal."
        )

    return {
        "local_score": local_score,
        "rgb_local_score": rgb_local_score,
        "thermal_local_score": thermal_local_score,
        "precision_fusion_used": precision_fusion_used,
    }


def _summarize_values(values):
    if values.numel() == 0:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": values.mean().item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def transport_rerank(
    query_token_embeddings,
    query_token_ids,
    query_global_embeddings,
    gallery_rgb_patch_tokens,
    gallery_thermal_patch_tokens,
    global_similarity,
    gallery_rgb_precision_weights=None,
    gallery_thermal_precision_weights=None,
    gallery_precision_weight_valid=None,
    topk=128,
    lambda_weight=0.1,
    patch_topr=4,
    token_topm=8,
    eps=1e-6,
    iters=20,
    fuse_mode="avg",
    score_norm="none",
    use_patch_filter=False,
    use_token_filter=False,
    vocab_size=49408,
    num_special_tokens=3,
    return_stats=False,
):
    """
    OT-style local rerank over the top-K gallery candidates of each query.

    The transport cost is 1 - cosine_similarity and the local score is
    sum(P * similarity), where P is the Sinkhorn transport plan.

    Important behavior:
        Only the ordering inside each query's global top-K is changed.
        All non-top-K candidates keep their original scores unchanged.
    """
    if global_similarity is None:
        raise ValueError("global_similarity must not be None.")

    num_query, num_gallery = global_similarity.shape
    stats = {
        "transport_applied": False,
        "transport_status": "skipped",
        "transport_topk": int(topk),
        "transport_lambda": float(lambda_weight),
    }
    if num_query == 0 or num_gallery == 0:
        if return_stats:
            return global_similarity, stats
        return global_similarity

    if query_token_embeddings is None or gallery_rgb_patch_tokens is None:
        stats["transport_status"] = "missing_local_features"
        if return_stats:
            return global_similarity, stats
        return global_similarity

    effective_topk = min(int(topk), num_gallery)
    topk_scores, topk_indices = torch.topk(
        global_similarity,
        k=effective_topk,
        dim=1,
        largest=True,
        sorted=True,
    )

    local_scores = global_similarity.new_zeros((num_query, effective_topk))
    rgb_local_scores = global_similarity.new_zeros((num_query, effective_topk))
    thermal_local_scores = global_similarity.new_zeros((num_query, effective_topk))
    empty_token_queries = 0
    precision_used_pairs = 0
    precision_fallback_pairs = 0

    for q_idx in range(num_query):
        query_global_embedding = None
        if query_global_embeddings is not None:
            query_global_embedding = query_global_embeddings[q_idx]

        selected_text_tokens = select_text_tokens(
            query_token_embeddings=query_token_embeddings[q_idx],
            query_token_ids=None
            if query_token_ids is None
            else query_token_ids[q_idx],
            query_global_embedding=query_global_embedding,
            token_topm=token_topm,
            use_token_filter=use_token_filter,
            vocab_size=vocab_size,
            num_special_tokens=num_special_tokens,
        )
        if selected_text_tokens.shape[0] == 0:
            empty_token_queries += 1
            continue

        for k_idx in range(effective_topk):
            gallery_index = int(topk_indices[q_idx, k_idx].item())
            pair_scores = pair_transport_score(
                text_tokens=selected_text_tokens,
                rgb_patch_tokens=gallery_rgb_patch_tokens[gallery_index],
                thermal_patch_tokens=None
                if gallery_thermal_patch_tokens is None
                else gallery_thermal_patch_tokens[gallery_index],
                query_global_embedding=query_global_embedding,
                patch_topr=patch_topr,
                eps=eps,
                iters=iters,
                fuse_mode=fuse_mode,
                use_patch_filter=use_patch_filter,
                rgb_precision_weight=None
                if gallery_rgb_precision_weights is None
                else gallery_rgb_precision_weights[gallery_index],
                thermal_precision_weight=None
                if gallery_thermal_precision_weights is None
                else gallery_thermal_precision_weights[gallery_index],
                precision_weight_valid=None
                if gallery_precision_weight_valid is None
                else gallery_precision_weight_valid[gallery_index],
            )
            local_scores[q_idx, k_idx] = pair_scores["local_score"]
            rgb_local_scores[q_idx, k_idx] = pair_scores["rgb_local_score"]
            thermal_local_scores[q_idx, k_idx] = pair_scores["thermal_local_score"]
            if str(fuse_mode).lower() == "precision":
                if pair_scores["precision_fusion_used"]:
                    precision_used_pairs += 1
                else:
                    precision_fallback_pairs += 1

    score_norm = str(score_norm).lower()
    safe_local_std = None
    if score_norm == "zscore":
        local_mean = local_scores.mean(dim=1, keepdim=True)
        local_std = local_scores.std(dim=1, unbiased=False, keepdim=True)
        safe_local_std = local_std.clamp_min(float(eps))
        normalized_local_scores = (local_scores - local_mean) / safe_local_std
    elif score_norm == "none":
        normalized_local_scores = local_scores
    else:
        raise ValueError(
            "Unsupported transport_score_norm="
            f"{score_norm}. Supported: none, zscore."
        )

    reranked_similarity = global_similarity.clone()
    fused_topk_scores = topk_scores + float(lambda_weight) * normalized_local_scores
    rerank_order = torch.argsort(
        fused_topk_scores, dim=1, descending=True
    )  # [Q, K]
    reranked_topk_indices = torch.gather(topk_indices, dim=1, index=rerank_order)
    reranked_similarity.scatter_(
        1,
        reranked_topk_indices,
        topk_scores,
    )

    num_rank_changed = (
        rerank_order
        != torch.arange(
            effective_topk, device=rerank_order.device
        ).unsqueeze(0)
    ).sum().item()

    stats.update(
        {
            "transport_applied": True,
            "transport_status": "ok",
            "effective_topk": effective_topk,
            "num_query": num_query,
            "num_gallery": num_gallery,
            "num_empty_token_queries": empty_token_queries,
            "transport_fuse_mode": str(fuse_mode).lower(),
            "transport_score_norm": score_norm,
            "raw_local_mean": _summarize_values(local_scores.mean(dim=1)),
            "raw_local_std": _summarize_values(
                local_scores.std(dim=1, unbiased=False)
            ),
            "used_local_mean": _summarize_values(
                normalized_local_scores.mean(dim=1)
            ),
            "used_local_std": _summarize_values(
                normalized_local_scores.std(dim=1, unbiased=False)
            ),
            "rgb_local_mean": _summarize_values(
                rgb_local_scores.mean(dim=1)
            ),
            "thermal_local_mean": _summarize_values(
                thermal_local_scores.mean(dim=1)
            ),
            "safe_local_std_min": 0.0
            if safe_local_std is None
            else safe_local_std.min().item(),
            "num_rank_changed": int(num_rank_changed),
            "precision_used_pairs": int(precision_used_pairs),
            "precision_fallback_pairs": int(precision_fallback_pairs),
        }
    )
    if return_stats:
        return reranked_similarity, stats
    return reranked_similarity


class TransportReranker(object):
    """
    Evaluation-only transport reranker over top-K gallery candidates.
    """

    def __init__(
        self,
        topk: int = 128,
        lambda_weight: float = 0.1,
        patch_topr: int = 4,
        token_topm: int = 8,
        eps: float = 1e-6,
        iters: int = 20,
        fuse_mode: str = "avg",
        score_norm: str = "none",
        use_patch_filter: bool = False,
        use_token_filter: bool = False,
        vocab_size: int = 49408,
        num_special_tokens: int = 3,
    ):
        self.topk = int(topk)
        self.lambda_weight = float(lambda_weight)
        self.patch_topr = int(patch_topr)
        self.token_topm = int(token_topm)
        self.eps = float(eps)
        self.iters = int(iters)
        self.fuse_mode = str(fuse_mode)
        self.score_norm = str(score_norm)
        self.use_patch_filter = bool(use_patch_filter)
        self.use_token_filter = bool(use_token_filter)
        self.vocab_size = int(vocab_size)
        self.num_special_tokens = int(num_special_tokens)

        self._validate_hyperparams()

    def _validate_hyperparams(self) -> None:
        if self.topk <= 0:
            raise ValueError(f"transport_topk must be > 0, got {self.topk}.")
        if self.lambda_weight < 0.0:
            raise ValueError(
                "transport_lambda must be >= 0, "
                f"got {self.lambda_weight}."
            )
        if self.patch_topr <= 0:
            raise ValueError(
                "transport_patch_topr must be > 0, "
                f"got {self.patch_topr}."
            )
        if self.token_topm <= 0:
            raise ValueError(
                "transport_token_topm must be > 0, "
                f"got {self.token_topm}."
            )
        if self.eps <= 0.0:
            raise ValueError(f"transport_eps must be > 0, got {self.eps}.")
        if self.iters <= 0:
            raise ValueError(
                f"transport_iters must be > 0, got {self.iters}."
            )

    def rerank(
        self,
        query_token_embeddings: Optional[torch.Tensor],
        query_token_ids: Optional[torch.Tensor],
        query_global_embeddings: Optional[torch.Tensor],
        gallery_rgb_patch_tokens: Optional[torch.Tensor],
        gallery_thermal_patch_tokens: Optional[torch.Tensor],
        global_similarity: torch.Tensor,
        gallery_rgb_precision_weights: Optional[torch.Tensor] = None,
        gallery_thermal_precision_weights: Optional[torch.Tensor] = None,
        gallery_precision_weight_valid: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        return transport_rerank(
            query_token_embeddings=query_token_embeddings,
            query_token_ids=query_token_ids,
            query_global_embeddings=query_global_embeddings,
            gallery_rgb_patch_tokens=gallery_rgb_patch_tokens,
            gallery_thermal_patch_tokens=gallery_thermal_patch_tokens,
            gallery_rgb_precision_weights=gallery_rgb_precision_weights,
            gallery_thermal_precision_weights=gallery_thermal_precision_weights,
            gallery_precision_weight_valid=gallery_precision_weight_valid,
            global_similarity=global_similarity,
            topk=self.topk,
            lambda_weight=self.lambda_weight,
            patch_topr=self.patch_topr,
            token_topm=self.token_topm,
            eps=self.eps,
            iters=self.iters,
            fuse_mode=self.fuse_mode,
            score_norm=self.score_norm,
            use_patch_filter=self.use_patch_filter,
            use_token_filter=self.use_token_filter,
            vocab_size=self.vocab_size,
            num_special_tokens=self.num_special_tokens,
            return_stats=return_stats,
        )


def build_transport_reranker(args) -> Optional[TransportReranker]:
    """
    Create a transport reranker only when the explicit flag is on.
    """
    config = build_transport_rerank_config(args)
    if not config["use_transport_rerank"]:
        return None

    return TransportReranker(
        topk=config["transport_topk"],
        lambda_weight=config["transport_lambda"],
        patch_topr=config["transport_patch_topr"],
        token_topm=config["transport_token_topm"],
        eps=config["transport_eps"],
        iters=config["transport_iters"],
        fuse_mode=config["transport_fuse_mode"],
        score_norm=config["transport_score_norm"],
        use_patch_filter=config["transport_use_patch_filter"],
        use_token_filter=config["transport_use_token_filter"],
        vocab_size=int(getattr(args, "vocab_size", 49408)),
    )
