import torch
import torch.nn.functional as F


class QueryAdaptiveLocalReranker(object):
    """
    Inference-only query-adaptive local reranker over top-K gallery candidates.

    Expected tensor shapes:
        query_token_embeddings:      [Q, Lt, D]
        query_token_ids:             [Q, Lt]
        query_global_embeddings:     [Q, D]
        gallery_rgb_patch_tokens:    [G, Np, D]
        gallery_thermal_patch_tokens:[G, Np, D] or None
        global_similarity:           [Q, G]
    """

    def __init__(
        self,
        topk=128,
        alpha=0.2,
        token_topm=8,
        patch_topr=4,
        use_score_norm=False,
        score_norm_type="zscore",
        use_query_gate=False,
        query_gate_type="top12_margin",
        query_gate_margin=0.05,
        vocab_size=49408,
        num_special_tokens=3,
        query_chunk_size=64,
        candidate_chunk_size=16,
        score_norm_eps=1e-6,
    ):
        self.topk = int(topk)
        self.alpha = float(alpha)
        self.token_topm = int(token_topm)
        self.patch_topr = int(patch_topr)
        self.use_score_norm = bool(use_score_norm)
        self.score_norm_type = str(score_norm_type).lower()
        self.use_query_gate = bool(use_query_gate)
        self.query_gate_type = str(query_gate_type).lower()
        self.query_gate_margin = float(query_gate_margin)
        self.vocab_size = int(vocab_size)
        self.num_special_tokens = int(num_special_tokens)
        self.query_chunk_size = int(query_chunk_size)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.score_norm_eps = float(score_norm_eps)

        if self.topk <= 0:
            raise ValueError(f"rerank_topk must be > 0, got {self.topk}.")
        if self.token_topm <= 0:
            raise ValueError(
                f"rerank_token_topm must be > 0, got {self.token_topm}."
            )
        if self.patch_topr <= 0:
            raise ValueError(
                f"rerank_patch_topr must be > 0, got {self.patch_topr}."
            )
        if self.vocab_size <= self.num_special_tokens:
            raise ValueError(
                "vocab_size must be larger than num_special_tokens, "
                f"got vocab_size={self.vocab_size}, "
                f"num_special_tokens={self.num_special_tokens}."
            )
        if self.score_norm_eps <= 0:
            raise ValueError(
                f"score_norm_eps must be > 0, got {self.score_norm_eps}."
            )
        if self.use_score_norm and self.score_norm_type != "zscore":
            raise ValueError(
                "Unsupported qalr_score_norm_type="
                f"{self.score_norm_type}. Only 'zscore' is currently supported."
            )
        if self.use_query_gate and self.query_gate_type != "top12_margin":
            raise ValueError(
                "Unsupported qalr_query_gate_type="
                f"{self.query_gate_type}. Only 'top12_margin' is currently supported."
            )

    def _masked_softmax(self, values, valid_mask, dim):
        safe_values = values.masked_fill(~valid_mask, float("-inf"))
        max_values = safe_values.max(dim=dim, keepdim=True).values
        max_values = torch.where(
            torch.isfinite(max_values), max_values, torch.zeros_like(max_values)
        )
        weights = torch.exp(safe_values - max_values) * valid_mask.float()
        denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        return weights / denom

    def _build_valid_text_mask(self, query_token_ids):
        # Token ids in [1, vocab_size - num_special_tokens - 1] are regular BPE tokens.
        special_token_start = self.vocab_size - self.num_special_tokens
        return (query_token_ids > 0) & (query_token_ids < special_token_start)

    def _build_patch_pool(self, gallery_rgb_patch_tokens, gallery_thermal_patch_tokens):
        rgb_patch_tokens = gallery_rgb_patch_tokens.float()
        if gallery_thermal_patch_tokens is None:
            return rgb_patch_tokens

        thermal_patch_tokens = gallery_thermal_patch_tokens.float()
        if thermal_patch_tokens.shape[1] == 0:
            return rgb_patch_tokens
        return torch.cat([rgb_patch_tokens, thermal_patch_tokens], dim=1)

    def _select_salient_tokens(
        self,
        query_token_embeddings,
        query_token_ids,
        query_global_embeddings,
    ):
        valid_token_mask = self._build_valid_text_mask(query_token_ids)
        query_token_embeddings = F.normalize(
            query_token_embeddings.float(), p=2, dim=-1
        )
        query_global_embeddings = F.normalize(
            query_global_embeddings.float(), p=2, dim=-1
        )

        saliency = F.cosine_similarity(
            query_token_embeddings,
            query_global_embeddings.unsqueeze(1),
            dim=-1,
        )
        saliency = saliency.masked_fill(~valid_token_mask, float("-inf"))

        effective_topm = min(self.token_topm, saliency.shape[1])
        topm_saliency, topm_indices = torch.topk(
            saliency, k=effective_topm, dim=1, largest=True, sorted=False
        )
        topm_valid_mask = torch.isfinite(topm_saliency)
        safe_topm_indices = topm_indices.masked_fill(~topm_valid_mask, 0)

        gathered_tokens = torch.gather(
            query_token_embeddings,
            dim=1,
            index=safe_topm_indices.unsqueeze(-1).expand(
                -1, -1, query_token_embeddings.shape[-1]
            ),
        )
        topm_weights = self._masked_softmax(
            topm_saliency, valid_mask=topm_valid_mask, dim=1
        )
        return gathered_tokens, topm_weights

    def _normalize_local_scores(self, local_scores):
        if not self.use_score_norm:
            return local_scores, None

        if self.score_norm_type != "zscore":
            raise ValueError(
                "Unsupported qalr_score_norm_type="
                f"{self.score_norm_type}. Only 'zscore' is currently supported."
            )

        local_mean = local_scores.mean(dim=1, keepdim=True)
        local_std = local_scores.std(dim=1, unbiased=False, keepdim=True)
        safe_local_std = local_std.clamp_min(self.score_norm_eps)
        local_scores = (local_scores - local_mean) / safe_local_std
        return local_scores, safe_local_std

    def _build_query_gate_mask(self, global_similarity):
        num_query, num_gallery = global_similarity.shape
        if not self.use_query_gate:
            return torch.ones(
                num_query, dtype=torch.bool, device=global_similarity.device
            ), None

        if self.query_gate_type != "top12_margin":
            raise ValueError(
                "Unsupported qalr_query_gate_type="
                f"{self.query_gate_type}. Only 'top12_margin' is currently supported."
            )

        if num_gallery < 2:
            return torch.zeros(
                num_query, dtype=torch.bool, device=global_similarity.device
            ), torch.full(
                (num_query,),
                float("inf"),
                dtype=global_similarity.dtype,
                device=global_similarity.device,
            )

        top2_scores = torch.topk(
            global_similarity, k=2, dim=1, largest=True, sorted=True
        ).values
        top12_margin = top2_scores[:, 0] - top2_scores[:, 1]
        query_gate_mask = top12_margin < self.query_gate_margin
        return query_gate_mask, top12_margin

    def _summarize_querywise_tensor(self, values):
        if values.numel() == 0:
            return {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        return {
            "avg": values.mean().item(),
            "min": values.min().item(),
            "max": values.max().item(),
        }

    def rerank(
        self,
        query_token_embeddings,
        query_token_ids,
        query_global_embeddings,
        gallery_rgb_patch_tokens,
        gallery_thermal_patch_tokens,
        global_similarity,
        return_stats=False,
    ):
        num_query, num_gallery = global_similarity.shape
        if num_query == 0 or num_gallery == 0:
            if return_stats:
                return global_similarity, {}
            return global_similarity

        patch_pool = self._build_patch_pool(
            gallery_rgb_patch_tokens, gallery_thermal_patch_tokens
        )
        if patch_pool.shape[1] == 0:
            if return_stats:
                return global_similarity, {}
            return global_similarity

        patch_pool = F.normalize(patch_pool.float(), p=2, dim=-1)
        salient_query_tokens, salient_token_weights = self._select_salient_tokens(
            query_token_embeddings=query_token_embeddings,
            query_token_ids=query_token_ids,
            query_global_embeddings=query_global_embeddings,
        )

        effective_topk = min(self.topk, num_gallery)
        effective_patch_topr = min(self.patch_topr, patch_pool.shape[1])
        topk_scores, topk_indices = torch.topk(
            global_similarity,
            k=effective_topk,
            dim=1,
            largest=True,
            sorted=False,
        )
        query_gate_mask, top12_margin = self._build_query_gate_mask(global_similarity)
        reranked_similarity = global_similarity.clone()
        raw_local_means = []
        raw_local_stds = []
        used_local_means = []
        used_local_stds = []
        safe_local_std_mins = []

        for q_start in range(0, num_query, self.query_chunk_size):
            q_end = min(q_start + self.query_chunk_size, num_query)
            query_tokens_chunk = salient_query_tokens[q_start:q_end]
            token_weights_chunk = salient_token_weights[q_start:q_end]
            topk_indices_chunk = topk_indices[q_start:q_end]
            query_gate_mask_chunk = query_gate_mask[q_start:q_end]
            local_scores_chunk = torch.zeros_like(topk_scores[q_start:q_end])

            for c_start in range(0, effective_topk, self.candidate_chunk_size):
                c_end = min(c_start + self.candidate_chunk_size, effective_topk)
                candidate_indices = topk_indices_chunk[:, c_start:c_end]
                candidate_patch_tokens = patch_pool[candidate_indices]

                token_patch_similarity = torch.einsum(
                    "bmd,bkpd->bkmp",
                    query_tokens_chunk,
                    candidate_patch_tokens,
                )
                top_patch_scores = torch.topk(
                    token_patch_similarity,
                    k=effective_patch_topr,
                    dim=-1,
                    largest=True,
                    sorted=False,
                ).values
                token_local_scores = top_patch_scores.mean(dim=-1)
                local_scores_chunk[:, c_start:c_end] = torch.sum(
                    token_local_scores * token_weights_chunk.unsqueeze(1),
                    dim=-1,
                )

            raw_local_means.append(local_scores_chunk.mean(dim=1).detach().cpu())
            raw_local_stds.append(
                local_scores_chunk.std(dim=1, unbiased=False).detach().cpu()
            )
            fused_local_scores_chunk, safe_local_std = self._normalize_local_scores(
                local_scores_chunk
            )
            if self.use_score_norm:
                used_local_means.append(
                    fused_local_scores_chunk.mean(dim=1).detach().cpu()
                )
                used_local_stds.append(
                    fused_local_scores_chunk.std(dim=1, unbiased=False).detach().cpu()
                )
                safe_local_std_mins.append(safe_local_std.min().detach().cpu())

            final_topk_scores = topk_scores[q_start:q_end] + self.alpha * fused_local_scores_chunk
            if self.use_query_gate:
                final_topk_scores = torch.where(
                    query_gate_mask_chunk.unsqueeze(1),
                    final_topk_scores,
                    topk_scores[q_start:q_end],
                )
            reranked_similarity[q_start:q_end].scatter_(
                dim=1,
                index=topk_indices_chunk,
                src=final_topk_scores,
            )

        if not return_stats:
            return reranked_similarity

        raw_local_means = torch.cat(raw_local_means, dim=0)
        raw_local_stds = torch.cat(raw_local_stds, dim=0)
        stats = {
            "raw_local_mean": self._summarize_querywise_tensor(raw_local_means),
            "raw_local_std": self._summarize_querywise_tensor(raw_local_stds),
        }
        if self.use_score_norm:
            used_local_means = torch.cat(used_local_means, dim=0)
            used_local_stds = torch.cat(used_local_stds, dim=0)
            safe_local_std_mins = torch.stack(safe_local_std_mins, dim=0)
            stats.update(
                {
                    "used_local_mean": self._summarize_querywise_tensor(
                        used_local_means
                    ),
                    "used_local_std": self._summarize_querywise_tensor(
                        used_local_stds
                    ),
                    "safe_local_std_min": safe_local_std_mins.min().item(),
                }
            )
        if top12_margin is not None:
            stats["query_gate_margin"] = self._summarize_querywise_tensor(
                top12_margin.detach().cpu()
            )
        stats["gated_query_ratio"] = query_gate_mask.float().mean().item()

        return reranked_similarity, stats
