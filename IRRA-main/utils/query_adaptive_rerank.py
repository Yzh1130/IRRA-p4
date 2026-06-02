import math

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
        use_focus_topn=False,
        focus_topn=16,
        vocab_size=49408,
        num_special_tokens=3,
        query_chunk_size=64,
        candidate_chunk_size=16,
        score_norm_eps=1e-6,
        use_qalr_v2=False,
        qalr_v2_alpha_min=0.0,
        qalr_v2_alpha_max=0.01,
        qalr_v2_margin_low=0.0,
        qalr_v2_margin_high=0.02,
        qalr_v2_freeze_top1_margin=0.02,
        qalr_v2_promote_margin=0.0,
        qalr_v2_modality_fuse="concat",
        qalr_v2_top1_guard=False,
        qalr_v2_promote_only=False,
        qalr_v2_diag_only=False,
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
        self.use_focus_topn = bool(use_focus_topn)
        self.focus_topn = int(focus_topn)
        self.vocab_size = int(vocab_size)
        self.num_special_tokens = int(num_special_tokens)
        self.query_chunk_size = int(query_chunk_size)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.score_norm_eps = float(score_norm_eps)
        self.use_qalr_v2 = bool(use_qalr_v2)
        self.qalr_v2_alpha_min = float(qalr_v2_alpha_min)
        self.qalr_v2_alpha_max = float(qalr_v2_alpha_max)
        self.qalr_v2_margin_low = float(qalr_v2_margin_low)
        self.qalr_v2_margin_high = float(qalr_v2_margin_high)
        self.qalr_v2_freeze_top1_margin = float(qalr_v2_freeze_top1_margin)
        self.qalr_v2_promote_margin = float(qalr_v2_promote_margin)
        self.qalr_v2_modality_fuse = str(qalr_v2_modality_fuse).lower()
        self.qalr_v2_top1_guard = bool(qalr_v2_top1_guard)
        self.qalr_v2_promote_only = bool(qalr_v2_promote_only)
        self.qalr_v2_diag_only = bool(qalr_v2_diag_only)

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
        if self.use_focus_topn and self.focus_topn <= 0:
            raise ValueError(f"qalr_focus_topn must be > 0, got {self.focus_topn}.")
        if self.qalr_v2_alpha_max < self.qalr_v2_alpha_min:
            raise ValueError(
                "qalr_v2_alpha_max must be >= qalr_v2_alpha_min, got "
                f"{self.qalr_v2_alpha_max} < {self.qalr_v2_alpha_min}."
            )
        if self.qalr_v2_modality_fuse not in {
            "concat",
            "avg",
            "rgb",
            "thermal",
            "precision",
        }:
            raise ValueError(
                "Unsupported qalr_v2_modality_fuse="
                f"{self.qalr_v2_modality_fuse!r}."
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

        local_scores, safe_local_std = self._zscore_querywise(local_scores)
        return local_scores, safe_local_std

    def _zscore_querywise(self, scores):
        score_mean = scores.mean(dim=1, keepdim=True)
        score_std = scores.std(dim=1, unbiased=False, keepdim=True)
        safe_score_std = score_std.clamp_min(self.score_norm_eps)
        return (scores - score_mean) / safe_score_std, safe_score_std

    def _compute_top12_margin(self, global_similarity):
        num_query, num_gallery = global_similarity.shape
        if num_gallery < 2:
            return torch.full(
                (num_query,),
                float("inf"),
                dtype=global_similarity.dtype,
                device=global_similarity.device,
            )
        top2_scores = torch.topk(
            global_similarity, k=2, dim=1, largest=True, sorted=True
        ).values
        return top2_scores[:, 0] - top2_scores[:, 1]

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

        top12_margin = self._compute_top12_margin(global_similarity)
        query_gate_mask = top12_margin < self.query_gate_margin
        return query_gate_mask, top12_margin

    def _build_qalr_v2_alpha(self, top12_margin):
        alpha_min = self.qalr_v2_alpha_min
        alpha_max = self.qalr_v2_alpha_max
        if self.qalr_v2_margin_high <= self.qalr_v2_margin_low:
            use_max = top12_margin <= self.qalr_v2_margin_low
            alpha_eff = torch.where(
                use_max,
                top12_margin.new_full(top12_margin.shape, alpha_max),
                top12_margin.new_full(top12_margin.shape, alpha_min),
            )
        else:
            ramp = (
                (self.qalr_v2_margin_high - top12_margin)
                / (self.qalr_v2_margin_high - self.qalr_v2_margin_low)
            ).clamp(0.0, 1.0)
            alpha_eff = alpha_min + ramp * (alpha_max - alpha_min)
        return alpha_eff.clamp(max=alpha_max)

    def _compute_local_scores_from_pool(
        self,
        query_tokens_chunk,
        token_weights_chunk,
        patch_pool,
        topk_indices_chunk,
    ):
        if patch_pool is None or patch_pool.shape[1] == 0:
            return topk_indices_chunk.new_zeros(
                topk_indices_chunk.shape,
                dtype=query_tokens_chunk.dtype,
            )
        effective_patch_topr = min(self.patch_topr, patch_pool.shape[1])
        local_scores_chunk = query_tokens_chunk.new_zeros(topk_indices_chunk.shape)
        for c_start in range(0, topk_indices_chunk.shape[1], self.candidate_chunk_size):
            c_end = min(c_start + self.candidate_chunk_size, topk_indices_chunk.shape[1])
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
        return local_scores_chunk

    def _build_diagnostics(
        self,
        qids,
        gids,
        original_order,
        ranking_order,
        topk_indices,
        local_scores_topk,
        top12_margin,
        alpha_eff,
        effective_topk,
        modality_fuse,
    ):
        if qids is None or gids is None:
            return [], {}

        qids_cpu = qids.detach().cpu().long().view(-1)
        gids_cpu = gids.detach().cpu().long().view(-1)
        original_cpu = original_order.detach().cpu().long()
        ranking_cpu = ranking_order.detach().cpu().long()
        topk_cpu = topk_indices.detach().cpu().long()
        local_cpu = (
            local_scores_topk.detach().cpu().float()
            if torch.is_tensor(local_scores_topk)
            else None
        )
        margin_cpu = top12_margin.detach().cpu().float()
        alpha_cpu = alpha_eff.detach().cpu().float()

        queries = []
        changed_count = 0
        improved_count = 0
        hurt_count = 0
        neutral_count = 0
        for query_index in range(original_cpu.shape[0]):
            q_pid = int(qids_cpu[query_index].item())
            old_top1_index = int(original_cpu[query_index, 0].item())
            new_top1_index = int(ranking_cpu[query_index, 0].item())
            old_top1_pid = int(gids_cpu[old_top1_index].item())
            new_top1_pid = int(gids_cpu[new_top1_index].item())
            gt_match_old = old_top1_pid == q_pid
            gt_match_new = new_top1_pid == q_pid
            changed_top1 = old_top1_index != new_top1_index
            improved_top1 = (not gt_match_old) and gt_match_new
            hurt_top1 = gt_match_old and (not gt_match_new)
            neutral_changed_top1 = (
                changed_top1 and (not improved_top1) and (not hurt_top1)
            )

            if changed_top1:
                changed_count += 1
            if improved_top1:
                improved_count += 1
            if hurt_top1:
                hurt_count += 1
            if neutral_changed_top1:
                neutral_count += 1

            local_old = None
            local_new = None
            if local_cpu is not None:
                local_old = self._lookup_topk_local_score(
                    topk_cpu[query_index],
                    local_cpu[query_index],
                    old_top1_index,
                )
                local_new = self._lookup_topk_local_score(
                    topk_cpu[query_index],
                    local_cpu[query_index],
                    new_top1_index,
                )

            queries.append(
                {
                    "query_index": int(query_index),
                    "q_pid": q_pid,
                    "old_top1_index": old_top1_index,
                    "old_top1_pid": old_top1_pid,
                    "new_top1_index": new_top1_index,
                    "new_top1_pid": new_top1_pid,
                    "gt_match_old": bool(gt_match_old),
                    "gt_match_new": bool(gt_match_new),
                    "changed_top1": bool(changed_top1),
                    "improved_top1": bool(improved_top1),
                    "hurt_top1": bool(hurt_top1),
                    "neutral_changed_top1": bool(neutral_changed_top1),
                    "global_top12_margin": self._safe_float(
                        margin_cpu[query_index].item()
                    ),
                    "alpha_eff": self._safe_float(alpha_cpu[query_index].item()),
                    "local_score_old_top1": local_old,
                    "local_score_new_top1": local_new,
                    "modality_fuse": str(modality_fuse),
                    "rerank_topk": int(effective_topk),
                    "focus_topn": (
                        int(min(self.focus_topn, effective_topk))
                        if self.use_focus_topn
                        else int(effective_topk)
                    ),
                }
            )

        summary = {
            "total_queries": int(original_cpu.shape[0]),
            "changed_top1_count": int(changed_count),
            "improved_top1_count": int(improved_count),
            "hurt_top1_count": int(hurt_count),
            "neutral_changed_top1_count": int(neutral_count),
            "net_top1_gain": int(improved_count - hurt_count),
        }
        return queries, summary

    def _lookup_topk_local_score(self, topk_indices, local_scores, target_index):
        matches = (topk_indices == int(target_index)).nonzero(as_tuple=False)
        if matches.numel() == 0:
            return None
        score = local_scores[int(matches[0, 0].item())].item()
        return self._safe_float(score)

    def _safe_float(self, value):
        value = float(value)
        if not math.isfinite(value):
            return None
        return value

    def _rank_positions_desc(self, scores):
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        positions = (
            torch.arange(scores.shape[1], device=scores.device)
            .unsqueeze(0)
            .expand_as(order)
        )
        ranks.scatter_(dim=1, index=order, src=positions)
        return ranks + 1

    def _candidate_precision_scores(
        self,
        rgb_local_scores,
        thermal_local_scores,
        topk_indices,
        gallery_rgb_precision_weights=None,
        gallery_thermal_precision_weights=None,
        gallery_precision_weight_valid=None,
    ):
        if (
            gallery_rgb_precision_weights is None
            or gallery_thermal_precision_weights is None
        ):
            return None, None

        rgb_weight = gallery_rgb_precision_weights[topk_indices].float()
        thermal_weight = gallery_thermal_precision_weights[topk_indices].float()
        weight_sum = (rgb_weight + thermal_weight).clamp_min(self.score_norm_eps)
        rgb_weight = rgb_weight / weight_sum
        thermal_weight = thermal_weight / weight_sum
        precision_scores = (
            rgb_local_scores * rgb_weight
            + thermal_local_scores * thermal_weight
        )
        valid_mask = torch.ones_like(precision_scores, dtype=torch.bool)
        if gallery_precision_weight_valid is not None:
            valid_mask = gallery_precision_weight_valid[topk_indices].bool()
            avg_scores = 0.5 * (rgb_local_scores + thermal_local_scores)
            precision_scores = torch.where(valid_mask, precision_scores, avg_scores)
        return precision_scores, valid_mask

    def candidate_oracle_diagnostics(
        self,
        query_token_embeddings,
        query_token_ids,
        query_global_embeddings,
        gallery_rgb_patch_tokens,
        gallery_thermal_patch_tokens,
        global_similarity,
        qids,
        gids,
        gallery_rgb_precision_weights=None,
        gallery_thermal_precision_weights=None,
        gallery_precision_weight_valid=None,
        save_scores=False,
    ):
        num_query, num_gallery = global_similarity.shape
        effective_topk = min(self.topk, num_gallery)
        config = {
            "topk": int(effective_topk),
            "requested_topk": int(self.topk),
            "token_topm": int(self.token_topm),
            "patch_topr": int(self.patch_topr),
            "save_scores": bool(save_scores),
        }
        if num_query == 0 or num_gallery == 0 or effective_topk <= 0:
            return {
                "config": config,
                "summary": {
                    "total_queries": int(num_query),
                    "old_top1_correct": 0,
                    "gt_in_topk_count": 0,
                    "wrong_old_top1_but_gt_in_topk": 0,
                    "oracle_topk_possible_gain": 0,
                    "oracle_topk_possible_R1": 0.0,
                },
                "queries": [],
                "candidates": [],
            }

        original_order = torch.argsort(global_similarity, dim=1, descending=True)
        topk_indices = original_order[:, :effective_topk]
        topk_global_scores = torch.gather(global_similarity, dim=1, index=topk_indices)
        top12_margin = self._compute_top12_margin(global_similarity)
        top1_scores = topk_global_scores[:, 0]
        if num_gallery >= 2:
            top2_scores = torch.topk(
                global_similarity, k=2, dim=1, largest=True, sorted=True
            ).values[:, 1]
        else:
            top2_scores = torch.full_like(top1_scores, float("-inf"))

        rgb_patch_pool = F.normalize(gallery_rgb_patch_tokens.float(), p=2, dim=-1)
        if (
            gallery_thermal_patch_tokens is not None
            and gallery_thermal_patch_tokens.shape[1] > 0
        ):
            thermal_patch_pool = F.normalize(
                gallery_thermal_patch_tokens.float(), p=2, dim=-1
            )
        else:
            thermal_patch_pool = None
        concat_patch_pool = F.normalize(
            self._build_patch_pool(
                gallery_rgb_patch_tokens, gallery_thermal_patch_tokens
            ).float(),
            p=2,
            dim=-1,
        )

        salient_query_tokens, salient_token_weights = self._select_salient_tokens(
            query_token_embeddings=query_token_embeddings,
            query_token_ids=query_token_ids,
            query_global_embeddings=query_global_embeddings,
        )
        concat_local_scores = global_similarity.new_zeros((num_query, effective_topk))
        rgb_local_scores = global_similarity.new_zeros((num_query, effective_topk))
        thermal_local_scores = global_similarity.new_zeros((num_query, effective_topk))
        for q_start in range(0, num_query, self.query_chunk_size):
            q_end = min(q_start + self.query_chunk_size, num_query)
            query_tokens_chunk = salient_query_tokens[q_start:q_end]
            token_weights_chunk = salient_token_weights[q_start:q_end]
            topk_indices_chunk = topk_indices[q_start:q_end]
            concat_local_scores[q_start:q_end] = self._compute_local_scores_from_pool(
                query_tokens_chunk,
                token_weights_chunk,
                concat_patch_pool,
                topk_indices_chunk,
            )
            rgb_local_scores[q_start:q_end] = self._compute_local_scores_from_pool(
                query_tokens_chunk,
                token_weights_chunk,
                rgb_patch_pool,
                topk_indices_chunk,
            )
            if thermal_patch_pool is None:
                thermal_local_scores[q_start:q_end] = rgb_local_scores[q_start:q_end]
            else:
                thermal_local_scores[q_start:q_end] = self._compute_local_scores_from_pool(
                    query_tokens_chunk,
                    token_weights_chunk,
                    thermal_patch_pool,
                    topk_indices_chunk,
                )

        avg_local_scores = 0.5 * (rgb_local_scores + thermal_local_scores)
        precision_local_scores, precision_valid_mask = self._candidate_precision_scores(
            rgb_local_scores=rgb_local_scores,
            thermal_local_scores=thermal_local_scores,
            topk_indices=topk_indices,
            gallery_rgb_precision_weights=gallery_rgb_precision_weights,
            gallery_thermal_precision_weights=gallery_thermal_precision_weights,
            gallery_precision_weight_valid=gallery_precision_weight_valid,
        )

        concat_ranks = self._rank_positions_desc(concat_local_scores)
        rgb_ranks = self._rank_positions_desc(rgb_local_scores)
        thermal_ranks = self._rank_positions_desc(thermal_local_scores)
        avg_ranks = self._rank_positions_desc(avg_local_scores)
        precision_ranks = (
            self._rank_positions_desc(precision_local_scores)
            if precision_local_scores is not None
            else None
        )

        qids_cpu = qids.detach().cpu().long().view(-1)
        gids_cpu = gids.detach().cpu().long().view(-1)
        topk_indices_cpu = topk_indices.detach().cpu().long()
        topk_global_scores_cpu = topk_global_scores.detach().cpu().float()
        top1_scores_cpu = top1_scores.detach().cpu().float()
        top2_scores_cpu = top2_scores.detach().cpu().float()
        top12_margin_cpu = top12_margin.detach().cpu().float()
        concat_scores_cpu = concat_local_scores.detach().cpu().float()
        rgb_scores_cpu = rgb_local_scores.detach().cpu().float()
        thermal_scores_cpu = thermal_local_scores.detach().cpu().float()
        avg_scores_cpu = avg_local_scores.detach().cpu().float()
        precision_scores_cpu = (
            precision_local_scores.detach().cpu().float()
            if precision_local_scores is not None
            else None
        )
        precision_valid_cpu = (
            precision_valid_mask.detach().cpu().bool()
            if precision_valid_mask is not None
            else None
        )
        concat_ranks_cpu = concat_ranks.detach().cpu().long()
        rgb_ranks_cpu = rgb_ranks.detach().cpu().long()
        thermal_ranks_cpu = thermal_ranks.detach().cpu().long()
        avg_ranks_cpu = avg_ranks.detach().cpu().long()
        precision_ranks_cpu = (
            precision_ranks.detach().cpu().long()
            if precision_ranks is not None
            else None
        )

        queries = []
        candidates = []
        old_top1_correct = 0
        gt_in_topk_count = 0
        wrong_old_top1_but_gt_in_topk = 0

        for query_index in range(num_query):
            q_pid = int(qids_cpu[query_index].item())
            old_top1_gallery_index = int(topk_indices_cpu[query_index, 0].item())
            old_top1_pid = int(gids_cpu[old_top1_gallery_index].item())
            old_top1_is_gt = old_top1_pid == q_pid
            if old_top1_is_gt:
                old_top1_correct += 1

            candidate_pids = gids_cpu[topk_indices_cpu[query_index]]
            gt_mask = candidate_pids.eq(q_pid)
            has_gt_in_topk = bool(gt_mask.any().item())
            if has_gt_in_topk:
                gt_in_topk_count += 1
                if not old_top1_is_gt:
                    wrong_old_top1_but_gt_in_topk += 1
                gt_positions = gt_mask.nonzero(as_tuple=False).view(-1)
                best_gt_global_rank = int(gt_positions[0].item()) + 1
                best_gt_local_concat_rank = int(
                    concat_ranks_cpu[query_index, gt_positions].min().item()
                )
                best_gt_local_rgb_rank = int(
                    rgb_ranks_cpu[query_index, gt_positions].min().item()
                )
                best_gt_local_thermal_rank = int(
                    thermal_ranks_cpu[query_index, gt_positions].min().item()
                )
            else:
                best_gt_global_rank = None
                best_gt_local_concat_rank = None
                best_gt_local_rgb_rank = None
                best_gt_local_thermal_rank = None

            queries.append(
                {
                    "query_index": int(query_index),
                    "q_pid": q_pid,
                    "old_top1_gallery_index": old_top1_gallery_index,
                    "old_top1_pid": old_top1_pid,
                    "old_top1_is_gt": bool(old_top1_is_gt),
                    "has_gt_in_topk": bool(has_gt_in_topk),
                    "best_gt_global_rank": best_gt_global_rank,
                    "best_gt_local_concat_rank": best_gt_local_concat_rank,
                    "best_gt_local_rgb_rank": best_gt_local_rgb_rank,
                    "best_gt_local_thermal_rank": best_gt_local_thermal_rank,
                    "topk": int(effective_topk),
                    "token_topm": int(self.token_topm),
                    "patch_topr": int(self.patch_topr),
                }
            )

            old_concat_score = concat_scores_cpu[query_index, 0].item()
            old_rgb_score = rgb_scores_cpu[query_index, 0].item()
            old_thermal_score = thermal_scores_cpu[query_index, 0].item()
            for candidate_rank in range(effective_topk):
                candidate_gallery_index = int(
                    topk_indices_cpu[query_index, candidate_rank].item()
                )
                candidate_pid = int(gids_cpu[candidate_gallery_index].item())
                is_gt_match = candidate_pid == q_pid
                candidate = {
                    "query_index": int(query_index),
                    "q_pid": q_pid,
                    "candidate_rank_global": int(candidate_rank + 1),
                    "candidate_gallery_index": candidate_gallery_index,
                    "candidate_pid": candidate_pid,
                    "is_gt_match": bool(is_gt_match),
                    "is_old_top1": bool(candidate_rank == 0),
                    "global_score": self._safe_float(
                        topk_global_scores_cpu[query_index, candidate_rank].item()
                    ),
                    "global_top1_score": self._safe_float(
                        top1_scores_cpu[query_index].item()
                    ),
                    "global_top2_score": self._safe_float(
                        top2_scores_cpu[query_index].item()
                    ),
                    "global_top12_margin": self._safe_float(
                        top12_margin_cpu[query_index].item()
                    ),
                    "local_concat_score": self._safe_float(
                        concat_scores_cpu[query_index, candidate_rank].item()
                    ),
                    "local_rgb_score": self._safe_float(
                        rgb_scores_cpu[query_index, candidate_rank].item()
                    ),
                    "local_thermal_score": self._safe_float(
                        thermal_scores_cpu[query_index, candidate_rank].item()
                    ),
                    "local_avg_score": self._safe_float(
                        avg_scores_cpu[query_index, candidate_rank].item()
                    ),
                    "local_precision_score": (
                        self._safe_float(
                            precision_scores_cpu[
                                query_index, candidate_rank
                            ].item()
                        )
                        if precision_scores_cpu is not None
                        else None
                    ),
                    "local_precision_valid": (
                        bool(precision_valid_cpu[query_index, candidate_rank].item())
                        if precision_valid_cpu is not None
                        else False
                    ),
                    "local_concat_rank_within_topk": int(
                        concat_ranks_cpu[query_index, candidate_rank].item()
                    ),
                    "local_rgb_rank_within_topk": int(
                        rgb_ranks_cpu[query_index, candidate_rank].item()
                    ),
                    "local_thermal_rank_within_topk": int(
                        thermal_ranks_cpu[query_index, candidate_rank].item()
                    ),
                    "local_avg_rank_within_topk": int(
                        avg_ranks_cpu[query_index, candidate_rank].item()
                    ),
                    "local_precision_rank_within_topk": (
                        int(precision_ranks_cpu[query_index, candidate_rank].item())
                        if precision_ranks_cpu is not None
                        else None
                    ),
                    "local_gap_to_old_top1_concat": self._safe_float(
                        concat_scores_cpu[query_index, candidate_rank].item()
                        - old_concat_score
                    ),
                    "local_gap_to_old_top1_rgb": self._safe_float(
                        rgb_scores_cpu[query_index, candidate_rank].item()
                        - old_rgb_score
                    ),
                    "local_gap_to_old_top1_thermal": self._safe_float(
                        thermal_scores_cpu[query_index, candidate_rank].item()
                        - old_thermal_score
                    ),
                    "local_gap_to_old_top1_avg": self._safe_float(
                        avg_scores_cpu[query_index, candidate_rank].item()
                        - avg_scores_cpu[query_index, 0].item()
                    ),
                    "old_top1_gallery_index": old_top1_gallery_index,
                    "old_top1_pid": old_top1_pid,
                    "old_top1_is_gt": bool(old_top1_is_gt),
                }
                candidates.append(candidate)

        oracle_possible_gain = int(wrong_old_top1_but_gt_in_topk)
        oracle_possible_r1 = (
            100.0 * float(old_top1_correct + oracle_possible_gain) / float(num_query)
            if num_query > 0
            else 0.0
        )
        summary = {
            "total_queries": int(num_query),
            "old_top1_correct": int(old_top1_correct),
            "gt_in_topk_count": int(gt_in_topk_count),
            "wrong_old_top1_but_gt_in_topk": int(wrong_old_top1_but_gt_in_topk),
            "oracle_topk_possible_gain": int(oracle_possible_gain),
            "oracle_topk_possible_R1": float(oracle_possible_r1),
        }
        return {
            "config": config,
            "summary": summary,
            "queries": queries,
            "candidates": candidates,
        }

    def gate_rerank(
        self,
        query_token_embeddings,
        query_token_ids,
        query_global_embeddings,
        gallery_rgb_patch_tokens,
        gallery_thermal_patch_tokens,
        global_similarity,
        qids=None,
        gids=None,
        mode="concat",
        candidate_rank_max=2,
        local_rank_max=1,
        global_margin_max=0.01,
        local_gap_min=0.01,
        return_stats=False,
        return_diagnostics=False,
    ):
        mode = str(mode).lower()
        if mode not in {"concat", "avg", "rgb", "thermal"}:
            raise ValueError(f"Unsupported qalr_v3_gate_mode={mode!r}.")

        candidate_rank_max = int(candidate_rank_max)
        local_rank_max = int(local_rank_max)
        global_margin_max = float(global_margin_max)
        local_gap_min = float(local_gap_min)

        num_query, num_gallery = global_similarity.shape
        if (
            num_query == 0
            or num_gallery == 0
            or candidate_rank_max <= 1
            or local_rank_max <= 0
        ):
            stats = {
                "diagnostic_summary": {
                    "total_queries": int(num_query),
                    "changed_top1_count": 0,
                    "improved_top1_count": 0,
                    "hurt_top1_count": 0,
                    "neutral_changed_top1_count": 0,
                    "net_top1_gain": 0,
                },
                "query_diagnostics": [],
                "effective_topk": int(min(self.topk, num_gallery))
                if num_gallery > 0
                else 0,
                "qalr_v3_gate_mode": mode,
                "qalr_v3_gate_candidate_rank_max": int(candidate_rank_max),
                "qalr_v3_gate_local_rank_max": int(local_rank_max),
                "qalr_v3_gate_global_margin_max": float(global_margin_max),
                "qalr_v3_gate_local_gap_min": float(local_gap_min),
            }
            if return_stats:
                return global_similarity, stats
            return global_similarity

        effective_topk = min(self.topk, num_gallery)
        original_order = torch.argsort(global_similarity, dim=1, descending=True)
        topk_indices = original_order[:, :effective_topk]
        top12_margin = self._compute_top12_margin(global_similarity)

        rgb_patch_pool = F.normalize(gallery_rgb_patch_tokens.float(), p=2, dim=-1)
        if (
            gallery_thermal_patch_tokens is not None
            and gallery_thermal_patch_tokens.shape[1] > 0
        ):
            thermal_patch_pool = F.normalize(
                gallery_thermal_patch_tokens.float(), p=2, dim=-1
            )
        else:
            thermal_patch_pool = None
        concat_patch_pool = F.normalize(
            self._build_patch_pool(
                gallery_rgb_patch_tokens, gallery_thermal_patch_tokens
            ).float(),
            p=2,
            dim=-1,
        )

        salient_query_tokens, salient_token_weights = self._select_salient_tokens(
            query_token_embeddings=query_token_embeddings,
            query_token_ids=query_token_ids,
            query_global_embeddings=query_global_embeddings,
        )
        concat_local_scores = global_similarity.new_zeros((num_query, effective_topk))
        rgb_local_scores = global_similarity.new_zeros((num_query, effective_topk))
        thermal_local_scores = global_similarity.new_zeros((num_query, effective_topk))
        for q_start in range(0, num_query, self.query_chunk_size):
            q_end = min(q_start + self.query_chunk_size, num_query)
            query_tokens_chunk = salient_query_tokens[q_start:q_end]
            token_weights_chunk = salient_token_weights[q_start:q_end]
            topk_indices_chunk = topk_indices[q_start:q_end]
            concat_local_scores[q_start:q_end] = self._compute_local_scores_from_pool(
                query_tokens_chunk,
                token_weights_chunk,
                concat_patch_pool,
                topk_indices_chunk,
            )
            rgb_local_scores[q_start:q_end] = self._compute_local_scores_from_pool(
                query_tokens_chunk,
                token_weights_chunk,
                rgb_patch_pool,
                topk_indices_chunk,
            )
            if thermal_patch_pool is None:
                thermal_local_scores[q_start:q_end] = rgb_local_scores[q_start:q_end]
            else:
                thermal_local_scores[q_start:q_end] = self._compute_local_scores_from_pool(
                    query_tokens_chunk,
                    token_weights_chunk,
                    thermal_patch_pool,
                    topk_indices_chunk,
                )

        avg_local_scores = 0.5 * (rgb_local_scores + thermal_local_scores)
        if mode == "concat":
            local_scores = concat_local_scores
        elif mode == "avg":
            local_scores = avg_local_scores
        elif mode == "rgb":
            local_scores = rgb_local_scores
        else:
            local_scores = thermal_local_scores

        local_ranks = self._rank_positions_desc(local_scores)
        position = (
            torch.arange(effective_topk, device=global_similarity.device)
            .unsqueeze(0)
            .expand(num_query, -1)
        )
        candidate_rank = position + 1
        local_gap = local_scores - local_scores[:, :1]
        eligible_mask = (
            (position > 0)
            & (candidate_rank <= candidate_rank_max)
            & (local_ranks <= local_rank_max)
            & (top12_margin.unsqueeze(1) < global_margin_max)
            & (local_gap > local_gap_min)
        )

        selection_scores = local_scores.masked_fill(~eligible_mask, float("-inf"))
        selected_local_scores, selected_positions = selection_scores.max(dim=1)
        has_selection = torch.isfinite(selected_local_scores)
        ranking_order = original_order.clone()
        for query_index in has_selection.nonzero(as_tuple=False).view(-1):
            query_index = int(query_index.item())
            selected_pos = int(selected_positions[query_index].item())
            selected_index = topk_indices[query_index, selected_pos]
            current_order = ranking_order[query_index]
            remaining_order = current_order[current_order != selected_index]
            ranking_order[query_index] = torch.cat(
                [selected_index.view(1), remaining_order],
                dim=0,
            )

        rank_scores = torch.arange(
            num_gallery,
            0,
            -1,
            device=global_similarity.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(num_query, -1)
        reranked_similarity = torch.empty(
            global_similarity.shape,
            device=global_similarity.device,
            dtype=torch.float32,
        )
        reranked_similarity.scatter_(
            dim=1,
            index=ranking_order,
            src=rank_scores,
        )

        queries, summary = self._build_gate_diagnostics(
            qids=qids,
            gids=gids,
            original_order=original_order,
            ranking_order=ranking_order,
            topk_indices=topk_indices,
            local_scores=local_scores,
            local_ranks=local_ranks,
            local_gap=local_gap,
            top12_margin=top12_margin,
            selected_positions=selected_positions,
            has_selection=has_selection,
            mode=mode,
            candidate_rank_max=candidate_rank_max,
            local_rank_max=local_rank_max,
            global_margin_max=global_margin_max,
            local_gap_min=local_gap_min,
            include_queries=return_diagnostics,
        )
        topk_membership_safe = torch.equal(
            torch.sort(ranking_order[:, :effective_topk], dim=1).values,
            torch.sort(original_order[:, :effective_topk], dim=1).values,
        )
        tail_membership_safe = torch.equal(
            torch.sort(ranking_order[:, effective_topk:], dim=1).values,
            torch.sort(original_order[:, effective_topk:], dim=1).values,
        )
        stats = {
            "diagnostic_summary": summary,
            "query_diagnostics": queries,
            "effective_topk": int(effective_topk),
            "changed_top1_count": int(summary.get("changed_top1_count", 0)),
            "qalr_v3_gate_enabled": True,
            "qalr_v3_gate_mode": mode,
            "qalr_v3_gate_candidate_rank_max": int(candidate_rank_max),
            "qalr_v3_gate_local_rank_max": int(local_rank_max),
            "qalr_v3_gate_global_margin_max": float(global_margin_max),
            "qalr_v3_gate_local_gap_min": float(local_gap_min),
            "qalr_v3_gate_topk_boundary_safe": bool(
                topk_membership_safe and tail_membership_safe
            ),
        }
        if return_stats:
            return reranked_similarity, stats
        return reranked_similarity

    def _build_gate_diagnostics(
        self,
        qids,
        gids,
        original_order,
        ranking_order,
        topk_indices,
        local_scores,
        local_ranks,
        local_gap,
        top12_margin,
        selected_positions,
        has_selection,
        mode,
        candidate_rank_max,
        local_rank_max,
        global_margin_max,
        local_gap_min,
        include_queries,
    ):
        original_cpu = original_order.detach().cpu().long()
        ranking_cpu = ranking_order.detach().cpu().long()
        topk_cpu = topk_indices.detach().cpu().long()
        local_scores_cpu = local_scores.detach().cpu().float()
        local_ranks_cpu = local_ranks.detach().cpu().long()
        local_gap_cpu = local_gap.detach().cpu().float()
        margin_cpu = top12_margin.detach().cpu().float()
        selected_cpu = selected_positions.detach().cpu().long()
        has_selection_cpu = has_selection.detach().cpu().bool()
        qids_cpu = qids.detach().cpu().long().view(-1) if qids is not None else None
        gids_cpu = gids.detach().cpu().long().view(-1) if gids is not None else None

        queries = []
        changed_count = 0
        improved_count = 0
        hurt_count = 0
        neutral_count = 0
        old_correct_count = 0
        new_correct_count = 0
        for query_index in range(original_cpu.shape[0]):
            old_top1_index = int(original_cpu[query_index, 0].item())
            new_top1_index = int(ranking_cpu[query_index, 0].item())
            changed_top1 = old_top1_index != new_top1_index

            q_pid = None
            old_top1_pid = None
            new_top1_pid = None
            gt_match_old = False
            gt_match_new = False
            if qids_cpu is not None and gids_cpu is not None:
                q_pid = int(qids_cpu[query_index].item())
                old_top1_pid = int(gids_cpu[old_top1_index].item())
                new_top1_pid = int(gids_cpu[new_top1_index].item())
                gt_match_old = old_top1_pid == q_pid
                gt_match_new = new_top1_pid == q_pid
                if gt_match_old:
                    old_correct_count += 1
                if gt_match_new:
                    new_correct_count += 1

            improved_top1 = (not gt_match_old) and gt_match_new
            hurt_top1 = gt_match_old and (not gt_match_new)
            neutral_changed_top1 = (
                changed_top1 and (not improved_top1) and (not hurt_top1)
            )
            if changed_top1:
                changed_count += 1
            if improved_top1:
                improved_count += 1
            if hurt_top1:
                hurt_count += 1
            if neutral_changed_top1:
                neutral_count += 1

            selected_candidate_global_rank = None
            selected_candidate_local_rank = None
            selected_local_gap = None
            if bool(has_selection_cpu[query_index].item()):
                selected_pos = int(selected_cpu[query_index].item())
                selected_candidate_global_rank = selected_pos + 1
                selected_candidate_local_rank = int(
                    local_ranks_cpu[query_index, selected_pos].item()
                )
                selected_local_gap = self._safe_float(
                    local_gap_cpu[query_index, selected_pos].item()
                )

            if include_queries:
                queries.append(
                    {
                        "query_index": int(query_index),
                        "q_pid": q_pid,
                        "old_top1_index": old_top1_index,
                        "old_top1_pid": old_top1_pid,
                        "new_top1_index": new_top1_index,
                        "new_top1_pid": new_top1_pid,
                        "changed_top1": bool(changed_top1),
                        "improved_top1": bool(improved_top1),
                        "hurt_top1": bool(hurt_top1),
                        "neutral_changed_top1": bool(neutral_changed_top1),
                        "global_top12_margin": self._safe_float(
                            margin_cpu[query_index].item()
                        ),
                        "local_gap": selected_local_gap,
                        "local_score_old_top1": self._safe_float(
                            local_scores_cpu[query_index, 0].item()
                        ),
                        "local_score_new_top1": self._lookup_topk_local_score(
                            topk_cpu[query_index],
                            local_scores_cpu[query_index],
                            new_top1_index,
                        ),
                        "selected_candidate_global_rank": selected_candidate_global_rank,
                        "selected_candidate_local_rank": selected_candidate_local_rank,
                        "gate_mode": str(mode),
                        "gate_candidate_rank_max": int(candidate_rank_max),
                        "gate_local_rank_max": int(local_rank_max),
                        "gate_global_margin_max": float(global_margin_max),
                        "gate_local_gap_min": float(local_gap_min),
                    }
                )

        total_queries = int(original_cpu.shape[0])
        summary = {
            "total_queries": total_queries,
            "changed_top1_count": int(changed_count),
            "improved_top1_count": int(improved_count),
            "hurt_top1_count": int(hurt_count),
            "neutral_changed_top1_count": int(neutral_count),
            "net_top1_gain": int(improved_count - hurt_count),
            "old_top1_correct": int(old_correct_count),
            "new_top1_correct": int(new_correct_count),
            "old_R1": (
                100.0 * float(old_correct_count) / float(total_queries)
                if total_queries > 0 and qids_cpu is not None and gids_cpu is not None
                else None
            ),
            "R1": (
                100.0 * float(new_correct_count) / float(total_queries)
                if total_queries > 0 and qids_cpu is not None and gids_cpu is not None
                else None
            ),
        }
        return queries, summary

    def _build_focus_candidate_mask(self, global_similarity, topk_indices, effective_topk):
        num_query = global_similarity.shape[0]
        if not self.use_focus_topn:
            return torch.ones_like(topk_indices, dtype=torch.bool), effective_topk

        effective_focus_topn = min(self.focus_topn, effective_topk)
        focus_indices = torch.topk(
            global_similarity,
            k=effective_focus_topn,
            dim=1,
            largest=True,
            sorted=True,
        ).indices
        focus_candidate_mask = (
            topk_indices.unsqueeze(-1) == focus_indices.unsqueeze(1)
        ).any(dim=-1)
        if num_query == 0:
            return focus_candidate_mask, effective_focus_topn
        return focus_candidate_mask, effective_focus_topn

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

    def _rerank_v2(
        self,
        query_token_embeddings,
        query_token_ids,
        query_global_embeddings,
        gallery_rgb_patch_tokens,
        gallery_thermal_patch_tokens,
        gallery_rgb_precision_weights,
        gallery_thermal_precision_weights,
        gallery_precision_weight_valid,
        global_similarity,
        qids=None,
        gids=None,
        return_stats=False,
        return_diagnostics=False,
    ):
        num_query, num_gallery = global_similarity.shape
        if num_query == 0 or num_gallery == 0:
            if return_stats:
                return global_similarity, {}
            return global_similarity

        rgb_patch_pool = F.normalize(gallery_rgb_patch_tokens.float(), p=2, dim=-1)
        if gallery_thermal_patch_tokens is not None and gallery_thermal_patch_tokens.shape[1] > 0:
            thermal_patch_pool = F.normalize(
                gallery_thermal_patch_tokens.float(), p=2, dim=-1
            )
        else:
            thermal_patch_pool = None
        concat_patch_pool = F.normalize(
            self._build_patch_pool(gallery_rgb_patch_tokens, gallery_thermal_patch_tokens).float(),
            p=2,
            dim=-1,
        )

        salient_query_tokens, salient_token_weights = self._select_salient_tokens(
            query_token_embeddings=query_token_embeddings,
            query_token_ids=query_token_ids,
            query_global_embeddings=query_global_embeddings,
        )

        effective_topk = min(self.topk, num_gallery)
        original_order = torch.argsort(global_similarity, dim=1, descending=True)
        topk_indices = original_order[:, :effective_topk]
        topk_scores = torch.gather(global_similarity, dim=1, index=topk_indices)
        top12_margin = self._compute_top12_margin(global_similarity)
        alpha_eff = self._build_qalr_v2_alpha(top12_margin)
        query_gate_mask, _ = self._build_query_gate_mask(global_similarity)
        focus_candidate_mask, effective_focus_topn = self._build_focus_candidate_mask(
            global_similarity=global_similarity,
            topk_indices=topk_indices,
            effective_topk=effective_topk,
        )

        ranking_order = original_order.clone()
        used_local_scores = global_similarity.new_zeros((num_query, effective_topk))
        raw_local_means = []
        raw_local_stds = []
        used_local_means = []
        used_local_stds = []
        safe_local_std_mins = []
        rgb_local_means = []
        thermal_local_means = []
        concat_local_means = []
        top1_guard_freeze_count = 0
        promote_only_freeze_count = 0
        precision_used_pair_count = 0
        precision_fallback_pair_count = 0

        for q_start in range(0, num_query, self.query_chunk_size):
            q_end = min(q_start + self.query_chunk_size, num_query)
            query_tokens_chunk = salient_query_tokens[q_start:q_end]
            token_weights_chunk = salient_token_weights[q_start:q_end]
            topk_indices_chunk = topk_indices[q_start:q_end]
            query_gate_mask_chunk = query_gate_mask[q_start:q_end]
            focus_candidate_mask_chunk = focus_candidate_mask[q_start:q_end]
            topk_scores_chunk = topk_scores[q_start:q_end]
            margin_chunk = top12_margin[q_start:q_end]
            alpha_chunk = alpha_eff[q_start:q_end].unsqueeze(1)

            concat_local_scores_chunk = self._compute_local_scores_from_pool(
                query_tokens_chunk,
                token_weights_chunk,
                concat_patch_pool,
                topk_indices_chunk,
            )
            rgb_local_scores_chunk = self._compute_local_scores_from_pool(
                query_tokens_chunk,
                token_weights_chunk,
                rgb_patch_pool,
                topk_indices_chunk,
            )
            if thermal_patch_pool is None:
                thermal_local_scores_chunk = rgb_local_scores_chunk
            else:
                thermal_local_scores_chunk = self._compute_local_scores_from_pool(
                    query_tokens_chunk,
                    token_weights_chunk,
                    thermal_patch_pool,
                    topk_indices_chunk,
                )

            modality_fuse = self.qalr_v2_modality_fuse
            precision_used_mask = None
            if modality_fuse == "concat":
                local_scores_chunk = concat_local_scores_chunk
            elif modality_fuse == "avg":
                local_scores_chunk = 0.5 * (
                    rgb_local_scores_chunk + thermal_local_scores_chunk
                )
            elif modality_fuse == "rgb":
                local_scores_chunk = rgb_local_scores_chunk
            elif modality_fuse == "thermal":
                local_scores_chunk = thermal_local_scores_chunk
            elif modality_fuse == "precision":
                if (
                    gallery_rgb_precision_weights is not None
                    and gallery_thermal_precision_weights is not None
                ):
                    rgb_weight = gallery_rgb_precision_weights[topk_indices_chunk].float()
                    thermal_weight = gallery_thermal_precision_weights[
                        topk_indices_chunk
                    ].float()
                    weight_valid = torch.ones_like(rgb_weight, dtype=torch.bool)
                    if gallery_precision_weight_valid is not None:
                        weight_valid = gallery_precision_weight_valid[
                            topk_indices_chunk
                        ].bool()
                    weight_sum = (rgb_weight + thermal_weight).clamp_min(
                        self.score_norm_eps
                    )
                    rgb_weight = rgb_weight / weight_sum
                    thermal_weight = thermal_weight / weight_sum
                    local_scores_chunk = (
                        rgb_local_scores_chunk * rgb_weight
                        + thermal_local_scores_chunk * thermal_weight
                    )
                    avg_scores = 0.5 * (
                        rgb_local_scores_chunk + thermal_local_scores_chunk
                    )
                    local_scores_chunk = torch.where(
                        weight_valid,
                        local_scores_chunk,
                        avg_scores,
                    )
                    precision_used_mask = weight_valid
                else:
                    local_scores_chunk = 0.5 * (
                        rgb_local_scores_chunk + thermal_local_scores_chunk
                    )
            else:
                raise ValueError(
                    "Unsupported qalr_v2_modality_fuse="
                    f"{self.qalr_v2_modality_fuse!r}."
                )

            raw_local_means.append(local_scores_chunk.mean(dim=1).detach().cpu())
            raw_local_stds.append(
                local_scores_chunk.std(dim=1, unbiased=False).detach().cpu()
            )
            rgb_local_means.append(rgb_local_scores_chunk.mean(dim=1).detach().cpu())
            thermal_local_means.append(
                thermal_local_scores_chunk.mean(dim=1).detach().cpu()
            )
            concat_local_means.append(
                concat_local_scores_chunk.mean(dim=1).detach().cpu()
            )
            if precision_used_mask is not None:
                precision_used_pair_count += int(precision_used_mask.sum().item())
                precision_fallback_pair_count += int(
                    (~precision_used_mask).sum().item()
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
            used_local_scores[q_start:q_end] = fused_local_scores_chunk

            if self.use_score_norm:
                base_topk_scores_chunk, _ = self._zscore_querywise(topk_scores_chunk)
            else:
                base_topk_scores_chunk = topk_scores_chunk
            final_topk_scores = base_topk_scores_chunk + alpha_chunk * fused_local_scores_chunk
            if self.use_focus_topn:
                final_topk_scores = torch.where(
                    focus_candidate_mask_chunk,
                    final_topk_scores,
                    base_topk_scores_chunk,
                )
            if self.use_query_gate:
                final_topk_scores = torch.where(
                    query_gate_mask_chunk.unsqueeze(1),
                    final_topk_scores,
                    base_topk_scores_chunk,
                )

            freeze_top1_mask = torch.zeros(
                q_end - q_start,
                dtype=torch.bool,
                device=global_similarity.device,
            )
            if self.qalr_v2_top1_guard:
                guard_mask = margin_chunk >= self.qalr_v2_freeze_top1_margin
                freeze_top1_mask = freeze_top1_mask | guard_mask
                top1_guard_freeze_count += int(guard_mask.sum().item())
            if self.qalr_v2_promote_only:
                if effective_topk > 1:
                    local_advantage = (
                        fused_local_scores_chunk[:, 1:].max(dim=1).values
                        - fused_local_scores_chunk[:, 0]
                    )
                    allow_promote = local_advantage > self.qalr_v2_promote_margin
                else:
                    allow_promote = torch.zeros_like(freeze_top1_mask)
                promote_freeze_mask = ~allow_promote
                freeze_top1_mask = freeze_top1_mask | promote_freeze_mask
                promote_only_freeze_count += int(promote_freeze_mask.sum().item())
            if freeze_top1_mask.any():
                row_max = final_topk_scores.max(dim=1).values
                epsilon = row_max.abs() * 1e-6 + 1e-6
                final_topk_scores = final_topk_scores.clone()
                final_topk_scores[freeze_top1_mask, 0] = (
                    row_max[freeze_top1_mask] + epsilon[freeze_top1_mask]
                )

            topk_rerank_order = torch.argsort(
                final_topk_scores, dim=1, descending=True, stable=True
            )
            reranked_topk_indices = torch.gather(
                topk_indices_chunk, dim=1, index=topk_rerank_order
            )
            if self.use_query_gate:
                reranked_topk_indices = torch.where(
                    query_gate_mask_chunk.unsqueeze(1),
                    reranked_topk_indices,
                    topk_indices_chunk,
                )
            ranking_order[q_start:q_end, :effective_topk] = reranked_topk_indices

        rank_scores = torch.arange(
            num_gallery,
            0,
            -1,
            device=global_similarity.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(num_query, -1)
        reranked_similarity = torch.empty(
            global_similarity.shape,
            device=global_similarity.device,
            dtype=torch.float32,
        )
        reranked_similarity.scatter_(
            dim=1,
            index=ranking_order,
            src=rank_scores,
        )
        output_similarity = global_similarity if self.qalr_v2_diag_only else reranked_similarity

        reranked_query_mask = (
            ranking_order[:, :effective_topk] != original_order[:, :effective_topk]
        ).any(dim=1)
        changed_top1_mask = ranking_order[:, 0] != original_order[:, 0]
        gated_query_count = (
            int(query_gate_mask.sum().item()) if self.use_query_gate else int(num_query)
        )
        reranked_query_count = int(reranked_query_mask.sum().item())
        changed_top1_count = int(changed_top1_mask.sum().item())
        topk_membership_safe = torch.equal(
            torch.sort(ranking_order[:, :effective_topk], dim=1).values,
            torch.sort(original_order[:, :effective_topk], dim=1).values,
        )
        tail_membership_safe = torch.equal(
            torch.sort(ranking_order[:, effective_topk:], dim=1).values,
            torch.sort(original_order[:, effective_topk:], dim=1).values,
        )
        topk_boundary_safe = bool(topk_membership_safe and tail_membership_safe)
        if self.use_query_gate and (~query_gate_mask).any():
            ungated_order_safe = torch.equal(
                ranking_order[~query_gate_mask],
                original_order[~query_gate_mask],
            )
            topk_boundary_safe = bool(topk_boundary_safe and ungated_order_safe)

        if not return_stats:
            return output_similarity

        raw_local_means = torch.cat(raw_local_means, dim=0)
        raw_local_stds = torch.cat(raw_local_stds, dim=0)
        rgb_local_means = torch.cat(rgb_local_means, dim=0)
        thermal_local_means = torch.cat(thermal_local_means, dim=0)
        concat_local_means = torch.cat(concat_local_means, dim=0)
        queries, diag_summary = self._build_diagnostics(
            qids=qids,
            gids=gids,
            original_order=original_order,
            ranking_order=ranking_order,
            topk_indices=topk_indices,
            local_scores_topk=used_local_scores,
            top12_margin=top12_margin,
            alpha_eff=alpha_eff,
            effective_topk=effective_topk,
            modality_fuse=self.qalr_v2_modality_fuse,
        ) if return_diagnostics else ([], {})
        stats = {
            "raw_local_mean": self._summarize_querywise_tensor(raw_local_means),
            "raw_local_std": self._summarize_querywise_tensor(raw_local_stds),
            "rgb_local_mean": self._summarize_querywise_tensor(rgb_local_means),
            "thermal_local_mean": self._summarize_querywise_tensor(
                thermal_local_means
            ),
            "concat_local_mean": self._summarize_querywise_tensor(
                concat_local_means
            ),
            "effective_topk": float(effective_topk),
            "reranked_query_count": reranked_query_count,
            "gated_query_count": gated_query_count,
            "changed_top1_count": changed_top1_count,
            "topk_boundary_safe": topk_boundary_safe,
            "fusion_norm": (
                "querywise_global_local_zscore"
                if self.use_score_norm
                else "legacy_raw_global_plus_local"
            ),
            "qalr_v2_enabled": True,
            "qalr_v2_diag_only": bool(self.qalr_v2_diag_only),
            "qalr_v2_modality_fuse": self.qalr_v2_modality_fuse,
            "qalr_v2_alpha_min": self.qalr_v2_alpha_min,
            "qalr_v2_alpha_max": self.qalr_v2_alpha_max,
            "qalr_v2_margin_low": self.qalr_v2_margin_low,
            "qalr_v2_margin_high": self.qalr_v2_margin_high,
            "qalr_v2_top1_guard": bool(self.qalr_v2_top1_guard),
            "qalr_v2_promote_only": bool(self.qalr_v2_promote_only),
            "qalr_v2_top1_guard_freeze_count": int(top1_guard_freeze_count),
            "qalr_v2_promote_only_freeze_count": int(promote_only_freeze_count),
            "precision_used_pair_count": int(precision_used_pair_count),
            "precision_fallback_pair_count": int(precision_fallback_pair_count),
            "alpha_eff": self._summarize_querywise_tensor(
                alpha_eff.detach().cpu()
            ),
            "query_gate_margin": self._summarize_querywise_tensor(
                top12_margin.detach().cpu()
            ),
            "gated_query_ratio": query_gate_mask.float().mean().item(),
            "focused_candidate_ratio": focus_candidate_mask.float().mean().item(),
            "effective_focus_topn": float(effective_focus_topn),
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
        if return_diagnostics:
            stats["query_diagnostics"] = queries
            stats["diagnostic_summary"] = diag_summary
            stats["_proposed_similarity"] = reranked_similarity
        return output_similarity, stats

    def rerank(
        self,
        query_token_embeddings,
        query_token_ids,
        query_global_embeddings,
        gallery_rgb_patch_tokens,
        gallery_thermal_patch_tokens,
        global_similarity,
        gallery_rgb_precision_weights=None,
        gallery_thermal_precision_weights=None,
        gallery_precision_weight_valid=None,
        qids=None,
        gids=None,
        return_stats=False,
        return_diagnostics=False,
    ):
        if self.use_qalr_v2:
            return self._rerank_v2(
                query_token_embeddings=query_token_embeddings,
                query_token_ids=query_token_ids,
                query_global_embeddings=query_global_embeddings,
                gallery_rgb_patch_tokens=gallery_rgb_patch_tokens,
                gallery_thermal_patch_tokens=gallery_thermal_patch_tokens,
                gallery_rgb_precision_weights=gallery_rgb_precision_weights,
                gallery_thermal_precision_weights=gallery_thermal_precision_weights,
                gallery_precision_weight_valid=gallery_precision_weight_valid,
                global_similarity=global_similarity,
                qids=qids,
                gids=gids,
                return_stats=return_stats,
                return_diagnostics=return_diagnostics,
            )

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
        original_order = torch.argsort(global_similarity, dim=1, descending=True)
        topk_indices = original_order[:, :effective_topk]
        topk_scores = torch.gather(global_similarity, dim=1, index=topk_indices)
        query_gate_mask, top12_margin = self._build_query_gate_mask(global_similarity)
        diagnostic_top12_margin = None
        diagnostic_alpha_eff = None
        if return_diagnostics:
            diagnostic_top12_margin = (
                top12_margin
                if top12_margin is not None
                else self._compute_top12_margin(global_similarity)
            )
            diagnostic_alpha_eff = global_similarity.new_full(
                (num_query,), float(self.alpha)
            )
        focus_candidate_mask, effective_focus_topn = self._build_focus_candidate_mask(
            global_similarity=global_similarity,
            topk_indices=topk_indices,
            effective_topk=effective_topk,
        )
        ranking_order = original_order.clone()
        raw_local_means = []
        raw_local_stds = []
        used_local_means = []
        used_local_stds = []
        safe_local_std_mins = []
        diagnostic_local_scores = (
            global_similarity.new_zeros((num_query, effective_topk))
            if return_diagnostics
            else None
        )

        for q_start in range(0, num_query, self.query_chunk_size):
            q_end = min(q_start + self.query_chunk_size, num_query)
            query_tokens_chunk = salient_query_tokens[q_start:q_end]
            token_weights_chunk = salient_token_weights[q_start:q_end]
            topk_indices_chunk = topk_indices[q_start:q_end]
            query_gate_mask_chunk = query_gate_mask[q_start:q_end]
            focus_candidate_mask_chunk = focus_candidate_mask[q_start:q_end]
            topk_scores_chunk = topk_scores[q_start:q_end]
            local_scores_chunk = torch.zeros_like(topk_scores_chunk)

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
            if diagnostic_local_scores is not None:
                diagnostic_local_scores[q_start:q_end] = fused_local_scores_chunk
            if self.use_score_norm:
                used_local_means.append(
                    fused_local_scores_chunk.mean(dim=1).detach().cpu()
                )
                used_local_stds.append(
                    fused_local_scores_chunk.std(dim=1, unbiased=False).detach().cpu()
                )
                safe_local_std_mins.append(safe_local_std.min().detach().cpu())

            if self.use_score_norm:
                base_topk_scores_chunk, _ = self._zscore_querywise(topk_scores_chunk)
            else:
                base_topk_scores_chunk = topk_scores_chunk
            final_topk_scores = (
                base_topk_scores_chunk + self.alpha * fused_local_scores_chunk
            )
            if self.use_focus_topn:
                final_topk_scores = torch.where(
                    focus_candidate_mask_chunk,
                    final_topk_scores,
                    base_topk_scores_chunk,
                )
            if self.use_query_gate:
                final_topk_scores = torch.where(
                    query_gate_mask_chunk.unsqueeze(1),
                    final_topk_scores,
                    base_topk_scores_chunk,
                )
            topk_rerank_order = torch.argsort(
                final_topk_scores, dim=1, descending=True, stable=True
            )
            reranked_topk_indices = torch.gather(
                topk_indices_chunk, dim=1, index=topk_rerank_order
            )
            if self.use_query_gate:
                reranked_topk_indices = torch.where(
                    query_gate_mask_chunk.unsqueeze(1),
                    reranked_topk_indices,
                    topk_indices_chunk,
                )
            ranking_order[q_start:q_end, :effective_topk] = reranked_topk_indices

        rank_scores = torch.arange(
            num_gallery,
            0,
            -1,
            device=global_similarity.device,
            dtype=torch.float32,
        ).unsqueeze(0).expand(num_query, -1)
        reranked_similarity = torch.empty(
            global_similarity.shape,
            device=global_similarity.device,
            dtype=torch.float32,
        )
        reranked_similarity.scatter_(
            dim=1,
            index=ranking_order,
            src=rank_scores,
        )
        reranked_query_mask = (
            ranking_order[:, :effective_topk] != original_order[:, :effective_topk]
        ).any(dim=1)
        changed_top1_mask = ranking_order[:, 0] != original_order[:, 0]
        if self.use_query_gate:
            gated_query_count = int(query_gate_mask.sum().item())
        else:
            gated_query_count = int(num_query)
        reranked_query_count = int(reranked_query_mask.sum().item())
        changed_top1_count = int(changed_top1_mask.sum().item())

        topk_membership_safe = torch.equal(
            torch.sort(ranking_order[:, :effective_topk], dim=1).values,
            torch.sort(original_order[:, :effective_topk], dim=1).values,
        )
        tail_membership_safe = torch.equal(
            torch.sort(ranking_order[:, effective_topk:], dim=1).values,
            torch.sort(original_order[:, effective_topk:], dim=1).values,
        )
        topk_boundary_safe = bool(topk_membership_safe and tail_membership_safe)
        if self.use_query_gate and (~query_gate_mask).any():
            ungated_order_safe = torch.equal(
                ranking_order[~query_gate_mask],
                original_order[~query_gate_mask],
            )
            topk_boundary_safe = bool(topk_boundary_safe and ungated_order_safe)

        if not return_stats:
            return reranked_similarity

        raw_local_means = torch.cat(raw_local_means, dim=0)
        raw_local_stds = torch.cat(raw_local_stds, dim=0)
        queries, diag_summary = self._build_diagnostics(
            qids=qids,
            gids=gids,
            original_order=original_order,
            ranking_order=ranking_order,
            topk_indices=topk_indices,
            local_scores_topk=diagnostic_local_scores,
            top12_margin=diagnostic_top12_margin,
            alpha_eff=diagnostic_alpha_eff,
            effective_topk=effective_topk,
            modality_fuse="concat",
        ) if return_diagnostics else ([], {})
        stats = {
            "raw_local_mean": self._summarize_querywise_tensor(raw_local_means),
            "raw_local_std": self._summarize_querywise_tensor(raw_local_stds),
            "effective_topk": float(effective_topk),
            "reranked_query_count": reranked_query_count,
            "gated_query_count": gated_query_count,
            "changed_top1_count": changed_top1_count,
            "topk_boundary_safe": topk_boundary_safe,
            "fusion_norm": (
                "querywise_global_local_zscore"
                if self.use_score_norm
                else "legacy_raw_global_plus_local"
            ),
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
        stats["focused_candidate_ratio"] = focus_candidate_mask.float().mean().item()
        stats["effective_focus_topn"] = float(effective_focus_topn)
        if return_diagnostics:
            stats["query_diagnostics"] = queries
            stats["diagnostic_summary"] = diag_summary
            stats["_proposed_similarity"] = reranked_similarity

        return reranked_similarity, stats
