from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging

from utils.query_adaptive_rerank import QueryAdaptiveLocalReranker
from utils.transport_rerank import build_transport_reranker
from utils.mtp_text import build_rgb_text, build_thermal_text
from utils.simple_tokenizer import SimpleTokenizer


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    # similarity: [num_query, num_gallery], larger means more similar.
    if get_mAP:
        # Full sort is needed when mAP/mINP are required.
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # For CMC only, top-k is faster than full sort.
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    # Predicted gallery pid for each query rank position.
    pred_labels = g_pids[indices.cpu()]  # q * k
    # Binary match matrix: True if predicted pid equals query pid.
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    # Compute CMC curve.
    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    # mINP considers the rank position of the last relevant match.
    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    # mAP is mean of per-query average precision.
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")
        self.last_eval_result = None
        self._mtp_tokenizer = None

    def _get_mtp_tokenizer(self):
        if self._mtp_tokenizer is None:
            self._mtp_tokenizer = SimpleTokenizer()
        return self._mtp_tokenizer

    def _decode_caption_batch(self, caption_ids):
        tokenizer = self._get_mtp_tokenizer()
        sot_token = int(tokenizer.encoder["<|startoftext|>"])
        eot_token = int(tokenizer.encoder["<|endoftext|>"])
        caption_ids_cpu = caption_ids.detach().cpu()
        captions = []
        for row in caption_ids_cpu:
            token_ids = []
            for token_id in row.tolist():
                token_id = int(token_id)
                if token_id == 0 or token_id == sot_token:
                    continue
                if token_id == eot_token:
                    break
                token_ids.append(token_id)
            if token_ids:
                captions.append(" ".join(tokenizer.decode(token_ids).split()))
            else:
                captions.append("")
        return captions

    def _tokenize_text_batch(self, texts, text_length, device):
        tokenizer = self._get_mtp_tokenizer()
        sot_token = int(tokenizer.encoder["<|startoftext|>"])
        eot_token = int(tokenizer.encoder["<|endoftext|>"])
        output = torch.zeros(len(texts), int(text_length), dtype=torch.long)
        for row_index, text in enumerate(texts):
            token_ids = [sot_token] + tokenizer.encode(text) + [eot_token]
            if len(token_ids) > int(text_length):
                token_ids = token_ids[: int(text_length)]
                token_ids[-1] = eot_token
            output[row_index, : len(token_ids)] = torch.tensor(
                token_ids, dtype=torch.long
            )
        return output.to(device)

    def _build_mtp_query_token_views(self, caption_ids, model):
        captions = self._decode_caption_batch(caption_ids)
        color_drop_mode = str(
            getattr(getattr(model, "args", None), "mtp_color_drop_mode", "delete")
        )
        rgb_texts = [build_rgb_text(caption) for caption in captions]
        th_texts = [
            build_thermal_text(caption, color_drop_mode=color_drop_mode)
            for caption in captions
        ]
        text_length = int(caption_ids.shape[1])
        device = caption_ids.device
        caption_ids_rgb = self._tokenize_text_batch(
            rgb_texts, text_length=text_length, device=device
        )
        caption_ids_th = self._tokenize_text_batch(
            th_texts, text_length=text_length, device=device
        )
        return caption_ids_rgb, caption_ids_th

    def _encode_text_feature(self, model, caption, return_local_tokens=False):
        if return_local_tokens and hasattr(model, "encode_text_eval"):
            try:
                return model.encode_text_eval(
                    caption, return_local_tokens=True
                )
            except TypeError:
                pass
        if return_local_tokens and hasattr(model, "encode_text_eval_with_tokens"):
            return model.encode_text_eval_with_tokens(caption)
        if hasattr(model, "encode_text_eval"):
            return model.encode_text_eval(caption)
        # Prefer explicit deterministic text encoder when provided by model.
        if hasattr(model, "encode_text_deterministic"):
            if return_local_tokens:
                raise AttributeError(
                    "Model must implement encode_text_eval(..., return_local_tokens=True) "
                    "or encode_text_eval_with_tokens() when local eval tokens are requested."
                )
            return model.encode_text_deterministic(caption)
        if return_local_tokens:
            raise AttributeError(
                "Model must implement an eval-time text token return path when "
                "local eval tokens are requested."
            )
        return model.encode_text(caption)

    def _encode_gallery_feature(
        self, model, img, thermal_img=None, return_local_tokens=False
    ):
        if return_local_tokens and hasattr(model, "encode_rgbt_eval"):
            try:
                return model.encode_rgbt_eval(
                    img,
                    thermal_img,
                    return_local_tokens=True,
                )
            except TypeError:
                pass
        if return_local_tokens and hasattr(model, "encode_rgbt_eval_with_patches"):
            return model.encode_rgbt_eval_with_patches(img, thermal_img)
        if hasattr(model, "encode_rgbt_eval"):
            return model.encode_rgbt_eval(img, thermal_img)
        # Prefer explicit deterministic fused visual encoder when available.
        if thermal_img is not None and hasattr(model, "encode_rgbt_deterministic"):
            if return_local_tokens:
                raise AttributeError(
                    "Model must implement encode_rgbt_eval(..., return_local_tokens=True) "
                    "or encode_rgbt_eval_with_patches() when local eval tokens are requested."
                )
            return model.encode_rgbt_deterministic(img, thermal_img)
        if thermal_img is None:
            if return_local_tokens:
                raise AttributeError(
                    "Model must implement an eval-time visual token return path when "
                    "local eval tokens are requested."
                )
            return model.encode_image(img)
        # Fallback for models without dedicated RGBT helper.
        if return_local_tokens:
            raise AttributeError(
                "Model must implement an eval-time visual token return path when "
                "local eval tokens are requested."
            )
        rgb_feat = model.encode_image(img)
        t_feat = model.encode_image(thermal_img)
        return rgb_feat + t_feat

    def _compute_embedding(
        self, model, return_local_tokens=False, use_mtp_eval_fusion=False
    ):
        # Encode all queries and gallery samples once, then evaluate by matrix ops.
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        qfeats_rgb, qfeats_th = [], []
        gfeats_rgb, gfeats_th = [], []
        query_token_embeddings, query_token_ids, query_global_embeddings = [], [], []
        gallery_rgb_patch_tokens, gallery_thermal_patch_tokens = [], []
        gallery_rgb_precision_weights = []
        gallery_thermal_precision_weights = []
        gallery_precision_weight_valid = []
        need_rich_gallery_outputs = bool(return_local_tokens or use_mtp_eval_fusion)
        # Encode text queries.
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_outputs = self._encode_text_feature(
                    model,
                    caption,
                    return_local_tokens=return_local_tokens,
                )
            qids.append(pid.view(-1)) # flatten 
            if return_local_tokens:
                qfeats.append(text_outputs["global_feat"])
                query_token_embeddings.append(text_outputs["token_embeddings"])
                query_token_ids.append(caption)
                query_global_embeddings.append(
                    text_outputs["global_token_embedding"]
                )
            else:
                qfeats.append(text_outputs)
            if use_mtp_eval_fusion:
                caption_rgb, caption_th = self._build_mtp_query_token_views(
                    caption, model
                )
                with torch.no_grad():
                    text_outputs_rgb = self._encode_text_feature(
                        model,
                        caption_rgb,
                        return_local_tokens=False,
                    )
                    text_outputs_th = self._encode_text_feature(
                        model,
                        caption_th,
                        return_local_tokens=False,
                    )
                if isinstance(text_outputs_rgb, dict):
                    text_outputs_rgb = text_outputs_rgb["global_feat"]
                if isinstance(text_outputs_th, dict):
                    text_outputs_th = text_outputs_th["global_feat"]
                qfeats_rgb.append(text_outputs_rgb)
                qfeats_th.append(text_outputs_th)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        if use_mtp_eval_fusion:
            qfeats_rgb = torch.cat(qfeats_rgb, 0)
            qfeats_th = torch.cat(qfeats_th, 0)
        if return_local_tokens:
            query_token_embeddings = torch.cat(query_token_embeddings, 0)
            query_token_ids = torch.cat(query_token_ids, 0)
            query_global_embeddings = torch.cat(query_global_embeddings, 0)

        # Encode gallery images.
        for data in self.img_loader:
            # Compatible with:
            #   (pid, img) for RGB-only gallery
            #   (pid, rgb_img, thermal_img) for RGB-T gallery
            if len(data) == 3:
                pid, img, thermal_img = data
                img = img.to(device)
                thermal_img = thermal_img.to(device)
            else:
                pid, img = data
                img = img.to(device)
                thermal_img = None
            with torch.no_grad():
                image_outputs = self._encode_gallery_feature(
                    model,
                    img,
                    thermal_img,
                    return_local_tokens=need_rich_gallery_outputs,
                )
            gids.append(pid.view(-1)) # flatten 
            if need_rich_gallery_outputs:
                if not isinstance(image_outputs, dict):
                    image_outputs = {"global_feat": image_outputs}
                gfeats.append(image_outputs["global_feat"])
                if use_mtp_eval_fusion:
                    gfeats_rgb.append(
                        image_outputs.get("rgb_global_feat", image_outputs["global_feat"])
                    )
                    gfeats_th.append(
                        image_outputs.get(
                            "thermal_global_feat",
                            torch.zeros_like(image_outputs["global_feat"]),
                        )
                    )
                if return_local_tokens:
                    gallery_rgb_patch_tokens.append(
                        image_outputs["rgb_patch_tokens"]
                    )
                    gallery_thermal_patch_tokens.append(
                        image_outputs["thermal_patch_tokens"]
                    )
                    batch_size = image_outputs["global_feat"].shape[0]
                    rgb_precision_weight = image_outputs.get("rgb_precision_weight", None)
                    thermal_precision_weight = image_outputs.get(
                        "thermal_precision_weight", None
                    )
                    precision_weight_valid = image_outputs.get(
                        "precision_weight_valid", None
                    )
                    if rgb_precision_weight is None:
                        rgb_precision_weight = image_outputs["global_feat"].new_full(
                            (batch_size,), 0.5
                        )
                    if thermal_precision_weight is None:
                        thermal_precision_weight = image_outputs["global_feat"].new_full(
                            (batch_size,), 0.5
                        )
                    if precision_weight_valid is None:
                        precision_weight_valid = torch.zeros(
                            batch_size,
                            device=image_outputs["global_feat"].device,
                            dtype=torch.bool,
                        )
                    gallery_rgb_precision_weights.append(
                        rgb_precision_weight.float()
                    )
                    gallery_thermal_precision_weights.append(
                        thermal_precision_weight.float()
                    )
                    gallery_precision_weight_valid.append(
                        precision_weight_valid.bool()
                    )
            else:
                gfeats.append(image_outputs)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        if use_mtp_eval_fusion:
            gfeats_rgb = torch.cat(gfeats_rgb, 0)
            gfeats_th = torch.cat(gfeats_th, 0)

        if return_local_tokens:
            gallery_rgb_patch_tokens = torch.cat(gallery_rgb_patch_tokens, 0)
            gallery_thermal_patch_tokens = torch.cat(
                gallery_thermal_patch_tokens, 0
            )
            gallery_rgb_precision_weights = torch.cat(
                gallery_rgb_precision_weights, 0
            )
            gallery_thermal_precision_weights = torch.cat(
                gallery_thermal_precision_weights, 0
            )
            gallery_precision_weight_valid = torch.cat(
                gallery_precision_weight_valid, 0
            )
            outputs = {
                "qfeats": qfeats,
                "gfeats": gfeats,
                "qids": qids,
                "gids": gids,
                "query_token_embeddings": query_token_embeddings,
                "query_token_ids": query_token_ids,
                "query_global_embeddings": query_global_embeddings,
                "gallery_rgb_patch_tokens": gallery_rgb_patch_tokens,
                "gallery_thermal_patch_tokens": gallery_thermal_patch_tokens,
                "gallery_rgb_precision_weights": gallery_rgb_precision_weights,
                "gallery_thermal_precision_weights": gallery_thermal_precision_weights,
                "gallery_precision_weight_valid": gallery_precision_weight_valid,
            }
            if use_mtp_eval_fusion:
                outputs.update(
                    {
                        "qfeats_rgb": qfeats_rgb,
                        "qfeats_th": qfeats_th,
                        "gfeats_rgb": gfeats_rgb,
                        "gfeats_th": gfeats_th,
                    }
                )
            return outputs
        if use_mtp_eval_fusion:
            return {
                "qfeats": qfeats,
                "gfeats": gfeats,
                "qids": qids,
                "gids": gids,
                "qfeats_rgb": qfeats_rgb,
                "qfeats_th": qfeats_th,
                "gfeats_rgb": gfeats_rgb,
                "gfeats_th": gfeats_th,
            }
        return qfeats, gfeats, qids, gids

    def _compute_embedding_with_local(self, model):
        # Backward-compatible wrapper for existing local-rerank callers.
        return self._compute_embedding(model, return_local_tokens=True)
    
    def eval(self, model, i2t_metric=False, return_local_tokens=False):
        use_qalr = bool(getattr(getattr(model, "args", None), "use_qalr", False))
        use_transport_rerank = bool(
            getattr(getattr(model, "args", None), "use_transport_rerank", False)
        )
        use_mtp_eval_fusion = bool(
            getattr(getattr(model, "args", None), "use_modality_text_prompt", False)
        )
        return_local_tokens = bool(
            return_local_tokens or use_qalr or use_transport_rerank
        )

        # 1) Get embeddings.
        if return_local_tokens or use_mtp_eval_fusion:
            embedding_outputs = self._compute_embedding(
                model,
                return_local_tokens=return_local_tokens,
                use_mtp_eval_fusion=use_mtp_eval_fusion,
            )
            qfeats = embedding_outputs["qfeats"]
            gfeats = embedding_outputs["gfeats"]
            qids = embedding_outputs["qids"]
            gids = embedding_outputs["gids"]
        else:
            qfeats, gfeats, qids, gids = self._compute_embedding(model)

        # 2) L2 normalize before cosine similarity.
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        # 3) Text-to-image similarity matrix.
        similarity = qfeats @ gfeats.t()
        if use_mtp_eval_fusion:
            alpha = float(getattr(model.args, "mtp_eval_fusion_alpha", 0.12))
            beta = float(getattr(model.args, "mtp_eval_fusion_beta", 0.03))
            qfeats_rgb = F.normalize(embedding_outputs["qfeats_rgb"], p=2, dim=1)
            qfeats_th = F.normalize(embedding_outputs["qfeats_th"], p=2, dim=1)
            gfeats_rgb = F.normalize(embedding_outputs["gfeats_rgb"], p=2, dim=1)
            gfeats_th = F.normalize(embedding_outputs["gfeats_th"], p=2, dim=1)
            similarity = (
                similarity
                + alpha * (qfeats_rgb @ gfeats_rgb.t())
                + beta * (qfeats_th @ gfeats_th.t())
            )
        baseline_similarity = similarity.clone()
        if use_qalr:
            use_qalr_score_norm = bool(
                getattr(model.args, "use_qalr_score_norm", False)
            )
            qalr_score_norm_type = str(
                getattr(model.args, "qalr_score_norm_type", "zscore")
            )
            use_qalr_query_gate = bool(
                getattr(model.args, "use_qalr_query_gate", False)
            )
            qalr_query_gate_type = str(
                getattr(model.args, "qalr_query_gate_type", "top12_margin")
            )
            qalr_query_gate_margin = float(
                getattr(model.args, "qalr_query_gate_margin", 0.05)
            )
            use_qalr_focus_topn = bool(
                getattr(model.args, "use_qalr_focus_topn", False)
            )
            qalr_focus_topn = int(getattr(model.args, "qalr_focus_topn", 16))
            reranker = QueryAdaptiveLocalReranker(
                topk=int(getattr(model.args, "rerank_topk", 128)),
                alpha=float(getattr(model.args, "rerank_alpha", 0.2)),
                token_topm=int(getattr(model.args, "rerank_token_topm", 8)),
                patch_topr=int(getattr(model.args, "rerank_patch_topr", 4)),
                use_score_norm=use_qalr_score_norm,
                score_norm_type=qalr_score_norm_type,
                use_query_gate=use_qalr_query_gate,
                query_gate_type=qalr_query_gate_type,
                query_gate_margin=qalr_query_gate_margin,
                use_focus_topn=use_qalr_focus_topn,
                focus_topn=qalr_focus_topn,
                vocab_size=int(getattr(model.args, "vocab_size", 49408)),
            )
            similarity, rerank_stats = reranker.rerank(
                query_token_embeddings=embedding_outputs["query_token_embeddings"],
                query_token_ids=embedding_outputs["query_token_ids"],
                query_global_embeddings=embedding_outputs["query_global_embeddings"],
                gallery_rgb_patch_tokens=embedding_outputs["gallery_rgb_patch_tokens"],
                gallery_thermal_patch_tokens=embedding_outputs["gallery_thermal_patch_tokens"],
                global_similarity=similarity,
                return_stats=True,
            )
            self.logger.info(
                "QALR enabled: topk=%d, alpha=%.4f, token_topm=%d, patch_topr=%d, use_qalr_score_norm=%s, qalr_score_norm_type=%s, use_qalr_query_gate=%s, qalr_query_gate_type=%s, qalr_query_gate_margin=%.6f, use_qalr_focus_topn=%s, qalr_focus_topn=%d",
                getattr(model.args, "rerank_topk", 128),
                getattr(model.args, "rerank_alpha", 0.2),
                getattr(model.args, "rerank_token_topm", 8),
                getattr(model.args, "rerank_patch_topr", 4),
                use_qalr_score_norm,
                qalr_score_norm_type,
                use_qalr_query_gate,
                qalr_query_gate_type,
                qalr_query_gate_margin,
                use_qalr_focus_topn,
                qalr_focus_topn,
            )
            raw_local_mean = rerank_stats.get("raw_local_mean", {})
            raw_local_std = rerank_stats.get("raw_local_std", {})
            log_msg = (
                "QALR local score summary: raw_mean_avg=%.4f, raw_std_avg=%.4f, "
                "raw_std_min=%.6f"
            )
            log_args = [
                raw_local_mean.get("avg", 0.0),
                raw_local_std.get("avg", 0.0),
                raw_local_std.get("min", 0.0),
            ]
            if use_qalr_score_norm:
                used_local_mean = rerank_stats.get("used_local_mean", {})
                used_local_std = rerank_stats.get("used_local_std", {})
                log_msg += (
                    ", used_mean_avg=%.4f, used_std_avg=%.4f, safe_std_min=%.6f"
                )
                log_args.extend(
                    [
                        used_local_mean.get("avg", 0.0),
                        used_local_std.get("avg", 0.0),
                        rerank_stats.get("safe_local_std_min", 0.0),
                    ]
                )
            log_msg += ", gated_query_ratio=%.4f"
            log_args.append(rerank_stats.get("gated_query_ratio", 0.0))
            log_msg += ", focused_candidate_ratio=%.4f"
            log_args.append(rerank_stats.get("focused_candidate_ratio", 0.0))
            if use_qalr_query_gate:
                query_gate_margin = rerank_stats.get("query_gate_margin", {})
                log_msg += ", top12_margin_avg=%.4f"
                log_args.extend(
                    [
                        query_gate_margin.get("avg", 0.0),
                    ]
                )
            if use_qalr_focus_topn:
                log_msg += ", effective_focus_topn=%.0f/%.0f"
                log_args.extend(
                    [
                        rerank_stats.get("effective_focus_topn", 0.0),
                        rerank_stats.get("effective_topk", 0.0),
                    ]
                )
            self.logger.info(log_msg, *log_args)

        if use_transport_rerank:
            if use_qalr:
                self.logger.warning(
                    "Both use_qalr and use_transport_rerank are enabled. "
                    "Transport rerank will use baseline global similarity and override "
                    "the earlier QALR scores."
                )
            reranker = build_transport_reranker(model.args)
            if reranker is None:
                raise RuntimeError(
                    "use_transport_rerank=True but transport reranker was not built."
                )
            similarity, transport_stats = reranker.rerank(
                query_token_embeddings=embedding_outputs["query_token_embeddings"],
                query_token_ids=embedding_outputs["query_token_ids"],
                query_global_embeddings=embedding_outputs["query_global_embeddings"],
                gallery_rgb_patch_tokens=embedding_outputs["gallery_rgb_patch_tokens"],
                gallery_thermal_patch_tokens=embedding_outputs["gallery_thermal_patch_tokens"],
                gallery_rgb_precision_weights=embedding_outputs.get(
                    "gallery_rgb_precision_weights", None
                ),
                gallery_thermal_precision_weights=embedding_outputs.get(
                    "gallery_thermal_precision_weights", None
                ),
                gallery_precision_weight_valid=embedding_outputs.get(
                    "gallery_precision_weight_valid", None
                ),
                global_similarity=baseline_similarity,
                return_stats=True,
            )
            self.logger.info("[EVAL] transport rerank enabled")
            self.logger.info(
                "[EVAL] topk=%d, lambda=%.4f, fuse_mode=%s, token_topm=%d, patch_topr=%d, eps=%.2e, iters=%d, score_norm=%s, use_patch_filter=%s, use_token_filter=%s",
                getattr(model.args, "transport_topk", 128),
                getattr(model.args, "transport_lambda", 0.1),
                getattr(model.args, "transport_fuse_mode", "avg"),
                getattr(model.args, "transport_token_topm", 8),
                getattr(model.args, "transport_patch_topr", 4),
                getattr(model.args, "transport_eps", 1e-6),
                getattr(model.args, "transport_iters", 20),
                getattr(model.args, "transport_score_norm", "none"),
                getattr(model.args, "transport_use_patch_filter", False),
                getattr(model.args, "transport_use_token_filter", False),
            )
            self.logger.info(
                "[EVAL] transport local score summary: raw_mean_avg=%.4f, raw_std_avg=%.4f, used_mean_avg=%.4f, used_std_avg=%.4f, rgb_mean_avg=%.4f, ir_mean_avg=%.4f, effective_topk=%d, empty_token_queries=%d, rank_changed=%d, precision_used_pairs=%d, precision_fallback_pairs=%d",
                transport_stats.get("raw_local_mean", {}).get("avg", 0.0),
                transport_stats.get("raw_local_std", {}).get("avg", 0.0),
                transport_stats.get("used_local_mean", {}).get("avg", 0.0),
                transport_stats.get("used_local_std", {}).get("avg", 0.0),
                transport_stats.get("rgb_local_mean", {}).get("avg", 0.0),
                transport_stats.get("thermal_local_mean", {}).get("avg", 0.0),
                int(transport_stats.get("effective_topk", 0)),
                int(transport_stats.get("num_empty_token_queries", 0)),
                int(transport_stats.get("num_rank_changed", 0)),
                int(transport_stats.get("precision_used_pairs", 0)),
                int(transport_stats.get("precision_fallback_pairs", 0)),
            )

        # 4) Compute retrieval metrics.
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            # Optional image-to-text evaluation.
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        self.last_eval_result = {
            "R1": float(t2i_cmc[0]),
            "R5": float(t2i_cmc[4]),
            "R10": float(t2i_cmc[9]),
            "mAP": float(t2i_mAP),
            "mINP": float(t2i_mINP),
        }
        
        return t2i_cmc[0]
