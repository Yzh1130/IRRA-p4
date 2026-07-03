from prettytable import PrettyTable
import hashlib
import json
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging

from utils.query_adaptive_rerank import QueryAdaptiveLocalReranker
from utils.transport_rerank import build_transport_reranker
from utils.mtp_text import build_rgb_text, build_thermal_text
from utils.simple_tokenizer import SimpleTokenizer
from utils.attribute_parser import COLOR_CLASSES, parse_basic_attributes


def _stable_hash(value):
    encoded = json.dumps(value, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _tensor_shape(tensor):
    return list(tensor.shape) if torch.is_tensor(tensor) else []


USPC_SLOT_RERANK_COLOR_SLOTS = {
    "upper_color": ("upper", "{} upper clothing"),
    "lower_color": ("lower", "{} lower clothing"),
    "shoe_color": ("shoes", "{} shoes"),
    "bag_color": ("bag", "{} bag"),
}

USPC_SLOT_RERANK_HAIR_LABELS = ("short", "long")


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
    def __init__(self, img_loader, txt_loader, source="eval"):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")
        self.last_eval_result = None
        self._mtp_tokenizer = None
        self._eval_degrade_generators = {}
        self.source = source

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

    def _get_eval_degrade_generator(self, device, seed):
        key = str(device)
        if key not in self._eval_degrade_generators:
            try:
                generator = torch.Generator(device=device)
            except (TypeError, RuntimeError):
                device_type = getattr(device, "type", str(device).split(":")[0])
                generator = torch.Generator(device=device_type)
            self._eval_degrade_generators[key] = generator.manual_seed(int(seed))
        return self._eval_degrade_generators[key]

    def _clamp_like_eval_image(self, value, reference):
        if not torch.is_tensor(value):
            return value
        if value.dim() <= 1:
            return value
        reduce_dims = tuple(range(1, reference.dim()))
        ref = reference.detach().float()
        min_value = ref.amin(dim=reduce_dims, keepdim=True).to(dtype=value.dtype)
        max_value = ref.amax(dim=reduce_dims, keepdim=True).to(dtype=value.dtype)
        return torch.max(torch.min(value, max_value), min_value)

    def _add_eval_noise(self, tensor, strength, seed):
        if tensor is None:
            return None
        if float(strength) <= 0.0:
            return tensor
        generator = self._get_eval_degrade_generator(tensor.device, seed)
        noise = torch.randn(
            tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype,
            generator=generator,
        )
        noisy = tensor + float(strength) * noise
        return self._clamp_like_eval_image(noisy, tensor)

    def _blur_eval_image(self, tensor, kernel_size):
        if tensor is None:
            return None
        kernel_size = int(kernel_size)
        if kernel_size <= 1:
            return tensor
        pad_total = kernel_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padded = F.pad(
            tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="replicate",
        )
        return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)

    def _apply_eval_degradation(self, model, img, thermal_img):
        model_args = getattr(model, "args", None)
        mode = str(getattr(model_args, "eval_degrade_mode", "none")).lower()
        if mode == "none":
            return img, thermal_img

        strength = float(getattr(model_args, "eval_degrade_strength", 1.0))
        blur_kernel = int(getattr(model_args, "eval_degrade_blur_kernel", 7))
        seed = int(getattr(model_args, "eval_degrade_seed", 2026))

        if mode == "rgb_missing":
            img = torch.zeros_like(img)
        elif mode == "ir_missing":
            if thermal_img is not None:
                thermal_img = torch.zeros_like(thermal_img)
        elif mode == "rgb_noise":
            img = self._add_eval_noise(img, strength, seed)
        elif mode == "ir_noise":
            thermal_img = self._add_eval_noise(thermal_img, strength, seed)
        elif mode == "both_noise":
            img = self._add_eval_noise(img, strength, seed)
            thermal_img = self._add_eval_noise(thermal_img, strength, seed)
        elif mode == "rgb_blur":
            img = self._blur_eval_image(img, blur_kernel)
        elif mode == "ir_blur":
            thermal_img = self._blur_eval_image(thermal_img, blur_kernel)
        elif mode == "both_blur":
            img = self._blur_eval_image(img, blur_kernel)
            thermal_img = self._blur_eval_image(thermal_img, blur_kernel)
        else:
            raise ValueError(f"Unsupported eval_degrade_mode: {mode}.")
        return img, thermal_img

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
            img, thermal_img = self._apply_eval_degradation(
                model,
                img,
                thermal_img,
            )
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

    def _log_feature_summary(self, qfeats, gfeats):
        qfeats_float = qfeats.detach().float()
        gfeats_float = gfeats.detach().float()
        self.logger.info(
            "[EVAL_FEATURE_SUMMARY] source=%s image_features_shape=%s text_features_shape=%s image_features_norm_mean=%.6f text_features_norm_mean=%.6f",
            self.source,
            _tensor_shape(gfeats),
            _tensor_shape(qfeats),
            float(gfeats_float.norm(p=2, dim=1).mean().item()) if gfeats_float.numel() else 0.0,
            float(qfeats_float.norm(p=2, dim=1).mean().item()) if qfeats_float.numel() else 0.0,
        )

    def _log_score_summary(self, similarity):
        sim_float = similarity.detach().float()
        if sim_float.numel():
            top1 = torch.argmax(sim_float, dim=1).detach().cpu().tolist()
            sim_mean = float(sim_float.mean().item())
            sim_std = float(sim_float.std(unbiased=False).item())
        else:
            top1 = []
            sim_mean = 0.0
            sim_std = 0.0
        self.logger.info(
            "[EVAL_SCORE_SUMMARY] source=%s sim_shape=%s sim_mean=%.6f sim_std=%.6f sim_top1_hash=%s",
            self.source,
            _tensor_shape(similarity),
            sim_mean,
            sim_std,
            _stable_hash(top1),
        )

    def _build_eval_order_hash(self, qids, gids, similarity):
        return _stable_hash(
            {
                "txt_pids": qids.detach().cpu().view(-1).tolist(),
                "img_pids": gids.detach().cpu().view(-1).tolist(),
                "score_shape": _tensor_shape(similarity),
            }
        )

    def _score_semantics(
        self,
        use_qalr,
        use_qalr_v2,
        use_qalr_v3_gate,
        use_qalr_v3_two_stage_gate,
        use_transport_rerank,
        use_uspc_slot_rerank,
        use_mtp_eval_fusion,
    ):
        if use_transport_rerank:
            return "transport_rerank_score"
        if use_qalr_v3_two_stage_gate:
            return "qalr_v3_two_stage_gate_rank_score"
        if use_qalr_v3_gate:
            return "qalr_v3_gate_rank_score"
        if use_qalr_v2:
            return "qalr_v2_rank_score"
        if use_qalr:
            return "qalr_rank_score"
        if use_uspc_slot_rerank:
            return "uspc_slot_rerank_score"
        if use_mtp_eval_fusion:
            return "global_cosine_similarity_with_mtp_eval_fusion"
        return "global_cosine_similarity"

    def _collect_qalr_args(self, model_args):
        return {
            "rerank_topk": int(getattr(model_args, "rerank_topk", 128)),
            "rerank_alpha": float(getattr(model_args, "rerank_alpha", 0.2)),
            "rerank_token_topm": int(
                getattr(model_args, "rerank_token_topm", 8)
            ),
            "rerank_patch_topr": int(
                getattr(model_args, "rerank_patch_topr", 4)
            ),
            "use_qalr_score_norm": bool(
                getattr(model_args, "use_qalr_score_norm", False)
            ),
            "qalr_score_norm_type": str(
                getattr(model_args, "qalr_score_norm_type", "zscore")
            ),
            "use_qalr_query_gate": bool(
                getattr(model_args, "use_qalr_query_gate", False)
            ),
            "qalr_query_gate_type": str(
                getattr(model_args, "qalr_query_gate_type", "top12_margin")
            ),
            "qalr_query_gate_margin": float(
                getattr(model_args, "qalr_query_gate_margin", 0.05)
            ),
            "use_qalr_focus_topn": bool(
                getattr(model_args, "use_qalr_focus_topn", False)
            ),
            "qalr_focus_topn": int(getattr(model_args, "qalr_focus_topn", 16)),
            "use_qalr_v2": bool(getattr(model_args, "use_qalr_v2", False)),
            "qalr_v2_save_diag": bool(
                getattr(model_args, "qalr_v2_save_diag", False)
            ),
            "qalr_v2_diag_file": str(
                getattr(model_args, "qalr_v2_diag_file", "")
            ),
            "qalr_v2_alpha_min": float(
                getattr(model_args, "qalr_v2_alpha_min", 0.0)
            ),
            "qalr_v2_alpha_max": float(
                getattr(model_args, "qalr_v2_alpha_max", 0.01)
            ),
            "qalr_v2_margin_low": float(
                getattr(model_args, "qalr_v2_margin_low", 0.0)
            ),
            "qalr_v2_margin_high": float(
                getattr(model_args, "qalr_v2_margin_high", 0.02)
            ),
            "qalr_v2_freeze_top1_margin": float(
                getattr(model_args, "qalr_v2_freeze_top1_margin", 0.02)
            ),
            "qalr_v2_promote_margin": float(
                getattr(model_args, "qalr_v2_promote_margin", 0.0)
            ),
            "qalr_v2_modality_fuse": str(
                getattr(model_args, "qalr_v2_modality_fuse", "concat")
            ),
            "qalr_v2_top1_guard": bool(
                getattr(model_args, "qalr_v2_top1_guard", False)
            ),
            "qalr_v2_promote_only": bool(
                getattr(model_args, "qalr_v2_promote_only", False)
            ),
            "qalr_v2_diag_only": bool(
                getattr(model_args, "qalr_v2_diag_only", False)
            ),
            "qalr_v3_candidate_diag": bool(
                getattr(model_args, "qalr_v3_candidate_diag", False)
            ),
            "qalr_v3_candidate_topk": int(
                getattr(model_args, "qalr_v3_candidate_topk", 64)
            ),
            "qalr_v3_candidate_token_topm": int(
                getattr(model_args, "qalr_v3_candidate_token_topm", 8)
            ),
            "qalr_v3_candidate_patch_topr": int(
                getattr(model_args, "qalr_v3_candidate_patch_topr", 4)
            ),
            "qalr_v3_candidate_save_scores": bool(
                getattr(model_args, "qalr_v3_candidate_save_scores", False)
            ),
            "use_qalr_v3_gate": bool(
                getattr(model_args, "use_qalr_v3_gate", False)
            ),
            "qalr_v3_gate_mode": str(
                getattr(model_args, "qalr_v3_gate_mode", "concat")
            ),
            "qalr_v3_gate_topk": int(
                getattr(model_args, "qalr_v3_gate_topk", 64)
            ),
            "qalr_v3_gate_token_topm": int(
                getattr(model_args, "qalr_v3_gate_token_topm", 8)
            ),
            "qalr_v3_gate_patch_topr": int(
                getattr(model_args, "qalr_v3_gate_patch_topr", 4)
            ),
            "qalr_v3_gate_candidate_rank_max": int(
                getattr(model_args, "qalr_v3_gate_candidate_rank_max", 2)
            ),
            "qalr_v3_gate_local_rank_max": int(
                getattr(model_args, "qalr_v3_gate_local_rank_max", 1)
            ),
            "qalr_v3_gate_global_margin_max": float(
                getattr(model_args, "qalr_v3_gate_global_margin_max", 0.01)
            ),
            "qalr_v3_gate_local_gap_min": float(
                getattr(model_args, "qalr_v3_gate_local_gap_min", 0.01)
            ),
            "qalr_v3_gate_save_diag": bool(
                getattr(model_args, "qalr_v3_gate_save_diag", False)
            ),
            "qalr_v3_gate_diag_file": str(
                getattr(model_args, "qalr_v3_gate_diag_file", "")
            ),
            "use_qalr_v3_two_stage_gate": bool(
                getattr(model_args, "use_qalr_v3_two_stage_gate", False)
            ),
            "qalr_v3_stage1_mode": str(
                getattr(model_args, "qalr_v3_stage1_mode", "avg")
            ),
            "qalr_v3_stage1_global_margin_max": float(
                getattr(model_args, "qalr_v3_stage1_global_margin_max", 0.008)
            ),
            "qalr_v3_stage1_local_gap_min": float(
                getattr(model_args, "qalr_v3_stage1_local_gap_min", 0.015)
            ),
            "qalr_v3_stage2_mode": str(
                getattr(model_args, "qalr_v3_stage2_mode", "avg")
            ),
            "qalr_v3_stage2_global_margin_min": float(
                getattr(model_args, "qalr_v3_stage2_global_margin_min", 0.008)
            ),
            "qalr_v3_stage2_global_margin_max": float(
                getattr(model_args, "qalr_v3_stage2_global_margin_max", 0.010)
            ),
            "qalr_v3_stage2_local_gap_min": float(
                getattr(model_args, "qalr_v3_stage2_local_gap_min", 0.011)
            ),
            "qalr_v3_two_stage_topk": int(
                getattr(model_args, "qalr_v3_two_stage_topk", 64)
            ),
            "qalr_v3_two_stage_candidate_rank_max": int(
                getattr(model_args, "qalr_v3_two_stage_candidate_rank_max", 2)
            ),
            "qalr_v3_two_stage_local_rank_max": int(
                getattr(model_args, "qalr_v3_two_stage_local_rank_max", 1)
            ),
            "qalr_v3_two_stage_token_topm": int(
                getattr(model_args, "qalr_v3_two_stage_token_topm", 8)
            ),
            "qalr_v3_two_stage_patch_topr": int(
                getattr(model_args, "qalr_v3_two_stage_patch_topr", 4)
            ),
            "qalr_v3_two_stage_save_diag": bool(
                getattr(model_args, "qalr_v3_two_stage_save_diag", False)
            ),
            "qalr_v3_two_stage_diag_file": str(
                getattr(model_args, "qalr_v3_two_stage_diag_file", "")
            ),
        }

    def _maybe_save_eval_scores(
        self,
        model_args,
        similarity,
        raw_similarity,
        qids,
        gids,
        metrics,
        use_qalr,
        use_qalr_v2,
        use_qalr_v3_gate,
        use_qalr_v3_two_stage_gate,
        use_transport_rerank,
        use_uspc_slot_rerank,
        use_mtp_eval_fusion,
    ):
        if model_args is None or not bool(
            getattr(model_args, "save_eval_scores", False)
        ):
            return
        if self.source == "train_validation":
            self.logger.warning(
                "[EVAL_SCORE_DUMP] skipped during train_validation"
            )
            return

        save_file = str(getattr(model_args, "eval_scores_file", "")).strip()
        if not save_file:
            raise ValueError(
                "--save_eval_scores=True requires --eval_scores_file to be set."
            )

        sim_cpu = similarity.detach().cpu().float()
        txt_pids = qids.detach().cpu().long().view(-1)
        img_pids = gids.detach().cpu().long().view(-1)
        order_hash = self._build_eval_order_hash(txt_pids, img_pids, sim_cpu)
        score_semantics = self._score_semantics(
            use_qalr=use_qalr,
            use_qalr_v2=use_qalr_v2,
            use_qalr_v3_gate=use_qalr_v3_gate,
            use_qalr_v3_two_stage_gate=use_qalr_v3_two_stage_gate,
            use_transport_rerank=use_transport_rerank,
            use_uspc_slot_rerank=use_uspc_slot_rerank,
            use_mtp_eval_fusion=use_mtp_eval_fusion,
        )
        metadata = {
            "dataset_name": str(getattr(model_args, "dataset_name", "")),
            "eval_dataset": str(getattr(model_args, "eval_dataset", "")),
            "score_shape": list(sim_cpu.shape),
            "score_semantics": score_semantics,
            "use_qalr": bool(use_qalr),
            "use_qalr_v2": bool(use_qalr_v2),
            "use_qalr_v3_gate": bool(use_qalr_v3_gate),
            "use_qalr_v3_two_stage_gate": bool(use_qalr_v3_two_stage_gate),
            "use_qalr_v3_candidate_diag": bool(
                getattr(model_args, "qalr_v3_candidate_diag", False)
            ),
            "qalr_args": self._collect_qalr_args(model_args),
            "use_transport_rerank": bool(use_transport_rerank),
            "use_uspc_slot_rerank": bool(use_uspc_slot_rerank),
            "use_modality_text_prompt": bool(use_mtp_eval_fusion),
            "config_file": str(getattr(model_args, "config_file", "")),
            "test_ckpt_file": str(getattr(model_args, "test_ckpt_file", "")),
            "model_name": str(getattr(model_args, "name", "")),
            "output_dir": str(getattr(model_args, "output_dir", "")),
            "eval_source": str(self.source),
            "order_hash": order_hash,
            "raw_similarity_saved": bool(
                getattr(model_args, "eval_scores_save_raw", False)
            ),
        }
        payload = {
            "sim": sim_cpu,
            "txt_pids": txt_pids,
            "img_pids": img_pids,
            "txt_indices": torch.arange(sim_cpu.shape[0], dtype=torch.long),
            "img_indices": torch.arange(sim_cpu.shape[1], dtype=torch.long),
            "metrics": dict(metrics),
            "metadata": metadata,
            "order_hash": order_hash,
        }
        if bool(getattr(model_args, "eval_scores_save_raw", False)):
            if raw_similarity is None:
                raise RuntimeError(
                    "--eval_scores_save_raw was set, but raw similarity is unavailable."
                )
            payload["raw_sim"] = raw_similarity.detach().cpu().float()

        save_file = os.path.abspath(save_file)
        save_dir = os.path.dirname(save_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(payload, save_file)
        self.logger.info(
            "[EVAL_SCORE_DUMP] path=%s shape=%s order_hash=%s semantics=%s raw_saved=%s",
            save_file,
            list(sim_cpu.shape),
            order_hash,
            score_semantics,
            bool("raw_sim" in payload),
        )

    def _compute_metric_summary(self, similarity, qids, gids):
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
            similarity=similarity,
            q_pids=qids,
            g_pids=gids,
            max_rank=10,
            get_mAP=True,
        )
        return {
            "R1": float(t2i_cmc[0].item()),
            "R5": float(t2i_cmc[4].item()),
            "R10": float(t2i_cmc[9].item()),
            "mAP": float(t2i_mAP.item()),
            "mINP": float(t2i_mINP.item()),
        }

    def _save_qalr_diagnostics(
        self,
        model_args,
        rerank_stats,
        before_metrics,
        after_metrics,
        use_qalr_v2,
    ):
        if model_args is None or not bool(
            getattr(model_args, "qalr_v2_save_diag", False)
        ):
            return
        diag_file = str(getattr(model_args, "qalr_v2_diag_file", "")).strip()
        if not diag_file:
            default_name = "qalr_v2_diag.json" if use_qalr_v2 else "qalr_diag.json"
            diag_file = os.path.join(
                str(getattr(model_args, "output_dir", ".")),
                default_name,
            )
        diag_file = os.path.abspath(diag_file)
        diag_dir = os.path.dirname(diag_file)
        if diag_dir:
            os.makedirs(diag_dir, exist_ok=True)

        summary = dict(rerank_stats.get("diagnostic_summary", {}))
        summary.setdefault(
            "total_queries",
            len(rerank_stats.get("query_diagnostics", [])),
        )
        summary.setdefault(
            "changed_top1_count",
            int(rerank_stats.get("changed_top1_count", 0)),
        )
        summary.setdefault("improved_top1_count", 0)
        summary.setdefault("hurt_top1_count", 0)
        summary.setdefault("neutral_changed_top1_count", 0)
        summary.setdefault(
            "net_top1_gain",
            int(summary.get("improved_top1_count", 0))
            - int(summary.get("hurt_top1_count", 0)),
        )
        for prefix, metrics in [
            ("before", before_metrics),
            ("after", after_metrics),
        ]:
            if metrics is None:
                continue
            for key, value in metrics.items():
                summary[f"{prefix}_{key}"] = float(value)

        payload = {
            "summary": summary,
            "queries": list(rerank_stats.get("query_diagnostics", [])),
            "config": {
                "use_qalr_v2": bool(use_qalr_v2),
                "use_qalr": bool(getattr(model_args, "use_qalr", False)),
                "rerank_topk": int(getattr(model_args, "rerank_topk", 128)),
                "rerank_alpha": float(getattr(model_args, "rerank_alpha", 0.2)),
                "rerank_token_topm": int(
                    getattr(model_args, "rerank_token_topm", 8)
                ),
                "rerank_patch_topr": int(
                    getattr(model_args, "rerank_patch_topr", 4)
                ),
                "use_qalr_score_norm": bool(
                    getattr(model_args, "use_qalr_score_norm", False)
                ),
                "use_qalr_query_gate": bool(
                    getattr(model_args, "use_qalr_query_gate", False)
                ),
                "use_qalr_focus_topn": bool(
                    getattr(model_args, "use_qalr_focus_topn", False)
                ),
                "qalr_focus_topn": int(
                    getattr(model_args, "qalr_focus_topn", 16)
                ),
                "qalr_v2_alpha_min": float(
                    getattr(model_args, "qalr_v2_alpha_min", 0.0)
                ),
                "qalr_v2_alpha_max": float(
                    getattr(model_args, "qalr_v2_alpha_max", 0.01)
                ),
                "qalr_v2_margin_low": float(
                    getattr(model_args, "qalr_v2_margin_low", 0.0)
                ),
                "qalr_v2_margin_high": float(
                    getattr(model_args, "qalr_v2_margin_high", 0.02)
                ),
                "qalr_v2_freeze_top1_margin": float(
                    getattr(model_args, "qalr_v2_freeze_top1_margin", 0.02)
                ),
                "qalr_v2_promote_margin": float(
                    getattr(model_args, "qalr_v2_promote_margin", 0.0)
                ),
                "qalr_v2_modality_fuse": str(
                    getattr(model_args, "qalr_v2_modality_fuse", "concat")
                ),
                "qalr_v2_top1_guard": bool(
                    getattr(model_args, "qalr_v2_top1_guard", False)
                ),
                "qalr_v2_promote_only": bool(
                    getattr(model_args, "qalr_v2_promote_only", False)
                ),
                "qalr_v2_diag_only": bool(
                    getattr(model_args, "qalr_v2_diag_only", False)
                ),
            },
        }
        with open(diag_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        self.logger.info(
            "[QALR_V2_DIAG_DUMP] path=%s queries=%d",
            diag_file,
            len(payload["queries"]),
        )

    def _log_qalr_diag_summary(self, prefix, rerank_stats):
        summary = rerank_stats.get("diagnostic_summary", {})
        changed = int(
            summary.get(
                "changed_top1_count",
                rerank_stats.get("changed_top1_count", 0),
            )
        )
        improved = int(summary.get("improved_top1_count", 0))
        hurt = int(summary.get("hurt_top1_count", 0))
        neutral = int(summary.get("neutral_changed_top1_count", 0))
        net = int(summary.get("net_top1_gain", improved - hurt))
        self.logger.info(
            "%s changed=%d improved=%d hurt=%d neutral=%d net=%d",
            prefix,
            changed,
            improved,
            hurt,
            neutral,
            net,
        )

    def _save_qalr_v3_candidate_diagnostics(self, model_args, payload):
        if model_args is None or not bool(
            getattr(model_args, "qalr_v3_candidate_diag", False)
        ):
            return
        diag_file = str(
            getattr(model_args, "qalr_v3_candidate_diag_file", "")
        ).strip()
        if not diag_file:
            diag_file = os.path.join(
                str(getattr(model_args, "output_dir", ".")),
                "qalr_v3_candidate_diag.json",
            )
        diag_file = os.path.abspath(diag_file)
        diag_dir = os.path.dirname(diag_file)
        if diag_dir:
            os.makedirs(diag_dir, exist_ok=True)
        with open(diag_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

        summary = payload.get("summary", {})
        self.logger.info(
            "[QALR_V3_CANDIDATE_DIAG] path=%s total=%d old_top1_correct=%d gt_in_topk=%d wrong_old_top1_but_gt_in_topk=%d oracle_gain=%d oracle_R1=%.6f",
            diag_file,
            int(summary.get("total_queries", 0)),
            int(summary.get("old_top1_correct", 0)),
            int(summary.get("gt_in_topk_count", 0)),
            int(summary.get("wrong_old_top1_but_gt_in_topk", 0)),
            int(summary.get("oracle_topk_possible_gain", 0)),
            float(summary.get("oracle_topk_possible_R1", 0.0)),
        )

    def _save_qalr_v3_gate_diagnostics(self, model_args, rerank_stats):
        if model_args is None or not bool(
            getattr(model_args, "qalr_v3_gate_save_diag", False)
        ):
            return
        diag_file = str(getattr(model_args, "qalr_v3_gate_diag_file", "")).strip()
        if not diag_file:
            diag_file = os.path.join(
                str(getattr(model_args, "output_dir", ".")),
                "qalr_v3_gate_diag.json",
            )
        diag_file = os.path.abspath(diag_file)
        diag_dir = os.path.dirname(diag_file)
        if diag_dir:
            os.makedirs(diag_dir, exist_ok=True)

        summary = dict(rerank_stats.get("diagnostic_summary", {}))
        payload = {
            "summary": summary,
            "queries": list(rerank_stats.get("query_diagnostics", [])),
            "config": {
                "use_qalr_v3_gate": bool(
                    getattr(model_args, "use_qalr_v3_gate", False)
                ),
                "qalr_v3_gate_mode": str(
                    getattr(model_args, "qalr_v3_gate_mode", "concat")
                ),
                "qalr_v3_gate_topk": int(
                    getattr(model_args, "qalr_v3_gate_topk", 64)
                ),
                "qalr_v3_gate_token_topm": int(
                    getattr(model_args, "qalr_v3_gate_token_topm", 8)
                ),
                "qalr_v3_gate_patch_topr": int(
                    getattr(model_args, "qalr_v3_gate_patch_topr", 4)
                ),
                "qalr_v3_gate_candidate_rank_max": int(
                    getattr(model_args, "qalr_v3_gate_candidate_rank_max", 2)
                ),
                "qalr_v3_gate_local_rank_max": int(
                    getattr(model_args, "qalr_v3_gate_local_rank_max", 1)
                ),
                "qalr_v3_gate_global_margin_max": float(
                    getattr(model_args, "qalr_v3_gate_global_margin_max", 0.01)
                ),
                "qalr_v3_gate_local_gap_min": float(
                    getattr(model_args, "qalr_v3_gate_local_gap_min", 0.01)
                ),
                "dataset_name": str(getattr(model_args, "dataset_name", "")),
                "eval_dataset": str(getattr(model_args, "eval_dataset", "")),
                "config_file": str(getattr(model_args, "config_file", "")),
                "test_ckpt_file": str(getattr(model_args, "test_ckpt_file", "")),
                "model_name": str(getattr(model_args, "name", "")),
                "score_source": "pre_qalr_v3_gate_global_similarity",
            },
        }
        with open(diag_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        self.logger.info(
            "[QALR_V3_GATE_DIAG_DUMP] path=%s queries=%d",
            diag_file,
            len(payload["queries"]),
        )

    def _save_qalr_v3_two_stage_diagnostics(self, model_args, rerank_stats):
        if model_args is None or not bool(
            getattr(model_args, "qalr_v3_two_stage_save_diag", False)
        ):
            return
        diag_file = str(
            getattr(model_args, "qalr_v3_two_stage_diag_file", "")
        ).strip()
        if not diag_file:
            diag_file = os.path.join(
                str(getattr(model_args, "output_dir", ".")),
                "qalr_v3_two_stage_gate_diag.json",
            )
        diag_file = os.path.abspath(diag_file)
        diag_dir = os.path.dirname(diag_file)
        if diag_dir:
            os.makedirs(diag_dir, exist_ok=True)

        summary = dict(rerank_stats.get("diagnostic_summary", {}))
        payload = {
            "summary": summary,
            "queries": list(rerank_stats.get("query_diagnostics", [])),
            "config": {
                "use_qalr_v3_two_stage_gate": bool(
                    getattr(model_args, "use_qalr_v3_two_stage_gate", False)
                ),
                "qalr_v3_stage1_mode": str(
                    getattr(model_args, "qalr_v3_stage1_mode", "avg")
                ),
                "qalr_v3_stage1_global_margin_max": float(
                    getattr(model_args, "qalr_v3_stage1_global_margin_max", 0.008)
                ),
                "qalr_v3_stage1_local_gap_min": float(
                    getattr(model_args, "qalr_v3_stage1_local_gap_min", 0.015)
                ),
                "qalr_v3_stage2_mode": str(
                    getattr(model_args, "qalr_v3_stage2_mode", "avg")
                ),
                "qalr_v3_stage2_global_margin_min": float(
                    getattr(model_args, "qalr_v3_stage2_global_margin_min", 0.008)
                ),
                "qalr_v3_stage2_global_margin_max": float(
                    getattr(model_args, "qalr_v3_stage2_global_margin_max", 0.010)
                ),
                "qalr_v3_stage2_local_gap_min": float(
                    getattr(model_args, "qalr_v3_stage2_local_gap_min", 0.011)
                ),
                "qalr_v3_two_stage_topk": int(
                    getattr(model_args, "qalr_v3_two_stage_topk", 64)
                ),
                "qalr_v3_two_stage_candidate_rank_max": int(
                    getattr(model_args, "qalr_v3_two_stage_candidate_rank_max", 2)
                ),
                "qalr_v3_two_stage_local_rank_max": int(
                    getattr(model_args, "qalr_v3_two_stage_local_rank_max", 1)
                ),
                "qalr_v3_two_stage_token_topm": int(
                    getattr(model_args, "qalr_v3_two_stage_token_topm", 8)
                ),
                "qalr_v3_two_stage_patch_topr": int(
                    getattr(model_args, "qalr_v3_two_stage_patch_topr", 4)
                ),
                "dataset_name": str(getattr(model_args, "dataset_name", "")),
                "eval_dataset": str(getattr(model_args, "eval_dataset", "")),
                "config_file": str(getattr(model_args, "config_file", "")),
                "test_ckpt_file": str(getattr(model_args, "test_ckpt_file", "")),
                "model_name": str(getattr(model_args, "name", "")),
                "score_source": "pre_qalr_v3_two_stage_global_similarity",
            },
        }
        with open(diag_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        self.logger.info(
            "[QALR_V3_TWO_STAGE_DIAG_DUMP] path=%s queries=%d",
            diag_file,
            len(payload["queries"]),
        )

    def _get_query_raw_captions(self, expected_count):
        txt_dataset = getattr(self.txt_loader, "dataset", None)
        raw_captions = getattr(txt_dataset, "captions", None)
        if raw_captions is None:
            raise RuntimeError(
                "use_uspc_slot_rerank=True requires txt_loader.dataset.captions."
            )
        raw_captions = [
            "" if caption is None else str(caption)
            for caption in list(raw_captions)
        ]
        if len(raw_captions) != int(expected_count):
            raise RuntimeError(
                "use_uspc_slot_rerank=True requires raw captions to align with "
                f"text queries, got {len(raw_captions)} captions for "
                f"{int(expected_count)} queries."
            )
        return raw_captions

    def _as_hw_pair(self, value, default):
        if isinstance(value, str):
            cleaned = (
                value.replace("(", " ")
                .replace(")", " ")
                .replace("[", " ")
                .replace("]", " ")
                .replace(",", " ")
                .replace("x", " ")
                .replace("X", " ")
            )
            parts = [part for part in cleaned.split() if part]
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
            if len(parts) == 1:
                one = int(parts[0])
                return one, one
            return default
        if isinstance(value, (list, tuple)):
            if len(value) >= 2:
                return int(value[0]), int(value[1])
            if len(value) == 1:
                one = int(value[0])
                return one, one
            return default
        if value is None:
            return default
        one = int(value)
        return one, one

    def _infer_uspc_slot_rerank_grid(self, patch_count, model_args):
        patch_count = int(patch_count)
        img_h, img_w = self._as_hw_pair(
            getattr(model_args, "img_size", None),
            (384, 128),
        )
        stride_h, stride_w = self._as_hw_pair(
            getattr(model_args, "stride_size", None),
            (16, 16),
        )
        stride_h = max(1, int(stride_h))
        stride_w = max(1, int(stride_w))
        cfg_h = max(1, int(img_h) // stride_h)
        cfg_w = max(1, int(img_w) // stride_w)
        if cfg_h * cfg_w == patch_count:
            return cfg_h, cfg_w
        if cfg_w > 0 and patch_count % cfg_w == 0:
            return max(1, patch_count // cfg_w), cfg_w
        if cfg_h > 0 and patch_count % cfg_h == 0:
            return cfg_h, max(1, patch_count // cfg_h)
        return max(1, patch_count), 1

    def _row_bounds_for_rerank(self, grid_h, start_frac, end_frac):
        start = int(round(float(start_frac) * float(grid_h)))
        end = int(round(float(end_frac) * float(grid_h)))
        start = max(0, min(int(grid_h), start))
        end = max(0, min(int(grid_h), end))
        if end <= start:
            end = min(int(grid_h), start + 1)
        return start, end

    def _mean_fixed_region_for_rerank(
        self, patch_grid, row_start, row_end, col_mask=None
    ):
        if col_mask is None:
            region = patch_grid[:, row_start:row_end, :, :]
        else:
            region = patch_grid[:, row_start:row_end, col_mask, :]
        if region.numel() == 0:
            return patch_grid.new_zeros(
                patch_grid.shape[0],
                patch_grid.shape[-1],
            )
        return region.reshape(region.shape[0], -1, region.shape[-1]).mean(dim=1)

    def _pool_fixed_regions_for_rerank(self, gallery_rgb_patch_tokens, model_args):
        if gallery_rgb_patch_tokens is None:
            raise RuntimeError(
                "use_uspc_slot_rerank=True requires gallery RGB patch tokens."
            )
        if gallery_rgb_patch_tokens.dim() != 3:
            raise ValueError(
                "USPC slot rerank RGB patch tokens must have shape [G, N, C], "
                f"got {tuple(gallery_rgb_patch_tokens.shape)}."
            )
        if gallery_rgb_patch_tokens.shape[1] <= 0:
            raise ValueError("USPC slot rerank RGB patch tokens are empty.")

        patch_tokens = gallery_rgb_patch_tokens.float()
        grid_h, grid_w = self._infer_uspc_slot_rerank_grid(
            patch_tokens.shape[1],
            model_args,
        )
        patch_count = grid_h * grid_w
        patch_grid = patch_tokens[:, :patch_count, :].reshape(
            patch_tokens.shape[0],
            grid_h,
            grid_w,
            patch_tokens.shape[-1],
        )

        head_rows = self._row_bounds_for_rerank(grid_h, 0.00, 0.20)
        upper_rows = self._row_bounds_for_rerank(grid_h, 0.20, 0.55)
        lower_rows = self._row_bounds_for_rerank(grid_h, 0.55, 0.85)
        shoes_rows = self._row_bounds_for_rerank(grid_h, 0.85, 1.00)
        bag_rows = self._row_bounds_for_rerank(grid_h, 0.25, 0.75)
        edge_cols = max(1, grid_w // 4)
        bag_col_mask = torch.zeros(
            grid_w,
            dtype=torch.bool,
            device=patch_tokens.device,
        )
        bag_col_mask[:edge_cols] = True
        bag_col_mask[-edge_cols:] = True

        region_feats = {
            "head": self._mean_fixed_region_for_rerank(patch_grid, *head_rows),
            "upper": self._mean_fixed_region_for_rerank(patch_grid, *upper_rows),
            "lower": self._mean_fixed_region_for_rerank(patch_grid, *lower_rows),
            "shoes": self._mean_fixed_region_for_rerank(patch_grid, *shoes_rows),
            "bag": self._mean_fixed_region_for_rerank(
                patch_grid, *bag_rows, col_mask=bag_col_mask
            ),
        }
        return {
            name: F.normalize(feat.float(), p=2, dim=1)
            for name, feat in region_feats.items()
        }

    def _build_slot_prompts_for_query(
        self, parsed_attrs, query_index, skip_negative_bag=True
    ):
        slots = []
        targets = parsed_attrs["targets"]
        masks = parsed_attrs["masks"]

        for attr_name, (region_name, prompt_template) in (
            USPC_SLOT_RERANK_COLOR_SLOTS.items()
        ):
            is_valid = bool(masks[attr_name][query_index].item())
            if not is_valid:
                continue
            color_index = int(targets[attr_name][query_index].item())
            if color_index < 0 or color_index >= len(COLOR_CLASSES):
                continue
            color_name = COLOR_CLASSES[color_index]
            slots.append(
                {
                    "attr_name": attr_name,
                    "region_name": region_name,
                    "prompt": prompt_template.format(color_name),
                }
            )

        if bool(masks["bag_present"][query_index].item()):
            bag_target = int(targets["bag_present"][query_index].item())
            if bag_target == 1:
                slots.append(
                    {
                        "attr_name": "bag_present",
                        "region_name": "bag",
                        "prompt": "bag",
                    }
                )
            elif not skip_negative_bag:
                slots.append(
                    {
                        "attr_name": "bag_present",
                        "region_name": "bag",
                        "prompt": "no bag",
                    }
                )

        if bool(masks["hair_length"][query_index].item()):
            hair_target = int(targets["hair_length"][query_index].item())
            if 0 <= hair_target < len(USPC_SLOT_RERANK_HAIR_LABELS):
                slots.append(
                    {
                        "attr_name": "hair_length",
                        "region_name": "head",
                        "prompt": (
                            f"{USPC_SLOT_RERANK_HAIR_LABELS[hair_target]} hair"
                        ),
                    }
                )

        return slots

    def _encode_slot_prompt_embeddings(self, model, prompts, device):
        unique_prompts = list(dict.fromkeys(prompts))
        if not unique_prompts:
            return {}
        text_length = int(getattr(getattr(model, "args", None), "text_length", 77))
        prompt_ids = self._tokenize_text_batch(
            unique_prompts,
            text_length=text_length,
            device=device,
        )
        with torch.no_grad():
            prompt_outputs = self._encode_text_feature(
                model,
                prompt_ids,
                return_local_tokens=False,
            )
        if isinstance(prompt_outputs, dict):
            prompt_feats = prompt_outputs["global_feat"]
        else:
            prompt_feats = prompt_outputs
        prompt_feats = F.normalize(prompt_feats.float(), p=2, dim=1)
        return {
            prompt: prompt_feats[index]
            for index, prompt in enumerate(unique_prompts)
        }

    def _normalize_uspc_slot_scores(self, scores, score_norm):
        score_norm = str(score_norm).lower()
        if score_norm == "none":
            return scores
        if score_norm != "zscore":
            raise ValueError(
                "Unsupported uspc_slot_rerank_score_norm="
                f"{score_norm!r}. Supported: 'zscore', 'none'."
            )
        score_mean = scores.mean()
        score_std = scores.std(unbiased=False).clamp_min(1e-6)
        return (scores - score_mean) / score_std

    def _apply_uspc_slot_rerank(self, model, embedding_outputs, global_similarity):
        model_args = getattr(model, "args", None)
        topk = int(getattr(model_args, "uspc_slot_rerank_topk", 50))
        alpha = float(getattr(model_args, "uspc_slot_rerank_alpha", 0.0))
        score_norm = str(
            getattr(model_args, "uspc_slot_rerank_score_norm", "zscore")
        ).lower()
        region_mode = str(
            getattr(model_args, "uspc_slot_rerank_region_mode", "fixed_stripe")
        ).lower()
        patch_source = str(
            getattr(model_args, "uspc_slot_rerank_patch_source", "rgb")
        ).lower()
        min_valid_slots = int(
            getattr(model_args, "uspc_slot_rerank_min_valid_slots", 1)
        )
        skip_negative_bag = bool(
            getattr(model_args, "uspc_slot_rerank_skip_negative_bag", True)
        )

        if topk <= 0:
            raise ValueError(f"uspc_slot_rerank_topk must be > 0, got {topk}.")
        if min_valid_slots <= 0:
            raise ValueError(
                "uspc_slot_rerank_min_valid_slots must be > 0, "
                f"got {min_valid_slots}."
            )
        if score_norm not in {"zscore", "none"}:
            raise ValueError(
                "Unsupported uspc_slot_rerank_score_norm="
                f"{score_norm!r}. Supported: 'zscore', 'none'."
            )
        if region_mode != "fixed_stripe":
            raise ValueError(
                "USPC-v2C currently supports only "
                "--uspc_slot_rerank_region_mode fixed_stripe."
            )
        if patch_source != "rgb":
            raise NotImplementedError(
                "USPC-v2C first version supports only "
                "--uspc_slot_rerank_patch_source rgb."
            )
        if not isinstance(embedding_outputs, dict):
            raise RuntimeError(
                "use_uspc_slot_rerank=True requires rich evaluator outputs."
            )

        num_query, num_gallery = global_similarity.shape
        if num_query == 0 or num_gallery == 0:
            return global_similarity, {
                "uspc_slot_rerank_active": False,
                "uspc_slot_rerank_topk": topk,
                "uspc_slot_rerank_alpha": alpha,
                "uspc_slot_rerank_valid_query_count": 0,
                "uspc_slot_rerank_avg_valid_slots": 0.0,
                "uspc_slot_rerank_score_mean": 0.0,
                "uspc_slot_rerank_score_std": 0.0,
            }

        raw_captions = self._get_query_raw_captions(num_query)
        parsed_attrs = parse_basic_attributes(
            raw_captions,
            device=global_similarity.device,
        )
        query_slots = []
        slot_prompts = []
        valid_slot_counts = []
        for query_index in range(num_query):
            slots = self._build_slot_prompts_for_query(
                parsed_attrs,
                query_index,
                skip_negative_bag=skip_negative_bag,
            )
            query_slots.append(slots)
            valid_slot_counts.append(len(slots))
            if len(slots) >= min_valid_slots:
                slot_prompts.extend([slot["prompt"] for slot in slots])

        prompt_embeddings = self._encode_slot_prompt_embeddings(
            model,
            slot_prompts,
            device=global_similarity.device,
        )
        region_feats = self._pool_fixed_regions_for_rerank(
            embedding_outputs.get("gallery_rgb_patch_tokens", None),
            model_args,
        )

        effective_topk = min(topk, num_gallery)
        original_order = torch.argsort(global_similarity, dim=1, descending=True)
        topk_indices = original_order[:, :effective_topk]
        topk_scores = torch.gather(global_similarity, dim=1, index=topk_indices)
        ranking_order = original_order.clone()
        valid_query_mask = torch.zeros(
            num_query,
            dtype=torch.bool,
            device=global_similarity.device,
        )
        raw_slot_score_values = []

        for query_index, slots in enumerate(query_slots):
            if len(slots) < min_valid_slots:
                continue
            candidate_indices = topk_indices[query_index]
            per_slot_scores = []
            for slot in slots:
                prompt_feat = prompt_embeddings.get(slot["prompt"], None)
                region_feat = region_feats.get(slot["region_name"], None)
                if prompt_feat is None or region_feat is None:
                    continue
                per_slot_scores.append(region_feat[candidate_indices] @ prompt_feat)
            if len(per_slot_scores) < min_valid_slots:
                continue

            raw_slot_scores = torch.stack(per_slot_scores, dim=0).mean(dim=0)
            normalized_slot_scores = self._normalize_uspc_slot_scores(
                raw_slot_scores,
                score_norm=score_norm,
            )
            final_topk_scores = (
                topk_scores[query_index]
                + alpha * normalized_slot_scores
            )
            topk_rerank_order = torch.argsort(
                final_topk_scores,
                dim=0,
                descending=True,
                stable=True,
            )
            ranking_order[query_index, :effective_topk] = torch.gather(
                candidate_indices,
                dim=0,
                index=topk_rerank_order,
            )
            valid_query_mask[query_index] = True
            raw_slot_score_values.append(raw_slot_scores.detach().float())

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

        valid_slot_tensor = torch.tensor(
            valid_slot_counts,
            dtype=torch.float32,
            device=global_similarity.device,
        )
        if raw_slot_score_values:
            raw_slot_scores = torch.cat(raw_slot_score_values, dim=0)
            score_mean = float(raw_slot_scores.mean().item())
            score_std = float(raw_slot_scores.std(unbiased=False).item())
        else:
            score_mean = 0.0
            score_std = 0.0
        reranked_query_count = int(
            (
                ranking_order[:, :effective_topk]
                != original_order[:, :effective_topk]
            ).any(dim=1).sum().item()
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
            "uspc_slot_rerank_active": True,
            "uspc_slot_rerank_topk": effective_topk,
            "uspc_slot_rerank_alpha": alpha,
            "uspc_slot_rerank_score_norm": score_norm,
            "uspc_slot_rerank_valid_query_count": int(
                valid_query_mask.sum().item()
            ),
            "uspc_slot_rerank_avg_valid_slots": float(
                valid_slot_tensor.mean().item()
            ) if valid_slot_tensor.numel() else 0.0,
            "uspc_slot_rerank_score_mean": score_mean,
            "uspc_slot_rerank_score_std": score_std,
            "uspc_slot_rerank_reranked_query_count": reranked_query_count,
            "uspc_slot_rerank_topk_boundary_safe": bool(
                topk_membership_safe and tail_membership_safe
            ),
        }
        return reranked_similarity, stats
    
    def eval(self, model, i2t_metric=False, return_local_tokens=False):
        use_qalr = bool(getattr(getattr(model, "args", None), "use_qalr", False))
        use_qalr_v2 = bool(
            getattr(getattr(model, "args", None), "use_qalr_v2", False)
        )
        use_qalr_v3_candidate_diag = bool(
            getattr(
                getattr(model, "args", None),
                "qalr_v3_candidate_diag",
                False,
            )
        )
        use_qalr_v3_gate = bool(
            getattr(getattr(model, "args", None), "use_qalr_v3_gate", False)
        )
        use_qalr_v3_two_stage_gate = bool(
            getattr(
                getattr(model, "args", None),
                "use_qalr_v3_two_stage_gate",
                False,
            )
        )
        use_transport_rerank = bool(
            getattr(getattr(model, "args", None), "use_transport_rerank", False)
        )
        use_uspc_slot_rerank = bool(
            getattr(getattr(model, "args", None), "use_uspc_slot_rerank", False)
        )
        use_mtp_eval_fusion = bool(
            getattr(getattr(model, "args", None), "use_modality_text_prompt", False)
        )
        if use_uspc_slot_rerank and (
            use_qalr
            or use_qalr_v2
            or use_qalr_v3_gate
            or use_qalr_v3_two_stage_gate
            or use_transport_rerank
        ):
            self.logger.warning(
                "USPC slot rerank is an inference-time rerank experiment and should "
                "not be mixed with QALR/QALR-v2 or transport rerank. Disable "
                "--use_qalr, --use_qalr_v2, --use_qalr_v3_gate, "
                "--use_qalr_v3_two_stage_gate and --use_transport_rerank before "
                "running --use_uspc_slot_rerank."
            )
            raise RuntimeError(
                "use_uspc_slot_rerank=True is mutually exclusive with QALR, "
                "QALR-v2, QALR-v3 gate, QALR-v3 two-stage gate or transport rerank."
            )
        return_local_tokens = bool(
            return_local_tokens
            or use_qalr
            or use_qalr_v2
            or use_qalr_v3_candidate_diag
            or use_qalr_v3_gate
            or use_qalr_v3_two_stage_gate
            or use_transport_rerank
            or use_uspc_slot_rerank
        )
        model_args = getattr(model, "args", None)
        self.logger.info(
            "[EVAL_MODEL_FLAGS] source=%s model_training=%s args_training=%s eval_use_fused_mu=%s eval_use_mu_only=%s use_precision_fusion=%s use_global_uncertainty=%s use_rgbt_add_fusion=%s use_batch_uncertainty=%s use_id_uncertainty=%s use_qalr=%s use_qalr_v2=%s qalr_v3_candidate_diag=%s use_qalr_v3_gate=%s use_qalr_v3_two_stage_gate=%s use_transport_rerank=%s use_uspc_slot_rerank=%s use_modality_text_prompt=%s",
            self.source,
            bool(getattr(model, "training", False)),
            bool(getattr(model_args, "training", False)),
            bool(getattr(model_args, "eval_use_fused_mu", True)),
            bool(getattr(model_args, "eval_use_mu_only", False)),
            getattr(model_args, "use_precision_fusion", None),
            bool(getattr(model_args, "use_global_uncertainty", False)),
            bool(getattr(model_args, "use_rgbt_add_fusion", True)),
            bool(getattr(model_args, "use_batch_uncertainty", True)),
            bool(getattr(model_args, "use_id_uncertainty", True)),
            use_qalr,
            use_qalr_v2,
            use_qalr_v3_candidate_diag,
            use_qalr_v3_gate,
            use_qalr_v3_two_stage_gate,
            use_transport_rerank,
            use_uspc_slot_rerank,
            use_mtp_eval_fusion,
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

        self._log_feature_summary(qfeats, gfeats)

        # 2) L2 normalize before cosine similarity.
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        # 3) Text-to-image similarity matrix.
        similarity = qfeats @ gfeats.t()
        if use_mtp_eval_fusion:
            alpha = float(getattr(model.args, "mtp_eval_fusion_alpha", 0.10))
            beta = float(getattr(model.args, "mtp_eval_fusion_beta", 0.00))
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
        if use_qalr_v3_candidate_diag:
            candidate_diag_reranker = QueryAdaptiveLocalReranker(
                topk=int(getattr(model.args, "qalr_v3_candidate_topk", 64)),
                alpha=0.0,
                token_topm=int(
                    getattr(model.args, "qalr_v3_candidate_token_topm", 8)
                ),
                patch_topr=int(
                    getattr(model.args, "qalr_v3_candidate_patch_topr", 4)
                ),
                vocab_size=int(getattr(model.args, "vocab_size", 49408)),
            )
            candidate_diag_payload = candidate_diag_reranker.candidate_oracle_diagnostics(
                query_token_embeddings=embedding_outputs["query_token_embeddings"],
                query_token_ids=embedding_outputs["query_token_ids"],
                query_global_embeddings=embedding_outputs["query_global_embeddings"],
                gallery_rgb_patch_tokens=embedding_outputs["gallery_rgb_patch_tokens"],
                gallery_thermal_patch_tokens=embedding_outputs[
                    "gallery_thermal_patch_tokens"
                ],
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
                qids=qids,
                gids=gids,
                save_scores=bool(
                    getattr(model.args, "qalr_v3_candidate_save_scores", False)
                ),
            )
            candidate_diag_payload["config"].update(
                {
                    "dataset_name": str(getattr(model.args, "dataset_name", "")),
                    "eval_dataset": str(getattr(model.args, "eval_dataset", "")),
                    "config_file": str(getattr(model.args, "config_file", "")),
                    "test_ckpt_file": str(getattr(model.args, "test_ckpt_file", "")),
                    "model_name": str(getattr(model.args, "name", "")),
                    "score_source": "pre_qalr_global_similarity",
                }
            )
            self._save_qalr_v3_candidate_diagnostics(
                model_args=model.args,
                payload=candidate_diag_payload,
            )
        if use_qalr_v3_two_stage_gate:
            two_stage_reranker = QueryAdaptiveLocalReranker(
                topk=int(getattr(model.args, "qalr_v3_two_stage_topk", 64)),
                alpha=0.0,
                token_topm=int(
                    getattr(model.args, "qalr_v3_two_stage_token_topm", 8)
                ),
                patch_topr=int(
                    getattr(model.args, "qalr_v3_two_stage_patch_topr", 4)
                ),
                vocab_size=int(getattr(model.args, "vocab_size", 49408)),
            )
            similarity, two_stage_stats = two_stage_reranker.two_stage_gate_rerank(
                query_token_embeddings=embedding_outputs["query_token_embeddings"],
                query_token_ids=embedding_outputs["query_token_ids"],
                query_global_embeddings=embedding_outputs["query_global_embeddings"],
                gallery_rgb_patch_tokens=embedding_outputs["gallery_rgb_patch_tokens"],
                gallery_thermal_patch_tokens=embedding_outputs[
                    "gallery_thermal_patch_tokens"
                ],
                global_similarity=similarity,
                qids=qids,
                gids=gids,
                stage1_mode=str(getattr(model.args, "qalr_v3_stage1_mode", "avg")),
                stage1_global_margin_max=float(
                    getattr(model.args, "qalr_v3_stage1_global_margin_max", 0.008)
                ),
                stage1_local_gap_min=float(
                    getattr(model.args, "qalr_v3_stage1_local_gap_min", 0.015)
                ),
                stage2_mode=str(getattr(model.args, "qalr_v3_stage2_mode", "avg")),
                stage2_global_margin_min=float(
                    getattr(model.args, "qalr_v3_stage2_global_margin_min", 0.008)
                ),
                stage2_global_margin_max=float(
                    getattr(model.args, "qalr_v3_stage2_global_margin_max", 0.010)
                ),
                stage2_local_gap_min=float(
                    getattr(model.args, "qalr_v3_stage2_local_gap_min", 0.011)
                ),
                candidate_rank_max=int(
                    getattr(model.args, "qalr_v3_two_stage_candidate_rank_max", 2)
                ),
                local_rank_max=int(
                    getattr(model.args, "qalr_v3_two_stage_local_rank_max", 1)
                ),
                return_stats=True,
                return_diagnostics=bool(
                    getattr(model.args, "qalr_v3_two_stage_save_diag", False)
                ),
            )
            if use_qalr or use_qalr_v2 or use_qalr_v3_gate:
                self.logger.warning(
                    "QALR v3 two-stage gate is enabled together with another "
                    "QALR variant. Two-stage gate takes precedence and earlier "
                    "QALR variants are skipped."
                )
            two_stage_summary = two_stage_stats.get("diagnostic_summary", {})
            self.logger.info(
                "[QALR_V3_TWO_STAGE] enabled=True topk=%d token_topm=%d patch_topr=%d candidate_rank_max=%d local_rank_max=%d stage1_mode=%s stage1_margin_max=%.6f stage1_gap_min=%.6f stage2_mode=%s stage2_margin_min=%.6f stage2_margin_max=%.6f stage2_gap_min=%.6f topk_boundary_safe=%s",
                int(getattr(model.args, "qalr_v3_two_stage_topk", 64)),
                int(getattr(model.args, "qalr_v3_two_stage_token_topm", 8)),
                int(getattr(model.args, "qalr_v3_two_stage_patch_topr", 4)),
                int(getattr(model.args, "qalr_v3_two_stage_candidate_rank_max", 2)),
                int(getattr(model.args, "qalr_v3_two_stage_local_rank_max", 1)),
                str(getattr(model.args, "qalr_v3_stage1_mode", "avg")),
                float(getattr(model.args, "qalr_v3_stage1_global_margin_max", 0.008)),
                float(getattr(model.args, "qalr_v3_stage1_local_gap_min", 0.015)),
                str(getattr(model.args, "qalr_v3_stage2_mode", "avg")),
                float(getattr(model.args, "qalr_v3_stage2_global_margin_min", 0.008)),
                float(getattr(model.args, "qalr_v3_stage2_global_margin_max", 0.010)),
                float(getattr(model.args, "qalr_v3_stage2_local_gap_min", 0.011)),
                str(
                    two_stage_stats.get(
                        "qalr_v3_two_stage_topk_boundary_safe",
                        False,
                    )
                ),
            )
            self.logger.info(
                "[QALR_V3_TWO_STAGE_DIAG] changed=%d improved=%d hurt=%d neutral=%d net=%d stage1=%d stage2=%d R1=%s",
                int(two_stage_summary.get("changed_top1_count", 0)),
                int(two_stage_summary.get("improved_top1_count", 0)),
                int(two_stage_summary.get("hurt_top1_count", 0)),
                int(two_stage_summary.get("neutral_changed_top1_count", 0)),
                int(two_stage_summary.get("net_top1_gain", 0)),
                int(two_stage_summary.get("stage1_hit_count", 0)),
                int(two_stage_summary.get("stage2_hit_count", 0)),
                str(two_stage_summary.get("R1", None)),
            )
            self._save_qalr_v3_two_stage_diagnostics(
                model_args=model.args,
                rerank_stats=two_stage_stats,
            )
        if use_qalr_v3_gate and not use_qalr_v3_two_stage_gate:
            gate_reranker = QueryAdaptiveLocalReranker(
                topk=int(getattr(model.args, "qalr_v3_gate_topk", 64)),
                alpha=0.0,
                token_topm=int(getattr(model.args, "qalr_v3_gate_token_topm", 8)),
                patch_topr=int(getattr(model.args, "qalr_v3_gate_patch_topr", 4)),
                vocab_size=int(getattr(model.args, "vocab_size", 49408)),
            )
            similarity, gate_stats = gate_reranker.gate_rerank(
                query_token_embeddings=embedding_outputs["query_token_embeddings"],
                query_token_ids=embedding_outputs["query_token_ids"],
                query_global_embeddings=embedding_outputs["query_global_embeddings"],
                gallery_rgb_patch_tokens=embedding_outputs["gallery_rgb_patch_tokens"],
                gallery_thermal_patch_tokens=embedding_outputs[
                    "gallery_thermal_patch_tokens"
                ],
                global_similarity=similarity,
                qids=qids,
                gids=gids,
                mode=str(getattr(model.args, "qalr_v3_gate_mode", "concat")),
                candidate_rank_max=int(
                    getattr(model.args, "qalr_v3_gate_candidate_rank_max", 2)
                ),
                local_rank_max=int(
                    getattr(model.args, "qalr_v3_gate_local_rank_max", 1)
                ),
                global_margin_max=float(
                    getattr(model.args, "qalr_v3_gate_global_margin_max", 0.01)
                ),
                local_gap_min=float(
                    getattr(model.args, "qalr_v3_gate_local_gap_min", 0.01)
                ),
                return_stats=True,
                return_diagnostics=bool(
                    getattr(model.args, "qalr_v3_gate_save_diag", False)
                ),
            )
            if use_qalr or use_qalr_v2:
                self.logger.warning(
                    "QALR v3 gate is enabled together with old QALR/QALR-v2. "
                    "QALR v3 gate takes precedence and earlier QALR variants are skipped."
                )
            gate_summary = gate_stats.get("diagnostic_summary", {})
            self.logger.info(
                "[QALR_V3_GATE] enabled=True mode=%s topk=%d token_topm=%d patch_topr=%d candidate_rank_max=%d local_rank_max=%d global_margin_max=%.6f local_gap_min=%.6f topk_boundary_safe=%s",
                str(getattr(model.args, "qalr_v3_gate_mode", "concat")),
                int(getattr(model.args, "qalr_v3_gate_topk", 64)),
                int(getattr(model.args, "qalr_v3_gate_token_topm", 8)),
                int(getattr(model.args, "qalr_v3_gate_patch_topr", 4)),
                int(getattr(model.args, "qalr_v3_gate_candidate_rank_max", 2)),
                int(getattr(model.args, "qalr_v3_gate_local_rank_max", 1)),
                float(getattr(model.args, "qalr_v3_gate_global_margin_max", 0.01)),
                float(getattr(model.args, "qalr_v3_gate_local_gap_min", 0.01)),
                str(gate_stats.get("qalr_v3_gate_topk_boundary_safe", False)),
            )
            self.logger.info(
                "[QALR_V3_GATE_DIAG] changed=%d improved=%d hurt=%d neutral=%d net=%d R1=%s",
                int(gate_summary.get("changed_top1_count", 0)),
                int(gate_summary.get("improved_top1_count", 0)),
                int(gate_summary.get("hurt_top1_count", 0)),
                int(gate_summary.get("neutral_changed_top1_count", 0)),
                int(gate_summary.get("net_top1_gain", 0)),
                str(gate_summary.get("R1", None)),
            )
            self._save_qalr_v3_gate_diagnostics(
                model_args=model.args,
                rerank_stats=gate_stats,
            )
        if use_uspc_slot_rerank:
            similarity, uspc_slot_rerank_stats = self._apply_uspc_slot_rerank(
                model=model,
                embedding_outputs=embedding_outputs,
                global_similarity=similarity,
            )
            if bool(getattr(model.args, "debug_uspc_slot_rerank", False)):
                self.logger.info(
                    "[USPC_SLOT_RERANK] uspc_slot_rerank_active=%s, "
                    "uspc_slot_rerank_topk=%d, "
                    "uspc_slot_rerank_alpha=%.6f, "
                    "uspc_slot_rerank_valid_query_count=%d, "
                    "uspc_slot_rerank_avg_valid_slots=%.4f, "
                    "uspc_slot_rerank_score_mean=%.6f, "
                    "uspc_slot_rerank_score_std=%.6f, "
                    "uspc_slot_rerank_reranked_query_count=%d, "
                    "uspc_slot_rerank_topk_boundary_safe=%s",
                    uspc_slot_rerank_stats.get("uspc_slot_rerank_active", False),
                    int(uspc_slot_rerank_stats.get("uspc_slot_rerank_topk", 0)),
                    float(
                        uspc_slot_rerank_stats.get(
                            "uspc_slot_rerank_alpha", 0.0
                        )
                    ),
                    int(
                        uspc_slot_rerank_stats.get(
                            "uspc_slot_rerank_valid_query_count", 0
                        )
                    ),
                    float(
                        uspc_slot_rerank_stats.get(
                            "uspc_slot_rerank_avg_valid_slots", 0.0
                        )
                    ),
                    float(
                        uspc_slot_rerank_stats.get(
                            "uspc_slot_rerank_score_mean", 0.0
                        )
                    ),
                    float(
                        uspc_slot_rerank_stats.get(
                            "uspc_slot_rerank_score_std", 0.0
                        )
                    ),
                    int(
                        uspc_slot_rerank_stats.get(
                            "uspc_slot_rerank_reranked_query_count", 0
                        )
                    ),
                    str(
                        uspc_slot_rerank_stats.get(
                            "uspc_slot_rerank_topk_boundary_safe", False
                        )
                    ),
                )
        if (
            use_qalr
            and not use_qalr_v2
            and not use_qalr_v3_gate
            and not use_qalr_v3_two_stage_gate
        ):
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
                qids=qids,
                gids=gids,
                return_stats=True,
                return_diagnostics=bool(
                    getattr(model.args, "qalr_v2_save_diag", False)
                ),
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
            log_msg += (
                ", reranked_query_count=%d, gated_query_count=%d, "
                "changed_top1_count=%d, topk_boundary_safe=%s, fusion_norm=%s"
            )
            log_args.extend(
                [
                    int(rerank_stats.get("reranked_query_count", 0)),
                    int(rerank_stats.get("gated_query_count", 0)),
                    int(rerank_stats.get("changed_top1_count", 0)),
                    str(rerank_stats.get("topk_boundary_safe", False)),
                    str(rerank_stats.get("fusion_norm", "unknown")),
                ]
            )
            self.logger.info(log_msg, *log_args)
            if bool(getattr(model.args, "qalr_v2_save_diag", False)):
                before_metrics = self._compute_metric_summary(
                    baseline_similarity,
                    qids,
                    gids,
                )
                after_metrics = self._compute_metric_summary(similarity, qids, gids)
                self._log_qalr_diag_summary("[QALR_V2_DIAG]", rerank_stats)
                self._save_qalr_diagnostics(
                    model_args=model.args,
                    rerank_stats=rerank_stats,
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    use_qalr_v2=False,
                )

        if use_qalr_v2 and not use_qalr_v3_gate and not use_qalr_v3_two_stage_gate:
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
                use_qalr_v2=True,
                qalr_v2_alpha_min=float(
                    getattr(model.args, "qalr_v2_alpha_min", 0.0)
                ),
                qalr_v2_alpha_max=float(
                    getattr(model.args, "qalr_v2_alpha_max", 0.01)
                ),
                qalr_v2_margin_low=float(
                    getattr(model.args, "qalr_v2_margin_low", 0.0)
                ),
                qalr_v2_margin_high=float(
                    getattr(model.args, "qalr_v2_margin_high", 0.02)
                ),
                qalr_v2_freeze_top1_margin=float(
                    getattr(model.args, "qalr_v2_freeze_top1_margin", 0.02)
                ),
                qalr_v2_promote_margin=float(
                    getattr(model.args, "qalr_v2_promote_margin", 0.0)
                ),
                qalr_v2_modality_fuse=str(
                    getattr(model.args, "qalr_v2_modality_fuse", "concat")
                ),
                qalr_v2_top1_guard=bool(
                    getattr(model.args, "qalr_v2_top1_guard", False)
                ),
                qalr_v2_promote_only=bool(
                    getattr(model.args, "qalr_v2_promote_only", False)
                ),
                qalr_v2_diag_only=bool(
                    getattr(model.args, "qalr_v2_diag_only", False)
                ),
            )
            similarity, rerank_stats = reranker.rerank(
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
                global_similarity=similarity,
                qids=qids,
                gids=gids,
                return_stats=True,
                return_diagnostics=True,
            )
            proposed_similarity = rerank_stats.pop("_proposed_similarity", similarity)
            before_metrics = self._compute_metric_summary(
                baseline_similarity,
                qids,
                gids,
            )
            proposed_metrics = self._compute_metric_summary(
                proposed_similarity,
                qids,
                gids,
            )
            if bool(getattr(model.args, "qalr_v2_diag_only", False)):
                similarity = baseline_similarity
                after_metrics = before_metrics
            else:
                after_metrics = proposed_metrics
            self.logger.info(
                "[QALR_V2] enabled=True diag_only=%s topk=%d alpha_max=%.6f",
                bool(getattr(model.args, "qalr_v2_diag_only", False)),
                int(getattr(model.args, "rerank_topk", 128)),
                float(getattr(model.args, "qalr_v2_alpha_max", 0.01)),
            )
            self.logger.info(
                "[QALR_V2_CONFIG] alpha_min=%.6f alpha_max=%.6f margin_low=%.6f margin_high=%.6f top1_guard=%s freeze_top1_margin=%.6f promote_only=%s promote_margin=%.6f modality_fuse=%s",
                float(getattr(model.args, "qalr_v2_alpha_min", 0.0)),
                float(getattr(model.args, "qalr_v2_alpha_max", 0.01)),
                float(getattr(model.args, "qalr_v2_margin_low", 0.0)),
                float(getattr(model.args, "qalr_v2_margin_high", 0.02)),
                bool(getattr(model.args, "qalr_v2_top1_guard", False)),
                float(getattr(model.args, "qalr_v2_freeze_top1_margin", 0.02)),
                bool(getattr(model.args, "qalr_v2_promote_only", False)),
                float(getattr(model.args, "qalr_v2_promote_margin", 0.0)),
                str(getattr(model.args, "qalr_v2_modality_fuse", "concat")),
            )
            self._log_qalr_diag_summary("[QALR_V2_DIAG]", rerank_stats)
            self._save_qalr_diagnostics(
                model_args=model.args,
                rerank_stats=rerank_stats,
                before_metrics=before_metrics,
                after_metrics=proposed_metrics,
                use_qalr_v2=True,
            )

        if use_transport_rerank:
            if (
                use_qalr
                or use_qalr_v2
                or use_qalr_v3_gate
                or use_qalr_v3_two_stage_gate
            ):
                self.logger.warning(
                    "QALR/QALR-v2/QALR-v3 gate/QALR-v3 two-stage gate "
                    "and use_transport_rerank are both enabled. "
                    "Transport rerank will use baseline global similarity and "
                    "override the earlier QALR scores."
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

        self._log_score_summary(similarity)

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
        self._maybe_save_eval_scores(
            model_args=model_args,
            similarity=similarity,
            raw_similarity=baseline_similarity,
            qids=qids,
            gids=gids,
            metrics=self.last_eval_result,
            use_qalr=use_qalr,
            use_qalr_v2=use_qalr_v2,
            use_qalr_v3_gate=use_qalr_v3_gate,
            use_qalr_v3_two_stage_gate=use_qalr_v3_two_stage_gate,
            use_transport_rerank=use_transport_rerank,
            use_uspc_slot_rerank=use_uspc_slot_rerank,
            use_mtp_eval_fusion=use_mtp_eval_fusion,
        )
        
        return t2i_cmc[0]
