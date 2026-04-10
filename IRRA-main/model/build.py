from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import objectives
from .clip_model import (
    LayerNorm,
    QuickGELU,
    Transformer,
    build_CLIP_from_openai_pretrained,
    convert_weights,
)
from .prototype_bank import IdentityPrototypeBank
from .uncertainty import GlobalUncertaintyModeler


def apply_image_local_filter(
    global_feat, patch_tokens, top_r, gamma, temp, detach_global
):
    """
    Stateless image-only local filtering v1 helper.

    Args:
        global_feat: [B, C]
        patch_tokens: [B, N, C]
    Returns:
        refined_global: [B, C]
    """
    if global_feat.dim() != 2:
        raise ValueError(
            f"global_feat must have shape [B, C], got {tuple(global_feat.shape)}."
        )
    if patch_tokens.dim() != 3:
        raise ValueError(
            f"patch_tokens must have shape [B, N, C], got {tuple(patch_tokens.shape)}."
        )
    if patch_tokens.shape[0] != global_feat.shape[0]:
        raise ValueError(
            "global_feat and patch_tokens must share the same batch size, got "
            f"{global_feat.shape[0]} and {patch_tokens.shape[0]}."
        )
    if patch_tokens.shape[-1] != global_feat.shape[-1]:
        raise ValueError(
            "global_feat and patch_tokens must share the same feature dim, got "
            f"{global_feat.shape[-1]} and {patch_tokens.shape[-1]}."
        )
    if temp <= 0:
        raise ValueError(f"temp must be > 0, got {temp}.")

    global_feat = global_feat.float()
    patch_tokens = patch_tokens.float()
    if gamma <= 0:
        return global_feat
    if patch_tokens.shape[1] == 0:
        return global_feat

    score_global = global_feat.detach() if detach_global else global_feat
    relevance = F.cosine_similarity(
        patch_tokens, score_global.unsqueeze(1), dim=-1
    )  # [B, N]

    # Clamp requested top-r to the available patch count for safe fallback.
    top_r = max(1, min(int(top_r), patch_tokens.shape[1]))
    top_scores, top_indices = torch.topk(
        relevance, k=top_r, dim=1, largest=True, sorted=False
    )  # [B, top_r]
    gather_index = top_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
    top_patches = torch.gather(patch_tokens, dim=1, index=gather_index)  # [B, top_r, C]

    weights = torch.softmax(top_scores / float(temp), dim=1)  # [B, top_r]
    local_feat = torch.sum(weights.unsqueeze(-1) * top_patches, dim=1)  # [B, C]
    refined_global = F.normalize(
        global_feat + float(gamma) * local_feat, p=2, dim=-1
    )
    return refined_global


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        # CLIP backbone (shared by RGB and Thermal).
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size, args=args
        )
        self.embed_dim = base_cfg["embed_dim"]

        # Temperature scalar used by global alignment losses.
        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        # Ablation switches.
        self.use_rgbt_add_fusion = getattr(args, "use_rgbt_add_fusion", True)
        self.use_rgbt_moe_adapter = bool(
            getattr(args, "use_rgbt_moe_adapter", False)
        )
        self.use_mum = bool(getattr(args, "use_mum", False))
        self.use_global_uncertainty = bool(
            getattr(args, "use_global_uncertainty", False)
        )
        self.uncertainty_dist_type = str(
            getattr(args, "uncertainty_dist_type", "diag_gaussian")
        ).lower()
        self.logvar_min = float(getattr(args, "logvar_min", -6.0))
        self.logvar_max = float(getattr(args, "logvar_max", 4.0))
        self.uncertainty_loss_type = str(
            getattr(args, "uncertainty_loss_type", "w2")
        ).lower()
        self.uncertainty_eps = float(getattr(args, "uncertainty_eps", 1e-6))
        configured_precision_fusion = getattr(args, "use_precision_fusion", None)
        if configured_precision_fusion is None:
            # Default: enable precision fusion whenever global uncertainty is enabled.
            self.use_precision_fusion = self.use_global_uncertainty
        else:
            self.use_precision_fusion = bool(configured_precision_fusion)
        self.eval_use_fused_mu = bool(getattr(args, "eval_use_fused_mu", True))
        self.eval_use_mu_only = bool(getattr(args, "eval_use_mu_only", False))
        self.use_modality_text_prompt = bool(
            getattr(args, "use_modality_text_prompt", False)
        )
        self.mtp_rgb_ctx_len = int(getattr(args, "mtp_rgb_ctx_len", 4))
        self.mtp_th_ctx_len = int(getattr(args, "mtp_th_ctx_len", 4))
        if self.use_global_uncertainty and self.uncertainty_dist_type != "diag_gaussian":
            raise ValueError(
                f"Unsupported uncertainty_dist_type={self.uncertainty_dist_type}. "
                "Only 'diag_gaussian' is currently implemented."
            )
        if self.mtp_rgb_ctx_len < 0:
            raise ValueError(
                f"mtp_rgb_ctx_len must be >= 0, got {self.mtp_rgb_ctx_len}."
            )
        if self.mtp_th_ctx_len < 0:
            raise ValueError(
                f"mtp_th_ctx_len must be >= 0, got {self.mtp_th_ctx_len}."
            )

        # Global uncertainty heads (pooled global feature -> mu/logvar).
        # They are only activated by use_global_uncertainty switch.
        self.text_mu_head = nn.Linear(self.embed_dim, self.embed_dim)
        self.text_logvar_head = nn.Linear(self.embed_dim, self.embed_dim)
        self.rgb_mu_head = nn.Linear(self.embed_dim, self.embed_dim)
        self.rgb_logvar_head = nn.Linear(self.embed_dim, self.embed_dim)
        self.thermal_mu_head = nn.Linear(self.embed_dim, self.embed_dim)
        self.thermal_logvar_head = nn.Linear(self.embed_dim, self.embed_dim)
        self._init_global_uncertainty_heads()

        self.use_batch_uncertainty = getattr(
            args, "use_batch_uncertainty", getattr(args, "mum_use_batch_var", True)
        )
        self.use_id_uncertainty = getattr(
            args, "use_id_uncertainty", getattr(args, "mum_use_id_var", True)
        )
        self.use_prob_sampling = getattr(
            args, "use_prob_sampling", getattr(args, "mum_sample_enabled", False)
        )
        configured_bank_size = int(getattr(args, "memory_bank_size", 0) or 0)
        bank_size = (
            max(self.num_classes, configured_bank_size)
            if configured_bank_size > 0
            else self.num_classes
        )
        self.global_uncertainty = GlobalUncertaintyModeler(
            num_classes=bank_size,
            feat_dim=self.embed_dim,
            omega=getattr(args, "omega", getattr(args, "mum_omega", 0.5)),
            scale=getattr(
                args, "uncertainty_scale", getattr(args, "mum_scale", 1.0)
            ),
            use_batch_var=self.use_batch_uncertainty,
            use_id_var=self.use_id_uncertainty,
            sample_enabled=self.use_prob_sampling,
            # Keep evaluation deterministic regardless of use_prob_sampling.
            sample_on_eval=False,
            memory_momentum=getattr(args, "mum_memory_momentum", 0.9),
            eps=getattr(args, "mum_eps", self.uncertainty_eps),
        )

        # Optional identity prototype bank over fused visual global features.
        # Keep it fully disabled unless use_proto_calib=True.
        self.use_proto_calib = bool(getattr(args, "use_proto_calib", False))
        self.lambda_proto_v = float(getattr(args, "lambda_proto_v", 1.0))
        self.lambda_proto_t = float(getattr(args, "lambda_proto_t", 1.0))
        self.proto_temp = float(getattr(args, "proto_temp", 0.07))
        self.proto_momentum = float(getattr(args, "proto_momentum", 0.9))
        self.proto_warmup_epochs = int(getattr(args, "proto_warmup_epochs", 0))
        self.proto_runtime_epoch = 1
        if self.proto_temp <= 0:
            raise ValueError(f"proto_temp must be > 0, got {self.proto_temp}.")
        if self.proto_warmup_epochs < 0:
            raise ValueError(
                "proto_warmup_epochs must be >= 0, "
                f"got {self.proto_warmup_epochs}."
            )
        self.prototype_bank = None
        if self.use_proto_calib:
            # Default integration uses zero init; the bank module also supports
            # random-normalized init for later experiments.
            self.prototype_bank = IdentityPrototypeBank(
                num_classes=self.num_classes,
                feat_dim=self.embed_dim,
                momentum=self.proto_momentum,
                init_mode="zero",
                eps=self.uncertainty_eps,
            )

        # Optional image-only local filtering over patch tokens.
        self.use_image_local_filter = bool(
            getattr(args, "use_image_local_filter", False)
        )
        # Runtime switch defaults to legacy behavior until the trainer updates it.
        self.img_local_filter_active = self.use_image_local_filter
        self.img_local_topk = int(getattr(args, "img_local_topk", 64))
        self.img_local_unc_lambda = float(
            getattr(args, "img_local_unc_lambda", 0.3)
        )
        self.img_local_fuse_gamma = float(
            getattr(args, "img_local_fuse_gamma", 0.1)
        )
        self.img_local_score_temp = float(
            getattr(args, "img_local_score_temp", 0.07)
        )
        self.img_local_detach_global = bool(
            getattr(args, "img_local_detach_global", True)
        )
        self.use_img_local_filter = bool(
            getattr(args, "use_img_local_filter", False)
        )
        self.img_local_topr = int(getattr(args, "img_local_topr", 4))
        self.img_local_gamma = float(getattr(args, "img_local_gamma", 0.1))
        self.img_local_temp = float(getattr(args, "img_local_temp", 0.07))
        if self.img_local_topk <= 0:
            raise ValueError(
                f"img_local_topk must be > 0, got {self.img_local_topk}."
            )
        if self.img_local_score_temp <= 0:
            raise ValueError(
                "img_local_score_temp must be > 0, "
                f"got {self.img_local_score_temp}."
            )
        if self.img_local_topr <= 0:
            raise ValueError(
                f"img_local_topr must be > 0, got {self.img_local_topr}."
            )
        if self.img_local_temp <= 0:
            raise ValueError(
                f"img_local_temp must be > 0, got {self.img_local_temp}."
            )
        self.img_local_unc_head = None

        if "id" in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if "mlm" in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(
                self.embed_dim, self.embed_dim // 64, batch_first=True
            )
            self.cross_modal_transformer = Transformer(
                width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64
            )
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5

            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict(
                    [
                        ("dense", nn.Linear(self.embed_dim, self.embed_dim)),
                        ("gelu", QuickGELU()),
                        ("ln", LayerNorm(self.embed_dim)),
                        ("fc", nn.Linear(self.embed_dim, args.vocab_size)),
                    ]
                )
            )
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        if self.use_image_local_filter:
            # Only instantiate the local uncertainty head when the module is enabled.
            # This keeps baseline initialization and RNG consumption unchanged when off.
            self.img_local_unc_head = nn.Linear(self.embed_dim, 1)
            nn.init.constant_(self.img_local_unc_head.weight.data, 0.0)
            nn.init.constant_(self.img_local_unc_head.bias.data, 0.0)
            self.effective_img_local_fuse_gamma = self.img_local_fuse_gamma

    def set_img_local_runtime(self, active, fuse_gamma):
        if not self.use_image_local_filter:
            self.img_local_filter_active = False
            if hasattr(self, "effective_img_local_fuse_gamma"):
                delattr(self, "effective_img_local_fuse_gamma")
            return

        runtime_gamma = float(fuse_gamma)
        self.img_local_filter_active = bool(active) and runtime_gamma > 0.0
        if self.img_local_filter_active:
            self.effective_img_local_fuse_gamma = runtime_gamma
        else:
            self.effective_img_local_fuse_gamma = 0.0

    def set_img_local_filter_active(self, active):
        default_runtime_gamma = self.img_local_fuse_gamma if active else 0.0
        self.set_img_local_runtime(active, default_runtime_gamma)

    def set_proto_runtime_epoch(self, epoch):
        self.proto_runtime_epoch = int(epoch)

    def _proto_calib_enabled(self):
        return self.use_proto_calib and self.prototype_bank is not None

    def should_update_proto_bank(self):
        if not self._proto_calib_enabled():
            return False
        return self.training

    def should_enable_proto_loss(self):
        if not self._proto_calib_enabled():
            return False
        if not self.training:
            return False
        # Chosen warmup behavior: keep updating the bank, but delay prototype loss.
        return int(self.proto_runtime_epoch) > self.proto_warmup_epochs

    def _should_update_proto_bank(self):
        if not self.use_proto_calib or self.prototype_bank is None:
            return False
        if not self.training:
            return False
        return True

    def get_visual_prototype_matrix(self):
        if self.prototype_bank is None:
            return None
        return self.prototype_bank.get_normalized_prototypes()

    def get_visual_prototype_initialized_mask(self):
        if self.prototype_bank is None:
            return None
        return self.prototype_bank.get_initialized_mask()

    def update_visual_prototype_bank(self, fused_feats, pids):
        if not self._should_update_proto_bank():
            return {
                "updated": False,
                "num_updated_pids": 0,
                "num_newly_initialized_pids": 0,
                "num_initialized_pids": 0,
            }
        return self.prototype_bank.update(fused_feats, pids)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split("+")]
        print(f"Training Model with {self.current_task} tasks")

    def _init_global_uncertainty_heads(self):
        # Initialize mu head as near-identity projection and logvar head as zeros.
        for head in [self.text_mu_head, self.rgb_mu_head, self.thermal_mu_head]:
            nn.init.eye_(head.weight.data)
            nn.init.constant_(head.bias.data, 0.0)
        for head in [
            self.text_logvar_head,
            self.rgb_logvar_head,
            self.thermal_logvar_head,
        ]:
            nn.init.constant_(head.weight.data, 0.0)
            nn.init.constant_(head.bias.data, 0.0)

    def _use_diag_gaussian_global(self):
        return self.use_global_uncertainty and self.uncertainty_dist_type == "diag_gaussian"

    def _project_global_gaussian(self, feats, mu_head, logvar_head):
        # Keep head computation dtype-compatible with fp16-converted parameters.
        mu = mu_head(feats.to(mu_head.weight.dtype)).float()
        logvar = logvar_head(feats.to(logvar_head.weight.dtype)).float()
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        return mu, logvar

    def _encode_text_global_gaussian(self, text_global_feat):
        if not self._use_diag_gaussian_global():
            return text_global_feat, None
        return self._project_global_gaussian(
            text_global_feat, self.text_mu_head, self.text_logvar_head
        )

    def _encode_caption_ids_with_uncertainty(self, caption_ids):
        text_tokens = self.base_model.encode_text(caption_ids)  # [B, L_t, C]
        batch_index = torch.arange(text_tokens.shape[0], device=text_tokens.device)
        text_global_raw = text_tokens[
            batch_index, caption_ids.argmax(dim=-1)
        ].float()  # [B, C]
        text_mu, text_logvar = self._encode_text_global_gaussian(text_global_raw)
        return text_tokens, text_global_raw, text_mu, text_logvar

    def _encode_visual_global_gaussian(
        self, rgb_global_feat, thermal_global_feat, thermal_available
    ):
        if not self._use_diag_gaussian_global():
            return rgb_global_feat, None, thermal_global_feat, None

        rgb_mu, rgb_logvar = self._project_global_gaussian(
            rgb_global_feat, self.rgb_mu_head, self.rgb_logvar_head
        )
        if thermal_available:
            thermal_mu, thermal_logvar = self._project_global_gaussian(
                thermal_global_feat, self.thermal_mu_head, self.thermal_logvar_head
            )
        else:
            # Keep single-image datasets fully backward compatible.
            thermal_mu = torch.zeros_like(rgb_mu)
            thermal_logvar = torch.zeros_like(rgb_logvar)
        return rgb_mu, rgb_logvar, thermal_mu, thermal_logvar

    def _safe_var_from_logvar(self, logvar):
        # var = exp(logvar), with clamped logvar for numeric stability.
        clamped_logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        return torch.exp(clamped_logvar).clamp_min(self.uncertainty_eps)

    def _safe_logvar_from_var(self, var):
        # log(var + eps) with clamped var floor.
        safe_var = var.clamp_min(self.uncertainty_eps)
        logvar = torch.log(safe_var + self.uncertainty_eps)
        return torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)

    def _build_eval_precision_weights(
        self, rgb_logvar, thermal_logvar, thermal_available
    ):
        """
        Build scalar RGB / Thermal precision weights for eval-time local reranking.

        Source:
            The weights come from the same diagonal-Gaussian log-variance heads used
            by the model's global uncertainty branch.

        Normalization:
            precision = 1 / (var + eps), var = exp(logvar)
            modality scalar precision = mean over feature dimension
            weight_rgb = precision_rgb / (precision_rgb + precision_th + eps)
            weight_th = precision_th / (precision_rgb + precision_th + eps)

        When uncertainty information is unavailable, return 0.5 / 0.5 and mark the
        weights invalid so downstream code can safely fall back to avg fusion.
        """
        reference = None
        if rgb_logvar is not None:
            reference = rgb_logvar
        elif thermal_logvar is not None:
            reference = thermal_logvar
        else:
            return None, None, None

        batch_size = reference.shape[0]
        default_half = reference.new_full((batch_size,), 0.5).float()
        invalid_mask = torch.zeros(
            batch_size, device=reference.device, dtype=torch.bool
        )

        if (
            (not thermal_available)
            or (rgb_logvar is None)
            or (thermal_logvar is None)
            or (not self._use_diag_gaussian_global())
        ):
            return default_half, default_half, invalid_mask

        rgb_precision = 1.0 / (
            self._safe_var_from_logvar(rgb_logvar) + self.uncertainty_eps
        )
        thermal_precision = 1.0 / (
            self._safe_var_from_logvar(thermal_logvar) + self.uncertainty_eps
        )
        rgb_precision_scalar = rgb_precision.mean(dim=1)
        thermal_precision_scalar = thermal_precision.mean(dim=1)
        precision_sum = (
            rgb_precision_scalar + thermal_precision_scalar
        ).clamp_min(self.uncertainty_eps)
        rgb_weight = (rgb_precision_scalar / precision_sum).float()
        thermal_weight = (thermal_precision_scalar / precision_sum).float()
        valid_mask = torch.isfinite(rgb_weight) & torch.isfinite(thermal_weight)
        rgb_weight = torch.where(valid_mask, rgb_weight, default_half)
        thermal_weight = torch.where(valid_mask, thermal_weight, default_half)
        return rgb_weight, thermal_weight, valid_mask

    def _fuse_rgbt_global(
        self,
        rgb_mu,
        rgb_logvar,
        thermal_mu,
        thermal_logvar,
        thermal_available,
    ):
        """
        Fuse RGB/T thermal global features.

        If diagonal-Gaussian global uncertainty is enabled and use_precision_fusion=True:
          precision = 1 / (var + eps), var = exp(logvar)
          precision_fused = precision_rgb + precision_th
          mu_fused = (precision_rgb * mu_rgb + precision_th * mu_th) / precision_fused
          var_fused = 1 / precision_fused
          logvar_fused = log(var_fused + eps)

        Otherwise fallback to baseline-style deterministic add fusion.
        """
        if not self._use_diag_gaussian_global():
            if self.use_rgbt_add_fusion:
                fused_mu = rgb_mu + thermal_mu
            else:
                fused_mu = rgb_mu
            fused_logvar = torch.zeros_like(fused_mu)
            return fused_mu, fused_logvar

        rgb_var = self._safe_var_from_logvar(rgb_logvar)
        thermal_var = self._safe_var_from_logvar(thermal_logvar)

        if thermal_available and self.use_precision_fusion:
            precision_rgb = 1.0 / (rgb_var + self.uncertainty_eps)
            precision_th = 1.0 / (thermal_var + self.uncertainty_eps)
            precision_fused = (precision_rgb + precision_th).clamp_min(
                self.uncertainty_eps
            )
            fused_mu = (
                precision_rgb * rgb_mu + precision_th * thermal_mu
            ) / precision_fused
            fused_var = 1.0 / precision_fused
            fused_logvar = self._safe_logvar_from_var(fused_var)
            return fused_mu, fused_logvar

        # Fallback: keep baseline fused representation path.
        if self.use_rgbt_add_fusion and thermal_available:
            fused_mu = rgb_mu + thermal_mu
            fused_var = rgb_var + thermal_var
        else:
            fused_mu = rgb_mu
            fused_var = rgb_var
        fused_logvar = self._safe_logvar_from_var(fused_var)
        return fused_mu, fused_logvar

    def compute_uncertainty_distance(
        self,
        mu1,
        logvar1,
        mu2,
        logvar2,
        pairwise=False,
        squared=True,
    ):
        """
        Compute closed-form uncertainty distance for global Gaussian features.
        Currently supports diagonal-Gaussian W2 family.
        """
        if self.uncertainty_loss_type != "w2":
            raise ValueError(
                f"Unsupported uncertainty_loss_type={self.uncertainty_loss_type}. "
                "Only 'w2' is currently implemented."
            )
        return objectives.compute_w2_diag_distance(
            mu1=mu1,
            logvar1=logvar1,
            mu2=mu2,
            logvar2=logvar2,
            pairwise=pairwise,
            squared=squared,
            eps=self.uncertainty_eps,
            logvar_min=self.logvar_min,
            logvar_max=self.logvar_max,
        )

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q), self.ln_pre_i(k), self.ln_pre_i(v), need_weights=False
        )[0]
        x = x.permute(1, 0, 2)  # [B, L, C] -> [L, B, C]
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # [L, B, C] -> [B, L, C]
        x = self.ln_post(x)
        return x

    def _extract_patch_tokens(self, image_tokens, reference_feat=None):
        """
        Safely extract patch tokens from image backbone outputs.

        Returns:
            [B, N, C] when patch tokens exist
            [B, 0, C] when the current path has no patch tokens
            None when a safe shape cannot be inferred
        """
        if image_tokens is None:
            return None
        if image_tokens.dim() == 3:
            return image_tokens[:, 1:, :].float()
        if reference_feat is not None:
            return reference_feat.new_zeros(
                reference_feat.shape[0],
                0,
                reference_feat.shape[-1],
            )
        return None

    def _encode_visual_tokens(
        self, image, modality, return_adapter_aux=False
    ):
        if return_adapter_aux and self.use_rgbt_moe_adapter:
            return self.base_model.encode_image(
                image,
                modality=modality,
                return_adapter_aux=True,
            )
        image_tokens = self.base_model.encode_image(image, modality=modality)
        return image_tokens, None

    def _merge_rgbt_moe_adapter_aux(self, rgb_aux=None, thermal_aux=None):
        if rgb_aux is None and thermal_aux is None:
            return None

        merged_aux = {
            "layer_indices": [],
            "router_weights": {"rgb": {}, "thermal": {}},
            "rgb_shared_avg_weights": {},
            "rgb_specific_avg_weights": {},
            "thermal_shared_avg_weights": {},
            "thermal_specific_avg_weights": {},
            "shared_branch_pooled_features": {"rgb": {}, "thermal": {}},
        }

        layer_indices = set()
        if rgb_aux is not None:
            layer_indices.update(rgb_aux.get("layer_indices", []))
            for layer_aux in rgb_aux.get("per_layer", []):
                layer_key = str(layer_aux["layer_index"])
                layer_indices.add(int(layer_aux["layer_index"]))
                merged_aux["router_weights"]["rgb"][layer_key] = layer_aux[
                    "router_weights"
                ]
                merged_aux["rgb_shared_avg_weights"][layer_key] = layer_aux[
                    "avg_shared_weight"
                ]
                merged_aux["rgb_specific_avg_weights"][layer_key] = layer_aux[
                    "avg_rgb_specific_weight"
                ]
                merged_aux["shared_branch_pooled_features"]["rgb"][
                    layer_key
                ] = layer_aux["shared_pooled_feature"]

        if thermal_aux is not None:
            layer_indices.update(thermal_aux.get("layer_indices", []))
            for layer_aux in thermal_aux.get("per_layer", []):
                layer_key = str(layer_aux["layer_index"])
                layer_indices.add(int(layer_aux["layer_index"]))
                merged_aux["router_weights"]["thermal"][layer_key] = layer_aux[
                    "router_weights"
                ]
                merged_aux["thermal_shared_avg_weights"][layer_key] = layer_aux[
                    "avg_shared_weight"
                ]
                merged_aux["thermal_specific_avg_weights"][layer_key] = layer_aux[
                    "avg_thermal_specific_weight"
                ]
                merged_aux["shared_branch_pooled_features"]["thermal"][
                    layer_key
                ] = layer_aux["shared_pooled_feature"]

        merged_aux["layer_indices"] = sorted(layer_indices)
        return merged_aux

    def _maybe_apply_img_local_filter_v1(self, global_feat, patch_tokens):
        if not self.use_img_local_filter:
            return global_feat
        if global_feat is None or patch_tokens is None:
            return global_feat
        if patch_tokens.dim() != 3:
            return global_feat
        if patch_tokens.shape[0] != global_feat.shape[0]:
            return global_feat
        if patch_tokens.shape[-1] != global_feat.shape[-1]:
            return global_feat
        if patch_tokens.shape[1] == 0:
            return global_feat
        return apply_image_local_filter(
            global_feat=global_feat,
            patch_tokens=patch_tokens,
            top_r=self.img_local_topr,
            gamma=self.img_local_gamma,
            temp=self.img_local_temp,
            detach_global=self.img_local_detach_global,
        )

    def _apply_image_local_filter(self, image_tokens):
        """
        Build the visual global feature from CLIP image tokens.

        Args:
            image_tokens: [B, 1 + N, C], where token 0 is CLS and tokens 1..N are
                patch tokens.
        Returns:
            visual_global: [B, C]
                - baseline CLS token when img_local_filter_active=False
                - image-only locally filtered global feature otherwise
            local_stats: dict[str, torch.Tensor]
                Small scalar statistics for training-time observability.
        """
        if image_tokens.dim() == 2:
            # Safe fallback for backbones/paths that only expose a single global feature.
            return image_tokens.float(), None

        # Keep the baseline global definition unchanged: raw CLS token.
        g_global = image_tokens[:, 0, :].float()  # [B, C]
        runtime_gamma = float(getattr(self, "effective_img_local_fuse_gamma", 0.0))
        if (not self.img_local_filter_active) or runtime_gamma <= 0.0:
            return g_global, None

        patch_tokens = image_tokens[:, 1:, :].float()  # [B, N, C], no CLS
        if patch_tokens.shape[1] == 0:
            zero = g_global.new_zeros(())
            return g_global, {
                "avg_img_local_uncertainty": zero,
                "avg_img_local_keep_ratio": zero,
                "avg_img_local_weight_entropy": zero,
            }

        # Predict scalar uncertainty per patch token: u_i = softplus(head(v_i)).
        unc_logits = self.img_local_unc_head(
            patch_tokens.to(self.img_local_unc_head.weight.dtype)
        )
        patch_uncertainty = F.softplus(unc_logits.float()).squeeze(-1)  # [B, N]

        # Image-only relevance against the current baseline global feature.
        score_global = g_global.detach() if self.img_local_detach_global else g_global
        relevance = F.cosine_similarity(
            patch_tokens, score_global.unsqueeze(1), dim=-1
        ) / self.img_local_score_temp  # [B, N]

        # Final token scores: s_i = r_i - lambda_unc * u_i.
        scores = relevance - self.img_local_unc_lambda * patch_uncertainty  # [B, N]

        # Keep top-k tokens per image, mask all non-top-k scores before softmax.
        topk = max(1, min(self.img_local_topk, scores.shape[1]))
        _, topk_indices = torch.topk(scores, k=topk, dim=1, largest=True, sorted=False)
        topk_mask = torch.zeros_like(scores, dtype=torch.bool)  # [B, N]
        topk_mask.scatter_(1, topk_indices, True)
        masked_scores = scores.masked_fill(~topk_mask, float("-inf"))  # [B, N]
        # Softmax is evaluated only on top-k entries; subtract row-wise max for stability.
        masked_scores = masked_scores - masked_scores.max(dim=1, keepdim=True).values
        weights = torch.softmax(masked_scores, dim=1)  # [B, N]

        # Weighted local aggregation over the retained patch tokens only.
        g_local = torch.sum(weights.unsqueeze(-1) * patch_tokens, dim=1)  # [B, C]
        fused_global = (
            (1.0 - runtime_gamma) * g_global
            + runtime_gamma * g_local
        )
        weight_entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=1)
        local_stats = {
            "avg_img_local_uncertainty": patch_uncertainty.mean(),
            "avg_img_local_keep_ratio": scores.new_full(
                (), float(topk) / float(scores.shape[1])
            ),
            "avg_img_local_weight_entropy": weight_entropy.mean(),
        }
        return F.normalize(fused_global, p=2, dim=-1), local_stats  # [B, C]

    def encode_image(self, image, modality="rgb", return_adapter_aux=False):
        # image: [B, 3, H, W] -> image_tokens: [B, L_v, C]
        image_tokens, adapter_aux = self._encode_visual_tokens(
            image,
            modality=modality,
            return_adapter_aux=return_adapter_aux,
        )
        # global cls token: [B, C]
        image_global = image_tokens[:, 0, :].float()
        if return_adapter_aux:
            return image_global, adapter_aux
        return image_global

    def encode_text(self, text):
        # text: [B, L_t] -> text_tokens: [B, L_t, C]
        text_tokens = self.base_model.encode_text(text)
        batch_index = torch.arange(text_tokens.shape[0], device=text_tokens.device)
        # EOT global token: [B, C]
        return text_tokens[batch_index, text.argmax(dim=-1)].float()

    def encode_text_deterministic(self, text):
        # Deterministic global text feature used by retrieval evaluation.
        text_global_feat = self.encode_text(text)
        text_mu, _ = self._encode_text_global_gaussian(text_global_feat)
        # Default deterministic representation remains compatible: use mu if enabled.
        return text_mu

    def encode_rgbt_deterministic(self, rgb_image, thermal_image=None):
        # Deterministic fused visual feature used by retrieval evaluation.
        rgb_global_feat, thermal_global_feat, _, _, _, _ = self.encode_rgbt_global(
            rgb_image, thermal_image
        )
        rgb_mu, rgb_logvar, thermal_mu, thermal_logvar = self._encode_visual_global_gaussian(
            rgb_global_feat,
            thermal_global_feat,
            thermal_available=thermal_image is not None,
        )
        f_c, _ = self._fuse_rgbt_global(
            rgb_mu=rgb_mu,
            rgb_logvar=rgb_logvar,
            thermal_mu=thermal_mu,
            thermal_logvar=thermal_logvar,
            thermal_available=thermal_image is not None,
        )
        if self.eval_use_mu_only:
            return rgb_mu
        if self.eval_use_fused_mu:
            return f_c
        if self.use_rgbt_add_fusion:
            return rgb_mu + thermal_mu
        return rgb_mu

    def _maybe_eval_sampling(self, feats):
        """
        Keep retrieval evaluation deterministic.
        Sampling is only allowed in training mode when probabilistic sampling is enabled.
        """
        if (not self.training) or (not self.use_prob_sampling):
            return feats
        sigma = (
            feats.var(dim=0, unbiased=False, keepdim=True)
            .clamp_min(getattr(self.args, "mum_eps", 1e-6))
            .sqrt()
            .expand_as(feats)
        )
        sigma = sigma * getattr(self.args, "uncertainty_scale", 1.0)
        return feats + torch.randn_like(feats) * sigma

    def _encode_text_eval_outputs(self, text):
        """
        Shared eval-only text encoder that reuses the existing token activations.

        Returns:
            {
                "global_feat": [B, C], deterministic retrieval embedding
                "token_embeddings": [B, L_t, C], text token embeddings
                "global_token_embedding": [B, C], raw EOT token embedding
            }
        """
        text_tokens = self.base_model.encode_text(text)  # [B, L_t, C]
        batch_index = torch.arange(text_tokens.shape[0], device=text_tokens.device)
        text_global_raw = text_tokens[batch_index, text.argmax(dim=-1)].float()  # [B, C]
        text_feat, _ = self._encode_text_global_gaussian(text_global_raw)
        return {
            "global_feat": self._maybe_eval_sampling(text_feat),
            "token_embeddings": text_tokens.float(),
            "global_token_embedding": text_global_raw,
        }

    def encode_text_eval(self, text, return_local_tokens=False):
        if not return_local_tokens:
            feats = self.encode_text_deterministic(text)
            return self._maybe_eval_sampling(feats)
        return self._encode_text_eval_outputs(text)

    def _encode_rgbt_eval_outputs(self, rgb_image, thermal_image=None):
        """
        Shared eval-only visual encoder that reuses the existing image token activations.

        Returns:
            {
                "global_feat": [B, C], deterministic retrieval embedding
                "rgb_global_feat": [B, C], RGB global embedding before fused selection
                "thermal_global_feat": [B, C], Thermal global embedding before fused selection
                "rgb_patch_tokens": [B, Np, C]
                "thermal_patch_tokens": [B, Np, C] or [B, 0, C] when thermal is absent
                "rgb_precision_weight": [B], scalar modality weight for local rerank
                "thermal_precision_weight": [B], scalar modality weight for local rerank
                "precision_weight_valid": [B], whether precision weights are valid
            }
        """
        rgb_tokens, _ = self._encode_visual_tokens(
            rgb_image,
            modality="rgb",
            return_adapter_aux=False,
        )
        rgb_global_feat, _ = self._apply_image_local_filter(rgb_tokens)  # [B, C]
        rgb_patch_tokens = self._extract_patch_tokens(
            rgb_tokens, reference_feat=rgb_global_feat
        )
        rgb_global_feat = self._maybe_apply_img_local_filter_v1(
            rgb_global_feat, rgb_patch_tokens
        )

        if thermal_image is not None:
            thermal_tokens, _ = self._encode_visual_tokens(
                thermal_image,
                modality="thermal",
                return_adapter_aux=False,
            )
            thermal_global_feat, _ = self._apply_image_local_filter(
                thermal_tokens
            )  # [B, C]
            thermal_patch_tokens = self._extract_patch_tokens(
                thermal_tokens, reference_feat=thermal_global_feat
            )
            thermal_global_feat = self._maybe_apply_img_local_filter_v1(
                thermal_global_feat, thermal_patch_tokens
            )
        else:
            thermal_global_feat = torch.zeros_like(rgb_global_feat)
            thermal_patch_tokens = rgb_patch_tokens.new_zeros(
                rgb_patch_tokens.shape[0],
                0,
                rgb_patch_tokens.shape[-1],
            )

        rgb_mu, rgb_logvar, thermal_mu, thermal_logvar = self._encode_visual_global_gaussian(
            rgb_global_feat,
            thermal_global_feat,
            thermal_available=thermal_image is not None,
        )
        rgb_precision_weight, thermal_precision_weight, precision_weight_valid = (
            self._build_eval_precision_weights(
                rgb_logvar=rgb_logvar,
                thermal_logvar=thermal_logvar,
                thermal_available=thermal_image is not None,
            )
        )
        fused_feat, _ = self._fuse_rgbt_global(
            rgb_mu=rgb_mu,
            rgb_logvar=rgb_logvar,
            thermal_mu=thermal_mu,
            thermal_logvar=thermal_logvar,
            thermal_available=thermal_image is not None,
        )

        if self.eval_use_mu_only:
            global_feat = rgb_mu
        elif self.eval_use_fused_mu:
            global_feat = fused_feat
        elif self.use_rgbt_add_fusion:
            global_feat = rgb_mu + thermal_mu
        else:
            global_feat = rgb_mu

        return {
            "global_feat": self._maybe_eval_sampling(global_feat),
            "rgb_global_feat": self._maybe_eval_sampling(rgb_mu),
            "thermal_global_feat": self._maybe_eval_sampling(thermal_mu),
            "rgb_patch_tokens": rgb_patch_tokens,
            "thermal_patch_tokens": thermal_patch_tokens,
            "rgb_precision_weight": rgb_precision_weight,
            "thermal_precision_weight": thermal_precision_weight,
            "precision_weight_valid": precision_weight_valid,
        }

    def encode_rgbt_eval(self, rgb_image, thermal_image=None, return_local_tokens=False):
        if not return_local_tokens:
            feats = self.encode_rgbt_deterministic(rgb_image, thermal_image)
            return self._maybe_eval_sampling(feats)
        return self._encode_rgbt_eval_outputs(rgb_image, thermal_image)

    def encode_text_eval_with_tokens(self, text):
        """
        Backward-compatible eval helper for local rerank modules.

        Returns:
            {
                "global_feat": [B, C], deterministic retrieval embedding
                "token_embeddings": [B, L_t, C], text token embeddings
                "global_token_embedding": [B, C], raw EOT token embedding for saliency
            }
        """
        return self._encode_text_eval_outputs(text)

    def encode_rgbt_eval_with_patches(self, rgb_image, thermal_image=None):
        """
        Backward-compatible eval helper for local rerank modules.

        Returns:
            {
                "global_feat": [B, C], deterministic retrieval embedding
                "rgb_patch_tokens": [B, Np, C]
                "thermal_patch_tokens": [B, Np, C] or [B, 0, C] when thermal is absent
            }
        """
        return self._encode_rgbt_eval_outputs(rgb_image, thermal_image)

    def encode_rgbt_global(self, rgb_image, thermal_image=None):
        """
        Args:
            rgb_image: [B, 3, H, W]
            thermal_image: [B, 3, H, W] or None
        Returns:
            v_rgb_c: [B, C]
            v_t_c: [B, C]
            f_c: [B, C], fused by element-wise addition
            rgb_tokens: [B, L_v, C]
            moe_adapter_aux: dict or None
        """
        rgb_tokens, rgb_moe_adapter_aux = self._encode_visual_tokens(
            rgb_image,
            modality="rgb",
            return_adapter_aux=self.use_rgbt_moe_adapter,
        )
        v_rgb_c, rgb_local_stats = self._apply_image_local_filter(rgb_tokens)  # [B, C]
        rgb_patch_tokens = self._extract_patch_tokens(
            rgb_tokens, reference_feat=v_rgb_c
        )
        v_rgb_c = self._maybe_apply_img_local_filter_v1(v_rgb_c, rgb_patch_tokens)

        if thermal_image is not None:
            thermal_tokens, thermal_moe_adapter_aux = self._encode_visual_tokens(
                thermal_image,
                modality="thermal",
                return_adapter_aux=self.use_rgbt_moe_adapter,
            )
            v_t_c, thermal_local_stats = self._apply_image_local_filter(
                thermal_tokens
            )  # [B, C]
            thermal_patch_tokens = self._extract_patch_tokens(
                thermal_tokens, reference_feat=v_t_c
            )
            v_t_c = self._maybe_apply_img_local_filter_v1(
                v_t_c, thermal_patch_tokens
            )
        else:
            # Backward compatibility for old single-image datasets.
            v_t_c = torch.zeros_like(v_rgb_c)  # [B, C]
            thermal_local_stats = None
            thermal_moe_adapter_aux = None

        if self.use_rgbt_add_fusion:
            f_c = v_rgb_c + v_t_c  # [B, C]
        else:
            # Baseline ablation: RGB-only global feature.
            f_c = v_rgb_c
        local_stats = None
        modality_stats = [
            stats for stats in [rgb_local_stats, thermal_local_stats] if stats is not None
        ]
        if modality_stats:
            local_stats = {}
            for key in modality_stats[0].keys():
                local_stats[key] = torch.stack(
                    [stats[key] for stats in modality_stats], dim=0
                ).mean()
        moe_adapter_aux = self._merge_rgbt_moe_adapter_aux(
            rgb_aux=rgb_moe_adapter_aux,
            thermal_aux=thermal_moe_adapter_aux,
        )
        return v_rgb_c, v_t_c, f_c, rgb_tokens, local_stats, moe_adapter_aux

    def _deterministic_uncertainty_outputs(self, f_c, l_c):
        zero_sigma = torch.zeros_like(f_c)
        return {
            "fused_feat_det": f_c,           # [B, C]
            "text_feat_det": l_c,            # [B, C]
            "fused_feat_sampled": f_c,       # [B, C]
            "text_feat_sampled": l_c,        # [B, C]
            "sigma_batch_f": zero_sigma,     # [B, C]
            "sigma_batch_t": zero_sigma,     # [B, C]
            "sigma_id_f": zero_sigma,        # [B, C]
            "sigma_id_t": zero_sigma,        # [B, C]
            "sigma_unify_f": zero_sigma,     # [B, C]
            "sigma_unify_t": zero_sigma,     # [B, C]
        }

    def forward(self, batch):
        ret = {}

        # Input tensors:
        # images: [B, 3, H, W]
        # thermal_images: [B, 3, H, W] (optional for compatibility)
        # caption_ids: [B, L_t]
        images = batch["images"]
        thermal_images = batch.get("thermal_images", None)
        caption_ids = batch["caption_ids"]
        caption_ids_rgb = None
        caption_ids_th = None
        if self.use_modality_text_prompt:
            caption_ids_rgb = batch.get("caption_ids_rgb", caption_ids)
            caption_ids_th = batch.get("caption_ids_th", caption_ids)

        # 1) RGB/Thermal pooled global branch (before uncertainty heads).
        (
            v_rgb_c_raw,
            v_t_c_raw,
            _,
            rgb_tokens,
            img_local_stats,
            rgbt_moe_adapter_aux,
        ) = self.encode_rgbt_global(
            images, thermal_images
        )

        # 2) Text pooled global branch (before uncertainty heads).
        _, l_c_raw, text_mu, text_logvar = self._encode_caption_ids_with_uncertainty(
            caption_ids
        )
        text_feat_rgb = None
        text_feat_th = None
        if self.use_modality_text_prompt:
            _, _, text_feat_rgb, _ = self._encode_caption_ids_with_uncertainty(
                caption_ids_rgb
            )
            _, _, text_feat_th, _ = self._encode_caption_ids_with_uncertainty(
                caption_ids_th
            )

        # 3) Optional diagonal-Gaussian heads over global pooled features.
        # Returned mu/logvar are used by later uncertainty-aware losses/fusion.
        rgb_mu, rgb_logvar, thermal_mu, thermal_logvar = self._encode_visual_global_gaussian(
            v_rgb_c_raw,
            v_t_c_raw,
            thermal_available=thermal_images is not None,
        )
        fused_mu, fused_logvar = self._fuse_rgbt_global(
            rgb_mu=rgb_mu,
            rgb_logvar=rgb_logvar,
            thermal_mu=thermal_mu,
            thermal_logvar=thermal_logvar,
            thermal_available=thermal_images is not None,
        )

        # Keep existing training/eval flow compatible: deterministic embedding is mu.
        v_rgb_c = rgb_mu
        v_t_c = thermal_mu
        l_c = text_mu
        f_c = fused_mu

        # 4) MUM-style uncertainty over deterministic global embeddings.
        if self.use_mum:
            uncertainty_outputs = self.global_uncertainty(
                fused_feats=f_c,
                text_feats=l_c,
                pids=batch.get("pids", None),
            )
        else:
            uncertainty_outputs = self._deterministic_uncertainty_outputs(f_c, l_c)

        p_f = uncertainty_outputs["fused_feat_sampled"]  # [B, C]
        p_t = uncertainty_outputs["text_feat_sampled"]   # [B, C]
        f_det = uncertainty_outputs["fused_feat_det"]    # [B, C]
        t_det = uncertainty_outputs["text_feat_det"]     # [B, C]

        # Required outputs for downstream losses/logging.
        ret.update(uncertainty_outputs)
        if img_local_stats is not None:
            ret.update(img_local_stats)
        if rgbt_moe_adapter_aux is not None:
            ret["rgbt_moe_adapter_aux"] = rgbt_moe_adapter_aux
        ret.update(
            {
                "v_rgb_c": v_rgb_c,  # [B, C]
                "v_t_c": v_t_c,      # [B, C]
                "fused_mu": fused_mu,            # [B, C]
                "fused_logvar": fused_logvar,    # [B, C]
                # Canonical aliases requested by trainer side.
                "fused_det_feat": f_det,
                "text_det_feat": t_det,
                "fused_prob_feat": p_f,
                "text_prob_feat": p_t,
            }
        )
        if self.use_modality_text_prompt:
            ret.update(
                {
                    "text_feat_full": l_c,
                    "text_feat_rgb": text_feat_rgb,
                    "text_feat_th": text_feat_th,
                }
            )
        if self._use_diag_gaussian_global():
            ret.update(
                {
                    # Per-modality global Gaussian parameters for future
                    # uncertainty-specific alignment/fusion losses.
                    "text_mu": text_mu,               # [B, C]
                    "text_logvar": text_logvar,       # [B, C]
                    "rgb_mu": rgb_mu,                 # [B, C]
                    "rgb_logvar": rgb_logvar,         # [B, C]
                    "thermal_mu": thermal_mu,         # [B, C]
                    "thermal_logvar": thermal_logvar, # [B, C]
                }
            )

        logit_scale = self.logit_scale
        ret.update({"temperature": 1 / logit_scale})

        # Global alignment losses are computed on sampled features.
        if "itc" in self.current_task:
            ret.update({"itc_loss": objectives.compute_itc(p_f, p_t, logit_scale)})

        if "sdm" in self.current_task:
            loss_sdm_prob = objectives.compute_sdm(
                p_f, p_t, batch["pids"], logit_scale
            )
            ret.update(
                {
                    "loss_sdm_prob": loss_sdm_prob,
                    # Keep legacy key for compatibility with existing scripts.
                    "sdm_loss": loss_sdm_prob,
                }
            )
            if self.use_mum:
                ret.update(
                    {
                        "loss_sdm_det": objectives.compute_sdm(
                            f_det, t_det, batch["pids"], logit_scale
                        )
                    }
                )

        if "cmpm" in self.current_task:
            ret.update({"cmpm_loss": objectives.compute_cmpm(p_f, p_t, batch["pids"])})

        if "id" in self.current_task:
            # Keep ID branch deterministic.
            cls_dtype = self.classifier.weight.dtype
            image_logits = self.classifier(f_c.to(cls_dtype)).float()
            text_logits = self.classifier(l_c.to(cls_dtype)).float()
            ret.update(
                {
                    "id_loss": objectives.compute_id(
                        image_logits, text_logits, batch["pids"]
                    )
                    * self.args.id_loss_weight
                }
            )

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)
            ret.update({"img_acc": (image_pred == batch["pids"]).float().mean()})
            ret.update({"txt_acc": (text_pred == batch["pids"]).float().mean()})

        # Keep local/MLM branch with RGB image tokens to minimize changes.
        if "mlm" in self.current_task:
            mlm_ids = batch["mlm_ids"]
            mlm_feats = self.base_model.encode_text(mlm_ids)  # [B, L_t, C]
            x = self.cross_former(mlm_feats, rgb_tokens, rgb_tokens)  # [B, L_t, C]
            x = self.mlm_head(x)  # [B, L_t, vocab_size]

            scores = x.float().reshape(-1, self.args.vocab_size)  # [B*L_t, V]
            mlm_labels = batch["mlm_labels"].reshape(-1)  # [B*L_t]
            raw_mlm_loss = objectives.compute_mlm(scores, mlm_labels)
            ret.update(
                {
                    # Legacy weighted key kept for backward compatibility.
                    "mlm_loss": raw_mlm_loss * self.args.mlm_loss_weight
                }
            )
            # Requested probabilistic-training key (unweighted).
            ret.update({"loss_mlm": raw_mlm_loss})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({"mlm_acc": acc})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    convert_weights(model)
    return model
