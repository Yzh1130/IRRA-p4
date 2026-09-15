from collections import OrderedDict
import logging
import os.path as op

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
from .prototype_bank import DualIdentityPrototypeBank, IdentityPrototypeBank
from .mraf import (
    ModalityReliabilityAdaptiveFusion,
    mraf_mem_debug_active,
    mraf_mem_debug_log,
)
from .text_guided_masking import (
    apply_guided_mask_to_input_ids,
    build_guided_mlm_inputs_and_labels,
    build_informative_token_mask,
    build_noise_token_mask,
    compute_token_importance_scores,
    summarize_mask_stats,
)
from .uncertainty import GlobalUncertaintyModeler
from utils.attribute_parser import (
    get_basic_attribute_num_classes,
    get_uspc_slot_local_attr_names,
    get_uspc_slot_region_map,
)
from utils.options import validate_mraf_options


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
        validate_mraf_options(args)
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        # CLIP backbone (shared by RGB and Thermal).
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size, args=args
        )
        self.embed_dim = base_cfg["embed_dim"]
        self.mraf_enabled = bool(getattr(args, "mraf_enabled", False))
        if self.mraf_enabled:
            self.mraf_fusion = ModalityReliabilityAdaptiveFusion(
                feature_dim=self.embed_dim,
                hidden_dim=int(getattr(args, "mraf_hidden_dim", 128)),
                dropout=float(getattr(args, "mraf_dropout", 0.1)),
                gate_temperature=float(
                    getattr(args, "mraf_gate_temperature", 1.0)
                ),
                gate_logit_scale=float(
                    getattr(args, "mraf_gate_logit_scale", 1.0)
                ),
                gate_fp32=bool(getattr(args, "mraf_gate_fp32", False)),
                gate_base_grad_scale=float(
                    getattr(args, "mraf_gate_base_grad_scale", 1.0)
                ),
                equalize_modality_norm=bool(
                    getattr(args, "mraf_equalize_modality_norm", False)
                ),
                freeze_gate=bool(
                    getattr(args, "mraf_freeze_gate", False)
                ),
                rel_isolate_input=bool(
                    getattr(args, "mraf_rel_isolate_input", False)
                ),
            )

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
        self.use_noise_token_masking = bool(
            getattr(args, "use_noise_token_masking", False)
        )
        self.use_informative_token_masking = bool(
            getattr(args, "use_informative_token_masking", False)
        )
        self.noise_mask_ratio = float(getattr(args, "noise_mask_ratio", 0.15))
        self.info_mask_ratio = float(getattr(args, "info_mask_ratio", 0.15))
        self.token_mask_score_source = str(
            getattr(args, "token_mask_score_source", "none")
        ).lower()
        self.token_mask_min_keep = int(getattr(args, "token_mask_min_keep", 1))
        self.noise_mask_apply_to_mlm = bool(
            getattr(args, "noise_mask_apply_to_mlm", False)
        )
        self.info_mask_apply_to_mlm = bool(
            getattr(args, "info_mask_apply_to_mlm", True)
        )
        self.guided_mask_detach_score = bool(
            getattr(args, "guided_mask_detach_score", True)
        )
        self.lambda_guided_mtp = float(getattr(args, "lambda_guided_mtp", 1.0))
        self.debug_print_token_mask_stats = bool(
            getattr(args, "debug_print_token_mask_stats", False)
        )
        self.ablate_random_noise_mask = bool(
            getattr(args, "ablate_random_noise_mask", False)
        )
        self.ablate_random_info_mask = bool(
            getattr(args, "ablate_random_info_mask", False)
        )
        self.guided_mask_token_id = int(getattr(args, "vocab_size", 49408)) - 3
        self.guided_sot_token_id = int(getattr(args, "vocab_size", 49408)) - 2
        self.guided_eot_token_id = int(getattr(args, "vocab_size", 49408)) - 1
        self.guided_mask_special_token_ids = {
            0,
            self.guided_mask_token_id,
            self.guided_sot_token_id,
            self.guided_eot_token_id,
        }
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
        if not 0.0 <= self.noise_mask_ratio <= 1.0:
            raise ValueError(
                f"noise_mask_ratio must be in [0, 1], got {self.noise_mask_ratio}."
            )
        if not 0.0 <= self.info_mask_ratio <= 1.0:
            raise ValueError(
                f"info_mask_ratio must be in [0, 1], got {self.info_mask_ratio}."
            )
        if self.token_mask_score_source not in {"none", "grad_attn", "attn", "grad"}:
            raise ValueError(
                "token_mask_score_source must be one of "
                "['none', 'grad_attn', 'attn', 'grad'], "
                f"got {self.token_mask_score_source}."
            )
        if self.token_mask_min_keep < 0:
            raise ValueError(
                "token_mask_min_keep must be >= 0, "
                f"got {self.token_mask_min_keep}."
            )
        if self.lambda_guided_mtp < 0.0:
            raise ValueError(
                f"lambda_guided_mtp must be >= 0, got {self.lambda_guided_mtp}."
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
        self.proto_version = str(getattr(args, "proto_version", "v1")).lower()
        self.lambda_proto_v = float(getattr(args, "lambda_proto_v", 1.0))
        self.lambda_proto_t = float(getattr(args, "lambda_proto_t", 1.0))
        self.lambda_proto_self_v = getattr(args, "lambda_proto_self_v", None)
        self.lambda_proto_self_t = getattr(args, "lambda_proto_self_t", None)
        self.lambda_proto_cross_t2v = float(
            getattr(args, "lambda_proto_cross_t2v", 0.0)
        )
        self.lambda_proto_cross_v2t = float(
            getattr(args, "lambda_proto_cross_v2t", 0.0)
        )
        self.lambda_proto_align = float(getattr(args, "lambda_proto_align", 0.0))
        self.proto_temp = float(getattr(args, "proto_temp", 0.07))
        self.proto_momentum = float(getattr(args, "proto_momentum", 0.9))
        self.proto_warmup_epochs = int(getattr(args, "proto_warmup_epochs", 0))
        self.proto_update_source = str(
            getattr(args, "proto_update_source", "detached")
        ).lower()
        self.proto_init_file = str(getattr(args, "proto_init_file", ""))
        self.proto_init_strict = bool(getattr(args, "proto_init_strict", False))
        self.proto_init_load_visual = bool(
            getattr(args, "proto_init_load_visual", True)
        )
        self.proto_init_load_text = bool(getattr(args, "proto_init_load_text", True))
        self.proto_init_freeze_update_epochs = int(
            getattr(args, "proto_init_freeze_update_epochs", 0)
        )
        self.proto_init_loaded = False
        self.proto_init_visual_loaded = False
        self.proto_init_text_loaded = False
        self.proto_init_load_info = {
            "visual_loaded": False,
            "text_loaded": False,
            "visual_initialized_count": 0,
            "text_initialized_count": 0,
            "num_classes": self.num_classes,
            "feature_dim": self.embed_dim,
        }
        self.debug_proto_stats = bool(getattr(args, "debug_proto_stats", False))
        self.proto_runtime_epoch = 1
        if self.proto_version not in {"v1", "v2"}:
            raise ValueError(
                f"proto_version must be one of ['v1', 'v2'], got {self.proto_version}."
            )
        if self.proto_temp <= 0:
            raise ValueError(f"proto_temp must be > 0, got {self.proto_temp}.")
        if self.proto_warmup_epochs < 0:
            raise ValueError(
                "proto_warmup_epochs must be >= 0, "
                f"got {self.proto_warmup_epochs}."
            )
        if self.proto_update_source not in {"detached", "raw"}:
            raise ValueError(
                "proto_update_source must be one of ['detached', 'raw'], "
                f"got {self.proto_update_source}."
            )
        if self.proto_init_freeze_update_epochs < 0:
            raise ValueError(
                "proto_init_freeze_update_epochs must be >= 0, "
                f"got {self.proto_init_freeze_update_epochs}."
            )
        if self.proto_init_file and not (
            self.proto_init_load_visual or self.proto_init_load_text
        ):
            raise ValueError(
                "proto_init_file was provided, but both visual/text loading are disabled."
            )
        self.prototype_bank = None
        self.dual_prototype_bank = None
        if self.use_proto_calib:
            if self.proto_version == "v1":
                # Default integration uses zero init; the bank module also supports
                # random-normalized init for later experiments.
                self.prototype_bank = IdentityPrototypeBank(
                    num_classes=self.num_classes,
                    feat_dim=self.embed_dim,
                    momentum=self.proto_momentum,
                    init_mode="zero",
                    eps=self.uncertainty_eps,
                    update_source=self.proto_update_source,
                )
            else:
                self.dual_prototype_bank = DualIdentityPrototypeBank(
                    num_classes=self.num_classes,
                    feat_dim=self.embed_dim,
                    momentum=self.proto_momentum,
                    init_mode="zero",
                    eps=self.uncertainty_eps,
                    update_source=self.proto_update_source,
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
        self.use_attr_aux = bool(
            getattr(args, "use_attr_aux", False)
            and float(getattr(args, "lambda_attr_aux", 0.0)) > 0.0
        )
        self.attr_feature_source = str(
            getattr(args, "attr_feature_source", "fused_det")
        ).lower()
        if self.use_attr_aux and self.attr_feature_source not in {
            "fused_det",
            "fused_prob",
        }:
            raise ValueError(
                "attr_feature_source must be one of ['fused_det', 'fused_prob'], "
                f"got {self.attr_feature_source}."
            )
        self.use_uspc_slot_local = bool(
            getattr(args, "use_uspc_slot_local", False)
            and float(getattr(args, "lambda_uspc_slot_local", 0.0)) > 0.0
        )
        self.uspc_slot_patch_source = str(
            getattr(args, "uspc_slot_patch_source", "rgb")
        ).lower()
        self.uspc_slot_region_mode = str(
            getattr(args, "uspc_slot_region_mode", "fixed_stripe")
        ).lower()
        self.uspc_slot_loss_type = str(
            getattr(args, "uspc_slot_loss_type", "ce")
        ).lower()
        self.uspc_slot_detach_patch = bool(
            getattr(args, "uspc_slot_detach_patch", False)
        )
        self.debug_uspc_slot_local = bool(
            getattr(args, "debug_uspc_slot_local", False)
        )
        self.uspc_slot_attr_names = get_uspc_slot_local_attr_names()
        self.uspc_slot_region_map = get_uspc_slot_region_map()
        if self.use_uspc_slot_local:
            if self.uspc_slot_patch_source != "rgb":
                raise NotImplementedError(
                    "USPC-v2-B0 only guarantees --uspc_slot_patch_source rgb. "
                    f"Got {self.uspc_slot_patch_source!r}."
                )
            if self.uspc_slot_region_mode != "fixed_stripe":
                raise ValueError(
                    "uspc_slot_region_mode must be 'fixed_stripe', "
                    f"got {self.uspc_slot_region_mode!r}."
                )
            if self.uspc_slot_loss_type != "ce":
                raise ValueError(
                    "uspc_slot_loss_type must be 'ce', "
                    f"got {self.uspc_slot_loss_type!r}."
                )
            self.uspc_slot_grid_h, self.uspc_slot_grid_w = (
                self._configured_patch_grid(args)
            )

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

        if self.use_attr_aux:
            self.attr_head_gender = nn.Linear(self.embed_dim, 3)
            self.attr_head_upper_color = nn.Linear(self.embed_dim, 11)
            self.attr_head_lower_color = nn.Linear(self.embed_dim, 11)
            self.attr_head_shoe_color = nn.Linear(self.embed_dim, 11)
            self.attr_head_bag_color = nn.Linear(self.embed_dim, 11)
            self.attr_head_bag_present = nn.Linear(self.embed_dim, 2)
            self.attr_head_hair_length = nn.Linear(self.embed_dim, 2)

        if self.use_uspc_slot_local:
            attr_num_classes = get_basic_attribute_num_classes()
            self.uspc_slot_heads = nn.ModuleDict(
                {
                    attr_name: nn.Linear(
                        self.embed_dim,
                        int(attr_num_classes[attr_name]),
                    )
                    for attr_name in self.uspc_slot_attr_names
                }
            )

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

    def _proto_init_update_frozen(self):
        return (
            self.proto_init_loaded
            and self.proto_init_freeze_update_epochs > 0
            and int(self.proto_runtime_epoch) <= self.proto_init_freeze_update_epochs
        )

    def is_proto_update_frozen(self):
        return bool(self._proto_init_update_frozen())

    def _proto_calib_enabled(self):
        if not self.use_proto_calib:
            return False
        if self.proto_version == "v2":
            return self.dual_prototype_bank is not None
        return self.prototype_bank is not None

    def should_update_proto_bank(self):
        if not self._proto_calib_enabled():
            return False
        if self._proto_init_update_frozen():
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
        if not self._proto_calib_enabled():
            return False
        if not self.training:
            return False
        if self._proto_init_update_frozen():
            return False
        return True

    def _check_proto_init_metadata(self, init_data, strict, logger):
        meta = init_data.get("meta", {})
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            if strict:
                raise ValueError("proto init file 'meta' must be a dict in strict mode.")
            logger.warning("[PROTO_INIT_LOAD] meta is not a dict; tensor shape checks still apply")
            return

        meta_num_classes = meta.get("num_classes", None)
        meta_feature_dim = meta.get("feature_dim", meta.get("feat_dim", None))
        if strict and (meta_num_classes is None or meta_feature_dim is None):
            raise ValueError(
                "proto_init_strict=True requires meta.num_classes and meta.feature_dim."
            )
        if meta_num_classes is not None and int(meta_num_classes) != self.num_classes:
            message = (
                "proto init metadata num_classes mismatch: "
                f"file={int(meta_num_classes)}, model={self.num_classes}."
            )
            if strict:
                raise ValueError(message)
            logger.warning("[PROTO_INIT_LOAD] %s Tensor shape checks still apply.", message)
        if meta_feature_dim is not None and int(meta_feature_dim) != self.embed_dim:
            message = (
                "proto init metadata feature_dim mismatch: "
                f"file={int(meta_feature_dim)}, model={self.embed_dim}."
            )
            if strict:
                raise ValueError(message)
            logger.warning("[PROTO_INIT_LOAD] %s Tensor shape checks still apply.", message)

    @torch.no_grad()
    def load_proto_init_file(self, proto_init_file=None, logger=None):
        logger = logger or logging.getLogger("IRRA.train")
        proto_init_file = str(proto_init_file if proto_init_file is not None else self.proto_init_file)
        info = {
            "enabled": False,
            "file": proto_init_file,
            "proto_version": self.proto_version,
            "visual_loaded": False,
            "text_loaded": False,
            "visual_initialized_count": 0,
            "text_initialized_count": 0,
            "feature_dim": self.embed_dim,
            "num_classes": self.num_classes,
        }
        if proto_init_file == "":
            return info
        if not self.use_proto_calib:
            logger.info("[PROTO_INIT_LOAD] enabled=False")
            logger.info("[PROTO_INIT_LOAD] file=%s", proto_init_file)
            logger.info("[PROTO_INIT_LOAD] reason=use_proto_calib is False")
            return info
        if not self._proto_calib_enabled():
            raise RuntimeError(
                "proto_init_file was provided, but prototype calibration bank "
                "is not initialized."
            )
        if not op.isfile(proto_init_file):
            raise RuntimeError(f"proto_init_file='{proto_init_file}' does not exist.")

        logger.info(
            "[PROTO_INIT_LOAD] order=after checkpoint/resume load; "
            "proto_init_file has final prototype-bank priority"
        )
        init_data = torch.load(proto_init_file, map_location=torch.device("cpu"))
        if not isinstance(init_data, dict):
            raise ValueError(
                f"proto_init_file must contain a dict, got {type(init_data)}."
            )
        self._check_proto_init_metadata(
            init_data,
            strict=self.proto_init_strict,
            logger=logger,
        )

        if self.proto_version == "v2":
            load_info = self.dual_prototype_bank.load_offline_init(
                init_data,
                load_visual=self.proto_init_load_visual,
                load_text=self.proto_init_load_text,
            )
        else:
            load_info = self.prototype_bank.load_offline_init(
                init_data,
                load_visual=self.proto_init_load_visual,
                load_text=False,
            )

        self.proto_init_visual_loaded = bool(load_info.get("visual_loaded", False))
        self.proto_init_text_loaded = bool(load_info.get("text_loaded", False))
        self.proto_init_loaded = bool(
            self.proto_init_visual_loaded or self.proto_init_text_loaded
        )
        self.proto_init_load_info = dict(load_info)
        info.update(load_info)
        info["enabled"] = True

        logger.info("[PROTO_INIT_LOAD] enabled=True")
        logger.info("[PROTO_INIT_LOAD] file=%s", proto_init_file)
        logger.info("[PROTO_INIT_LOAD] proto_version=%s", self.proto_version)
        logger.info(
            "[PROTO_INIT_LOAD] visual_loaded=%s",
            bool(info["visual_loaded"]),
        )
        logger.info(
            "[PROTO_INIT_LOAD] text_loaded=%s",
            bool(info["text_loaded"]),
        )
        logger.info(
            "[PROTO_INIT_LOAD] visual_initialized_count=%d",
            int(info["visual_initialized_count"]),
        )
        logger.info(
            "[PROTO_INIT_LOAD] text_initialized_count=%d",
            int(info["text_initialized_count"]),
        )
        logger.info("[PROTO_INIT_LOAD] feature_dim=%d", int(info["feature_dim"]))
        logger.info("[PROTO_INIT_LOAD] num_classes=%d", int(info["num_classes"]))
        logger.info(
            "[PROTO_INIT_LOAD] freeze_update_epochs=%d",
            int(self.proto_init_freeze_update_epochs),
        )
        return info

    def get_visual_prototype_matrix(self):
        if self.proto_version == "v2":
            if self.dual_prototype_bank is None:
                return None
            return self.dual_prototype_bank.get_visual_normalized_prototypes()
        if self.prototype_bank is None:
            return None
        return self.prototype_bank.get_normalized_prototypes()

    def get_visual_prototype_initialized_mask(self):
        if self.proto_version == "v2":
            if self.dual_prototype_bank is None:
                return None
            return self.dual_prototype_bank.get_visual_initialized_mask()
        if self.prototype_bank is None:
            return None
        return self.prototype_bank.get_initialized_mask()

    def get_text_prototype_matrix(self):
        if self.proto_version != "v2" or self.dual_prototype_bank is None:
            return None
        return self.dual_prototype_bank.get_text_normalized_prototypes()

    def get_text_prototype_initialized_mask(self):
        if self.proto_version != "v2" or self.dual_prototype_bank is None:
            return None
        return self.dual_prototype_bank.get_text_initialized_mask()

    def update_visual_prototype_bank(self, fused_feats, pids):
        if not self._should_update_proto_bank():
            return {
                "updated": False,
                "num_updated_pids": 0,
                "num_newly_initialized_pids": 0,
                "num_initialized_pids": 0,
            }
        if self.proto_version == "v2":
            return self.dual_prototype_bank.update(
                visual_feats=fused_feats,
                text_feats=None,
                pids=pids,
            )
        return self.prototype_bank.update(fused_feats, pids)

    def update_dual_prototype_bank(self, visual_feats, text_feats, pids):
        if not self._should_update_proto_bank() or self.proto_version != "v2":
            return {
                "updated": False,
                "num_updated_pids": 0,
                "num_newly_initialized_pids": 0,
                "num_initialized_pids": 0,
                "num_updated_visual_pids": 0,
                "num_updated_text_pids": 0,
                "num_newly_initialized_visual_pids": 0,
                "num_newly_initialized_text_pids": 0,
                "num_visual_initialized_pids": 0,
                "num_text_initialized_pids": 0,
                "proto_batch_unique_pids": 0,
                "proto_v_mean_norm": 0.0,
                "proto_t_mean_norm": 0.0,
            }
        return self.dual_prototype_bank.update(
            visual_feats=visual_feats,
            text_feats=text_feats,
            pids=pids,
        )

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split("+")]
        print(f"Training Model with {self.current_task} tasks")

    def _get_caption_weight_for_loss(self, batch, apply_flag_name, default=True):
        if not bool(getattr(self.args, "use_synth_loss_weighting", False)):
            return None
        if not bool(getattr(self.args, apply_flag_name, default)):
            return None

        caption_weight = batch.get("caption_weight", None)
        if caption_weight is None:
            return None
        caption_weight = caption_weight.to(
            device=batch["caption_ids"].device,
            dtype=torch.float32,
        ).reshape(-1)
        if caption_weight.numel() == 0:
            return None
        if bool(torch.all(caption_weight.eq(1.0)).item()):
            return None
        return caption_weight

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

    def _append_mga_text_outputs(
        self,
        batch,
        ret,
        ids_key,
        valid_key,
        weight_key,
        feat_key,
        ret_valid_key,
        ret_weight_key,
    ):
        if ids_key not in batch:
            return

        _, _, text_feat, _ = self._encode_caption_ids_with_uncertainty(
            batch[ids_key]
        )
        ret[feat_key] = text_feat

        valid = batch.get(valid_key, None)
        if torch.is_tensor(valid):
            ret[ret_valid_key] = valid.to(device=text_feat.device).reshape(-1).bool()
        else:
            ret[ret_valid_key] = torch.ones(
                text_feat.shape[0],
                device=text_feat.device,
                dtype=torch.bool,
            )

        weight = batch.get(weight_key, None)
        if torch.is_tensor(weight):
            ret[ret_weight_key] = weight.to(
                device=text_feat.device,
                dtype=text_feat.dtype,
            ).reshape(-1)
        else:
            ret[ret_weight_key] = torch.ones(
                text_feat.shape[0],
                device=text_feat.device,
                dtype=text_feat.dtype,
            )

    def _append_mga_aux_text_outputs(self, batch, ret):
        if bool(getattr(self.args, "use_mga_pseudo", False)):
            self._append_mga_text_outputs(
                batch=batch,
                ret=ret,
                ids_key="mga_pseudo_ids",
                valid_key="mga_has_pseudo",
                weight_key="mga_pseudo_weight",
                feat_key="mga_pseudo_text_feat",
                ret_valid_key="mga_pseudo_valid",
                ret_weight_key="mga_pseudo_weight",
            )

        if bool(getattr(self.args, "use_mga_attr_align", False)):
            self._append_mga_text_outputs(
                batch=batch,
                ret=ret,
                ids_key="mga_rgb_color_ids",
                valid_key="mga_has_rgb_color",
                weight_key="mga_rgb_color_weight",
                feat_key="mga_rgb_color_text_feat",
                ret_valid_key="mga_rgb_color_valid",
                ret_weight_key="mga_rgb_color_weight",
            )
            self._append_mga_text_outputs(
                batch=batch,
                ret=ret,
                ids_key="mga_t_shape_ids",
                valid_key="mga_has_t_shape",
                weight_key="mga_t_shape_weight",
                feat_key="mga_t_shape_text_feat",
                ret_valid_key="mga_t_shape_valid",
                ret_weight_key="mga_t_shape_weight",
            )

        if bool(getattr(self.args, "use_mga_negative_loss", False)):
            self._append_mga_text_outputs(
                batch=batch,
                ret=ret,
                ids_key="mga_negative_ids",
                valid_key="mga_has_negative",
                weight_key="mga_negative_weight",
                feat_key="mga_negative_text_feat",
                ret_valid_key="mga_negative_valid",
                ret_weight_key="mga_negative_weight",
            )

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

    def _log_mraf_mem_callsite(self, phase, entry, tensor, thermal_available):
        if self.mraf_enabled and thermal_available and mraf_mem_debug_active(tensor.device):
            mraf_mem_debug_log(
                logging.getLogger("IRRA.train"),
                f"build.callsite.{phase}",
                tensor.device,
                entry=entry,
            )

    def _fuse_rgbt_global(
        self,
        rgb_mu,
        rgb_logvar,
        thermal_mu,
        thermal_logvar,
        thermal_available,
        return_mraf_aux=False,
        mem_debug_entry="global",
    ):
        """
        Fuse RGB/T thermal global features.

        MRAF, when enabled, replaces both deterministic addition and precision
        fusion. Argument validation rejects simultaneous MRAF and effective
        uncertainty precision fusion before model construction.

        If diagonal-Gaussian global uncertainty is enabled and use_precision_fusion=True:
          precision = 1 / (var + eps), var = exp(logvar)
          precision_fused = precision_rgb + precision_th
          mu_fused = (precision_rgb * mu_rgb + precision_th * mu_th) / precision_fused
          var_fused = 1 / precision_fused
          logvar_fused = log(var_fused + eps)

        Otherwise fallback to baseline-style deterministic add fusion.
        """
        mraf_aux = None
        if self.mraf_enabled and thermal_available:
            mem_debug = mraf_mem_debug_active(rgb_mu.device)
            if mem_debug:
                mraf_mem_debug_log(
                    logging.getLogger("IRRA.train"),
                    "build.mraf.before",
                    rgb_mu.device,
                    entry=mem_debug_entry,
                )
                fused_mu, mraf_aux = self.mraf_fusion(
                    rgb_mu,
                    thermal_mu,
                    return_aux=True,
                    mem_debug_entry=mem_debug_entry,
                )
                mraf_mem_debug_log(
                    logging.getLogger("IRRA.train"),
                    "build.mraf.after",
                    rgb_mu.device,
                    entry=mem_debug_entry,
                )
            else:
                fused_mu, mraf_aux = self.mraf_fusion(
                    rgb_mu,
                    thermal_mu,
                    return_aux=True,
                )
            if self._use_diag_gaussian_global():
                rgb_var = self._safe_var_from_logvar(rgb_logvar)
                thermal_var = self._safe_var_from_logvar(thermal_logvar)
                alpha_rgb = mraf_aux["alpha_rgb"].to(dtype=rgb_var.dtype)
                alpha_ir = mraf_aux["alpha_ir"].to(dtype=thermal_var.dtype)
                fused_var = 4.0 * (
                    alpha_rgb.square() * rgb_var
                    + alpha_ir.square() * thermal_var
                )
                fused_logvar = self._safe_logvar_from_var(fused_var)
            else:
                fused_logvar = torch.zeros_like(fused_mu)
            if return_mraf_aux:
                return fused_mu, fused_logvar, mraf_aux
            return fused_mu, fused_logvar

        if not self._use_diag_gaussian_global():
            if self.use_rgbt_add_fusion:
                fused_mu = rgb_mu + thermal_mu
            else:
                fused_mu = rgb_mu
            fused_logvar = torch.zeros_like(fused_mu)
            if return_mraf_aux:
                return fused_mu, fused_logvar, mraf_aux
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
            if return_mraf_aux:
                return fused_mu, fused_logvar, mraf_aux
            return fused_mu, fused_logvar

        # Fallback: keep baseline fused representation path.
        if self.use_rgbt_add_fusion and thermal_available:
            fused_mu = rgb_mu + thermal_mu
            fused_var = rgb_var + thermal_var
        else:
            fused_mu = rgb_mu
            fused_var = rgb_var
        fused_logvar = self._safe_logvar_from_var(fused_var)
        if return_mraf_aux:
            return fused_mu, fused_logvar, mraf_aux
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

    def _noise_token_masking_active(self):
        # A: train-only text input perturbation. Default False keeps baseline exact.
        return self.training and self.use_noise_token_masking

    def _informative_token_masking_active(self):
        # B: train-only guided MLM/MTP target selection. Eval/test never enter it.
        return self.training and self.use_informative_token_masking

    def _guided_text_masking_requested(self):
        # A and B are independent. With both off, this is False and no score/mask
        # tensors are built, so the baseline forward path stays unchanged.
        return (
            self._noise_token_masking_active()
            or self._informative_token_masking_active()
        )

    def _guided_score_needed(self):
        if self.token_mask_score_source == "none":
            return False
        noise_needs_score = (
            self._noise_token_masking_active()
            and not self.ablate_random_noise_mask
        )
        info_needs_score = (
            self._informative_token_masking_active()
            and not self.ablate_random_info_mask
        )
        return noise_needs_score or info_needs_score

    def _guided_mask_mode(self, enabled, ablate_random, token_scores, prefix):
        if not enabled:
            return "disabled"
        if ablate_random:
            return "random_ablate"
        if token_scores is None:
            return "random_fallback"
        return f"guided_{self.token_mask_score_source}_{prefix}"

    def _compute_guided_token_scores(self, caption_ids, attention_mask):
        if not self._guided_score_needed():
            return None

        with torch.no_grad():
            score_tokens = self.base_model.encode_text(caption_ids)
            batch_index = torch.arange(score_tokens.shape[0], device=score_tokens.device)
            text_global = score_tokens[batch_index, caption_ids.argmax(dim=-1)]
            token_scores = compute_token_importance_scores(
                input_ids=caption_ids,
                token_embeddings=score_tokens,
                global_embedding=text_global,
                attention_mask=attention_mask,
                special_token_ids=self.guided_mask_special_token_ids,
                score_source=self.token_mask_score_source,
                detach_score=self.guided_mask_detach_score,
            )
        return token_scores

    def _build_guided_mask_aux(self, caption_ids):
        if not self._guided_text_masking_requested():
            return None

        # Shared lightweight score source for A/B. A samples low-score tokens;
        # B samples high-score tokens. Random ablations skip score computation.
        attention_mask = caption_ids.ne(0)
        candidate_mask = attention_mask.clone()
        for token_id in self.guided_mask_special_token_ids:
            candidate_mask = candidate_mask & caption_ids.ne(int(token_id))
        token_scores = self._compute_guided_token_scores(
            caption_ids=caption_ids,
            attention_mask=attention_mask,
        )
        noise_mask_mode = self._guided_mask_mode(
            enabled=self._noise_token_masking_active(),
            ablate_random=self.ablate_random_noise_mask,
            token_scores=token_scores,
            prefix="noise",
        )
        info_mask_mode = self._guided_mask_mode(
            enabled=self._informative_token_masking_active(),
            ablate_random=self.ablate_random_info_mask,
            token_scores=token_scores,
            prefix="info",
        )
        noise_mask = build_noise_token_mask(
            input_ids=caption_ids,
            attention_mask=attention_mask,
            token_scores=token_scores,
            enabled=self.use_noise_token_masking,
            is_training=self.training,
            mask_ratio=self.noise_mask_ratio,
            min_keep=self.token_mask_min_keep,
            special_token_ids=self.guided_mask_special_token_ids,
            ablate_random=self.ablate_random_noise_mask,
        )
        info_mask = build_informative_token_mask(
            input_ids=caption_ids,
            attention_mask=attention_mask,
            token_scores=token_scores,
            enabled=self.use_informative_token_masking,
            is_training=self.training,
            mask_ratio=self.info_mask_ratio,
            min_keep=self.token_mask_min_keep,
            special_token_ids=self.guided_mask_special_token_ids,
            ablate_random=self.ablate_random_info_mask,
        )

        aux = {
            "token_scores": token_scores,
            "noise_mask": noise_mask,
            "info_mask": info_mask,
            "guided_mask_stats": None,
            "noise_mask_mode": noise_mask_mode,
            "info_mask_mode": info_mask_mode,
        }
        if self.debug_print_token_mask_stats:
            aux["guided_mask_stats"] = summarize_mask_stats(
                token_scores=token_scores,
                noise_mask=noise_mask,
                info_mask=info_mask,
                candidate_mask=candidate_mask,
                enabled_noise=self.use_noise_token_masking,
                enabled_info=self.use_informative_token_masking,
                noise_mask_mode=noise_mask_mode,
                info_mask_mode=info_mask_mode,
            )
        return aux

    def _apply_noise_mask_to_caption_ids(self, caption_ids, guided_mask_aux=None):
        # A only changes the main training text encoder input.
        if guided_mask_aux is None:
            return caption_ids
        return apply_guided_mask_to_input_ids(
            caption_ids,
            token_mask=guided_mask_aux.get("noise_mask"),
            mask_token_id=self.guided_mask_token_id,
            enabled=self._noise_token_masking_active(),
        )

    def _apply_guided_masks_to_mlm_ids(self, mlm_ids, guided_mask_aux=None):
        if guided_mask_aux is None:
            return mlm_ids

        masked_ids = mlm_ids
        if self.use_noise_token_masking and self.noise_mask_apply_to_mlm:
            masked_ids = apply_guided_mask_to_input_ids(
                masked_ids,
                token_mask=guided_mask_aux.get("noise_mask"),
                mask_token_id=self.guided_mask_token_id,
                enabled=self._noise_token_masking_active(),
            )
        return masked_ids

    def _guided_informative_mlm_active(self):
        return self._informative_token_masking_active() and self.info_mask_apply_to_mlm

    def _build_guided_informative_mlm(self, caption_ids, guided_mask_aux=None):
        # B only changes MLM/MTP prediction targets and reuses the existing MLM head.
        if not self._guided_informative_mlm_active():
            return None, None
        if guided_mask_aux is None:
            return None, None
        return build_guided_mlm_inputs_and_labels(
            input_ids=caption_ids,
            informative_mask=guided_mask_aux.get("info_mask"),
            mask_token_id=self.guided_mask_token_id,
            ignore_index=0,
            enabled=True,
        )

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
        self._log_mraf_mem_callsite(
            "before", "eval", rgb_mu, thermal_image is not None
        )
        f_c, _ = self._fuse_rgbt_global(
            rgb_mu=rgb_mu,
            rgb_logvar=rgb_logvar,
            thermal_mu=thermal_mu,
            thermal_logvar=thermal_logvar,
            thermal_available=thermal_image is not None,
            mem_debug_entry="eval",
        )
        self._log_mraf_mem_callsite(
            "after", "eval", rgb_mu, thermal_image is not None
        )
        if self.mraf_enabled:
            return f_c
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
        if self.mraf_enabled:
            self._log_mraf_mem_callsite(
                "before", "eval", rgb_mu, thermal_image is not None
            )
            fused_feat, _, mraf_aux = self._fuse_rgbt_global(
                rgb_mu=rgb_mu,
                rgb_logvar=rgb_logvar,
                thermal_mu=thermal_mu,
                thermal_logvar=thermal_logvar,
                thermal_available=thermal_image is not None,
                return_mraf_aux=True,
                mem_debug_entry="eval",
            )
            self._log_mraf_mem_callsite(
                "after", "eval", rgb_mu, thermal_image is not None
            )
        else:
            fused_feat, _ = self._fuse_rgbt_global(
                rgb_mu=rgb_mu,
                rgb_logvar=rgb_logvar,
                thermal_mu=thermal_mu,
                thermal_logvar=thermal_logvar,
                thermal_available=thermal_image is not None,
            )
            mraf_aux = None

        if self.mraf_enabled:
            global_feat = fused_feat
        elif self.eval_use_mu_only:
            global_feat = rgb_mu
        elif self.eval_use_fused_mu:
            global_feat = fused_feat
        elif self.use_rgbt_add_fusion:
            global_feat = rgb_mu + thermal_mu
        else:
            global_feat = rgb_mu

        outputs = {
            "global_feat": self._maybe_eval_sampling(global_feat),
            "rgb_global_feat": self._maybe_eval_sampling(rgb_mu),
            "thermal_global_feat": self._maybe_eval_sampling(thermal_mu),
            "rgb_patch_tokens": rgb_patch_tokens,
            "thermal_patch_tokens": thermal_patch_tokens,
            "rgb_precision_weight": rgb_precision_weight,
            "thermal_precision_weight": thermal_precision_weight,
            "precision_weight_valid": precision_weight_valid,
        }
        if mraf_aux is not None:
            outputs.update(
                {
                    "mraf_alpha": mraf_aux["alpha"],
                    "mraf_alpha_rgb": mraf_aux["alpha_rgb"],
                    "mraf_alpha_ir": mraf_aux["alpha_ir"],
                    "mraf_logits": mraf_aux["logits"],
                    "raw_mraf_logits": mraf_aux["raw_logits"],
                    "scaled_mraf_logits": mraf_aux["scaled_logits"],
                }
            )
        return outputs

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

    def encode_rgbt_global(
        self, rgb_image, thermal_image=None, return_patch_tokens=False
    ):
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
            patch_outputs: optional dict when return_patch_tokens=True
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
            thermal_patch_tokens = v_rgb_c.new_zeros(
                v_rgb_c.shape[0],
                0,
                v_rgb_c.shape[-1],
            )

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
        if return_patch_tokens:
            patch_outputs = {
                "rgb_patch_feats": rgb_patch_tokens,
                "thermal_patch_feats": thermal_patch_tokens,
            }
            return (
                v_rgb_c,
                v_t_c,
                f_c,
                rgb_tokens,
                local_stats,
                moe_adapter_aux,
                patch_outputs,
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

    def _forward_attr_heads(self, attr_feat):
        def _head_logits(head):
            return head(attr_feat.to(head.weight.dtype)).float()

        return {
            "attr_logits_gender": _head_logits(self.attr_head_gender),
            "attr_logits_upper_color": _head_logits(self.attr_head_upper_color),
            "attr_logits_lower_color": _head_logits(self.attr_head_lower_color),
            "attr_logits_shoe_color": _head_logits(self.attr_head_shoe_color),
            "attr_logits_bag_color": _head_logits(self.attr_head_bag_color),
            "attr_logits_bag_present": _head_logits(self.attr_head_bag_present),
            "attr_logits_hair_length": _head_logits(self.attr_head_hair_length),
        }

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

    def _configured_patch_grid(self, args):
        img_h, img_w = self._as_hw_pair(getattr(args, "img_size", None), (384, 128))
        stride_h, stride_w = self._as_hw_pair(
            getattr(args, "stride_size", None),
            (16, 16),
        )
        stride_h = max(1, int(stride_h))
        stride_w = max(1, int(stride_w))
        return max(1, img_h // stride_h), max(1, img_w // stride_w)

    def _runtime_patch_grid(self, patch_count):
        patch_count = int(patch_count)
        cfg_h = max(1, int(getattr(self, "uspc_slot_grid_h", 1)))
        cfg_w = max(1, int(getattr(self, "uspc_slot_grid_w", 1)))
        if cfg_h * cfg_w == patch_count:
            return cfg_h, cfg_w
        if cfg_w > 0 and patch_count % cfg_w == 0:
            return max(1, patch_count // cfg_w), cfg_w
        if cfg_h > 0 and patch_count % cfg_h == 0:
            return cfg_h, max(1, patch_count // cfg_h)
        return patch_count, 1

    def _row_bounds(self, grid_h, start_frac, end_frac):
        start = int(round(float(start_frac) * float(grid_h)))
        end = int(round(float(end_frac) * float(grid_h)))
        start = max(0, min(int(grid_h), start))
        end = max(0, min(int(grid_h), end))
        if end <= start:
            end = min(int(grid_h), start + 1)
        return start, end

    def _mean_region(self, patch_grid, row_start, row_end, col_mask=None):
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

    def _pool_uspc_fixed_regions(self, patch_tokens):
        if patch_tokens is None:
            raise KeyError("USPC slot-local heads require rgb patch tokens.")
        if patch_tokens.dim() != 3:
            raise ValueError(
                "USPC slot-local patch tokens must have shape [B, N, C], "
                f"got {tuple(patch_tokens.shape)}."
            )
        if patch_tokens.shape[1] <= 0:
            raise ValueError("USPC slot-local patch tokens are empty.")

        patch_tokens = patch_tokens.float()
        grid_h, grid_w = self._runtime_patch_grid(patch_tokens.shape[1])
        patch_count = grid_h * grid_w
        patch_grid = patch_tokens[:, :patch_count, :].reshape(
            patch_tokens.shape[0],
            grid_h,
            grid_w,
            patch_tokens.shape[-1],
        )

        head_rows = self._row_bounds(grid_h, 0.00, 0.20)
        upper_rows = self._row_bounds(grid_h, 0.20, 0.55)
        lower_rows = self._row_bounds(grid_h, 0.55, 0.85)
        shoes_rows = self._row_bounds(grid_h, 0.85, 1.00)
        bag_rows = self._row_bounds(grid_h, 0.25, 0.75)
        edge_cols = max(1, grid_w // 4)
        bag_col_mask = torch.zeros(
            grid_w,
            dtype=torch.bool,
            device=patch_tokens.device,
        )
        bag_col_mask[:edge_cols] = True
        bag_col_mask[-edge_cols:] = True

        return {
            "head": self._mean_region(patch_grid, *head_rows),
            "upper": self._mean_region(patch_grid, *upper_rows),
            "lower": self._mean_region(patch_grid, *lower_rows),
            "shoes": self._mean_region(patch_grid, *shoes_rows),
            "bag": self._mean_region(patch_grid, *bag_rows, col_mask=bag_col_mask),
        }

    def _forward_uspc_slot_local_heads(self, patch_tokens):
        if not self.use_uspc_slot_local:
            return {}
        region_feats = self._pool_uspc_fixed_regions(patch_tokens)
        head_region_feats = (
            {key: value.detach() for key, value in region_feats.items()}
            if self.uspc_slot_detach_patch
            else region_feats
        )
        logits = {}
        for attr_name in self.uspc_slot_attr_names:
            region_name = self.uspc_slot_region_map[attr_name]
            head = self.uspc_slot_heads[attr_name]
            region_feat = head_region_feats[region_name]
            logits[attr_name] = head(region_feat.to(head.weight.dtype)).float()
        return {
            "logits": logits,
            "region_feats": region_feats,
        }

    def forward(self, batch, mdrc_feature_only=False):
        ret = {}

        # Input tensors:
        # images: [B, 3, H, W]
        # thermal_images: [B, 3, H, W] (optional for compatibility)
        # caption_ids: [B, L_t]
        images = batch["images"]
        thermal_images = batch.get("thermal_images", None)
        if mdrc_feature_only:
            # MDRC degraded branch: image-only feature extraction for
            # clean-degraded consistency. No text encoding and no baseline losses.
            (
                v_rgb_c_raw,
                v_t_c_raw,
                _,
                _,
                img_local_stats,
                _,
            ) = self.encode_rgbt_global(images, thermal_images)
            rgb_mu, rgb_logvar, thermal_mu, thermal_logvar = self._encode_visual_global_gaussian(
                v_rgb_c_raw,
                v_t_c_raw,
                thermal_available=thermal_images is not None,
            )
            if self.mraf_enabled:
                self._log_mraf_mem_callsite(
                    "before", "degraded_feature_only", rgb_mu,
                    thermal_images is not None,
                )
                fused_mu, fused_logvar, mraf_aux = self._fuse_rgbt_global(
                    rgb_mu=rgb_mu,
                    rgb_logvar=rgb_logvar,
                    thermal_mu=thermal_mu,
                    thermal_logvar=thermal_logvar,
                    thermal_available=thermal_images is not None,
                    return_mraf_aux=True,
                    mem_debug_entry="degraded_feature_only",
                )
                self._log_mraf_mem_callsite(
                    "after", "degraded_feature_only", rgb_mu,
                    thermal_images is not None,
                )
            else:
                fused_mu, fused_logvar = self._fuse_rgbt_global(
                    rgb_mu=rgb_mu,
                    rgb_logvar=rgb_logvar,
                    thermal_mu=thermal_mu,
                    thermal_logvar=thermal_logvar,
                    thermal_available=thermal_images is not None,
                )
                mraf_aux = None
            ret.update(
                {
                    "v_rgb_c": rgb_mu,
                    "v_t_c": thermal_mu,
                    "fused_mu": fused_mu,
                    "fused_logvar": fused_logvar,
                    "fused_det_feat": fused_mu,
                    "fused_prob_feat": fused_mu,
                    "mdrc_feature_only": True,
                }
            )
            if img_local_stats is not None:
                ret.update(img_local_stats)
            if mraf_aux is not None:
                ret.update(
                    {
                        "mraf_alpha": mraf_aux["alpha"],
                        "mraf_alpha_rgb": mraf_aux["alpha_rgb"],
                        "mraf_alpha_ir": mraf_aux["alpha_ir"],
                        "mraf_logits": mraf_aux["logits"],
                        "raw_mraf_logits": mraf_aux["raw_logits"],
                        "scaled_mraf_logits": mraf_aux["scaled_logits"],
                    }
                )
            if self._use_diag_gaussian_global():
                ret.update(
                    {
                        "rgb_mu": rgb_mu,
                        "rgb_logvar": rgb_logvar,
                        "thermal_mu": thermal_mu,
                        "thermal_logvar": thermal_logvar,
                    }
                )
            return ret

        caption_ids = batch["caption_ids"]
        caption_weight_for_sdm = self._get_caption_weight_for_loss(
            batch,
            "synth_weight_apply_to_sdm",
            default=True,
        )
        caption_weight_for_mlm = self._get_caption_weight_for_loss(
            batch,
            "synth_weight_apply_to_mlm",
            default=True,
        )
        caption_ids_full_clean = None
        caption_ids_rgb = None
        caption_ids_th = None
        if self.use_modality_text_prompt:
            if "caption_ids_full_clean" not in batch:
                raise KeyError(
                    "use_modality_text_prompt=True requires batch['caption_ids_full_clean'] "
                    "for clean full-text encoding."
                )
            caption_ids_full_clean = batch["caption_ids_full_clean"]
            caption_ids_rgb = batch.get("caption_ids_rgb", caption_ids)
            caption_ids_th = batch.get("caption_ids_th", caption_ids)

        # 1) RGB/Thermal pooled global branch (before uncertainty heads).
        local_patch_outputs = None
        if self.use_uspc_slot_local:
            (
                v_rgb_c_raw,
                v_t_c_raw,
                _,
                rgb_tokens,
                img_local_stats,
                rgbt_moe_adapter_aux,
                local_patch_outputs,
            ) = self.encode_rgbt_global(
                images,
                thermal_images,
                return_patch_tokens=True,
            )
        else:
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
        # A/B boundary:
        # - A masks low-score tokens in this main text input during training.
        # - B does not touch this input; it only builds guided MLM/MTP targets below.
        # - A+B can run together using separate masks; both off keeps caption_ids.
        guided_mask_aux = self._build_guided_mask_aux(caption_ids)
        caption_ids_for_text = self._apply_noise_mask_to_caption_ids(
            caption_ids,
            guided_mask_aux=guided_mask_aux,
        )
        text_tokens, l_c_raw, text_mu, text_logvar = (
            self._encode_caption_ids_with_uncertainty(caption_ids_for_text)
        )
        text_feat_full = None
        text_feat_rgb = None
        text_feat_th = None
        if self.use_modality_text_prompt:
            _, _, text_feat_full, _ = self._encode_caption_ids_with_uncertainty(
                caption_ids_full_clean
            )
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
        if self.mraf_enabled:
            self._log_mraf_mem_callsite(
                "before", "clean", rgb_mu, thermal_images is not None
            )
            fused_mu, fused_logvar, mraf_aux = self._fuse_rgbt_global(
                rgb_mu=rgb_mu,
                rgb_logvar=rgb_logvar,
                thermal_mu=thermal_mu,
                thermal_logvar=thermal_logvar,
                thermal_available=thermal_images is not None,
                return_mraf_aux=True,
                mem_debug_entry="clean",
            )
            self._log_mraf_mem_callsite(
                "after", "clean", rgb_mu, thermal_images is not None
            )
        else:
            fused_mu, fused_logvar = self._fuse_rgbt_global(
                rgb_mu=rgb_mu,
                rgb_logvar=rgb_logvar,
                thermal_mu=thermal_mu,
                thermal_logvar=thermal_logvar,
                thermal_available=thermal_images is not None,
            )
            mraf_aux = None

        # Keep existing training/eval flow compatible: deterministic embedding is mu.
        v_rgb_c = rgb_mu
        v_t_c = thermal_mu
        l_c = text_mu
        f_c = fused_mu

        hn_fused_det_feat = None
        hn_images = batch.get("hn_rgb_images", None)
        if hn_images is not None:
            if hn_images.dim() == 5:
                hn_batch_size, hn_num_neg = hn_images.shape[:2]
                hn_images_flat = hn_images.reshape(-1, *hn_images.shape[2:])
            elif hn_images.dim() == 4:
                hn_batch_size = hn_images.shape[0]
                hn_num_neg = 1
                hn_images_flat = hn_images
            else:
                raise ValueError(
                    "hn_rgb_images must have shape [B, N, C, H, W] or "
                    f"[B, C, H, W], got {tuple(hn_images.shape)}."
                )

            hn_t_images = batch.get("hn_t_images", None)
            if thermal_images is None:
                hn_t_images = None
            if hn_t_images is not None:
                if hn_t_images.dim() == 5:
                    hn_t_images_flat = hn_t_images.reshape(
                        -1, *hn_t_images.shape[2:]
                    )
                elif hn_t_images.dim() == 4:
                    hn_t_images_flat = hn_t_images
                else:
                    raise ValueError(
                        "hn_t_images must have shape [B, N, C, H, W] or "
                        f"[B, C, H, W], got {tuple(hn_t_images.shape)}."
                    )
            else:
                hn_t_images_flat = None

            hn_v_rgb_raw, hn_v_t_raw, _, _, _, _ = self.encode_rgbt_global(
                hn_images_flat, hn_t_images_flat
            )
            (
                hn_rgb_mu,
                hn_rgb_logvar,
                hn_thermal_mu,
                hn_thermal_logvar,
            ) = self._encode_visual_global_gaussian(
                hn_v_rgb_raw,
                hn_v_t_raw,
                thermal_available=hn_t_images_flat is not None,
            )
            self._log_mraf_mem_callsite(
                "before", "hardneg", hn_rgb_mu, hn_t_images_flat is not None
            )
            hn_fused_mu, _ = self._fuse_rgbt_global(
                rgb_mu=hn_rgb_mu,
                rgb_logvar=hn_rgb_logvar,
                thermal_mu=hn_thermal_mu,
                thermal_logvar=hn_thermal_logvar,
                thermal_available=hn_t_images_flat is not None,
                mem_debug_entry="hardneg",
            )
            self._log_mraf_mem_callsite(
                "after", "hardneg", hn_rgb_mu, hn_t_images_flat is not None
            )
            hn_fused_det_feat = hn_fused_mu.reshape(
                hn_batch_size, hn_num_neg, -1
            )

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
        if guided_mask_aux is not None:
            ret["guided_mask_aux"] = guided_mask_aux
            guided_mask_stats = guided_mask_aux.get("guided_mask_stats", None)
            if isinstance(guided_mask_stats, dict):
                ret["noise_mask_avg_tokens"] = f_c.new_tensor(
                    guided_mask_stats.get("avg_noise_mask_tokens", 0.0)
                )
                ret["noise_mask_valid_ratio"] = f_c.new_tensor(
                    guided_mask_stats.get("noise_mask_valid_ratio", 0.0)
                )
                ret["info_mask_avg_tokens"] = f_c.new_tensor(
                    guided_mask_stats.get("avg_info_mask_tokens", 0.0)
                )
                ret["info_mask_valid_ratio"] = f_c.new_tensor(
                    guided_mask_stats.get("info_mask_valid_ratio", 0.0)
                )
                ret["noise_info_overlap_ratio"] = f_c.new_tensor(
                    guided_mask_stats.get("noise_info_overlap_ratio", 0.0)
                )
                ret["token_score_mean"] = f_c.new_tensor(
                    guided_mask_stats.get("token_score_valid_mean", 0.0)
                )
                ret["token_score_var"] = f_c.new_tensor(
                    guided_mask_stats.get("token_score_var", 0.0)
                )
                ret["noise_mask_mode"] = guided_mask_stats.get(
                    "noise_mask_mode", "none"
                )
                ret["info_mask_mode"] = guided_mask_stats.get(
                    "info_mask_mode", "none"
                )
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
        if mraf_aux is not None:
            ret.update(
                {
                    "mraf_alpha": mraf_aux["alpha"],
                    "mraf_alpha_rgb": mraf_aux["alpha_rgb"],
                    "mraf_alpha_ir": mraf_aux["alpha_ir"],
                    "mraf_logits": mraf_aux["logits"],
                    "raw_mraf_logits": mraf_aux["raw_logits"],
                    "scaled_mraf_logits": mraf_aux["scaled_logits"],
                }
            )
        if hn_fused_det_feat is not None:
            ret["hn_fused_det_feat"] = hn_fused_det_feat
        self._append_mga_aux_text_outputs(batch, ret)
        if self.use_attr_aux:
            attr_feat = p_f if self.attr_feature_source == "fused_prob" else f_det
            ret.update(self._forward_attr_heads(attr_feat))
        if self.use_uspc_slot_local:
            if local_patch_outputs is None:
                raise RuntimeError(
                    "USPC slot-local is enabled but patch outputs were not collected."
                )
            rgb_patch_feats = local_patch_outputs.get("rgb_patch_feats", None)
            ret["rgb_patch_feats"] = rgb_patch_feats
            if self.uspc_slot_patch_source in {"thermal", "concat_rgbt"}:
                ret["thermal_patch_feats"] = local_patch_outputs.get(
                    "thermal_patch_feats",
                    None,
                )
            slot_outputs = self._forward_uspc_slot_local_heads(rgb_patch_feats)
            ret["uspc_slot_logits"] = slot_outputs["logits"]
            if self.debug_uspc_slot_local:
                ret["uspc_slot_region_feats"] = slot_outputs["region_feats"]
        if self.use_modality_text_prompt:
            ret.update(
                {
                    "text_feat_full": text_feat_full,
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
                p_f,
                p_t,
                batch["pids"],
                logit_scale,
                sample_weight=caption_weight_for_sdm,
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
                            f_det,
                            t_det,
                            batch["pids"],
                            logit_scale,
                            sample_weight=caption_weight_for_sdm,
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
        # B on: replace random MLM targets with guided informative targets.
        # B off: use the dataloader's original random mlm_ids/mlm_labels unchanged.
        if "mlm" in self.current_task:
            guided_mlm_active = False
            mlm_ids, mlm_labels = self._build_guided_informative_mlm(
                caption_ids,
                guided_mask_aux=guided_mask_aux,
            )
            if mlm_ids is not None and mlm_labels is not None:
                guided_mlm_active = True
            else:
                mlm_ids = self._apply_guided_masks_to_mlm_ids(
                    batch["mlm_ids"],
                    guided_mask_aux=guided_mask_aux,
                )
                mlm_labels = batch["mlm_labels"]

            mlm_feats = self.base_model.encode_text(mlm_ids)  # [B, L_t, C]
            x = self.cross_former(mlm_feats, rgb_tokens, rgb_tokens)  # [B, L_t, C]
            x = self.mlm_head(x)  # [B, L_t, vocab_size]

            scores = x.float().reshape(-1, self.args.vocab_size)  # [B*L_t, V]
            mlm_labels = mlm_labels.reshape(-1)  # [B*L_t]
            mlm_label_mask = mlm_labels.ne(0)
            if bool(mlm_label_mask.any()):
                raw_mlm_loss = objectives.compute_mlm(
                    scores,
                    mlm_labels,
                    sample_weight=caption_weight_for_mlm,
                    batch_size=x.shape[0],
                )
            else:
                raw_mlm_loss = scores.new_zeros(())
            guided_mtp_scale = (
                self.lambda_guided_mtp if guided_mlm_active else 1.0
            )
            ret.update(
                {
                    # Legacy weighted key kept for backward compatibility.
                    "mlm_loss": (
                        raw_mlm_loss
                        * self.args.mlm_loss_weight
                        * guided_mtp_scale
                    )
                }
            )
            # Requested probabilistic-training key (unweighted).
            ret.update({"loss_mlm": raw_mlm_loss * guided_mtp_scale})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_label_mask, as_tuple=False).flatten()
            if mlm_label_idx.numel() > 0:
                acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            else:
                acc = scores.new_zeros(())
            ret.update({"mlm_acc": acc})
            if guided_mlm_active:
                guided_valid_tokens = mlm_label_mask.sum()
                ret["guided_mtp_valid_tokens"] = guided_valid_tokens.float()
                ret["guided_mtp_lambda"] = scores.new_tensor(guided_mtp_scale)

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    convert_weights(model)
    if model.mraf_enabled and model.mraf_fusion.gate_fp32:
        # convert_weights recursively halves every Linear. Restore only the
        # optional MRAF scorer; its _apply override also protects later
        # model.half()/to(dtype=...) calls.
        model.mraf_fusion.scorer.float()
    return model
