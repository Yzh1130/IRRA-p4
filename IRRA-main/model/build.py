from collections import OrderedDict

import torch
import torch.nn as nn

from model import objectives
from .clip_model import (
    LayerNorm,
    QuickGELU,
    Transformer,
    build_CLIP_from_openai_pretrained,
    convert_weights,
)
from .uncertainty import GlobalUncertaintyModeler


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        # CLIP backbone (shared by RGB and Thermal).
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice, args.img_size, args.stride_size
        )
        self.embed_dim = base_cfg["embed_dim"]

        # Temperature scalar used by global alignment losses.
        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        # Ablation switches.
        self.use_rgbt_add_fusion = getattr(args, "use_rgbt_add_fusion", True)
        self.use_mum = getattr(
            args, "use_mum", getattr(args, "use_global_uncertainty", False)
        )
        self.use_batch_uncertainty = getattr(
            args, "use_batch_uncertainty", getattr(args, "mum_use_batch_var", True)
        )
        self.use_id_uncertainty = getattr(
            args, "use_id_uncertainty", getattr(args, "mum_use_id_var", True)
        )
        self.use_prob_sampling = getattr(
            args, "use_prob_sampling", getattr(args, "mum_sample_enabled", False)
        )
        self.use_global_uncertainty = self.use_mum
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
            eps=getattr(args, "mum_eps", 1e-6),
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

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split("+")]
        print(f"Training Model with {self.current_task} tasks")

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q), self.ln_pre_i(k), self.ln_pre_i(v), need_weights=False
        )[0]
        x = x.permute(1, 0, 2)  # [B, L, C] -> [L, B, C]
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # [L, B, C] -> [B, L, C]
        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        # image: [B, 3, H, W] -> image_tokens: [B, L_v, C]
        image_tokens = self.base_model.encode_image(image)
        # global cls token: [B, C]
        return image_tokens[:, 0, :].float()

    def encode_text(self, text):
        # text: [B, L_t] -> text_tokens: [B, L_t, C]
        text_tokens = self.base_model.encode_text(text)
        batch_index = torch.arange(text_tokens.shape[0], device=text_tokens.device)
        # EOT global token: [B, C]
        return text_tokens[batch_index, text.argmax(dim=-1)].float()

    def encode_text_deterministic(self, text):
        # Deterministic global text feature used by retrieval evaluation.
        return self.encode_text(text)

    def encode_rgbt_deterministic(self, rgb_image, thermal_image=None):
        # Deterministic fused visual feature used by retrieval evaluation.
        _, _, f_c, _ = self.encode_rgbt_global(rgb_image, thermal_image)
        return f_c

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

    def encode_text_eval(self, text):
        feats = self.encode_text_deterministic(text)
        return self._maybe_eval_sampling(feats)

    def encode_rgbt_eval(self, rgb_image, thermal_image=None):
        feats = self.encode_rgbt_deterministic(rgb_image, thermal_image)
        return self._maybe_eval_sampling(feats)

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
        """
        rgb_tokens = self.base_model.encode_image(rgb_image)  # [B, L_v, C]
        v_rgb_c = rgb_tokens[:, 0, :].float()  # [B, C]

        if thermal_image is not None:
            thermal_tokens = self.base_model.encode_image(thermal_image)  # [B, L_v, C]
            v_t_c = thermal_tokens[:, 0, :].float()  # [B, C]
        else:
            # Backward compatibility for old single-image datasets.
            v_t_c = torch.zeros_like(v_rgb_c)  # [B, C]

        if self.use_rgbt_add_fusion:
            f_c = v_rgb_c + v_t_c  # [B, C]
        else:
            # Baseline ablation: RGB-only global feature.
            f_c = v_rgb_c
        return v_rgb_c, v_t_c, f_c, rgb_tokens

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

        # 1) RGB/Thermal global branch.
        v_rgb_c, v_t_c, f_c, rgb_tokens = self.encode_rgbt_global(
            images, thermal_images
        )

        # 2) Text global branch.
        text_tokens = self.base_model.encode_text(caption_ids)  # [B, L_t, C]
        batch_index = torch.arange(text_tokens.shape[0], device=text_tokens.device)
        l_c = text_tokens[batch_index, caption_ids.argmax(dim=-1)].float()  # [B, C]

        # 3) MUM-style uncertainty over (f_c, l_c).
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

        # Required outputs for downstream losses/logging.
        ret.update(uncertainty_outputs)
        ret.update(
            {
                "v_rgb_c": v_rgb_c,  # [B, C]
                "v_t_c": v_t_c,      # [B, C]
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

        if "cmpm" in self.current_task:
            ret.update({"cmpm_loss": objectives.compute_cmpm(p_f, p_t, batch["pids"])})

        if "id" in self.current_task:
            # Keep ID branch deterministic.
            image_logits = self.classifier(f_c.half()).float()
            text_logits = self.classifier(l_c.half()).float()
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
