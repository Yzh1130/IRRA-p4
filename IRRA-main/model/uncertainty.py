import torch
import torch.nn as nn


class IdentityVarianceMemoryBank(nn.Module):
    """
    Store identity-level running mean/variance for fused image and text features.
    """

    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.9, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.eps = eps

        # [num_classes, feat_dim]
        self.register_buffer("mean_f", torch.zeros(num_classes, feat_dim))
        self.register_buffer("var_f", torch.ones(num_classes, feat_dim) * eps)
        self.register_buffer("mean_t", torch.zeros(num_classes, feat_dim))
        self.register_buffer("var_t", torch.ones(num_classes, feat_dim) * eps)
        # [num_classes], indicates whether one identity has been initialized
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, fused_feats: torch.Tensor, text_feats: torch.Tensor, pids: torch.Tensor):
        """
        Args:
            fused_feats: [B, C]
            text_feats: [B, C]
            pids: [B]
        """
        if pids is None or pids.numel() == 0:
            return

        pids = pids.long()
        unique_ids = torch.unique(pids)
        m = self.momentum

        for pid in unique_ids:
            mask = pids == pid
            if mask.sum() == 0:
                continue

            cur_f = fused_feats[mask].detach()
            cur_t = text_feats[mask].detach()

            cur_mean_f = cur_f.mean(dim=0)
            cur_var_f = cur_f.var(dim=0, unbiased=False)
            cur_mean_t = cur_t.mean(dim=0)
            cur_var_t = cur_t.var(dim=0, unbiased=False)

            idx = int(pid.item())
            if not bool(self.initialized[idx]):
                self.mean_f[idx] = cur_mean_f
                self.var_f[idx] = cur_var_f.clamp_min(self.eps)
                self.mean_t[idx] = cur_mean_t
                self.var_t[idx] = cur_var_t.clamp_min(self.eps)
                self.initialized[idx] = True
                continue

            old_mean_f = self.mean_f[idx]
            old_var_f = self.var_f[idx]
            old_mean_t = self.mean_t[idx]
            old_var_t = self.var_t[idx]

            # EMA with between-mean correction term.
            new_mean_f = m * old_mean_f + (1.0 - m) * cur_mean_f
            new_var_f = (
                m * old_var_f
                + (1.0 - m) * cur_var_f
                + m * (1.0 - m) * (old_mean_f - cur_mean_f).pow(2)
            )
            new_mean_t = m * old_mean_t + (1.0 - m) * cur_mean_t
            new_var_t = (
                m * old_var_t
                + (1.0 - m) * cur_var_t
                + m * (1.0 - m) * (old_mean_t - cur_mean_t).pow(2)
            )

            self.mean_f[idx] = new_mean_f
            self.var_f[idx] = new_var_f.clamp_min(self.eps)
            self.mean_t[idx] = new_mean_t
            self.var_t[idx] = new_var_t.clamp_min(self.eps)

    def gather_identity_variance(
        self,
        pids: torch.Tensor,
        fallback_var_f: torch.Tensor,
        fallback_var_t: torch.Tensor,
    ):
        """
        Args:
            pids: [B]
            fallback_var_f: [B, C], fallback when identity is unseen
            fallback_var_t: [B, C], fallback when identity is unseen
        Returns:
            var_id_f: [B, C]
            var_id_t: [B, C]
        """
        if pids is None:
            return fallback_var_f, fallback_var_t

        pids = pids.long()
        var_id_f = self.var_f[pids]
        var_id_t = self.var_t[pids]
        seen = self.initialized[pids].unsqueeze(1)  # [B, 1]

        var_id_f = torch.where(seen, var_id_f, fallback_var_f)
        var_id_t = torch.where(seen, var_id_t, fallback_var_t)
        return var_id_f, var_id_t


class GlobalUncertaintyModeler(nn.Module):
    """
    MUM-style uncertainty for global fused-image/text features.
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        omega: float = 0.5,
        scale: float = 1.0,
        use_batch_var: bool = True,
        use_id_var: bool = True,
        sample_enabled: bool = True,
        sample_on_eval: bool = False,
        memory_momentum: float = 0.9,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.omega = omega
        self.scale = scale
        self.use_batch_var = use_batch_var
        self.use_id_var = use_id_var
        self.sample_enabled = sample_enabled
        self.sample_on_eval = sample_on_eval
        self.eps = eps

        self.memory_bank = IdentityVarianceMemoryBank(
            num_classes=num_classes,
            feat_dim=feat_dim,
            momentum=memory_momentum,
            eps=eps,
        )

    def _batch_variance(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, C] -> batch var [1, C] -> [B, C]
        var = feats.var(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        return var.expand(feats.shape[0], -1)

    def _unify_sigma(self, sigma_batch: torch.Tensor, sigma_id: torch.Tensor) -> torch.Tensor:
        if self.use_batch_var and self.use_id_var:
            sigma_unify = self.scale * (self.omega * sigma_batch + (1.0 - self.omega) * sigma_id)
        elif self.use_batch_var:
            sigma_unify = self.scale * sigma_batch
        elif self.use_id_var:
            sigma_unify = self.scale * sigma_id
        else:
            sigma_unify = torch.full_like(sigma_batch, self.eps)
        return sigma_unify.clamp_min(self.eps)

    def _sample(self, feats: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Default eval is deterministic; sampling in eval requires explicit opt-in.
        if self.sample_enabled and (self.training or self.sample_on_eval):
            eps = torch.randn_like(feats)
            return feats + eps * sigma
        return feats

    def forward(self, fused_feats: torch.Tensor, text_feats: torch.Tensor, pids: torch.Tensor = None):
        """
        Args:
            fused_feats: [B, C]
            text_feats: [B, C]
            pids: [B]
        Returns:
            dict with deterministic features, sampled features, and uncertainty terms.
        """
        # 1) Batch-level variance [B, C]
        var_batch_f = self._batch_variance(fused_feats)
        var_batch_t = self._batch_variance(text_feats)

        # 2) Identity-level variance from memory bank [B, C]
        var_id_f, var_id_t = self.memory_bank.gather_identity_variance(
            pids=pids,
            fallback_var_f=var_batch_f,
            fallback_var_t=var_batch_t,
        )

        # Use variance-derived std as sigma for reparameterization.
        sigma_batch_f = torch.sqrt(var_batch_f + self.eps)
        sigma_batch_t = torch.sqrt(var_batch_t + self.eps)
        sigma_id_f = torch.sqrt(var_id_f + self.eps)
        sigma_id_t = torch.sqrt(var_id_t + self.eps)

        # 3) Unified uncertainty [B, C]
        sigma_unify_f = self._unify_sigma(sigma_batch_f, sigma_id_f)
        sigma_unify_t = self._unify_sigma(sigma_batch_t, sigma_id_t)

        # 4) Probabilistic sampling [B, C]
        sampled_f = self._sample(fused_feats, sigma_unify_f)
        sampled_t = self._sample(text_feats, sigma_unify_t)

        if self.training:
            self.memory_bank.update(
                fused_feats=fused_feats,
                text_feats=text_feats,
                pids=pids,
            )

        return {
            # Deterministic globals
            "fused_feat_det": fused_feats,      # [B, C]
            "text_feat_det": text_feats,        # [B, C]
            # Sampled globals
            "fused_feat_sampled": sampled_f,    # [B, C]
            "text_feat_sampled": sampled_t,     # [B, C]
            # Uncertainty terms
            "sigma_batch_f": sigma_batch_f,     # [B, C]
            "sigma_batch_t": sigma_batch_t,     # [B, C]
            "sigma_id_f": sigma_id_f,           # [B, C]
            "sigma_id_t": sigma_id_t,           # [B, C]
            "sigma_unify_f": sigma_unify_f,     # [B, C]
            "sigma_unify_t": sigma_unify_t,     # [B, C]
        }
