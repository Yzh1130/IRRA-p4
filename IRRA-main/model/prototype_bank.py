import torch
import torch.distributed as dist
import torch.nn as nn


class IdentityPrototypeBank(nn.Module):
    """
    One visual prototype per training identity.

    The bank is updated from detached fused visual global features only.
    Stored prototypes are returned in normalized form for downstream use.
    """

    def __init__(
        self,
        num_classes,
        feat_dim,
        momentum=0.9,
        init_mode="zero",
        eps=1e-12,
        sync_distributed=True,
        update_source="detached",
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}.")
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be > 0, got {feat_dim}.")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError(
                f"momentum must be in [0, 1), got {momentum}."
            )
        update_source = str(update_source).lower()
        if update_source not in {"detached", "raw"}:
            raise ValueError(
                "update_source must be one of ['detached', 'raw'], "
                f"got {update_source}."
            )

        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.sync_distributed = bool(sync_distributed)
        self.init_mode = str(init_mode).lower()
        self.update_source = update_source

        init_bank = self._build_init_bank(self.init_mode)
        self.register_buffer("prototype_memory", init_bank)
        self.register_buffer(
            "initialized", self._build_initialized_mask(self.init_mode)
        )

    def _safe_l2_normalize(self, x, dim=1):
        norm = x.norm(p=2, dim=dim, keepdim=True).clamp_min(self.eps)
        return x / norm

    def _build_initialized_mask(self, init_mode):
        init_mode = str(init_mode).lower()
        if init_mode == "zero":
            return torch.zeros(self.num_classes, dtype=torch.bool)
        if init_mode == "random":
            return torch.ones(self.num_classes, dtype=torch.bool)
        raise ValueError(
            f"Unsupported prototype init_mode={init_mode}. "
            "Expected 'zero' or 'random'."
        )

    def _build_init_bank(self, init_mode):
        init_mode = str(init_mode).lower()
        if init_mode == "zero":
            return torch.zeros(self.num_classes, self.feat_dim, dtype=torch.float32)
        if init_mode == "random":
            init_bank = torch.randn(
                self.num_classes, self.feat_dim, dtype=torch.float32
            )
            return self._safe_l2_normalize(init_bank, dim=1)
        raise ValueError(
            f"Unsupported prototype init_mode={init_mode}. "
            "Expected 'zero' or 'random'."
        )

    @torch.no_grad()
    def reset_parameters(self, init_mode=None):
        if init_mode is None:
            init_mode = self.init_mode
        init_mode = str(init_mode).lower()
        init_bank = self._build_init_bank(init_mode).to(
            device=self.prototype_memory.device,
            dtype=self.prototype_memory.dtype,
        )
        self.prototype_memory.copy_(init_bank)
        init_mask = self._build_initialized_mask(init_mode).to(
            device=self.initialized.device
        )
        self.initialized.copy_(init_mask)
        self.init_mode = init_mode

    @torch.no_grad()
    def update(self, fused_feats, pids):
        update_info = {
            "updated": False,
            "num_updated_pids": 0,
            "num_newly_initialized_pids": 0,
            "num_initialized_pids": int(self.initialized.sum().item()),
        }
        if fused_feats is None or pids is None:
            return update_info
        if fused_feats.ndim != 2:
            raise ValueError(
                "fused_feats must have shape [B, C], "
                f"got shape {tuple(fused_feats.shape)}."
            )

        pids = pids.reshape(-1).long()
        if pids.numel() == 0:
            return update_info

        valid_mask = (pids >= 0) & (pids < self.num_classes)
        if not valid_mask.any():
            return update_info

        if self.update_source == "detached":
            fused_feats = fused_feats.detach()
        fused_feats = fused_feats.float()[valid_mask]
        pids = pids[valid_mask]

        batch_sums = fused_feats.new_zeros(self.num_classes, self.feat_dim)
        batch_counts = fused_feats.new_zeros(self.num_classes)
        batch_sums.index_add_(0, pids, fused_feats)
        batch_counts.index_add_(
            0, pids, torch.ones_like(pids, dtype=batch_counts.dtype)
        )

        if self.sync_distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)

        updated_pids = torch.nonzero(batch_counts > 0, as_tuple=False).squeeze(1)
        if updated_pids.numel() == 0:
            return update_info

        batch_means = batch_sums[updated_pids] / batch_counts[updated_pids].unsqueeze(
            1
        ).clamp_min(1.0)
        batch_means = self._safe_l2_normalize(batch_means, dim=1)

        momentum = self.momentum
        existing_mask = self.initialized[updated_pids]
        new_mask = ~existing_mask

        if new_mask.any():
            new_pids = updated_pids[new_mask]
            self.prototype_memory[new_pids] = batch_means[new_mask]
            self.initialized[new_pids] = True
            update_info["num_newly_initialized_pids"] = int(new_pids.numel())

        if existing_mask.any():
            existing_pids = updated_pids[existing_mask]
            existing_proto = self.prototype_memory[existing_pids]
            existing_means = batch_means[existing_mask]
            self.prototype_memory[existing_pids] = (
                momentum * existing_proto + (1.0 - momentum) * existing_means
            )

        update_info["updated"] = True
        update_info["num_updated_pids"] = int(updated_pids.numel())
        update_info["num_initialized_pids"] = int(self.initialized.sum().item())
        return update_info

    def get_normalized_prototypes(self):
        return self._safe_l2_normalize(self.prototype_memory.float(), dim=1)

    def get_initialized_mask(self):
        return self.initialized

    def forward(self):
        return self.get_normalized_prototypes()


class DualIdentityPrototypeBank(nn.Module):
    """
    One visual prototype and one text prototype per training identity.

    The visual bank is updated from fused visual global features, and the text
    bank is updated from text global features. Stored prototypes are normalized
    after every EMA update and returned in normalized form for downstream use.
    """

    def __init__(
        self,
        num_classes,
        feat_dim,
        momentum=0.9,
        init_mode="zero",
        eps=1e-12,
        sync_distributed=True,
        update_source="detached",
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}.")
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be > 0, got {feat_dim}.")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError(
                f"momentum must be in [0, 1), got {momentum}."
            )
        update_source = str(update_source).lower()
        if update_source not in {"detached", "raw"}:
            raise ValueError(
                "update_source must be one of ['detached', 'raw'], "
                f"got {update_source}."
            )

        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.sync_distributed = bool(sync_distributed)
        self.init_mode = str(init_mode).lower()
        self.update_source = update_source

        self.register_buffer(
            "visual_prototypes", self._build_init_bank(self.init_mode)
        )
        self.register_buffer(
            "text_prototypes", self._build_init_bank(self.init_mode)
        )
        self.register_buffer(
            "visual_initialized", self._build_initialized_mask(self.init_mode)
        )
        self.register_buffer(
            "text_initialized", self._build_initialized_mask(self.init_mode)
        )

    def _safe_l2_normalize(self, x, dim=1):
        norm = x.norm(p=2, dim=dim, keepdim=True).clamp_min(self.eps)
        return x / norm

    def _build_initialized_mask(self, init_mode):
        init_mode = str(init_mode).lower()
        if init_mode == "zero":
            return torch.zeros(self.num_classes, dtype=torch.bool)
        if init_mode == "random":
            return torch.ones(self.num_classes, dtype=torch.bool)
        raise ValueError(
            f"Unsupported prototype init_mode={init_mode}. "
            "Expected 'zero' or 'random'."
        )

    def _build_init_bank(self, init_mode):
        init_mode = str(init_mode).lower()
        if init_mode == "zero":
            return torch.zeros(self.num_classes, self.feat_dim, dtype=torch.float32)
        if init_mode == "random":
            init_bank = torch.randn(
                self.num_classes, self.feat_dim, dtype=torch.float32
            )
            return self._safe_l2_normalize(init_bank, dim=1)
        raise ValueError(
            f"Unsupported prototype init_mode={init_mode}. "
            "Expected 'zero' or 'random'."
        )

    @torch.no_grad()
    def reset_parameters(self, init_mode=None):
        if init_mode is None:
            init_mode = self.init_mode
        init_mode = str(init_mode).lower()
        visual_bank = self._build_init_bank(init_mode).to(
            device=self.visual_prototypes.device,
            dtype=self.visual_prototypes.dtype,
        )
        text_bank = self._build_init_bank(init_mode).to(
            device=self.text_prototypes.device,
            dtype=self.text_prototypes.dtype,
        )
        init_mask = self._build_initialized_mask(init_mode)
        self.visual_prototypes.copy_(visual_bank)
        self.text_prototypes.copy_(text_bank)
        self.visual_initialized.copy_(
            init_mask.to(device=self.visual_initialized.device)
        )
        self.text_initialized.copy_(
            init_mask.to(device=self.text_initialized.device)
        )
        self.init_mode = init_mode

    def _empty_update_info(self, prefix):
        return {
            f"{prefix}_updated": False,
            f"num_updated_{prefix}_pids": 0,
            f"num_newly_initialized_{prefix}_pids": 0,
            f"num_{prefix}_initialized_pids": 0,
            f"{prefix}_mean_norm": 0.0,
        }

    @torch.no_grad()
    def _update_one_bank(self, feats, pids, prototypes, initialized, prefix):
        update_info = self._empty_update_info(prefix)
        update_info[f"num_{prefix}_initialized_pids"] = int(
            initialized.sum().item()
        )
        if feats is None or pids is None:
            return update_info
        if feats.ndim != 2:
            raise ValueError(
                f"{prefix}_feats must have shape [B, C], "
                f"got shape {tuple(feats.shape)}."
            )

        pids = pids.reshape(-1).long()
        if pids.numel() == 0:
            return update_info
        if pids.shape[0] != feats.shape[0]:
            raise ValueError(
                f"pids must have shape [B] matching {prefix}_feats, "
                f"got {pids.shape[0]} and {feats.shape[0]}."
            )

        valid_mask = (pids >= 0) & (pids < self.num_classes)
        if not valid_mask.any():
            return update_info

        if self.update_source == "detached":
            feats = feats.detach()
        feats = feats.float()[valid_mask]
        pids = pids[valid_mask]

        batch_sums = feats.new_zeros(self.num_classes, self.feat_dim)
        batch_counts = feats.new_zeros(self.num_classes)
        batch_sums.index_add_(0, pids, feats)
        batch_counts.index_add_(
            0, pids, torch.ones_like(pids, dtype=batch_counts.dtype)
        )

        if self.sync_distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)

        updated_pids = torch.nonzero(batch_counts > 0, as_tuple=False).squeeze(1)
        if updated_pids.numel() == 0:
            return update_info

        batch_means = batch_sums[updated_pids] / batch_counts[updated_pids].unsqueeze(
            1
        ).clamp_min(1.0)
        batch_means = self._safe_l2_normalize(batch_means, dim=1)

        momentum = self.momentum
        existing_mask = initialized[updated_pids]
        new_mask = ~existing_mask

        if new_mask.any():
            new_pids = updated_pids[new_mask]
            prototypes[new_pids] = batch_means[new_mask]
            initialized[new_pids] = True
            update_info[f"num_newly_initialized_{prefix}_pids"] = int(
                new_pids.numel()
            )

        if existing_mask.any():
            existing_pids = updated_pids[existing_mask]
            existing_proto = prototypes[existing_pids]
            existing_means = batch_means[existing_mask]
            prototypes[existing_pids] = self._safe_l2_normalize(
                momentum * existing_proto + (1.0 - momentum) * existing_means,
                dim=1,
            )

        update_info[f"{prefix}_updated"] = True
        update_info[f"num_updated_{prefix}_pids"] = int(updated_pids.numel())
        update_info[f"num_{prefix}_initialized_pids"] = int(initialized.sum().item())
        update_info[f"{prefix}_mean_norm"] = float(
            batch_means.norm(p=2, dim=1).mean().item()
        )
        return update_info

    @torch.no_grad()
    def update(self, visual_feats, text_feats, pids):
        info = {
            "updated": False,
            "num_updated_pids": 0,
            "num_newly_initialized_pids": 0,
            "num_initialized_pids": 0,
            "proto_batch_unique_pids": 0,
            "proto_v_mean_norm": 0.0,
            "proto_t_mean_norm": 0.0,
        }
        if pids is not None:
            flat_pids = pids.reshape(-1).long()
            valid_mask = (flat_pids >= 0) & (flat_pids < self.num_classes)
            if valid_mask.any():
                info["proto_batch_unique_pids"] = int(
                    torch.unique(flat_pids[valid_mask]).numel()
                )

        visual_info = self._update_one_bank(
            visual_feats,
            pids,
            self.visual_prototypes,
            self.visual_initialized,
            "visual",
        )
        text_info = self._update_one_bank(
            text_feats,
            pids,
            self.text_prototypes,
            self.text_initialized,
            "text",
        )

        info.update(visual_info)
        info.update(text_info)
        info["updated"] = bool(
            visual_info.get("visual_updated", False)
            or text_info.get("text_updated", False)
        )
        info["num_updated_pids"] = max(
            int(visual_info.get("num_updated_visual_pids", 0)),
            int(text_info.get("num_updated_text_pids", 0)),
        )
        info["num_newly_initialized_pids"] = max(
            int(visual_info.get("num_newly_initialized_visual_pids", 0)),
            int(text_info.get("num_newly_initialized_text_pids", 0)),
        )
        info["num_initialized_pids"] = min(
            int(self.visual_initialized.sum().item()),
            int(self.text_initialized.sum().item()),
        )
        info["proto_v_mean_norm"] = float(
            visual_info.get("visual_mean_norm", 0.0)
        )
        info["proto_t_mean_norm"] = float(text_info.get("text_mean_norm", 0.0))
        return info

    def get_visual_normalized_prototypes(self):
        return self._safe_l2_normalize(self.visual_prototypes.float(), dim=1)

    def get_text_normalized_prototypes(self):
        return self._safe_l2_normalize(self.text_prototypes.float(), dim=1)

    def get_visual_initialized_mask(self):
        return self.visual_initialized

    def get_text_initialized_mask(self):
        return self.text_initialized

    def forward(self):
        return {
            "visual_prototypes": self.get_visual_normalized_prototypes(),
            "text_prototypes": self.get_text_normalized_prototypes(),
            "visual_initialized": self.get_visual_initialized_mask(),
            "text_initialized": self.get_text_initialized_mask(),
        }
