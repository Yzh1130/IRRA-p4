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

        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.sync_distributed = bool(sync_distributed)
        self.init_mode = str(init_mode).lower()

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

        fused_feats = fused_feats.detach().float()[valid_mask]
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
