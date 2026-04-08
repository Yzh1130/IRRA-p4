import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class BottleneckAdapter(nn.Module):
    """
    Lightweight residual adapter over the last feature dimension.

    Structure:
        down_proj -> activation -> dropout -> up_proj

    Args:
        embed_dim: feature dimension C of the incoming tokens/features.
        bottleneck_dim: bottleneck hidden size used by the adapter.
        dropout: dropout probability applied between activation and up projection.

    Input:
        x: [B, C] or [B, L, C]

    Output:
        adapted_delta: same shape as x

    Initialization:
        The up projection is zero-initialized so the adapter starts from an
        identity-like state once inserted into a residual path.
    """

    def __init__(self, embed_dim, bottleneck_dim, dropout=0.1):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}.")
        if bottleneck_dim <= 0:
            raise ValueError(
                f"bottleneck_dim must be > 0, got {bottleneck_dim}."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.embed_dim = int(embed_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.dropout_prob = float(dropout)

        self.down_proj = nn.Linear(self.embed_dim, self.bottleneck_dim)
        self.activation = QuickGELU()
        self.dropout = nn.Dropout(self.dropout_prob)
        self.up_proj = nn.Linear(self.bottleneck_dim, self.embed_dim)
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.constant_(self.down_proj.bias, 0.0)
        nn.init.constant_(self.up_proj.weight, 0.0)
        nn.init.constant_(self.up_proj.bias, 0.0)

    def forward(self, x):
        if x.dim() not in (2, 3):
            raise ValueError(
                f"x must have shape [B, C] or [B, L, C], got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                "The last dim of x must match embed_dim, got "
                f"{x.shape[-1]} vs {self.embed_dim}."
            )

        hidden = self.down_proj(x.to(self.down_proj.weight.dtype))
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        delta = self.up_proj(hidden)
        return delta.to(x.dtype)


class ModalityAwareRouter(nn.Module):
    """
    Dense masked router for the visual RGBT MoE adapter.

    Routing space is fixed to three experts in this order:
        [shared, rgb, thermal]

    Masked routing:
        - RGB samples can only use [shared, rgb]
        - Thermal samples can only use [shared, thermal]

    Args:
        embed_dim: pooled token feature dimension C.
        router_temp: softmax temperature used after expert masking.

    Inputs:
        pooled_feat: [B, C]
        modality: explicit modality specifier. Supported forms:
            - single string: "rgb" or "thermal" for the whole batch
            - list/tuple of strings of length B
            - LongTensor [B] with ids: rgb->0, thermal->1

    Outputs:
        router_weights: [B, 3], float32, aligned with [shared, rgb, thermal]
    """

    MODALITY_TO_ID = {"rgb": 0, "thermal": 1}
    EXPERT_ORDER = ("shared", "rgb", "thermal")

    def __init__(self, embed_dim, router_temp=1.0):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}.")
        if router_temp <= 0:
            raise ValueError(
                f"router_temp must be > 0, got {router_temp}."
            )

        self.embed_dim = int(embed_dim)
        self.router_temp = float(router_temp)
        self.modality_embedding = nn.Embedding(2, self.embed_dim)
        self.router_proj = nn.Linear(self.embed_dim * 2, len(self.EXPERT_ORDER))
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.modality_embedding.weight, std=0.02)
        nn.init.normal_(self.router_proj.weight, std=1e-3)
        nn.init.constant_(self.router_proj.bias, 0.0)

    def _modality_to_tensor(self, modality, batch_size, device):
        if isinstance(modality, str):
            key = modality.lower()
            if key not in self.MODALITY_TO_ID:
                raise ValueError(
                    f"Unsupported modality={modality}. "
                    "Expected 'rgb' or 'thermal'."
                )
            return torch.full(
                (batch_size,),
                self.MODALITY_TO_ID[key],
                device=device,
                dtype=torch.long,
            )

        if isinstance(modality, (list, tuple)):
            if len(modality) != batch_size:
                raise ValueError(
                    "When modality is a list/tuple, its length must equal batch "
                    f"size. Got {len(modality)} vs {batch_size}."
                )
            modality_ids = []
            for item in modality:
                if not isinstance(item, str):
                    raise ValueError(
                        "List/tuple modality items must be strings 'rgb' or "
                        f"'thermal', got {type(item)}."
                    )
                key = item.lower()
                if key not in self.MODALITY_TO_ID:
                    raise ValueError(
                        f"Unsupported modality item={item}. "
                        "Expected 'rgb' or 'thermal'."
                    )
                modality_ids.append(self.MODALITY_TO_ID[key])
            return torch.tensor(modality_ids, device=device, dtype=torch.long)

        if torch.is_tensor(modality):
            if modality.dim() == 0:
                modality = modality.reshape(1)
            if modality.dim() != 1:
                raise ValueError(
                    "Tensor modality must have shape [B], got "
                    f"{tuple(modality.shape)}."
                )
            if modality.shape[0] != batch_size:
                raise ValueError(
                    "Tensor modality batch size mismatch: "
                    f"{modality.shape[0]} vs {batch_size}."
                )
            modality = modality.to(device=device, dtype=torch.long)
            valid_mask = (modality == 0) | (modality == 1)
            if not bool(valid_mask.all()):
                raise ValueError(
                    "Tensor modality must use ids rgb->0, thermal->1 only."
                )
            return modality

        raise TypeError(
            "modality must be a string, a list/tuple of strings, or a LongTensor."
        )

    def _build_expert_mask(self, modality_ids):
        batch_size = modality_ids.shape[0]
        mask = torch.zeros(
            batch_size,
            len(self.EXPERT_ORDER),
            device=modality_ids.device,
            dtype=torch.bool,
        )
        is_rgb = modality_ids == self.MODALITY_TO_ID["rgb"]
        is_thermal = modality_ids == self.MODALITY_TO_ID["thermal"]

        # shared expert is always allowed
        mask[:, 0] = True
        mask[is_rgb, 1] = True
        mask[is_thermal, 2] = True
        return mask

    def forward(self, pooled_feat, modality):
        if pooled_feat.dim() != 2:
            raise ValueError(
                "pooled_feat must have shape [B, C], got "
                f"{tuple(pooled_feat.shape)}."
            )
        if pooled_feat.shape[-1] != self.embed_dim:
            raise ValueError(
                "The last dim of pooled_feat must match embed_dim, got "
                f"{pooled_feat.shape[-1]} vs {self.embed_dim}."
            )

        batch_size = pooled_feat.shape[0]
        modality_ids = self._modality_to_tensor(
            modality, batch_size=batch_size, device=pooled_feat.device
        )
        modality_feat = self.modality_embedding(modality_ids)

        router_input = torch.cat(
            [
                pooled_feat.to(self.router_proj.weight.dtype),
                modality_feat.to(self.router_proj.weight.dtype),
            ],
            dim=-1,
        )
        logits = self.router_proj(router_input).float()
        expert_mask = self._build_expert_mask(modality_ids)
        masked_logits = logits.masked_fill(~expert_mask, -1e4)
        router_weights = torch.softmax(
            masked_logits / self.router_temp, dim=-1
        )
        return router_weights


class RGBTMoEAdapter(nn.Module):
    """
    Visual RGBT MoE adapter with one shared expert and two modality-specific experts.

    Experts:
        - shared_adapter
        - rgb_adapter
        - thermal_adapter

    Inputs:
        x: [B, C] or [B, L, C]
        modality: explicit modality specifier. Supported forms:
            - "rgb" or "thermal"
            - list/tuple[str] with length B
            - LongTensor [B], where rgb->0 and thermal->1

    Outputs:
        adapted_x: same shape as x
        router_weights: [B, 3], float32, ordered as [shared, rgb, thermal]
        shared_branch_output: same shape as x, raw output of shared_adapter(x)

    Modality behavior:
        RGB:
            x + w_shared * shared_adapter(x) + w_rgb * rgb_adapter(x)
        Thermal:
            x + w_shared * shared_adapter(x) + w_t * thermal_adapter(x)
    """

    def __init__(self, embed_dim, args):
        super().__init__()
        bottleneck_dim = int(getattr(args, "moe_adapter_dim", 64))
        dropout = float(getattr(args, "moe_adapter_dropout", 0.1))
        router_temp = float(getattr(args, "moe_router_temp", 1.0))

        self.embed_dim = int(embed_dim)
        self.shared_adapter = BottleneckAdapter(
            embed_dim=self.embed_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
        self.rgb_adapter = BottleneckAdapter(
            embed_dim=self.embed_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
        self.thermal_adapter = BottleneckAdapter(
            embed_dim=self.embed_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
        self.router = ModalityAwareRouter(
            embed_dim=self.embed_dim,
            router_temp=router_temp,
        )

    def _pool_router_input(self, x):
        if x.dim() == 2:
            pooled = x
        elif x.dim() == 3:
            # By convention, token 0 is the pooled/CLS token in the current CLIP
            # vision branch. If later a different pooled feature is available,
            # the caller can pass [B, C] directly instead.
            pooled = x[:, 0, :]
        else:
            raise ValueError(
                f"x must have shape [B, C] or [B, L, C], got {tuple(x.shape)}."
            )

        if pooled.shape[-1] != self.embed_dim:
            raise ValueError(
                "The last dim of x must match embed_dim, got "
                f"{pooled.shape[-1]} vs {self.embed_dim}."
            )
        return pooled

    def _expand_weights(self, router_weights, target):
        view_shape = [router_weights.shape[0]] + [1] * (target.dim() - 1)
        return router_weights.to(target.dtype).view(*view_shape, -1)

    def forward(self, x, modality):
        pooled_feat = self._pool_router_input(x)
        router_weights = self.router(pooled_feat, modality)

        shared_branch_output = self.shared_adapter(x)
        rgb_branch_output = self.rgb_adapter(x)
        thermal_branch_output = self.thermal_adapter(x)

        expanded_weights = self._expand_weights(router_weights, shared_branch_output)
        shared_weight = expanded_weights[..., 0]
        rgb_weight = expanded_weights[..., 1]
        thermal_weight = expanded_weights[..., 2]

        adapted_x = (
            x
            + shared_weight * shared_branch_output
            + rgb_weight * rgb_branch_output
            + thermal_weight * thermal_branch_output
        )
        return adapted_x, router_weights, shared_branch_output
