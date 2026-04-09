import logging
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import objectives
from utils.comm import get_rank, synchronize
from utils.meter import AverageMeter
from utils.metrics import Evaluator


def _to_scalar(value):
    # Convert tensor/number into python float for stable meter/log writing.
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().mean().item()
    return float(value)


def _safe_l2_normalize(feats, dim=1, eps=1e-12):
    norm = feats.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return feats / norm


def _compute_batch_mean_shared_consistency_layer_loss(rgb_feat, thermal_feat):
    rgb_feat = _safe_l2_normalize(rgb_feat.float(), dim=1)
    thermal_feat = _safe_l2_normalize(thermal_feat.float(), dim=1)
    cosine_sim = F.cosine_similarity(rgb_feat, thermal_feat, dim=1)
    return 1.0 - cosine_sim.mean()


def _compute_id_pair_shared_consistency_layer_loss(rgb_feat, thermal_feat, pids):
    zero = rgb_feat.new_zeros(())
    if not torch.is_tensor(pids):
        return zero

    pids = pids.reshape(-1)
    if pids.shape[0] != rgb_feat.shape[0] or pids.shape[0] != thermal_feat.shape[0]:
        return zero

    rgb_feat = _safe_l2_normalize(rgb_feat.float(), dim=1)
    thermal_feat = _safe_l2_normalize(thermal_feat.float(), dim=1)
    rgb_pids = pids.to(device=rgb_feat.device)
    thermal_pids = pids.to(device=thermal_feat.device)

    pid_terms = []
    for pid in torch.unique(rgb_pids):
        rgb_mask = rgb_pids == pid
        thermal_mask = thermal_pids == pid
        if not bool(rgb_mask.any()) or not bool(thermal_mask.any()):
            continue

        rgb_pid_feat = _safe_l2_normalize(
            rgb_feat[rgb_mask].mean(dim=0, keepdim=True), dim=1
        )
        thermal_pid_feat = _safe_l2_normalize(
            thermal_feat[thermal_mask].mean(dim=0, keepdim=True), dim=1
        )
        pid_terms.append(
            1.0 - F.cosine_similarity(rgb_pid_feat, thermal_pid_feat, dim=1).squeeze(0)
        )

    if not pid_terms:
        return zero
    return torch.stack(pid_terms).mean()


def _compute_shared_consistency_layer_loss(rgb_feat, thermal_feat, pids, mode):
    mode = str(mode).lower()
    if mode == "batch_mean":
        return _compute_batch_mean_shared_consistency_layer_loss(
            rgb_feat, thermal_feat
        )
    if mode == "id_pair":
        return _compute_id_pair_shared_consistency_layer_loss(
            rgb_feat, thermal_feat, pids
        )
    raise ValueError(f"Unsupported moe_shared_consistency_mode={mode}.")


def _compute_shared_consistency_batch_stats(rgb_feat, thermal_feat, pids):
    rgb_count = int(rgb_feat.shape[0]) if torch.is_tensor(rgb_feat) and rgb_feat.dim() == 2 else 0
    thermal_count = (
        int(thermal_feat.shape[0])
        if torch.is_tensor(thermal_feat) and thermal_feat.dim() == 2
        else 0
    )
    common_pid_count = 0
    if (
        torch.is_tensor(pids)
        and rgb_count > 0
        and thermal_count > 0
    ):
        pids = pids.reshape(-1)
        if pids.shape[0] == rgb_count and pids.shape[0] == thermal_count:
            common_pid_count = int(torch.unique(pids).numel())
    return rgb_count, thermal_count, common_pid_count


def _resolve_shared_consistency_layer_keys(args, rgb_shared_feats, thermal_shared_feats):
    layer_keys = sorted(
        set(rgb_shared_feats.keys()).intersection(thermal_shared_feats.keys()),
        key=lambda x: int(x),
    )
    layer_spec = str(getattr(args, "moe_shared_consistency_layers", "all")).strip()
    if not layer_spec or layer_spec.lower() == "all":
        return layer_keys

    requested_layer_keys = []
    requested_seen = set()
    for token in layer_spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            layer_key = str(int(token))
        except ValueError as exc:
            raise ValueError(
                "moe_shared_consistency_layers must be 'all' or a comma-separated "
                f"list of integers, got {layer_spec!r}."
            ) from exc
        if layer_key in requested_seen:
            continue
        requested_seen.add(layer_key)
        requested_layer_keys.append(layer_key)

    if not requested_layer_keys:
        return layer_keys

    available_layer_keys = set(layer_keys)
    return [layer_key for layer_key in requested_layer_keys if layer_key in available_layer_keys]


def _resolve_moe_router_warmup_target(args, epoch):
    if not bool(getattr(args, "use_rgbt_moe_adapter_a2", False)):
        return 0.5, False

    warmup_epochs = int(getattr(args, "moe_router_warmup_epochs", 0))
    if warmup_epochs <= 0 or int(epoch) > warmup_epochs:
        return 0.5, False

    target_start = float(getattr(args, "moe_router_shared_target_start", 0.8))
    target_end = float(getattr(args, "moe_router_shared_target_end", 0.5))
    if not 0.0 <= target_start <= 1.0:
        raise ValueError(
            "moe_router_shared_target_start must be in [0, 1], "
            f"got {target_start}."
        )
    if not 0.0 <= target_end <= 1.0:
        raise ValueError(
            "moe_router_shared_target_end must be in [0, 1], "
            f"got {target_end}."
        )

    if warmup_epochs == 1:
        return target_start, True

    progress = float(int(epoch) - 1) / float(warmup_epochs - 1)
    target_shared = target_start + (target_end - target_start) * progress
    return target_shared, True


def _compute_router_warmup_term(weights, specific_index, target_shared):
    allowed_weights = torch.stack([weights[:, 0], weights[:, specific_index]], dim=1)
    target = allowed_weights.new_tensor([target_shared, 1.0 - target_shared]).view(1, 2)
    return F.mse_loss(allowed_weights, target.expand_as(allowed_weights))


def _get_moe_aux_schedule_config(args):
    schedule_mode = str(getattr(args, "moe_aux_schedule_mode", "constant")).lower()
    stage1_end = int(getattr(args, "moe_aux_stage1_end", 10))
    stage2_end = int(getattr(args, "moe_aux_stage2_end", 30))
    return schedule_mode, stage1_end, stage2_end


def resolve_moe_aux_effective_lambdas(args, epoch):
    schedule_mode, stage1_end, stage2_end = _get_moe_aux_schedule_config(args)
    default_router_balance = float(getattr(args, "lambda_router_balance", 0.01))
    default_shared_consistency = float(
        getattr(args, "lambda_shared_consistency", 0.05)
    )

    if schedule_mode == "constant":
        return "constant", default_router_balance, default_shared_consistency

    if schedule_mode == "three_stage":
        if stage1_end > stage2_end:
            raise ValueError(
                "moe_aux_stage1_end must be <= moe_aux_stage2_end, "
                f"got {stage1_end} and {stage2_end}."
            )
        epoch = int(epoch)
        if epoch <= stage1_end:
            return (
                "stage1",
                float(getattr(args, "lambda_router_balance_stage1", 0.03)),
                float(getattr(args, "lambda_shared_consistency_stage1", 0.10)),
            )
        if epoch <= stage2_end:
            return (
                "stage2",
                float(getattr(args, "lambda_router_balance_stage2", 0.02)),
                float(getattr(args, "lambda_shared_consistency_stage2", 0.08)),
            )
        return (
            "stage3",
            float(getattr(args, "lambda_router_balance_stage3", 0.01)),
            float(getattr(args, "lambda_shared_consistency_stage3", 0.05)),
        )

    raise ValueError(f"Unsupported moe_aux_schedule_mode={schedule_mode}.")


def log_moe_aux_schedule_state(logger, args, epoch, prefix="Epoch"):
    if not bool(getattr(args, "use_rgbt_moe_adapter", False)):
        return

    schedule_mode, stage1_end, stage2_end = _get_moe_aux_schedule_config(args)
    stage_name, effective_lambda_router_balance, effective_lambda_shared_consistency = (
        resolve_moe_aux_effective_lambdas(args, epoch)
    )
    logger.info(
        "%s [MOE] aux schedule state: current_epoch=%d (1-based), "
        "moe_aux_schedule_mode=%s, current_moe_aux_stage=%s, moe_aux_stage1_end=%d, "
        "moe_aux_stage2_end=%d, effective_lambda_router_balance=%.6f, "
        "effective_lambda_shared_consistency=%.6f",
        prefix,
        int(epoch),
        schedule_mode,
        stage_name,
        stage1_end,
        stage2_end,
        effective_lambda_router_balance,
        effective_lambda_shared_consistency,
    )


def _get_img_local_stage_config(args):
    use_image_local_filter = bool(getattr(args, "use_image_local_filter", False))
    stage_mode = str(getattr(args, "img_local_stage_mode", "always")).lower()
    stage_start = int(getattr(args, "img_local_stage_start", 1))
    stage_end = int(getattr(args, "img_local_stage_end", 9999))
    return use_image_local_filter, stage_mode, stage_start, stage_end


def _get_img_local_gamma_schedule_config(args):
    gamma_schedule_mode = str(
        getattr(args, "img_local_gamma_schedule_mode", "constant")
    ).lower()
    gamma_stage1_end = int(getattr(args, "img_local_gamma_stage1_end", 6))
    gamma_stage2_end = int(getattr(args, "img_local_gamma_stage2_end", 12))
    gamma_stage1_value = float(getattr(args, "img_local_gamma_stage1_value", 0.05))
    gamma_stage2_value = float(getattr(args, "img_local_gamma_stage2_value", 0.02))
    gamma_stage3_value = float(getattr(args, "img_local_gamma_stage3_value", 0.0))
    return (
        gamma_schedule_mode,
        gamma_stage1_end,
        gamma_stage2_end,
        gamma_stage1_value,
        gamma_stage2_value,
        gamma_stage3_value,
    )


def resolve_img_local_filter_active(args, epoch):
    use_image_local_filter, stage_mode, stage_start, stage_end = _get_img_local_stage_config(
        args
    )
    if not use_image_local_filter:
        return False
    if stage_mode == "always":
        return True
    if stage_mode == "window":
        return stage_start <= int(epoch) <= stage_end
    raise ValueError(f"Unsupported img_local_stage_mode={stage_mode}.")


def resolve_img_local_effective_fuse_gamma(args, epoch, active=None):
    use_image_local_filter = bool(getattr(args, "use_image_local_filter", False))
    if not use_image_local_filter:
        return 0.0

    if active is None:
        active = resolve_img_local_filter_active(args, epoch)
    if not active:
        return 0.0

    raw_fuse_gamma = float(getattr(args, "img_local_fuse_gamma", 0.1))
    (
        gamma_schedule_mode,
        gamma_stage1_end,
        gamma_stage2_end,
        gamma_stage1_value,
        gamma_stage2_value,
        gamma_stage3_value,
    ) = _get_img_local_gamma_schedule_config(args)

    if gamma_schedule_mode == "constant":
        return raw_fuse_gamma
    if gamma_schedule_mode == "three_stage":
        if gamma_stage1_end > gamma_stage2_end:
            raise ValueError(
                "img_local_gamma_stage1_end must be <= img_local_gamma_stage2_end, "
                f"got {gamma_stage1_end} and {gamma_stage2_end}."
            )
        epoch = int(epoch)
        if epoch <= gamma_stage1_end:
            return gamma_stage1_value
        if epoch <= gamma_stage2_end:
            return gamma_stage2_value
        return gamma_stage3_value
    raise ValueError(
        f"Unsupported img_local_gamma_schedule_mode={gamma_schedule_mode}."
    )


def resolve_img_local_runtime(args, epoch):
    active = resolve_img_local_filter_active(args, epoch)
    effective_fuse_gamma = resolve_img_local_effective_fuse_gamma(
        args, epoch, active=active
    )
    return active, effective_fuse_gamma


def _unwrap_runtime_model(model):
    target_model = model
    visited = set()
    while hasattr(target_model, "module") and id(target_model) not in visited:
        visited.add(id(target_model))
        target_model = target_model.module
    return target_model


def set_model_proto_runtime_epoch(model, epoch):
    target_model = _unwrap_runtime_model(model)
    if hasattr(target_model, "set_proto_runtime_epoch"):
        target_model.set_proto_runtime_epoch(epoch)
    else:
        target_model.proto_runtime_epoch = int(epoch)
    applied_epoch = int(getattr(target_model, "proto_runtime_epoch", epoch))
    if target_model is not model:
        setattr(model, "proto_runtime_epoch", applied_epoch)
    return applied_epoch


def _get_proto_calib_runtime_state(args, model):
    proto_enabled = bool(getattr(args, "use_proto_calib", False))
    state = {
        "proto_enabled": proto_enabled,
        "proto_loss_active": False,
        "proto_update_active": False,
        "num_initialized_pids": 0,
        "num_classes": 0,
        "initialized_mask": None,
    }
    if not proto_enabled:
        return state

    target_model = _unwrap_runtime_model(model)
    state["num_classes"] = int(getattr(target_model, "num_classes", 0))
    if hasattr(target_model, "should_enable_proto_loss"):
        state["proto_loss_active"] = bool(target_model.should_enable_proto_loss())
    if hasattr(target_model, "should_update_proto_bank"):
        state["proto_update_active"] = bool(target_model.should_update_proto_bank())
    if hasattr(target_model, "get_visual_prototype_initialized_mask"):
        initialized_mask = target_model.get_visual_prototype_initialized_mask()
        if initialized_mask is not None:
            state["initialized_mask"] = initialized_mask
            state["num_initialized_pids"] = int(initialized_mask.sum().item())
    return state


def log_proto_calib_state(logger, args, model, epoch, prefix="Epoch"):
    state = _get_proto_calib_runtime_state(args, model)
    logger.info(
        "%s proto calib state: "
        "current_epoch=%d (1-based), "
        "use_proto_calib=%s, "
        "proto_loss_active=%s, "
        "proto_update_active=%s, "
        "proto_warmup_mode=%s, "
        "proto_warmup_epochs=%d, "
        "lambda_proto_v=%.6f, "
        "lambda_proto_t=%.6f, "
        "proto_temp=%.6f, "
        "proto_momentum=%.6f, "
        "proto_initialized_pids=%d/%d",
        prefix,
        int(epoch),
        state["proto_enabled"],
        state["proto_loss_active"],
        state["proto_update_active"],
        "disable_loss_only",
        int(getattr(args, "proto_warmup_epochs", 0)),
        float(getattr(args, "lambda_proto_v", 1.0)),
        float(getattr(args, "lambda_proto_t", 1.0)),
        float(getattr(args, "proto_temp", 0.07)),
        float(getattr(args, "proto_momentum", 0.9)),
        state["num_initialized_pids"],
        state["num_classes"],
    )


def clear_model_img_local_runtime(model):
    target_model = _unwrap_runtime_model(model)
    target_model.img_local_filter_active = False
    if hasattr(target_model, "effective_img_local_fuse_gamma"):
        delattr(target_model, "effective_img_local_fuse_gamma")
    if target_model is not model:
        setattr(model, "img_local_filter_active", False)
        if hasattr(model, "effective_img_local_fuse_gamma"):
            delattr(model, "effective_img_local_fuse_gamma")
    return False, 0.0


def set_model_img_local_runtime(model, active, fuse_gamma):
    target_model = _unwrap_runtime_model(model)
    if hasattr(target_model, "set_img_local_runtime"):
        target_model.set_img_local_runtime(active=active, fuse_gamma=fuse_gamma)
    elif hasattr(target_model, "set_img_local_filter_active"):
        target_model.set_img_local_filter_active(active)
        target_model.effective_img_local_fuse_gamma = float(
            fuse_gamma if getattr(target_model, "img_local_filter_active", False) else 0.0
        )
    else:
        target_model.img_local_filter_active = bool(
            getattr(target_model, "use_image_local_filter", False) and active
        )
        target_model.effective_img_local_fuse_gamma = float(
            fuse_gamma if target_model.img_local_filter_active else 0.0
        )
    applied_active = bool(getattr(target_model, "img_local_filter_active", False))
    applied_fuse_gamma = float(
        getattr(target_model, "effective_img_local_fuse_gamma", 0.0)
    )
    if target_model is not model:
        setattr(model, "img_local_filter_active", applied_active)
        if hasattr(target_model, "effective_img_local_fuse_gamma"):
            setattr(model, "effective_img_local_fuse_gamma", applied_fuse_gamma)
        elif hasattr(model, "effective_img_local_fuse_gamma"):
            delattr(model, "effective_img_local_fuse_gamma")
    return applied_active, applied_fuse_gamma


def set_model_img_local_filter_active(model, active):
    target_model = _unwrap_runtime_model(model)
    default_fuse_gamma = float(
        getattr(target_model, "img_local_fuse_gamma", 0.0) if active else 0.0
    )
    applied_active, _ = set_model_img_local_runtime(
        model, active=active, fuse_gamma=default_fuse_gamma
    )
    return applied_active


def log_img_local_filter_v1_train_config(logger, args):
    if not bool(getattr(args, "use_img_local_filter", False)):
        return
    logger.info("[LOCAL] image-only local filter enabled")
    logger.info("topr=%d", int(getattr(args, "img_local_topr", 4)))
    logger.info("gamma=%.6f", float(getattr(args, "img_local_gamma", 0.1)))
    logger.info("temp=%.6f", float(getattr(args, "img_local_temp", 0.07)))
    logger.info(
        "detach_global=%s",
        bool(getattr(args, "img_local_detach_global", True)),
    )
    if float(getattr(args, "img_local_gamma", 0.1)) <= 0.0:
        logger.info("[LOCAL] gamma<=0, fallback to original global feature")


def log_img_local_filter_v1_eval_state(logger, model, prefix="EVAL"):
    target_model = _unwrap_runtime_model(model)
    enabled = bool(
        getattr(
            target_model,
            "use_img_local_filter",
            getattr(getattr(target_model, "args", None), "use_img_local_filter", False),
        )
    )
    if not enabled:
        logger.info(
            "[LOCAL][%s] image-only local filter enabled=False",
            str(prefix).upper(),
        )
        return
    gamma = float(
        getattr(
            target_model,
            "img_local_gamma",
            getattr(getattr(target_model, "args", None), "img_local_gamma", 0.1),
        )
    )
    logger.info(
        "[LOCAL][%s] image-only local filter enabled=%s, topr=%d, gamma=%.6f, temp=%.6f, detach_global=%s",
        str(prefix).upper(),
        enabled,
        int(
            getattr(
                target_model,
                "img_local_topr",
                getattr(getattr(target_model, "args", None), "img_local_topr", 4),
            )
        ),
        gamma,
        float(
            getattr(
                target_model,
                "img_local_temp",
                getattr(getattr(target_model, "args", None), "img_local_temp", 0.07),
            )
        ),
        bool(
            getattr(
                target_model,
                "img_local_detach_global",
                getattr(
                    getattr(target_model, "args", None),
                    "img_local_detach_global",
                    True,
                ),
            )
        ),
    )
    if enabled and gamma <= 0.0:
        logger.info(
            "[LOCAL][%s] gamma<=0, current runtime falls back to original global feature",
            str(prefix).upper(),
        )


def log_rgbt_moe_adapter_config(logger, args, model):
    if not bool(getattr(args, "use_rgbt_moe_adapter", False)):
        return

    target_model = _unwrap_runtime_model(model)
    enabled_layers = []
    visual = getattr(getattr(target_model, "base_model", None), "visual", None)
    transformer = getattr(visual, "transformer", None)
    if transformer is not None:
        enabled_layers = list(
            getattr(transformer, "rgbt_moe_adapter_layers", enabled_layers)
        )
    if not enabled_layers:
        layer_spec = str(getattr(args, "moe_adapter_layers", "")).strip()
        if layer_spec:
            enabled_layers = [int(v.strip()) for v in layer_spec.split(",") if v.strip()]

    logger.info(
        "[MOE] use_rgbt_moe_adapter=%s, use_rgbt_moe_adapter_a2=%s, shared_consistency_mode=%s, moe_shared_consistency_layers=%s, log_moe_pid_stats=%s, enabled_visual_layers(0-based)=%s, moe_adapter_dim=%d, moe_adapter_dropout=%.6f, moe_router_temp=%.6f, lambda_router_balance=%.6f, lambda_shared_consistency=%.6f, lambda_router_warmup=%.6f, moe_router_warmup_epochs=%d, moe_router_shared_target_start=%.6f, moe_router_shared_target_end=%.6f",
        bool(getattr(args, "use_rgbt_moe_adapter", False)),
        bool(getattr(args, "use_rgbt_moe_adapter_a2", False)),
        str(getattr(args, "moe_shared_consistency_mode", "batch_mean")).lower(),
        str(getattr(args, "moe_shared_consistency_layers", "all")).strip(),
        bool(getattr(args, "log_moe_pid_stats", False)),
        enabled_layers,
        int(getattr(args, "moe_adapter_dim", 64)),
        float(getattr(args, "moe_adapter_dropout", 0.1)),
        float(getattr(args, "moe_router_temp", 1.0)),
        float(getattr(args, "lambda_router_balance", 0.01)),
        float(getattr(args, "lambda_shared_consistency", 0.05)),
        float(getattr(args, "lambda_router_warmup", 0.0)),
        int(getattr(args, "moe_router_warmup_epochs", 0)),
        float(getattr(args, "moe_router_shared_target_start", 0.8)),
        float(getattr(args, "moe_router_shared_target_end", 0.5)),
    )


def log_img_local_filter_state(
    logger, args, epoch, active, effective_fuse_gamma, prefix="Epoch"
):
    (
        use_image_local_filter,
        img_local_stage_mode,
        img_local_stage_start,
        img_local_stage_end,
    ) = _get_img_local_stage_config(args)
    (
        img_local_gamma_schedule_mode,
        img_local_gamma_stage1_end,
        img_local_gamma_stage2_end,
        img_local_gamma_stage1_value,
        img_local_gamma_stage2_value,
        img_local_gamma_stage3_value,
    ) = _get_img_local_gamma_schedule_config(args)
    logger.info(
        "%s img local filter state: "
        "current_epoch=%d (1-based), "
        "use_image_local_filter=%s, "
        "img_local_stage_mode=%s, "
        "img_local_stage_start=%d, "
        "img_local_stage_end=%d, "
        "img_local_filter_active=%s, "
        "img_local_gamma_schedule_mode=%s, "
        "img_local_fuse_gamma=%.6f, "
        "effective_img_local_fuse_gamma=%.6f, "
        "img_local_gamma_stage1_end=%d, "
        "img_local_gamma_stage2_end=%d, "
        "img_local_gamma_stage1_value=%.6f, "
        "img_local_gamma_stage2_value=%.6f, "
        "img_local_gamma_stage3_value=%.6f",
        prefix,
        epoch,
        use_image_local_filter,
        img_local_stage_mode,
        img_local_stage_start,
        img_local_stage_end,
        active,
        img_local_gamma_schedule_mode,
        float(getattr(args, "img_local_fuse_gamma", 0.1)),
        float(effective_fuse_gamma),
        img_local_gamma_stage1_end,
        img_local_gamma_stage2_end,
        img_local_gamma_stage1_value,
        img_local_gamma_stage2_value,
        img_local_gamma_stage3_value,
    )


def apply_img_local_filter_stage(
    model, args, epoch, logger=None, prefix="Epoch", runtime=None
):
    if runtime is None:
        runtime = resolve_img_local_runtime(args, epoch)
    requested_active, requested_fuse_gamma = runtime
    if not bool(getattr(args, "use_image_local_filter", False)):
        applied_active, applied_fuse_gamma = clear_model_img_local_runtime(model)
        if logger is not None:
            log_img_local_filter_state(
                logger,
                args,
                epoch,
                applied_active,
                applied_fuse_gamma,
                prefix=prefix,
            )
        return applied_active

    applied_active, applied_fuse_gamma = set_model_img_local_runtime(
        model,
        active=requested_active,
        fuse_gamma=requested_fuse_gamma,
    )
    if logger is not None:
        log_img_local_filter_state(
            logger,
            args,
            epoch,
            applied_active,
            applied_fuse_gamma,
            prefix=prefix,
        )
    return applied_active


def _build_total_loss(args, ret):
    """
    When use_mum is enabled:
      total_loss = loss_sdm_prob + lambda_det * loss_sdm_det + lambda_mlm * loss_mlm
    Otherwise keep legacy behavior:
      total_loss = sum(all keys containing 'loss')
    """
    if getattr(args, "use_mum", False):
        loss_sdm_prob = ret.get("loss_sdm_prob", ret.get("sdm_loss", None))
        if loss_sdm_prob is None:
            raise KeyError(
                "use_mum=True requires model to return 'loss_sdm_prob' or 'sdm_loss'."
            )
        loss_sdm_det = ret.get("loss_sdm_det", None)
        if loss_sdm_det is None:
            loss_sdm_det = loss_sdm_prob.new_zeros(())
        loss_mlm = ret.get("loss_mlm", ret.get("mlm_loss", None))
        if loss_mlm is None:
            loss_mlm = loss_sdm_prob.new_zeros(())
        total_loss = (
            loss_sdm_prob
            + getattr(args, "lambda_det", 0.5) * loss_sdm_det
            + getattr(args, "lambda_mlm", 1.0) * loss_mlm
        )
    else:
        # Legacy mode: sum all losses but avoid double-counting SDM alias
        # ('sdm_loss' and 'loss_sdm_prob' point to the same term).
        total_loss = None
        for k, v in ret.items():
            if "loss" not in k:
                continue
            if k == "loss_sdm_prob" and "sdm_loss" in ret:
                continue
            # These terms are added explicitly with dedicated weights below.
            if k in {
                "loss_unc_total",
                "loss_unc_text_rgb",
                "loss_unc_rgb_thermal",
                "loss_unc_text_thermal",
                "loss_var_reg",
                "loss_dist_sdm_total",
                "loss_dist_sdm_text_rgb",
                "loss_dist_sdm_rgb_thermal",
            }:
                continue
            total_loss = v if total_loss is None else total_loss + v

        if total_loss is None:
            raise KeyError("No loss term found in model outputs.")

    if getattr(args, "use_global_uncertainty", False):
        if "loss_unc_total" in ret:
            total_loss = total_loss + getattr(args, "lambda_unc", 0.1) * ret["loss_unc_total"]
        if "loss_var_reg" in ret:
            total_loss = total_loss + getattr(args, "lambda_var_reg", 1e-4) * ret["loss_var_reg"]
    if getattr(args, "use_distribution_sdm_aux", False):
        if "loss_dist_sdm_total" in ret:
            total_loss = total_loss + getattr(args, "lambda_dist_sdm", 1.0) * ret["loss_dist_sdm_total"]
    return total_loss


def _compute_rgbt_moe_adapter_terms(args, ret, batch, epoch):
    if not bool(getattr(args, "use_rgbt_moe_adapter", False)):
        return {}

    zero = batch["images"].new_zeros(())
    use_a2 = bool(getattr(args, "use_rgbt_moe_adapter_a2", False))
    shared_consistency_mode = str(
        getattr(args, "moe_shared_consistency_mode", "batch_mean")
    ).lower()
    (
        _,
        effective_lambda_router_balance,
        effective_lambda_shared_consistency,
    ) = resolve_moe_aux_effective_lambdas(args, epoch)
    out = {
        "loss_router_balance": zero,
        "loss_router_warmup": zero,
        "loss_shared_consistency": zero,
        "effective_lambda_router_balance": zero.new_tensor(
            effective_lambda_router_balance
        ),
        "effective_lambda_shared_consistency": zero.new_tensor(
            effective_lambda_shared_consistency
        ),
        "shared_consistency_active_layer_count": zero,
        "shared_cons_valid_common_pid_count": zero,
        "shared_cons_valid_rgb_count": zero,
        "shared_cons_valid_thermal_count": zero,
        "rgb_shared_weight_mean": zero,
        "rgb_specific_weight_mean": zero,
        "thermal_shared_weight_mean": zero,
        "thermal_specific_weight_mean": zero,
        "router_warmup_target_shared": zero,
        "shared_consistency_active_layers": "none",
    }

    aux = ret.get("rgbt_moe_adapter_aux", None)
    if not isinstance(aux, dict):
        return out

    router_weights = aux.get("router_weights", {})
    shared_branch_feats = aux.get("shared_branch_pooled_features", {})
    rgb_router_weights = router_weights.get("rgb", {})
    thermal_router_weights = router_weights.get("thermal", {})

    rgb_shared_means = []
    rgb_specific_means = []
    thermal_shared_means = []
    thermal_specific_means = []
    router_balance_terms = []
    router_warmup_terms = []

    uniform_target = zero.new_tensor([0.5, 0.5])
    router_warmup_target, router_warmup_active = _resolve_moe_router_warmup_target(
        args, epoch
    )
    if router_warmup_active:
        out["router_warmup_target_shared"] = zero.new_tensor(router_warmup_target)

    for layer_key, weights in rgb_router_weights.items():
        if not torch.is_tensor(weights) or weights.numel() == 0:
            continue
        weights = weights.float()
        shared_mean = weights[:, 0].mean()
        specific_mean = weights[:, 1].mean()
        rgb_shared_means.append(shared_mean)
        rgb_specific_means.append(specific_mean)
        router_balance_terms.append(
            F.mse_loss(torch.stack([shared_mean, specific_mean]), uniform_target)
        )
        if router_warmup_active:
            router_warmup_terms.append(
                _compute_router_warmup_term(
                    weights=weights,
                    specific_index=1,
                    target_shared=router_warmup_target,
                )
            )

    for layer_key, weights in thermal_router_weights.items():
        if not torch.is_tensor(weights) or weights.numel() == 0:
            continue
        weights = weights.float()
        shared_mean = weights[:, 0].mean()
        specific_mean = weights[:, 2].mean()
        thermal_shared_means.append(shared_mean)
        thermal_specific_means.append(specific_mean)
        router_balance_terms.append(
            F.mse_loss(torch.stack([shared_mean, specific_mean]), uniform_target)
        )
        if router_warmup_active:
            router_warmup_terms.append(
                _compute_router_warmup_term(
                    weights=weights,
                    specific_index=2,
                    target_shared=router_warmup_target,
                )
            )

    if rgb_shared_means:
        out["rgb_shared_weight_mean"] = torch.stack(rgb_shared_means).mean()
    if rgb_specific_means:
        out["rgb_specific_weight_mean"] = torch.stack(rgb_specific_means).mean()
    if thermal_shared_means:
        out["thermal_shared_weight_mean"] = torch.stack(thermal_shared_means).mean()
    if thermal_specific_means:
        out["thermal_specific_weight_mean"] = torch.stack(
            thermal_specific_means
        ).mean()
    if router_balance_terms:
        out["loss_router_balance"] = torch.stack(router_balance_terms).mean()
    if router_warmup_terms:
        out["loss_router_warmup"] = torch.stack(router_warmup_terms).mean()

    rgb_shared_feats = shared_branch_feats.get("rgb", {})
    thermal_shared_feats = shared_branch_feats.get("thermal", {})
    shared_consistency_terms = []
    shared_consistency_active_layers = []
    shared_consistency_pids = batch.get("pids", None)
    shared_consistency_stats_set = False
    for layer_key in _resolve_shared_consistency_layer_keys(
        args,
        rgb_shared_feats=rgb_shared_feats,
        thermal_shared_feats=thermal_shared_feats,
    ):
        rgb_feat = rgb_shared_feats[layer_key]
        thermal_feat = thermal_shared_feats[layer_key]
        if not torch.is_tensor(rgb_feat) or not torch.is_tensor(thermal_feat):
            continue
        if rgb_feat.dim() != 2 or thermal_feat.dim() != 2:
            continue
        if rgb_feat.shape != thermal_feat.shape:
            continue
        if rgb_feat.shape[0] == 0:
            continue
        if not shared_consistency_stats_set:
            (
                valid_rgb_count,
                valid_thermal_count,
                valid_common_pid_count,
            ) = _compute_shared_consistency_batch_stats(
                rgb_feat=rgb_feat,
                thermal_feat=thermal_feat,
                pids=shared_consistency_pids,
            )
            out["shared_cons_valid_rgb_count"] = zero.new_tensor(valid_rgb_count)
            out["shared_cons_valid_thermal_count"] = zero.new_tensor(
                valid_thermal_count
            )
            out["shared_cons_valid_common_pid_count"] = zero.new_tensor(
                valid_common_pid_count
            )
            shared_consistency_stats_set = True
        shared_consistency_active_layers.append(layer_key)
        shared_consistency_terms.append(
            _compute_shared_consistency_layer_loss(
                rgb_feat=rgb_feat,
                thermal_feat=thermal_feat,
                pids=shared_consistency_pids,
                mode=shared_consistency_mode,
            )
        )

    if shared_consistency_terms:
        out["loss_shared_consistency"] = torch.stack(
            shared_consistency_terms
        ).mean()
    out["shared_consistency_active_layer_count"] = zero.new_tensor(
        len(shared_consistency_active_layers)
    )
    if shared_consistency_active_layers:
        out["shared_consistency_active_layers"] = ",".join(
            shared_consistency_active_layers
        )

    return out


def _add_rgbt_moe_adapter_terms_to_total_loss(args, total_loss, ret, epoch):
    if not bool(getattr(args, "use_rgbt_moe_adapter", False)):
        return total_loss
    (
        _,
        default_effective_lambda_router_balance,
        default_effective_lambda_shared_consistency,
    ) = resolve_moe_aux_effective_lambdas(args, epoch)
    effective_lambda_router_balance = float(
        _to_scalar(
            ret.get(
                "effective_lambda_router_balance",
                default_effective_lambda_router_balance,
            )
        )
    )
    effective_lambda_shared_consistency = float(
        _to_scalar(
            ret.get(
                "effective_lambda_shared_consistency",
                default_effective_lambda_shared_consistency,
            )
        )
    )
    if "loss_router_balance" in ret:
        total_loss = (
            total_loss
            + effective_lambda_router_balance * ret["loss_router_balance"]
        )
    if "loss_router_warmup" in ret:
        total_loss = total_loss + float(
            getattr(args, "lambda_router_warmup", 0.0)
        ) * ret["loss_router_warmup"]
    if "loss_shared_consistency" in ret:
        total_loss = (
            total_loss
            + effective_lambda_shared_consistency * ret["loss_shared_consistency"]
        )
    return total_loss


def _compute_hardneg_aux_terms(args, ret, batch, epoch):
    if not bool(getattr(args, "use_hardneg_aux", False)):
        return {}

    zero = batch["images"].new_zeros(())
    if int(epoch) < int(getattr(args, "hn_start_epoch", 1)):
        return {
            "loss_hn": zero,
            "hn_active": False,
            "hn_num_valid_queries": 0.0,
            "hn_avg_pos_sim": zero,
            "hn_avg_neg_sim": zero,
            "hn_avg_margin_gap": zero,
        }

    fused_feat = ret.get("fused_det_feat", ret.get("fused_prob_feat", None))
    text_feat = ret.get("text_det_feat", ret.get("text_prob_feat", None))
    pids = batch.get("pids", None)
    if fused_feat is None or text_feat is None or pids is None:
        raise KeyError(
            "Hard-negative auxiliary loss requires fused/text global features "
            "and batch['pids']."
        )

    hn_out = objectives.compute_hard_negative_aux_loss(
        query_features=text_feat,
        gallery_features=fused_feat,
        query_pid=pids,
        gallery_pid=pids,
        margin=float(getattr(args, "hn_margin", 0.05)),
        topn=int(getattr(args, "hn_topn", 3)),
    )
    return {
        "loss_hn": hn_out["loss"],
        "hn_active": True,
        "hn_num_valid_queries": hn_out["num_valid_queries"],
        "hn_avg_pos_sim": hn_out["avg_pos_sim"],
        "hn_avg_neg_sim": hn_out["avg_neg_sim"],
        "hn_avg_margin_gap": hn_out["avg_margin_gap"],
    }


def _add_hardneg_aux_to_total_loss(args, total_loss, ret, epoch):
    if not bool(getattr(args, "use_hardneg_aux", False)):
        return total_loss
    if int(epoch) < int(getattr(args, "hn_start_epoch", 5)):
        return total_loss
    if "loss_hn" in ret:
        total_loss = total_loss + float(getattr(args, "lambda_hn", 0.05)) * ret["loss_hn"]
    return total_loss


def _compute_proto_calibration_terms(args, model, ret, batch):
    if not bool(getattr(args, "use_proto_calib", False)):
        return {}

    target_model = _unwrap_runtime_model(model)
    state = _get_proto_calib_runtime_state(args, model)
    proto_out = {
        "proto_loss_active": float(state["proto_loss_active"]),
        "proto_num_valid_targets": 0.0,
    }
    if not hasattr(target_model, "get_visual_prototype_matrix"):
        return proto_out

    prototype_matrix = target_model.get_visual_prototype_matrix()
    if prototype_matrix is None:
        return proto_out

    fused_feat = ret.get("fused_det_feat", None)
    text_feat = ret.get("text_det_feat", None)
    pids = batch.get("pids", None)
    if fused_feat is None or text_feat is None or pids is None:
        raise KeyError(
            "Prototype calibration requires 'fused_det_feat', "
            "'text_det_feat', and batch['pids']."
        )

    zero = fused_feat.new_zeros(())
    if not state["proto_loss_active"]:
        proto_out.update({"loss_proto_v": zero, "loss_proto_t": zero})
        return proto_out

    fused_feat = _safe_l2_normalize(fused_feat.float(), dim=1)
    text_feat = _safe_l2_normalize(text_feat.float(), dim=1)
    prototype_matrix = _safe_l2_normalize(prototype_matrix.float(), dim=1)
    pids = pids.long()

    proto_temp = float(getattr(args, "proto_temp", 0.07))
    if proto_temp <= 0:
        raise ValueError(f"proto_temp must be > 0, got {proto_temp}.")

    initialized_mask = state["initialized_mask"]
    if initialized_mask is None:
        initialized_mask = torch.ones(
            prototype_matrix.shape[0],
            device=prototype_matrix.device,
            dtype=torch.bool,
        )
    else:
        initialized_mask = initialized_mask.to(
            device=prototype_matrix.device, dtype=torch.bool
        )

    valid_target_mask = initialized_mask[pids]
    proto_out["proto_num_valid_targets"] = float(valid_target_mask.sum().item())
    if not valid_target_mask.any():
        proto_out.update({"loss_proto_v": zero, "loss_proto_t": zero})
        return proto_out

    valid_fused_feat = fused_feat[valid_target_mask]
    valid_text_feat = text_feat[valid_target_mask]
    valid_pids = pids[valid_target_mask]

    logits_v2p = valid_fused_feat @ prototype_matrix.t()
    logits_t2p = valid_text_feat @ prototype_matrix.t()
    logits_v2p = logits_v2p / proto_temp
    logits_t2p = logits_t2p / proto_temp

    # Mask still-uninitialized prototypes to avoid invalid competition / NaNs.
    logits_v2p = logits_v2p.masked_fill(~initialized_mask.unsqueeze(0), -1e4)
    logits_t2p = logits_t2p.masked_fill(~initialized_mask.unsqueeze(0), -1e4)

    proto_out.update(
        {
            "loss_proto_v": F.cross_entropy(logits_v2p, valid_pids),
            "loss_proto_t": F.cross_entropy(logits_t2p, valid_pids),
        }
    )
    return proto_out


def _add_proto_calibration_to_total_loss(args, total_loss, ret):
    if not bool(getattr(args, "use_proto_calib", False)):
        return total_loss
    if "loss_proto_v" in ret:
        total_loss = total_loss + float(getattr(args, "lambda_proto_v", 1.0)) * ret["loss_proto_v"]
    if "loss_proto_t" in ret:
        total_loss = total_loss + float(getattr(args, "lambda_proto_t", 1.0)) * ret["loss_proto_t"]
    return total_loss


def update_model_proto_bank_after_step(model, ret, batch):
    target_model = _unwrap_runtime_model(model)
    if not hasattr(target_model, "update_visual_prototype_bank"):
        return {
            "updated": False,
            "num_updated_pids": 0,
            "num_newly_initialized_pids": 0,
            "num_initialized_pids": 0,
        }
    fused_feat = ret.get("fused_det_feat", None)
    if fused_feat is None:
        return {
            "updated": False,
            "num_updated_pids": 0,
            "num_newly_initialized_pids": 0,
            "num_initialized_pids": 0,
        }
    return target_model.update_visual_prototype_bank(
        fused_feats=fused_feat,
        pids=batch.get("pids", None),
    )


def _mean_positive_pairwise_distance(dist_matrix, pids=None):
    """
    Average pairwise distance over positive matches.

    If pids are provided, positives are defined by pid equality (same as SDM label logic).
    Otherwise, fallback to diagonal matched pairs (index-aligned assumption).
    """
    if pids is None:
        return torch.diagonal(dist_matrix).mean()

    pids = pids.reshape(-1)
    pos_mask = pids.unsqueeze(1).eq(pids.unsqueeze(0))
    if pos_mask.any():
        return dist_matrix[pos_mask].mean()
    return torch.diagonal(dist_matrix).mean()


def _compute_uncertainty_alignment_terms(args, ret, batch):
    """
    Compute global uncertainty alignment losses from model outputs.

    Uses closed-form diagonal Gaussian W2^2 on global features:
      - text <-> rgb
      - rgb <-> thermal
      - text <-> thermal (optional, weighted by lambda_tt)

    Assumption:
      Current dataloader provides paired image-text samples per batch row.
      When repeated identities appear in a batch, we use pid-equality positive masks.
    """
    use_global_uncertainty = bool(getattr(args, "use_global_uncertainty", False))
    use_distribution_sdm_aux = bool(getattr(args, "use_distribution_sdm_aux", False))
    if use_distribution_sdm_aux and not use_global_uncertainty:
        raise ValueError(
            "use_distribution_sdm_aux=True requires --use_global_uncertainty."
        )
    if not use_global_uncertainty:
        return {}

    required = [
        "text_mu",
        "text_logvar",
        "rgb_mu",
        "rgb_logvar",
        "thermal_mu",
        "thermal_logvar",
    ]
    missing = [k for k in required if k not in ret]
    if missing:
        raise KeyError(
            "use_global_uncertainty=True requires model outputs to include "
            f"{missing}, but got keys: {list(ret.keys())}"
        )

    text_mu = ret["text_mu"]
    text_logvar = ret["text_logvar"]
    rgb_mu = ret["rgb_mu"]
    rgb_logvar = ret["rgb_logvar"]
    thermal_mu = ret["thermal_mu"]
    thermal_logvar = ret["thermal_logvar"]
    pids = batch.get("pids", None)
    has_thermal = "thermal_images" in batch

    eps = float(getattr(args, "uncertainty_eps", 1e-6))
    logvar_min = float(getattr(args, "logvar_min", -6.0))
    logvar_max = float(getattr(args, "logvar_max", 4.0))

    tr_dist = objectives.compute_w2_diag_pairwise_matrix(
        mu1=text_mu,
        logvar1=text_logvar,
        mu2=rgb_mu,
        logvar2=rgb_logvar,
        eps=eps,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )
    loss_unc_text_rgb = _mean_positive_pairwise_distance(tr_dist, pids)

    rt_dist = None
    if has_thermal:
        rt_dist = objectives.compute_w2_diag_pairwise_matrix(
            mu1=rgb_mu,
            logvar1=rgb_logvar,
            mu2=thermal_mu,
            logvar2=thermal_logvar,
            eps=eps,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        loss_unc_rgb_thermal = _mean_positive_pairwise_distance(rt_dist, pids)
    else:
        loss_unc_rgb_thermal = loss_unc_text_rgb.new_zeros(())

    if has_thermal and float(getattr(args, "lambda_tt", 0.0)) > 0.0:
        tt_dist = objectives.compute_w2_diag_pairwise_matrix(
            mu1=text_mu,
            logvar1=text_logvar,
            mu2=thermal_mu,
            logvar2=thermal_logvar,
            eps=eps,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        loss_unc_text_thermal = _mean_positive_pairwise_distance(tt_dist, pids)
    else:
        loss_unc_text_thermal = loss_unc_text_rgb.new_zeros(())

    loss_unc_total = (
        float(getattr(args, "lambda_tr", 1.0)) * loss_unc_text_rgb
        + float(getattr(args, "lambda_rt", 1.0)) * loss_unc_rgb_thermal
        + float(getattr(args, "lambda_tt", 0.0)) * loss_unc_text_thermal
    )

    if use_distribution_sdm_aux:
        if pids is None:
            raise KeyError(
                "use_distribution_sdm_aux=True requires batch to provide 'pids'."
            )
        dist_sdm_temp = float(getattr(args, "distribution_sdm_temperature", 1.0))
        image_ids = batch.get("image_ids", None)
        loss_dist_sdm_text_rgb = objectives.compute_distribution_sdm_from_distance(
            distance_matrix=tr_dist,
            pid=pids,
            image_id=image_ids,
            epsilon=eps,
            temperature=dist_sdm_temp,
        )

        if has_thermal and bool(
            getattr(args, "use_distribution_sdm_rgb_thermal", False)
        ):
            loss_dist_sdm_rgb_thermal = objectives.compute_distribution_sdm_from_distance(
                distance_matrix=rt_dist,
                pid=pids,
                image_id=image_ids,
                epsilon=eps,
                temperature=dist_sdm_temp,
            )
        else:
            loss_dist_sdm_rgb_thermal = loss_dist_sdm_text_rgb.new_zeros(())

        loss_dist_sdm_total = (
            float(getattr(args, "lambda_dist_sdm_tr", 1.0)) * loss_dist_sdm_text_rgb
            + float(getattr(args, "lambda_dist_sdm_rt", 1.0))
            * loss_dist_sdm_rgb_thermal
        )

    # Lightweight logvar regularization to avoid explosion/collapse.
    var_reg_terms = [text_logvar.pow(2).mean(), rgb_logvar.pow(2).mean()]
    if has_thermal:
        var_reg_terms.append(thermal_logvar.pow(2).mean())
    if "fused_logvar" in ret and torch.is_tensor(ret["fused_logvar"]):
        var_reg_terms.append(ret["fused_logvar"].pow(2).mean())
    loss_var_reg = sum(var_reg_terms) / len(var_reg_terms)

    out = {
        "loss_unc_total": loss_unc_total,
        "loss_unc_text_rgb": loss_unc_text_rgb,
        "loss_unc_rgb_thermal": loss_unc_rgb_thermal,
        "loss_unc_text_thermal": loss_unc_text_thermal,
        "loss_var_reg": loss_var_reg,
        "mean_text_logvar": text_logvar.mean(),
        "mean_rgb_logvar": rgb_logvar.mean(),
        "mean_thermal_logvar": thermal_logvar.mean(),
    }
    if use_distribution_sdm_aux:
        out.update(
            {
                "loss_dist_sdm_total": loss_dist_sdm_total,
                "loss_dist_sdm_text_rgb": loss_dist_sdm_text_rgb,
                "loss_dist_sdm_rgb_thermal": loss_dist_sdm_rgb_thermal,
            }
        )
    if "fused_logvar" in ret and torch.is_tensor(ret["fused_logvar"]):
        out["mean_fused_logvar"] = ret["fused_logvar"].mean()
    return out


def do_train(
    start_epoch,
    args,
    model,
    train_loader,
    evaluator,
    optimizer,
    scheduler,
    checkpointer,
):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch

    arguments = {"num_epoch": num_epoch, "iteration": 0}

    logger = logging.getLogger("IRRA.train")
    logger.info("start training")
    log_img_local_filter_v1_train_config(logger, args)
    log_rgbt_moe_adapter_config(logger, args, model)

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "loss_sdm_prob": AverageMeter(),
        "loss_sdm_det": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "loss_mlm": AverageMeter(),
        "loss_unc_total": AverageMeter(),
        "loss_unc_text_rgb": AverageMeter(),
        "loss_unc_rgb_thermal": AverageMeter(),
        "loss_unc_text_thermal": AverageMeter(),
        "loss_dist_sdm_total": AverageMeter(),
        "loss_dist_sdm_text_rgb": AverageMeter(),
        "loss_dist_sdm_rgb_thermal": AverageMeter(),
        "loss_router_balance": AverageMeter(),
        "loss_router_warmup": AverageMeter(),
        "loss_shared_consistency": AverageMeter(),
        "effective_lambda_router_balance": AverageMeter(),
        "effective_lambda_shared_consistency": AverageMeter(),
        "shared_consistency_active_layer_count": AverageMeter(),
        "shared_cons_valid_common_pid_count": AverageMeter(),
        "shared_cons_valid_rgb_count": AverageMeter(),
        "shared_cons_valid_thermal_count": AverageMeter(),
        "loss_hn": AverageMeter(),
        "hn_active": AverageMeter(),
        "hn_num_valid_queries": AverageMeter(),
        "hn_avg_pos_sim": AverageMeter(),
        "hn_avg_neg_sim": AverageMeter(),
        "hn_avg_margin_gap": AverageMeter(),
        "loss_proto_v": AverageMeter(),
        "loss_proto_t": AverageMeter(),
        "proto_loss_active": AverageMeter(),
        "proto_num_valid_targets": AverageMeter(),
        "proto_num_updated_pids": AverageMeter(),
        "proto_num_newly_initialized_pids": AverageMeter(),
        "proto_num_initialized_pids": AverageMeter(),
        "loss_var_reg": AverageMeter(),
        "mean_text_logvar": AverageMeter(),
        "mean_rgb_logvar": AverageMeter(),
        "mean_thermal_logvar": AverageMeter(),
        "mean_fused_logvar": AverageMeter(),
        "avg_fused_batch_uncertainty": AverageMeter(),
        "avg_text_batch_uncertainty": AverageMeter(),
        "rgb_shared_weight_mean": AverageMeter(),
        "rgb_specific_weight_mean": AverageMeter(),
        "thermal_shared_weight_mean": AverageMeter(),
        "thermal_specific_weight_mean": AverageMeter(),
        "router_warmup_target_shared": AverageMeter(),
        "avg_img_local_uncertainty": AverageMeter(),
        "avg_img_local_keep_ratio": AverageMeter(),
        "avg_img_local_weight_entropy": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_top1 = 0.0

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        epoch_img_local_runtime = resolve_img_local_runtime(args, epoch)
        img_local_filter_active = apply_img_local_filter_stage(
            model,
            args,
            epoch,
            logger=logger,
            prefix="Epoch",
            runtime=epoch_img_local_runtime,
        )
        set_model_proto_runtime_epoch(model, epoch)
        model.train()
        log_proto_calib_state(logger, args, model, epoch, prefix="Epoch")
        log_moe_aux_schedule_state(logger, args, epoch, prefix="Epoch")

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)
            ret.update(_compute_uncertainty_alignment_terms(args, ret, batch))
            original_total_loss = _build_total_loss(args, ret)
            ret.update(_compute_rgbt_moe_adapter_terms(args, ret, batch, epoch))
            if getattr(args, "use_hardneg_aux", False):
                ret.update(_compute_hardneg_aux_terms(args, ret, batch, epoch))
            ret.update(_compute_proto_calibration_terms(args, model, ret, batch))
            total_loss = _add_rgbt_moe_adapter_terms_to_total_loss(
                args, original_total_loss, ret, epoch
            )
            total_loss = _add_proto_calibration_to_total_loss(args, total_loss, ret)
            if getattr(args, "use_hardneg_aux", False):
                total_loss = _add_hardneg_aux_to_total_loss(
                    args, total_loss, ret, epoch
                )

            batch_size = batch["images"].shape[0]
            meters["loss"].update(_to_scalar(total_loss), batch_size)
            meters["sdm_loss"].update(_to_scalar(ret.get("sdm_loss", 0)), batch_size)
            meters["loss_sdm_prob"].update(
                _to_scalar(ret.get("loss_sdm_prob", ret.get("sdm_loss", 0))),
                batch_size,
            )
            meters["loss_sdm_det"].update(
                _to_scalar(ret.get("loss_sdm_det", 0)),
                batch_size,
            )
            meters["itc_loss"].update(_to_scalar(ret.get("itc_loss", 0)), batch_size)
            meters["id_loss"].update(_to_scalar(ret.get("id_loss", 0)), batch_size)
            meters["mlm_loss"].update(_to_scalar(ret.get("mlm_loss", 0)), batch_size)
            meters["loss_mlm"].update(
                _to_scalar(ret.get("loss_mlm", ret.get("mlm_loss", 0))), batch_size
            )
            meters["loss_unc_total"].update(
                _to_scalar(ret.get("loss_unc_total", 0)), batch_size
            )
            meters["loss_unc_text_rgb"].update(
                _to_scalar(ret.get("loss_unc_text_rgb", 0)), batch_size
            )
            meters["loss_unc_rgb_thermal"].update(
                _to_scalar(ret.get("loss_unc_rgb_thermal", 0)), batch_size
            )
            meters["loss_unc_text_thermal"].update(
                _to_scalar(ret.get("loss_unc_text_thermal", 0)), batch_size
            )
            meters["loss_dist_sdm_total"].update(
                _to_scalar(ret.get("loss_dist_sdm_total", 0)), batch_size
            )
            meters["loss_dist_sdm_text_rgb"].update(
                _to_scalar(ret.get("loss_dist_sdm_text_rgb", 0)), batch_size
            )
            meters["loss_dist_sdm_rgb_thermal"].update(
                _to_scalar(ret.get("loss_dist_sdm_rgb_thermal", 0)), batch_size
            )
            meters["loss_router_balance"].update(
                _to_scalar(ret.get("loss_router_balance", 0)), batch_size
            )
            meters["loss_router_warmup"].update(
                _to_scalar(ret.get("loss_router_warmup", 0)), batch_size
            )
            meters["loss_shared_consistency"].update(
                _to_scalar(ret.get("loss_shared_consistency", 0)), batch_size
            )
            if "effective_lambda_router_balance" in ret:
                meters["effective_lambda_router_balance"].update(
                    _to_scalar(ret["effective_lambda_router_balance"]), batch_size
                )
            if "effective_lambda_shared_consistency" in ret:
                meters["effective_lambda_shared_consistency"].update(
                    _to_scalar(ret["effective_lambda_shared_consistency"]),
                    batch_size,
                )
            meters["shared_consistency_active_layer_count"].update(
                _to_scalar(ret.get("shared_consistency_active_layer_count", 0)),
                batch_size,
            )
            meters["shared_cons_valid_common_pid_count"].update(
                _to_scalar(ret.get("shared_cons_valid_common_pid_count", 0)),
                batch_size,
            )
            meters["shared_cons_valid_rgb_count"].update(
                _to_scalar(ret.get("shared_cons_valid_rgb_count", 0)),
                batch_size,
            )
            meters["shared_cons_valid_thermal_count"].update(
                _to_scalar(ret.get("shared_cons_valid_thermal_count", 0)),
                batch_size,
            )
            if getattr(args, "use_hardneg_aux", False):
                meters["loss_hn"].update(_to_scalar(ret.get("loss_hn", 0)), batch_size)
                meters["hn_active"].update(_to_scalar(ret.get("hn_active", 0)), 1)
                meters["hn_num_valid_queries"].update(
                    _to_scalar(ret.get("hn_num_valid_queries", 0)), 1
                )
                meters["hn_avg_pos_sim"].update(
                    _to_scalar(ret.get("hn_avg_pos_sim", 0)), batch_size
                )
                meters["hn_avg_neg_sim"].update(
                    _to_scalar(ret.get("hn_avg_neg_sim", 0)), batch_size
                )
                meters["hn_avg_margin_gap"].update(
                    _to_scalar(ret.get("hn_avg_margin_gap", 0)), batch_size
                )
            meters["loss_proto_v"].update(
                _to_scalar(ret.get("loss_proto_v", 0)), batch_size
            )
            meters["loss_proto_t"].update(
                _to_scalar(ret.get("loss_proto_t", 0)), batch_size
            )
            meters["proto_loss_active"].update(
                _to_scalar(ret.get("proto_loss_active", 0)), 1
            )
            meters["proto_num_valid_targets"].update(
                _to_scalar(ret.get("proto_num_valid_targets", 0)), 1
            )
            meters["loss_var_reg"].update(
                _to_scalar(ret.get("loss_var_reg", 0)), batch_size
            )
            meters["mean_text_logvar"].update(
                _to_scalar(ret.get("mean_text_logvar", 0)), batch_size
            )
            meters["mean_rgb_logvar"].update(
                _to_scalar(ret.get("mean_rgb_logvar", 0)), batch_size
            )
            meters["mean_thermal_logvar"].update(
                _to_scalar(ret.get("mean_thermal_logvar", 0)), batch_size
            )
            meters["mean_fused_logvar"].update(
                _to_scalar(ret.get("mean_fused_logvar", 0)), batch_size
            )

            if "sigma_batch_f" in ret:
                meters["avg_fused_batch_uncertainty"].update(
                    _to_scalar(ret["sigma_batch_f"]), batch_size
                )
            if "sigma_batch_t" in ret:
                meters["avg_text_batch_uncertainty"].update(
                    _to_scalar(ret["sigma_batch_t"]), batch_size
                )
            if "rgb_shared_weight_mean" in ret:
                meters["rgb_shared_weight_mean"].update(
                    _to_scalar(ret["rgb_shared_weight_mean"]), batch_size
                )
            if "rgb_specific_weight_mean" in ret:
                meters["rgb_specific_weight_mean"].update(
                    _to_scalar(ret["rgb_specific_weight_mean"]), batch_size
                )
            if "thermal_shared_weight_mean" in ret:
                meters["thermal_shared_weight_mean"].update(
                    _to_scalar(ret["thermal_shared_weight_mean"]), batch_size
                )
            if "thermal_specific_weight_mean" in ret:
                meters["thermal_specific_weight_mean"].update(
                    _to_scalar(ret["thermal_specific_weight_mean"]), batch_size
                )
            if "router_warmup_target_shared" in ret:
                meters["router_warmup_target_shared"].update(
                    _to_scalar(ret["router_warmup_target_shared"]), batch_size
                )
            if "avg_img_local_uncertainty" in ret:
                meters["avg_img_local_uncertainty"].update(
                    _to_scalar(ret["avg_img_local_uncertainty"]), batch_size
                )
            if "avg_img_local_keep_ratio" in ret:
                meters["avg_img_local_keep_ratio"].update(
                    _to_scalar(ret["avg_img_local_keep_ratio"]), batch_size
                )
            if "avg_img_local_weight_entropy" in ret:
                meters["avg_img_local_weight_entropy"].update(
                    _to_scalar(ret["avg_img_local_weight_entropy"]), batch_size
                )

            meters["img_acc"].update(_to_scalar(ret.get("img_acc", 0)), batch_size)
            meters["txt_acc"].update(_to_scalar(ret.get("txt_acc", 0)), batch_size)
            meters["mlm_acc"].update(_to_scalar(ret.get("mlm_acc", 0)), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            proto_update_info = update_model_proto_bank_after_step(model, ret, batch)
            meters["proto_num_updated_pids"].update(
                float(proto_update_info.get("num_updated_pids", 0)), 1
            )
            meters["proto_num_newly_initialized_pids"].update(
                float(proto_update_info.get("num_newly_initialized_pids", 0)), 1
            )
            meters["proto_num_initialized_pids"].update(
                float(proto_update_info.get("num_initialized_pids", 0)), 1
            )
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.count > 0:
                        if (
                            not bool(getattr(args, "log_moe_pid_stats", False))
                            and k in {
                                "shared_cons_valid_common_pid_count",
                                "shared_cons_valid_rgb_count",
                                "shared_cons_valid_thermal_count",
                            }
                        ):
                            continue
                        info_str += f", {k}: {v.avg:.4f}"
                active_layers = ret.get("shared_consistency_active_layers", "none")
                info_str += f", shared_consistency_active_layers: {active_layers}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar("temperature", _to_scalar(ret["temperature"]), epoch)
        for k, v in meters.items():
            if v.count > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % eval_period == 0 and get_rank() == 0:
            img_local_filter_active = apply_img_local_filter_stage(
                model,
                args,
                epoch,
                logger=logger,
                prefix="Validation",
                runtime=epoch_img_local_runtime,
            )
            log_img_local_filter_v1_eval_state(logger, model, prefix="VALIDATION")
            logger.info("Validation Results - Epoch: {}".format(epoch))
            if args.distributed:
                top1 = evaluator.eval(model.module.eval())
            else:
                top1 = evaluator.eval(model.eval())

            torch.cuda.empty_cache()
            if best_top1 < top1:
                best_top1 = top1
                arguments["epoch"] = epoch
                checkpointer.save("best", **arguments)

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")
    log_img_local_filter_v1_eval_state(logger, model, prefix="INFERENCE")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    evaluator.eval(model.eval())
