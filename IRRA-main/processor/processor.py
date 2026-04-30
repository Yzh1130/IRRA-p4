import logging
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import objectives
from utils.comm import get_rank, synchronize
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.mtp_text import build_rgb_text, build_thermal_text
from utils.simple_tokenizer import SimpleTokenizer


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


def _truncate_log_text(text, max_chars=160):
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _decode_caption_token_ids_for_log(caption_ids, tokenizer):
    token_ids = caption_ids.detach().cpu().reshape(-1).tolist()
    sot_token = int(tokenizer.encoder["<|startoftext|>"])
    eot_token = int(tokenizer.encoder["<|endoftext|>"])
    filtered = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == 0 or token_id == sot_token:
            continue
        if token_id == eot_token:
            break
        filtered.append(token_id)
    if not filtered:
        return ""
    return " ".join(tokenizer.decode(filtered).split())


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


def _linear_ramp(cur_epoch, start_epoch, ramp_epochs, target):
    cur_epoch = int(cur_epoch)
    start_epoch = int(start_epoch)
    ramp_epochs = int(ramp_epochs)
    target = float(target)
    if ramp_epochs <= 0:
        return target if cur_epoch >= start_epoch else 0.0
    if cur_epoch < start_epoch:
        return 0.0
    if cur_epoch >= start_epoch + ramp_epochs:
        return target
    ratio = float(cur_epoch - start_epoch) / float(ramp_epochs)
    return target * ratio


def _linear_interp(cur_epoch, start_epoch, ramp_epochs, start_value, end_value):
    cur_epoch = int(cur_epoch)
    start_epoch = int(start_epoch)
    ramp_epochs = int(ramp_epochs)
    start_value = float(start_value)
    end_value = float(end_value)
    if ramp_epochs <= 0:
        return end_value if cur_epoch >= start_epoch else start_value
    if cur_epoch < start_epoch:
        return start_value
    if cur_epoch >= start_epoch + ramp_epochs:
        return end_value
    ratio = float(cur_epoch - start_epoch) / float(ramp_epochs)
    return start_value + ratio * (end_value - start_value)


def resolve_mtp_effective_lambdas(args, epoch):
    schedule_mode = str(getattr(args, "mtp_stage_mode", "three_stage")).lower()
    stage1_end = int(getattr(args, "mtp_stage1_end", 10))
    stage2_end = int(getattr(args, "mtp_stage2_end", 45))

    if schedule_mode == "constant":
        return (
            "constant",
            float(getattr(args, "lambda_rgb_txt_stage3", 0.10)),
            float(getattr(args, "lambda_th_txt_stage3", 0.04)),
            float(getattr(args, "lambda_txt_cons_stage3", 0.005)),
        )

    if schedule_mode == "three_stage":
        if stage1_end > stage2_end:
            raise ValueError(
                "mtp_stage1_end must be <= mtp_stage2_end, "
                f"got {stage1_end} and {stage2_end}."
            )
        cur_epoch = int(epoch)
        if cur_epoch <= 0:
            # Keep existing 1-based training-loop behavior while tolerating 0-based calls.
            cur_epoch = cur_epoch + 1

        lambda_rgb_txt_stage2 = float(getattr(args, "lambda_rgb_txt_stage2", 0.08))
        lambda_th_txt_stage2 = float(getattr(args, "lambda_th_txt_stage2", 0.03))
        lambda_txt_cons_stage2 = float(getattr(args, "lambda_txt_cons_stage2", 0.003))
        lambda_rgb_txt_stage3 = float(getattr(args, "lambda_rgb_txt_stage3", 0.10))
        lambda_th_txt_stage3 = float(getattr(args, "lambda_th_txt_stage3", 0.04))
        lambda_txt_cons_stage3 = float(getattr(args, "lambda_txt_cons_stage3", 0.005))

        if cur_epoch <= stage1_end:
            return (
                "stage1",
                0.0,
                0.0,
                0.0,
            )
        if cur_epoch <= stage2_end:
            stage2_start_epoch = stage1_end + 1
            stage2_ramp_epochs = int(getattr(args, "mtp_stage2_ramp_epochs", 0))
            return (
                "stage2",
                _linear_ramp(
                    cur_epoch, stage2_start_epoch, stage2_ramp_epochs, lambda_rgb_txt_stage2
                ),
                _linear_ramp(
                    cur_epoch, stage2_start_epoch, stage2_ramp_epochs, lambda_th_txt_stage2
                ),
                _linear_ramp(
                    cur_epoch,
                    stage2_start_epoch,
                    stage2_ramp_epochs,
                    lambda_txt_cons_stage2,
                ),
            )
        stage3_start_epoch = stage2_end + 1
        stage3_ramp_epochs = int(getattr(args, "mtp_stage3_ramp_epochs", 0))
        return (
            "stage3",
            _linear_interp(
                cur_epoch,
                stage3_start_epoch,
                stage3_ramp_epochs,
                lambda_rgb_txt_stage2,
                lambda_rgb_txt_stage3,
            ),
            _linear_interp(
                cur_epoch,
                stage3_start_epoch,
                stage3_ramp_epochs,
                lambda_th_txt_stage2,
                lambda_th_txt_stage3,
            ),
            _linear_interp(
                cur_epoch,
                stage3_start_epoch,
                stage3_ramp_epochs,
                lambda_txt_cons_stage2,
                lambda_txt_cons_stage3,
            ),
        )

    raise ValueError(f"Unsupported mtp_stage_mode={schedule_mode}.")


def _resolve_mtp_current_phase(args, epoch, stage_name):
    schedule_mode = str(getattr(args, "mtp_stage_mode", "three_stage")).lower()
    if schedule_mode != "three_stage":
        return stage_name

    cur_epoch = int(epoch)
    if cur_epoch <= 0:
        cur_epoch = cur_epoch + 1

    stage1_end = int(getattr(args, "mtp_stage1_end", 10))
    stage2_end = int(getattr(args, "mtp_stage2_end", 45))
    stage2_ramp_epochs = int(getattr(args, "mtp_stage2_ramp_epochs", 0))
    stage3_ramp_epochs = int(getattr(args, "mtp_stage3_ramp_epochs", 0))

    if stage_name == "stage1":
        return "stage1"
    if stage_name == "stage2":
        stage2_start_epoch = stage1_end + 1
        if stage2_ramp_epochs > 0 and cur_epoch < stage2_start_epoch + stage2_ramp_epochs:
            return "stage2_ramp"
        return "stage2_hold"
    if stage_name == "stage3":
        stage3_start_epoch = stage2_end + 1
        if stage3_ramp_epochs > 0 and cur_epoch < stage3_start_epoch + stage3_ramp_epochs:
            return "stage3_ramp"
        return "stage3_hold"
    return stage_name


def log_mtp_schedule_state(logger, args, epoch):
    if not bool(getattr(args, "use_modality_text_prompt", False)):
        return
    (
        stage_name,
        effective_lambda_rgb_txt,
        effective_lambda_th_txt,
        effective_lambda_txt_cons,
    ) = resolve_mtp_effective_lambdas(args, epoch)
    mtp_stage2_ramp_epochs = int(getattr(args, "mtp_stage2_ramp_epochs", 0))
    mtp_stage3_ramp_epochs = int(getattr(args, "mtp_stage3_ramp_epochs", 0))
    current_mtp_phase = _resolve_mtp_current_phase(args, epoch, stage_name)
    logger.info("[MTP] current_epoch=%d", int(epoch))
    logger.info("[MTP] mtp_stage2_ramp_epochs=%d", mtp_stage2_ramp_epochs)
    logger.info("[MTP] mtp_stage3_ramp_epochs=%d", mtp_stage3_ramp_epochs)
    logger.info("[MTP] current_mtp_phase=%s", current_mtp_phase)
    logger.info("[MTP] effective_lambda_rgb_txt=%.6f", effective_lambda_rgb_txt)
    logger.info("[MTP] effective_lambda_th_txt=%.6f", effective_lambda_th_txt)
    logger.info("[MTP] effective_lambda_txt_cons=%.6f", effective_lambda_txt_cons)


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
        "proto_init_loaded": False,
        "proto_update_frozen": False,
        "num_initialized_pids": 0,
        "num_visual_initialized_pids": 0,
        "num_text_initialized_pids": 0,
        "num_valid_classes": 0,
        "num_classes": 0,
        "initialized_mask": None,
        "text_initialized_mask": None,
    }
    if not proto_enabled:
        return state

    target_model = _unwrap_runtime_model(model)
    state["num_classes"] = int(getattr(target_model, "num_classes", 0))
    if hasattr(target_model, "should_enable_proto_loss"):
        state["proto_loss_active"] = bool(target_model.should_enable_proto_loss())
    if hasattr(target_model, "should_update_proto_bank"):
        state["proto_update_active"] = bool(target_model.should_update_proto_bank())
    state["proto_init_loaded"] = bool(getattr(target_model, "proto_init_loaded", False))
    if hasattr(target_model, "is_proto_update_frozen"):
        state["proto_update_frozen"] = bool(target_model.is_proto_update_frozen())
    if hasattr(target_model, "get_visual_prototype_initialized_mask"):
        initialized_mask = target_model.get_visual_prototype_initialized_mask()
        if initialized_mask is not None:
            state["initialized_mask"] = initialized_mask
            state["num_visual_initialized_pids"] = int(initialized_mask.sum().item())
            state["num_initialized_pids"] = int(initialized_mask.sum().item())
    if hasattr(target_model, "get_text_prototype_initialized_mask"):
        text_initialized_mask = target_model.get_text_prototype_initialized_mask()
        if text_initialized_mask is not None:
            state["text_initialized_mask"] = text_initialized_mask
            state["num_text_initialized_pids"] = int(
                text_initialized_mask.sum().item()
            )
            if state["initialized_mask"] is not None:
                visual_mask = state["initialized_mask"].to(
                    device=text_initialized_mask.device, dtype=torch.bool
                )
                text_mask = text_initialized_mask.to(dtype=torch.bool)
                state["num_valid_classes"] = int((visual_mask & text_mask).sum().item())
                state["num_initialized_pids"] = state["num_valid_classes"]
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
        "proto_init_loaded=%s, "
        "proto_update_frozen=%s, "
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
        state["proto_init_loaded"],
        state["proto_update_frozen"],
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
    logger.info("[LOCAL_V1] legacy image-only local filter v1 enabled")
    logger.info("topr=%d", int(getattr(args, "img_local_topr", 4)))
    logger.info("gamma=%.6f", float(getattr(args, "img_local_gamma", 0.1)))
    logger.info("temp=%.6f", float(getattr(args, "img_local_temp", 0.07)))
    logger.info(
        "detach_global=%s",
        bool(getattr(args, "img_local_detach_global", True)),
    )
    if float(getattr(args, "img_local_gamma", 0.1)) <= 0.0:
        logger.info("[LOCAL_V1] gamma<=0, fallback to original global feature")


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
            "[LOCAL_V1][%s] legacy image-only local filter v1 enabled=False",
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
        "[LOCAL_V1][%s] legacy image-only local filter v1 enabled=%s, topr=%d, gamma=%.6f, temp=%.6f, detach_global=%s",
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
            "[LOCAL_V1][%s] gamma<=0, current runtime falls back to original global feature",
            str(prefix).upper(),
        )


def log_img_local_filter_train_config(logger, args):
    if not bool(getattr(args, "use_image_local_filter", False)):
        return
    logger.info(
        "[LOCAL] image-only local filter config: "
        "use_image_local_filter=%s, "
        "img_local_stage_mode=%s, "
        "img_local_stage_start=%d, "
        "img_local_stage_end=%d, "
        "img_local_topk=%d, "
        "img_local_topr=%d, "
        "img_local_unc_lambda=%.6f, "
        "img_local_fuse_gamma=%.6f, "
        "img_local_gamma_schedule_mode=%s, "
        "img_local_gamma_stage1_end=%d, "
        "img_local_gamma_stage2_end=%d, "
        "img_local_gamma_stage1_value=%.6f, "
        "img_local_gamma_stage2_value=%.6f, "
        "img_local_gamma_stage3_value=%.6f, "
        "img_local_score_temp=%.6f, "
        "img_local_detach_global=%s",
        bool(getattr(args, "use_image_local_filter", False)),
        str(getattr(args, "img_local_stage_mode", "always")).lower(),
        int(getattr(args, "img_local_stage_start", 1)),
        int(getattr(args, "img_local_stage_end", 9999)),
        int(getattr(args, "img_local_topk", 64)),
        int(getattr(args, "img_local_topr", 4)),
        float(getattr(args, "img_local_unc_lambda", 0.3)),
        float(getattr(args, "img_local_fuse_gamma", 0.1)),
        str(getattr(args, "img_local_gamma_schedule_mode", "constant")).lower(),
        int(getattr(args, "img_local_gamma_stage1_end", 6)),
        int(getattr(args, "img_local_gamma_stage2_end", 12)),
        float(getattr(args, "img_local_gamma_stage1_value", 0.05)),
        float(getattr(args, "img_local_gamma_stage2_value", 0.02)),
        float(getattr(args, "img_local_gamma_stage3_value", 0.0)),
        float(getattr(args, "img_local_score_temp", 0.07)),
        bool(getattr(args, "img_local_detach_global", True)),
    )


def log_img_local_filter_eval_state(logger, model, prefix="EVAL"):
    target_model = _unwrap_runtime_model(model)
    model_args = getattr(target_model, "args", None)
    configured = bool(
        getattr(
            target_model,
            "use_image_local_filter",
            getattr(model_args, "use_image_local_filter", False),
        )
    )
    enabled = bool(
        configured and getattr(target_model, "img_local_filter_active", False)
    )
    logger.info(
        "[LOCAL][%s] image-only local filter enabled=%s, effective_img_local_fuse_gamma=%.6f, img_local_topk=%d, img_local_topr=%d, img_local_unc_lambda=%.6f, img_local_score_temp=%.6f",
        str(prefix).upper(),
        enabled,
        float(getattr(target_model, "effective_img_local_fuse_gamma", 0.0)),
        int(
            getattr(
                target_model,
                "img_local_topk",
                getattr(model_args, "img_local_topk", 64),
            )
        ),
        int(
            getattr(
                target_model,
                "img_local_topr",
                getattr(model_args, "img_local_topr", 4),
            )
        ),
        float(
            getattr(
                target_model,
                "img_local_unc_lambda",
                getattr(model_args, "img_local_unc_lambda", 0.3),
            )
        ),
        float(
            getattr(
                target_model,
                "img_local_score_temp",
                getattr(model_args, "img_local_score_temp", 0.07),
            )
        ),
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
        "img_local_topk=%d, "
        "img_local_topr=%d, "
        "img_local_unc_lambda=%.6f, "
        "img_local_score_temp=%.6f, "
        "img_local_detach_global=%s, "
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
        int(getattr(args, "img_local_topk", 64)),
        int(getattr(args, "img_local_topr", 4)),
        float(getattr(args, "img_local_unc_lambda", 0.3)),
        float(getattr(args, "img_local_score_temp", 0.07)),
        bool(getattr(args, "img_local_detach_global", True)),
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


def _get_caption_weight_for_loss(args, batch, apply_flag_name, default=True):
    if not bool(getattr(args, "use_synth_loss_weighting", False)):
        return None
    if not bool(getattr(args, apply_flag_name, default)):
        return None

    caption_weight = batch.get("caption_weight", None)
    if caption_weight is None:
        return None
    caption_weight = caption_weight.float().reshape(-1)
    if caption_weight.numel() == 0:
        return None
    if bool(torch.all(caption_weight.eq(1.0)).item()):
        return None
    return caption_weight


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


def _compute_mtp_text_prompt_terms(args, ret, batch, epoch):
    if not bool(getattr(args, "use_modality_text_prompt", False)):
        return {}

    zero = batch["images"].new_zeros(())
    (
        _,
        effective_lambda_rgb_txt,
        effective_lambda_th_txt,
        effective_lambda_txt_cons,
    ) = resolve_mtp_effective_lambdas(args, epoch)
    out = {
        "loss_rgb_txt": zero,
        "loss_th_txt": zero,
        "loss_txt_cons": zero,
        "effective_lambda_rgb_txt": zero.new_tensor(effective_lambda_rgb_txt),
        "effective_lambda_th_txt": zero.new_tensor(effective_lambda_th_txt),
        "effective_lambda_txt_cons": zero.new_tensor(effective_lambda_txt_cons),
    }

    f_rgb = ret.get("v_rgb_c", ret.get("rgb_mu", None))
    f_th = ret.get("v_t_c", ret.get("thermal_mu", None))
    e_full = ret.get("text_feat_full", ret.get("text_det_feat", None))
    e_rgb = ret.get("text_feat_rgb", None)
    e_th = ret.get("text_feat_th", None)
    pids = batch.get("pids", None)
    if (
        f_rgb is None
        or e_full is None
        or e_rgb is None
        or e_th is None
        or pids is None
    ):
        raise KeyError(
            "MTP losses require model outputs 'v_rgb_c', 'text_feat_full', "
            "'text_feat_rgb', 'text_feat_th' and batch['pids']."
        )
    caption_weight_for_sdm = _get_caption_weight_for_loss(
        args,
        batch,
        "synth_weight_apply_to_sdm",
        default=True,
    )

    temperature = ret.get("temperature", None)
    if temperature is None:
        logit_scale = f_rgb.new_tensor(1.0 / float(getattr(args, "temperature", 0.02)))
    elif torch.is_tensor(temperature):
        logit_scale = 1.0 / temperature.to(device=f_rgb.device, dtype=f_rgb.dtype).clamp_min(1e-12)
    else:
        logit_scale = f_rgb.new_tensor(1.0 / max(float(temperature), 1e-12))

    out["loss_rgb_txt"] = objectives.compute_sdm(
        f_rgb,
        e_rgb,
        pids,
        logit_scale,
        sample_weight=caption_weight_for_sdm,
    )
    if "thermal_images" in batch and f_th is not None:
        out["loss_th_txt"] = objectives.compute_sdm(
            f_th,
            e_th,
            pids,
            logit_scale,
            sample_weight=caption_weight_for_sdm,
        )

    e_rgb_th = F.normalize(e_rgb.float() + e_th.float(), p=2, dim=1)
    out["loss_txt_cons"] = 1.0 - F.cosine_similarity(
        e_full.float(), e_rgb_th, dim=1
    ).mean()

    return out


def _add_mtp_terms_to_total_loss(args, total_loss, ret, epoch):
    if not bool(getattr(args, "use_modality_text_prompt", False)):
        return total_loss

    (
        _,
        default_lambda_rgb_txt,
        default_lambda_th_txt,
        default_lambda_txt_cons,
    ) = resolve_mtp_effective_lambdas(args, epoch)
    effective_lambda_rgb_txt = float(
        _to_scalar(ret.get("effective_lambda_rgb_txt", default_lambda_rgb_txt))
    )
    effective_lambda_th_txt = float(
        _to_scalar(ret.get("effective_lambda_th_txt", default_lambda_th_txt))
    )
    effective_lambda_txt_cons = float(
        _to_scalar(ret.get("effective_lambda_txt_cons", default_lambda_txt_cons))
    )

    if "loss_rgb_txt" in ret:
        total_loss = total_loss + effective_lambda_rgb_txt * ret["loss_rgb_txt"]
    if "loss_th_txt" in ret:
        total_loss = total_loss + effective_lambda_th_txt * ret["loss_th_txt"]
    if "loss_txt_cons" in ret:
        total_loss = total_loss + effective_lambda_txt_cons * ret["loss_txt_cons"]
    return total_loss


def _compute_hardneg_aux_terms(args, ret, batch, epoch):
    if not bool(getattr(args, "use_hardneg_aux", False)):
        return {}

    zero = batch["images"].new_zeros(())
    hardneg_version = str(getattr(args, "hardneg_version", "v1")).lower()
    hn_direction = str(getattr(args, "hn_direction", "both")).lower()
    lambda_hn = float(getattr(args, "lambda_hn", 0.05))

    def _inactive_hardneg_terms():
        return {
            "loss_hn": zero,
            "loss_hn_weighted": zero,
            "hn_active": False,
            "hn_num_valid_queries": 0.0,
            "hn_avg_pos_sim": zero,
            "hn_avg_neg_sim": zero,
            "hn_avg_margin_gap": zero,
            "hn_violation_rate": zero,
            "hn_t2i_loss": zero,
            "hn_i2t_loss": zero,
            "hn_t2i_valid": 0.0,
            "hn_i2t_valid": 0.0,
            "hn_avg_top1_neg_sim": zero,
            "hn_avg_topk_neg_sim": zero,
            "hn_version": hardneg_version,
            "hn_direction": "t2i" if hardneg_version == "v1" else hn_direction,
        }

    if int(epoch) < int(getattr(args, "hn_start_epoch", 1)):
        return _inactive_hardneg_terms()

    fused_feat = ret.get("fused_det_feat", ret.get("fused_prob_feat", None))
    text_feat = ret.get("text_det_feat", ret.get("text_prob_feat", None))
    pids = batch.get("pids", None)
    if fused_feat is None or text_feat is None or pids is None:
        raise KeyError(
            "Hard-negative auxiliary loss requires fused/text global features "
            "and batch['pids']."
        )

    if hardneg_version == "v1":
        hn_out = objectives.compute_hard_negative_aux_loss(
            query_features=text_feat,
            gallery_features=fused_feat,
            query_pid=pids,
            gallery_pid=pids,
            margin=float(getattr(args, "hn_margin", 0.05)),
            topn=int(getattr(args, "hn_topn", 3)),
        )
        hn_direction = "t2i"
    elif hardneg_version == "v2":
        hn_out = objectives.compute_hard_negative_aux_loss_v2(
            text_features=text_feat,
            image_features=fused_feat,
            text_pid=pids,
            image_pid=pids,
            margin=float(getattr(args, "hn_margin", 0.05)),
            topn=int(getattr(args, "hn_topn", 3)),
            temperature=float(getattr(args, "hn_temperature", 0.07)),
            loss_type=str(getattr(args, "hn_loss_type", "softplus")).lower(),
            direction=hn_direction,
            false_neg_margin=float(getattr(args, "hn_false_neg_margin", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported hardneg_version={hardneg_version}.")

    loss_hn = hn_out["loss"]
    return {
        "loss_hn": loss_hn,
        "loss_hn_weighted": loss_hn * lambda_hn,
        "hn_active": True,
        "hn_num_valid_queries": hn_out["num_valid_queries"],
        "hn_avg_pos_sim": hn_out["avg_pos_sim"],
        "hn_avg_neg_sim": hn_out["avg_neg_sim"],
        "hn_avg_margin_gap": hn_out["avg_margin_gap"],
        "hn_violation_rate": hn_out.get("violation_rate", zero),
        "hn_t2i_loss": hn_out.get("t2i_loss", loss_hn),
        "hn_i2t_loss": hn_out.get("i2t_loss", zero),
        "hn_t2i_valid": hn_out.get("t2i_valid", hn_out["num_valid_queries"]),
        "hn_i2t_valid": hn_out.get("i2t_valid", 0.0),
        "hn_avg_top1_neg_sim": hn_out.get("avg_top1_neg_sim", zero),
        "hn_avg_topk_neg_sim": hn_out.get("avg_topk_neg_sim", hn_out["avg_neg_sim"]),
        "hn_version": hardneg_version,
        "hn_direction": hn_direction,
    }


def _add_hardneg_aux_to_total_loss(args, total_loss, ret, epoch):
    if not bool(getattr(args, "use_hardneg_aux", False)):
        return total_loss
    if int(epoch) < int(getattr(args, "hn_start_epoch", 5)):
        return total_loss
    if "loss_hn_weighted" in ret:
        total_loss = total_loss + ret["loss_hn_weighted"]
    elif "loss_hn" in ret:
        total_loss = total_loss + float(getattr(args, "lambda_hn", 0.05)) * ret["loss_hn"]
    return total_loss


def _compute_offline_hardneg_terms(args, ret, batch, epoch):
    if not bool(getattr(args, "use_offline_hardneg", False)):
        return {}

    zero = batch["images"].new_zeros(())
    lambda_offhn = float(getattr(args, "offline_hn_lambda", 0.05))
    margin = float(getattr(args, "offline_hn_margin", 0.05))
    temperature = float(getattr(args, "offline_hn_temperature", 0.07))
    if temperature <= 0:
        raise ValueError(
            f"offline_hn_temperature must be > 0, got {temperature}."
        )

    def _inactive_offhn_terms():
        return {
            "loss_offhn": zero,
            "loss_offhn_weighted": zero,
            "offhn_active": False,
            "offhn_num_valid": 0.0,
            "offhn_avg_pos_sim": zero,
            "offhn_avg_neg_sim": zero,
            "offhn_avg_margin_gap": zero,
            "offhn_violation_rate": zero,
            "offhn_lambda": zero.new_tensor(lambda_offhn),
            "offhn_margin": zero.new_tensor(margin),
            "offhn_temperature": zero.new_tensor(temperature),
        }

    if int(epoch) < int(getattr(args, "offline_hn_start_epoch", 1)):
        return _inactive_offhn_terms()

    text_feat = ret.get("text_det_feat", ret.get("text_prob_feat", None))
    pos_fused_feat = ret.get("fused_det_feat", ret.get("fused_prob_feat", None))
    hn_fused_feat = ret.get("hn_fused_det_feat", None)
    hn_valid_mask = batch.get("hn_valid_mask", None)
    hn_neg_pid = batch.get("hn_neg_pid", None)
    pids = batch.get("pids", None)
    if (
        text_feat is None
        or pos_fused_feat is None
        or hn_fused_feat is None
        or hn_valid_mask is None
        or hn_neg_pid is None
        or pids is None
    ):
        return _inactive_offhn_terms()

    if hn_fused_feat.dim() == 2:
        hn_fused_feat = hn_fused_feat.unsqueeze(1)
    if hn_valid_mask.dim() == 1:
        hn_valid_mask = hn_valid_mask.unsqueeze(1)
    if hn_neg_pid.dim() == 1:
        hn_neg_pid = hn_neg_pid.unsqueeze(1)

    text_norm = F.normalize(text_feat.float(), p=2, dim=1)
    pos_norm = F.normalize(pos_fused_feat.float(), p=2, dim=1)
    hn_norm = F.normalize(hn_fused_feat.float(), p=2, dim=2)

    sim_pos = F.cosine_similarity(text_norm, pos_norm, dim=1).unsqueeze(1)
    sim_pos = sim_pos.expand_as(hn_valid_mask.float())
    sim_neg = F.cosine_similarity(text_norm.unsqueeze(1), hn_norm, dim=2)

    valid_mask = hn_valid_mask.bool()
    valid_mask = valid_mask & hn_neg_pid.ne(pids.reshape(-1, 1))
    if not valid_mask.any():
        return _inactive_offhn_terms()

    margin_gap = sim_neg - sim_pos + margin
    raw = margin_gap / temperature
    per_pair_loss = F.softplus(raw) * temperature
    valid_loss = per_pair_loss[valid_mask]
    loss_offhn = valid_loss.mean()
    valid_margin_gap = margin_gap[valid_mask]

    return {
        "loss_offhn": loss_offhn,
        "loss_offhn_weighted": loss_offhn * lambda_offhn,
        "offhn_active": True,
        "offhn_num_valid": float(valid_mask.sum().item()),
        "offhn_avg_pos_sim": sim_pos[valid_mask].mean(),
        "offhn_avg_neg_sim": sim_neg[valid_mask].mean(),
        "offhn_avg_margin_gap": valid_margin_gap.mean(),
        "offhn_violation_rate": valid_margin_gap.gt(0).float().mean(),
        "offhn_lambda": zero.new_tensor(lambda_offhn),
        "offhn_margin": zero.new_tensor(margin),
        "offhn_temperature": zero.new_tensor(temperature),
    }


def _add_offline_hardneg_to_total_loss(args, total_loss, ret, epoch):
    if not bool(getattr(args, "use_offline_hardneg", False)):
        return total_loss
    if int(epoch) < int(getattr(args, "offline_hn_start_epoch", 1)):
        return total_loss
    if "loss_offhn_weighted" in ret:
        total_loss = total_loss + ret["loss_offhn_weighted"]
    elif "loss_offhn" in ret:
        total_loss = total_loss + float(
            getattr(args, "offline_hn_lambda", 0.05)
        ) * ret["loss_offhn"]
    return total_loss


def _resolve_proto_version(args):
    return str(getattr(args, "proto_version", "v1")).lower()


def _resolve_lambda_proto_self_v(args):
    value = getattr(args, "lambda_proto_self_v", None)
    if value is None:
        return float(getattr(args, "lambda_proto_v", 1.0))
    return float(value)


def _resolve_lambda_proto_self_t(args):
    value = getattr(args, "lambda_proto_self_t", None)
    if value is None:
        return float(getattr(args, "lambda_proto_t", 1.0))
    return float(value)


def _masked_prototype_ce(feats, prototypes, initialized_mask, pids, proto_temp):
    if feats is None or prototypes is None or initialized_mask is None or pids is None:
        if feats is not None:
            return feats.new_zeros(()), 0
        if prototypes is not None:
            return prototypes.new_zeros(()), 0
        if pids is not None:
            return torch.zeros((), device=pids.device), 0
        return torch.zeros(()), 0
    zero = feats.new_zeros(())
    if feats.dim() != 2 or prototypes.dim() != 2:
        raise ValueError(
            "Prototype CE requires [B, C] features and [K, C] prototypes, "
            f"got {tuple(feats.shape)} and {tuple(prototypes.shape)}."
        )
    if feats.shape[1] != prototypes.shape[1]:
        raise ValueError(
            "Prototype CE feature dimension mismatch: "
            f"{feats.shape[1]} vs {prototypes.shape[1]}."
        )

    num_classes = prototypes.shape[0]
    pids = pids.reshape(-1).to(device=feats.device).long()
    if pids.shape[0] != feats.shape[0]:
        raise ValueError(
            f"pids must match feature batch size, got {pids.shape[0]} and {feats.shape[0]}."
        )
    initialized_mask = initialized_mask.to(device=feats.device, dtype=torch.bool)
    valid_pid_mask = (pids >= 0) & (pids < num_classes)
    if not valid_pid_mask.any() or not initialized_mask.any():
        return zero, 0

    candidate_pids = pids[valid_pid_mask]
    candidate_feats = feats[valid_pid_mask]
    target_initialized = initialized_mask[candidate_pids]
    if not target_initialized.any():
        return zero, 0

    valid_feats = candidate_feats[target_initialized]
    valid_pids = candidate_pids[target_initialized]
    valid_feats = _safe_l2_normalize(valid_feats.float(), dim=1)
    prototypes = _safe_l2_normalize(
        prototypes.to(device=valid_feats.device).float(), dim=1
    )

    logits = valid_feats @ prototypes.t()
    logits = logits / float(proto_temp)
    logits = logits.masked_fill(~initialized_mask.unsqueeze(0), -1e4)
    return F.cross_entropy(logits, valid_pids), int(valid_pids.numel())


def _prototype_alignment_loss(
    visual_prototypes,
    text_prototypes,
    visual_initialized_mask,
    text_initialized_mask,
    pids,
):
    reference = visual_prototypes if visual_prototypes is not None else text_prototypes
    if reference is None:
        if pids is not None:
            return torch.zeros((), device=pids.device), 0, 0.0
        return torch.zeros(()), 0, 0.0
    zero = reference.new_zeros(())
    if (
        visual_prototypes is None
        or text_prototypes is None
        or visual_initialized_mask is None
        or text_initialized_mask is None
        or pids is None
    ):
        return zero, 0, 0.0

    num_classes = visual_prototypes.shape[0]
    pids = pids.reshape(-1).to(device=visual_prototypes.device).long()
    valid_pid_mask = (pids >= 0) & (pids < num_classes)
    if not valid_pid_mask.any():
        return zero, 0, 0.0

    unique_pids = torch.unique(pids[valid_pid_mask])
    both_initialized = (
        visual_initialized_mask.to(device=visual_prototypes.device, dtype=torch.bool)
        & text_initialized_mask.to(device=visual_prototypes.device, dtype=torch.bool)
    )
    unique_pids = unique_pids[both_initialized[unique_pids]]
    if unique_pids.numel() == 0:
        return zero, 0, 0.0

    visual_proto = _safe_l2_normalize(visual_prototypes.float(), dim=1)[unique_pids]
    text_proto = _safe_l2_normalize(
        text_prototypes.to(device=visual_prototypes.device).float(), dim=1
    )[unique_pids]
    cos = F.cosine_similarity(visual_proto, text_proto, dim=1)
    return (1.0 - cos).mean(), int(unique_pids.numel()), float(cos.mean().item())


def _compute_proto_calibration_terms_v2(args, model, ret, batch):
    target_model = _unwrap_runtime_model(model)
    state = _get_proto_calib_runtime_state(args, model)
    proto_out = {
        "proto_version": "v2",
        "proto_loss_active": float(state["proto_loss_active"]),
        "proto_init_loaded": float(state["proto_init_loaded"]),
        "proto_update_frozen": float(state["proto_update_frozen"]),
        "proto_num_valid_targets": 0.0,
    }

    if not (
        hasattr(target_model, "get_visual_prototype_matrix")
        and hasattr(target_model, "get_text_prototype_matrix")
    ):
        return proto_out

    visual_prototypes = target_model.get_visual_prototype_matrix()
    text_prototypes = target_model.get_text_prototype_matrix()
    if visual_prototypes is None or text_prototypes is None:
        return proto_out

    fused_feat = ret.get("fused_det_feat", None)
    text_feat = ret.get("text_det_feat", None)
    pids = batch.get("pids", None)
    if fused_feat is None or text_feat is None or pids is None:
        raise KeyError(
            "Prototype calibration v2 requires 'fused_det_feat', "
            "'text_det_feat', and batch['pids']."
        )

    zero = fused_feat.new_zeros(())
    visual_initialized = target_model.get_visual_prototype_initialized_mask()
    text_initialized = target_model.get_text_prototype_initialized_mask()
    if visual_initialized is None:
        visual_initialized = torch.zeros(
            visual_prototypes.shape[0], device=visual_prototypes.device, dtype=torch.bool
        )
    if text_initialized is None:
        text_initialized = torch.zeros(
            text_prototypes.shape[0], device=text_prototypes.device, dtype=torch.bool
        )
    visual_initialized = visual_initialized.to(
        device=visual_prototypes.device, dtype=torch.bool
    )
    text_initialized = text_initialized.to(
        device=text_prototypes.device, dtype=torch.bool
    )
    both_initialized = visual_initialized.to(device=text_initialized.device) & text_initialized
    proto_out.update(
        {
            "loss_proto_self_v": zero,
            "loss_proto_self_t": zero,
            "loss_proto_cross_t2v": zero,
            "loss_proto_cross_v2t": zero,
            "loss_proto_align": zero,
            "loss_proto_total": zero,
            "loss_proto_v": zero,
            "loss_proto_t": zero,
            "proto_v_initialized_count": float(visual_initialized.sum().item()),
            "proto_t_initialized_count": float(text_initialized.sum().item()),
            "proto_valid_classes": float(both_initialized.sum().item()),
            "proto_batch_unique_pids": 0.0,
            "proto_v_mean_norm": 0.0,
            "proto_t_mean_norm": 0.0,
            "proto_align_cos_mean": 0.0,
        }
    )

    flat_pids = pids.reshape(-1).to(device=fused_feat.device).long()
    valid_pid_mask = (flat_pids >= 0) & (flat_pids < visual_prototypes.shape[0])
    if valid_pid_mask.any():
        proto_out["proto_batch_unique_pids"] = float(
            torch.unique(flat_pids[valid_pid_mask]).numel()
        )
    if visual_initialized.any():
        proto_out["proto_v_mean_norm"] = float(
            visual_prototypes[visual_initialized].norm(p=2, dim=1).mean().item()
        )
    if text_initialized.any():
        proto_out["proto_t_mean_norm"] = float(
            text_prototypes[text_initialized].norm(p=2, dim=1).mean().item()
        )

    if not state["proto_loss_active"]:
        return proto_out

    proto_temp = float(getattr(args, "proto_temp", 0.07))
    if proto_temp <= 0:
        raise ValueError(f"proto_temp must be > 0, got {proto_temp}.")

    loss_self_v, valid_self_v = _masked_prototype_ce(
        fused_feat, visual_prototypes, visual_initialized, pids, proto_temp
    )
    loss_self_t, valid_self_t = _masked_prototype_ce(
        text_feat, text_prototypes, text_initialized, pids, proto_temp
    )
    loss_cross_t2v, valid_cross_t2v = _masked_prototype_ce(
        text_feat, visual_prototypes, visual_initialized, pids, proto_temp
    )
    loss_cross_v2t, valid_cross_v2t = _masked_prototype_ce(
        fused_feat, text_prototypes, text_initialized, pids, proto_temp
    )
    loss_align, _, align_cos_mean = _prototype_alignment_loss(
        visual_prototypes=visual_prototypes,
        text_prototypes=text_prototypes,
        visual_initialized_mask=visual_initialized,
        text_initialized_mask=text_initialized,
        pids=pids,
    )

    lambda_self_v = _resolve_lambda_proto_self_v(args)
    lambda_self_t = _resolve_lambda_proto_self_t(args)
    lambda_cross_t2v = float(getattr(args, "lambda_proto_cross_t2v", 0.0))
    lambda_cross_v2t = float(getattr(args, "lambda_proto_cross_v2t", 0.0))
    lambda_align = float(getattr(args, "lambda_proto_align", 0.0))
    loss_total = (
        lambda_self_v * loss_self_v
        + lambda_self_t * loss_self_t
        + lambda_cross_t2v * loss_cross_t2v
        + lambda_cross_v2t * loss_cross_v2t
        + lambda_align * loss_align
    )

    proto_out.update(
        {
            "loss_proto_self_v": loss_self_v,
            "loss_proto_self_t": loss_self_t,
            "loss_proto_cross_t2v": loss_cross_t2v,
            "loss_proto_cross_v2t": loss_cross_v2t,
            "loss_proto_align": loss_align,
            "loss_proto_total": loss_total,
            # Legacy meter aliases.
            "loss_proto_v": loss_self_v,
            "loss_proto_t": loss_self_t,
            "proto_num_valid_targets": float(
                max(valid_self_v, valid_self_t, valid_cross_t2v, valid_cross_v2t)
            ),
            "proto_align_cos_mean": align_cos_mean,
        }
    )
    return proto_out


def _compute_proto_calibration_terms(args, model, ret, batch):
    if not bool(getattr(args, "use_proto_calib", False)):
        return {}
    if _resolve_proto_version(args) == "v2":
        return _compute_proto_calibration_terms_v2(args, model, ret, batch)

    target_model = _unwrap_runtime_model(model)
    state = _get_proto_calib_runtime_state(args, model)
    proto_out = {
        "proto_version": "v1",
        "proto_loss_active": float(state["proto_loss_active"]),
        "proto_init_loaded": float(state["proto_init_loaded"]),
        "proto_update_frozen": float(state["proto_update_frozen"]),
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
    initialized_count = float(initialized_mask.sum().item())
    proto_out.update(
        {
            "loss_proto_v": zero,
            "loss_proto_t": zero,
            "loss_proto_self_v": zero,
            "loss_proto_self_t": zero,
            "loss_proto_cross_t2v": zero,
            "loss_proto_cross_v2t": zero,
            "loss_proto_align": zero,
            "loss_proto_total": zero,
            "proto_v_initialized_count": initialized_count,
            "proto_t_initialized_count": 0.0,
            "proto_valid_classes": initialized_count,
            "proto_batch_unique_pids": 0.0,
            "proto_v_mean_norm": 0.0,
            "proto_t_mean_norm": 0.0,
            "proto_align_cos_mean": 0.0,
        }
    )
    flat_pids = pids.reshape(-1).to(device=fused_feat.device).long()
    valid_pid_mask_for_stats = (
        (flat_pids >= 0) & (flat_pids < prototype_matrix.shape[0])
    )
    if valid_pid_mask_for_stats.any():
        proto_out["proto_batch_unique_pids"] = float(
            torch.unique(flat_pids[valid_pid_mask_for_stats]).numel()
        )
    if initialized_mask.any():
        proto_out["proto_v_mean_norm"] = float(
            prototype_matrix[initialized_mask].norm(p=2, dim=1).mean().item()
        )
    if not state["proto_loss_active"]:
        return proto_out

    fused_feat = _safe_l2_normalize(fused_feat.float(), dim=1)
    text_feat = _safe_l2_normalize(text_feat.float(), dim=1)
    prototype_matrix = _safe_l2_normalize(prototype_matrix.float(), dim=1)
    pids = pids.long()

    proto_temp = float(getattr(args, "proto_temp", 0.07))
    if proto_temp <= 0:
        raise ValueError(f"proto_temp must be > 0, got {proto_temp}.")

    valid_target_mask = initialized_mask[pids]
    proto_out["proto_num_valid_targets"] = float(valid_target_mask.sum().item())
    if not valid_target_mask.any():
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

    loss_proto_v = F.cross_entropy(logits_v2p, valid_pids)
    loss_proto_t = F.cross_entropy(logits_t2p, valid_pids)
    loss_proto_total = (
        float(getattr(args, "lambda_proto_v", 1.0)) * loss_proto_v
        + float(getattr(args, "lambda_proto_t", 1.0)) * loss_proto_t
    )
    proto_out.update(
        {
            "loss_proto_v": loss_proto_v,
            "loss_proto_t": loss_proto_t,
            "loss_proto_self_v": loss_proto_v,
            "loss_proto_self_t": loss_proto_t,
            "loss_proto_total": loss_proto_total,
        }
    )
    return proto_out


def _add_proto_calibration_to_total_loss(args, total_loss, ret):
    if not bool(getattr(args, "use_proto_calib", False)):
        return total_loss
    if _resolve_proto_version(args) == "v2":
        if "loss_proto_total" in ret:
            total_loss = total_loss + ret["loss_proto_total"]
        return total_loss
    if "loss_proto_v" in ret:
        total_loss = total_loss + float(getattr(args, "lambda_proto_v", 1.0)) * ret["loss_proto_v"]
    if "loss_proto_t" in ret:
        total_loss = total_loss + float(getattr(args, "lambda_proto_t", 1.0)) * ret["loss_proto_t"]
    return total_loss


def update_model_proto_bank_after_step(model, ret, batch):
    target_model = _unwrap_runtime_model(model)
    empty_info = {
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
        "proto_init_loaded": 0.0,
        "proto_update_frozen": 0.0,
    }
    empty_info["proto_init_loaded"] = float(
        bool(getattr(target_model, "proto_init_loaded", False))
    )
    if hasattr(target_model, "is_proto_update_frozen"):
        empty_info["proto_update_frozen"] = float(
            bool(target_model.is_proto_update_frozen())
        )
    proto_version = str(getattr(target_model, "proto_version", "v1")).lower()
    fused_feat = ret.get("fused_det_feat", None)
    if fused_feat is None:
        return empty_info
    if proto_version == "v2":
        if not hasattr(target_model, "update_dual_prototype_bank"):
            return empty_info
        text_feat = ret.get("text_det_feat", None)
        if text_feat is None:
            return empty_info
        update_info = target_model.update_dual_prototype_bank(
            visual_feats=fused_feat,
            text_feats=text_feat,
            pids=batch.get("pids", None),
        )
        update_info.setdefault("proto_init_loaded", empty_info["proto_init_loaded"])
        update_info.setdefault("proto_update_frozen", empty_info["proto_update_frozen"])
        return update_info
    if not hasattr(target_model, "update_visual_prototype_bank"):
        return empty_info
    update_info = target_model.update_visual_prototype_bank(
        fused_feats=fused_feat,
        pids=batch.get("pids", None),
    )
    update_info.setdefault("proto_init_loaded", empty_info["proto_init_loaded"])
    update_info.setdefault("proto_update_frozen", empty_info["proto_update_frozen"])
    return update_info


def _mean_positive_pairwise_distance(dist_matrix, pids=None, sample_weight=None):
    """
    Average pairwise distance over positive matches.

    If pids are provided, positives are defined by pid equality (same as SDM label logic).
    Otherwise, fallback to diagonal matched pairs (index-aligned assumption).
    """
    if pids is None:
        per_sample_dist = torch.diagonal(dist_matrix)
        if sample_weight is None:
            return per_sample_dist.mean()
        sample_weight = sample_weight.to(
            device=per_sample_dist.device,
            dtype=per_sample_dist.dtype,
        ).reshape(-1)
        weight_sum = sample_weight.sum().clamp_min(1e-8)
        return (per_sample_dist * sample_weight).sum() / weight_sum

    pids = pids.reshape(-1)
    pos_mask = pids.unsqueeze(1).eq(pids.unsqueeze(0))
    if pos_mask.any():
        if sample_weight is None:
            return dist_matrix[pos_mask].mean()
        sample_weight = sample_weight.to(
            device=dist_matrix.device,
            dtype=dist_matrix.dtype,
        ).reshape(-1)
        if sample_weight.shape[0] != dist_matrix.shape[0]:
            raise ValueError(
                "sample_weight for uncertainty loss must have shape [batch_size], "
                f"got {tuple(sample_weight.shape)} for batch_size={dist_matrix.shape[0]}."
            )
        pos_count = pos_mask.sum(dim=1)
        valid_query = pos_count.gt(0)
        per_query_dist = (
            dist_matrix.masked_fill(~pos_mask, 0.0).sum(dim=1)
            / pos_count.clamp_min(1).to(dist_matrix.dtype)
        )
        effective_weight = sample_weight * valid_query.to(sample_weight.dtype)
        weight_sum = effective_weight.sum().clamp_min(1e-8)
        return (per_query_dist * effective_weight).sum() / weight_sum

    per_sample_dist = torch.diagonal(dist_matrix)
    if sample_weight is None:
        return per_sample_dist.mean()
    sample_weight = sample_weight.to(
        device=per_sample_dist.device,
        dtype=per_sample_dist.dtype,
    ).reshape(-1)
    weight_sum = sample_weight.sum().clamp_min(1e-8)
    return (per_sample_dist * sample_weight).sum() / weight_sum


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
    caption_weight_for_unc = _get_caption_weight_for_loss(
        args,
        batch,
        "synth_weight_apply_to_unc",
        default=False,
    )

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
    loss_unc_text_rgb = _mean_positive_pairwise_distance(
        tr_dist,
        pids,
        sample_weight=caption_weight_for_unc,
    )

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
        loss_unc_text_thermal = _mean_positive_pairwise_distance(
            tt_dist,
            pids,
            sample_weight=caption_weight_for_unc,
        )
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

    arguments = {"num_epoch": num_epoch, "iteration": 0, "epoch": 0}

    logger = logging.getLogger("IRRA.train")
    logger.info("start training")
    log_img_local_filter_v1_train_config(logger, args)
    log_img_local_filter_train_config(logger, args)
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
        "loss_rgb_txt": AverageMeter(),
        "loss_th_txt": AverageMeter(),
        "loss_txt_cons": AverageMeter(),
        "effective_lambda_rgb_txt": AverageMeter(),
        "effective_lambda_th_txt": AverageMeter(),
        "effective_lambda_txt_cons": AverageMeter(),
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
        "loss_hn_weighted": AverageMeter(),
        "hn_active": AverageMeter(),
        "hn_num_valid_queries": AverageMeter(),
        "hn_avg_pos_sim": AverageMeter(),
        "hn_avg_neg_sim": AverageMeter(),
        "hn_avg_margin_gap": AverageMeter(),
        "hn_violation_rate": AverageMeter(),
        "hn_t2i_loss": AverageMeter(),
        "hn_i2t_loss": AverageMeter(),
        "hn_t2i_valid": AverageMeter(),
        "hn_i2t_valid": AverageMeter(),
        "hn_avg_top1_neg_sim": AverageMeter(),
        "hn_avg_topk_neg_sim": AverageMeter(),
        "loss_offhn": AverageMeter(),
        "loss_offhn_weighted": AverageMeter(),
        "offhn_active": AverageMeter(),
        "offhn_num_valid": AverageMeter(),
        "offhn_avg_pos_sim": AverageMeter(),
        "offhn_avg_neg_sim": AverageMeter(),
        "offhn_avg_margin_gap": AverageMeter(),
        "offhn_violation_rate": AverageMeter(),
        "offhn_lambda": AverageMeter(),
        "offhn_margin": AverageMeter(),
        "offhn_temperature": AverageMeter(),
        "loss_proto_v": AverageMeter(),
        "loss_proto_t": AverageMeter(),
        "loss_proto_self_v": AverageMeter(),
        "loss_proto_self_t": AverageMeter(),
        "loss_proto_cross_t2v": AverageMeter(),
        "loss_proto_cross_v2t": AverageMeter(),
        "loss_proto_align": AverageMeter(),
        "loss_proto_total": AverageMeter(),
        "proto_loss_active": AverageMeter(),
        "proto_init_loaded": AverageMeter(),
        "proto_update_frozen": AverageMeter(),
        "proto_num_valid_targets": AverageMeter(),
        "proto_v_initialized_count": AverageMeter(),
        "proto_t_initialized_count": AverageMeter(),
        "proto_valid_classes": AverageMeter(),
        "proto_batch_unique_pids": AverageMeter(),
        "proto_v_mean_norm": AverageMeter(),
        "proto_t_mean_norm": AverageMeter(),
        "proto_align_cos_mean": AverageMeter(),
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
        "noise_mask_avg_tokens": AverageMeter(),
        "noise_mask_valid_ratio": AverageMeter(),
        "info_mask_avg_tokens": AverageMeter(),
        "info_mask_valid_ratio": AverageMeter(),
        "noise_info_overlap_ratio": AverageMeter(),
        "guided_mtp_valid_tokens": AverageMeter(),
        "guided_mtp_lambda": AverageMeter(),
        "token_score_mean": AverageMeter(),
        "token_score_var": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_top1 = 0.0
    mtp_text_example_logged = False
    use_modality_text_prompt = bool(
        getattr(args, "use_modality_text_prompt", False)
    )
    mtp_text_tokenizer = (
        SimpleTokenizer() if use_modality_text_prompt else None
    )

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
        log_mtp_schedule_state(logger, args, epoch)

        for n_iter, batch in enumerate(train_loader):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            if use_modality_text_prompt and (not mtp_text_example_logged):
                full_clean_example = _decode_caption_token_ids_for_log(
                    batch.get("caption_ids_full_clean", batch["caption_ids"])[0],
                    mtp_text_tokenizer,
                )
                if "caption_ids_rgb" in batch:
                    rgb_example = _decode_caption_token_ids_for_log(
                        batch["caption_ids_rgb"][0], mtp_text_tokenizer
                    )
                else:
                    rgb_example = build_rgb_text(full_clean_example)
                if "caption_ids_th" in batch:
                    th_example = _decode_caption_token_ids_for_log(
                        batch["caption_ids_th"][0], mtp_text_tokenizer
                    )
                else:
                    th_example = build_thermal_text(
                        full_clean_example,
                        color_drop_mode=str(
                            getattr(args, "mtp_color_drop_mode", "delete")
                        ),
                    )
                logger.info(
                    "[MTP][TEXT_EXAMPLE] full_clean=%s",
                    _truncate_log_text(full_clean_example),
                )
                logger.info(
                    "[MTP][TEXT_EXAMPLE] rgb=%s",
                    _truncate_log_text(rgb_example),
                )
                logger.info(
                    "[MTP][TEXT_EXAMPLE] th=%s",
                    _truncate_log_text(th_example),
                )
                mtp_text_example_logged = True

            ret = model(batch)
            ret.update(_compute_uncertainty_alignment_terms(args, ret, batch))
            original_total_loss = _build_total_loss(args, ret)
            ret.update(_compute_rgbt_moe_adapter_terms(args, ret, batch, epoch))
            if getattr(args, "use_hardneg_aux", False):
                ret.update(_compute_hardneg_aux_terms(args, ret, batch, epoch))
            if getattr(args, "use_offline_hardneg", False):
                ret.update(_compute_offline_hardneg_terms(args, ret, batch, epoch))
            ret.update(_compute_proto_calibration_terms(args, model, ret, batch))
            ret.update(_compute_mtp_text_prompt_terms(args, ret, batch, epoch))
            total_loss = _add_rgbt_moe_adapter_terms_to_total_loss(
                args, original_total_loss, ret, epoch
            )
            total_loss = _add_proto_calibration_to_total_loss(args, total_loss, ret)
            if getattr(args, "use_hardneg_aux", False):
                total_loss = _add_hardneg_aux_to_total_loss(
                    args, total_loss, ret, epoch
                )
            if getattr(args, "use_offline_hardneg", False):
                total_loss = _add_offline_hardneg_to_total_loss(
                    args, total_loss, ret, epoch
                )
            total_loss = _add_mtp_terms_to_total_loss(args, total_loss, ret, epoch)

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
            if "loss_rgb_txt" in ret:
                meters["loss_rgb_txt"].update(
                    _to_scalar(ret["loss_rgb_txt"]), batch_size
                )
            if "loss_th_txt" in ret:
                meters["loss_th_txt"].update(
                    _to_scalar(ret["loss_th_txt"]), batch_size
                )
            if "loss_txt_cons" in ret:
                meters["loss_txt_cons"].update(
                    _to_scalar(ret["loss_txt_cons"]), batch_size
                )
            if "effective_lambda_rgb_txt" in ret:
                meters["effective_lambda_rgb_txt"].update(
                    _to_scalar(ret["effective_lambda_rgb_txt"]), batch_size
                )
            if "effective_lambda_th_txt" in ret:
                meters["effective_lambda_th_txt"].update(
                    _to_scalar(ret["effective_lambda_th_txt"]), batch_size
                )
            if "effective_lambda_txt_cons" in ret:
                meters["effective_lambda_txt_cons"].update(
                    _to_scalar(ret["effective_lambda_txt_cons"]), batch_size
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
                meters["loss_hn_weighted"].update(
                    _to_scalar(ret.get("loss_hn_weighted", 0)), batch_size
                )
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
                meters["hn_violation_rate"].update(
                    _to_scalar(ret.get("hn_violation_rate", 0)), batch_size
                )
                meters["hn_t2i_loss"].update(
                    _to_scalar(ret.get("hn_t2i_loss", 0)), batch_size
                )
                meters["hn_i2t_loss"].update(
                    _to_scalar(ret.get("hn_i2t_loss", 0)), batch_size
                )
                meters["hn_t2i_valid"].update(_to_scalar(ret.get("hn_t2i_valid", 0)), 1)
                meters["hn_i2t_valid"].update(_to_scalar(ret.get("hn_i2t_valid", 0)), 1)
                meters["hn_avg_top1_neg_sim"].update(
                    _to_scalar(ret.get("hn_avg_top1_neg_sim", 0)), batch_size
                )
                meters["hn_avg_topk_neg_sim"].update(
                    _to_scalar(ret.get("hn_avg_topk_neg_sim", 0)), batch_size
                )
            if getattr(args, "use_offline_hardneg", False):
                meters["loss_offhn"].update(
                    _to_scalar(ret.get("loss_offhn", 0)), batch_size
                )
                meters["loss_offhn_weighted"].update(
                    _to_scalar(ret.get("loss_offhn_weighted", 0)), batch_size
                )
                meters["offhn_active"].update(
                    _to_scalar(ret.get("offhn_active", 0)), 1
                )
                meters["offhn_num_valid"].update(
                    _to_scalar(ret.get("offhn_num_valid", 0)), 1
                )
                meters["offhn_avg_pos_sim"].update(
                    _to_scalar(ret.get("offhn_avg_pos_sim", 0)), batch_size
                )
                meters["offhn_avg_neg_sim"].update(
                    _to_scalar(ret.get("offhn_avg_neg_sim", 0)), batch_size
                )
                meters["offhn_avg_margin_gap"].update(
                    _to_scalar(ret.get("offhn_avg_margin_gap", 0)), batch_size
                )
                meters["offhn_violation_rate"].update(
                    _to_scalar(ret.get("offhn_violation_rate", 0)), batch_size
                )
                meters["offhn_lambda"].update(
                    _to_scalar(ret.get("offhn_lambda", 0)), 1
                )
                meters["offhn_margin"].update(
                    _to_scalar(ret.get("offhn_margin", 0)), 1
                )
                meters["offhn_temperature"].update(
                    _to_scalar(ret.get("offhn_temperature", 0)), 1
                )
            meters["loss_proto_v"].update(
                _to_scalar(ret.get("loss_proto_v", 0)), batch_size
            )
            meters["loss_proto_t"].update(
                _to_scalar(ret.get("loss_proto_t", 0)), batch_size
            )
            if bool(getattr(args, "use_proto_calib", False)):
                for proto_meter_name in [
                    "loss_proto_self_v",
                    "loss_proto_self_t",
                    "loss_proto_cross_t2v",
                    "loss_proto_cross_v2t",
                    "loss_proto_align",
                    "loss_proto_total",
                    "proto_init_loaded",
                    "proto_update_frozen",
                    "proto_v_initialized_count",
                    "proto_t_initialized_count",
                    "proto_valid_classes",
                    "proto_batch_unique_pids",
                    "proto_v_mean_norm",
                    "proto_t_mean_norm",
                    "proto_align_cos_mean",
                ]:
                    meters[proto_meter_name].update(
                        _to_scalar(ret.get(proto_meter_name, 0)), 1
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
            if bool(getattr(args, "debug_print_token_mask_stats", False)):
                if "noise_mask_avg_tokens" in ret:
                    meters["noise_mask_avg_tokens"].update(
                        _to_scalar(ret["noise_mask_avg_tokens"]), batch_size
                    )
                if "noise_mask_valid_ratio" in ret:
                    meters["noise_mask_valid_ratio"].update(
                        _to_scalar(ret["noise_mask_valid_ratio"]), batch_size
                    )
                if "info_mask_avg_tokens" in ret:
                    meters["info_mask_avg_tokens"].update(
                        _to_scalar(ret["info_mask_avg_tokens"]), batch_size
                    )
                if "info_mask_valid_ratio" in ret:
                    meters["info_mask_valid_ratio"].update(
                        _to_scalar(ret["info_mask_valid_ratio"]), batch_size
                    )
                if "noise_info_overlap_ratio" in ret:
                    meters["noise_info_overlap_ratio"].update(
                        _to_scalar(ret["noise_info_overlap_ratio"]), batch_size
                    )
                if "guided_mtp_valid_tokens" in ret:
                    meters["guided_mtp_valid_tokens"].update(
                        _to_scalar(ret["guided_mtp_valid_tokens"]), 1
                    )
                if "guided_mtp_lambda" in ret:
                    meters["guided_mtp_lambda"].update(
                        _to_scalar(ret["guided_mtp_lambda"]), 1
                    )
                if "token_score_mean" in ret:
                    meters["token_score_mean"].update(
                        _to_scalar(ret["token_score_mean"]), batch_size
                    )
                if "token_score_var" in ret:
                    meters["token_score_var"].update(
                        _to_scalar(ret["token_score_var"]), batch_size
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
                hardneg_debug_meter_names = {
                    "hn_t2i_loss",
                    "hn_i2t_loss",
                    "hn_t2i_valid",
                    "hn_i2t_valid",
                    "hn_avg_top1_neg_sim",
                    "hn_avg_topk_neg_sim",
                }
                proto_debug_meter_names = {
                    "proto_batch_unique_pids",
                    "proto_v_mean_norm",
                    "proto_t_mean_norm",
                    "proto_align_cos_mean",
                }
                debug_hardneg_stats = bool(getattr(args, "debug_hardneg_stats", False))
                debug_proto_stats = bool(getattr(args, "debug_proto_stats", False))
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
                        if (not debug_hardneg_stats) and k in hardneg_debug_meter_names:
                            continue
                        if (not debug_proto_stats) and k in proto_debug_meter_names:
                            continue
                        info_str += f", {k}: {v.avg:.4f}"
                active_layers = ret.get("shared_consistency_active_layers", "none")
                info_str += f", shared_consistency_active_layers: {active_layers}"
                if bool(getattr(args, "use_proto_calib", False)):
                    info_str += (
                        f", proto_version: {ret.get('proto_version', getattr(args, 'proto_version', 'v1'))}"
                    )
                if bool(getattr(args, "use_hardneg_aux", False)):
                    info_str += (
                        f", hn_version: {ret.get('hn_version', getattr(args, 'hardneg_version', 'v1'))}"
                    )
                    info_str += (
                        f", hn_direction: {ret.get('hn_direction', getattr(args, 'hn_direction', 'both'))}"
                    )
                if bool(getattr(args, "debug_print_token_mask_stats", False)):
                    info_str += (
                        f", noise_mask_mode: {ret.get('noise_mask_mode', 'none')}"
                    )
                    info_str += (
                        f", info_mask_mode: {ret.get('info_mask_mode', 'none')}"
                    )
                if bool(getattr(args, "debug_print_synth_weight_stats", False)):
                    caption_is_synth = batch.get("caption_is_synth", None)
                    caption_weight = batch.get("caption_weight", None)
                    if caption_is_synth is not None and caption_weight is not None:
                        caption_is_synth = caption_is_synth.reshape(-1)
                        caption_weight = caption_weight.float().reshape(-1)
                        info_str += (
                            f", [SYNTH_WEIGHT][BATCH] num_synth="
                            f"{int(caption_is_synth.float().sum().item())}"
                        )
                        info_str += (
                            f" mean_w="
                            f"{caption_weight.mean().item():.4f}"
                        )
                        info_str += (
                            f" min_w="
                            f"{caption_weight.min().item():.4f}"
                        )
                        info_str += (
                            f" max_w="
                            f"{caption_weight.max().item():.4f}"
                        )
                    else:
                        info_str += ", [SYNTH_WEIGHT][BATCH] missing"
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
                logger=None,
                prefix="VALIDATION",
                runtime=epoch_img_local_runtime,
            )
            log_img_local_filter_eval_state(logger, model, prefix="VALIDATION")
            logger.info(
                "[MTP][VALIDATION] enabled=%s",
                bool(getattr(args, "use_modality_text_prompt", False)),
            )
            logger.info(
                "[MTP][VALIDATION] alpha=%.6f",
                float(getattr(args, "mtp_eval_fusion_alpha", 0.10)),
            )
            logger.info(
                "[MTP][VALIDATION] beta=%.6f",
                float(getattr(args, "mtp_eval_fusion_beta", 0.00)),
            )
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
    model_args = getattr(model, "args", None)
    logger.info(
        "[MTP][VALIDATION] enabled=%s",
        bool(getattr(model_args, "use_modality_text_prompt", False)),
    )
    logger.info(
        "[MTP][VALIDATION] alpha=%.6f",
        float(getattr(model_args, "mtp_eval_fusion_alpha", 0.10)),
    )
    logger.info(
        "[MTP][VALIDATION] beta=%.6f",
        float(getattr(model_args, "mtp_eval_fusion_beta", 0.00)),
    )
    log_img_local_filter_eval_state(logger, model, prefix="INFERENCE")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    evaluator.eval(model.eval())
