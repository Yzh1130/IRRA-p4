import hashlib
import json
import logging
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from datasets.bases import tokenize
from model import build_model, objectives
from model.objectives import (
    compute_dcalign_lite_loss,
    compute_uspc_loss,
    resolve_dcalign_lite_lambda,
)
from utils.attribute_parser import (
    get_uspc_slot_local_attr_names,
    parse_basic_attributes,
)
from utils.checkpoint import load_state_dict
from utils.comm import get_rank, synchronize
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.mdrc import (
    ALLOWED_DEGRADE_MODES,
    degrade_rgbt_batch,
    mdrc_feature_consistency,
)
from utils.mtp_text import build_rgb_text, build_thermal_text
from utils.simple_tokenizer import SimpleTokenizer
from utils.uspc_text_views import build_uspc_text_views


EVAL_STATE_KEYWORDS = (
    "proto",
    "prototype",
    "unc",
    "uncertainty",
    "logvar",
    "fusion",
    "fused",
)

EVAL_RUNTIME_ATTR_NAMES = (
    "proto_runtime_epoch",
    "proto_init_loaded",
    "proto_init_visual_loaded",
    "proto_init_text_loaded",
    "proto_init_load_info",
    "img_local_filter_active",
    "effective_img_local_fuse_gamma",
    "eval_use_fused_mu",
    "eval_use_mu_only",
    "use_precision_fusion",
    "use_global_uncertainty",
    "use_rgbt_add_fusion",
    "use_batch_uncertainty",
    "use_id_uncertainty",
    "use_prob_sampling",
)

_MGA_WARNED_KEYS = set()


def _to_scalar(value):
    # Convert tensor/number into python float for stable meter/log writing.
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().mean().item()
    return float(value)


def _resolve_metric_ckpt_dir(args):
    metric_ckpt_dir = str(getattr(args, "metric_ckpt_dir", "") or "").strip()
    if metric_ckpt_dir:
        return metric_ckpt_dir
    return os.path.join(args.output_dir, "metric_ckpts")


def _json_safe_metadata_value(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {
            str(k): _json_safe_metadata_value(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_hash(value):
    encoded = json.dumps(
        _json_safe_metadata_value(value),
        ensure_ascii=True,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _args_to_plain_dict(args):
    if args is None:
        return {}
    if hasattr(args, "items"):
        source = dict(args)
    elif hasattr(args, "__dict__"):
        source = vars(args)
    else:
        return {}
    return {
        str(k): _json_safe_metadata_value(v)
        for k, v in source.items()
    }


def _loader_drop_last(loader):
    if hasattr(loader, "drop_last"):
        return bool(loader.drop_last)
    batch_sampler = getattr(loader, "batch_sampler", None)
    if hasattr(batch_sampler, "drop_last"):
        return bool(batch_sampler.drop_last)
    return False


def _loader_shuffle(loader):
    sampler = getattr(loader, "sampler", None)
    sampler_name = sampler.__class__.__name__ if sampler is not None else "None"
    return sampler_name in {"RandomSampler", "RandomIdentitySampler", "RandomIdentitySampler_DDP"}


def _dataset_image_paths(dataset):
    if hasattr(dataset, "img_paths"):
        return list(getattr(dataset, "img_paths"))
    paths = []
    if hasattr(dataset, "img_paths_rgb"):
        paths.extend(list(getattr(dataset, "img_paths_rgb")))
    if hasattr(dataset, "img_paths_t"):
        paths.extend(list(getattr(dataset, "img_paths_t")))
    return paths


def _eval_order_hashes(img_loader, txt_loader):
    img_dataset = getattr(img_loader, "dataset", None)
    txt_dataset = getattr(txt_loader, "dataset", None)
    image_pids = list(getattr(img_dataset, "image_pids", []))
    caption_pids = list(getattr(txt_dataset, "caption_pids", []))
    img_paths = _dataset_image_paths(img_dataset)
    captions = list(getattr(txt_dataset, "captions", []))
    return {
        "image_ids_hash": _stable_hash(image_pids),
        "caption_ids_hash": _stable_hash(caption_pids),
        "pid_hash": _stable_hash({"image_pids": image_pids, "caption_pids": caption_pids}),
        "img_path_hash": _stable_hash(img_paths),
        "caption_text_hash": _stable_hash(captions),
    }


def log_eval_pipeline_summary(logger, source, function, img_loader, txt_loader, args):
    img_dataset = getattr(img_loader, "dataset", None)
    txt_dataset = getattr(txt_loader, "dataset", None)
    img_sampler = getattr(img_loader, "sampler", None)
    txt_sampler = getattr(txt_loader, "sampler", None)
    image_pids = list(getattr(img_dataset, "image_pids", []))
    caption_pids = list(getattr(txt_dataset, "caption_pids", []))
    split = (
        str(getattr(args, "val_dataset", "test"))
        if source == "train_validation"
        else str(getattr(args, "eval_dataset", "test"))
    )
    num_ids = len(set(image_pids) | set(caption_pids))
    logger.info(
        "[EVAL_PIPELINE] source=%s function=%s split=%s loader_len=%s/%s batch_size=%s/%s shuffle=%s/%s drop_last=%s/%s sampler=%s/%s dataset_class=%s/%s num_images=%d num_captions=%d num_ids=%d",
        source,
        function,
        split,
        len(img_loader),
        len(txt_loader),
        getattr(img_loader, "batch_size", None),
        getattr(txt_loader, "batch_size", None),
        _loader_shuffle(img_loader),
        _loader_shuffle(txt_loader),
        _loader_drop_last(img_loader),
        _loader_drop_last(txt_loader),
        img_sampler.__class__.__name__ if img_sampler is not None else "None",
        txt_sampler.__class__.__name__ if txt_sampler is not None else "None",
        img_dataset.__class__.__name__ if img_dataset is not None else "None",
        txt_dataset.__class__.__name__ if txt_dataset is not None else "None",
        len(image_pids),
        len(caption_pids),
        num_ids,
    )
    logger.info(
        "[EVAL_PIPELINE_FLAGS] source=%s args_training=%s eval_use_fused_mu=%s eval_use_mu_only=%s use_precision_fusion=%s use_global_uncertainty=%s use_rgbt_add_fusion=%s use_batch_uncertainty=%s use_id_uncertainty=%s use_qalr=%s use_transport_rerank=%s use_modality_text_prompt=%s",
        source,
        bool(getattr(args, "training", False)),
        bool(getattr(args, "eval_use_fused_mu", True)),
        bool(getattr(args, "eval_use_mu_only", False)),
        getattr(args, "use_precision_fusion", None),
        bool(getattr(args, "use_global_uncertainty", False)),
        bool(getattr(args, "use_rgbt_add_fusion", True)),
        bool(getattr(args, "use_batch_uncertainty", True)),
        bool(getattr(args, "use_id_uncertainty", True)),
        bool(getattr(args, "use_qalr", False)),
        bool(getattr(args, "use_transport_rerank", False)),
        bool(getattr(args, "use_modality_text_prompt", False)),
    )
    hashes = _eval_order_hashes(img_loader, txt_loader)
    logger.info(
        "[EVAL_ORDER_HASH] source=%s image_ids_hash=%s caption_ids_hash=%s pid_hash=%s img_path_hash=%s caption_text_hash=%s",
        source,
        hashes["image_ids_hash"],
        hashes["caption_ids_hash"],
        hashes["pid_hash"],
        hashes["img_path_hash"],
        hashes["caption_text_hash"],
    )


def _has_eval_state_keyword(name):
    lower_name = str(name).lower()
    return any(keyword in lower_name for keyword in EVAL_STATE_KEYWORDS)


def _keyword_state_keys(state_dict):
    return [
        str(k)
        for k in state_dict.keys()
        if _has_eval_state_keyword(k)
    ]


def _get_target_state_dict(model):
    target_model = _unwrap_runtime_model(model)
    return target_model.state_dict()


def _collect_plain_tensor_attrs(model):
    target_model = _unwrap_runtime_model(model)
    state_dict_keys = set(target_model.state_dict().keys())
    plain_tensors = {}
    plain_tensor_shapes = {}
    for module_name, module in target_model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for attr_name, value in module.__dict__.items():
            if attr_name.startswith("_"):
                continue
            full_name = f"{prefix}{attr_name}"
            if full_name in state_dict_keys:
                continue
            if torch.is_tensor(value):
                plain_tensors[full_name] = value.detach().cpu()
                plain_tensor_shapes[full_name] = list(value.shape)
    return plain_tensors, plain_tensor_shapes


def _collect_eval_extra_state(model):
    target_model = _unwrap_runtime_model(model)
    runtime_attrs = {}
    for attr_name in EVAL_RUNTIME_ATTR_NAMES:
        if hasattr(target_model, attr_name):
            runtime_attrs[attr_name] = _json_safe_metadata_value(
                getattr(target_model, attr_name)
            )

    plain_tensors, plain_tensor_shapes = _collect_plain_tensor_attrs(model)
    return {
        "runtime_attrs": runtime_attrs,
        "plain_tensor_attrs": plain_tensors,
        "plain_tensor_attr_shapes": plain_tensor_shapes,
        "keyword_state_dict_keys": _keyword_state_keys(_get_target_state_dict(model)),
    }


def _set_plain_tensor_attr(model, full_name, value):
    target_model = _unwrap_runtime_model(model)
    module_name, _, attr_name = str(full_name).rpartition(".")
    modules = dict(target_model.named_modules())
    module = modules.get(module_name, None) if module_name else target_model
    if module is None or not attr_name:
        return False
    if not torch.is_tensor(value):
        return False

    current_value = getattr(module, attr_name, None)
    restored_value = value.detach().clone()
    if torch.is_tensor(current_value):
        restored_value = restored_value.to(
            device=current_value.device,
            dtype=current_value.dtype,
        )
    setattr(module, attr_name, restored_value)
    return True


def restore_eval_extra_state(model, extra_state, logger=None):
    if not isinstance(extra_state, dict):
        return {"runtime_attrs": 0, "plain_tensor_attrs": 0}

    target_model = _unwrap_runtime_model(model)
    restored_runtime = 0
    for attr_name, value in extra_state.get("runtime_attrs", {}).items():
        if attr_name in EVAL_RUNTIME_ATTR_NAMES or hasattr(target_model, attr_name):
            setattr(target_model, attr_name, value)
            restored_runtime += 1

    restored_plain_tensors = 0
    for full_name, value in extra_state.get("plain_tensor_attrs", {}).items():
        if _set_plain_tensor_attr(model, full_name, value):
            restored_plain_tensors += 1

    if logger is not None:
        logger.info(
            "[FULL_STATE_EXTRA_RESTORE] runtime_attrs=%d plain_tensor_attrs=%d",
            restored_runtime,
            restored_plain_tensors,
        )
    return {
        "runtime_attrs": restored_runtime,
        "plain_tensor_attrs": restored_plain_tensors,
    }


def is_full_eval_state_checkpoint(checkpoint):
    return (
        isinstance(checkpoint, dict)
        and bool(checkpoint.get("full_eval_state_checkpoint", False))
        and checkpoint.get("checkpoint_type", "") == "full_eval_state"
    )


def _tensor_stats_for_log(tensor):
    tensor = tensor.detach().float()
    if tensor.numel() == 0:
        return "shape={} empty=True".format(list(tensor.shape))
    return (
        "shape={} mean={:.6f} std={:.6f} norm={:.6f}".format(
            list(tensor.shape),
            float(tensor.mean().item()),
            float(tensor.std(unbiased=False).item()),
            float(tensor.norm(p=2).item()),
        )
    )


def log_eval_state_summary(
    logger,
    model,
    stage,
    checkpoint=None,
    extra_state=None,
):
    model_state_dict = _get_target_state_dict(model)
    model_keyword_keys = _keyword_state_keys(model_state_dict)
    logger.info(
        "[STATE_SUMMARY] stage=%s model_keyword_key_count=%d",
        stage,
        len(model_keyword_keys),
    )
    logger.info(
        "[STATE_SUMMARY] stage=%s model_keyword_keys=%s",
        stage,
        model_keyword_keys,
    )

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model", None), dict):
        checkpoint_keyword_keys = _keyword_state_keys(checkpoint["model"])
    else:
        checkpoint_keyword_keys = []
    logger.info(
        "[STATE_SUMMARY] stage=%s checkpoint_keyword_key_count=%d",
        stage,
        len(checkpoint_keyword_keys),
    )
    if checkpoint_keyword_keys:
        logger.info(
            "[STATE_SUMMARY] stage=%s checkpoint_keyword_keys=%s",
            stage,
            checkpoint_keyword_keys,
        )

    plain_tensors, plain_tensor_shapes = _collect_plain_tensor_attrs(model)
    logger.info(
        "[STATE_SUMMARY] stage=%s plain_tensor_attr_count=%d plain_tensor_attrs=%s",
        stage,
        len(plain_tensors),
        plain_tensor_shapes,
    )

    proto_keys = [
        key
        for key in model_state_dict.keys()
        if "proto" in str(key).lower() or "prototype" in str(key).lower()
    ]
    for key in proto_keys:
        value = model_state_dict[key]
        if not torch.is_tensor(value):
            continue
        if value.dtype == torch.bool or "initialized" in str(key).lower():
            logger.info(
                "[STATE_SUMMARY] stage=%s prototype_state=%s shape=%s initialized_count=%d",
                stage,
                key,
                list(value.shape),
                int(value.bool().sum().item()),
            )
        else:
            logger.info(
                "[STATE_SUMMARY] stage=%s prototype_state=%s %s",
                stage,
                key,
                _tensor_stats_for_log(value),
            )

    if isinstance(extra_state, dict):
        logger.info(
            "[STATE_SUMMARY] stage=%s extra_state_keys=%s",
            stage,
            list(extra_state.keys()),
        )
        logger.info(
            "[STATE_SUMMARY] stage=%s extra_runtime_keys=%s extra_plain_tensor_keys=%s",
            stage,
            list(extra_state.get("runtime_attrs", {}).keys()),
            list(extra_state.get("plain_tensor_attrs", {}).keys()),
        )


def load_full_eval_state_checkpoint(model, checkpoint_file, logger=None, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    if not is_full_eval_state_checkpoint(checkpoint):
        raise ValueError(f"Not a full eval state checkpoint: {checkpoint_file}")
    if "model" not in checkpoint or not isinstance(checkpoint["model"], dict):
        raise KeyError(f"Full eval state checkpoint has no model state: {checkpoint_file}")

    load_state_dict(model, checkpoint["model"])
    if logger is not None:
        logger.info("[FULL_STATE_CKPT_LOAD] path=%s", checkpoint_file)
    return checkpoint


def _save_full_eval_state_checkpoint(
    model,
    args,
    epoch,
    metrics,
    reason,
    logger,
):
    if reason not in {"epoch", "best"}:
        raise ValueError(f"Unsupported full eval checkpoint reason: {reason}")

    metric_ckpt_dir = _resolve_metric_ckpt_dir(args)
    os.makedirs(metric_ckpt_dir, exist_ok=True)

    r1 = float(metrics.get("R1", 0.0)) if isinstance(metrics, dict) else 0.0
    map_value = float(metrics.get("mAP", 0.0)) if isinstance(metrics, dict) else 0.0
    prefix = "full_epoch" if reason == "epoch" else "full_best_epoch"
    ckpt_name = f"{prefix}_{int(epoch):03d}_R1_{r1:.3f}_mAP_{map_value:.3f}"
    ckpt_path = os.path.join(metric_ckpt_dir, f"{ckpt_name}.pth")

    extra_state = _collect_eval_extra_state(model)
    checkpoint = {
        "checkpoint_type": "full_eval_state",
        "full_eval_state_checkpoint": True,
        "model": model.state_dict(),
        "epoch": int(epoch),
        "metrics": {
            "R1": r1,
            "R5": float(metrics.get("R5", 0.0)) if isinstance(metrics, dict) else 0.0,
            "R10": float(metrics.get("R10", 0.0)) if isinstance(metrics, dict) else 0.0,
            "mAP": map_value,
            "mINP": float(metrics.get("mINP", 0.0)) if isinstance(metrics, dict) else 0.0,
        },
        "args": _args_to_plain_dict(args),
        "extra_eval_state": extra_state,
    }
    log_eval_state_summary(
        logger,
        model,
        stage="train_validation_before_save",
        checkpoint=checkpoint,
        extra_state=extra_state,
    )
    torch.save(checkpoint, ckpt_path)
    logger.info(
        "[FULL_STATE_CKPT_SAVE] path=%s epoch=%d R1=%.3f",
        ckpt_path,
        int(epoch),
        r1,
    )
    return ckpt_path


def _parse_verify_ckpt_epochs(args):
    raw_epochs = str(getattr(args, "verify_ckpt_epochs", "") or "").strip()
    if raw_epochs == "":
        return None
    epochs = set()
    for item in raw_epochs.split(","):
        item = item.strip()
        if item == "":
            continue
        try:
            epoch = int(item)
        except ValueError as exc:
            raise ValueError(
                f"--verify_ckpt_epochs must be comma-separated integers, got {raw_epochs!r}"
            ) from exc
        if epoch <= 0:
            raise ValueError(
                f"--verify_ckpt_epochs entries must be positive, got {epoch}."
            )
        epochs.add(epoch)
    return epochs


def _should_verify_ckpt_epoch(args, epoch):
    if not bool(getattr(args, "verify_ckpt_immediately_after_save", False)):
        return False
    verify_epochs = _parse_verify_ckpt_epochs(args)
    return verify_epochs is None or int(epoch) in verify_epochs


def _eval_model_with_existing_evaluator(evaluator, model, args):
    if bool(getattr(args, "distributed", False)) and hasattr(model, "module"):
        evaluator.eval(model.module.eval())
    else:
        evaluator.eval(model.eval())
    metrics = getattr(evaluator, "last_eval_result", None)
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "R1": float(metrics.get("R1", 0.0)),
        "R5": float(metrics.get("R5", 0.0)),
        "R10": float(metrics.get("R10", 0.0)),
        "mAP": float(metrics.get("mAP", 0.0)),
        "mINP": float(metrics.get("mINP", 0.0)),
    }


def _log_immediate_reeval_metrics(logger, tag, epoch, metrics, ckpt_path=None):
    if ckpt_path is None:
        logger.info(
            "%s epoch=%d R1=%.3f R5=%.3f R10=%.3f mAP=%.3f mINP=%.3f",
            tag,
            int(epoch),
            float(metrics.get("R1", 0.0)),
            float(metrics.get("R5", 0.0)),
            float(metrics.get("R10", 0.0)),
            float(metrics.get("mAP", 0.0)),
            float(metrics.get("mINP", 0.0)),
        )
        return
    logger.info(
        "%s ckpt=%s epoch=%d R1=%.3f R5=%.3f R10=%.3f mAP=%.3f mINP=%.3f",
        tag,
        ckpt_path,
        int(epoch),
        float(metrics.get("R1", 0.0)),
        float(metrics.get("R5", 0.0)),
        float(metrics.get("R10", 0.0)),
        float(metrics.get("mAP", 0.0)),
        float(metrics.get("mINP", 0.0)),
    )


def _verify_checkpoint_immediately_after_save(
    args,
    model,
    evaluator,
    ckpt_path,
    epoch,
    original_metrics,
    epoch_img_local_runtime,
    logger,
):
    if not ckpt_path:
        logger.warning(
            "[IMMEDIATE_REEVAL_SKIP] epoch=%d reason=missing_full_state_checkpoint",
            int(epoch),
        )
        return

    second_metrics = _eval_model_with_existing_evaluator(evaluator, model, args)
    _log_immediate_reeval_metrics(
        logger,
        "[IMMEDIATE_REEVAL_ORIGINAL]",
        epoch,
        second_metrics,
    )

    target_model = _unwrap_runtime_model(model)
    num_classes = int(getattr(target_model, "num_classes", 0))
    if num_classes <= 0:
        raise ValueError(
            "Cannot build immediate verification model because num_classes is missing."
        )

    device = next(target_model.parameters()).device
    reloaded_model = build_model(args, num_classes=num_classes)
    reloaded_model.to(device)
    checkpoint = load_full_eval_state_checkpoint(
        reloaded_model,
        ckpt_path,
        logger=logger,
    )
    apply_img_local_filter_stage(
        reloaded_model,
        args,
        epoch,
        logger=None,
        prefix="IMMEDIATE_RELOAD",
        runtime=epoch_img_local_runtime,
    )
    extra_eval_state = checkpoint.get("extra_eval_state", {})
    restore_eval_extra_state(reloaded_model, extra_eval_state, logger=logger)
    log_eval_state_summary(
        logger,
        reloaded_model,
        stage="immediate_reloaded_after_load",
        checkpoint=checkpoint,
        extra_state=extra_eval_state,
    )

    reloaded_metrics = _eval_model_with_existing_evaluator(
        evaluator,
        reloaded_model,
        args,
    )
    _log_immediate_reeval_metrics(
        logger,
        "[IMMEDIATE_REEVAL_RELOADED]",
        epoch,
        reloaded_metrics,
        ckpt_path=ckpt_path,
    )
    logger.info(
        "[IMMEDIATE_REEVAL_COMPARE] epoch=%d original_validation_R1=%.3f original_second_eval_R1=%.3f reloaded_R1=%.3f",
        int(epoch),
        float(original_metrics.get("R1", 0.0)),
        float(second_metrics.get("R1", 0.0)),
        float(reloaded_metrics.get("R1", 0.0)),
    )
    del reloaded_model
    torch.cuda.empty_cache()


def _save_metric_checkpoint(
    checkpointer,
    args,
    epoch,
    r1,
    metrics,
    arguments,
    reason,
    logger,
):
    if reason not in {"epoch", "best"}:
        raise ValueError(f"Unsupported metric checkpoint reason: {reason}")

    metric_ckpt_dir = _resolve_metric_ckpt_dir(args)
    os.makedirs(metric_ckpt_dir, exist_ok=True)

    r1 = float(r1)
    map_value = float(metrics.get("mAP", 0.0)) if isinstance(metrics, dict) else 0.0
    prefix = "epoch" if reason == "epoch" else "best_epoch"
    ckpt_name = f"{prefix}_{int(epoch):03d}_R1_{r1:.3f}_mAP_{map_value:.3f}"

    metric_arguments = dict(arguments)
    metric_arguments["epoch"] = int(epoch)

    metric_checkpointer = type(checkpointer)(
        checkpointer.model,
        checkpointer.optimizer,
        checkpointer.scheduler,
        metric_ckpt_dir,
        True,
        logger=logger,
    )
    metric_checkpointer.save(ckpt_name, **metric_arguments)

    ckpt_path = os.path.join(metric_ckpt_dir, f"{ckpt_name}.pth")
    json_path = os.path.join(metric_ckpt_dir, f"{ckpt_name}.json")
    metadata = {
        "path": ckpt_path,
        "epoch": int(epoch),
        "reason": str(reason),
        "R1": r1,
        "R5": float(metrics.get("R5", 0.0)) if isinstance(metrics, dict) else 0.0,
        "R10": float(metrics.get("R10", 0.0)) if isinstance(metrics, dict) else 0.0,
        "mAP": map_value,
        "mINP": float(metrics.get("mINP", 0.0)) if isinstance(metrics, dict) else 0.0,
        "output_dir": str(getattr(args, "output_dir", "")),
        "val_dataset": str(getattr(args, "val_dataset", "")),
        "batch_size": int(getattr(args, "batch_size", 0)),
        "test_batch_size": int(getattr(args, "test_batch_size", 0)),
        "checkpoint_arguments": {
            str(k): _json_safe_metadata_value(v) for k, v in metric_arguments.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    logger.info(
        "[METRIC_CKPT_SAVE] path=%s epoch=%d R1=%.3f mAP=%.3f reason=%s",
        ckpt_path,
        int(epoch),
        r1,
        map_value,
        reason,
    )
    return ckpt_path


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


def _uspc_enabled(args):
    return bool(getattr(args, "use_uspc", False))


def _uspc_should_encode_views(args):
    return _uspc_enabled(args) and float(getattr(args, "lambda_uspc", 0.0)) > 0.0


def _dcalign_lite_enabled(args):
    return bool(
        getattr(args, "use_dcalign_lite", False)
        and float(getattr(args, "lambda_dcalign", 0.0)) > 0.0
    )


def _dcalign_should_encode_views(args, epoch):
    return (
        _dcalign_lite_enabled(args)
        and float(resolve_dcalign_lite_lambda(args=args, epoch=epoch)) > 0.0
    )


def _semantic_alignment_should_encode_views(args, epoch):
    return _uspc_should_encode_views(args) or _dcalign_should_encode_views(args, epoch)


def _empty_dcalign_lite_terms(args, reference, epoch):
    if not torch.is_tensor(reference):
        raise TypeError("DCAlign-lite inactive terms require a tensor reference.")
    zero = reference.new_zeros(())
    return {
        "loss_dcalign": zero,
        "loss_dcalign_raw": zero,
        "loss_dcalign_color": zero,
        "loss_dcalign_struct": zero,
        "loss_dcalign_full": zero,
        "dcalign_lambda_effective": zero.new_tensor(
            float(resolve_dcalign_lite_lambda(args=args, epoch=epoch))
        ),
        "dcalign_active": zero,
        "dcalign_color_weight": zero.new_tensor(
            float(getattr(args, "dcalign_color_weight", 1.0))
        ),
        "dcalign_struct_weight": zero.new_tensor(
            float(getattr(args, "dcalign_struct_weight", 1.0))
        ),
        "dcalign_full_weight": zero.new_tensor(
            float(getattr(args, "dcalign_full_weight", 0.25))
        ),
    }


def _uspc_normalize_raw_captions(raw_captions):
    if isinstance(raw_captions, str):
        return [raw_captions]
    if raw_captions is None:
        return []
    return ["" if caption is None else str(caption) for caption in list(raw_captions)]


def _uspc_tokenize_text_batch(texts, tokenizer, text_length, device):
    token_tensors = [
        tokenize(text, tokenizer=tokenizer, text_length=text_length)
        for text in texts
    ]
    if not token_tensors:
        raise ValueError("USPC text view tokenization received an empty text list.")
    return torch.stack(token_tensors, dim=0).to(device)


def _uspc_encode_text_tokens(model, token_ids):
    target_model = _unwrap_runtime_model(model)
    if hasattr(target_model, "encode_text_deterministic"):
        return target_model.encode_text_deterministic(token_ids)
    if hasattr(target_model, "encode_text"):
        return target_model.encode_text(token_ids)
    raise AttributeError(
        "USPC text view encoding requires model.encode_text_deterministic() "
        "or model.encode_text()."
    )


def _uspc_select_ret_feature(ret, primary_key, fallback_key, display_name):
    feature = ret.get(primary_key, None)
    if feature is None:
        feature = ret.get(fallback_key, None)
    if feature is None:
        raise KeyError(
            f"USPC requires {display_name}; tried ret['{primary_key}'] "
            f"then ret['{fallback_key}']."
        )
    return feature


def _uspc_shape_string(value):
    if torch.is_tensor(value):
        return str(list(value.shape))
    return "None"


def _build_uspc_forward_inputs(args, model, batch, ret, tokenizer):
    if "raw_captions" not in batch:
        raise KeyError(
            "USPC is enabled with lambda_uspc > 0, but batch['raw_captions'] "
            "is missing. Ensure the training dataset returns raw captions."
        )
    raw_captions = _uspc_normalize_raw_captions(batch["raw_captions"])
    if not raw_captions:
        raise ValueError("USPC is enabled with lambda_uspc > 0, but raw_captions is empty.")

    try:
        views = build_uspc_text_views(
            raw_captions,
            mode=str(getattr(args, "uspc_text_view_mode", "rule")),
        )
        device = batch["caption_ids"].device
        text_length = int(getattr(args, "text_length", batch["caption_ids"].shape[1]))
        color_ids = _uspc_tokenize_text_batch(
            views["color"],
            tokenizer=tokenizer,
            text_length=text_length,
            device=device,
        )
        struct_ids = _uspc_tokenize_text_batch(
            views["struct"],
            tokenizer=tokenizer,
            text_length=text_length,
            device=device,
        )
        color_text_feat = _uspc_encode_text_tokens(model, color_ids)
        struct_text_feat = _uspc_encode_text_tokens(model, struct_ids)

        full_text_feat = ret.get("text_mu", None)
        if full_text_feat is None:
            full_text_feat = ret.get("text_det_feat", None)
        if full_text_feat is None:
            full_ids = _uspc_tokenize_text_batch(
                views["full"],
                tokenizer=tokenizer,
                text_length=text_length,
                device=device,
            )
            full_text_feat = _uspc_encode_text_tokens(model, full_ids)

        return {
            "raw_captions": raw_captions,
            "views": views,
            "rgb_feat": _uspc_select_ret_feature(ret, "rgb_mu", "v_rgb_c", "rgb_feat"),
            "thermal_feat": _uspc_select_ret_feature(
                ret, "thermal_mu", "v_t_c", "thermal_feat"
            ),
            "fused_feat": _uspc_select_ret_feature(
                ret, "fused_mu", "fused_det_feat", "fused_feat"
            ),
            "full_text_feat": full_text_feat,
            "color_text_feat": color_text_feat,
            "struct_text_feat": struct_text_feat,
            "rgb_logvar": ret.get("rgb_logvar", None),
            "thermal_logvar": ret.get("thermal_logvar", None),
            "fused_logvar": ret.get("fused_logvar", None),
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to build/encode USPC semantic text views: {exc}") from exc


def _log_uspc_debug_examples(logger, raw_captions, uspc_inputs=None):
    for idx, raw_caption in enumerate(list(raw_captions)[:2]):
        logger.info(
            "[USPC][RAW_CAPTION_EXAMPLE] idx=%d raw=%s",
            idx,
            _truncate_log_text(raw_caption),
        )
    if uspc_inputs is None:
        return

    views = uspc_inputs["views"]
    for idx, color_view in enumerate(list(views.get("color", []))[:2]):
        logger.info(
            "[USPC][COLOR_VIEW_EXAMPLE] idx=%d color=%s",
            idx,
            _truncate_log_text(color_view),
        )
    for idx, struct_view in enumerate(list(views.get("struct", []))[:2]):
        logger.info(
            "[USPC][STRUCT_VIEW_EXAMPLE] idx=%d struct=%s",
            idx,
            _truncate_log_text(struct_view),
        )
    logger.info(
        "[USPC][FEATURE_SHAPES] rgb_feat=%s thermal_feat=%s fused_feat=%s full_text_feat=%s color_text_feat=%s struct_text_feat=%s",
        _uspc_shape_string(uspc_inputs.get("rgb_feat")),
        _uspc_shape_string(uspc_inputs.get("thermal_feat")),
        _uspc_shape_string(uspc_inputs.get("fused_feat")),
        _uspc_shape_string(uspc_inputs.get("full_text_feat")),
        _uspc_shape_string(uspc_inputs.get("color_text_feat")),
        _uspc_shape_string(uspc_inputs.get("struct_text_feat")),
    )
    logger.info(
        "[USPC][LOGVAR_AVAILABLE] rgb_logvar=%s thermal_logvar=%s fused_logvar=%s",
        torch.is_tensor(uspc_inputs.get("rgb_logvar")),
        torch.is_tensor(uspc_inputs.get("thermal_logvar")),
        torch.is_tensor(uspc_inputs.get("fused_logvar")),
    )


def _log_uspc_debug_status(logger, args, uspc_terms=None):
    lambda_uspc = float(getattr(args, "lambda_uspc", 0.0))
    if uspc_terms is None:
        logger.info(
            "[USPC][LOSS_STATUS] active=0 lambda_uspc=%.6f loss_uspc=0.000000 "
            "loss_uspc_raw=0.000000 loss_uspc_color=0.000000 "
            "loss_uspc_struct=0.000000 loss_uspc_full=0.000000 "
            "uspc_rgb_conf_mean=1.000000 uspc_thermal_conf_mean=1.000000 "
            "uspc_fused_conf_mean=1.000000",
            lambda_uspc,
        )
        return

    logger.info(
        "[USPC][LOSS_STATUS] active=%d lambda_uspc=%.6f loss_uspc=%.6f "
        "loss_uspc_raw=%.6f loss_uspc_color=%.6f loss_uspc_struct=%.6f "
        "loss_uspc_full=%.6f uspc_rgb_conf_mean=%.6f "
        "uspc_thermal_conf_mean=%.6f uspc_fused_conf_mean=%.6f",
        int(round(_to_scalar(uspc_terms.get("uspc_active", 0)))),
        lambda_uspc,
        _to_scalar(uspc_terms.get("loss_uspc", 0)),
        _to_scalar(uspc_terms.get("loss_uspc_raw", 0)),
        _to_scalar(uspc_terms.get("loss_uspc_color", 0)),
        _to_scalar(uspc_terms.get("loss_uspc_struct", 0)),
        _to_scalar(uspc_terms.get("loss_uspc_full", 0)),
        _to_scalar(uspc_terms.get("uspc_rgb_conf_mean", 1)),
        _to_scalar(uspc_terms.get("uspc_thermal_conf_mean", 1)),
        _to_scalar(uspc_terms.get("uspc_fused_conf_mean", 1)),
    )


def _log_dcalign_debug_status(logger, args, epoch, dcalign_terms=None):
    lambda_dcalign = float(getattr(args, "lambda_dcalign", 0.0))
    lambda_effective = float(resolve_dcalign_lite_lambda(args=args, epoch=epoch))
    color_weight = float(getattr(args, "dcalign_color_weight", 1.0))
    struct_weight = float(getattr(args, "dcalign_struct_weight", 1.0))
    full_weight = float(getattr(args, "dcalign_full_weight", 0.25))
    detach_text = bool(getattr(args, "dcalign_detach_text_views", False))
    detach_image = bool(getattr(args, "dcalign_detach_image_feats", False))
    if dcalign_terms is None:
        logger.info(
            "[DCALIGN][LOSS_STATUS] active=0 lambda_dcalign=%.6f "
            "lambda_effective=%.6f loss_dcalign=0.000000 "
            "loss_dcalign_raw=0.000000 loss_dcalign_color=0.000000 "
            "loss_dcalign_struct=0.000000 loss_dcalign_full=0.000000 "
            "weights=(%.6f,%.6f,%.6f) detach_text=%s detach_image=%s",
            lambda_dcalign,
            lambda_effective,
            color_weight,
            struct_weight,
            full_weight,
            detach_text,
            detach_image,
        )
        return

    logger.info(
        "[DCALIGN][LOSS_STATUS] active=%d lambda_dcalign=%.6f "
        "lambda_effective=%.6f loss_dcalign=%.6f loss_dcalign_raw=%.6f "
        "loss_dcalign_color=%.6f loss_dcalign_struct=%.6f "
        "loss_dcalign_full=%.6f weights=(%.6f,%.6f,%.6f) "
        "detach_text=%s detach_image=%s",
        int(round(_to_scalar(dcalign_terms.get("dcalign_active", 0)))),
        lambda_dcalign,
        _to_scalar(
            dcalign_terms.get("dcalign_lambda_effective", lambda_effective)
        ),
        _to_scalar(dcalign_terms.get("loss_dcalign", 0)),
        _to_scalar(dcalign_terms.get("loss_dcalign_raw", 0)),
        _to_scalar(dcalign_terms.get("loss_dcalign_color", 0)),
        _to_scalar(dcalign_terms.get("loss_dcalign_struct", 0)),
        _to_scalar(dcalign_terms.get("loss_dcalign_full", 0)),
        _to_scalar(dcalign_terms.get("dcalign_color_weight", color_weight)),
        _to_scalar(dcalign_terms.get("dcalign_struct_weight", struct_weight)),
        _to_scalar(dcalign_terms.get("dcalign_full_weight", full_weight)),
        detach_text,
        detach_image,
    )


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


def _build_total_loss(args, ret, epoch=None):
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
                "loss_attr_aux_raw",
                "loss_attr_aux_weighted",
                "loss_uspc_slot_local_raw",
                "loss_uspc_slot_local_weighted",
                "loss_dcalign",
                "loss_dcalign_raw",
                "loss_dcalign_color",
                "loss_dcalign_struct",
                "loss_dcalign_full",
                "loss_mdrc",
                "loss_mdrc_feature",
                "loss_mdrc_similarity",
                "loss_mdrc_weighted",
                "mdrc_loss_scale",
                "loss_mga_pseudo_sdm",
                "loss_mga_rgb_color",
                "loss_mga_t_shape",
                "loss_mga_negative",
                "loss_rde_consensus",
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


ATTR_AUX_NAMES = (
    "gender",
    "upper_color",
    "lower_color",
    "shoe_color",
    "bag_color",
    "bag_present",
    "hair_length",
)
USPC_SLOT_LOCAL_ATTR_NAMES = get_uspc_slot_local_attr_names()


def _attr_aux_enabled(args):
    return bool(
        getattr(args, "use_attr_aux", False)
        and float(getattr(args, "lambda_attr_aux", 0.0)) > 0.0
    )


def _empty_attr_aux_terms(reference, active=False):
    zero = reference.new_zeros(())
    out = {
        "loss_attr_aux_raw": zero,
        "loss_attr_aux_weighted": zero,
        "attr_active": zero.new_tensor(1.0 if active else 0.0),
    }
    for attr_name in ATTR_AUX_NAMES:
        out[f"attr_valid_{attr_name}"] = zero
    return out


def _compute_attr_aux_terms(args, ret, batch, epoch):
    if not _attr_aux_enabled(args):
        return {}

    if "images" not in batch or not torch.is_tensor(batch["images"]):
        raise KeyError("Attribute auxiliary loss requires batch['images'] tensor.")
    reference = batch["images"]

    if int(epoch) <= int(getattr(args, "attr_warmup_epochs", 0)):
        return _empty_attr_aux_terms(reference, active=False)
    active_until_epoch = int(getattr(args, "attr_active_until_epoch", -1))
    if active_until_epoch > 0 and int(epoch) > active_until_epoch:
        return _empty_attr_aux_terms(reference, active=False)

    schema = str(getattr(args, "attr_schema", "basic_v1")).lower()
    if schema != "basic_v1":
        raise ValueError(f"Unsupported attr_schema={schema!r}. Expected 'basic_v1'.")
    if "raw_captions" not in batch:
        raise KeyError(
            "use_attr_aux=True requires batch['raw_captions']; the first "
            "USPC-v2-A version parses captions dynamically in processor."
        )

    missing_logits = [
        f"attr_logits_{attr_name}"
        for attr_name in ATTR_AUX_NAMES
        if f"attr_logits_{attr_name}" not in ret
    ]
    if missing_logits:
        raise KeyError(
            "use_attr_aux=True requires model to return attribute logits, "
            f"missing {missing_logits}. Ensure the model was built with "
            "--use_attr_aux and lambda_attr_aux > 0."
        )

    parsed = parse_basic_attributes(
        batch["raw_captions"],
        device=reference.device,
    )
    targets = parsed["targets"]
    masks = parsed["masks"]
    loss_type = str(getattr(args, "attr_loss_type", "ce")).lower()
    if loss_type not in {"ce", "bce"}:
        raise ValueError(f"Unsupported attr_loss_type={loss_type!r}.")

    losses = []
    out = _empty_attr_aux_terms(reference, active=True)
    for attr_name in ATTR_AUX_NAMES:
        logits = ret[f"attr_logits_{attr_name}"]
        if not torch.is_tensor(logits):
            raise TypeError(f"ret['attr_logits_{attr_name}'] must be a tensor.")
        if logits.dim() != 2:
            raise ValueError(
                f"ret['attr_logits_{attr_name}'] must have shape [B, C], "
                f"got {tuple(logits.shape)}."
            )
        mask = masks[attr_name].to(device=logits.device, dtype=torch.bool).reshape(-1)
        target = targets[attr_name].to(device=logits.device).reshape(-1).long()
        if mask.shape[0] != logits.shape[0] or target.shape[0] != logits.shape[0]:
            raise ValueError(
                f"Attribute batch mismatch for {attr_name}: "
                f"logits={tuple(logits.shape)}, target={tuple(target.shape)}, "
                f"mask={tuple(mask.shape)}."
            )

        valid_count = int(mask.sum().item())
        out[f"attr_valid_{attr_name}"] = logits.new_tensor(float(valid_count))
        if valid_count <= 0:
            continue

        valid_logits = logits[mask].float()
        valid_target = target[mask]
        if loss_type == "ce":
            attr_loss = F.cross_entropy(valid_logits, valid_target)
        else:
            target_onehot = F.one_hot(
                valid_target,
                num_classes=valid_logits.shape[1],
            ).to(device=valid_logits.device, dtype=valid_logits.dtype)
            attr_loss = F.binary_cross_entropy_with_logits(
                valid_logits,
                target_onehot,
            )
        losses.append(attr_loss)

    if losses:
        loss_raw = torch.stack(losses).mean()
    else:
        loss_raw = reference.new_zeros(())
    loss_weighted = float(getattr(args, "lambda_attr_aux", 0.0)) * loss_raw
    out["loss_attr_aux_raw"] = loss_raw
    out["loss_attr_aux_weighted"] = loss_weighted
    out["attr_parser_stats"] = parsed.get("stats", {})
    out["attr_loss_type"] = loss_type
    out["attr_feature_source"] = str(getattr(args, "attr_feature_source", "fused_det"))
    return out


def _add_attr_aux_to_total_loss(args, total_loss, ret):
    if not _attr_aux_enabled(args):
        return total_loss
    if "loss_attr_aux_weighted" not in ret:
        raise KeyError(
            "Attribute auxiliary loss is enabled but 'loss_attr_aux_weighted' "
            "is missing from processor outputs."
        )
    return total_loss + ret["loss_attr_aux_weighted"]


def _uspc_slot_local_enabled(args):
    return bool(
        getattr(args, "use_uspc_slot_local", False)
        and float(getattr(args, "lambda_uspc_slot_local", 0.0)) > 0.0
    )


def _empty_uspc_slot_local_terms(reference, active=False):
    zero = reference.new_zeros(())
    out = {
        "loss_uspc_slot_local_raw": zero,
        "loss_uspc_slot_local_weighted": zero,
        "uspc_slot_local_active": zero.new_tensor(1.0 if active else 0.0),
    }
    for attr_name in USPC_SLOT_LOCAL_ATTR_NAMES:
        out[f"uspc_slot_valid_{attr_name}"] = zero
    return out


def _compute_uspc_slot_local_terms(args, ret, batch, epoch):
    if not _uspc_slot_local_enabled(args):
        return {}

    if "images" not in batch or not torch.is_tensor(batch["images"]):
        raise KeyError("USPC slot-local loss requires batch['images'] tensor.")
    reference = batch["images"]

    if int(epoch) <= int(getattr(args, "uspc_slot_warmup_epochs", 0)):
        return _empty_uspc_slot_local_terms(reference, active=False)
    active_until_epoch = int(getattr(args, "uspc_slot_active_until_epoch", -1))
    if active_until_epoch > 0 and int(epoch) > active_until_epoch:
        return _empty_uspc_slot_local_terms(reference, active=False)

    patch_source = str(getattr(args, "uspc_slot_patch_source", "rgb")).lower()
    if patch_source != "rgb":
        raise NotImplementedError(
            "USPC-v2-B0 only guarantees --uspc_slot_patch_source rgb. "
            f"Got {patch_source!r}."
        )
    loss_type = str(getattr(args, "uspc_slot_loss_type", "ce")).lower()
    if loss_type != "ce":
        raise ValueError(f"Unsupported uspc_slot_loss_type={loss_type!r}.")
    if "raw_captions" not in batch:
        raise KeyError(
            "use_uspc_slot_local=True requires batch['raw_captions']; "
            "dataset/collate stay unchanged and processor parses captions dynamically."
        )
    if "uspc_slot_logits" not in ret:
        raise KeyError(
            "use_uspc_slot_local=True requires model to return "
            "ret['uspc_slot_logits']. Ensure the model was built with "
            "--use_uspc_slot_local and lambda_uspc_slot_local > 0."
        )

    logits_by_attr = ret["uspc_slot_logits"]
    if not isinstance(logits_by_attr, dict):
        raise TypeError("ret['uspc_slot_logits'] must be a dict of tensors.")

    parsed = parse_basic_attributes(
        batch["raw_captions"],
        device=reference.device,
    )
    targets = parsed["targets"]
    masks = parsed["masks"]

    losses = []
    out = _empty_uspc_slot_local_terms(reference, active=True)
    for attr_name in USPC_SLOT_LOCAL_ATTR_NAMES:
        if attr_name not in logits_by_attr:
            raise KeyError(f"ret['uspc_slot_logits'] is missing {attr_name!r}.")
        logits = logits_by_attr[attr_name]
        if not torch.is_tensor(logits):
            raise TypeError(f"USPC slot logits for {attr_name} must be a tensor.")
        if logits.dim() != 2:
            raise ValueError(
                f"USPC slot logits for {attr_name} must have shape [B, C], "
                f"got {tuple(logits.shape)}."
            )

        mask = masks[attr_name].to(device=logits.device, dtype=torch.bool).reshape(-1)
        target = targets[attr_name].to(device=logits.device).reshape(-1).long()
        if mask.shape[0] != logits.shape[0] or target.shape[0] != logits.shape[0]:
            raise ValueError(
                f"USPC slot batch mismatch for {attr_name}: "
                f"logits={tuple(logits.shape)}, target={tuple(target.shape)}, "
                f"mask={tuple(mask.shape)}."
            )

        valid_count = int(mask.sum().item())
        out[f"uspc_slot_valid_{attr_name}"] = logits.new_tensor(float(valid_count))
        if valid_count <= 0:
            continue
        losses.append(F.cross_entropy(logits[mask].float(), target[mask]))

    if losses:
        loss_raw = torch.stack(losses).mean()
    else:
        loss_raw = reference.new_zeros(())
    loss_weighted = float(getattr(args, "lambda_uspc_slot_local", 0.0)) * loss_raw
    out["loss_uspc_slot_local_raw"] = loss_raw
    out["loss_uspc_slot_local_weighted"] = loss_weighted
    out["uspc_slot_parser_stats"] = parsed.get("stats", {})
    out["uspc_slot_patch_source"] = patch_source
    out["uspc_slot_loss_type"] = loss_type
    return out


def _add_uspc_slot_local_to_total_loss(args, total_loss, ret):
    if not _uspc_slot_local_enabled(args):
        return total_loss
    if "loss_uspc_slot_local_weighted" not in ret:
        raise KeyError(
            "USPC slot-local loss is enabled but "
            "'loss_uspc_slot_local_weighted' is missing from processor outputs."
        )
    return total_loss + ret["loss_uspc_slot_local_weighted"]


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


def _get_first_tensor_from_ret(ret, keys):
    for key in keys:
        value = ret.get(key, None)
        if torch.is_tensor(value):
            return value, key
    return None, "missing"


def _mga_warn_once(key, message):
    if key in _MGA_WARNED_KEYS:
        return
    _MGA_WARNED_KEYS.add(key)
    logging.getLogger("IRRA.train").warning(message)


def _temperature_to_tensor(temperature, reference):
    if temperature is None:
        return reference.new_tensor(float(0.02), dtype=torch.float32)
    if torch.is_tensor(temperature):
        return temperature.to(device=reference.device, dtype=torch.float32).reshape(-1)[0]
    return reference.new_tensor(float(temperature), dtype=torch.float32)


def weighted_sdm_or_contrastive_loss(
    image_feat,
    text_feat,
    pids,
    weights,
    temperature=None,
):
    """Weighted one-to-one symmetric InfoNCE fallback for MGA auxiliary text.

    This fallback does not merge same-pid positives like SDM, so duplicate pids
    in a batch are less precisely modeled. It is used here because MGA needs
    stable per-sample weighting.
    """
    if image_feat.ndim != 2 or text_feat.ndim != 2:
        raise ValueError(
            "weighted_sdm_or_contrastive_loss expects 2D image/text features."
        )
    if image_feat.shape != text_feat.shape:
        raise ValueError(
            "weighted_sdm_or_contrastive_loss feature shape mismatch: "
            f"{tuple(image_feat.shape)} vs {tuple(text_feat.shape)}."
        )
    batch_size = image_feat.shape[0]
    zero = image_feat.new_zeros(())
    if batch_size < 2:
        return zero

    if weights is None:
        weights = image_feat.new_ones(batch_size, dtype=torch.float32)
    else:
        weights = weights.to(device=image_feat.device, dtype=torch.float32).reshape(-1)
    if weights.numel() != batch_size:
        raise ValueError(
            "weighted_sdm_or_contrastive_loss weight length mismatch: "
            f"{weights.numel()} vs {batch_size}."
        )
    weights = weights.clamp(0.0, 1.0)
    weight_sum = weights.sum().clamp_min(1e-6)
    if float(weight_sum.detach().cpu()) <= 1e-6:
        return zero

    temp = _temperature_to_tensor(temperature, image_feat).clamp_min(1e-6)
    image_clean = torch.nan_to_num(
        image_feat.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    text_clean = torch.nan_to_num(
        text_feat.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    image_norm = F.normalize(image_clean, p=2, dim=1)
    text_norm = F.normalize(text_clean, p=2, dim=1)
    logits = image_norm @ text_norm.t() / temp
    target = torch.arange(batch_size, device=image_feat.device)
    loss_i2t = F.cross_entropy(logits, target, reduction="none")
    loss_t2i = F.cross_entropy(logits.t(), target, reduction="none")
    loss = ((loss_i2t + loss_t2i) * 0.5 * weights).sum() / weight_sum
    return loss


def _cosine_diag_similarity(image_feat, text_feat):
    image_clean = torch.nan_to_num(
        image_feat.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    text_clean = torch.nan_to_num(
        text_feat.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    image_norm = F.normalize(image_clean, p=2, dim=1)
    text_norm = F.normalize(text_clean, p=2, dim=1)
    return (image_norm * text_norm).sum(dim=1).clamp(-1.0, 1.0)


def _empty_rde_stats(reference):
    zero = reference.new_zeros(())
    return {
        "rde_clean_ratio": zero,
        "rde_uncertain_ratio": zero,
        "rde_noisy_ratio": zero,
        "rde_mean_conf": zero,
        "rde_count": 0.0,
    }


def get_rde_effective_weight(
    base_weight,
    pos_sim=None,
    args=None,
    epoch=None,
):
    base_weight = base_weight.float().reshape(-1).clamp(0.0, 1.0)
    stats = _empty_rde_stats(base_weight)
    if args is None or not bool(getattr(args, "use_rde_robust", False)):
        return base_weight, stats

    warmup_epochs = int(getattr(args, "rde_warmup_epochs", 10))
    if (
        epoch is None
        or int(epoch) <= warmup_epochs
    ) and not bool(getattr(args, "rde_enable_pseudo_during_warmup", False)):
        return torch.zeros_like(base_weight), stats

    conf_source = str(getattr(args, "rde_conf_source", "hybrid")).lower()
    if pos_sim is None:
        online_conf = base_weight
        if conf_source in {"online_sim", "hybrid"}:
            _mga_warn_once(
                "rde_online_sim_missing",
                "[RDE] pos_sim is missing; falling back to MGA base weight for "
                "online confidence.",
            )
    else:
        online_conf = ((pos_sim.to(base_weight.device).float().reshape(-1) + 1.0) * 0.5)
        online_conf = online_conf.clamp(0.0, 1.0)
        if online_conf.numel() != base_weight.numel():
            _mga_warn_once(
                "rde_online_sim_shape",
                "[RDE] pos_sim shape does not match base_weight; falling back "
                "to MGA base weight for online confidence.",
            )
            online_conf = base_weight

    if conf_source == "mga_weight":
        conf = base_weight
    elif conf_source == "online_sim":
        conf = online_conf
    elif conf_source == "hybrid":
        conf = 0.5 * base_weight + 0.5 * online_conf
    else:
        raise ValueError(f"Unsupported rde_conf_source={conf_source}.")
    conf = conf.clamp(0.0, 1.0)

    clean_threshold = float(getattr(args, "rde_clean_threshold", 0.70))
    uncertain_threshold = float(getattr(args, "rde_uncertain_threshold", 0.40))
    clean_mask = conf.ge(clean_threshold)
    uncertain_mask = conf.ge(uncertain_threshold) & conf.lt(clean_threshold)
    noisy_mask = conf.lt(uncertain_threshold)

    effective_weight = torch.zeros_like(base_weight)
    effective_weight = torch.where(clean_mask, base_weight, effective_weight)
    effective_weight = torch.where(
        uncertain_mask,
        base_weight * float(getattr(args, "rde_uncertain_weight", 0.5)),
        effective_weight,
    )
    effective_weight = torch.where(
        noisy_mask,
        base_weight * float(getattr(args, "rde_noisy_weight", 0.1)),
        effective_weight,
    )
    count = float(base_weight.numel())
    if count > 0:
        stats = {
            "rde_clean_ratio": clean_mask.float().mean(),
            "rde_uncertain_ratio": uncertain_mask.float().mean(),
            "rde_noisy_ratio": noisy_mask.float().mean(),
            "rde_mean_conf": conf.mean(),
            "rde_count": count,
        }
    return effective_weight.clamp(0.0, 1.0), stats


def _merge_rde_stats_into_terms(terms, stats):
    count = float(stats.get("rde_count", 0.0) or 0.0)
    if count <= 0:
        return
    prev_count = float(terms.get("_rde_stats_count", 0.0) or 0.0)
    total_count = prev_count + count
    for key in (
        "rde_clean_ratio",
        "rde_uncertain_ratio",
        "rde_noisy_ratio",
        "rde_mean_conf",
    ):
        prev = terms.get(key, None)
        cur = stats.get(key, None)
        if cur is None:
            continue
        if prev is None or not torch.is_tensor(prev):
            terms[key] = cur
        else:
            terms[key] = (prev * prev_count + cur * count) / max(total_count, 1e-6)
    terms["_rde_stats_count"] = total_count


def _mga_reference_tensor(ret, batch):
    for source in (ret, batch):
        for value in source.values():
            if torch.is_tensor(value):
                return value
    raise KeyError("MGA auxiliary loss requires at least one tensor in ret or batch.")


def _mga_empty_terms(reference):
    zero = reference.new_zeros(())
    return {
        "loss_mga_pseudo_sdm": zero,
        "loss_mga_rgb_color": zero,
        "loss_mga_t_shape": zero,
        "loss_mga_negative": zero,
        "loss_rde_consensus": zero,
        "mga_valid_pseudo_ratio": zero,
        "mga_valid_rgb_color_ratio": zero,
        "mga_valid_t_shape_ratio": zero,
        "rde_clean_ratio": zero,
        "rde_uncertain_ratio": zero,
        "rde_noisy_ratio": zero,
        "rde_mean_conf": zero,
        "conf_hn_active_ratio": zero,
        "_rde_stats_count": 0.0,
    }


def _mga_valid_ratio(valid, reference):
    if valid is None:
        return reference.new_zeros(())
    valid = valid.to(device=reference.device).reshape(-1)
    if valid.numel() == 0:
        return reference.new_zeros(())
    return valid.float().mean()


def _mga_subset_for_alignment(
    image_feat,
    text_feat,
    valid,
    weights,
    min_valid,
    warn_key,
):
    zero = image_feat.new_zeros(())
    if valid is None:
        return None, None, None, zero
    valid = valid.to(device=image_feat.device).reshape(-1).bool()
    batch_size = image_feat.shape[0]
    if (
        text_feat.shape[0] != batch_size
        or valid.numel() != batch_size
        or (weights is not None and weights.reshape(-1).numel() != batch_size)
    ):
        _mga_warn_once(
            warn_key,
            "[MGA] skipping auxiliary loss because feature/valid/weight batch "
            "dimensions do not match.",
        )
        return None, None, None, zero
    if int(valid.sum().item()) < int(min_valid):
        return None, None, None, zero
    if weights is None:
        weights = image_feat.new_ones(batch_size, dtype=torch.float32)
    else:
        weights = weights.to(device=image_feat.device, dtype=torch.float32).reshape(-1)
    weights = weights.clamp(0.0, 1.0)
    if float(weights[valid].sum().detach().cpu()) <= 1e-6:
        return None, None, None, zero
    return image_feat[valid], text_feat[valid], weights[valid], zero


def _compute_confidence_hard_negative_terms(args, ret, batch, reference):
    if not bool(getattr(args, "use_confidence_hard_negative", False)):
        return {}

    _mga_warn_once(
        "conf_hn_original_caption_only",
        "[CONF_HN] current hard-negative branches use original captions/features; "
        "no pseudo hard-negative mask is applied. conf_hn_active_ratio reports "
        "pseudo confidence eligibility only.",
    )
    zero = reference.new_zeros(())
    valid = ret.get("mga_pseudo_valid", batch.get("mga_has_pseudo", None))
    weight = ret.get("mga_pseudo_weight", batch.get("mga_pseudo_weight", None))
    if valid is None or weight is None:
        return {"conf_hn_active_ratio": zero}

    valid = valid.to(device=reference.device).reshape(-1).bool()
    weight = weight.to(device=reference.device, dtype=torch.float32).reshape(-1)
    if valid.numel() != weight.numel() or valid.numel() == 0 or not valid.any():
        return {"conf_hn_active_ratio": zero}

    min_conf = float(getattr(args, "conf_hn_min_weight", 0.5))
    active = weight[valid].ge(min_conf)
    return {"conf_hn_active_ratio": active.float().mean()}


def _compute_mga_aux_terms(args, ret, batch, epoch):
    use_pseudo = bool(getattr(args, "use_mga_pseudo", False))
    use_attr = bool(getattr(args, "use_mga_attr_align", False))
    use_negative = bool(getattr(args, "use_mga_negative_loss", False))
    use_rde = bool(getattr(args, "use_rde_robust", False))
    use_conf_hn = bool(getattr(args, "use_confidence_hard_negative", False))
    if not (use_pseudo or use_attr or use_negative or use_rde or use_conf_hn):
        return {}

    reference = _mga_reference_tensor(ret, batch)
    terms = _mga_empty_terms(reference)
    terms.update(_compute_confidence_hard_negative_terms(args, ret, batch, reference))
    temperature = ret.get("temperature", None)
    pids = batch.get("pids", None)

    if use_pseudo:
        valid = ret.get("mga_pseudo_valid", batch.get("mga_has_pseudo", None))
        terms["mga_valid_pseudo_ratio"] = _mga_valid_ratio(valid, reference)
        pseudo_valid_mask = None
        pseudo_image_feat = None
        pseudo_text_feat = None
        pseudo_effective_weight = None
        if (
            (
                float(getattr(args, "lambda_mga_pseudo_sdm", 0.0)) > 0.0
                or use_rde
                or float(getattr(args, "lambda_rde_consensus", 0.0)) > 0.0
            )
            and "fused_det_feat" in ret
            and "mga_pseudo_text_feat" in ret
        ):
            image_feat, text_feat, weights, zero = _mga_subset_for_alignment(
                ret["fused_det_feat"],
                ret["mga_pseudo_text_feat"],
                valid,
                ret.get("mga_pseudo_weight", None),
                min_valid=2,
                warn_key="pseudo_shape",
            )
            if image_feat is not None:
                pseudo_valid_mask = valid.to(
                    device=ret["fused_det_feat"].device
                ).reshape(-1).bool()
                pseudo_image_feat = image_feat
                pseudo_text_feat = text_feat
                pos_sim = _cosine_diag_similarity(image_feat, text_feat)
                effective_weight, rde_stats = get_rde_effective_weight(
                    weights,
                    pos_sim=pos_sim,
                    args=args,
                    epoch=epoch,
                )
                pseudo_effective_weight = effective_weight
                if use_rde:
                    _merge_rde_stats_into_terms(terms, rde_stats)
                if float(getattr(args, "lambda_mga_pseudo_sdm", 0.0)) > 0.0:
                    terms["loss_mga_pseudo_sdm"] = weighted_sdm_or_contrastive_loss(
                        image_feat,
                        text_feat,
                        pids,
                        effective_weight,
                        temperature=temperature,
                    )
            else:
                terms["loss_mga_pseudo_sdm"] = zero

        if (
            use_rde
            and float(getattr(args, "lambda_rde_consensus", 0.0)) > 0.0
            and epoch is not None
            and int(epoch) > int(getattr(args, "rde_warmup_epochs", 10))
            and pseudo_valid_mask is not None
            and pseudo_effective_weight is not None
            and "text_det_feat" in ret
        ):
            active_mask = pseudo_effective_weight.gt(0.0)
            if active_mask.any():
                orig_text_feat = ret["text_det_feat"][pseudo_valid_mask][active_mask]
                fused_feat = pseudo_image_feat[active_mask]
                pseudo_feat = pseudo_text_feat[active_mask]
                active_weight = pseudo_effective_weight[active_mask].to(
                    device=fused_feat.device,
                    dtype=fused_feat.dtype,
                )
                sim_orig = _cosine_diag_similarity(fused_feat, orig_text_feat)
                sim_pseudo = _cosine_diag_similarity(fused_feat, pseudo_feat)
                consistency_gap = (sim_orig - sim_pseudo).abs()
                terms["loss_rde_consensus"] = (
                    active_weight * consistency_gap.to(active_weight.dtype)
                ).mean()

    if use_attr:
        rgb_valid = ret.get("mga_rgb_color_valid", batch.get("mga_has_rgb_color", None))
        t_valid = ret.get("mga_t_shape_valid", batch.get("mga_has_t_shape", None))
        terms["mga_valid_rgb_color_ratio"] = _mga_valid_ratio(rgb_valid, reference)
        terms["mga_valid_t_shape_ratio"] = _mga_valid_ratio(t_valid, reference)

        if (
            float(getattr(args, "lambda_mga_rgb_color", 0.0)) > 0.0
            and "mga_rgb_color_text_feat" in ret
        ):
            rgb_feat, rgb_source = _get_first_tensor_from_ret(
                ret, ("rgb_mu", "v_rgb_c", "rgb_det_feat")
            )
            if rgb_feat is None:
                _mga_warn_once(
                    "rgb_feature_missing",
                    "[MGA] skipping loss_mga_rgb_color because no RGB global "
                    "feature key is available (tried rgb_mu, v_rgb_c, rgb_det_feat).",
                )
            else:
                image_feat, text_feat, weights, zero = _mga_subset_for_alignment(
                    rgb_feat,
                    ret["mga_rgb_color_text_feat"],
                    rgb_valid,
                    ret.get("mga_rgb_color_weight", None),
                    min_valid=2,
                    warn_key="rgb_color_shape",
                )
                if image_feat is not None:
                    pos_sim = _cosine_diag_similarity(image_feat, text_feat)
                    effective_weight, rde_stats = get_rde_effective_weight(
                        weights,
                        pos_sim=pos_sim,
                        args=args,
                        epoch=epoch,
                    )
                    if use_rde:
                        _merge_rde_stats_into_terms(terms, rde_stats)
                    terms["loss_mga_rgb_color"] = weighted_sdm_or_contrastive_loss(
                        image_feat,
                        text_feat,
                        pids,
                        effective_weight,
                        temperature=temperature,
                    )
                    terms["mga_rgb_color_feature_source"] = rgb_source
                else:
                    terms["loss_mga_rgb_color"] = zero

        if (
            float(getattr(args, "lambda_mga_t_shape", 0.0)) > 0.0
            and "mga_t_shape_text_feat" in ret
        ):
            thermal_feat, thermal_source = _get_first_tensor_from_ret(
                ret, ("thermal_mu", "v_t_c", "thermal_det_feat")
            )
            if thermal_feat is None:
                _mga_warn_once(
                    "thermal_feature_missing",
                    "[MGA] skipping loss_mga_t_shape because no thermal global "
                    "feature key is available (tried thermal_mu, v_t_c, thermal_det_feat).",
                )
            else:
                image_feat, text_feat, weights, zero = _mga_subset_for_alignment(
                    thermal_feat,
                    ret["mga_t_shape_text_feat"],
                    t_valid,
                    ret.get("mga_t_shape_weight", None),
                    min_valid=2,
                    warn_key="t_shape_shape",
                )
                if image_feat is not None:
                    pos_sim = _cosine_diag_similarity(image_feat, text_feat)
                    effective_weight, rde_stats = get_rde_effective_weight(
                        weights,
                        pos_sim=pos_sim,
                        args=args,
                        epoch=epoch,
                    )
                    if use_rde:
                        _merge_rde_stats_into_terms(terms, rde_stats)
                    terms["loss_mga_t_shape"] = weighted_sdm_or_contrastive_loss(
                        image_feat,
                        text_feat,
                        pids,
                        effective_weight,
                        temperature=temperature,
                    )
                    terms["mga_t_shape_feature_source"] = thermal_source
                else:
                    terms["loss_mga_t_shape"] = zero

    if (
        use_negative
        and float(getattr(args, "lambda_mga_negative", 0.0)) > 0.0
        and "fused_det_feat" in ret
        and "mga_negative_text_feat" in ret
    ):
        neg_valid = ret.get("mga_negative_valid", batch.get("mga_has_negative", None))
        image_feat, text_feat, weights, zero = _mga_subset_for_alignment(
            ret["fused_det_feat"],
            ret["mga_negative_text_feat"],
            neg_valid,
            ret.get("mga_negative_weight", None),
            min_valid=1,
            warn_key="negative_shape",
        )
        if image_feat is not None:
            image_clean = torch.nan_to_num(
                image_feat.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            text_clean = torch.nan_to_num(
                text_feat.float(), nan=0.0, posinf=0.0, neginf=0.0
            )
            image_norm = F.normalize(image_clean, p=2, dim=1)
            text_norm = F.normalize(text_clean, p=2, dim=1)
            sim = (image_norm * text_norm).sum(dim=1)
            margin = float(getattr(args, "mga_neg_margin", 0.2))
            terms["loss_mga_negative"] = (
                F.relu(sim - margin) * weights.to(sim.device, dtype=sim.dtype)
            ).mean()
        else:
            terms["loss_mga_negative"] = zero

    return terms


def _add_mga_aux_to_total_loss(args, total_loss, ret):
    if bool(getattr(args, "use_mga_pseudo", False)):
        lambda_pseudo = float(getattr(args, "lambda_mga_pseudo_sdm", 0.0))
        if lambda_pseudo > 0.0 and "loss_mga_pseudo_sdm" in ret:
            total_loss = total_loss + lambda_pseudo * ret["loss_mga_pseudo_sdm"]
    if bool(getattr(args, "use_mga_attr_align", False)):
        lambda_rgb = float(getattr(args, "lambda_mga_rgb_color", 0.0))
        lambda_t = float(getattr(args, "lambda_mga_t_shape", 0.0))
        if lambda_rgb > 0.0 and "loss_mga_rgb_color" in ret:
            total_loss = total_loss + lambda_rgb * ret["loss_mga_rgb_color"]
        if lambda_t > 0.0 and "loss_mga_t_shape" in ret:
            total_loss = total_loss + lambda_t * ret["loss_mga_t_shape"]
    if bool(getattr(args, "use_mga_negative_loss", False)):
        lambda_negative = float(getattr(args, "lambda_mga_negative", 0.0))
        if lambda_negative > 0.0 and "loss_mga_negative" in ret:
            total_loss = total_loss + lambda_negative * ret["loss_mga_negative"]
    if bool(getattr(args, "use_rde_robust", False)):
        lambda_consensus = float(getattr(args, "lambda_rde_consensus", 0.0))
        if lambda_consensus > 0.0 and "loss_rde_consensus" in ret:
            total_loss = total_loss + lambda_consensus * ret["loss_rde_consensus"]
    return total_loss


def _compute_inbatch_hard_negative_terms(args, ret, batch, epoch):
    if not bool(getattr(args, "use_hard_negative_loss", False)):
        return {}

    zero = batch["images"].new_zeros(())
    lambda_hn = float(getattr(args, "lambda_hn", 0.0))
    hn_direction = str(getattr(args, "hn_direction", "t2i")).lower()

    def _inactive_terms():
        return {
            "loss_hn": zero,
            "loss_hn_weighted": zero,
            "hn_active": False,
            "hn_num_valid_queries": 0.0,
            "hn_mean_pos_sim": zero,
            "hn_mean_neg_sim": zero,
            "hn_margin_violation_rate": zero,
            # Legacy aliases keep existing meter/debug surfaces usable.
            "hn_avg_pos_sim": zero,
            "hn_avg_neg_sim": zero,
            "hn_violation_rate": zero,
            "hn_t2i_loss": zero,
            "hn_i2t_loss": zero,
            "hn_t2i_valid": 0.0,
            "hn_i2t_valid": 0.0,
            "hn_direction": hn_direction,
            "hn_image_feature_source": "inactive",
            "hn_text_feature_source": "inactive",
        }

    if int(epoch) <= int(getattr(args, "hn_warmup_epochs", 5)):
        return _inactive_terms()

    text_feat, text_source = _get_first_tensor_from_ret(
        ret,
        ["text_prob_feat", "text_det_feat", "text_mu"],
    )
    if bool(getattr(args, "hn_use_fused_image_feature", False)):
        image_keys = ["fused_det_feat", "fused_mu", "fused_prob_feat", "v_rgb_c"]
    else:
        image_keys = ["fused_prob_feat", "fused_det_feat", "fused_mu", "v_rgb_c"]
    image_feat, image_source = _get_first_tensor_from_ret(ret, image_keys)
    pids = batch.get("pids", None)
    if text_feat is None or image_feat is None or pids is None:
        raise KeyError(
            "use_hard_negative_loss=True requires text/image global features "
            "and batch['pids']."
        )

    hn_out = objectives.compute_inbatch_hard_negative_loss(
        text_features=text_feat,
        image_features=image_feat,
        pids=pids,
        margin=float(getattr(args, "hn_margin", 0.05)),
        topk=int(getattr(args, "hn_topk", 1)),
        direction=hn_direction,
        detach_neg=bool(getattr(args, "hn_detach_neg", False)),
    )
    loss_hn = hn_out["loss"]
    mean_pos_sim = hn_out["mean_pos_sim"]
    mean_neg_sim = hn_out["mean_neg_sim"]
    violation_rate = hn_out["margin_violation_rate"]
    return {
        "loss_hn": loss_hn,
        "loss_hn_weighted": loss_hn * lambda_hn,
        "hn_active": True,
        "hn_num_valid_queries": hn_out["num_valid_queries"],
        "hn_mean_pos_sim": mean_pos_sim,
        "hn_mean_neg_sim": mean_neg_sim,
        "hn_margin_violation_rate": violation_rate,
        # Legacy aliases keep existing hard-negative meters/debug checks readable.
        "hn_avg_pos_sim": mean_pos_sim,
        "hn_avg_neg_sim": mean_neg_sim,
        "hn_violation_rate": violation_rate,
        "hn_t2i_loss": hn_out.get("t2i_loss", zero),
        "hn_i2t_loss": hn_out.get("i2t_loss", zero),
        "hn_t2i_valid": hn_out.get("t2i_valid", 0.0),
        "hn_i2t_valid": hn_out.get("i2t_valid", 0.0),
        "hn_direction": hn_direction,
        "hn_image_feature_source": image_source,
        "hn_text_feature_source": text_source,
    }


def _add_inbatch_hard_negative_to_total_loss(args, total_loss, ret, epoch):
    if not bool(getattr(args, "use_hard_negative_loss", False)):
        return total_loss
    if int(epoch) <= int(getattr(args, "hn_warmup_epochs", 5)):
        return total_loss
    if "loss_hn_weighted" in ret:
        return total_loss + ret["loss_hn_weighted"]
    if "loss_hn" in ret:
        return total_loss + float(getattr(args, "lambda_hn", 0.0)) * ret["loss_hn"]
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


MDRC_IMAGE_FEATURE_KEYS = (
    "fused_det_feat",
    "fused_prob_feat",
    "fused_mu",
    "fused_feat_det",
    "fused_feat_sampled",
)
def _mdrc_enabled(args):
    return bool(getattr(args, "mdrc_enable", False))


def _mdrc_loss_scale(args):
    if args is None:
        return 1.0
    return float(getattr(args, "mdrc_loss_scale", 1.0))


def _resolve_mdrc_lambda(args, epoch):
    if not _mdrc_enabled(args):
        return 0.0
    target_lambda = float(getattr(args, "mdrc_lambda", 0.05))
    if target_lambda <= 0.0:
        return 0.0
    warmup_epoch = max(0, int(getattr(args, "mdrc_warmup_epoch", 3)))
    cur_epoch = int(epoch)
    if warmup_epoch > 0 and cur_epoch < warmup_epoch:
        return 0.0
    if warmup_epoch <= 0:
        return target_lambda
    ramp_epochs = max(1, warmup_epoch)
    ramp_step = max(0, cur_epoch - warmup_epoch + 1)
    ramp_factor = min(1.0, float(ramp_step) / float(ramp_epochs))
    return target_lambda * ramp_factor


def _extract_mdrc_feature(ret, candidate_keys, feature_name, required=True):
    for key in candidate_keys:
        value = ret.get(key, None)
        if torch.is_tensor(value) and value.dim() == 2:
            return value, key
    if not required:
        return None, "none"
    raise KeyError(
        f"MDRC requires a 2D {feature_name} feature in one of "
        f"{list(candidate_keys)}, but got keys: {list(ret.keys())}."
    )


def _empty_mdrc_terms(reference, lambda_effective=0.0, loss_scale=1.0):
    zero = reference.new_zeros(())
    return {
        "loss_mdrc": zero,
        "loss_mdrc_feature": zero,
        "loss_mdrc_similarity": zero,
        "loss_mdrc_weighted": zero,
        "weighted_mdrc": zero,
        "effective_weighted_mdrc": zero,
        "mdrc_lambda": zero.new_tensor(float(lambda_effective)),
        "mdrc_loss_scale": zero.new_tensor(float(loss_scale)),
        "mdrc_active": zero,
        "mdrc_feature_only_degraded": zero,
        "mdrc_delta_rgb": zero,
        "mdrc_delta_ir": zero,
        "mdrc_cos_clean_degraded": zero,
        "mdrc_degraded_requires_grad": False,
        "mdrc_image_feature_key": "none",
        "mdrc_text_feature_key": "none",
    }


def _compute_mdrc_terms(args, model, batch, clean_ret, epoch):
    """
    MDRC v1.1: training-stage clean-degraded feature consistency.

    The clean batch keeps the baseline loss path. When enabled after warmup, a
    degraded RGB/T view uses an image-only forward and is aligned to detached
    clean image features. The degraded branch does not compute MLM/ID/SDM losses.
    """
    reference = batch["images"]
    lambda_effective = _resolve_mdrc_lambda(args, epoch)
    if lambda_effective <= 0.0:
        return _empty_mdrc_terms(
            reference,
            lambda_effective=lambda_effective,
            loss_scale=_mdrc_loss_scale(args),
        )

    clean_img_feat, image_feature_key = _extract_mdrc_feature(
        clean_ret,
        MDRC_IMAGE_FEATURE_KEYS,
        "image",
        required=True,
    )
    degraded_rgb, degraded_ir = degrade_rgbt_batch(
        batch["images"],
        batch.get("thermal_images", None),
        prob=float(getattr(args, "mdrc_prob", 0.5)),
        modes=getattr(args, "mdrc_degrade_modes", None),
    )
    delta_rgb = (degraded_rgb.detach().float() - batch["images"].detach().float()).abs().mean()
    if torch.is_tensor(degraded_ir) and "thermal_images" in batch:
        delta_ir = (
            degraded_ir.detach().float()
            - batch["thermal_images"].detach().float()
        ).abs().mean()
    else:
        delta_ir = delta_rgb.new_zeros(())

    degraded_batch = dict(batch)
    degraded_batch["images"] = degraded_rgb
    if "thermal_images" in batch:
        degraded_batch["thermal_images"] = degraded_ir

    degraded_ret = model(degraded_batch, mdrc_feature_only=True)
    degraded_img_feat, degraded_image_feature_key = _extract_mdrc_feature(
        degraded_ret,
        (image_feature_key,) + tuple(
            key for key in MDRC_IMAGE_FEATURE_KEYS if key != image_feature_key
        ),
        "degraded image",
        required=True,
    )
    degraded_requires_grad = bool(degraded_img_feat.requires_grad)
    if not degraded_requires_grad:
        raise RuntimeError(
            "MDRC degraded image feature does not require grad. "
            "The degraded feature-only branch must remain in the autograd graph."
        )

    loss_feature = mdrc_feature_consistency(clean_img_feat.detach(), degraded_img_feat)
    loss_similarity = loss_feature.new_zeros(())
    loss_mdrc = loss_feature
    lambda_tensor = loss_mdrc.new_tensor(float(lambda_effective))
    scale_tensor = loss_mdrc.new_tensor(float(_mdrc_loss_scale(args)))
    loss_weighted = lambda_tensor * scale_tensor * loss_mdrc
    cos_clean_degraded = F.cosine_similarity(
        F.normalize(clean_img_feat.detach().float(), p=2, dim=-1, eps=1e-12),
        F.normalize(degraded_img_feat.detach().float(), p=2, dim=-1, eps=1e-12),
        dim=-1,
    ).mean()

    return {
        "loss_mdrc": loss_mdrc,
        "loss_mdrc_feature": loss_feature,
        "loss_mdrc_similarity": loss_similarity,
        "loss_mdrc_weighted": loss_weighted,
        "weighted_mdrc": loss_weighted,
        "effective_weighted_mdrc": loss_weighted,
        "mdrc_lambda": lambda_tensor,
        "mdrc_loss_scale": scale_tensor,
        "mdrc_active": loss_mdrc.new_ones(()),
        "mdrc_feature_only_degraded": loss_mdrc.new_ones(()),
        "mdrc_delta_rgb": delta_rgb,
        "mdrc_delta_ir": delta_ir,
        "mdrc_cos_clean_degraded": cos_clean_degraded,
        "mdrc_degraded_requires_grad": degraded_requires_grad,
        "mdrc_image_feature_key": degraded_image_feature_key,
        "mdrc_text_feature_key": "disabled_v1_1",
    }


def _add_mdrc_to_total_loss(args, total_loss, mdrc_terms):
    if not _mdrc_enabled(args):
        return total_loss
    if not mdrc_terms:
        return total_loss
    return (
        total_loss
        + mdrc_terms["mdrc_lambda"]
        * mdrc_terms["mdrc_loss_scale"]
        * mdrc_terms["loss_mdrc"]
    )


def _log_mdrc_debug_status(
    logger,
    args,
    epoch,
    n_iter,
    baseline_loss,
    total_loss,
    mdrc_terms=None,
):
    reference = baseline_loss if torch.is_tensor(baseline_loss) else total_loss
    empty_terms = _empty_mdrc_terms(
        reference,
        lambda_effective=_resolve_mdrc_lambda(args, epoch),
        loss_scale=_mdrc_loss_scale(args),
    )
    terms = mdrc_terms if mdrc_terms is not None else empty_terms
    active = bool(_to_scalar(terms.get("mdrc_active", 0)) > 0.0)
    reason = "active" if active else "warmup_or_disabled"
    logger.info(
        "[MDRC_DEBUG] active=%s reason=%s epoch=%d iter=%d "
        "baseline_loss=%.6f loss_mdrc=%.6f lambda=%.6f "
        "mdrc_loss_scale=%.6f weighted_mdrc=%.6f "
        "effective_weighted_mdrc=%.6f total_loss=%.6f delta_rgb=%.6f "
        "delta_ir=%.6f cos_clean_degraded=%.6f degraded_requires_grad=%s "
        "feature_only_degraded=%s image_feature_key=%s text_feature_key=%s",
        active,
        reason,
        int(epoch),
        int(n_iter) + 1,
        _to_scalar(baseline_loss),
        _to_scalar(terms.get("loss_mdrc", 0)),
        _to_scalar(terms.get("mdrc_lambda", 0)),
        _to_scalar(terms.get("mdrc_loss_scale", _mdrc_loss_scale(args))),
        _to_scalar(terms.get("weighted_mdrc", terms.get("loss_mdrc_weighted", 0))),
        _to_scalar(
            terms.get(
                "effective_weighted_mdrc",
                terms.get("weighted_mdrc", terms.get("loss_mdrc_weighted", 0)),
            )
        ),
        _to_scalar(total_loss),
        _to_scalar(terms.get("mdrc_delta_rgb", 0)),
        _to_scalar(terms.get("mdrc_delta_ir", 0)),
        _to_scalar(terms.get("mdrc_cos_clean_degraded", 0)),
        bool(terms.get("mdrc_degraded_requires_grad", False)),
        bool(_to_scalar(terms.get("mdrc_feature_only_degraded", 0)) > 0.0),
        str(terms.get("mdrc_image_feature_key", "none")),
        str(terms.get("mdrc_text_feature_key", "none")),
    )


def log_mdrc_config(logger, args):
    if not _mdrc_enabled(args):
        return
    logger.info(
        "[MDRC] enable=%s feature_only_degraded=True lambda=%.6f loss_scale=%.6f prob=%.6f warmup_epoch=%d temp=%.6f modes=%s",
        bool(getattr(args, "mdrc_enable", False)),
        float(getattr(args, "mdrc_lambda", 0.05)),
        _mdrc_loss_scale(args),
        float(getattr(args, "mdrc_prob", 0.5)),
        int(getattr(args, "mdrc_warmup_epoch", 3)),
        float(getattr(args, "mdrc_temp", 0.07)),
        list(getattr(args, "mdrc_degrade_modes", [])),
    )


# Fixed batch-level probability of applying the ablation degradation augmentation.
DEGRADE_AUG_PROB = 0.5


def _degrade_aug_mode(args):
    return str(getattr(args, "degrade_aug_mode", "none"))


def _degrade_aug_enabled(args):
    # Ablation-only single-branch degradation augmentation. It is disabled whenever
    # MDRC is on, so MDRC's clean-degraded consistency training is never altered.
    return _degrade_aug_mode(args) != "none" and not _mdrc_enabled(args)


def log_degrade_aug_config(logger, args):
    # Only log (and thus only ever change behavior) when a mode is requested; the
    # default "none" leaves the baseline path and its logs completely unchanged.
    if _degrade_aug_mode(args) == "none":
        return
    logger.info(
        "[DEGRADE_AUG] mode=%s prob=%.1f mdrc_enable=%s active=%s",
        _degrade_aug_mode(args),
        DEGRADE_AUG_PROB,
        bool(_mdrc_enabled(args)),
        bool(_degrade_aug_enabled(args)),
    )


def _apply_degrade_aug_to_batch(args, batch):
    """
    Ablation degradation augmentation (single branch, no consistency loss).

    With probability DEGRADE_AUG_PROB, degrade the image inputs in-place on the
    batch dict so the degraded tensors replace the originals for ALL downstream
    baseline losses (SDM/MLM/ID). Returns (applied, mode_name) for logging.

    - "mdrc_style": sample one of the 6 MDRC modes uniformly and apply it to the
      whole batch (reuses degrade_rgbt_batch with prob=1.0).
    - "moddrop": zero out a whole modality (RGB or IR) with equal probability
      (reuses degrade_rgbt_batch rgb_drop/ir_drop with prob=1.0).
    """
    if random.random() >= DEGRADE_AUG_PROB:
        return False, "none"

    mode = _degrade_aug_mode(args)
    rgb_imgs = batch["images"]
    ir_imgs = batch.get("thermal_images", None)

    if mode == "mdrc_style":
        chosen = random.choice(list(ALLOWED_DEGRADE_MODES))
        applied_name = chosen
    elif mode == "moddrop":
        chosen = random.choice(["rgb_drop", "ir_drop"])
        applied_name = "moddrop_rgb" if chosen == "rgb_drop" else "moddrop_ir"
    else:
        return False, "none"

    degraded_rgb, degraded_ir = degrade_rgbt_batch(
        rgb_imgs, ir_imgs, prob=1.0, modes=[chosen]
    )
    batch["images"] = degraded_rgb
    if torch.is_tensor(ir_imgs):
        batch["thermal_images"] = degraded_ir
    return True, applied_name


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
    log_mdrc_config(logger, args)
    log_degrade_aug_config(logger, args)

    meters = {
        "loss": AverageMeter(),
        "loss_total": AverageMeter(),
        "loss_baseline": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "loss_sdm_prob": AverageMeter(),
        "loss_sdm_det": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "loss_mlm": AverageMeter(),
        "loss_attr_aux_raw": AverageMeter(),
        "loss_attr_aux_weighted": AverageMeter(),
        "attr_active": AverageMeter(),
        "attr_valid_gender": AverageMeter(),
        "attr_valid_upper_color": AverageMeter(),
        "attr_valid_lower_color": AverageMeter(),
        "attr_valid_shoe_color": AverageMeter(),
        "attr_valid_bag_color": AverageMeter(),
        "attr_valid_bag_present": AverageMeter(),
        "attr_valid_hair_length": AverageMeter(),
        "loss_uspc_slot_local_raw": AverageMeter(),
        "loss_uspc_slot_local_weighted": AverageMeter(),
        "uspc_slot_local_active": AverageMeter(),
        "uspc_slot_valid_upper_color": AverageMeter(),
        "uspc_slot_valid_lower_color": AverageMeter(),
        "uspc_slot_valid_shoe_color": AverageMeter(),
        "uspc_slot_valid_bag_color": AverageMeter(),
        "uspc_slot_valid_bag_present": AverageMeter(),
        "uspc_slot_valid_hair_length": AverageMeter(),
        "loss_uspc": AverageMeter(),
        "loss_uspc_raw": AverageMeter(),
        "loss_uspc_color": AverageMeter(),
        "loss_uspc_struct": AverageMeter(),
        "loss_uspc_full": AverageMeter(),
        "uspc_rgb_conf_mean": AverageMeter(),
        "uspc_thermal_conf_mean": AverageMeter(),
        "uspc_fused_conf_mean": AverageMeter(),
        "uspc_active": AverageMeter(),
        "loss_dcalign": AverageMeter(),
        "loss_dcalign_raw": AverageMeter(),
        "loss_dcalign_color": AverageMeter(),
        "loss_dcalign_struct": AverageMeter(),
        "loss_dcalign_full": AverageMeter(),
        "dcalign_lambda_effective": AverageMeter(),
        "dcalign_active": AverageMeter(),
        "dcalign_color_weight": AverageMeter(),
        "dcalign_struct_weight": AverageMeter(),
        "dcalign_full_weight": AverageMeter(),
        "loss_mdrc": AverageMeter(),
        "loss_mdrc_feature": AverageMeter(),
        "loss_mdrc_similarity": AverageMeter(),
        "loss_mdrc_weighted": AverageMeter(),
        "effective_weighted_mdrc": AverageMeter(),
        "mdrc_lambda": AverageMeter(),
        "mdrc_loss_scale": AverageMeter(),
        "mdrc_active": AverageMeter(),
        "mdrc_feature_only_degraded": AverageMeter(),
        "loss_rgb_txt": AverageMeter(),
        "loss_th_txt": AverageMeter(),
        "loss_txt_cons": AverageMeter(),
        "loss_mga_pseudo_sdm": AverageMeter(),
        "loss_mga_rgb_color": AverageMeter(),
        "loss_mga_t_shape": AverageMeter(),
        "loss_mga_negative": AverageMeter(),
        "loss_rde_consensus": AverageMeter(),
        "mga_valid_pseudo_ratio": AverageMeter(),
        "mga_valid_rgb_color_ratio": AverageMeter(),
        "mga_valid_t_shape_ratio": AverageMeter(),
        "rde_clean_ratio": AverageMeter(),
        "rde_uncertain_ratio": AverageMeter(),
        "rde_noisy_ratio": AverageMeter(),
        "rde_mean_conf": AverageMeter(),
        "conf_hn_active_ratio": AverageMeter(),
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
        "hn_mean_pos_sim": AverageMeter(),
        "hn_mean_neg_sim": AverageMeter(),
        "hn_margin_violation_rate": AverageMeter(),
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
    use_uspc = _uspc_enabled(args)
    use_dcalign_lite = _dcalign_lite_enabled(args)
    use_attr_aux = _attr_aux_enabled(args)
    use_uspc_slot_local = _uspc_slot_local_enabled(args)
    uspc_text_tokenizer = (
        SimpleTokenizer() if (use_uspc or use_dcalign_lite) else None
    )
    use_modality_text_prompt = bool(
        getattr(args, "use_modality_text_prompt", False)
    )
    mtp_text_tokenizer = (
        SimpleTokenizer() if use_modality_text_prompt else None
    )

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        uspc_debug_logged_this_epoch = False
        dcalign_debug_logged_this_epoch = False
        mdrc_debug_logged_this_epoch = False
        degrade_aug_logged_this_epoch = False
        attr_debug_batches_logged_this_epoch = 0
        uspc_slot_debug_batches_logged_this_epoch = 0
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

            if _degrade_aug_enabled(args):
                # Ablation single-branch degradation augmentation: replace the
                # image inputs before the forward so the degraded tensors flow
                # into all baseline losses. No second branch, no consistency loss.
                degrade_applied, degrade_mode_name = _apply_degrade_aug_to_batch(
                    args, batch
                )
                if not degrade_aug_logged_this_epoch:
                    logger.info(
                        "[DEGRADE_AUG] epoch=%d iter=%d applied=%s mode=%s",
                        epoch,
                        n_iter + 1,
                        bool(degrade_applied),
                        degrade_mode_name,
                    )
                    degrade_aug_logged_this_epoch = True

            ret = model(batch)
            uspc_debug_should_log = (
                use_uspc
                and bool(getattr(args, "debug_uspc", False))
                and n_iter == 0
                and (not uspc_debug_logged_this_epoch)
            )
            dcalign_debug_should_log = (
                use_dcalign_lite
                and bool(getattr(args, "dcalign_debug", False))
                and n_iter == 0
                and (not dcalign_debug_logged_this_epoch)
            )
            uspc_inputs = None
            if _semantic_alignment_should_encode_views(args, epoch):
                uspc_inputs = _build_uspc_forward_inputs(
                    args,
                    model,
                    batch,
                    ret,
                    tokenizer=uspc_text_tokenizer,
                )
            ret.update(_compute_uncertainty_alignment_terms(args, ret, batch))
            original_total_loss = _build_total_loss(args, ret, epoch=epoch)
            ret.update(_compute_rgbt_moe_adapter_terms(args, ret, batch, epoch))
            if getattr(args, "use_hard_negative_loss", False):
                ret.update(_compute_inbatch_hard_negative_terms(args, ret, batch, epoch))
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
            if getattr(args, "use_hard_negative_loss", False):
                total_loss = _add_inbatch_hard_negative_to_total_loss(
                    args, total_loss, ret, epoch
                )
            if getattr(args, "use_hardneg_aux", False):
                total_loss = _add_hardneg_aux_to_total_loss(
                    args, total_loss, ret, epoch
                )
            if getattr(args, "use_offline_hardneg", False):
                total_loss = _add_offline_hardneg_to_total_loss(
                    args, total_loss, ret, epoch
                )
            total_loss = _add_mtp_terms_to_total_loss(args, total_loss, ret, epoch)
            mga_terms = _compute_mga_aux_terms(args, ret, batch, epoch)
            if mga_terms:
                ret.update(mga_terms)
                total_loss = _add_mga_aux_to_total_loss(args, total_loss, ret)
            attr_terms = None
            if use_attr_aux:
                attr_terms = _compute_attr_aux_terms(args, ret, batch, epoch)
                ret.update(attr_terms)
                total_loss = _add_attr_aux_to_total_loss(args, total_loss, ret)
                if (
                    bool(getattr(args, "debug_attr_aux", False))
                    and attr_debug_batches_logged_this_epoch < 2
                ):
                    logger.info(
                        "[ATTR_AUX][DEBUG] epoch=%d iter=%d active=%d "
                        "lambda_attr_aux=%.6f loss_raw=%.6f loss_weighted=%.6f "
                        "feature_source=%s loss_type=%s stats=%s",
                        epoch,
                        n_iter + 1,
                        int(round(_to_scalar(attr_terms.get("attr_active", 0)))),
                        float(getattr(args, "lambda_attr_aux", 0.0)),
                        _to_scalar(attr_terms.get("loss_attr_aux_raw", 0)),
                        _to_scalar(attr_terms.get("loss_attr_aux_weighted", 0)),
                        str(attr_terms.get("attr_feature_source", "unknown")),
                        str(attr_terms.get("attr_loss_type", "unknown")),
                        attr_terms.get("attr_parser_stats", {}),
                    )
                    attr_debug_batches_logged_this_epoch += 1
            uspc_slot_terms = None
            if use_uspc_slot_local:
                uspc_slot_terms = _compute_uspc_slot_local_terms(
                    args,
                    ret,
                    batch,
                    epoch,
                )
                ret.update(uspc_slot_terms)
                total_loss = _add_uspc_slot_local_to_total_loss(
                    args,
                    total_loss,
                    ret,
                )
                if (
                    bool(getattr(args, "debug_uspc_slot_local", False))
                    and uspc_slot_debug_batches_logged_this_epoch < 2
                ):
                    logger.info(
                        "[USPC_SLOT_LOCAL][DEBUG] epoch=%d iter=%d active=%d "
                        "lambda=%.6f loss_raw=%.6f loss_weighted=%.6f "
                        "patch_source=%s loss_type=%s stats=%s",
                        epoch,
                        n_iter + 1,
                        int(
                            round(
                                _to_scalar(
                                    uspc_slot_terms.get(
                                        "uspc_slot_local_active",
                                        0,
                                    )
                                )
                            )
                        ),
                        float(getattr(args, "lambda_uspc_slot_local", 0.0)),
                        _to_scalar(
                            uspc_slot_terms.get("loss_uspc_slot_local_raw", 0)
                        ),
                        _to_scalar(
                            uspc_slot_terms.get(
                                "loss_uspc_slot_local_weighted",
                                0,
                            )
                        ),
                        str(uspc_slot_terms.get("uspc_slot_patch_source", "unknown")),
                        str(uspc_slot_terms.get("uspc_slot_loss_type", "unknown")),
                        uspc_slot_terms.get("uspc_slot_parser_stats", {}),
                    )
                    uspc_slot_debug_batches_logged_this_epoch += 1
            uspc_terms = None
            if _uspc_should_encode_views(args):
                if uspc_inputs is None:
                    raise RuntimeError(
                        "USPC loss is requested but semantic text view inputs were "
                        "not built."
                    )
                uspc_terms = compute_uspc_loss(
                    rgb_feat=uspc_inputs["rgb_feat"],
                    thermal_feat=uspc_inputs["thermal_feat"],
                    fused_feat=uspc_inputs["fused_feat"],
                    full_text_feat=uspc_inputs["full_text_feat"],
                    color_text_feat=uspc_inputs["color_text_feat"],
                    struct_text_feat=uspc_inputs["struct_text_feat"],
                    pids=batch.get("pids", None),
                    rgb_logvar=uspc_inputs.get("rgb_logvar", None),
                    thermal_logvar=uspc_inputs.get("thermal_logvar", None),
                    fused_logvar=uspc_inputs.get("fused_logvar", None),
                    args=args,
                    epoch=epoch,
                )
                # compute_uspc_loss already applies lambda_uspc internally.
                total_loss = total_loss + uspc_terms["loss_uspc"]

            dcalign_terms = None
            if use_dcalign_lite:
                if _dcalign_should_encode_views(args, epoch):
                    if uspc_inputs is None:
                        raise RuntimeError(
                            "DCAlign-lite loss is requested but semantic text view "
                            "inputs were not built."
                        )
                    dcalign_terms = compute_dcalign_lite_loss(
                        rgb_feat=uspc_inputs["rgb_feat"],
                        thermal_feat=uspc_inputs["thermal_feat"],
                        fused_feat=uspc_inputs["fused_feat"],
                        full_text_feat=uspc_inputs["full_text_feat"],
                        color_text_feat=uspc_inputs["color_text_feat"],
                        struct_text_feat=uspc_inputs["struct_text_feat"],
                        pids=batch.get("pids", None),
                        args=args,
                        epoch=epoch,
                    )
                else:
                    dcalign_terms = _empty_dcalign_lite_terms(
                        args,
                        batch["images"],
                        epoch,
                    )
                ret.update(dcalign_terms)
                total_loss = total_loss + dcalign_terms["loss_dcalign"]

            baseline_total_loss = total_loss
            mdrc_terms = None
            if _mdrc_enabled(args):
                mdrc_terms = _compute_mdrc_terms(
                    args,
                    model,
                    batch,
                    ret,
                    epoch,
                )
                ret.update(mdrc_terms)
                total_loss = _add_mdrc_to_total_loss(args, total_loss, mdrc_terms)

            if not mdrc_debug_logged_this_epoch:
                _log_mdrc_debug_status(
                    logger=logger,
                    args=args,
                    epoch=epoch,
                    n_iter=n_iter,
                    baseline_loss=baseline_total_loss,
                    total_loss=total_loss,
                    mdrc_terms=mdrc_terms,
                )
                mdrc_debug_logged_this_epoch = True

            if uspc_debug_should_log:
                if uspc_inputs is None:
                    _log_uspc_debug_examples(
                        logger,
                        _uspc_normalize_raw_captions(batch.get("raw_captions", [])),
                        uspc_inputs=None,
                    )
                else:
                    _log_uspc_debug_examples(
                        logger,
                        uspc_inputs["raw_captions"],
                        uspc_inputs=uspc_inputs,
                    )
                if uspc_terms is not None:
                    _log_uspc_debug_status(logger, args, uspc_terms=uspc_terms)
                uspc_debug_logged_this_epoch = True

            if dcalign_debug_should_log:
                if uspc_inputs is None:
                    _log_uspc_debug_examples(
                        logger,
                        _uspc_normalize_raw_captions(batch.get("raw_captions", [])),
                        uspc_inputs=None,
                    )
                else:
                    _log_uspc_debug_examples(
                        logger,
                        uspc_inputs["raw_captions"],
                        uspc_inputs=uspc_inputs,
                    )
                _log_dcalign_debug_status(
                    logger,
                    args,
                    epoch,
                    dcalign_terms=dcalign_terms,
                )
                dcalign_debug_logged_this_epoch = True

            batch_size = batch["images"].shape[0]
            meters["loss"].update(_to_scalar(total_loss), batch_size)
            if mdrc_terms is not None:
                meters["loss_total"].update(_to_scalar(total_loss), batch_size)
                meters["loss_baseline"].update(
                    _to_scalar(baseline_total_loss), batch_size
                )
                for mdrc_loss_name in [
                    "loss_mdrc",
                    "loss_mdrc_feature",
                    "loss_mdrc_similarity",
                    "loss_mdrc_weighted",
                    "effective_weighted_mdrc",
                ]:
                    meters[mdrc_loss_name].update(
                        _to_scalar(mdrc_terms.get(mdrc_loss_name, 0)),
                        batch_size,
                    )
                for mdrc_stat_name in [
                    "mdrc_lambda",
                    "mdrc_loss_scale",
                    "mdrc_active",
                    "mdrc_feature_only_degraded",
                ]:
                    meters[mdrc_stat_name].update(
                        _to_scalar(mdrc_terms.get(mdrc_stat_name, 0)),
                        1,
                    )
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
            if mga_terms:
                for mga_loss_name in [
                    "loss_mga_pseudo_sdm",
                    "loss_mga_rgb_color",
                    "loss_mga_t_shape",
                    "loss_mga_negative",
                    "loss_rde_consensus",
                ]:
                    meters[mga_loss_name].update(
                        _to_scalar(mga_terms.get(mga_loss_name, 0)),
                        batch_size,
                    )
                for mga_ratio_name in [
                    "mga_valid_pseudo_ratio",
                    "mga_valid_rgb_color_ratio",
                    "mga_valid_t_shape_ratio",
                    "rde_clean_ratio",
                    "rde_uncertain_ratio",
                    "rde_noisy_ratio",
                    "rde_mean_conf",
                    "conf_hn_active_ratio",
                ]:
                    meters[mga_ratio_name].update(
                        _to_scalar(mga_terms.get(mga_ratio_name, 0)),
                        1,
                    )
            if attr_terms is not None:
                meters["loss_attr_aux_raw"].update(
                    _to_scalar(attr_terms.get("loss_attr_aux_raw", 0)),
                    batch_size,
                )
                meters["loss_attr_aux_weighted"].update(
                    _to_scalar(attr_terms.get("loss_attr_aux_weighted", 0)),
                    batch_size,
                )
                meters["attr_active"].update(
                    _to_scalar(attr_terms.get("attr_active", 0)),
                    1,
                )
                for attr_name in ATTR_AUX_NAMES:
                    meter_name = f"attr_valid_{attr_name}"
                    meters[meter_name].update(
                        _to_scalar(attr_terms.get(meter_name, 0)),
                        1,
                    )
            if uspc_slot_terms is not None:
                meters["loss_uspc_slot_local_raw"].update(
                    _to_scalar(
                        uspc_slot_terms.get("loss_uspc_slot_local_raw", 0)
                    ),
                    batch_size,
                )
                meters["loss_uspc_slot_local_weighted"].update(
                    _to_scalar(
                        uspc_slot_terms.get("loss_uspc_slot_local_weighted", 0)
                    ),
                    batch_size,
                )
                meters["uspc_slot_local_active"].update(
                    _to_scalar(uspc_slot_terms.get("uspc_slot_local_active", 0)),
                    1,
                )
                for attr_name in USPC_SLOT_LOCAL_ATTR_NAMES:
                    meter_name = f"uspc_slot_valid_{attr_name}"
                    meters[meter_name].update(
                        _to_scalar(uspc_slot_terms.get(meter_name, 0)),
                        1,
                    )
            if uspc_terms is not None:
                for uspc_loss_name in [
                    "loss_uspc",
                    "loss_uspc_raw",
                    "loss_uspc_color",
                    "loss_uspc_struct",
                    "loss_uspc_full",
                ]:
                    meters[uspc_loss_name].update(
                        _to_scalar(uspc_terms.get(uspc_loss_name, 0)),
                        batch_size,
                    )
                for uspc_stat_name in [
                    "uspc_rgb_conf_mean",
                    "uspc_thermal_conf_mean",
                    "uspc_fused_conf_mean",
                    "uspc_active",
                ]:
                    meters[uspc_stat_name].update(
                        _to_scalar(uspc_terms.get(uspc_stat_name, 0)),
                        1,
                    )
            if dcalign_terms is not None:
                for dcalign_loss_name in [
                    "loss_dcalign",
                    "loss_dcalign_raw",
                    "loss_dcalign_color",
                    "loss_dcalign_struct",
                    "loss_dcalign_full",
                ]:
                    meters[dcalign_loss_name].update(
                        _to_scalar(dcalign_terms.get(dcalign_loss_name, 0)),
                        batch_size,
                    )
                for dcalign_stat_name in [
                    "dcalign_lambda_effective",
                    "dcalign_active",
                    "dcalign_color_weight",
                    "dcalign_struct_weight",
                    "dcalign_full_weight",
                ]:
                    meters[dcalign_stat_name].update(
                        _to_scalar(dcalign_terms.get(dcalign_stat_name, 0)),
                        1,
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
            if getattr(args, "use_hard_negative_loss", False) or getattr(args, "use_hardneg_aux", False):
                meters["loss_hn"].update(_to_scalar(ret.get("loss_hn", 0)), batch_size)
                meters["loss_hn_weighted"].update(
                    _to_scalar(ret.get("loss_hn_weighted", 0)), batch_size
                )
                meters["hn_active"].update(_to_scalar(ret.get("hn_active", 0)), 1)
                meters["hn_num_valid_queries"].update(
                    _to_scalar(ret.get("hn_num_valid_queries", 0)), 1
                )
                meters["hn_mean_pos_sim"].update(
                    _to_scalar(ret.get("hn_mean_pos_sim", ret.get("hn_avg_pos_sim", 0))),
                    batch_size,
                )
                meters["hn_mean_neg_sim"].update(
                    _to_scalar(ret.get("hn_mean_neg_sim", ret.get("hn_avg_neg_sim", 0))),
                    batch_size,
                )
                meters["hn_margin_violation_rate"].update(
                    _to_scalar(
                        ret.get(
                            "hn_margin_violation_rate",
                            ret.get("hn_violation_rate", 0),
                        )
                    ),
                    batch_size,
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
                debug_hardneg_stats = bool(
                    getattr(args, "debug_hardneg_stats", False)
                    or getattr(args, "hn_debug", False)
                )
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
                if _mdrc_enabled(args):
                    info_str += (
                        f", mdrc_image_feature_key: {ret.get('mdrc_image_feature_key', 'none')}"
                    )
                    info_str += (
                        f", mdrc_text_feature_key: {ret.get('mdrc_text_feature_key', 'none')}"
                    )
                if bool(getattr(args, "use_proto_calib", False)):
                    info_str += (
                        f", proto_version: {ret.get('proto_version', getattr(args, 'proto_version', 'v1'))}"
                    )
                if bool(getattr(args, "use_hard_negative_loss", False)):
                    info_str += (
                        f", hn_direction: {ret.get('hn_direction', getattr(args, 'hn_direction', 't2i'))}"
                    )
                    if bool(getattr(args, "hn_debug", False)):
                        info_str += (
                            f", hn_image_feature_source: {ret.get('hn_image_feature_source', 'unknown')}"
                        )
                        info_str += (
                            f", hn_text_feature_source: {ret.get('hn_text_feature_source', 'unknown')}"
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

            debug_max_train_batches = int(
                getattr(args, "debug_max_train_batches", 0)
            )
            if debug_max_train_batches > 0 and (n_iter + 1) >= debug_max_train_batches:
                logger.info(
                    "[DEBUG] stop epoch after %d train batches because "
                    "debug_max_train_batches=%d",
                    n_iter + 1,
                    debug_max_train_batches,
                )
                break

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
            evaluator.source = "train_validation"
            log_eval_pipeline_summary(
                logger,
                source="train_validation",
                function="Evaluator.eval",
                img_loader=evaluator.img_loader,
                txt_loader=evaluator.txt_loader,
                args=args,
            )
            if args.distributed:
                top1 = evaluator.eval(model.module.eval())
            else:
                top1 = evaluator.eval(model.eval())
            eval_metrics = getattr(evaluator, "last_eval_result", None)
            if not isinstance(eval_metrics, dict):
                eval_metrics = {}
            metric_r1 = float(eval_metrics.get("R1", top1))
            eval_metrics.setdefault("R1", metric_r1)

            torch.cuda.empty_cache()
            if bool(getattr(args, "save_all_epoch_ckpt", False)):
                _save_metric_checkpoint(
                    checkpointer=checkpointer,
                    args=args,
                    epoch=epoch,
                    r1=metric_r1,
                    metrics=eval_metrics,
                    arguments=arguments,
                    reason="epoch",
                    logger=logger,
                )
            should_verify_ckpt = _should_verify_ckpt_epoch(args, epoch)
            full_epoch_ckpt_path = None
            if bool(getattr(args, "save_full_eval_state_ckpt", False)) or should_verify_ckpt:
                full_epoch_ckpt_path = _save_full_eval_state_checkpoint(
                    model=model,
                    args=args,
                    epoch=epoch,
                    metrics=eval_metrics,
                    reason="epoch",
                    logger=logger,
                )
            if best_top1 < top1:
                best_top1 = top1
                arguments["epoch"] = epoch
                checkpointer.save("best", **arguments)
                if bool(getattr(args, "save_metric_best_ckpt", False)):
                    _save_metric_checkpoint(
                        checkpointer=checkpointer,
                        args=args,
                        epoch=epoch,
                        r1=metric_r1,
                        metrics=eval_metrics,
                        arguments=arguments,
                        reason="best",
                        logger=logger,
                    )
                if bool(getattr(args, "save_full_eval_state_ckpt", False)):
                    _save_full_eval_state_checkpoint(
                        model=model,
                        args=args,
                        epoch=epoch,
                        metrics=eval_metrics,
                        reason="best",
                        logger=logger,
                    )
            if should_verify_ckpt:
                _verify_checkpoint_immediately_after_save(
                    args=args,
                    model=model,
                    evaluator=evaluator,
                    ckpt_path=full_epoch_ckpt_path,
                    epoch=epoch,
                    original_metrics=eval_metrics,
                    epoch_img_local_runtime=epoch_img_local_runtime,
                    logger=logger,
                )

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader, eval_source="test_py"):
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
    log_eval_pipeline_summary(
        logger,
        source=eval_source,
        function="do_inference->Evaluator.eval",
        img_loader=test_img_loader,
        txt_loader=test_txt_loader,
        args=model_args,
    )
    evaluator = Evaluator(test_img_loader, test_txt_loader, source=eval_source)
    evaluator.eval(model.eval())
