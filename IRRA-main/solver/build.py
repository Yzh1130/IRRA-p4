import csv
import logging
import math
import os

import torch

from .lr_scheduler import LRSchedulerWithWarmup


OPTIMIZER_PARAM_GROUP_FIELDS = (
    "epoch",
    "group_index",
    "group_name",
    "parameter_count",
    "parameter_numel",
    "initial_lr",
    "current_lr",
    "weight_decay",
    "scorer_lr_multiplier",
    "parameter_names",
)


def _model_core(model):
    return model.module if hasattr(model, "module") else model


def get_mraf_scorer_named_parameters(model):
    """Locate scorer parameters by module identity, including DP/DDP wrappers."""
    core = _model_core(model)
    fusion = getattr(core, "mraf_fusion", None)
    scorer = getattr(fusion, "scorer", None)
    if scorer is None:
        return []
    names_by_id = {
        id(parameter): name for name, parameter in core.named_parameters()
    }
    result = []
    for parameter in scorer.parameters():
        name = names_by_id.get(id(parameter))
        if name is None:
            raise RuntimeError(
                "An MRAF scorer parameter is not registered on the model."
            )
        result.append((name, parameter))
    return result


def _legacy_parameter_options(args, key):
    lr = args.lr
    weight_decay = args.weight_decay
    if "cross" in key:
        lr = args.lr * args.lr_factor
    if "bias" in key:
        lr = args.lr * args.bias_lr_factor
        weight_decay = args.weight_decay_bias
    if "classifier" in key or "mlm_head" in key:
        lr = args.lr * args.lr_factor
    return float(lr), float(weight_decay)


def _validate_optimizer_parameter_partition(model, param_groups):
    required = [parameter for parameter in model.parameters() if parameter.requires_grad]
    grouped = [
        parameter
        for group in param_groups
        for parameter in group["params"]
    ]
    grouped_ids = [id(parameter) for parameter in grouped]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("Optimizer parameter groups contain duplicate parameters.")
    if set(grouped_ids) != {id(parameter) for parameter in required}:
        raise RuntimeError(
            "Optimizer parameter groups do not cover every requires_grad "
            "parameter exactly once."
        )


def optimizer_param_group_rows(optimizer, model, epoch=0):
    names_by_id = {
        id(parameter): name
        for name, parameter in _model_core(model).named_parameters()
    }
    scorer_ids = {
        id(parameter)
        for _, parameter in get_mraf_scorer_named_parameters(model)
    }
    rows = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = list(group["params"])
        parameter_names = [
            names_by_id.get(id(parameter), f"<unregistered:{id(parameter)}>")
            for parameter in parameters
        ]
        is_scorer = bool(parameters) and all(
            id(parameter) in scorer_ids for parameter in parameters
        )
        rows.append(
            {
                "epoch": int(epoch),
                "group_index": index,
                "group_name": group.get(
                    "group_name",
                    "mraf_scorer" if is_scorer else "legacy_parameter",
                ),
                "parameter_count": len(parameters),
                "parameter_numel": sum(parameter.numel() for parameter in parameters),
                "initial_lr": float(group.get("initial_lr", group["lr"])),
                "current_lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "scorer_lr_multiplier": (
                    float(group.get("mraf_scorer_lr_multiplier", 1.0))
                    if is_scorer
                    else 1.0
                ),
                "parameter_names": ",".join(parameter_names),
            }
        )
    return rows


def append_optimizer_param_groups_tsv(
    optimizer,
    model,
    output_dir,
    epoch=0,
):
    """Append a graph-free optimizer group snapshot on rank zero."""
    if not output_dir or int(os.environ.get("RANK", "0")) != 0:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "optimizer_param_groups.tsv")
    write_header = not os.path.exists(path)
    rows = optimizer_param_group_rows(optimizer, model, epoch=epoch)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OPTIMIZER_PARAM_GROUP_FIELDS,
            delimiter="\t",
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return path


def build_optimizer(args, model):
    # Build parameter groups with different lr/weight_decay policies.
    params = []
    scorer_multiplier = float(
        getattr(args, "mraf_scorer_lr_multiplier", 1.0)
    )
    if not math.isfinite(scorer_multiplier) or scorer_multiplier <= 0.0:
        raise ValueError("--mraf_scorer_lr_multiplier must be > 0.")
    scorer_named = get_mraf_scorer_named_parameters(model)
    scorer_ids = {id(parameter) for _, parameter in scorer_named}
    if scorer_multiplier != 1.0 and not scorer_named:
        raise ValueError(
            "--mraf_scorer_lr_multiplier differs from 1.0, but the model "
            "has no enabled MRAF scorer."
        )

    print(f'Using {args.lr_factor} times learning rate for random init module ')

    scorer_groups = {}
    for key, value in model.named_parameters():
        if not value.requires_grad:
            # Skip frozen parameters.
            continue
        lr, weight_decay = _legacy_parameter_options(args, key)
        if scorer_multiplier != 1.0 and id(value) in scorer_ids:
            group_key = (
                "mraf_scorer_bias"
                if float(weight_decay) == float(args.weight_decay_bias)
                else "mraf_scorer_weight"
            )
            group = scorer_groups.setdefault(
                group_key,
                {
                    "params": [],
                    "lr": float(args.lr) * scorer_multiplier,
                    "weight_decay": weight_decay,
                    "group_name": group_key,
                    "parameter_names": [],
                    "mraf_scorer_lr_multiplier": scorer_multiplier,
                },
            )
            group["params"].append(value)
            group["parameter_names"].append(key)
        else:
            params.append(
                {"params": [value], "lr": lr, "weight_decay": weight_decay}
            )
    params.extend(scorer_groups.values())
    _validate_optimizer_parameter_partition(model, params)

    # Select optimizer implementation by args.
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer: {args.optimizer}")

    rows = optimizer_param_group_rows(optimizer, model, epoch=0)
    logger = logging.getLogger("IRRA.train")
    for row in rows:
        logger.info(
            "[OPTIMIZER_GROUP] name=%s parameter_count=%d numel=%d "
            "initial_lr=%.12g current_lr=%.12g weight_decay=%.12g "
            "scorer_multiplier=%.6g parameters=%s",
            row["group_name"],
            row["parameter_count"],
            row["parameter_numel"],
            row["initial_lr"],
            row["current_lr"],
            row["weight_decay"],
            row["scorer_lr_multiplier"],
            row["parameter_names"],
        )
    append_optimizer_param_groups_tsv(
        optimizer,
        model,
        getattr(args, "output_dir", ""),
        epoch=0,
    )

    return optimizer


def build_lr_scheduler(args, optimizer):
    # Wrap optimizer with warmup + configurable decay strategy.
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
