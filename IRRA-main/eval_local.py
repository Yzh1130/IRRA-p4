import argparse
import os.path as op

import torch

from datasets import build_dataloader
from model import build_model
from processor.processor import apply_img_local_filter_stage, log_img_local_filter_eval_state
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.logger import setup_logger
from utils.metrics import Evaluator


DEFAULT_ARGS = {
    "training": False,
    "distributed": False,
    "name": "eval_local",
    "output_dir": "logs",
    "pretrain_choice": "ViT-B/16",
    "temperature": 0.02,
    "img_aug": False,
    "cmt_depth": 4,
    "MLM": False,
    "loss_names": "sdm+id+mlm",
    "mlm_loss_weight": 1.0,
    "id_loss_weight": 1.0,
    "use_mum": False,
    "use_global_uncertainty": False,
    "uncertainty_dist_type": "diag_gaussian",
    "logvar_min": -6.0,
    "logvar_max": 4.0,
    "use_precision_fusion": None,
    "uncertainty_loss_type": "w2",
    "uncertainty_eps": 1e-6,
    "eval_use_fused_mu": True,
    "eval_use_mu_only": False,
    "use_rgbt_add_fusion": True,
    "use_rgbt_moe_adapter": False,
    "moe_adapter_layers": "6,7,8,9,10,11",
    "moe_adapter_dim": 64,
    "moe_adapter_dropout": 0.1,
    "moe_router_temp": 1.0,
    "lambda_router_balance": 0.01,
    "lambda_shared_consistency": 0.05,
    "use_img_local_filter": False,
    "img_local_topr": 4,
    "img_local_gamma": 0.1,
    "img_local_temp": 0.07,
    "use_image_local_filter": False,
    "img_local_stage_mode": "always",
    "img_local_stage_start": 1,
    "img_local_stage_end": 9999,
    "img_local_topk": 64,
    "img_local_unc_lambda": 0.3,
    "img_local_fuse_gamma": 0.1,
    "img_local_gamma_schedule_mode": "constant",
    "img_local_gamma_stage1_end": 6,
    "img_local_gamma_stage2_end": 12,
    "img_local_gamma_stage1_value": 0.05,
    "img_local_gamma_stage2_value": 0.02,
    "img_local_gamma_stage3_value": 0.0,
    "img_local_score_temp": 0.07,
    "img_local_detach_global": True,
    "img_size": (384, 128),
    "stride_size": 16,
    "text_length": 77,
    "vocab_size": 49408,
    "dataset_name": None,
    "root_dir": None,
    "test_batch_size": 512,
    "num_workers": 8,
    "eval_dataset": "test",
    "val_dataset": "test",
    "use_qalr": False,
    "use_transport_rerank": False,
}


def _build_parser():
    parser = argparse.ArgumentParser(description="IRRA eval-only local entry")
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="optional training configs.yaml used as the base eval config",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="dataset name used for eval dataloader",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="dataset root directory",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        required=True,
        help="checkpoint path used for eval-only model loading",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        choices=["val", "test"],
        default=None,
        help="eval split used by eval-only dataloader",
    )
    parser.add_argument(
        "--runtime_epoch",
        type=int,
        default=None,
        help="runtime epoch used by image-local scheduling; defaults to checkpoint epoch",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="eval batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="dataloader workers",
    )
    parser.add_argument(
        "--pretrain_choice",
        type=str,
        default=None,
        help="CLIP backbone name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="initial temperature value",
    )
    parser.add_argument(
        "--cmt_depth",
        type=int,
        default=None,
        help="cross modal transformer depth",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="image size used by eval transforms",
    )
    parser.add_argument(
        "--stride_size",
        type=int,
        default=None,
        help="vision transformer stride size",
    )
    parser.add_argument(
        "--text_length",
        type=int,
        default=None,
        help="text sequence length",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="token vocabulary size",
    )
    parser.add_argument(
        "--MLM",
        action="store_true",
        default=None,
        help="keep MLM-related modules enabled when the checkpoint needs them",
    )
    parser.add_argument(
        "--loss_names",
        type=str,
        default=None,
        help="loss-name string used to build the checkpoint architecture",
    )

    rgbt_add_group = parser.add_mutually_exclusive_group()
    rgbt_add_group.add_argument(
        "--use_rgbt_add_fusion",
        dest="use_rgbt_add_fusion",
        action="store_true",
        help="enable RGB-T additive fusion",
    )
    rgbt_add_group.add_argument(
        "--no_use_rgbt_add_fusion",
        dest="use_rgbt_add_fusion",
        action="store_false",
        help="disable RGB-T additive fusion",
    )
    parser.set_defaults(use_rgbt_add_fusion=None)

    global_unc_group = parser.add_mutually_exclusive_group()
    global_unc_group.add_argument(
        "--use_global_uncertainty",
        dest="use_global_uncertainty",
        action="store_true",
        help="enable global uncertainty heads",
    )
    global_unc_group.add_argument(
        "--no_use_global_uncertainty",
        dest="use_global_uncertainty",
        action="store_false",
        help="disable global uncertainty heads",
    )
    parser.set_defaults(use_global_uncertainty=None)

    parser.add_argument(
        "--uncertainty_dist_type",
        type=str,
        default=None,
        help="global uncertainty distribution type",
    )
    parser.add_argument(
        "--uncertainty_loss_type",
        type=str,
        default=None,
        help="uncertainty alignment loss type",
    )

    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument(
        "--use_precision_fusion",
        dest="use_precision_fusion",
        action="store_true",
        help="enable precision-weighted RGB-T fusion",
    )
    precision_group.add_argument(
        "--no_use_precision_fusion",
        dest="use_precision_fusion",
        action="store_false",
        help="disable precision-weighted RGB-T fusion",
    )
    parser.set_defaults(use_precision_fusion=None)

    eval_fused_group = parser.add_mutually_exclusive_group()
    eval_fused_group.add_argument(
        "--eval_use_fused_mu",
        dest="eval_use_fused_mu",
        action="store_true",
        help="use fused mu as visual eval embedding",
    )
    eval_fused_group.add_argument(
        "--no_eval_use_fused_mu",
        dest="eval_use_fused_mu",
        action="store_false",
        help="disable fused-mu eval embedding",
    )
    parser.set_defaults(eval_use_fused_mu=None)

    moe_group = parser.add_mutually_exclusive_group()
    moe_group.add_argument(
        "--use_rgbt_moe_adapter",
        dest="use_rgbt_moe_adapter",
        action="store_true",
        help="enable visual RGBT MoE adapter",
    )
    moe_group.add_argument(
        "--no_use_rgbt_moe_adapter",
        dest="use_rgbt_moe_adapter",
        action="store_false",
        help="disable visual RGBT MoE adapter",
    )
    parser.set_defaults(use_rgbt_moe_adapter=None)

    parser.add_argument(
        "--moe_adapter_layers",
        type=str,
        default=None,
        help="comma-separated MoE adapter layers",
    )
    parser.add_argument(
        "--moe_adapter_dim",
        type=int,
        default=None,
        help="MoE adapter bottleneck dim",
    )
    parser.add_argument(
        "--moe_adapter_dropout",
        type=float,
        default=None,
        help="MoE adapter dropout",
    )
    parser.add_argument(
        "--moe_router_temp",
        type=float,
        default=None,
        help="MoE router temperature",
    )
    parser.add_argument(
        "--lambda_router_balance",
        type=float,
        default=None,
        help="router balance weight",
    )
    parser.add_argument(
        "--lambda_shared_consistency",
        type=float,
        default=None,
        help="shared consistency weight",
    )

    local_group = parser.add_mutually_exclusive_group()
    local_group.add_argument(
        "--use_image_local_filter",
        dest="use_image_local_filter",
        action="store_true",
        help="enable image-only local filter at eval time",
    )
    local_group.add_argument(
        "--no_use_image_local_filter",
        dest="use_image_local_filter",
        action="store_false",
        help="disable image-only local filter at eval time",
    )
    parser.set_defaults(use_image_local_filter=None)

    parser.add_argument(
        "--img_local_stage_mode",
        type=str,
        choices=["always", "window"],
        default=None,
        help="image-local stage mode",
    )
    parser.add_argument(
        "--img_local_stage_start",
        type=int,
        default=None,
        help="image-local stage start epoch",
    )
    parser.add_argument(
        "--img_local_stage_end",
        type=int,
        default=None,
        help="image-local stage end epoch",
    )
    parser.add_argument(
        "--img_local_topk",
        type=int,
        default=None,
        help="image-local top-k patch count",
    )
    parser.add_argument(
        "--img_local_topr",
        type=int,
        default=None,
        help="legacy v1 image-local top-r patch count",
    )
    parser.add_argument(
        "--img_local_unc_lambda",
        type=float,
        default=None,
        help="image-local uncertainty penalty weight",
    )
    parser.add_argument(
        "--img_local_fuse_gamma",
        type=float,
        default=None,
        help="image-local fuse gamma",
    )
    parser.add_argument(
        "--img_local_gamma_schedule_mode",
        type=str,
        choices=["constant", "three_stage"],
        default=None,
        help="image-local gamma schedule mode",
    )
    parser.add_argument(
        "--img_local_score_temp",
        type=float,
        default=None,
        help="image-local score temperature",
    )

    detach_group = parser.add_mutually_exclusive_group()
    detach_group.add_argument(
        "--img_local_detach_global",
        dest="img_local_detach_global",
        action="store_true",
        help="detach global feature in image-local scoring",
    )
    detach_group.add_argument(
        "--no_img_local_detach_global",
        dest="img_local_detach_global",
        action="store_false",
        help="keep global feature trainable in image-local scoring",
    )
    parser.set_defaults(img_local_detach_global=None)

    return parser


def _load_base_args(config_file):
    if config_file:
        args = load_train_configs(config_file)
    else:
        args = argparse.Namespace()
    for key, value in DEFAULT_ARGS.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def _apply_optional_override(args, key, value):
    if value is not None:
        setattr(args, key, value)


def _merge_cli_overrides(args, cli_args):
    for key in [
        "dataset_name",
        "root_dir",
        "test_batch_size",
        "num_workers",
        "pretrain_choice",
        "temperature",
        "cmt_depth",
        "loss_names",
        "use_rgbt_add_fusion",
        "use_global_uncertainty",
        "uncertainty_dist_type",
        "use_precision_fusion",
        "uncertainty_loss_type",
        "eval_use_fused_mu",
        "use_rgbt_moe_adapter",
        "moe_adapter_layers",
        "moe_adapter_dim",
        "moe_adapter_dropout",
        "moe_router_temp",
        "lambda_router_balance",
        "lambda_shared_consistency",
        "use_image_local_filter",
        "img_local_stage_mode",
        "img_local_stage_start",
        "img_local_stage_end",
        "img_local_topk",
        "img_local_topr",
        "img_local_unc_lambda",
        "img_local_fuse_gamma",
        "img_local_gamma_schedule_mode",
        "img_local_score_temp",
        "img_local_detach_global",
        "eval_dataset",
        "runtime_epoch",
        "MLM",
        "stride_size",
        "text_length",
        "vocab_size",
    ]:
        _apply_optional_override(args, key, getattr(cli_args, key))

    if cli_args.img_size is not None:
        args.img_size = tuple(cli_args.img_size)

    args.resume_ckpt = cli_args.resume_ckpt
    args.resume_ckpt_file = cli_args.resume_ckpt
    args.training = False
    args.distributed = False
    return args


def _validate_required_args(args):
    if not args.dataset_name:
        raise ValueError("--dataset_name is required unless provided by --config_file.")
    if not args.root_dir:
        raise ValueError("--root_dir is required unless provided by --config_file.")


if __name__ == "__main__":
    cli_args = _build_parser().parse_args()

    args = _load_base_args(cli_args.config_file)
    args = _merge_cli_overrides(args, cli_args)
    _validate_required_args(args)

    ckpt_path = op.abspath(args.resume_ckpt)
    if not op.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log_dir = op.dirname(ckpt_path)
    if getattr(args, "output_dir", None) in {None, "", "logs"}:
        args.output_dir = log_dir

    logger = setup_logger("IRRA", save_dir=log_dir, if_train=False)
    logger.info("Eval-only local entry")
    logger.info("resume_ckpt=%s", ckpt_path)
    logger.info("dataset_name=%s, root_dir=%s", args.dataset_name, args.root_dir)
    logger.info("eval_dataset=%s", getattr(args, "eval_dataset", "test"))
    logger.info(
        "[EVAL_LOCAL][START] ckpt=%s use_image_local_filter=%s topk=%d topr=%d unc_lambda=%.6f gamma=%.6f score_temp=%.6f detach_global=%s",
        ckpt_path,
        bool(getattr(args, "use_image_local_filter", False)),
        int(getattr(args, "img_local_topk", 64)),
        int(getattr(args, "img_local_topr", 4)),
        float(getattr(args, "img_local_unc_lambda", 0.3)),
        float(getattr(args, "img_local_fuse_gamma", 0.1)),
        float(getattr(args, "img_local_score_temp", 0.07)),
        bool(getattr(args, "img_local_detach_global", True)),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)

    checkpointer = Checkpointer(model)
    checkpoint = checkpointer.load(ckpt_path)
    runtime_epoch = (
        int(args.runtime_epoch)
        if getattr(args, "runtime_epoch", None) is not None
        else int(checkpoint.get("epoch", 1)) if isinstance(checkpoint, dict) else 1
    )

    apply_img_local_filter_stage(
        model,
        args,
        runtime_epoch,
        logger=None,
        prefix="VALIDATION",
    )
    log_img_local_filter_eval_state(logger, model, prefix="VALIDATION")

    model.to(device)
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())
    logger.info("Final t2i R1: %.3f", float(top1))
    eval_result = evaluator.last_eval_result or {}
    logger.info(
        "[EVAL_LOCAL][SUMMARY] ckpt=%s gamma=%.6f topk=%d topr=%d R1=%.3f R5=%.3f R10=%.3f mAP=%.3f mINP=%.3f",
        ckpt_path,
        float(getattr(args, "img_local_fuse_gamma", 0.1)),
        int(getattr(args, "img_local_topk", 64)),
        int(getattr(args, "img_local_topr", 4)),
        float(eval_result.get("R1", float(top1))),
        float(eval_result.get("R5", 0.0)),
        float(eval_result.get("R10", 0.0)),
        float(eval_result.get("mAP", 0.0)),
        float(eval_result.get("mINP", 0.0)),
    )
