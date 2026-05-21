import argparse
import copy
import os.path as op

import torch

from datasets import build_dataloader
from model import build_model
from processor.processor import (
    apply_img_local_filter_stage,
    do_inference,
    is_full_eval_state_checkpoint,
    load_full_eval_state_checkpoint,
    log_eval_state_summary,
    restore_eval_extra_state,
)
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.logger import setup_logger


def _apply_qalr_test_overrides(args, cli_args):
    qalr_defaults = {
        "use_qalr": False,
        "rerank_topk": 128,
        "rerank_alpha": 0.2,
        "rerank_token_topm": 8,
        "rerank_patch_topr": 4,
        "use_qalr_score_norm": False,
        "qalr_score_norm_type": "zscore",
        "use_qalr_query_gate": False,
        "qalr_query_gate_type": "top12_margin",
        "qalr_query_gate_margin": 0.05,
        "use_qalr_focus_topn": False,
        "qalr_focus_topn": 16,
        "use_uspc_slot_rerank": False,
        "uspc_slot_rerank_topk": 50,
        "uspc_slot_rerank_alpha": 0.0,
        "uspc_slot_rerank_score_norm": "zscore",
        "uspc_slot_rerank_region_mode": "fixed_stripe",
        "uspc_slot_rerank_patch_source": "rgb",
        "uspc_slot_rerank_min_valid_slots": 1,
        "uspc_slot_rerank_skip_negative_bag": True,
        "debug_uspc_slot_rerank": False,
    }
    for key, value in qalr_defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    if cli_args.use_qalr is not None:
        args.use_qalr = cli_args.use_qalr
    if cli_args.rerank_topk is not None:
        args.rerank_topk = cli_args.rerank_topk
    if cli_args.rerank_alpha is not None:
        args.rerank_alpha = cli_args.rerank_alpha
    if cli_args.rerank_token_topm is not None:
        args.rerank_token_topm = cli_args.rerank_token_topm
    if cli_args.rerank_patch_topr is not None:
        args.rerank_patch_topr = cli_args.rerank_patch_topr
    if cli_args.use_qalr_score_norm is not None:
        args.use_qalr_score_norm = cli_args.use_qalr_score_norm
    if cli_args.qalr_score_norm_type is not None:
        args.qalr_score_norm_type = cli_args.qalr_score_norm_type
    if cli_args.use_qalr_query_gate is not None:
        args.use_qalr_query_gate = cli_args.use_qalr_query_gate
    if cli_args.qalr_query_gate_type is not None:
        args.qalr_query_gate_type = cli_args.qalr_query_gate_type
    if cli_args.qalr_query_gate_margin is not None:
        args.qalr_query_gate_margin = cli_args.qalr_query_gate_margin
    if cli_args.use_qalr_focus_topn is not None:
        args.use_qalr_focus_topn = cli_args.use_qalr_focus_topn
    if cli_args.qalr_focus_topn is not None:
        args.qalr_focus_topn = cli_args.qalr_focus_topn
    if cli_args.use_uspc_slot_rerank is not None:
        args.use_uspc_slot_rerank = cli_args.use_uspc_slot_rerank
    if cli_args.uspc_slot_rerank_topk is not None:
        args.uspc_slot_rerank_topk = cli_args.uspc_slot_rerank_topk
    if cli_args.uspc_slot_rerank_alpha is not None:
        args.uspc_slot_rerank_alpha = cli_args.uspc_slot_rerank_alpha
    if cli_args.uspc_slot_rerank_score_norm is not None:
        args.uspc_slot_rerank_score_norm = cli_args.uspc_slot_rerank_score_norm
    if cli_args.uspc_slot_rerank_region_mode is not None:
        args.uspc_slot_rerank_region_mode = cli_args.uspc_slot_rerank_region_mode
    if cli_args.uspc_slot_rerank_patch_source is not None:
        args.uspc_slot_rerank_patch_source = cli_args.uspc_slot_rerank_patch_source
    if cli_args.uspc_slot_rerank_min_valid_slots is not None:
        args.uspc_slot_rerank_min_valid_slots = cli_args.uspc_slot_rerank_min_valid_slots
    if cli_args.uspc_slot_rerank_skip_negative_bag is not None:
        args.uspc_slot_rerank_skip_negative_bag = cli_args.uspc_slot_rerank_skip_negative_bag
    if cli_args.debug_uspc_slot_rerank is not None:
        args.debug_uspc_slot_rerank = cli_args.debug_uspc_slot_rerank
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument(
        "--config_file", default="logs/CUHK-PEDES/iira/configs.yaml"
    )
    parser.add_argument(
        "--test_ckpt_file",
        type=str,
        default="",
        help="checkpoint file to evaluate; defaults to output_dir/best.pth",
    )
    parser.add_argument(
        "--use_train_validation_pipeline",
        default=False,
        action="store_true",
        help="build eval dataloaders through the same training validation branch used by train.py",
    )
    parser.add_argument(
        "--use_qalr",
        default=None,
        action="store_true",
        help="enable inference-only query-adaptive local reranking",
    )
    parser.add_argument(
        "--no_use_qalr",
        dest="use_qalr",
        default=None,
        action="store_false",
        help="disable inference-only query-adaptive local reranking",
    )
    parser.add_argument(
        "--rerank_topk",
        type=int,
        default=None,
        help="number of top-ranked gallery candidates per query used for local reranking",
    )
    parser.add_argument(
        "--rerank_alpha",
        type=float,
        default=None,
        help="fusion weight for local rerank score",
    )
    parser.add_argument(
        "--rerank_token_topm",
        type=int,
        default=None,
        help="number of salient text tokens kept for local reranking",
    )
    parser.add_argument(
        "--rerank_patch_topr",
        type=int,
        default=None,
        help="number of top-matching RGB/T patches averaged per selected text token",
    )
    parser.add_argument(
        "--use_qalr_score_norm",
        default=None,
        action="store_true",
        help="enable query-wise local score normalization inside QALR",
    )
    parser.add_argument(
        "--no_use_qalr_score_norm",
        dest="use_qalr_score_norm",
        default=None,
        action="store_false",
        help="disable query-wise local score normalization inside QALR",
    )
    parser.add_argument(
        "--qalr_score_norm_type",
        type=str,
        default=None,
        help="QALR local score normalization type",
    )
    parser.add_argument(
        "--use_qalr_query_gate",
        default=None,
        action="store_true",
        help="enable hard-query gating that decides whether each query uses QALR",
    )
    parser.add_argument(
        "--no_use_qalr_query_gate",
        dest="use_qalr_query_gate",
        default=None,
        action="store_false",
        help="disable hard-query gating for QALR",
    )
    parser.add_argument(
        "--qalr_query_gate_type",
        type=str,
        default=None,
        help="QALR query gating type",
    )
    parser.add_argument(
        "--qalr_query_gate_margin",
        type=float,
        default=None,
        help="margin threshold on global top1-top2 score gap for enabling QALR on a query",
    )
    parser.add_argument(
        "--use_qalr_focus_topn",
        default=None,
        action="store_true",
        help="enable focus top-n rerank inside the selected global top-K candidates",
    )
    parser.add_argument(
        "--no_use_qalr_focus_topn",
        dest="use_qalr_focus_topn",
        default=None,
        action="store_false",
        help="disable focus top-n rerank inside the selected global top-K candidates",
    )
    parser.add_argument(
        "--qalr_focus_topn",
        type=int,
        default=None,
        help="only apply local rerank fusion to the first n globally ranked candidates inside rerank_topk",
    )
    parser.add_argument(
        "--use_uspc_slot_rerank",
        default=None,
        action="store_true",
        help="enable USPC-v2C text-slot guided inference-time local reranking",
    )
    parser.add_argument(
        "--no_use_uspc_slot_rerank",
        dest="use_uspc_slot_rerank",
        default=None,
        action="store_false",
        help="disable USPC-v2C text-slot guided inference-time local reranking",
    )
    parser.add_argument(
        "--uspc_slot_rerank_topk",
        type=int,
        default=None,
        help="number of top-ranked gallery candidates per query used for USPC slot reranking",
    )
    parser.add_argument(
        "--uspc_slot_rerank_alpha",
        type=float,
        default=None,
        help="fusion weight for USPC slot rerank score",
    )
    parser.add_argument(
        "--uspc_slot_rerank_score_norm",
        type=str,
        default=None,
        choices=["zscore", "none"],
        help="query-wise USPC slot rerank score normalization",
    )
    parser.add_argument(
        "--uspc_slot_rerank_region_mode",
        type=str,
        default=None,
        choices=["fixed_stripe"],
        help="region pooling mode for USPC slot rerank",
    )
    parser.add_argument(
        "--uspc_slot_rerank_patch_source",
        type=str,
        default=None,
        choices=["rgb"],
        help="patch source for USPC slot rerank; v2C first version supports RGB only",
    )
    parser.add_argument(
        "--uspc_slot_rerank_min_valid_slots",
        type=int,
        default=None,
        help="minimum parsed valid slots required before applying USPC slot rerank",
    )
    parser.add_argument(
        "--uspc_slot_rerank_skip_negative_bag",
        default=None,
        action="store_true",
        help="skip no-bag text slots during USPC slot rerank",
    )
    parser.add_argument(
        "--no_uspc_slot_rerank_skip_negative_bag",
        dest="uspc_slot_rerank_skip_negative_bag",
        default=None,
        action="store_false",
        help="include no-bag text slots during USPC slot rerank",
    )
    parser.add_argument(
        "--debug_uspc_slot_rerank",
        default=None,
        action="store_true",
        help="print extra USPC slot rerank debug information",
    )
    parser.add_argument(
        "--no_debug_uspc_slot_rerank",
        dest="debug_uspc_slot_rerank",
        default=None,
        action="store_false",
        help="disable extra USPC slot rerank debug information",
    )
    cli_args = parser.parse_args()

    args = load_train_configs(cli_args.config_file)
    args = _apply_qalr_test_overrides(args, cli_args)
    args.training = False

    logger = setup_logger("IRRA", save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    if cli_args.use_train_validation_pipeline:
        loader_args = copy.deepcopy(args)
        loader_args.training = True
        loader_args.distributed = False
        args.eval_dataset = str(getattr(args, "val_dataset", "test"))
        _, test_img_loader, test_txt_loader, num_classes = build_dataloader(loader_args)
        logger.info(
            "[TEST_PIPELINE] use_train_validation_pipeline=True val_dataset=%s batch_size=%s",
            getattr(loader_args, "val_dataset", "test"),
            getattr(loader_args, "batch_size", None),
        )
    else:
        test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)

    checkpointer = Checkpointer(model)
    ckpt_file = op.join(args.output_dir, "best.pth")
    raw_checkpoint = None
    is_full_state_ckpt = False
    if str(cli_args.test_ckpt_file).strip():
        ckpt_file = op.abspath(cli_args.test_ckpt_file)
        if not op.isfile(ckpt_file):
            raise FileNotFoundError(f"--test_ckpt_file does not exist: {ckpt_file}")
        raw_checkpoint = torch.load(ckpt_file, map_location=torch.device("cpu"))
        is_full_state_ckpt = is_full_eval_state_checkpoint(raw_checkpoint)

    if is_full_state_ckpt:
        checkpoint = load_full_eval_state_checkpoint(
            model,
            ckpt_file,
            logger=logger,
            checkpoint=raw_checkpoint,
        )
    else:
        checkpoint = checkpointer.load(f=ckpt_file)
    if str(cli_args.test_ckpt_file).strip():
        logger.info("[TEST_CKPT_LOAD] path=%s", ckpt_file)
    best_epoch = int(checkpoint.get("epoch", 1)) if isinstance(checkpoint, dict) else 1
    apply_img_local_filter_stage(
        model, args, best_epoch, logger=None, prefix="INFERENCE"
    )
    extra_eval_state = (
        checkpoint.get("extra_eval_state", {})
        if is_full_state_ckpt and isinstance(checkpoint, dict)
        else {}
    )
    if is_full_state_ckpt:
        restore_eval_extra_state(model, extra_eval_state, logger=logger)
    if str(cli_args.test_ckpt_file).strip():
        log_eval_state_summary(
            logger,
            model,
            stage="test_after_load",
            checkpoint=raw_checkpoint if raw_checkpoint is not None else checkpoint,
            extra_state=extra_eval_state,
        )

    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader, eval_source="test_py")
