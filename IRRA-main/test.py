import argparse
import os.path as op

from datasets import build_dataloader
from model import build_model
from processor.processor import apply_img_local_filter_stage, do_inference
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
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument(
        "--config_file", default="logs/CUHK-PEDES/iira/configs.yaml"
    )
    parser.add_argument(
        "--use_qalr",
        default=None,
        action="store_true",
        help="enable inference-only query-adaptive local reranking",
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
        "--qalr_focus_topn",
        type=int,
        default=None,
        help="only apply local rerank fusion to the first n globally ranked candidates inside rerank_topk",
    )
    cli_args = parser.parse_args()

    args = load_train_configs(cli_args.config_file)
    args = _apply_qalr_test_overrides(args, cli_args)
    args.training = False

    logger = setup_logger("IRRA", save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)

    checkpointer = Checkpointer(model)
    checkpoint = checkpointer.load(f=op.join(args.output_dir, "best.pth"))
    best_epoch = int(checkpoint.get("epoch", 1)) if isinstance(checkpoint, dict) else 1
    apply_img_local_filter_stage(
        model, args, best_epoch, logger=logger, prefix="Inference"
    )

    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)
