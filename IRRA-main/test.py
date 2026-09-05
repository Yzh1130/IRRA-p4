import argparse
import math
import copy
import os.path as op

import torch

from datasets import build_dataloader
from datasets.rgbtpedes import RGBTPEDES
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
from utils.iotools import load_train_configs, read_json
from utils.logger import setup_logger


def _positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a positive float, got {value!r}"
        )
    return parsed


def _apply_mraf_test_overrides(args, cli_args):
    """Apply read-only runtime overrides to a copied loaded configuration."""
    runtime_args = copy.deepcopy(args)
    override = getattr(cli_args, "mraf_gate_logit_scale", None)
    if override is not None:
        runtime_args.mraf_gate_logit_scale = float(override)
    return runtime_args


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
        "use_qalr_v2": False,
        "qalr_v2_save_diag": False,
        "qalr_v2_diag_file": "",
        "qalr_v2_alpha_min": 0.0,
        "qalr_v2_alpha_max": 0.01,
        "qalr_v2_margin_low": 0.0,
        "qalr_v2_margin_high": 0.02,
        "qalr_v2_freeze_top1_margin": 0.02,
        "qalr_v2_promote_margin": 0.0,
        "qalr_v2_modality_fuse": "concat",
        "qalr_v2_top1_guard": False,
        "qalr_v2_promote_only": False,
        "qalr_v2_diag_only": False,
        "qalr_v3_candidate_diag": False,
        "qalr_v3_candidate_diag_file": "",
        "qalr_v3_candidate_topk": 64,
        "qalr_v3_candidate_token_topm": 8,
        "qalr_v3_candidate_patch_topr": 4,
        "qalr_v3_candidate_save_scores": False,
        "use_qalr_v3_gate": False,
        "qalr_v3_gate_mode": "concat",
        "qalr_v3_gate_topk": 64,
        "qalr_v3_gate_token_topm": 8,
        "qalr_v3_gate_patch_topr": 4,
        "qalr_v3_gate_candidate_rank_max": 2,
        "qalr_v3_gate_local_rank_max": 1,
        "qalr_v3_gate_global_margin_max": 0.01,
        "qalr_v3_gate_local_gap_min": 0.01,
        "qalr_v3_gate_save_diag": False,
        "qalr_v3_gate_diag_file": "",
        "use_qalr_v3_two_stage_gate": False,
        "qalr_v3_stage1_mode": "avg",
        "qalr_v3_stage1_global_margin_max": 0.008,
        "qalr_v3_stage1_local_gap_min": 0.015,
        "qalr_v3_stage2_mode": "avg",
        "qalr_v3_stage2_global_margin_min": 0.008,
        "qalr_v3_stage2_global_margin_max": 0.010,
        "qalr_v3_stage2_local_gap_min": 0.011,
        "qalr_v3_two_stage_topk": 64,
        "qalr_v3_two_stage_candidate_rank_max": 2,
        "qalr_v3_two_stage_local_rank_max": 1,
        "qalr_v3_two_stage_token_topm": 8,
        "qalr_v3_two_stage_patch_topr": 4,
        "qalr_v3_two_stage_save_diag": False,
        "qalr_v3_two_stage_diag_file": "",
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
    if cli_args.use_qalr_v2 is not None:
        args.use_qalr_v2 = cli_args.use_qalr_v2
    if cli_args.qalr_v2_save_diag is not None:
        args.qalr_v2_save_diag = cli_args.qalr_v2_save_diag
    if cli_args.qalr_v2_diag_file is not None:
        args.qalr_v2_diag_file = cli_args.qalr_v2_diag_file
    if cli_args.qalr_v2_alpha_min is not None:
        args.qalr_v2_alpha_min = cli_args.qalr_v2_alpha_min
    if cli_args.qalr_v2_alpha_max is not None:
        args.qalr_v2_alpha_max = cli_args.qalr_v2_alpha_max
    if cli_args.qalr_v2_margin_low is not None:
        args.qalr_v2_margin_low = cli_args.qalr_v2_margin_low
    if cli_args.qalr_v2_margin_high is not None:
        args.qalr_v2_margin_high = cli_args.qalr_v2_margin_high
    if cli_args.qalr_v2_freeze_top1_margin is not None:
        args.qalr_v2_freeze_top1_margin = cli_args.qalr_v2_freeze_top1_margin
    if cli_args.qalr_v2_promote_margin is not None:
        args.qalr_v2_promote_margin = cli_args.qalr_v2_promote_margin
    if cli_args.qalr_v2_modality_fuse is not None:
        args.qalr_v2_modality_fuse = cli_args.qalr_v2_modality_fuse
    if cli_args.qalr_v2_top1_guard is not None:
        args.qalr_v2_top1_guard = cli_args.qalr_v2_top1_guard
    if cli_args.qalr_v2_promote_only is not None:
        args.qalr_v2_promote_only = cli_args.qalr_v2_promote_only
    if cli_args.qalr_v2_diag_only is not None:
        args.qalr_v2_diag_only = cli_args.qalr_v2_diag_only
    if cli_args.qalr_v3_candidate_diag is not None:
        args.qalr_v3_candidate_diag = cli_args.qalr_v3_candidate_diag
    if cli_args.qalr_v3_candidate_diag_file is not None:
        args.qalr_v3_candidate_diag_file = cli_args.qalr_v3_candidate_diag_file
    if cli_args.qalr_v3_candidate_topk is not None:
        args.qalr_v3_candidate_topk = cli_args.qalr_v3_candidate_topk
    if cli_args.qalr_v3_candidate_token_topm is not None:
        args.qalr_v3_candidate_token_topm = cli_args.qalr_v3_candidate_token_topm
    if cli_args.qalr_v3_candidate_patch_topr is not None:
        args.qalr_v3_candidate_patch_topr = cli_args.qalr_v3_candidate_patch_topr
    if cli_args.qalr_v3_candidate_save_scores is not None:
        args.qalr_v3_candidate_save_scores = cli_args.qalr_v3_candidate_save_scores
    if cli_args.use_qalr_v3_gate is not None:
        args.use_qalr_v3_gate = cli_args.use_qalr_v3_gate
    if cli_args.qalr_v3_gate_mode is not None:
        args.qalr_v3_gate_mode = cli_args.qalr_v3_gate_mode
    if cli_args.qalr_v3_gate_topk is not None:
        args.qalr_v3_gate_topk = cli_args.qalr_v3_gate_topk
    if cli_args.qalr_v3_gate_token_topm is not None:
        args.qalr_v3_gate_token_topm = cli_args.qalr_v3_gate_token_topm
    if cli_args.qalr_v3_gate_patch_topr is not None:
        args.qalr_v3_gate_patch_topr = cli_args.qalr_v3_gate_patch_topr
    if cli_args.qalr_v3_gate_candidate_rank_max is not None:
        args.qalr_v3_gate_candidate_rank_max = cli_args.qalr_v3_gate_candidate_rank_max
    if cli_args.qalr_v3_gate_local_rank_max is not None:
        args.qalr_v3_gate_local_rank_max = cli_args.qalr_v3_gate_local_rank_max
    if cli_args.qalr_v3_gate_global_margin_max is not None:
        args.qalr_v3_gate_global_margin_max = cli_args.qalr_v3_gate_global_margin_max
    if cli_args.qalr_v3_gate_local_gap_min is not None:
        args.qalr_v3_gate_local_gap_min = cli_args.qalr_v3_gate_local_gap_min
    if cli_args.qalr_v3_gate_save_diag is not None:
        args.qalr_v3_gate_save_diag = cli_args.qalr_v3_gate_save_diag
    if cli_args.qalr_v3_gate_diag_file is not None:
        args.qalr_v3_gate_diag_file = cli_args.qalr_v3_gate_diag_file
    if cli_args.use_qalr_v3_two_stage_gate is not None:
        args.use_qalr_v3_two_stage_gate = cli_args.use_qalr_v3_two_stage_gate
    if cli_args.qalr_v3_stage1_mode is not None:
        args.qalr_v3_stage1_mode = cli_args.qalr_v3_stage1_mode
    if cli_args.qalr_v3_stage1_global_margin_max is not None:
        args.qalr_v3_stage1_global_margin_max = cli_args.qalr_v3_stage1_global_margin_max
    if cli_args.qalr_v3_stage1_local_gap_min is not None:
        args.qalr_v3_stage1_local_gap_min = cli_args.qalr_v3_stage1_local_gap_min
    if cli_args.qalr_v3_stage2_mode is not None:
        args.qalr_v3_stage2_mode = cli_args.qalr_v3_stage2_mode
    if cli_args.qalr_v3_stage2_global_margin_min is not None:
        args.qalr_v3_stage2_global_margin_min = cli_args.qalr_v3_stage2_global_margin_min
    if cli_args.qalr_v3_stage2_global_margin_max is not None:
        args.qalr_v3_stage2_global_margin_max = cli_args.qalr_v3_stage2_global_margin_max
    if cli_args.qalr_v3_stage2_local_gap_min is not None:
        args.qalr_v3_stage2_local_gap_min = cli_args.qalr_v3_stage2_local_gap_min
    if cli_args.qalr_v3_two_stage_topk is not None:
        args.qalr_v3_two_stage_topk = cli_args.qalr_v3_two_stage_topk
    if cli_args.qalr_v3_two_stage_candidate_rank_max is not None:
        args.qalr_v3_two_stage_candidate_rank_max = cli_args.qalr_v3_two_stage_candidate_rank_max
    if cli_args.qalr_v3_two_stage_local_rank_max is not None:
        args.qalr_v3_two_stage_local_rank_max = cli_args.qalr_v3_two_stage_local_rank_max
    if cli_args.qalr_v3_two_stage_token_topm is not None:
        args.qalr_v3_two_stage_token_topm = cli_args.qalr_v3_two_stage_token_topm
    if cli_args.qalr_v3_two_stage_patch_topr is not None:
        args.qalr_v3_two_stage_patch_topr = cli_args.qalr_v3_two_stage_patch_topr
    if cli_args.qalr_v3_two_stage_save_diag is not None:
        args.qalr_v3_two_stage_save_diag = cli_args.qalr_v3_two_stage_save_diag
    if cli_args.qalr_v3_two_stage_diag_file is not None:
        args.qalr_v3_two_stage_diag_file = cli_args.qalr_v3_two_stage_diag_file
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
        "--mraf_gate_logit_scale",
        type=_positive_float,
        default=None,
        help=(
            "read-only MRAF logit-scale override for this evaluation; "
            "does not modify configs.yaml or the checkpoint"
        ),
    )
    parser.add_argument(
        "--use_train_validation_pipeline",
        default=False,
        action="store_true",
        help="build eval dataloaders through the same training validation branch used by train.py",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="evaluation split; val requires an explicit --dev_split_file",
    )
    parser.add_argument(
        "--dev_split_file",
        type=str,
        default=None,
        help="explicit development split JSON required by --eval_split val",
    )
    parser.add_argument(
        "--save_eval_scores",
        default=False,
        action="store_true",
        help="save the final text-to-image eval score matrix for offline ensemble",
    )
    parser.add_argument(
        "--eval_scores_file",
        type=str,
        default="",
        help="output .pt file for --save_eval_scores",
    )
    parser.add_argument(
        "--eval_scores_save_raw",
        default=False,
        action="store_true",
        help="also save the pre-rerank raw similarity matrix when dumping eval scores",
    )
    parser.add_argument(
        "--eval_degrade_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "rgb_missing",
            "ir_missing",
            "rgb_noise",
            "ir_noise",
            "both_noise",
            "rgb_blur",
            "ir_blur",
            "both_blur",
            "rgb_brightness",
            "ir_brightness",
            "rgb_lowres",
            "ir_lowres",
            "rgb_jpeg",
            "ir_jpeg",
        ],
        help="evaluation-only temporary RGB/T degradation mode",
    )
    parser.add_argument(
        "--eval_degrade_strength",
        type=float,
        default=1.0,
        help=(
            "degradation strength: Gaussian std for *_noise, brightness factor "
            "for *_brightness, linear resize scale for *_lowres, or integer JPEG "
            "quality for *_jpeg"
        ),
    )
    parser.add_argument(
        "--eval_degrade_blur_kernel",
        type=int,
        default=7,
        help="evaluation-only blur kernel size for *_blur degrade modes",
    )
    parser.add_argument(
        "--eval_degrade_seed",
        type=int,
        default=2026,
        help="evaluation-only random seed for degradation noise",
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
        "--use_qalr_v2",
        default=None,
        action="store_true",
        help="enable safer QALR v2 local reranking",
    )
    parser.add_argument(
        "--no_use_qalr_v2",
        dest="use_qalr_v2",
        default=None,
        action="store_false",
        help="disable safer QALR v2 local reranking",
    )
    parser.add_argument(
        "--qalr_v2_save_diag",
        default=None,
        action="store_true",
        help="save QALR/QALR-v2 per-query diagnostics",
    )
    parser.add_argument(
        "--no_qalr_v2_save_diag",
        dest="qalr_v2_save_diag",
        default=None,
        action="store_false",
        help="disable QALR/QALR-v2 per-query diagnostics",
    )
    parser.add_argument(
        "--qalr_v2_diag_file",
        type=str,
        default=None,
        help="QALR-v2 diagnostics json output file",
    )
    parser.add_argument(
        "--qalr_v2_alpha_min",
        type=float,
        default=None,
        help="minimum adaptive alpha for QALR v2",
    )
    parser.add_argument(
        "--qalr_v2_alpha_max",
        type=float,
        default=None,
        help="maximum adaptive alpha for QALR v2",
    )
    parser.add_argument(
        "--qalr_v2_margin_low",
        type=float,
        default=None,
        help="margin where QALR v2 uses alpha_max",
    )
    parser.add_argument(
        "--qalr_v2_margin_high",
        type=float,
        default=None,
        help="margin where QALR v2 uses alpha_min",
    )
    parser.add_argument(
        "--qalr_v2_freeze_top1_margin",
        type=float,
        default=None,
        help="freeze top1 above this global margin when QALR v2 guard is enabled",
    )
    parser.add_argument(
        "--qalr_v2_promote_margin",
        type=float,
        default=None,
        help="minimum local advantage required for QALR v2 top1 promotion",
    )
    parser.add_argument(
        "--qalr_v2_modality_fuse",
        type=str,
        default=None,
        choices=["concat", "avg", "rgb", "thermal", "precision"],
        help="QALR v2 local score modality fusion",
    )
    parser.add_argument(
        "--qalr_v2_top1_guard",
        default=None,
        action="store_true",
        help="enable QALR v2 global top1 freeze guard",
    )
    parser.add_argument(
        "--no_qalr_v2_top1_guard",
        dest="qalr_v2_top1_guard",
        default=None,
        action="store_false",
        help="disable QALR v2 global top1 freeze guard",
    )
    parser.add_argument(
        "--qalr_v2_promote_only",
        default=None,
        action="store_true",
        help="allow QALR v2 top1 replacement only with enough local advantage",
    )
    parser.add_argument(
        "--no_qalr_v2_promote_only",
        dest="qalr_v2_promote_only",
        default=None,
        action="store_false",
        help="disable QALR v2 promote-only mode",
    )
    parser.add_argument(
        "--qalr_v2_diag_only",
        default=None,
        action="store_true",
        help="compute QALR v2 diagnostics without changing final eval scores",
    )
    parser.add_argument(
        "--no_qalr_v2_diag_only",
        dest="qalr_v2_diag_only",
        default=None,
        action="store_false",
        help="disable QALR v2 diagnostic-only mode",
    )
    parser.add_argument(
        "--qalr_v3_candidate_diag",
        default=None,
        action="store_true",
        help="save QALR v3 candidate-level oracle diagnostics",
    )
    parser.add_argument(
        "--no_qalr_v3_candidate_diag",
        dest="qalr_v3_candidate_diag",
        default=None,
        action="store_false",
        help="disable QALR v3 candidate-level oracle diagnostics",
    )
    parser.add_argument(
        "--qalr_v3_candidate_diag_file",
        type=str,
        default=None,
        help="QALR v3 candidate diagnostics json output file",
    )
    parser.add_argument(
        "--qalr_v3_candidate_topk",
        type=int,
        default=None,
        help="global top-K candidates saved for QALR v3 diagnostics",
    )
    parser.add_argument(
        "--qalr_v3_candidate_token_topm",
        type=int,
        default=None,
        help="number of salient text tokens used for QALR v3 diagnostics",
    )
    parser.add_argument(
        "--qalr_v3_candidate_patch_topr",
        type=int,
        default=None,
        help="number of top patches per token used for QALR v3 diagnostics",
    )
    parser.add_argument(
        "--qalr_v3_candidate_save_scores",
        default=None,
        action="store_true",
        help="save candidate-level score fields in QALR v3 diagnostics",
    )
    parser.add_argument(
        "--no_qalr_v3_candidate_save_scores",
        dest="qalr_v3_candidate_save_scores",
        default=None,
        action="store_false",
        help="omit optional candidate-level score fields in QALR v3 diagnostics",
    )
    parser.add_argument(
        "--use_qalr_v3_gate",
        default=None,
        action="store_true",
        help="enable QALR v3 gated top1 promotion rerank",
    )
    parser.add_argument(
        "--no_use_qalr_v3_gate",
        dest="use_qalr_v3_gate",
        default=None,
        action="store_false",
        help="disable QALR v3 gated top1 promotion rerank",
    )
    parser.add_argument(
        "--qalr_v3_gate_mode",
        type=str,
        default=None,
        choices=["concat", "avg", "rgb", "thermal"],
        help="QALR v3 gate local score mode",
    )
    parser.add_argument(
        "--qalr_v3_gate_topk",
        type=int,
        default=None,
        help="global top-K candidate pool for QALR v3 gate",
    )
    parser.add_argument(
        "--qalr_v3_gate_token_topm",
        type=int,
        default=None,
        help="number of salient text tokens used for QALR v3 gate",
    )
    parser.add_argument(
        "--qalr_v3_gate_patch_topr",
        type=int,
        default=None,
        help="number of top patches per token used for QALR v3 gate",
    )
    parser.add_argument(
        "--qalr_v3_gate_candidate_rank_max",
        type=int,
        default=None,
        help="maximum global candidate rank allowed for QALR v3 gate promotion",
    )
    parser.add_argument(
        "--qalr_v3_gate_local_rank_max",
        type=int,
        default=None,
        help="maximum local rank allowed for QALR v3 gate promotion",
    )
    parser.add_argument(
        "--qalr_v3_gate_global_margin_max",
        type=float,
        default=None,
        help="maximum global top1-top2 margin allowed for QALR v3 gate",
    )
    parser.add_argument(
        "--qalr_v3_gate_local_gap_min",
        type=float,
        default=None,
        help="minimum local score gap over old top1 required for QALR v3 gate",
    )
    parser.add_argument(
        "--qalr_v3_gate_save_diag",
        default=None,
        action="store_true",
        help="save QALR v3 gate per-query diagnostics",
    )
    parser.add_argument(
        "--no_qalr_v3_gate_save_diag",
        dest="qalr_v3_gate_save_diag",
        default=None,
        action="store_false",
        help="disable QALR v3 gate diagnostics",
    )
    parser.add_argument(
        "--qalr_v3_gate_diag_file",
        type=str,
        default=None,
        help="QALR v3 gate diagnostics json output file",
    )
    parser.add_argument(
        "--use_qalr_v3_two_stage_gate",
        default=None,
        action="store_true",
        help="enable QALR v3 two-stage gated top1 promotion rerank",
    )
    parser.add_argument(
        "--no_use_qalr_v3_two_stage_gate",
        dest="use_qalr_v3_two_stage_gate",
        default=None,
        action="store_false",
        help="disable QALR v3 two-stage gated top1 promotion rerank",
    )
    parser.add_argument(
        "--qalr_v3_stage1_mode",
        type=str,
        default=None,
        choices=["concat", "avg", "rgb", "thermal"],
        help="QALR v3 two-stage gate stage1 local score mode",
    )
    parser.add_argument(
        "--qalr_v3_stage1_global_margin_max",
        type=float,
        default=None,
        help="stage1 maximum global top1-top2 margin",
    )
    parser.add_argument(
        "--qalr_v3_stage1_local_gap_min",
        type=float,
        default=None,
        help="stage1 minimum local score gap over old top1",
    )
    parser.add_argument(
        "--qalr_v3_stage2_mode",
        type=str,
        default=None,
        choices=["concat", "avg", "rgb", "thermal"],
        help="QALR v3 two-stage gate stage2 local score mode",
    )
    parser.add_argument(
        "--qalr_v3_stage2_global_margin_min",
        type=float,
        default=None,
        help="stage2 minimum global top1-top2 margin",
    )
    parser.add_argument(
        "--qalr_v3_stage2_global_margin_max",
        type=float,
        default=None,
        help="stage2 maximum global top1-top2 margin",
    )
    parser.add_argument(
        "--qalr_v3_stage2_local_gap_min",
        type=float,
        default=None,
        help="stage2 minimum local score gap over old top1",
    )
    parser.add_argument(
        "--qalr_v3_two_stage_topk",
        type=int,
        default=None,
        help="global top-K candidate pool for QALR v3 two-stage gate",
    )
    parser.add_argument(
        "--qalr_v3_two_stage_candidate_rank_max",
        type=int,
        default=None,
        help="maximum global candidate rank allowed for QALR v3 two-stage gate",
    )
    parser.add_argument(
        "--qalr_v3_two_stage_local_rank_max",
        type=int,
        default=None,
        help="maximum local rank allowed for QALR v3 two-stage gate",
    )
    parser.add_argument(
        "--qalr_v3_two_stage_token_topm",
        type=int,
        default=None,
        help="number of salient text tokens used for QALR v3 two-stage gate",
    )
    parser.add_argument(
        "--qalr_v3_two_stage_patch_topr",
        type=int,
        default=None,
        help="number of top patches per token used for QALR v3 two-stage gate",
    )
    parser.add_argument(
        "--qalr_v3_two_stage_save_diag",
        default=None,
        action="store_true",
        help="save QALR v3 two-stage gate per-query diagnostics",
    )
    parser.add_argument(
        "--no_qalr_v3_two_stage_save_diag",
        dest="qalr_v3_two_stage_save_diag",
        default=None,
        action="store_false",
        help="disable QALR v3 two-stage gate diagnostics",
    )
    parser.add_argument(
        "--qalr_v3_two_stage_diag_file",
        type=str,
        default=None,
        help="QALR v3 two-stage gate diagnostics json output file",
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
    if cli_args.save_eval_scores and not str(cli_args.eval_scores_file).strip():
        raise ValueError(
            "--save_eval_scores requires --eval_scores_file, e.g. "
            "--eval_scores_file pack/scores/model_a.pt"
        )
    if float(cli_args.eval_degrade_strength) < 0.0:
        raise ValueError("--eval_degrade_strength must be >= 0.")
    if int(cli_args.eval_degrade_blur_kernel) <= 0:
        raise ValueError("--eval_degrade_blur_kernel must be > 0.")
    new_unit_strength_modes = {
        "rgb_brightness",
        "ir_brightness",
        "rgb_lowres",
        "ir_lowres",
    }
    if cli_args.eval_degrade_mode in new_unit_strength_modes and not (
        0.0 < float(cli_args.eval_degrade_strength) <= 1.0
    ):
        raise ValueError(
            f"--eval_degrade_strength={cli_args.eval_degrade_strength!r} is invalid "
            f"for {cli_args.eval_degrade_mode}; expected 0 < strength <= 1."
        )
    if cli_args.eval_degrade_mode in {"rgb_jpeg", "ir_jpeg"}:
        jpeg_quality = float(cli_args.eval_degrade_strength)
        if not jpeg_quality.is_integer() or not (1.0 <= jpeg_quality <= 100.0):
            raise ValueError(
                f"--eval_degrade_strength={cli_args.eval_degrade_strength!r} is "
                f"invalid for {cli_args.eval_degrade_mode}; expected an integer "
                "JPEG quality in [1, 100]."
            )

    args = load_train_configs(cli_args.config_file)
    args = _apply_qalr_test_overrides(args, cli_args)
    args = _apply_mraf_test_overrides(args, cli_args)
    args.training = False
    args.save_eval_scores = bool(cli_args.save_eval_scores)
    args.eval_scores_file = str(cli_args.eval_scores_file)
    args.eval_scores_save_raw = bool(cli_args.eval_scores_save_raw)
    args.eval_degrade_mode = str(cli_args.eval_degrade_mode)
    args.eval_degrade_strength = float(cli_args.eval_degrade_strength)
    args.eval_degrade_blur_kernel = int(cli_args.eval_degrade_blur_kernel)
    args.eval_degrade_seed = int(cli_args.eval_degrade_seed)
    if cli_args.dev_split_file is not None:
        args.dev_split_file = op.abspath(op.expanduser(cli_args.dev_split_file))
    args.config_file = op.abspath(cli_args.config_file)
    args.test_ckpt_file = ""

    logger = setup_logger("IRRA", save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    logger.info(
        "[MRAF_OVERRIDE] gate_logit_scale=%.6f source=%s",
        float(getattr(args, "mraf_gate_logit_scale", 1.0)),
        (
            "cli_override"
            if cli_args.mraf_gate_logit_scale is not None
            else "saved_config"
        ),
    )
    logger.info(
        "[DEGRADE] mode=%s strength=%.6f blur_kernel=%d seed=%d",
        args.eval_degrade_mode,
        args.eval_degrade_strength,
        args.eval_degrade_blur_kernel,
        args.eval_degrade_seed,
    )
    if args.eval_degrade_mode in {
        "rgb_brightness",
        "ir_brightness",
        "rgb_lowres",
        "ir_lowres",
        "rgb_jpeg",
        "ir_jpeg",
    }:
        logger.info(
            "[DEGRADE] processing_space=pixel [0,1] before re-normalization"
        )
    device = "cuda"

    args.eval_dataset = str(cli_args.eval_split)
    expected_dev_ids = None
    if args.eval_dataset == "val":
        if not str(cli_args.dev_split_file or "").strip():
            raise ValueError(
                "--eval_split val requires --dev_split_file on the command line; "
                "a path stored only in configs.yaml is not accepted."
            )
        if str(getattr(args, "dataset_name", "")) != "RGBT-PEDES":
            raise ValueError(
                "--eval_split val with --dev_split_file is currently restricted "
                "to dataset_name='RGBT-PEDES'."
            )
        dev_split_file = str(args.dev_split_file)
        if not op.isfile(dev_split_file):
            raise FileNotFoundError(
                f"--dev_split_file does not exist: {dev_split_file}"
            )
        dev_split = read_json(dev_split_file)
        if not isinstance(dev_split, dict) or "dev_ids" not in dev_split:
            raise ValueError(
                "--dev_split_file must be a JSON object containing 'dev_ids'."
            )
        dev_ids = dev_split["dev_ids"]
        if not isinstance(dev_ids, list):
            raise ValueError("--dev_split_file field 'dev_ids' must be a list.")
        if "n_dev" not in dev_split:
            raise ValueError("--dev_split_file must contain 'n_dev'.")
        expected_dev_ids = dev_split["n_dev"]
        if isinstance(expected_dev_ids, bool) or not isinstance(
            expected_dev_ids, int
        ):
            raise ValueError(
                "--dev_split_file field 'n_dev' must be an integer."
            )
        if expected_dev_ids != len(dev_ids):
            raise ValueError(
                "--dev_split_file mismatch: "
                f"n_dev={expected_dev_ids}, len(dev_ids)={len(dev_ids)}."
            )

        dev_dataset = RGBTPEDES(
            root=args.root_dir,
            verbose=False,
            dev_split_file=dev_split_file,
        )
        if dev_dataset.val_annos is dev_dataset.test_annos:
            raise RuntimeError(
                "Unsafe val split: RGBTPEDES.val_annos aliases test_annos."
            )
        val_unique_pids = {
            int(anno["id"]) for anno in dev_dataset.val_annos
        }
        if len(val_unique_pids) != expected_dev_ids:
            raise RuntimeError(
                "Unsafe val split identity count: "
                f"expected n_dev={expected_dev_ids}, got {len(val_unique_pids)}."
            )

    if cli_args.use_train_validation_pipeline:
        loader_args = copy.deepcopy(args)
        loader_args.training = True
        loader_args.distributed = False
        loader_args.val_dataset = args.eval_dataset
        _, test_img_loader, test_txt_loader, num_classes = build_dataloader(loader_args)
        logger.info(
            "[TEST_PIPELINE] use_train_validation_pipeline=True val_dataset=%s batch_size=%s",
            getattr(loader_args, "val_dataset", "test"),
            getattr(loader_args, "batch_size", None),
        )
    else:
        test_img_loader, test_txt_loader, num_classes = build_dataloader(args)

    image_pids = list(
        getattr(getattr(test_img_loader, "dataset", None), "image_pids", [])
    )
    caption_pids = list(
        getattr(getattr(test_txt_loader, "dataset", None), "caption_pids", [])
    )
    num_eval_ids = len(set(image_pids) | set(caption_pids))
    if expected_dev_ids is not None and num_eval_ids != expected_dev_ids:
        raise RuntimeError(
            "Selected val dataloaders do not match the development split: "
            f"expected n_dev={expected_dev_ids}, got {num_eval_ids}."
        )
    logger.info(
        "[EVAL_SPLIT] split=%s num_images=%d num_captions=%d num_ids=%d",
        args.eval_dataset,
        len(image_pids),
        len(caption_pids),
        num_eval_ids,
    )
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
    args.test_ckpt_file = ckpt_file
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
