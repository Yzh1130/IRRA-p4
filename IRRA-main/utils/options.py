import argparse
import math

from utils.config import load_config_mapping
from utils.mdrc import ALLOWED_DEGRADE_MODES, DEFAULT_DEGRADE_MODES


MRAF_PRECISION_FUSION_ERROR = (
    "MRAF and uncertainty precision fusion are mutually exclusive in the "
    "current implementation."
)


def validate_mraf_options(args):
    """Validate MRAF options for CLI and config-file model construction."""
    if int(getattr(args, "mraf_hidden_dim", 128)) <= 0:
        raise ValueError("--mraf_hidden_dim must be > 0.")
    dropout = float(getattr(args, "mraf_dropout", 0.1))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("--mraf_dropout must be in [0, 1).")
    if float(getattr(args, "mraf_gate_temperature", 1.0)) <= 0.0:
        raise ValueError("--mraf_gate_temperature must be > 0.")
    gate_logit_scale = float(getattr(args, "mraf_gate_logit_scale", 1.0))
    if not math.isfinite(gate_logit_scale) or gate_logit_scale <= 0.0:
        raise ValueError("--mraf_gate_logit_scale must be > 0.")
    scorer_lr_multiplier = float(
        getattr(args, "mraf_scorer_lr_multiplier", 1.0)
    )
    if (
        not math.isfinite(scorer_lr_multiplier)
        or scorer_lr_multiplier <= 0.0
    ):
        raise ValueError("--mraf_scorer_lr_multiplier must be > 0.")
    if float(getattr(args, "mraf_target_temperature", 0.1)) <= 0.0:
        raise ValueError("--mraf_target_temperature must be > 0.")
    if float(getattr(args, "mraf_rel_lambda", 0.01)) < 0.0:
        raise ValueError("--mraf_rel_lambda must be >= 0.")
    if float(getattr(args, "mraf_balance_lambda", 0.0)) < 0.0:
        raise ValueError("--mraf_balance_lambda must be >= 0.")
    base_grad_scale = float(
        getattr(args, "mraf_gate_base_grad_scale", 1.0)
    )
    if not 0.0 <= base_grad_scale <= 1.0:
        raise ValueError("--mraf_gate_base_grad_scale must be in [0.0, 1.0].")
    if int(getattr(args, "mraf_rel_warmup_epoch", 3)) < 0:
        raise ValueError("--mraf_rel_warmup_epoch must be >= 0.")
    utility_prior = getattr(args, "mraf_utility_prior", None)
    if utility_prior is not None:
        if not isinstance(utility_prior, (list, tuple)) or len(utility_prior) != 2:
            raise ValueError("--mraf_utility_prior requires exactly two values.")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in utility_prior
        ):
            raise ValueError("--mraf_utility_prior values must be finite and > 0.")

    configured_precision = getattr(args, "use_precision_fusion", None)
    effective_precision = (
        bool(getattr(args, "use_global_uncertainty", False))
        if configured_precision is None
        else bool(configured_precision)
    )
    if bool(getattr(args, "mraf_enabled", False)) and effective_precision:
        raise ValueError(MRAF_PRECISION_FUSION_ERROR)


def get_args():
    # Centralized CLI argument definitions for training/inference.
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument(
        "--config_file",
        default="",
        help=(
            "optional saved training config used as parser defaults; explicit "
            "CLI arguments take precedence"
        ),
    )
    # local_rank is injected by distributed launchers (torchrun).
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", type=int, default=100)
    parser.add_argument("--eval_period", type=int, default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')
    parser.add_argument("--load_ckpt_file", type=str, default="", help="load model weights only from ...")
    parser.add_argument("--save_all_epoch_ckpt", default=False, action='store_true', help="save an extra metric-named checkpoint after every validation")
    parser.add_argument(
        "--save_metric_best_ckpt",
        default=False,
        action="store_true",
        help=(
            "enable R1-selected best.pth, checkpoint_selection.tsv, and an "
            "extra non-overwritten metric-named checkpoint"
        ),
    )
    parser.add_argument("--metric_ckpt_dir", type=str, default="", help="directory for extra metric checkpoints; defaults to output_dir/metric_ckpts")
    parser.add_argument("--save_full_eval_state_ckpt", default=False, action='store_true', help="save full eval-state checkpoints after validation for exact reproduction diagnostics")
    parser.add_argument("--verify_ckpt_immediately_after_save", default=False, action='store_true', help="immediately re-evaluate in-memory and reloaded full-state checkpoints after validation")
    parser.add_argument("--verify_ckpt_epochs", type=str, default="", help="comma-separated 1-based epochs for immediate checkpoint verification, e.g. '9,13'; empty means all validation epochs")
    parser.add_argument("--save_last_checkpoint", default=False, action='store_true', help="save one additional last.pth after the final training epoch")
    parser.add_argument("--save_model_only", default=False, action='store_true', help="save only the model state_dict in checkpoints")

    ######################## model general settings ########################
    # pretrain_choice chooses which CLIP checkpoint to initialize.
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=False, action='store_true')

    ## cross modal transfomer setting
    # cmt_depth controls cross-modal transformer depth for MLM branch.
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=False, action='store_true', help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    # loss_names combines multiple tasks via '+'.
    parser.add_argument("--loss_names", default='sdm+id+mlm', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    mdrc_group = parser.add_argument_group("MDRC")
    mdrc_group.add_argument("--mdrc_enable", default=False, action='store_true', help="enable MDRC clean-degraded consistency training")
    mdrc_group.add_argument("--mdrc_lambda", type=float, default=0.05, help="maximum MDRC consistency loss weight")
    mdrc_group.add_argument("--mdrc_loss_scale", type=float, default=1.0, help="extra scale applied to MDRC loss before adding it to total loss")
    mdrc_group.add_argument("--mdrc_prob", type=float, default=0.5, help="per-sample probability of applying one MDRC degradation")
    mdrc_group.add_argument("--mdrc_warmup_epoch", type=int, default=3, help="1-based epoch at which MDRC starts; weight then linearly ramps")
    mdrc_group.add_argument("--mdrc_temp", type=float, default=0.07, help="temperature for MDRC image-text similarity distribution consistency")
    mdrc_group.add_argument("--mdrc_degrade_modes", type=str, nargs='+', default=list(DEFAULT_DEGRADE_MODES), help="MDRC degradation modes; supports space or comma separated values")
    mraf_group = parser.add_argument_group("MRAF")
    mraf_group.add_argument("--mraf_enabled", default=False, action='store_true', help="enable modality reliability-adaptive RGB-T fusion")
    mraf_group.add_argument("--mraf_hidden_dim", type=int, default=128, help="hidden dimension of the shared MRAF reliability scorer")
    mraf_group.add_argument("--mraf_dropout", type=float, default=0.1, help="dropout probability in the MRAF reliability scorer")
    mraf_group.add_argument("--mraf_gate_temperature", type=float, default=1.0, help="softmax temperature for MRAF gate weights")
    mraf_group.add_argument("--mraf_gate_logit_scale", type=float, default=1.0, help="positive multiplier applied to raw MRAF logits before temperature and softmax")
    mraf_group.add_argument("--mraf_gate_fp32", default=False, action='store_true', help="keep only the MRAF scorer, logits, alpha, and fusion arithmetic in FP32")
    mraf_group.add_argument("--mraf_gate_base_grad_scale", type=float, default=1.0, help="scale only base-task gate gradients into the MRAF scorer; valid range [0,1]")
    mraf_group.add_argument("--mraf_equalize_modality_norm", default=False, action='store_true', help="equalize RGB/IR feature norms to their per-sample mean before MRAF weighted fusion; the gate and scorer path is unchanged")
    mraf_group.add_argument("--mraf_freeze_gate", default=False, action='store_true', help="freeze MRAF gate weights at 0.5/0.5 for ablation; the scorer is still constructed but produces no effect on fusion")
    mraf_group.add_argument("--mraf_scorer_lr_multiplier", type=float, default=1.0, help="positive learning-rate multiplier applied only to MRAF scorer parameters")
    mraf_group.add_argument("--mraf_rel_lambda", type=float, default=0.01, help="maximum MRAF reliability supervision weight")
    mraf_group.add_argument("--mraf_rel_isolate_input", default=False, action='store_true', help="isolate reliability-loss scorer input gradients while preserving the main MRAF fusion gate")
    mraf_group.add_argument("--mraf_target_temperature", type=float, default=0.1, help="temperature for drift-derived MRAF reliability targets")
    mraf_group.add_argument("--mraf_utility_prior", type=float, nargs=2, default=None, metavar=("U_RGB", "U_IR"), help="optional positive RGB/IR utility prior for MRAF reliability targets")
    mraf_group.add_argument("--mraf_rel_warmup_epoch", type=int, default=3, help="number of epochs used to linearly warm up MRAF reliability supervision")
    mraf_group.add_argument("--mraf_balance_lambda", type=float, default=0.0, help="reserved MRAF balance-loss weight; inactive in the first implementation")
    # Ablation-only degradation data augmentation (single-branch; no consistency loss).
    # Active only when --mdrc_enable is off; "none" keeps the baseline path unchanged.
    parser.add_argument("--degrade_aug_mode", type=str, default="none", choices=["none", "mdrc_style", "moddrop"], help="ablation degradation augmentation applied to image inputs before the forward pass; 'mdrc_style' samples one of the 6 MDRC modes, 'moddrop' zeros a whole modality; disabled when --mdrc_enable is set")
    parser.add_argument("--use_proto_calib", default=False, action='store_true', help="enable identity prototype calibration bank during training only")
    parser.add_argument("--proto_version", type=str, default="v1", choices=["v1", "v2"], help="prototype calibration version used when --use_proto_calib is enabled")
    parser.add_argument("--lambda_proto_v", type=float, default=1.0, help="reserved visual prototype calibration loss weight")
    parser.add_argument("--lambda_proto_t", type=float, default=1.0, help="reserved text prototype calibration loss weight")
    parser.add_argument("--lambda_proto_self_v", type=float, default=None, help="v2 fused_det_feat -> visual prototype CE weight; defaults to lambda_proto_v when omitted")
    parser.add_argument("--lambda_proto_self_t", type=float, default=None, help="v2 text_det_feat -> text prototype CE weight; defaults to lambda_proto_t when omitted")
    parser.add_argument("--lambda_proto_cross_t2v", type=float, default=0.0, help="v2 text_det_feat -> visual prototype CE weight")
    parser.add_argument("--lambda_proto_cross_v2t", type=float, default=0.0, help="v2 fused_det_feat -> text prototype CE weight")
    parser.add_argument("--lambda_proto_align", type=float, default=0.0, help="v2 same-pid visual/text prototype alignment loss weight")
    parser.add_argument("--proto_temp", type=float, default=0.07, help="reserved temperature for prototype calibration logits")
    parser.add_argument("--proto_momentum", type=float, default=0.9, help="EMA momentum for identity prototype bank updates")
    parser.add_argument("--proto_warmup_epochs", type=int, default=0, help="number of warmup epochs before prototype losses are enabled")
    parser.add_argument("--proto_update_source", type=str, default="detached", choices=["detached", "raw"], help="feature source semantics for prototype bank updates")
    parser.add_argument("--proto_init_file", type=str, default="", help="offline prototype init .pt file path")
    parser.add_argument("--proto_init_strict", default=False, action='store_true', help="require prototype init file metadata to match num_classes and feature_dim")
    parser.set_defaults(proto_init_load_visual=True, proto_init_load_text=True)
    parser.add_argument("--proto_init_load_visual", dest="proto_init_load_visual", action='store_true', help="load visual prototypes from --proto_init_file")
    parser.add_argument("--no_proto_init_load_visual", dest="proto_init_load_visual", action='store_false', help="do not load visual prototypes from --proto_init_file")
    parser.add_argument("--proto_init_load_text", dest="proto_init_load_text", action='store_true', help="load text prototypes from --proto_init_file")
    parser.add_argument("--no_proto_init_load_text", dest="proto_init_load_text", action='store_false', help="do not load text prototypes from --proto_init_file")
    parser.add_argument("--proto_init_freeze_update_epochs", type=int, default=0, help="freeze prototype bank EMA updates for the first N epochs after loading --proto_init_file")
    parser.add_argument("--debug_proto_stats", default=False, action='store_true', help="print extra prototype calibration stats")
    parser.add_argument("--use_uspc", default=False, action='store_true', help="enable USPC auxiliary loss")
    parser.add_argument("--uspc_version", type=str, default="v1", choices=["v1", "v2a"], help="USPC version")
    parser.add_argument("--lambda_uspc", type=float, default=0.0, help="overall USPC loss weight")
    parser.add_argument("--uspc_color_weight", type=float, default=1.0, help="inner weight for RGB-color text view loss")
    parser.add_argument("--uspc_struct_weight", type=float, default=1.0, help="inner weight for Thermal-structure text view loss")
    parser.add_argument("--uspc_full_weight", type=float, default=1.0, help="inner weight for Fused-full text view loss")
    parser.add_argument("--uspc_uncertainty_weighting", default=False, action='store_true', help="weight USPC semantic losses by uncertainty confidence")
    parser.set_defaults(uspc_weight_detach=True)
    parser.add_argument("--uspc_weight_detach", dest="uspc_weight_detach", action='store_true', help="detach uncertainty weights used by USPC")
    parser.add_argument("--no_uspc_weight_detach", dest="uspc_weight_detach", action='store_false', help="do not detach uncertainty weights used by USPC")
    parser.add_argument("--uspc_warmup_epochs", type=int, default=0, help="number of warmup epochs before enabling USPC loss")
    parser.add_argument("--uspc_active_until_epoch", default=-1, type=int, help="If >0, use USPC only for 1-based epochs <= this value; disable USPC afterwards.")
    parser.add_argument("--uspc_text_view_mode", type=str, default="rule", choices=["rule"], help="USPC text semantic view construction mode")
    parser.add_argument("--uspc_loss_type", type=str, default="sdm", choices=["sdm", "ce"], help="USPC internal alignment loss type")
    parser.add_argument("--debug_uspc", default=False, action='store_true', help="print extra USPC debug information")
    parser.add_argument("--use_dcalign_lite", default=False, action='store_true', help="enable DCAlign-lite v1.1 semantic decoupled alignment")
    parser.add_argument("--lambda_dcalign", type=float, default=0.0, help="maximum DCAlign-lite v1.1 loss weight after warmup/ramp")
    parser.add_argument("--dcalign_color_weight", type=float, default=1.0, help="inner weight for DCAlign RGB-color text alignment")
    parser.add_argument("--dcalign_struct_weight", type=float, default=1.0, help="inner weight for DCAlign thermal-structure text alignment")
    parser.add_argument("--dcalign_full_weight", type=float, default=0.25, help="inner weight for DCAlign fused-full text alignment")
    parser.add_argument("--dcalign_warmup_epochs", type=int, default=1, help="number of 1-based epochs before DCAlign starts ramping")
    parser.add_argument("--dcalign_ramp_epochs", type=int, default=3, help="number of epochs used to linearly ramp DCAlign to lambda_dcalign")
    parser.add_argument("--dcalign_active_until_epoch", type=int, default=-1, help="If >=0, use DCAlign only for 1-based epochs <= this value")
    parser.add_argument("--dcalign_detach_text_views", default=False, action='store_true', help="detach DCAlign semantic text-view features before loss")
    parser.add_argument("--dcalign_detach_image_feats", default=False, action='store_true', help="detach DCAlign image features before loss")
    parser.add_argument("--dcalign_debug", default=False, action='store_true', help="print extra DCAlign-lite debug information")
    parser.add_argument("--use_attr_aux", default=False, action='store_true', help="enable caption attribute auxiliary loss")
    parser.add_argument("--lambda_attr_aux", type=float, default=0.0, help="weight for caption attribute auxiliary loss")
    parser.add_argument("--attr_schema", type=str, default="basic_v1", help="caption attribute parser schema")
    parser.add_argument("--attr_feature_source", type=str, default="fused_det", choices=["fused_det", "fused_prob"], help="feature source for attribute auxiliary head")
    parser.add_argument("--attr_loss_type", type=str, default="ce", choices=["ce", "bce"], help="attribute auxiliary loss type")
    parser.add_argument("--attr_warmup_epochs", type=int, default=0, help="number of warmup epochs before enabling attribute auxiliary loss")
    parser.add_argument("--attr_active_until_epoch", type=int, default=-1, help="If >0, use attribute auxiliary loss only for 1-based epochs <= this value")
    parser.add_argument("--debug_attr_aux", default=False, action='store_true', help="print extra attribute auxiliary loss debug information")
    parser.add_argument("--use_uspc_slot_local", default=False, action='store_true', help="enable USPC-v2-B0 fixed-region local attribute alignment")
    parser.add_argument("--lambda_uspc_slot_local", type=float, default=0.0, help="weight for USPC-v2-B0 fixed-region local attribute alignment")
    parser.add_argument("--uspc_slot_patch_source", type=str, default="rgb", choices=["rgb", "thermal", "concat_rgbt"], help="patch source for USPC slot-local loss; B0 only guarantees rgb")
    parser.add_argument("--uspc_slot_region_mode", type=str, default="fixed_stripe", choices=["fixed_stripe"], help="region pooling mode for USPC slot-local loss")
    parser.add_argument("--uspc_slot_loss_type", type=str, default="ce", choices=["ce"], help="loss type for USPC slot-local attribute heads")
    parser.add_argument("--uspc_slot_warmup_epochs", type=int, default=0, help="number of warmup epochs before enabling USPC slot-local loss")
    parser.add_argument("--uspc_slot_active_until_epoch", type=int, default=-1, help="If >0, use USPC slot-local loss only for 1-based epochs <= this value")
    parser.add_argument("--uspc_slot_detach_patch", default=False, action='store_true', help="detach patch region features before USPC slot-local heads")
    parser.add_argument("--debug_uspc_slot_local", default=False, action='store_true', help="print extra USPC slot-local debug information")
    parser.add_argument("--use_mga_pseudo", default=False, action='store_true', help="enable structured pseudo-caption auxiliary training")
    parser.add_argument("--mga_pseudo_json", type=str, default="", help="structured pseudo-caption JSONL file path")
    parser.add_argument("--mga_min_weight", type=float, default=0.3, help="minimum pseudo text weight required for auxiliary losses")
    parser.add_argument("--mga_pseudo_sample_prob", type=float, default=0.35, help="probability of sampling pseudo text as auxiliary supervision per train sample")
    parser.add_argument("--mga_attr_sample_prob", type=float, default=0.5, help="probability of sampling rgb_color / thermal_shape attribute text per train sample")
    parser.add_argument("--mga_debug_print_samples", type=int, default=0, help="print first N matched MGA pseudo training samples during dataloader build")
    parser.add_argument("--use_mga_attr_align", default=False, action='store_true', help="enable RGB/T decoupled attribute alignment loss")
    parser.add_argument("--use_mga_negative_loss", default=False, action='store_true', help="enable reserved negative/absence text margin loss")
    parser.add_argument("--lambda_mga_pseudo_sdm", type=float, default=0.0, help="weight for MGA pseudo-caption SDM auxiliary loss")
    parser.add_argument("--lambda_mga_rgb_color", type=float, default=0.0, help="weight for MGA RGB color attribute alignment loss")
    parser.add_argument("--lambda_mga_t_shape", type=float, default=0.0, help="weight for MGA thermal shape attribute alignment loss")
    parser.add_argument("--lambda_mga_negative", type=float, default=0.0, help="weight for reserved MGA negative/absence margin loss")
    parser.add_argument("--mga_neg_margin", type=float, default=0.2, help="margin for MGA negative/absence text loss")
    parser.add_argument("--use_rde_robust", default=False, action='store_true', help="enable RDE/DURA-style robust weighting mechanism")
    parser.add_argument("--rde_warmup_epochs", type=int, default=10, help="number of epochs before enabling robust pseudo/noisy weighting")
    parser.add_argument("--rde_enable_pseudo_during_warmup", default=False, action='store_true', help="keep MGA pseudo/attribute auxiliary losses active during RDE warmup")
    parser.add_argument("--rde_conf_source", type=str, default="hybrid", choices=["mga_weight", "online_sim", "hybrid"], help="clean confidence source for RDE robust weighting")
    parser.add_argument("--rde_clean_threshold", type=float, default=0.70, help="confidence threshold for clean samples")
    parser.add_argument("--rde_uncertain_threshold", type=float, default=0.40, help="confidence threshold for uncertain samples")
    parser.add_argument("--rde_uncertain_weight", type=float, default=0.5, help="loss weight assigned to uncertain samples")
    parser.add_argument("--rde_noisy_weight", type=float, default=0.1, help="loss weight assigned to noisy samples")
    parser.add_argument("--lambda_rde_consensus", type=float, default=0.0, help="weight for RDE robust consensus auxiliary loss")
    parser.add_argument("--use_confidence_hard_negative", default=False, action='store_true', help="enable confidence-aware hard negative training")
    parser.add_argument("--conf_hn_min_weight", type=float, default=0.5, help="minimum confidence required for strong hard negative participation")
    parser.add_argument("--use_hard_negative_loss", default=False, action='store_true', help="enable RDE-style in-batch hard-negative ranking loss")
    parser.add_argument("--use_hardneg_aux", default=False, action='store_true', help="enable legacy batch-internal hard-negative-aware auxiliary loss")
    parser.add_argument("--lambda_hn", type=float, default=0.0, help="weight for hard-negative loss")
    parser.add_argument("--hn_margin", type=float, default=0.05, help="margin used by hard-negative hinge/ranking loss")
    parser.add_argument("--hn_topk", type=int, default=1, help="top-k hardest in-batch negatives per query for --use_hard_negative_loss")
    parser.add_argument("--hn_warmup_epochs", type=int, default=5, help="number of 1-based warmup epochs before enabling --use_hard_negative_loss")
    parser.add_argument("--hn_direction", type=str, default="t2i", choices=["t2i", "i2t", "both"], help="hard-negative direction")
    parser.add_argument("--hn_detach_neg", default=False, action='store_true', help="detach negative similarities in in-batch hard-negative loss")
    parser.add_argument("--hn_use_fused_image_feature", default=False, action='store_true', help="prefer fused deterministic image feature when available for hard-negative loss")
    parser.add_argument("--hn_debug", default=False, action='store_true', help="print extra in-batch hard-negative debug information")
    parser.add_argument("--hn_topn", type=int, default=3, help="average the top-n hardest in-batch negatives per query")
    parser.add_argument("--hn_start_epoch", type=int, default=5, help="1-based epoch index to start hard-negative auxiliary loss")
    parser.add_argument("--hardneg_version", type=str, default="v1", choices=["v1", "v2"], help="hard-negative auxiliary loss version")
    parser.add_argument("--hn_temperature", type=float, default=0.07, help="temperature for hard-negative v2 softplus/logsumexp scaling")
    parser.add_argument("--hn_loss_type", type=str, default="softplus", choices=["softplus", "logsumexp"], help="hard-negative v2 loss form")
    parser.add_argument("--hn_false_neg_margin", type=float, default=0.0, help="optional hard-negative v2 false-negative guard margin")
    parser.add_argument("--debug_hardneg_stats", default=False, action='store_true', help="print extra hard-negative auxiliary stats")
    parser.add_argument("--use_offline_hardneg", default=False, action='store_true', help="enable offline hard-negative image auxiliary loss")
    parser.add_argument("--offline_hardneg_index_file", type=str, default="", help="json index file built by tools/build_offline_hardneg_index.py")
    parser.add_argument("--offline_hn_lambda", type=float, default=0.05, help="weight for offline hard-negative auxiliary loss")
    parser.add_argument("--offline_hn_margin", type=float, default=0.05, help="margin for offline hard-negative softplus loss")
    parser.add_argument("--offline_hn_temperature", type=float, default=0.07, help="temperature for offline hard-negative softplus loss")
    parser.add_argument("--offline_hn_num_neg", type=int, default=1, help="number of offline hard-negative images loaded per sample")
    parser.add_argument("--offline_hn_topk_train", type=int, default=20, help="candidate pool size used from each offline hard-negative list during training")
    parser.add_argument("--offline_hn_start_epoch", type=int, default=1, help="1-based epoch index to start offline hard-negative loss")
    parser.add_argument("--offline_hn_sample_mode", type=str, default="random", choices=["random", "top1"], help="how to sample train-time offline hard negatives")
    parser.add_argument("--debug_offline_hn_stats", default=False, action='store_true', help="print extra offline hard-negative stats")

    ######################## global uncertainty feature settings ########################
    # Feature-level uncertainty modeling over global embeddings only.
    parser.add_argument("--use_global_uncertainty", default=False, action='store_true', help="enable global diagonal-Gaussian feature heads")
    parser.add_argument("--uncertainty_dist_type", type=str, default="diag_gaussian", help="global uncertainty distribution type")
    parser.add_argument("--logvar_min", type=float, default=-6.0, help="minimum clamp value for predicted log-variance")
    parser.add_argument("--logvar_max", type=float, default=4.0, help="maximum clamp value for predicted log-variance")
    parser.add_argument("--use_precision_fusion", default=None, action='store_true', help="use precision-weighted RGB-T fusion for Gaussian globals")
    parser.add_argument("--no_use_precision_fusion", dest='use_precision_fusion', action='store_false', help="disable precision fusion and fallback to baseline fusion")
    parser.add_argument("--uncertainty_loss_type", type=str, default="w2", help="closed-form uncertainty distance type")
    parser.add_argument("--uncertainty_eps", type=float, default=1e-6, help="numeric epsilon for uncertainty computations")
    parser.add_argument("--lambda_unc", type=float, default=0.1, help="weight for uncertainty distribution alignment loss")
    parser.add_argument("--lambda_tr", type=float, default=1.0, help="inner weight for text-rgb uncertainty alignment")
    parser.add_argument("--lambda_rt", type=float, default=1.0, help="inner weight for rgb-thermal uncertainty alignment")
    parser.add_argument("--lambda_tt", type=float, default=0.0, help="inner weight for text-thermal uncertainty alignment")
    parser.add_argument("--lambda_var_reg", type=float, default=1e-4, help="weight for log-variance regularization")
    parser.add_argument("--use_distribution_sdm_aux", default=False, action='store_true', help="enable distribution-SDM auxiliary loss over global uncertainty distance matrices")
    parser.add_argument("--lambda_dist_sdm", type=float, default=1.0, help="outer weight for distribution-SDM auxiliary loss")
    parser.add_argument("--lambda_dist_sdm_tr", type=float, default=1.0, help="inner weight for text-rgb distribution-SDM term")
    parser.add_argument("--use_distribution_sdm_rgb_thermal", default=False, action='store_true', help="optionally enable rgb-thermal distribution-SDM term")
    parser.add_argument("--lambda_dist_sdm_rt", type=float, default=1.0, help="inner weight for rgb-thermal distribution-SDM term")
    parser.add_argument("--distribution_sdm_temperature", type=float, default=1.0, help="temperature for converting W2 distances into distribution-SDM logits")
    parser.add_argument("--eval_use_fused_mu", default=True, action='store_true', help="use fused_mu as visual embedding in evaluation when available")
    parser.add_argument("--no_eval_use_fused_mu", dest='eval_use_fused_mu', action='store_false', help="disable fused_mu in evaluation")
    parser.add_argument("--eval_use_mu_only", default=False, action='store_true', help="force evaluation to use rgb_mu only")
    parser.add_argument("--no_eval_use_mu_only", dest='eval_use_mu_only', action='store_false', help="disable rgb_mu-only evaluation")

    ######################## MUM / probabilistic alignment settings ########################
    parser.add_argument("--use_mum", default=False, action='store_true', help="use probabilistic global alignment")
    parser.add_argument("--omega", type=float, default=0.5, help="batch/id uncertainty mixing weight")
    parser.add_argument("--uncertainty_scale", type=float, default=1.0, help="global uncertainty scale")
    parser.add_argument("--memory_bank_size", type=int, default=0, help="0 means using num_classes as memory size")
    parser.add_argument("--lambda_det", type=float, default=0.5, help="weight for deterministic SDM loss")
    parser.add_argument("--lambda_mlm", type=float, default=1.0, help="weight for mlm loss in probabilistic training")
    parser.add_argument("--use_rgbt_add_fusion", default=True, action='store_true', help="use F_c = V_rgb_c + V_t_c")
    parser.add_argument("--no_use_rgbt_add_fusion", dest='use_rgbt_add_fusion', action='store_false', help="disable RGB-T add fusion, use RGB-only global feature")
    parser.add_argument("--use_batch_uncertainty", default=True, action='store_true', help="enable batch-level uncertainty")
    parser.add_argument("--no_use_batch_uncertainty", dest='use_batch_uncertainty', action='store_false', help="disable batch-level uncertainty")
    parser.add_argument("--use_id_uncertainty", default=True, action='store_true', help="enable identity-level uncertainty from memory bank")
    parser.add_argument("--no_use_id_uncertainty", dest='use_id_uncertainty', action='store_false', help="disable identity-level uncertainty")
    parser.add_argument("--use_prob_sampling", default=False, action='store_true', help="enable probabilistic reparameterization sampling")
    parser.add_argument("--no_use_prob_sampling", dest='use_prob_sampling', action='store_false', help="disable probabilistic reparameterization sampling")
    # Ablation presets:
    # 1) no MUM: --no_use_rgbt_add_fusion(optional baseline) --loss_names sdm+...
    #    with --use_mum omitted (False by default)
    # 2) batch-only UM: --use_mum --use_batch_uncertainty --no_use_id_uncertainty
    # 3) id-only UM: --use_mum --no_use_batch_uncertainty --use_id_uncertainty
    # 4) full MUM: --use_mum --use_batch_uncertainty --use_id_uncertainty
    # 5) deterministic training: --use_mum --no_use_prob_sampling (default)
    # 6) probabilistic training: --use_mum --use_prob_sampling

    ######################## visual RGBT MoE adapter settings ########################
    parser.add_argument("--use_rgbt_moe_adapter", default=False, action='store_true', help="enable the visual RGBT MoE adapter")
    parser.add_argument("--moe_adapter_layers", type=str, default='6,7,8,9,10,11', help="comma-separated visual block indices used by the RGBT MoE adapter")
    parser.add_argument("--moe_adapter_dim", type=int, default=64, help="bottleneck dim of the RGBT MoE adapter")
    parser.add_argument("--moe_adapter_dropout", type=float, default=0.1, help="dropout used by the RGBT MoE adapter")
    parser.add_argument("--moe_router_temp", type=float, default=1.0, help="router temperature of the RGBT MoE adapter")
    parser.add_argument("--lambda_router_balance", type=float, default=0.01, help="reserved router balance loss weight for the RGBT MoE adapter")
    parser.add_argument("--lambda_shared_consistency", type=float, default=0.05, help="reserved shared consistency loss weight for the RGBT MoE adapter")
    parser.add_argument("--moe_aux_schedule_mode", type=str, default="constant", choices=["constant", "three_stage"], help="schedule mode for MoE auxiliary loss weights")
    parser.add_argument("--moe_aux_stage1_end", type=int, default=10, help="inclusive end epoch of MoE auxiliary schedule stage 1")
    parser.add_argument("--moe_aux_stage2_end", type=int, default=30, help="inclusive end epoch of MoE auxiliary schedule stage 2")
    parser.add_argument("--lambda_router_balance_stage1", type=float, default=0.03, help="router balance weight used in MoE auxiliary schedule stage 1")
    parser.add_argument("--lambda_router_balance_stage2", type=float, default=0.02, help="router balance weight used in MoE auxiliary schedule stage 2")
    parser.add_argument("--lambda_router_balance_stage3", type=float, default=0.01, help="router balance weight used in MoE auxiliary schedule stage 3")
    parser.add_argument("--lambda_shared_consistency_stage1", type=float, default=0.10, help="shared consistency weight used in MoE auxiliary schedule stage 1")
    parser.add_argument("--lambda_shared_consistency_stage2", type=float, default=0.08, help="shared consistency weight used in MoE auxiliary schedule stage 2")
    parser.add_argument("--lambda_shared_consistency_stage3", type=float, default=0.05, help="shared consistency weight used in MoE auxiliary schedule stage 3")
    parser.add_argument("--moe_shared_consistency_mode", type=str, default="batch_mean", choices=["batch_mean", "id_pair"], help="shared consistency supervision mode for the RGBT MoE adapter")
    parser.add_argument("--moe_shared_consistency_layers", type=str, default='all', help="shared consistency layer selection: 'all' keeps the current A2.1 behavior, or provide a comma-separated layer list such as '10,11'")
    parser.add_argument("--log_moe_pid_stats", default=False, action='store_true', help="log pid-aware MoE statistics when available")
    parser.add_argument("--use_rgbt_moe_adapter_a2", default=False, action='store_true', help="enable A2-only training refinements for the visual RGBT MoE adapter")
    parser.add_argument("--lambda_router_warmup", type=float, default=0.0, help="A2-only extra loss weight for early shared-favoring router warmup")
    parser.add_argument("--moe_router_warmup_epochs", type=int, default=0, help="A2-only number of early epochs using shared-favoring router warmup")
    parser.add_argument("--moe_router_shared_target_start", type=float, default=0.8, help="A2-only target shared expert probability at the first warmup epoch")
    parser.add_argument("--moe_router_shared_target_end", type=float, default=0.5, help="A2-only target shared expert probability at the last warmup epoch")

    ######################## image-only local filtering v1 settings ########################
    # Lightweight patch selection/refinement over visual global features only.
    parser.add_argument("--use_img_local_filter", default=False, action='store_true', help="enable image-only local filtering v1 on visual global features")
    parser.add_argument("--img_local_topr", type=int, default=4, help="top-r patch count for image-only local filtering v1")
    parser.add_argument("--img_local_gamma", type=float, default=0.1, help="fusion weight gamma for image-only local filtering v1")
    parser.add_argument("--img_local_temp", type=float, default=0.07, help="temperature for image-only local filtering v1 softmax weighting")

    ######################## image-only local filtering settings ########################
    # Optional patch-token filtering over image features only.
    parser.add_argument("--use_image_local_filter", default=False, action='store_true', help="enable image-only local filtering before visual global alignment")
    parser.add_argument("--img_local_stage_mode", type=str, default="always", choices=["always", "window"], help="epoch scheduling mode for image-only local filtering")
    parser.add_argument("--img_local_stage_start", type=int, default=1, help="inclusive start epoch when img_local_stage_mode=window")
    parser.add_argument("--img_local_stage_end", type=int, default=9999, help="inclusive end epoch when img_local_stage_mode=window")
    parser.add_argument("--img_local_topk", type=int, default=64, help="number of patch tokens kept by image-only local filtering")
    parser.add_argument("--img_local_unc_lambda", type=float, default=0.3, help="penalty weight for patch uncertainty in image-only local filtering")
    parser.add_argument("--img_local_fuse_gamma", type=float, default=0.1, help="fusion weight for local aggregated feature")
    parser.add_argument("--img_local_gamma_schedule_mode", type=str, default="constant", choices=["constant", "three_stage"], help="runtime schedule mode for effective image-local fuse gamma")
    parser.add_argument("--img_local_gamma_stage1_end", type=int, default=6, help="inclusive end epoch of gamma stage 1 when img_local_gamma_schedule_mode=three_stage")
    parser.add_argument("--img_local_gamma_stage2_end", type=int, default=12, help="inclusive end epoch of gamma stage 2 when img_local_gamma_schedule_mode=three_stage")
    parser.add_argument("--img_local_gamma_stage1_value", type=float, default=0.05, help="effective gamma used in stage 1 when img_local_gamma_schedule_mode=three_stage")
    parser.add_argument("--img_local_gamma_stage2_value", type=float, default=0.02, help="effective gamma used in stage 2 when img_local_gamma_schedule_mode=three_stage")
    parser.add_argument("--img_local_gamma_stage3_value", type=float, default=0.0, help="effective gamma used in stage 3 when img_local_gamma_schedule_mode=three_stage")
    parser.add_argument("--img_local_score_temp", type=float, default=0.07, help="temperature for image-only patch relevance scores")
    parser.add_argument("--img_local_detach_global", default=True, action='store_true', help="detach global feature when computing image-only patch relevance (shared by legacy path and v1 helper)")
    parser.add_argument("--no_img_local_detach_global", dest='img_local_detach_global', action='store_false', help="keep global feature trainable inside image-only patch relevance")

    ######################## inference-only query-adaptive local rerank ########################
    # Only active during evaluation; training and baseline inference stay unchanged when disabled.
    parser.add_argument("--use_qalr", default=False, action='store_true', help="enable inference-only query-adaptive local reranking on top-K gallery candidates")
    parser.add_argument("--rerank_topk", type=int, default=128, help="number of top-ranked gallery candidates per query used for local reranking")
    parser.add_argument("--rerank_alpha", type=float, default=0.2, help="fusion weight for local rerank score: final = global + alpha * local")
    parser.add_argument("--rerank_token_topm", type=int, default=8, help="number of salient text tokens kept for local reranking")
    parser.add_argument("--rerank_patch_topr", type=int, default=4, help="number of top-matching RGB/T patches averaged per selected text token")
    parser.add_argument("--use_qalr_score_norm", default=False, action='store_true', help="enable query-wise local score normalization inside QALR")
    parser.add_argument("--qalr_score_norm_type", type=str, default="zscore", help="QALR local score normalization type")
    parser.add_argument("--use_qalr_query_gate", default=False, action='store_true', help="enable hard-query gating that decides whether each query uses QALR")
    parser.add_argument("--qalr_query_gate_type", type=str, default="top12_margin", help="QALR query gating type")
    parser.add_argument("--qalr_query_gate_margin", type=float, default=0.05, help="margin threshold on global top1-top2 score gap for enabling QALR on a query")
    parser.add_argument("--use_qalr_focus_topn", default=False, action='store_true', help="enable focus top-n rerank inside the selected global top-K candidates")
    parser.add_argument("--qalr_focus_topn", type=int, default=16, help="only apply local rerank fusion to the first n globally ranked candidates inside rerank_topk")
    parser.add_argument("--use_qalr_v2", default=False, action='store_true', help="enable safer QALR v2 local reranking")
    parser.add_argument("--qalr_v2_save_diag", default=False, action='store_true', help="save QALR/QALR-v2 per-query diagnostics")
    parser.add_argument("--qalr_v2_diag_file", type=str, default="", help="QALR-v2 diagnostics json output file")
    parser.add_argument("--qalr_v2_alpha_min", type=float, default=0.0, help="minimum adaptive alpha for QALR v2")
    parser.add_argument("--qalr_v2_alpha_max", type=float, default=0.01, help="maximum adaptive alpha for QALR v2")
    parser.add_argument("--qalr_v2_margin_low", type=float, default=0.0, help="margin where QALR v2 uses alpha_max")
    parser.add_argument("--qalr_v2_margin_high", type=float, default=0.02, help="margin where QALR v2 uses alpha_min")
    parser.add_argument("--qalr_v2_freeze_top1_margin", type=float, default=0.02, help="freeze global top1 above this margin when QALR v2 top1 guard is enabled")
    parser.add_argument("--qalr_v2_promote_margin", type=float, default=0.0, help="minimum local advantage required to replace global top1 in QALR v2 promote-only mode")
    parser.add_argument("--qalr_v2_modality_fuse", type=str, default="concat", choices=["concat", "avg", "rgb", "thermal", "precision"], help="QALR v2 local score modality fusion")
    parser.add_argument("--qalr_v2_top1_guard", default=False, action='store_true', help="enable QALR v2 global top1 freeze guard")
    parser.add_argument("--qalr_v2_promote_only", default=False, action='store_true', help="allow QALR v2 top1 replacement only with enough local advantage")
    parser.add_argument("--qalr_v2_diag_only", default=False, action='store_true', help="compute QALR v2 diagnostics without changing final eval scores")
    parser.add_argument("--qalr_v3_candidate_diag", default=False, action='store_true', help="save QALR v3 candidate-level oracle diagnostics without changing eval scores")
    parser.add_argument("--qalr_v3_candidate_diag_file", type=str, default="", help="QALR v3 candidate diagnostics json output file")
    parser.add_argument("--qalr_v3_candidate_topk", type=int, default=64, help="global top-K candidates saved for QALR v3 candidate diagnostics")
    parser.add_argument("--qalr_v3_candidate_token_topm", type=int, default=8, help="number of salient text tokens for QALR v3 candidate diagnostics")
    parser.add_argument("--qalr_v3_candidate_patch_topr", type=int, default=4, help="number of top patches per token for QALR v3 candidate diagnostics")
    parser.add_argument("--qalr_v3_candidate_save_scores", default=False, action='store_true', help="save candidate-level score fields in QALR v3 diagnostics")
    parser.add_argument("--use_qalr_v3_gate", default=False, action='store_true', help="enable QALR v3 gated top1 promotion rerank")
    parser.add_argument("--qalr_v3_gate_mode", type=str, default="concat", choices=["concat", "avg", "rgb", "thermal"], help="QALR v3 gate local score mode")
    parser.add_argument("--qalr_v3_gate_topk", type=int, default=64, help="global top-K candidate pool for QALR v3 gate")
    parser.add_argument("--qalr_v3_gate_token_topm", type=int, default=8, help="number of salient text tokens for QALR v3 gate")
    parser.add_argument("--qalr_v3_gate_patch_topr", type=int, default=4, help="number of top patches per token for QALR v3 gate")
    parser.add_argument("--qalr_v3_gate_candidate_rank_max", type=int, default=2, help="maximum global candidate rank allowed for QALR v3 gate promotion")
    parser.add_argument("--qalr_v3_gate_local_rank_max", type=int, default=1, help="maximum local rank allowed for QALR v3 gate promotion")
    parser.add_argument("--qalr_v3_gate_global_margin_max", type=float, default=0.01, help="maximum global top1-top2 margin allowed for QALR v3 gate")
    parser.add_argument("--qalr_v3_gate_local_gap_min", type=float, default=0.01, help="minimum local score gap over old top1 required for QALR v3 gate")
    parser.add_argument("--qalr_v3_gate_save_diag", default=False, action='store_true', help="save QALR v3 gate per-query diagnostics")
    parser.add_argument("--qalr_v3_gate_diag_file", type=str, default="", help="QALR v3 gate diagnostics json output file")
    parser.add_argument("--use_qalr_v3_two_stage_gate", default=False, action='store_true', help="enable QALR v3 two-stage gated top1 promotion rerank")
    parser.add_argument("--qalr_v3_stage1_mode", type=str, default="avg", choices=["concat", "avg", "rgb", "thermal"], help="QALR v3 two-stage gate stage1 local score mode")
    parser.add_argument("--qalr_v3_stage1_global_margin_max", type=float, default=0.008, help="stage1 maximum global top1-top2 margin")
    parser.add_argument("--qalr_v3_stage1_local_gap_min", type=float, default=0.015, help="stage1 minimum local score gap over old top1")
    parser.add_argument("--qalr_v3_stage2_mode", type=str, default="avg", choices=["concat", "avg", "rgb", "thermal"], help="QALR v3 two-stage gate stage2 local score mode")
    parser.add_argument("--qalr_v3_stage2_global_margin_min", type=float, default=0.008, help="stage2 minimum global top1-top2 margin")
    parser.add_argument("--qalr_v3_stage2_global_margin_max", type=float, default=0.010, help="stage2 maximum global top1-top2 margin")
    parser.add_argument("--qalr_v3_stage2_local_gap_min", type=float, default=0.011, help="stage2 minimum local score gap over old top1")
    parser.add_argument("--qalr_v3_two_stage_topk", type=int, default=64, help="global top-K candidate pool for QALR v3 two-stage gate")
    parser.add_argument("--qalr_v3_two_stage_candidate_rank_max", type=int, default=2, help="maximum global candidate rank allowed for QALR v3 two-stage gate")
    parser.add_argument("--qalr_v3_two_stage_local_rank_max", type=int, default=1, help="maximum local rank allowed for QALR v3 two-stage gate")
    parser.add_argument("--qalr_v3_two_stage_token_topm", type=int, default=8, help="number of salient text tokens for QALR v3 two-stage gate")
    parser.add_argument("--qalr_v3_two_stage_patch_topr", type=int, default=4, help="number of top patches per token for QALR v3 two-stage gate")
    parser.add_argument("--qalr_v3_two_stage_save_diag", default=False, action='store_true', help="save QALR v3 two-stage gate per-query diagnostics")
    parser.add_argument("--qalr_v3_two_stage_diag_file", type=str, default="", help="QALR v3 two-stage gate diagnostics json output file")

    ######################## inference-only USPC text-slot local rerank ########################
    # Evaluation-only USPC-v2C switch. This does not add training losses or checkpoint weights.
    parser.add_argument("--use_uspc_slot_rerank", default=False, action='store_true', help="enable USPC-v2C text-slot guided inference-time local reranking")
    parser.add_argument("--uspc_slot_rerank_topk", type=int, default=50, help="number of top-ranked gallery candidates per query used for USPC slot reranking")
    parser.add_argument("--uspc_slot_rerank_alpha", type=float, default=0.0, help="fusion weight for USPC slot rerank score: final = global + alpha * slot")
    parser.add_argument("--uspc_slot_rerank_score_norm", type=str, default="zscore", choices=["zscore", "none"], help="query-wise USPC slot rerank score normalization")
    parser.add_argument("--uspc_slot_rerank_region_mode", type=str, default="fixed_stripe", choices=["fixed_stripe"], help="region pooling mode for USPC slot rerank")
    parser.add_argument("--uspc_slot_rerank_patch_source", type=str, default="rgb", choices=["rgb"], help="patch source for USPC slot rerank; v2C first version supports RGB only")
    parser.add_argument("--uspc_slot_rerank_min_valid_slots", type=int, default=1, help="minimum parsed valid text slots required before applying USPC slot rerank to a query")
    parser.add_argument("--uspc_slot_rerank_skip_negative_bag", default=True, action='store_true', help="skip no-bag text slots during USPC slot rerank")
    parser.add_argument("--no_uspc_slot_rerank_skip_negative_bag", dest='uspc_slot_rerank_skip_negative_bag', action='store_false', help="include no-bag text slots during USPC slot rerank")
    parser.add_argument("--debug_uspc_slot_rerank", default=False, action='store_true', help="print extra USPC slot rerank debug information")

    ######################## inference-only transport rerank ########################
    # Placeholder CLI surface only. No training/evaluation path uses it until explicitly wired in.
    parser.add_argument("--use_transport_rerank", default=False, action='store_true', help="enable evaluation-only transport rerank placeholder")
    parser.add_argument("--transport_topk", type=int, default=128, help="number of top-ranked gallery candidates reserved for transport rerank")
    parser.add_argument("--transport_lambda", type=float, default=0.1, help="reserved fusion weight for transport rerank score")
    parser.add_argument("--transport_patch_topr", type=int, default=4, help="reserved number of top patches kept per transport matching step")
    parser.add_argument("--transport_token_topm", type=int, default=8, help="reserved number of salient text tokens kept for transport rerank")
    parser.add_argument("--transport_eps", type=float, default=1e-6, help="reserved numeric epsilon for transport rerank stability")
    parser.add_argument("--transport_iters", type=int, default=20, help="reserved number of transport solver iterations")
    parser.add_argument("--transport_fuse_mode", type=str, default="avg", help="reserved transport/global score fusion mode")
    parser.add_argument("--transport_score_norm", type=str, default="none", help="reserved transport score normalization mode")
    parser.add_argument("--transport_use_patch_filter", default=False, action='store_true', help="enable reserved patch pre-filtering inside transport rerank")
    parser.add_argument("--transport_use_token_filter", default=False, action='store_true', help="enable reserved token pre-filtering inside transport rerank")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## guided text token masking settings ########################
    # Guided text-side masking switches. They are no-op by default and do not add
    # standalone loss terms.
    parser.add_argument("--use_noise_token_masking", default=False, action='store_true', help="enable guided noisy-token masking")
    parser.add_argument("--use_informative_token_masking", default=False, action='store_true', help="enable guided informative-token MLM/MTP targets")
    parser.add_argument("--noise_mask_ratio", type=float, default=0.15, help="base ratio for noisy-token masking")
    parser.add_argument("--info_mask_ratio", type=float, default=0.15, help="base ratio for informative-token masking")
    parser.add_argument("--token_mask_score_source", type=str, default="none", choices=["none", "grad_attn", "attn", "grad"], help="token score source for guided masking; non-none uses the current lite text-feature proxy")
    parser.add_argument("--token_mask_min_keep", type=int, default=1, help="minimum number of non-special tokens to keep per text")
    parser.add_argument("--noise_mask_apply_to_mlm", default=False, action='store_true', help="also apply noisy-token mask to MLM input")
    info_mask_mlm_group = parser.add_mutually_exclusive_group()
    info_mask_mlm_group.add_argument("--info_mask_apply_to_mlm", dest="info_mask_apply_to_mlm", default=True, action='store_true', help="apply informative-token mask to MLM input")
    info_mask_mlm_group.add_argument("--no_info_mask_apply_to_mlm", dest="info_mask_apply_to_mlm", action='store_false', help="disable informative-token mask MLM application")
    guided_detach_group = parser.add_mutually_exclusive_group()
    guided_detach_group.add_argument("--guided_mask_detach_score", dest="guided_mask_detach_score", default=True, action='store_true', help="detach reserved token scores before guided masking")
    guided_detach_group.add_argument("--no_guided_mask_detach_score", dest="guided_mask_detach_score", action='store_false', help="keep reserved token scores attached")
    parser.add_argument("--lambda_guided_mtp", type=float, default=1.0, help="guided informative MLM/MTP loss multiplier")
    parser.add_argument("--debug_print_token_mask_stats", default=False, action='store_true', help="return guided token-mask debug statistics")
    parser.add_argument("--ablate_random_noise_mask", default=False, action='store_true', help="use random noisy-token masking ablation")
    parser.add_argument("--ablate_random_info_mask", default=False, action='store_true', help="use random informative-token masking ablation")

    ######################## modality-aware text prompting settings ########################
    parser.add_argument("--use_modality_text_prompt", default=False, action='store_true', help="enable modality-aware text prompting")
    parser.add_argument("--mtp_rgb_ctx_len", type=int, default=4, help="context token budget for rgb-friendly text prompt")
    parser.add_argument("--mtp_th_ctx_len", type=int, default=4, help="context token budget for thermal-friendly text prompt")
    parser.add_argument("--mtp_color_drop_mode", type=str, default='delete', choices=['delete', 'mask'], help="color-word processing mode for thermal-friendly text")
    parser.add_argument("--mtp_stage_mode", type=str, default='three_stage', choices=['constant', 'three_stage'], help="stage schedule mode for modality-text losses")
    parser.add_argument("--mtp_stage1_end", type=int, default=10, help="inclusive end epoch of modality-text stage 1")
    parser.add_argument("--mtp_stage2_end", type=int, default=45, help="inclusive end epoch of modality-text stage 2")
    parser.add_argument("--mtp_stage2_ramp_epochs", type=int, default=0, help="epochs used to smoothly ramp stage2 weights from stage1 values; 0 keeps legacy hard switch behavior")
    parser.add_argument("--mtp_stage3_ramp_epochs", type=int, default=0, help="epochs used to smoothly ramp stage3 weights from stage2 values; 0 keeps legacy hard switch behavior")
    parser.add_argument("--lambda_rgb_txt_stage1", type=float, default=0.0, help="stage1 weight for rgb-text auxiliary loss")
    parser.add_argument("--lambda_rgb_txt_stage2", type=float, default=0.08, help="stage2 weight for rgb-text auxiliary loss")
    parser.add_argument("--lambda_rgb_txt_stage3", type=float, default=0.10, help="stage3 weight for rgb-text auxiliary loss")
    parser.add_argument("--lambda_th_txt_stage1", type=float, default=0.0, help="stage1 weight for thermal-text auxiliary loss")
    parser.add_argument("--lambda_th_txt_stage2", type=float, default=0.03, help="stage2 weight for thermal-text auxiliary loss")
    parser.add_argument("--lambda_th_txt_stage3", type=float, default=0.04, help="stage3 weight for thermal-text auxiliary loss")
    parser.add_argument("--lambda_txt_cons_stage1", type=float, default=0.0, help="stage1 weight for text-consistency auxiliary loss")
    parser.add_argument("--lambda_txt_cons_stage2", type=float, default=0.003, help="stage2 weight for text-consistency auxiliary loss")
    parser.add_argument("--lambda_txt_cons_stage3", type=float, default=0.005, help="stage3 weight for text-consistency auxiliary loss")
    parser.add_argument("--mtp_eval_fusion_alpha", type=float, default=0.10, help="eval-time fusion alpha for modality-text prompting")
    parser.add_argument("--mtp_eval_fusion_beta", type=float, default=0.00, help="eval-time fusion beta for modality-text prompting")

    ######################## solver ########################
    # Optimizer hyperparameters.
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    # Dataset and dataloader hyperparameters.
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--dev_split_file", default="", help="optional RGBT-PEDES identity-level development split JSON")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--debug_max_train_batches", type=int, default=0, help="debug-only cap on train batches per epoch; <=0 keeps full training")
    parser.add_argument("--use_synth_caption_bank", default=False, action='store_true', help="enable synthetic caption bank append for train split only")
    parser.add_argument("--synth_caption_bank_file", type=str, default="", help="json file path of synthetic caption bank")
    parser.add_argument("--synth_caption_keep_per_image", type=int, default=2, help="max synthetic captions appended per image")
    parser.add_argument("--synth_caption_train_mode", type=str, default="append", choices=["append", "off"], help="train-time synthetic caption mode")
    parser.add_argument("--synth_caption_weight", type=float, default=None, help="alias for synth_loss_weight; only used when synth loss weighting is enabled")
    parser.add_argument("--use_synth_loss_weighting", default=False, action='store_true', help="enable lower per-sample loss weight metadata for synthetic captions")
    parser.add_argument("--synth_loss_weight", type=float, default=1.0, help="caption_weight assigned to synthetic captions when synth loss weighting is enabled")
    parser.add_argument("--synth_weight_apply_to_mlm", default=True, action='store_true', help="apply synthetic caption weights to MLM loss")
    parser.add_argument("--no_synth_weight_apply_to_mlm", dest='synth_weight_apply_to_mlm', action='store_false', help="do not apply synthetic caption weights to MLM loss")
    parser.add_argument("--synth_weight_apply_to_sdm", default=True, action='store_true', help="apply synthetic caption weights to SDM loss")
    parser.add_argument("--no_synth_weight_apply_to_sdm", dest='synth_weight_apply_to_sdm', action='store_false', help="do not apply synthetic caption weights to SDM loss")
    parser.add_argument("--synth_weight_apply_to_unc", default=False, action='store_true', help="apply synthetic caption weights to text-side uncertainty alignment loss")
    parser.add_argument("--debug_print_synth_weight_stats", default=False, action='store_true', help="print batch synthetic-caption weight stats every log period")
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    # A historical configs.yaml can provide the complete baseline protocol.
    # Only currently supported parser destinations are applied; explicit CLI
    # arguments below still take precedence. With no --config_file this path is
    # a no-op and preserves the legacy defaults exactly.
    config_cli, _ = parser.parse_known_args()
    if str(config_cli.config_file).strip():
        saved_config = load_config_mapping(config_cli.config_file)
        supported_destinations = {
            action.dest for action in parser._actions
        }
        parser.set_defaults(
            **{
                key: value
                for key, value in saved_config.items()
                if key in supported_destinations
            }
        )

    # Parse CLI args into namespace.
    args = parser.parse_args()
    normalized_mdrc_modes = []
    for item in args.mdrc_degrade_modes:
        for mode in str(item).split(","):
            mode = mode.strip()
            if mode:
                normalized_mdrc_modes.append(mode)
    invalid_mdrc_modes = [
        mode for mode in normalized_mdrc_modes if mode not in ALLOWED_DEGRADE_MODES
    ]
    if invalid_mdrc_modes:
        parser.error(
            "--mdrc_degrade_modes contains unsupported mode(s): "
            f"{invalid_mdrc_modes}. Expected one of {list(ALLOWED_DEGRADE_MODES)}."
        )
    if not normalized_mdrc_modes:
        parser.error("--mdrc_degrade_modes must include at least one mode.")
    args.mdrc_degrade_modes = normalized_mdrc_modes
    if float(args.mdrc_lambda) < 0.0:
        parser.error("--mdrc_lambda must be >= 0.")
    if float(args.mdrc_loss_scale) < 0.0:
        parser.error("--mdrc_loss_scale must be >= 0.")
    if not 0.0 <= float(args.mdrc_prob) <= 1.0:
        parser.error("--mdrc_prob must be in [0, 1].")
    if int(args.mdrc_warmup_epoch) < 0:
        parser.error("--mdrc_warmup_epoch must be >= 0.")
    if float(args.mdrc_temp) <= 0.0:
        parser.error("--mdrc_temp must be > 0.")
    try:
        validate_mraf_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.use_hard_negative_loss and args.use_hardneg_aux:
        parser.error(
            "--use_hard_negative_loss and legacy --use_hardneg_aux share "
            "loss_hn logging; enable only one hard-negative path."
        )
    if int(args.hn_topk) <= 0:
        parser.error("--hn_topk must be > 0.")
    if int(args.hn_warmup_epochs) < 0:
        parser.error("--hn_warmup_epochs must be >= 0.")
    if args.hn_debug:
        args.debug_hardneg_stats = True
    if args.synth_caption_weight is None:
        args.synth_caption_weight = args.synth_loss_weight
    else:
        args.synth_loss_weight = args.synth_caption_weight

    return args
