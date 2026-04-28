import argparse


def get_args():
    # Centralized CLI argument definitions for training/inference.
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
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
    parser.add_argument("--use_proto_calib", default=False, action='store_true', help="enable identity prototype calibration bank during training only")
    parser.add_argument("--lambda_proto_v", type=float, default=1.0, help="reserved visual prototype calibration loss weight")
    parser.add_argument("--lambda_proto_t", type=float, default=1.0, help="reserved text prototype calibration loss weight")
    parser.add_argument("--proto_temp", type=float, default=0.07, help="reserved temperature for prototype calibration logits")
    parser.add_argument("--proto_momentum", type=float, default=0.9, help="EMA momentum for identity prototype bank updates")
    parser.add_argument("--proto_warmup_epochs", type=int, default=0, help="number of warmup epochs before prototype losses are enabled")
    parser.add_argument("--use_hardneg_aux", default=False, action='store_true', help="enable batch-internal hard-negative-aware auxiliary loss")
    parser.add_argument("--lambda_hn", type=float, default=0.05, help="weight for hard-negative auxiliary loss")
    parser.add_argument("--hn_margin", type=float, default=0.05, help="margin used by hard-negative auxiliary hinge loss")
    parser.add_argument("--hn_topn", type=int, default=3, help="average the top-n hardest in-batch negatives per query")
    parser.add_argument("--hn_start_epoch", type=int, default=5, help="1-based epoch index to start hard-negative auxiliary loss")

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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_synth_caption_bank", default=False, action='store_true', help="enable synthetic caption bank append for train split only")
    parser.add_argument("--synth_caption_bank_file", type=str, default="", help="json file path of synthetic caption bank")
    parser.add_argument("--synth_caption_keep_per_image", type=int, default=2, help="max synthetic captions appended per image")
    parser.add_argument("--synth_caption_train_mode", type=str, default="append", choices=["append", "off"], help="train-time synthetic caption mode")
    parser.add_argument("--use_synth_loss_weighting", default=False, action='store_true', help="enable lower per-sample loss weight metadata for synthetic captions")
    parser.add_argument("--synth_loss_weight", type=float, default=1.0, help="caption_weight assigned to synthetic captions when synth loss weighting is enabled")
    parser.add_argument("--synth_weight_apply_to_mlm", default=True, action='store_true', help="apply synthetic caption weights to MLM loss")
    parser.add_argument("--no_synth_weight_apply_to_mlm", dest='synth_weight_apply_to_mlm', action='store_false', help="do not apply synthetic caption weights to MLM loss")
    parser.add_argument("--synth_weight_apply_to_sdm", default=True, action='store_true', help="apply synthetic caption weights to SDM loss")
    parser.add_argument("--no_synth_weight_apply_to_sdm", dest='synth_weight_apply_to_sdm', action='store_false', help="do not apply synthetic caption weights to SDM loss")
    parser.add_argument("--synth_weight_apply_to_unc", default=False, action='store_true', help="apply synthetic caption weights to text-side uncertainty alignment loss")
    parser.add_argument("--debug_print_synth_weight_stats", default=False, action='store_true', help="print batch synthetic-caption weight stats every log period")
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    # Parse CLI args into namespace.
    args = parser.parse_args()

    return args
