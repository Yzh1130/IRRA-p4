import argparse


def get_args():
    # Centralized CLI argument definitions for training/inference.
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    # local_rank is injected by distributed launchers (torchrun).
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="baseline", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

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

    ######################## MUM / probabilistic alignment settings ########################
    parser.add_argument("--use_mum", default=False, action='store_true', help="use probabilistic global alignment")
    parser.add_argument("--omega", type=float, default=0.5, help="batch/id uncertainty mixing weight")
    parser.add_argument("--uncertainty_scale", type=float, default=1.0, help="global uncertainty scale")
    parser.add_argument("--memory_bank_size", type=int, default=0, help="0 means using num_classes as memory size")
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
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

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
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    # Parse CLI args into namespace.
    args = parser.parse_args()

    return args
