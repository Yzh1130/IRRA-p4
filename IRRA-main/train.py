import os
import os.path as op
import random
import time

import numpy as np
import torch

from datasets import build_dataloader
from model import build_model
from processor.processor import do_train
from solver import build_lr_scheduler, build_optimizer
from utils.checkpoint import Checkpointer
from utils.comm import get_rank, synchronize
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from utils.metrics import Evaluator
from utils.options import get_args


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _build_train_or_test_dataloaders(args):
    """
    Training mode keeps the original 4-value dataloader contract.
    Test mode accepts the 3-value contract and sets train_loader to None.
    """
    dataloader_outputs = build_dataloader(args)
    if args.training:
        if len(dataloader_outputs) != 4:
            raise ValueError(
                "Training mode expects build_dataloader(args) to return "
                f"4 values, got {len(dataloader_outputs)}."
            )
        train_loader, val_img_loader, val_txt_loader, num_classes = dataloader_outputs
    else:
        if len(dataloader_outputs) != 3:
            raise ValueError(
                "Test mode expects build_dataloader(args) to return "
                f"3 values, got {len(dataloader_outputs)}."
            )
        val_img_loader, val_txt_loader, num_classes = dataloader_outputs
        train_loader = None
    return train_loader, val_img_loader, val_txt_loader, num_classes


if __name__ == "__main__":
    args = get_args()
    set_seed(1 + get_rank())
    name = args.name

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    device = "cuda"
    if args.training:
        cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        args.output_dir = op.join(
            args.output_dir, args.dataset_name, f"{cur_time}_{name}"
        )

    logger = setup_logger(
        "IRRA",
        save_dir=args.output_dir,
        if_train=args.training,
        distributed_rank=get_rank(),
    )
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("[WARMSTART] load_ckpt_file={}".format(args.load_ckpt_file))
    logger.info(str(args).replace(",", "\n"))
    if args.training:
        save_train_configs(args.output_dir, args)

    train_loader, val_img_loader, val_txt_loader, num_classes = (
        _build_train_or_test_dataloaders(args)
    )
    model = build_model(args, num_classes)
    logger.info(
        "Total params: %2.fM"
        % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    optimizer = None
    scheduler = None
    if args.training:
        optimizer = build_optimizer(args, model)
        scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        if args.training:
            checkpoint = checkpointer.resume(args.resume_ckpt_file)
            start_epoch = checkpoint.get("epoch", 0)
        else:
            checkpointer.load(args.resume_ckpt_file)
    elif args.load_ckpt_file:
        checkpointer.load(args.load_ckpt_file)
        start_epoch = 0
        logger.info(
            "Warm-start loaded model weights only from {}".format(
                args.load_ckpt_file
            )
        )

    if args.training:
        do_train(
            start_epoch,
            args,
            model,
            train_loader,
            evaluator,
            optimizer,
            scheduler,
            checkpointer,
        )
    else:
        logger.info("Enter testing mode")
        eval_model = model.module if args.distributed else model
        evaluator.eval(eval_model.eval())
