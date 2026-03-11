import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize


def set_seed(seed=0):
    # 固定所有随机源，尽可能保证实验可复现。
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cudnn.deterministic=True 追求确定性；benchmark=True 会在固定输入尺寸时提升性能。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # 1) 解析命令行参数。
    args = get_args()
    # 分布式训练时给每个 rank 使用不同随机种子，避免完全相同的数据顺序。
    set_seed(1+get_rank())
    name = args.name

    # 2) 根据环境变量判断是否启用 DDP。
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        # 本地进程绑定到对应 GPU，并初始化通信组。
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        # 等待所有进程完成初始化，防止部分进程抢跑。
        synchronize()
    
    device = "cuda"
    # 3) 组织输出目录：logs/<dataset>/<timestamp>_<exp_name>。
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    # 仅主进程写日志文件，避免重复写入。
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    # 将当前实验参数完整落盘，便于后续复现实验与推理加载。
    save_train_configs(args.output_dir, args)

    # 4) 构建数据加载器与模型。
    # train_loader：训练图文对；val_img/txt_loader：验证阶段图像库与文本查询。
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    if args.distributed:
        # 使用 DDP 包装模型，让梯度在多卡间自动同步。
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # 当前实现不广播 buffer（如 BN 统计量）；若后续依赖 BN 统计，应重新评估该参数。
            broadcast_buffers=False,
        )
    # 5) 构建优化器与学习率调度器。
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    # 6) 构建 checkpoint 与评估器。
    # 仅主进程负责保存 checkpoint，防止多进程并发写同一文件。
    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    start_epoch = 1
    if args.resume:
        # 断点恢复：除模型权重外，还会恢复优化器/调度器状态。
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    # 7) 进入训练主循环。
    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
