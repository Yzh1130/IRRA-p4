from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    # 测试脚本只需要一个配置文件路径：默认读取训练目录下保存的 configs.yaml。
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/iira/configs.yaml')
    args = parser.parse_args()
    # 直接复用训练期参数，保证测试与训练配置一致。
    args = load_train_configs(args.config_file)

    # 显式切换为推理模式，避免 build_dataloader 走训练分支。
    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    # 构建测试集加载器（图像 gallery + 文本 query）并创建模型。
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    # 加载训练得到的最优权重。
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)
    # 执行检索评估并输出 R@K/mAP 等指标。
    do_inference(model, test_img_loader, test_txt_loader)
