import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.comm import get_rank, synchronize
from utils.meter import AverageMeter
from utils.metrics import Evaluator


def _to_scalar(value):
    # Convert tensor/number into python float for stable meter/log writing.
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().mean().item()
    return float(value)


def _build_total_loss(args, ret):
    """
    When use_mum is enabled:
      total_loss = loss_sdm_prob + lambda_det * loss_sdm_det + lambda_mlm * loss_mlm
    Otherwise keep legacy behavior:
      total_loss = sum(all keys containing 'loss')
    """
    if getattr(args, "use_mum", False):
        loss_sdm_prob = ret.get("loss_sdm_prob", ret.get("sdm_loss", None))
        if loss_sdm_prob is None:
            raise KeyError(
                "use_mum=True requires model to return 'loss_sdm_prob' or 'sdm_loss'."
            )
        loss_sdm_det = ret.get("loss_sdm_det", None)
        if loss_sdm_det is None:
            loss_sdm_det = loss_sdm_prob.new_zeros(())
        loss_mlm = ret.get("loss_mlm", ret.get("mlm_loss", None))
        if loss_mlm is None:
            loss_mlm = loss_sdm_prob.new_zeros(())
        return (
            loss_sdm_prob
            + getattr(args, "lambda_det", 0.5) * loss_sdm_det
            + getattr(args, "lambda_mlm", 1.0) * loss_mlm
        )

    # Legacy mode: sum all losses but avoid double-counting SDM alias
    # ('sdm_loss' and 'loss_sdm_prob' point to the same term).
    total_loss = None
    for k, v in ret.items():
        if "loss" not in k:
            continue
        if k == "loss_sdm_prob" and "sdm_loss" in ret:
            continue
        total_loss = v if total_loss is None else total_loss + v

    if total_loss is None:
        raise KeyError("No loss term found in model outputs.")
    return total_loss


def do_train(
    start_epoch,
    args,
    model,
    train_loader,
    evaluator,
    optimizer,
    scheduler,
    checkpointer,
):
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch

    arguments = {"num_epoch": num_epoch, "iteration": 0}

    logger = logging.getLogger("IRRA.train")
    logger.info("start training")

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "loss_sdm_prob": AverageMeter(),
        "loss_sdm_det": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "loss_mlm": AverageMeter(),
        "avg_fused_batch_uncertainty": AverageMeter(),
        "avg_text_batch_uncertainty": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_top1 = 0.0

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)
            total_loss = _build_total_loss(args, ret)

            batch_size = batch["images"].shape[0]
            meters["loss"].update(_to_scalar(total_loss), batch_size)
            meters["sdm_loss"].update(_to_scalar(ret.get("sdm_loss", 0)), batch_size)
            meters["loss_sdm_prob"].update(
                _to_scalar(ret.get("loss_sdm_prob", ret.get("sdm_loss", 0))),
                batch_size,
            )
            meters["loss_sdm_det"].update(
                _to_scalar(ret.get("loss_sdm_det", 0)),
                batch_size,
            )
            meters["itc_loss"].update(_to_scalar(ret.get("itc_loss", 0)), batch_size)
            meters["id_loss"].update(_to_scalar(ret.get("id_loss", 0)), batch_size)
            meters["mlm_loss"].update(_to_scalar(ret.get("mlm_loss", 0)), batch_size)
            meters["loss_mlm"].update(
                _to_scalar(ret.get("loss_mlm", ret.get("mlm_loss", 0))), batch_size
            )

            if "sigma_batch_f" in ret:
                meters["avg_fused_batch_uncertainty"].update(
                    _to_scalar(ret["sigma_batch_f"]), batch_size
                )
            if "sigma_batch_t" in ret:
                meters["avg_text_batch_uncertainty"].update(
                    _to_scalar(ret["sigma_batch_t"]), batch_size
                )

            meters["img_acc"].update(_to_scalar(ret.get("img_acc", 0)), batch_size)
            meters["txt_acc"].update(_to_scalar(ret.get("txt_acc", 0)), batch_size)
            meters["mlm_acc"].update(_to_scalar(ret.get("mlm_acc", 0)), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.count > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        tb_writer.add_scalar("lr", scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar("temperature", _to_scalar(ret["temperature"]), epoch)
        for k, v in meters.items():
            if v.count > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % eval_period == 0 and get_rank() == 0:
            logger.info("Validation Results - Epoch: {}".format(epoch))
            if args.distributed:
                top1 = evaluator.eval(model.module.eval())
            else:
                top1 = evaluator.eval(model.eval())

            torch.cuda.empty_cache()
            if best_top1 < top1:
                best_top1 = top1
                arguments["epoch"] = epoch
                checkpointer.save("best", **arguments)

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):
    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    evaluator.eval(model.eval())
