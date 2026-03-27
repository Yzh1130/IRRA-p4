import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from model import objectives
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
        total_loss = (
            loss_sdm_prob
            + getattr(args, "lambda_det", 0.5) * loss_sdm_det
            + getattr(args, "lambda_mlm", 1.0) * loss_mlm
        )
    else:
        # Legacy mode: sum all losses but avoid double-counting SDM alias
        # ('sdm_loss' and 'loss_sdm_prob' point to the same term).
        total_loss = None
        for k, v in ret.items():
            if "loss" not in k:
                continue
            if k == "loss_sdm_prob" and "sdm_loss" in ret:
                continue
            # These terms are added explicitly with dedicated weights below.
            if k in {
                "loss_unc_total",
                "loss_unc_text_rgb",
                "loss_unc_rgb_thermal",
                "loss_unc_text_thermal",
                "loss_var_reg",
                "loss_dist_sdm_total",
                "loss_dist_sdm_text_rgb",
                "loss_dist_sdm_rgb_thermal",
            }:
                continue
            total_loss = v if total_loss is None else total_loss + v

        if total_loss is None:
            raise KeyError("No loss term found in model outputs.")

    if getattr(args, "use_global_uncertainty", False):
        if "loss_unc_total" in ret:
            total_loss = total_loss + getattr(args, "lambda_unc", 0.1) * ret["loss_unc_total"]
        if "loss_var_reg" in ret:
            total_loss = total_loss + getattr(args, "lambda_var_reg", 1e-4) * ret["loss_var_reg"]
    if getattr(args, "use_distribution_sdm_aux", False):
        if "loss_dist_sdm_total" in ret:
            total_loss = total_loss + getattr(args, "lambda_dist_sdm", 1.0) * ret["loss_dist_sdm_total"]
    return total_loss


def _mean_positive_pairwise_distance(dist_matrix, pids=None):
    """
    Average pairwise distance over positive matches.

    If pids are provided, positives are defined by pid equality (same as SDM label logic).
    Otherwise, fallback to diagonal matched pairs (index-aligned assumption).
    """
    if pids is None:
        return torch.diagonal(dist_matrix).mean()

    pids = pids.reshape(-1)
    pos_mask = pids.unsqueeze(1).eq(pids.unsqueeze(0))
    if pos_mask.any():
        return dist_matrix[pos_mask].mean()
    return torch.diagonal(dist_matrix).mean()


def _compute_uncertainty_alignment_terms(args, ret, batch):
    """
    Compute global uncertainty alignment losses from model outputs.

    Uses closed-form diagonal Gaussian W2^2 on global features:
      - text <-> rgb
      - rgb <-> thermal
      - text <-> thermal (optional, weighted by lambda_tt)

    Assumption:
      Current dataloader provides paired image-text samples per batch row.
      When repeated identities appear in a batch, we use pid-equality positive masks.
    """
    use_global_uncertainty = bool(getattr(args, "use_global_uncertainty", False))
    use_distribution_sdm_aux = bool(getattr(args, "use_distribution_sdm_aux", False))
    if use_distribution_sdm_aux and not use_global_uncertainty:
        raise ValueError(
            "use_distribution_sdm_aux=True requires --use_global_uncertainty."
        )
    if not use_global_uncertainty:
        return {}

    required = [
        "text_mu",
        "text_logvar",
        "rgb_mu",
        "rgb_logvar",
        "thermal_mu",
        "thermal_logvar",
    ]
    missing = [k for k in required if k not in ret]
    if missing:
        raise KeyError(
            "use_global_uncertainty=True requires model outputs to include "
            f"{missing}, but got keys: {list(ret.keys())}"
        )

    text_mu = ret["text_mu"]
    text_logvar = ret["text_logvar"]
    rgb_mu = ret["rgb_mu"]
    rgb_logvar = ret["rgb_logvar"]
    thermal_mu = ret["thermal_mu"]
    thermal_logvar = ret["thermal_logvar"]
    pids = batch.get("pids", None)
    has_thermal = "thermal_images" in batch

    eps = float(getattr(args, "uncertainty_eps", 1e-6))
    logvar_min = float(getattr(args, "logvar_min", -6.0))
    logvar_max = float(getattr(args, "logvar_max", 4.0))

    tr_dist = objectives.compute_w2_diag_pairwise_matrix(
        mu1=text_mu,
        logvar1=text_logvar,
        mu2=rgb_mu,
        logvar2=rgb_logvar,
        eps=eps,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )
    loss_unc_text_rgb = _mean_positive_pairwise_distance(tr_dist, pids)

    rt_dist = None
    if has_thermal:
        rt_dist = objectives.compute_w2_diag_pairwise_matrix(
            mu1=rgb_mu,
            logvar1=rgb_logvar,
            mu2=thermal_mu,
            logvar2=thermal_logvar,
            eps=eps,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        loss_unc_rgb_thermal = _mean_positive_pairwise_distance(rt_dist, pids)
    else:
        loss_unc_rgb_thermal = loss_unc_text_rgb.new_zeros(())

    if has_thermal and float(getattr(args, "lambda_tt", 0.0)) > 0.0:
        tt_dist = objectives.compute_w2_diag_pairwise_matrix(
            mu1=text_mu,
            logvar1=text_logvar,
            mu2=thermal_mu,
            logvar2=thermal_logvar,
            eps=eps,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        loss_unc_text_thermal = _mean_positive_pairwise_distance(tt_dist, pids)
    else:
        loss_unc_text_thermal = loss_unc_text_rgb.new_zeros(())

    loss_unc_total = (
        float(getattr(args, "lambda_tr", 1.0)) * loss_unc_text_rgb
        + float(getattr(args, "lambda_rt", 1.0)) * loss_unc_rgb_thermal
        + float(getattr(args, "lambda_tt", 0.0)) * loss_unc_text_thermal
    )

    if use_distribution_sdm_aux:
        if pids is None:
            raise KeyError(
                "use_distribution_sdm_aux=True requires batch to provide 'pids'."
            )
        dist_sdm_temp = float(getattr(args, "distribution_sdm_temperature", 1.0))
        image_ids = batch.get("image_ids", None)
        loss_dist_sdm_text_rgb = objectives.compute_distribution_sdm_from_distance(
            distance_matrix=tr_dist,
            pid=pids,
            image_id=image_ids,
            epsilon=eps,
            temperature=dist_sdm_temp,
        )

        if has_thermal and bool(
            getattr(args, "use_distribution_sdm_rgb_thermal", False)
        ):
            loss_dist_sdm_rgb_thermal = objectives.compute_distribution_sdm_from_distance(
                distance_matrix=rt_dist,
                pid=pids,
                image_id=image_ids,
                epsilon=eps,
                temperature=dist_sdm_temp,
            )
        else:
            loss_dist_sdm_rgb_thermal = loss_dist_sdm_text_rgb.new_zeros(())

        loss_dist_sdm_total = (
            float(getattr(args, "lambda_dist_sdm_tr", 1.0)) * loss_dist_sdm_text_rgb
            + float(getattr(args, "lambda_dist_sdm_rt", 1.0))
            * loss_dist_sdm_rgb_thermal
        )

    # Lightweight logvar regularization to avoid explosion/collapse.
    var_reg_terms = [text_logvar.pow(2).mean(), rgb_logvar.pow(2).mean()]
    if has_thermal:
        var_reg_terms.append(thermal_logvar.pow(2).mean())
    if "fused_logvar" in ret and torch.is_tensor(ret["fused_logvar"]):
        var_reg_terms.append(ret["fused_logvar"].pow(2).mean())
    loss_var_reg = sum(var_reg_terms) / len(var_reg_terms)

    out = {
        "loss_unc_total": loss_unc_total,
        "loss_unc_text_rgb": loss_unc_text_rgb,
        "loss_unc_rgb_thermal": loss_unc_rgb_thermal,
        "loss_unc_text_thermal": loss_unc_text_thermal,
        "loss_var_reg": loss_var_reg,
        "mean_text_logvar": text_logvar.mean(),
        "mean_rgb_logvar": rgb_logvar.mean(),
        "mean_thermal_logvar": thermal_logvar.mean(),
    }
    if use_distribution_sdm_aux:
        out.update(
            {
                "loss_dist_sdm_total": loss_dist_sdm_total,
                "loss_dist_sdm_text_rgb": loss_dist_sdm_text_rgb,
                "loss_dist_sdm_rgb_thermal": loss_dist_sdm_rgb_thermal,
            }
        )
    if "fused_logvar" in ret and torch.is_tensor(ret["fused_logvar"]):
        out["mean_fused_logvar"] = ret["fused_logvar"].mean()
    return out


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
        "loss_unc_total": AverageMeter(),
        "loss_unc_text_rgb": AverageMeter(),
        "loss_unc_rgb_thermal": AverageMeter(),
        "loss_unc_text_thermal": AverageMeter(),
        "loss_dist_sdm_total": AverageMeter(),
        "loss_dist_sdm_text_rgb": AverageMeter(),
        "loss_dist_sdm_rgb_thermal": AverageMeter(),
        "loss_var_reg": AverageMeter(),
        "mean_text_logvar": AverageMeter(),
        "mean_rgb_logvar": AverageMeter(),
        "mean_thermal_logvar": AverageMeter(),
        "mean_fused_logvar": AverageMeter(),
        "avg_fused_batch_uncertainty": AverageMeter(),
        "avg_text_batch_uncertainty": AverageMeter(),
        "avg_img_local_uncertainty": AverageMeter(),
        "avg_img_local_keep_ratio": AverageMeter(),
        "avg_img_local_weight_entropy": AverageMeter(),
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
            ret.update(_compute_uncertainty_alignment_terms(args, ret, batch))
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
            meters["loss_unc_total"].update(
                _to_scalar(ret.get("loss_unc_total", 0)), batch_size
            )
            meters["loss_unc_text_rgb"].update(
                _to_scalar(ret.get("loss_unc_text_rgb", 0)), batch_size
            )
            meters["loss_unc_rgb_thermal"].update(
                _to_scalar(ret.get("loss_unc_rgb_thermal", 0)), batch_size
            )
            meters["loss_unc_text_thermal"].update(
                _to_scalar(ret.get("loss_unc_text_thermal", 0)), batch_size
            )
            meters["loss_dist_sdm_total"].update(
                _to_scalar(ret.get("loss_dist_sdm_total", 0)), batch_size
            )
            meters["loss_dist_sdm_text_rgb"].update(
                _to_scalar(ret.get("loss_dist_sdm_text_rgb", 0)), batch_size
            )
            meters["loss_dist_sdm_rgb_thermal"].update(
                _to_scalar(ret.get("loss_dist_sdm_rgb_thermal", 0)), batch_size
            )
            meters["loss_var_reg"].update(
                _to_scalar(ret.get("loss_var_reg", 0)), batch_size
            )
            meters["mean_text_logvar"].update(
                _to_scalar(ret.get("mean_text_logvar", 0)), batch_size
            )
            meters["mean_rgb_logvar"].update(
                _to_scalar(ret.get("mean_rgb_logvar", 0)), batch_size
            )
            meters["mean_thermal_logvar"].update(
                _to_scalar(ret.get("mean_thermal_logvar", 0)), batch_size
            )
            meters["mean_fused_logvar"].update(
                _to_scalar(ret.get("mean_fused_logvar", 0)), batch_size
            )

            if "sigma_batch_f" in ret:
                meters["avg_fused_batch_uncertainty"].update(
                    _to_scalar(ret["sigma_batch_f"]), batch_size
                )
            if "sigma_batch_t" in ret:
                meters["avg_text_batch_uncertainty"].update(
                    _to_scalar(ret["sigma_batch_t"]), batch_size
                )
            if "avg_img_local_uncertainty" in ret:
                meters["avg_img_local_uncertainty"].update(
                    _to_scalar(ret["avg_img_local_uncertainty"]), batch_size
                )
            if "avg_img_local_keep_ratio" in ret:
                meters["avg_img_local_keep_ratio"].update(
                    _to_scalar(ret["avg_img_local_keep_ratio"]), batch_size
                )
            if "avg_img_local_weight_entropy" in ret:
                meters["avg_img_local_weight_entropy"].update(
                    _to_scalar(ret["avg_img_local_weight_entropy"]), batch_size
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
