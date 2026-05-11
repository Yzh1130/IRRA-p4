import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(
    image_fetures,
    text_fetures,
    pid,
    logit_scale,
    image_id=None,
    factor=0.3,
    epsilon=1e-8,
    sample_weight=None,
):
    """
    Similarity Distribution Matching
    """
    # 构造同 ID 的监督分布矩阵 labels（同 id 为 1，否则为 0）。
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        # 可选：融合 image_id 形成软标签，缓和单一 pid 标签的硬约束。
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    # 先做 L2 归一化，再计算图文余弦相似度矩阵。
    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    # 温度缩放后的相似度 logits。
    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # 将真实匹配标签归一化为分布。
    labels_distribute = labels / labels.sum(dim=1)

    # 预测分布与真实分布之间的 KL 形式损失（双向）。
    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    i2t_loss = torch.sum(i2t_loss, dim=1)
    t2i_loss = torch.sum(t2i_loss, dim=1)

    if sample_weight is None:
        loss = torch.mean(i2t_loss) + torch.mean(t2i_loss)
    else:
        sample_weight = sample_weight.to(
            device=image_fetures.device,
            dtype=i2t_loss.dtype,
        ).reshape(-1)
        if sample_weight.shape[0] != batch_size:
            raise ValueError(
                "sample_weight for SDM must have shape [batch_size], got "
                f"{tuple(sample_weight.shape)} for batch_size={batch_size}."
            )
        weight_sum = sample_weight.sum().clamp_min(epsilon)
        loss = (
            (i2t_loss * sample_weight).sum() / weight_sum
            + (t2i_loss * sample_weight).sum() / weight_sum
        )

    return loss


def _empty_inbatch_hard_negative_output(reference):
    zero = reference.new_zeros(())
    return {
        "loss": zero,
        "num_valid_queries": 0.0,
        "mean_pos_sim": zero,
        "mean_neg_sim": zero,
        "margin_violation_rate": zero,
        "t2i_loss": zero,
        "i2t_loss": zero,
        "t2i_valid": 0.0,
        "i2t_valid": 0.0,
    }


def _empty_inbatch_hard_negative_direction_output(reference):
    zero = reference.new_zeros(())
    return {
        "loss": zero,
        "num_valid_queries": 0.0,
        "mean_pos_sim": zero,
        "mean_neg_sim": zero,
        "margin_violation_rate": zero,
    }


def _compute_inbatch_hard_negative_direction(
    sim_matrix,
    query_pid,
    gallery_pid,
    margin,
    topk,
    detach_neg=False,
):
    zero = sim_matrix.new_zeros(())
    if sim_matrix.numel() == 0:
        return _empty_inbatch_hard_negative_direction_output(sim_matrix)

    finite_sim_mask = torch.isfinite(sim_matrix)
    sim_matrix = torch.nan_to_num(
        sim_matrix, nan=-1.0, posinf=1.0, neginf=-1.0
    ).clamp_(min=-1.0, max=1.0)

    pos_mask = query_pid.unsqueeze(1).eq(gallery_pid.unsqueeze(0))
    neg_mask = ~pos_mask
    valid_pos_mask = pos_mask & finite_sim_mask
    valid_neg_mask = neg_mask & finite_sim_mask

    pos_count = valid_pos_mask.sum(dim=1)
    neg_count = valid_neg_mask.sum(dim=1)
    valid_query_mask = (pos_count > 0) & (neg_count > 0)
    if not valid_query_mask.any():
        return _empty_inbatch_hard_negative_direction_output(sim_matrix)

    sim_pos = sim_matrix.masked_fill(~valid_pos_mask, float("-inf")).max(dim=1).values
    sim_pos = torch.nan_to_num(sim_pos, nan=0.0, posinf=1.0, neginf=-1.0)

    masked_neg_sim = sim_matrix.masked_fill(~valid_neg_mask, float("-inf"))
    effective_topk = min(int(topk), masked_neg_sim.shape[1])
    if effective_topk <= 0:
        return _empty_inbatch_hard_negative_direction_output(sim_matrix)

    topk_neg_sim = torch.topk(
        masked_neg_sim,
        k=effective_topk,
        dim=1,
        largest=True,
        sorted=False,
    ).values
    valid_topk_mask = torch.isfinite(topk_neg_sim)
    hardneg_count = valid_topk_mask.sum(dim=1)
    valid_query_mask = (
        valid_query_mask
        & hardneg_count.gt(0)
        & torch.isfinite(sim_pos)
    )
    if not valid_query_mask.any():
        return _empty_inbatch_hard_negative_direction_output(sim_matrix)

    neg_for_loss = topk_neg_sim.detach() if detach_neg else topk_neg_sim
    margin_gap = float(margin) + neg_for_loss - sim_pos.unsqueeze(1)
    margin_gap = margin_gap.masked_fill(~valid_topk_mask, 0.0)
    pair_loss = F.relu(margin_gap)
    per_query_loss = pair_loss.sum(dim=1) / hardneg_count.clamp_min(1).to(
        pair_loss.dtype
    )

    topk_neg_for_stats = topk_neg_sim.masked_fill(~valid_topk_mask, 0.0)
    per_query_neg_sim = topk_neg_for_stats.sum(dim=1) / hardneg_count.clamp_min(
        1
    ).to(topk_neg_for_stats.dtype)
    violation_gap = float(margin) + topk_neg_sim - sim_pos.unsqueeze(1)
    per_query_violation = (
        (violation_gap.gt(0) & valid_topk_mask).float().sum(dim=1)
        / hardneg_count.clamp_min(1).float()
    )

    return {
        "loss": per_query_loss[valid_query_mask].mean(),
        "num_valid_queries": float(valid_query_mask.sum().item()),
        "mean_pos_sim": sim_pos[valid_query_mask].mean(),
        "mean_neg_sim": per_query_neg_sim[valid_query_mask].mean(),
        "margin_violation_rate": per_query_violation[valid_query_mask].mean(),
    }


def compute_inbatch_hard_negative_loss(
    text_features,
    image_features,
    pids,
    margin,
    topk,
    direction,
    detach_neg=False,
    epsilon=1e-8,
):
    """
    In-batch hard-negative ranking loss over text-image similarities.

    Positives are same-pid pairs and negatives are different-pid pairs. Each query
    uses its maximum positive similarity and the top-k hardest valid negatives.
    """
    if int(topk) <= 0:
        raise ValueError(f"hn_topk must be > 0, got {topk}.")
    direction = str(direction).lower()
    if direction not in {"t2i", "i2t", "both"}:
        raise ValueError(f"hn_direction must be t2i, i2t or both, got {direction}.")
    if text_features.dim() != 2 or image_features.dim() != 2:
        raise ValueError(
            "In-batch hard-negative loss expects 2D features, "
            f"got {tuple(text_features.shape)} and {tuple(image_features.shape)}."
        )
    if text_features.shape[1] != image_features.shape[1]:
        raise ValueError(
            "text_features and image_features must share feature dim, got "
            f"{text_features.shape[1]} and {image_features.shape[1]}."
        )

    pids = pids.reshape(-1).to(device=text_features.device)
    if text_features.shape[0] != pids.shape[0] or image_features.shape[0] != pids.shape[0]:
        raise ValueError(
            "text_features, image_features and pids must share batch size, got "
            f"{text_features.shape[0]}, {image_features.shape[0]}, {pids.shape[0]}."
        )
    if text_features.shape[0] == 0 or image_features.shape[0] == 0:
        return _empty_inbatch_hard_negative_output(text_features)

    text_features = torch.nan_to_num(
        text_features.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    image_features = torch.nan_to_num(
        image_features.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    text_norm = F.normalize(text_features, p=2, dim=1, eps=epsilon)
    image_norm = F.normalize(image_features, p=2, dim=1, eps=epsilon)
    sim_matrix = text_norm @ image_norm.t()

    t2i = _compute_inbatch_hard_negative_direction(
        sim_matrix=sim_matrix,
        query_pid=pids,
        gallery_pid=pids,
        margin=margin,
        topk=topk,
        detach_neg=detach_neg,
    )
    i2t = _compute_inbatch_hard_negative_direction(
        sim_matrix=sim_matrix.t(),
        query_pid=pids,
        gallery_pid=pids,
        margin=margin,
        topk=topk,
        detach_neg=detach_neg,
    )

    if direction == "t2i":
        loss = t2i["loss"]
        selected = [t2i]
    elif direction == "i2t":
        loss = i2t["loss"]
        selected = [i2t]
    else:
        loss = 0.5 * (t2i["loss"] + i2t["loss"])
        selected = [t2i, i2t]

    total_valid = sum(float(item["num_valid_queries"]) for item in selected)
    zero = text_features.new_zeros(())

    def _weighted_stat(key):
        if total_valid <= 0:
            return zero
        out = zero
        for item in selected:
            out = out + item[key] * float(item["num_valid_queries"])
        return out / total_valid

    return {
        "loss": loss,
        "num_valid_queries": total_valid,
        "mean_pos_sim": _weighted_stat("mean_pos_sim"),
        "mean_neg_sim": _weighted_stat("mean_neg_sim"),
        "margin_violation_rate": _weighted_stat("margin_violation_rate"),
        "t2i_loss": t2i["loss"],
        "i2t_loss": i2t["loss"],
        "t2i_valid": t2i["num_valid_queries"],
        "i2t_valid": i2t["num_valid_queries"],
    }


def compute_distribution_sdm_from_distance(
    distance_matrix,
    pid,
    image_id=None,
    factor=0.3,
    epsilon=1e-8,
    temperature=1.0,
):
    """
    Distribution-aware SDM from pairwise distances.

    Args:
        distance_matrix: [B, B], pairwise distances (e.g., diagonal Gaussian W2^2).
        pid: [B], person IDs for positive-pair supervision.
        image_id: optional [B], same semantics as compute_sdm.
        factor: optional label fusion factor when image_id is provided.
        epsilon: numeric epsilon for stable log computation.
        temperature: positive scalar to scale distance logits.
    Returns:
        Scalar bidirectional SDM-style KL loss.
    """
    if distance_matrix.dim() != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(
            "distance_matrix for distribution-SDM must be square [B, B], "
            f"but got shape={tuple(distance_matrix.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"distribution_sdm_temperature must be > 0, got {temperature}")

    batch_size = distance_matrix.shape[0]
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id is not None:
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask

    labels_distribute = labels / labels.sum(dim=1, keepdim=True).clamp_min(epsilon)

    # Distances are converted to similarities by negative sign.
    logits_ab = -distance_matrix / temperature
    logits_ba = logits_ab.t()

    ab_pred = F.softmax(logits_ab, dim=1)
    ab_loss = ab_pred * (F.log_softmax(logits_ab, dim=1) - torch.log(labels_distribute + epsilon))
    ba_pred = F.softmax(logits_ba, dim=1)
    ba_loss = ba_pred * (F.log_softmax(logits_ba, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(ab_loss, dim=1)) + torch.mean(torch.sum(ba_loss, dim=1))
    return loss


def compute_mlm(scores, labels, sample_weight=None, batch_size=None, epsilon=1e-8):
    if sample_weight is not None:
        sample_weight = sample_weight.to(
            device=scores.device,
            dtype=scores.dtype,
        ).reshape(-1)
        if bool(torch.all(sample_weight.eq(1.0)).item()):
            ce = nn.CrossEntropyLoss(ignore_index=0)
            return ce(scores, labels)
        if batch_size is None:
            batch_size = int(sample_weight.shape[0])
        if int(batch_size) != int(sample_weight.shape[0]):
            raise ValueError(
                "batch_size must match sample_weight length for weighted MLM, got "
                f"{batch_size} and {sample_weight.shape[0]}."
            )
        if scores.shape[0] % int(batch_size) != 0:
            raise ValueError(
                "Flattened MLM scores must be divisible by batch_size, got "
                f"{scores.shape[0]} and {batch_size}."
            )

        seq_len = scores.shape[0] // int(batch_size)
        token_loss = F.cross_entropy(
            scores,
            labels,
            ignore_index=0,
            reduction="none",
        ).reshape(int(batch_size), seq_len)
        valid_mask = labels.reshape(int(batch_size), seq_len).ne(0)
        valid_count = valid_mask.sum(dim=1)
        valid_sample = valid_count.gt(0)
        sample_loss = token_loss.sum(dim=1) / valid_count.clamp_min(1).to(
            token_loss.dtype
        )
        effective_weight = sample_weight * valid_sample.to(sample_weight.dtype)
        weight_sum = effective_weight.sum().clamp_min(epsilon)
        return (sample_loss * effective_weight).sum() / weight_sum

    # label=0 代表非掩码位置，使用 ignore_index=0 忽略这些位置的损失。
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    # 正样本定义为同 batch 下对角线配对：第 i 张图对应第 i 条文本。
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # 特征归一化后点积等价于余弦相似度。
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # 双向 logits：image->text 与 text->image。
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    # 双向交叉熵平均，得到 ITC 损失。
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    # 图像/文本两个分支都进行身份分类监督。
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    # 返回双分支均值，保持损失量级稳定。
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    # 构造同 ID 匹配掩码。
    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    # 计算 image->text / text->image 的投影相似度。
    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # 真实匹配分布归一化，随后与预测分布做 KL 形式对齐。
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss


def compute_hard_negative_aux_loss(
    query_features,
    gallery_features,
    query_pid,
    gallery_pid=None,
    margin=0.2,
    topn=1,
    epsilon=1e-8,
):
    """
    Batch-internal hard-negative auxiliary hinge loss over a cross-modal similarity matrix.

    Positives are defined by pid equality. All same-pid samples are excluded from the
    negative pool by default. When multiple positives exist in-batch, their
    similarities are averaged to form pos_sim. For hard negatives, the top-n
    highest-similarity valid negatives are averaged per query.

    Loss per valid query:
        max(0, margin - pos_sim + hardneg_sim)
    """
    if topn <= 0:
        raise ValueError(f"hn_topn must be > 0, got {topn}.")
    if query_features.dim() != 2 or gallery_features.dim() != 2:
        raise ValueError(
            "Hard-negative auxiliary expects 2D features, "
            f"got {tuple(query_features.shape)} and {tuple(gallery_features.shape)}."
        )

    if gallery_pid is None:
        gallery_pid = query_pid

    query_pid = query_pid.reshape(-1)
    gallery_pid = gallery_pid.reshape(-1)
    if query_features.shape[0] != query_pid.shape[0]:
        raise ValueError(
            "query_features and query_pid must share the same batch dimension, "
            f"got {query_features.shape[0]} and {query_pid.shape[0]}."
        )
    if gallery_features.shape[0] != gallery_pid.shape[0]:
        raise ValueError(
            "gallery_features and gallery_pid must share the same batch dimension, "
            f"got {gallery_features.shape[0]} and {gallery_pid.shape[0]}."
        )

    zero = query_features.new_zeros(())
    if query_features.shape[0] == 0 or gallery_features.shape[0] == 0:
        return {
            "loss": zero,
            "avg_pos_sim": zero,
            "avg_neg_sim": zero,
            "avg_margin_gap": zero,
            "num_valid_queries": 0.0,
        }

    # Zero-fill invalid feature entries first so later cosine similarity stays finite.
    query_features = torch.nan_to_num(
        query_features.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    gallery_features = torch.nan_to_num(
        gallery_features.float(), nan=0.0, posinf=0.0, neginf=0.0
    )

    query_norm = query_features / query_features.norm(dim=1, keepdim=True).clamp_min(
        epsilon
    )
    gallery_norm = gallery_features / gallery_features.norm(
        dim=1, keepdim=True
    ).clamp_min(epsilon)
    sim_matrix = query_norm @ gallery_norm.t()
    finite_sim_mask = torch.isfinite(sim_matrix)
    sim_matrix = torch.nan_to_num(
        sim_matrix, nan=-1.0, posinf=1.0, neginf=-1.0
    ).clamp_(min=-1.0, max=1.0)

    pos_mask = query_pid.unsqueeze(1).eq(gallery_pid.unsqueeze(0))
    neg_mask = ~pos_mask
    valid_pos_mask = pos_mask & finite_sim_mask
    valid_neg_mask = neg_mask & finite_sim_mask

    pos_count = valid_pos_mask.sum(dim=1)
    neg_count = valid_neg_mask.sum(dim=1)
    valid_query_mask = (pos_count > 0) & (neg_count > 0)

    if not valid_query_mask.any():
        return {
            "loss": zero,
            "avg_pos_sim": zero,
            "avg_neg_sim": zero,
            "avg_margin_gap": zero,
            "num_valid_queries": 0.0,
        }

    pos_sim = (sim_matrix * valid_pos_mask.float()).sum(dim=1) / pos_count.clamp_min(
        1
    ).float()

    masked_neg_sim = sim_matrix.masked_fill(~valid_neg_mask, float("-inf"))
    topk = min(int(topn), masked_neg_sim.shape[1])
    topk_neg_sim = torch.topk(
        masked_neg_sim, k=topk, dim=1, largest=True, sorted=False
    ).values
    valid_topk_mask = torch.isfinite(topk_neg_sim)
    hardneg_count = valid_topk_mask.sum(dim=1)
    valid_query_mask = valid_query_mask & (hardneg_count > 0)
    if not valid_query_mask.any():
        return {
            "loss": zero,
            "avg_pos_sim": zero,
            "avg_neg_sim": zero,
            "avg_margin_gap": zero,
            "num_valid_queries": 0.0,
        }

    hardneg_sim = topk_neg_sim.masked_fill(~valid_topk_mask, 0.0).sum(dim=1) / (
        hardneg_count.clamp_min(1).float()
    )

    margin_gap = margin - pos_sim + hardneg_sim
    per_query_loss = F.relu(margin_gap)
    loss = per_query_loss[valid_query_mask].mean()
    valid_margin_gap = margin_gap[valid_query_mask]
    valid_top1_neg_sim = topk_neg_sim.masked_fill(~valid_topk_mask, float("-inf")).max(
        dim=1
    ).values[valid_query_mask]

    return {
        "loss": loss,
        "avg_pos_sim": pos_sim[valid_query_mask].mean(),
        "avg_neg_sim": hardneg_sim[valid_query_mask].mean(),
        "avg_margin_gap": valid_margin_gap.mean(),
        "violation_rate": valid_margin_gap.gt(0).float().mean(),
        "avg_top1_neg_sim": valid_top1_neg_sim.mean(),
        "avg_topk_neg_sim": hardneg_sim[valid_query_mask].mean(),
        "num_valid_queries": float(valid_query_mask.sum().item()),
    }


def _empty_hard_negative_v2_output(reference):
    zero = reference.new_zeros(())
    return {
        "loss": zero,
        "avg_pos_sim": zero,
        "avg_neg_sim": zero,
        "avg_margin_gap": zero,
        "violation_rate": zero,
        "avg_top1_neg_sim": zero,
        "avg_topk_neg_sim": zero,
        "num_valid_queries": 0.0,
        "t2i_loss": zero,
        "i2t_loss": zero,
        "t2i_valid": 0.0,
        "i2t_valid": 0.0,
    }


def _compute_hard_negative_v2_direction(
    sim_matrix,
    query_pid,
    gallery_pid,
    margin,
    topn,
    temperature,
    loss_type,
    false_neg_margin=0.0,
):
    zero = sim_matrix.new_zeros(())
    finite_sim_mask = torch.isfinite(sim_matrix)
    sim_matrix = torch.nan_to_num(
        sim_matrix, nan=-1.0, posinf=1.0, neginf=-1.0
    ).clamp_(min=-1.0, max=1.0)

    pos_mask = query_pid.unsqueeze(1).eq(gallery_pid.unsqueeze(0))
    neg_mask = ~pos_mask
    valid_pos_mask = pos_mask & finite_sim_mask
    valid_neg_mask = neg_mask & finite_sim_mask

    pos_count = valid_pos_mask.sum(dim=1)
    if not (pos_count > 0).any():
        return {
            "loss": zero,
            "avg_pos_sim": zero,
            "avg_neg_sim": zero,
            "avg_margin_gap": zero,
            "violation_rate": zero,
            "avg_top1_neg_sim": zero,
            "avg_topk_neg_sim": zero,
            "num_valid_queries": 0.0,
        }

    pos_sim = (sim_matrix * valid_pos_mask.float()).sum(dim=1) / pos_count.clamp_min(
        1
    ).float()

    if false_neg_margin > 0:
        false_neg_mask = valid_neg_mask & (
            sim_matrix >= (pos_sim.unsqueeze(1) - float(false_neg_margin))
        )
        valid_neg_mask = valid_neg_mask & ~false_neg_mask

    neg_count = valid_neg_mask.sum(dim=1)
    valid_query_mask = (pos_count > 0) & (neg_count > 0)
    if not valid_query_mask.any():
        return {
            "loss": zero,
            "avg_pos_sim": zero,
            "avg_neg_sim": zero,
            "avg_margin_gap": zero,
            "violation_rate": zero,
            "avg_top1_neg_sim": zero,
            "avg_topk_neg_sim": zero,
            "num_valid_queries": 0.0,
        }

    masked_neg_sim = sim_matrix.masked_fill(~valid_neg_mask, float("-inf"))
    topk = min(int(topn), masked_neg_sim.shape[1])
    topk_neg_sim = torch.topk(
        masked_neg_sim, k=topk, dim=1, largest=True, sorted=True
    ).values
    valid_topk_mask = torch.isfinite(topk_neg_sim)
    hardneg_count = valid_topk_mask.sum(dim=1)
    valid_query_mask = valid_query_mask & (hardneg_count > 0)
    if not valid_query_mask.any():
        return {
            "loss": zero,
            "avg_pos_sim": zero,
            "avg_neg_sim": zero,
            "avg_margin_gap": zero,
            "violation_rate": zero,
            "avg_top1_neg_sim": zero,
            "avg_topk_neg_sim": zero,
            "num_valid_queries": 0.0,
        }

    topk_neg_for_lse = topk_neg_sim.masked_fill(~valid_topk_mask, float("-inf"))
    hardneg_sim = torch.logsumexp(topk_neg_for_lse / temperature, dim=1) * temperature
    topk_mean_sim = topk_neg_sim.masked_fill(~valid_topk_mask, 0.0).sum(dim=1) / (
        hardneg_count.clamp_min(1).float()
    )
    top1_neg_sim = topk_neg_sim[:, 0]

    margin_gap = hardneg_sim - pos_sim + margin
    if loss_type == "softplus":
        per_query_loss = F.softplus(margin_gap / temperature) * temperature
    elif loss_type == "logsumexp":
        pairwise_gap = (
            topk_neg_for_lse - pos_sim.unsqueeze(1) + margin
        ) / temperature
        per_query_loss = F.softplus(torch.logsumexp(pairwise_gap, dim=1)) * temperature
    else:
        raise ValueError(
            f"hn_loss_type must be 'softplus' or 'logsumexp', got {loss_type}."
        )

    valid_loss = per_query_loss[valid_query_mask]
    valid_margin_gap = margin_gap[valid_query_mask]
    return {
        "loss": valid_loss.mean(),
        "avg_pos_sim": pos_sim[valid_query_mask].mean(),
        "avg_neg_sim": hardneg_sim[valid_query_mask].mean(),
        "avg_margin_gap": valid_margin_gap.mean(),
        "violation_rate": valid_margin_gap.gt(0).float().mean(),
        "avg_top1_neg_sim": top1_neg_sim[valid_query_mask].mean(),
        "avg_topk_neg_sim": topk_mean_sim[valid_query_mask].mean(),
        "num_valid_queries": float(valid_query_mask.sum().item()),
    }


def compute_hard_negative_aux_loss_v2(
    text_features,
    image_features,
    text_pid,
    image_pid=None,
    margin=0.05,
    topn=3,
    temperature=0.07,
    loss_type="softplus",
    direction="both",
    false_neg_margin=0.0,
    epsilon=1e-8,
):
    """
    Stable batch-internal hard-negative auxiliary loss over text-image similarities.

    Positives are all same-pid pairs. Negatives are different-pid pairs. Each anchor
    uses the top-k hardest in-batch negatives and a smooth softplus/logsumexp loss.
    """
    if topn <= 0:
        raise ValueError(f"hn_topn must be > 0, got {topn}.")
    if temperature <= 0:
        raise ValueError(f"hn_temperature must be > 0, got {temperature}.")
    if text_features.dim() != 2 or image_features.dim() != 2:
        raise ValueError(
            "Hard-negative v2 expects 2D features, "
            f"got {tuple(text_features.shape)} and {tuple(image_features.shape)}."
        )
    if direction not in {"t2i", "i2t", "both"}:
        raise ValueError(f"hn_direction must be t2i, i2t or both, got {direction}.")

    if image_pid is None:
        image_pid = text_pid
    text_pid = text_pid.reshape(-1)
    image_pid = image_pid.reshape(-1)
    if text_features.shape[0] != text_pid.shape[0]:
        raise ValueError(
            "text_features and text_pid must share the same batch dimension, "
            f"got {text_features.shape[0]} and {text_pid.shape[0]}."
        )
    if image_features.shape[0] != image_pid.shape[0]:
        raise ValueError(
            "image_features and image_pid must share the same batch dimension, "
            f"got {image_features.shape[0]} and {image_pid.shape[0]}."
        )

    zero = text_features.new_zeros(())
    if text_features.shape[0] == 0 or image_features.shape[0] == 0:
        return _empty_hard_negative_v2_output(text_features)

    text_features = torch.nan_to_num(
        text_features.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    image_features = torch.nan_to_num(
        image_features.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    text_norm = text_features / text_features.norm(dim=1, keepdim=True).clamp_min(
        epsilon
    )
    image_norm = image_features / image_features.norm(dim=1, keepdim=True).clamp_min(
        epsilon
    )
    sim_matrix = text_norm @ image_norm.t()

    t2i = _compute_hard_negative_v2_direction(
        sim_matrix=sim_matrix,
        query_pid=text_pid,
        gallery_pid=image_pid,
        margin=margin,
        topn=topn,
        temperature=temperature,
        loss_type=loss_type,
        false_neg_margin=false_neg_margin,
    )
    i2t = _compute_hard_negative_v2_direction(
        sim_matrix=sim_matrix.t(),
        query_pid=image_pid,
        gallery_pid=text_pid,
        margin=margin,
        topn=topn,
        temperature=temperature,
        loss_type=loss_type,
        false_neg_margin=false_neg_margin,
    )

    if direction == "t2i":
        loss = t2i["loss"]
        selected = [t2i]
    elif direction == "i2t":
        loss = i2t["loss"]
        selected = [i2t]
    else:
        loss = 0.5 * (t2i["loss"] + i2t["loss"])
        selected = [t2i, i2t]

    total_valid = sum(float(item["num_valid_queries"]) for item in selected)

    def _weighted_stat(key):
        if total_valid <= 0:
            return zero
        out = zero
        for item in selected:
            out = out + item[key] * float(item["num_valid_queries"])
        return out / total_valid

    return {
        "loss": loss,
        "avg_pos_sim": _weighted_stat("avg_pos_sim"),
        "avg_neg_sim": _weighted_stat("avg_neg_sim"),
        "avg_margin_gap": _weighted_stat("avg_margin_gap"),
        "violation_rate": _weighted_stat("violation_rate"),
        "avg_top1_neg_sim": _weighted_stat("avg_top1_neg_sim"),
        "avg_topk_neg_sim": _weighted_stat("avg_topk_neg_sim"),
        "num_valid_queries": total_valid,
        "t2i_loss": t2i["loss"],
        "i2t_loss": i2t["loss"],
        "t2i_valid": t2i["num_valid_queries"],
        "i2t_valid": i2t["num_valid_queries"],
    }


def _diag_gaussian_sigma_from_logvar(
    logvar,
    eps=1e-6,
    logvar_min=-6.0,
    logvar_max=4.0,
):
    """
    Convert log-variance to std for diagonal Gaussian with safety clamps.

    Args:
        logvar: [B, D] or [N, D], per-dimension log-variance.
        eps: scalar, minimum variance epsilon.
        logvar_min: scalar, lower clamp bound for logvar.
        logvar_max: scalar, upper clamp bound for logvar.
    Returns:
        sigma: same shape as logvar, where sigma = exp(0.5 * clamp(logvar)).
    """
    clamped_logvar = torch.clamp(logvar, min=logvar_min, max=logvar_max)
    sigma = torch.exp(0.5 * clamped_logvar)
    sigma = sigma.clamp_min(eps ** 0.5)
    return sigma


def compute_w2_diag_distance(
    mu1,
    logvar1,
    mu2,
    logvar2,
    pairwise=False,
    squared=True,
    eps=1e-6,
    logvar_min=-6.0,
    logvar_max=4.0,
):
    """
    Closed-form diagonal Gaussian W2 distance.

    For N(mu, diag(sigma^2)), diagonal Gaussian 2-Wasserstein (squared) is:
      W2^2 = ||mu1 - mu2||^2 + ||sigma1 - sigma2||^2
      sigma = exp(0.5 * logvar)

    Args:
        mu1: [B1, D], first mean tensor.
        logvar1: [B1, D], first log-variance tensor.
        mu2: [B2, D] for pairwise=True, else [B1, D].
        logvar2: [B2, D] for pairwise=True, else [B1, D].
        pairwise: bool, return full [B1, B2] distance matrix if True.
        squared: bool, return W2^2 if True, else return W2.
        eps: scalar epsilon for numeric stability.
        logvar_min/logvar_max: clamp range for logvar.
    Returns:
        dist:
          - [B1] for matched pairs (pairwise=False)
          - [B1, B2] for pairwise matrix (pairwise=True)
    """
    sigma1 = _diag_gaussian_sigma_from_logvar(
        logvar1, eps=eps, logvar_min=logvar_min, logvar_max=logvar_max
    )
    sigma2 = _diag_gaussian_sigma_from_logvar(
        logvar2, eps=eps, logvar_min=logvar_min, logvar_max=logvar_max
    )

    if pairwise:
        mu_term = torch.cdist(mu1, mu2, p=2).pow(2)
        sigma_term = torch.cdist(sigma1, sigma2, p=2).pow(2)
        dist_sq = mu_term + sigma_term
    else:
        dist_sq = (mu1 - mu2).pow(2).sum(dim=-1) + (sigma1 - sigma2).pow(2).sum(
            dim=-1
        )

    if squared:
        return dist_sq
    return torch.sqrt(dist_sq + eps)


def compute_w2_diag_matched_loss(
    mu1,
    logvar1,
    mu2,
    logvar2,
    reduction="mean",
    eps=1e-6,
    logvar_min=-6.0,
    logvar_max=4.0,
):
    """
    Matched-pair diagonal Gaussian W2^2 loss over a batch.

    Args:
        mu1, logvar1, mu2, logvar2: [B, D]
        reduction: one of {"mean", "sum", "none"}.
    Returns:
        Scalar loss for mean/sum or [B] vector for none.
    """
    dist_sq = compute_w2_diag_distance(
        mu1=mu1,
        logvar1=logvar1,
        mu2=mu2,
        logvar2=logvar2,
        pairwise=False,
        squared=True,
        eps=eps,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )

    if reduction == "mean":
        return dist_sq.mean()
    if reduction == "sum":
        return dist_sq.sum()
    if reduction == "none":
        return dist_sq
    raise ValueError(f"Unsupported reduction: {reduction}")


def compute_w2_diag_pairwise_matrix(
    mu1,
    logvar1,
    mu2,
    logvar2,
    eps=1e-6,
    logvar_min=-6.0,
    logvar_max=4.0,
):
    """
    Pairwise diagonal Gaussian W2^2 matrix.

    Args:
        mu1, logvar1: [B1, D]
        mu2, logvar2: [B2, D]
    Returns:
        dist_matrix: [B1, B2], W2^2 for each pair.
    """
    return compute_w2_diag_distance(
        mu1=mu1,
        logvar1=logvar1,
        mu2=mu2,
        logvar2=logvar2,
        pairwise=True,
        squared=True,
        eps=eps,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )


def _uspc_get_arg(args, name, default):
    if args is None:
        return default
    return getattr(args, name, default)


def _uspc_reference_tensor(*values):
    for value in values:
        if torch.is_tensor(value):
            return value
    return torch.zeros(())


def _uspc_zero_output(reference, confidence_value=1.0):
    zero = reference.new_zeros(())
    confidence = reference.new_full((), float(confidence_value))
    return {
        "loss_uspc": zero,
        "loss_uspc_raw": zero,
        "loss_uspc_color": zero,
        "loss_uspc_struct": zero,
        "loss_uspc_full": zero,
        "uspc_rgb_conf_mean": confidence,
        "uspc_thermal_conf_mean": confidence,
        "uspc_fused_conf_mean": confidence,
        "uspc_active": zero,
    }


def _uspc_require_feature(name, value):
    if not torch.is_tensor(value):
        raise KeyError(
            "USPC is active but required feature is missing or not a tensor: "
            f"{name}."
        )
    if value.dim() != 2:
        raise ValueError(
            f"USPC expects {name} to have shape [B, C], got {tuple(value.shape)}."
        )


def _uspc_clean_normalize(feat, eps=1e-12):
    feat = torch.nan_to_num(feat.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(feat, p=2, dim=1, eps=eps)


def _uspc_weighted_mean(loss_per_sample, sample_weight, eps=1e-6):
    if sample_weight is None:
        return loss_per_sample.mean()
    sample_weight = sample_weight.to(
        device=loss_per_sample.device,
        dtype=loss_per_sample.dtype,
    ).reshape(-1)
    if sample_weight.shape[0] != loss_per_sample.shape[0]:
        raise ValueError(
            "USPC sample weight must match batch size, got "
            f"{sample_weight.shape[0]} and {loss_per_sample.shape[0]}."
        )
    sample_weight = torch.nan_to_num(
        sample_weight, nan=1.0, posinf=1.0, neginf=1.0
    ).clamp_min(0.0)
    return (loss_per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(eps)


def _uspc_alignment_loss(
    image_feat,
    text_feat,
    pids=None,
    temperature=0.02,
    sample_weight=None,
    eps=1e-8,
):
    if image_feat.shape != text_feat.shape:
        raise ValueError(
            "USPC alignment expects paired image/text features with same shape, "
            f"got {tuple(image_feat.shape)} and {tuple(text_feat.shape)}."
        )
    if temperature <= 0:
        raise ValueError(f"USPC temperature must be > 0, got {temperature}.")

    image_feat = _uspc_clean_normalize(image_feat)
    text_feat = _uspc_clean_normalize(text_feat)
    batch_size = image_feat.shape[0]
    sim = image_feat @ text_feat.t()
    sim = sim / float(temperature)

    if pids is not None:
        pids = pids.reshape(-1).to(device=sim.device)
        if pids.shape[0] != batch_size:
            raise ValueError(
                "USPC pids must match batch size, got "
                f"{pids.shape[0]} and {batch_size}."
            )
        labels = pids.unsqueeze(1).eq(pids.unsqueeze(0)).float()
        labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(eps)
        per_i2t = -(labels * F.log_softmax(sim, dim=1)).sum(dim=1)
        per_t2i = -(labels.t() * F.log_softmax(sim.t(), dim=1)).sum(dim=1)
    else:
        target = torch.arange(batch_size, device=sim.device)
        per_i2t = F.cross_entropy(sim, target, reduction="none")
        per_t2i = F.cross_entropy(sim.t(), target, reduction="none")

    loss_i2t = _uspc_weighted_mean(per_i2t, sample_weight)
    # The first USPC version uses the same image-side confidence for the reverse
    # direction because batch rows are paired image/text samples.
    loss_t2i = _uspc_weighted_mean(per_t2i, sample_weight)
    return 0.5 * (loss_i2t + loss_t2i)


def _uspc_confidence_from_logvar(
    logvar,
    reference_feat,
    uncertainty_weighting=False,
    detach_weight=True,
    eps=1e-6,
):
    batch_size = reference_feat.shape[0]
    if (not uncertainty_weighting) or logvar is None:
        return reference_feat.new_ones(batch_size), reference_feat.new_ones(())
    if not torch.is_tensor(logvar):
        return reference_feat.new_ones(batch_size), reference_feat.new_ones(())
    if logvar.shape[0] != batch_size:
        raise ValueError(
            "USPC logvar batch size mismatch: "
            f"{logvar.shape[0]} vs {batch_size}."
        )

    logvar = torch.nan_to_num(logvar.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if logvar.dim() == 1:
        mean_logvar = logvar
    else:
        mean_logvar = logvar.reshape(batch_size, -1).mean(dim=1)
    mean_logvar = torch.clamp(mean_logvar, min=-30.0, max=30.0)
    conf = torch.exp(-mean_logvar)
    conf = torch.nan_to_num(conf, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)
    conf = conf / (conf.mean().detach() + eps)
    if detach_weight:
        conf = conf.detach()
    return conf, conf.detach().mean()


def compute_uspc_loss(
    rgb_feat,
    thermal_feat,
    fused_feat,
    full_text_feat,
    color_text_feat,
    struct_text_feat,
    pids=None,
    rgb_logvar=None,
    thermal_logvar=None,
    fused_logvar=None,
    args=None,
    epoch=0,
):
    """
    Compute USPC v1 feature-level semantic alignment loss.

    USPC v1 aligns:
      RGB feature     <-> color text view
      Thermal feature <-> structure/color-free text view
      Fused feature   <-> full text view

    This pure function is intentionally not connected to the training loop.
    """
    reference = _uspc_reference_tensor(
        rgb_feat,
        thermal_feat,
        fused_feat,
        full_text_feat,
        color_text_feat,
        struct_text_feat,
    )
    use_uspc = bool(_uspc_get_arg(args, "use_uspc", False))
    lambda_uspc = float(_uspc_get_arg(args, "lambda_uspc", 0.0))
    warmup_epochs = int(_uspc_get_arg(args, "uspc_warmup_epochs", 0))
    cur_epoch = int(epoch)
    active_until_epoch = int(_uspc_get_arg(args, "uspc_active_until_epoch", -1))
    if (
        (not use_uspc)
        or lambda_uspc <= 0.0
        or cur_epoch <= warmup_epochs
    ):
        return _uspc_zero_output(reference)
    if active_until_epoch > 0 and cur_epoch > active_until_epoch:
        return _uspc_zero_output(reference, confidence_value=0.0)

    version = str(_uspc_get_arg(args, "uspc_version", "v1")).lower()
    if version != "v1":
        raise ValueError(f"Unsupported USPC version: {version}. Expected 'v1'.")
    loss_type = str(_uspc_get_arg(args, "uspc_loss_type", "sdm")).lower()
    if loss_type != "sdm":
        raise NotImplementedError(
            "USPC loss_type='ce' is reserved but not implemented in v1."
        )

    required = {
        "rgb_feat": rgb_feat,
        "thermal_feat": thermal_feat,
        "fused_feat": fused_feat,
        "full_text_feat": full_text_feat,
        "color_text_feat": color_text_feat,
        "struct_text_feat": struct_text_feat,
    }
    for name, value in required.items():
        _uspc_require_feature(name, value)

    batch_size = rgb_feat.shape[0]
    feat_dim = rgb_feat.shape[1]
    for name, value in required.items():
        if value.shape[0] != batch_size or value.shape[1] != feat_dim:
            raise ValueError(
                "USPC feature shapes must all match [B, C]; "
                f"rgb_feat={tuple(rgb_feat.shape)}, {name}={tuple(value.shape)}."
            )

    temperature = float(_uspc_get_arg(args, "temperature", 0.02))
    uncertainty_weighting = bool(
        _uspc_get_arg(args, "uspc_uncertainty_weighting", False)
    )
    weight_detach = bool(_uspc_get_arg(args, "uspc_weight_detach", True))

    rgb_conf, rgb_conf_mean = _uspc_confidence_from_logvar(
        rgb_logvar,
        rgb_feat,
        uncertainty_weighting=uncertainty_weighting,
        detach_weight=weight_detach,
    )
    thermal_conf, thermal_conf_mean = _uspc_confidence_from_logvar(
        thermal_logvar,
        thermal_feat,
        uncertainty_weighting=uncertainty_weighting,
        detach_weight=weight_detach,
    )
    fused_conf, fused_conf_mean = _uspc_confidence_from_logvar(
        fused_logvar,
        fused_feat,
        uncertainty_weighting=uncertainty_weighting,
        detach_weight=weight_detach,
    )

    loss_color = _uspc_alignment_loss(
        rgb_feat,
        color_text_feat,
        pids=pids,
        temperature=temperature,
        sample_weight=rgb_conf,
    )
    loss_struct = _uspc_alignment_loss(
        thermal_feat,
        struct_text_feat,
        pids=pids,
        temperature=temperature,
        sample_weight=thermal_conf,
    )
    loss_full = _uspc_alignment_loss(
        fused_feat,
        full_text_feat,
        pids=pids,
        temperature=temperature,
        sample_weight=fused_conf,
    )

    color_weight = float(_uspc_get_arg(args, "uspc_color_weight", 1.0))
    struct_weight = float(_uspc_get_arg(args, "uspc_struct_weight", 1.0))
    full_weight = float(_uspc_get_arg(args, "uspc_full_weight", 1.0))
    loss_raw = (
        color_weight * loss_color
        + struct_weight * loss_struct
        + full_weight * loss_full
    )
    loss_uspc = lambda_uspc * loss_raw

    return {
        "loss_uspc": loss_uspc,
        "loss_uspc_raw": loss_raw,
        "loss_uspc_color": loss_color,
        "loss_uspc_struct": loss_struct,
        "loss_uspc_full": loss_full,
        "uspc_rgb_conf_mean": rgb_conf_mean,
        "uspc_thermal_conf_mean": thermal_conf_mean,
        "uspc_fused_conf_mean": fused_conf_mean,
        "uspc_active": loss_uspc.new_ones(()),
    }
