import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
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

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


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


def compute_mlm(scores, labels):
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

    return {
        "loss": loss,
        "avg_pos_sim": pos_sim[valid_query_mask].mean(),
        "avg_neg_sim": hardneg_sim[valid_query_mask].mean(),
        "avg_margin_gap": margin_gap[valid_query_mask].mean(),
        "num_valid_queries": float(valid_query_mask.sum().item()),
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
