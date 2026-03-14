from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    # similarity: [num_query, num_gallery], larger means more similar.
    if get_mAP:
        # Full sort is needed when mAP/mINP are required.
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # For CMC only, top-k is faster than full sort.
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    # Predicted gallery pid for each query rank position.
    pred_labels = g_pids[indices.cpu()]  # q * k
    # Binary match matrix: True if predicted pid equals query pid.
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    # Compute CMC curve.
    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    # mINP considers the rank position of the last relevant match.
    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    # mAP is mean of per-query average precision.
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    def _encode_text_feature(self, model, caption):
        if hasattr(model, "encode_text_eval"):
            return model.encode_text_eval(caption)
        # Prefer explicit deterministic text encoder when provided by model.
        if hasattr(model, "encode_text_deterministic"):
            return model.encode_text_deterministic(caption)
        return model.encode_text(caption)

    def _encode_gallery_feature(self, model, img, thermal_img=None):
        if hasattr(model, "encode_rgbt_eval"):
            return model.encode_rgbt_eval(img, thermal_img)
        # Prefer explicit deterministic fused visual encoder when available.
        if thermal_img is not None and hasattr(model, "encode_rgbt_deterministic"):
            return model.encode_rgbt_deterministic(img, thermal_img)
        if thermal_img is None:
            return model.encode_image(img)
        # Fallback for models without dedicated RGBT helper.
        rgb_feat = model.encode_image(img)
        t_feat = model.encode_image(thermal_img)
        return rgb_feat + t_feat

    def _compute_embedding(self, model):
        # Encode all queries and gallery samples once, then evaluate by matrix ops.
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # Encode text queries.
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = self._encode_text_feature(model, caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # Encode gallery images.
        for data in self.img_loader:
            # Compatible with:
            #   (pid, img) for RGB-only gallery
            #   (pid, rgb_img, thermal_img) for RGB-T gallery
            if len(data) == 3:
                pid, img, thermal_img = data
                img = img.to(device)
                thermal_img = thermal_img.to(device)
            else:
                pid, img = data
                img = img.to(device)
                thermal_img = None
            with torch.no_grad():
                img_feat = self._encode_gallery_feature(model, img, thermal_img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False):

        # 1) Get embeddings.
        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        # 2) L2 normalize before cosine similarity.
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        # 3) Text-to-image similarity matrix.
        similarity = qfeats @ gfeats.t()

        # 4) Compute retrieval metrics.
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            # Optional image-to-text evaluation.
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]
