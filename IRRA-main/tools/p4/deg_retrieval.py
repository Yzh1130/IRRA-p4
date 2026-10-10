"""Shared P4 training/diagnostic degradation, loss and PID batch counters.

Only a sampler configured with sdm/ce owns an independent generator. Its seed
is set once on the input device, and its state advances across all batches.
"""
from collections import Counter
import torch

from model import objectives
from utils.mdrc import DEFAULT_DEGRADE_MODES, _apply_mode


def resolve_deg_rng_seed(train_seed, deg_rng_seed=None):
    return int(train_seed) * 1000 + 7919 if deg_rng_seed is None else int(deg_rng_seed)


class DegRetrievalSampler:
    def __init__(self, loss_name="sdm", seed=8919, branch_prob=1.0):
        if loss_name not in ("none", "sdm", "ce"):
            raise ValueError("loss_name must be none, sdm or ce")
        if not 0.0 <= float(branch_prob) <= 1.0:
            raise ValueError("branch_prob must be in [0, 1]")
        self.loss_name = loss_name
        self.seed = int(seed)
        self.branch_prob = float(branch_prob)
        self.generator = None

    def _get_generator(self, device):
        device = torch.device(device)
        if self.generator is None:
            self.generator = torch.Generator(device=device)
            self.generator.manual_seed(self.seed)
        elif torch.device(self.generator.device) != device:
            raise ValueError("a P4 sampler must stay on one tensor device")
        return self.generator

    def sample(self, rgb, ir, image_ids):
        if self.loss_name == "none":
            # This return neither creates a generator nor draws any random value.
            return rgb, ir, {"enabled": False, "valid_modes": DEFAULT_DEGRADE_MODES}
        if not torch.is_tensor(ir) or rgb.shape[0] != ir.shape[0]:
            raise ValueError("P4 degraded retrieval requires paired RGB/IR tensors")
        image_ids = torch.as_tensor(image_ids, device=rgb.device).reshape(-1)
        if image_ids.numel() != rgb.shape[0] or ir.device != rgb.device:
            raise ValueError("image_ids and paired modalities must align on device")
        generator = self._get_generator(rgb.device)
        # A branch decision is made once per batch, even at probability 1.
        enabled = bool(torch.rand((), device=rgb.device, generator=generator).lt(
            self.branch_prob).item())
        if not enabled:
            return rgb, ir, {"enabled": False, "valid_modes": DEFAULT_DEGRADE_MODES}
        unique_ids, inverse = torch.unique(image_ids, sorted=True, return_inverse=True)
        choices = torch.randint(len(DEFAULT_DEGRADE_MODES), (unique_ids.numel(),),
                                device=rgb.device, generator=generator)
        unique_strength = torch.full((unique_ids.numel(),), 0.1, device=rgb.device)
        drop = choices.lt(2)
        attenuation = choices.ge(4)
        unique_strength[drop] = 0.0
        n_attenuation = int(attenuation.sum().item())
        if n_attenuation:
            unique_strength[attenuation] = torch.empty(n_attenuation, device=rgb.device).uniform_(
                0.2, 0.6, generator=generator)
        modes = choices[inverse]
        strengths = unique_strength[inverse]
        rgb_mask = (modes % 2).eq(0)
        ir_mask = ~rgb_mask
        degraded_rgb, degraded_ir = rgb.clone(), ir.clone()
        for index, mode in enumerate(DEFAULT_DEGRADE_MODES):
            selected = modes.eq(index).nonzero(as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            target = degraded_rgb if mode.startswith("rgb_") else degraded_ir
            # Noise samples use the same advancing generator, never an image seed.
            changed = _apply_mode(target.index_select(0, selected),
                                  torch.ones(selected.numel(), device=rgb.device,
                                             dtype=torch.bool),
                                  mode, generator=generator,
                                  factors=strengths.index_select(0, selected))
            target.index_copy_(0, selected, changed)
        meta = {
            "enabled": True, "valid_modes": DEFAULT_DEGRADE_MODES,
            "mode_indices": modes, "strengths": strengths,
            "rgb_mask": rgb_mask, "ir_mask": ir_mask,
            "unique_image_ids": unique_ids, "unique_mode_indices": choices,
            "unique_strengths": unique_strength,
            "mode_counts": {mode: int(modes.eq(i).sum().item())
                            for i, mode in enumerate(DEFAULT_DEGRADE_MODES)},
        }
        return degraded_rgb, degraded_ir, meta


def compute_deg_loss(image_feats, text_feats, pids, logit_scale, loss_name,
                     eps=1e-8, sample_weight=None):
    if loss_name == "sdm":
        # Same PID-only call as the clean branch: no image_id/factor override.
        return objectives.compute_sdm(image_feats, text_feats, pids, logit_scale,
                                      sample_weight=sample_weight)
    if loss_name == "ce":
        if sample_weight is not None:
            raise ValueError("P4 CE requires A's uniform PID targets (no synthetic weights)")
        return objectives.compute_multipos_ce(image_feats, text_feats, pids,
                                              logit_scale, eps)
    if loss_name == "none":
        return image_feats.new_zeros(())
    raise ValueError("loss_name must be none, sdm or ce")


def pid_targets(pids):
    pids = pids.reshape(-1, 1)
    labels = (pids - pids.t() == 0).float()
    return labels / labels.sum(dim=1)


def _label_rows(pids, image_ids):
    pids = torch.as_tensor(pids).detach().cpu().reshape(-1).tolist()
    image_ids = torch.as_tensor(image_ids).detach().cpu().reshape(-1).tolist()
    if len(pids) != len(image_ids):
        raise ValueError("pids and image_ids must have equal length")
    rows = {"positive_columns": [], "positive_unique_image_ids": [],
            "duplicate_positive_columns": [], "negative_columns": []}
    for pid in pids:
        positives = [i for i, candidate in enumerate(pids) if candidate == pid]
        n_unique = len({image_ids[i] for i in positives})
        rows["positive_columns"].append(len(positives))
        rows["positive_unique_image_ids"].append(n_unique)
        rows["duplicate_positive_columns"].append(len(positives) - n_unique)
        rows["negative_columns"].append(len(pids) - len(positives))
    return rows


def _summarize_rows(rows):
    return {key: {"mean": sum(values) / len(values) if values else 0.0,
                  "histogram": {str(k): v for k, v in sorted(Counter(values).items())}}
            for key, values in rows.items()}


def batch_label_statistics(pids, image_ids):
    # The gallery contains B image *columns*, including repeated image IDs.
    # i2t has one row per image column and B text columns, so the same per-PID
    # counts apply, while distinct image IDs refer to those text source images.
    rows = _label_rows(pids, image_ids)
    return {direction: _summarize_rows(rows) for direction in ("t2i", "i2t")}


class EpochCounters:
    def __init__(self):
        self.text_samples = 0
        self.image_ids = set()
        self.degraded_copies = 0
        self.calls = {"rgb_clean": 0, "ir_clean": 0, "degraded": 0,
                      "rgb_degraded": 0, "ir_degraded": 0}
        self.rows = {key: [] for key in _label_rows([], []).keys()}
        self.mode_counts = Counter()

    def add_batch(self, pids, image_ids, meta=None, encoder_calls=None,
                  clean_encoder_calls=None):
        ids = torch.as_tensor(image_ids).detach().cpu().reshape(-1).tolist()
        self.text_samples += len(ids)
        self.image_ids.update(ids)
        rows = _label_rows(pids, image_ids)
        for key in rows:
            self.rows[key].extend(rows[key])
        clean_calls = clean_encoder_calls if clean_encoder_calls is not None else {"rgb": 1, "ir": 1}
        self.calls["rgb_clean"] += int(clean_calls.get("rgb", 0))
        self.calls["ir_clean"] += int(clean_calls.get("ir", 0))
        if meta and meta.get("enabled"):
            self.degraded_copies += len(ids)
            self.mode_counts.update(meta["mode_counts"])
            for modality in ("rgb", "ir"):
                count = int((encoder_calls or {}).get(modality, 0))
                self.calls[modality + "_degraded"] += count
                self.calls["degraded"] += count

    def summary(self, epoch, seconds, peak_memory_bytes, partial=False):
        labels = _summarize_rows(self.rows)
        return {"epoch": int(epoch), "partial": bool(partial),
                "text_samples": self.text_samples,
                "unique_images": len(self.image_ids),
                "degraded_copies": self.degraded_copies,
                "image_encoder_calls": dict(self.calls),
                "label_statistics": {"t2i": labels, "i2t": labels},
                "degradation_mode_counts": dict(self.mode_counts),
                "seconds": float(seconds), "peak_memory_bytes": int(peak_memory_bytes)}
