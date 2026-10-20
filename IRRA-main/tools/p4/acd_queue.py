"""Detached FIFO negatives used only by the QueueCE training variant."""
from __future__ import annotations

import torch


class NegativeFeatureQueue:
    """Store prior normalized student embeddings without retaining their graph.

    ``enqueue`` expects the L2-normalized embeddings returned by the training
    encoder. It copies their FP32 values directly, without normalizing a second
    time and introducing another round of floating-point rounding.
    """

    def __init__(self, capacity):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("queue capacity must be an integer >= 1")
        self.capacity = capacity
        self.features = None
        self.pids = None

    def __len__(self):
        return 0 if self.features is None else self.features.shape[0]

    @staticmethod
    def _validate_rows(features, pids):
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape[1] < 1 or not features.is_floating_point()):
            raise ValueError("features must be a floating-point [rows, dimensions] tensor")
        if (not isinstance(pids, torch.Tensor) or pids.ndim != 1
                or pids.numel() != features.shape[0]):
            raise ValueError("Expected one PID per feature row")

    @torch.no_grad()
    def enqueue(self, features, pids):
        """Append this step's features after its update; keep the newest entries."""
        self._validate_rows(features, pids)
        if self.features is not None and features.shape[1] != self.features.shape[1]:
            raise ValueError("Queue feature dimension cannot change")
        if features.shape[0] == 0:
            return
        incoming = features.detach().to(dtype=torch.float32)
        labels = pids.detach().to(device=incoming.device, dtype=torch.long)
        if self.features is not None:
            incoming = torch.cat((self.features.to(incoming.device), incoming), dim=0)
            labels = torch.cat((self.pids.to(incoming.device), labels), dim=0)
        # Cloning also prevents a short retained slice from keeping a large
        # incoming batch's storage alive or aliasing mutable caller tensors.
        self.features = incoming[-self.capacity:].clone()
        self.pids = labels[-self.capacity:].clone()

    @torch.no_grad()
    def logits(self, text, pids):
        """Return detached 50-scale negatives, excluding every matching PID."""
        self._validate_rows(text, pids)
        if self.features is None:
            return text.new_empty((text.shape[0], 0))
        if text.shape[1] != self.features.shape[1]:
            raise ValueError("Text and queue feature dimensions must match")
        features = self.features.to(device=text.device, dtype=text.dtype)
        labels = pids.detach().to(device=text.device, dtype=torch.long)
        queue_labels = self.pids.to(text.device)
        values = 50.0 * (text.detach() @ features.t())
        return values.masked_fill(labels[:, None].eq(queue_labels[None, :]), -torch.inf)
