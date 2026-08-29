"""Loss construction for imbalanced wafer-map classification."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        probability = torch.exp(-cross_entropy)
        return (((1.0 - probability) ** self.gamma) * cross_entropy).mean()


def balanced_class_weights(
    labels: np.ndarray,
    num_classes: int,
    power: float = 1.0,
    max_weight: Optional[float] = None,
) -> torch.Tensor:
    """Return inverse-frequency weights with optional smoothing and clipping.

    ``power=1`` is the standard balanced formula. ``power=0.5`` takes its
    square root, which preserves minority emphasis without creating extreme
    gradients for very rare classes.
    """
    if not 0.0 <= power <= 1.0:
        raise ValueError(f"power must be between 0 and 1, got {power}")
    if max_weight is not None and max_weight <= 0:
        raise ValueError("max_weight must be positive")
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(float)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts.sum() / (num_classes * counts[present])
    weights[present] = np.power(weights[present], power)
    if max_weight is not None:
        weights[present] = np.minimum(weights[present], max_weight)
    return torch.from_numpy(weights)


def build_loss(
    config: Dict[str, Any],
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    name = str(config.get("name", "cross_entropy")).lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "weighted_cross_entropy":
        if class_weights is None:
            raise ValueError("weighted_cross_entropy requires class weights")
        return nn.CrossEntropyLoss(weight=class_weights)
    if name == "focal":
        return FocalLoss(gamma=float(config.get("gamma", 2.0)), weight=class_weights)
    raise ValueError(f"Unknown loss: {name}")
