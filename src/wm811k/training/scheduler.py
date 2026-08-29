"""Learning-rate scheduler factory."""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
    StepLR,
)


def build_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any],
    epochs: int,
) -> Optional[LRScheduler]:
    name = str(config.get("name", "none")).lower()
    if name in {"none", "off"}:
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "warmup_cosine":
        warmup_epochs = int(config.get("warmup_epochs", 5))
        if not 0 < warmup_epochs < epochs:
            raise ValueError("warmup_epochs must be between 1 and epochs - 1")
        warmup = LinearLR(
            optimizer,
            start_factor=float(config.get("start_factor", 0.1)),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - warmup_epochs, 1),
            eta_min=float(config.get("eta_min", 1e-6)),
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    if name == "step":
        return StepLR(
            optimizer,
            step_size=int(config.get("step_size", 10)),
            gamma=float(config.get("gamma", 0.1)),
        )
    raise ValueError(f"Unknown scheduler: {name}")
