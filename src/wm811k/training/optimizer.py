"""Optimizer factory."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn
from torch.optim import AdamW, Optimizer, SGD


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    name = str(config.get("name", "adamw")).lower()
    learning_rate = float(config.get("learning_rate", 3e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    if name == "adamw":
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}")
