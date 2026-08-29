"""Configuration-driven model registry."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from torch import nn

from wm811k.models.convnext import build_convnext_tiny
from wm811k.models.custom_cnn import build_custom_cnn
from wm811k.models.resnet import build_resnet18
from wm811k.models.swin import build_swin_tiny


ModelBuilder = Callable[..., nn.Module]

_MODEL_REGISTRY: Dict[str, ModelBuilder] = {
    "custom_cnn": build_custom_cnn,
    "resnet18": build_resnet18,
    "convnext_tiny": build_convnext_tiny,
    "swin_tiny": build_swin_tiny,
}


def available_models() -> List[str]:
    return sorted(_MODEL_REGISTRY)


def build_model(config: Dict[str, Any]) -> nn.Module:
    name = str(config.get("name", "")).lower()
    if name not in _MODEL_REGISTRY:
        choices = ", ".join(available_models())
        raise ValueError(f"Unknown model '{name}'. Available models: {choices}")
    kwargs = dict(config)
    kwargs.pop("name", None)
    kwargs.setdefault("input_channels", 2)
    kwargs.setdefault("num_classes", 9)
    kwargs.setdefault("pretrained", False)
    return _MODEL_REGISTRY[name](**kwargs)
