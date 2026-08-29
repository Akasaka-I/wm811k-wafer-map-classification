"""Swin Transformer model adapters."""

from torch import nn
from torchvision.models import Swin_T_Weights, swin_t

from wm811k.models.common import adapt_input_conv


def build_swin_tiny(
    input_channels: int,
    num_classes: int,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    weights = Swin_T_Weights.DEFAULT if pretrained else None
    model = swin_t(weights=weights)
    model.features[0][0] = adapt_input_conv(model.features[0][0], input_channels, pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model
