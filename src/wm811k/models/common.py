"""Shared model adaptation helpers."""

from __future__ import annotations

import torch
from torch import nn


def adapt_input_conv(layer: nn.Conv2d, input_channels: int, pretrained: bool) -> nn.Conv2d:
    if input_channels == layer.in_channels:
        return layer
    replacement = nn.Conv2d(
        input_channels,
        layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=layer.bias is not None,
        padding_mode=layer.padding_mode,
    )
    if pretrained:
        with torch.no_grad():
            average = layer.weight.mean(dim=1, keepdim=True)
            replacement.weight.copy_(average.repeat(1, input_channels, 1, 1))
            if layer.bias is not None:
                replacement.bias.copy_(layer.bias)
    return replacement
