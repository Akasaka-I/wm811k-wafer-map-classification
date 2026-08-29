"""ConvNeXt model adapters."""

from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from wm811k.models.common import adapt_input_conv


def build_convnext_tiny(
    input_channels: int,
    num_classes: int,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)
    model.features[0][0] = adapt_input_conv(model.features[0][0], input_channels, pretrained)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model
