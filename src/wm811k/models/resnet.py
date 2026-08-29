"""ResNet model adapters."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from wm811k.models.common import adapt_input_conv


def build_resnet18(
    input_channels: int,
    num_classes: int,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.conv1 = adapt_input_conv(model.conv1, input_channels, pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
