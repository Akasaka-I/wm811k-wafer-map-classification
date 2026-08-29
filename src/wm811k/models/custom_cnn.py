"""Small, interpretable CNN baseline."""

from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
        )


class CustomCNN(nn.Module):
    def __init__(self, input_channels: int = 2, num_classes: int = 9, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(input_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


def build_custom_cnn(input_channels: int, num_classes: int, **kwargs) -> nn.Module:
    return CustomCNN(
        input_channels=input_channels,
        num_classes=num_classes,
        dropout=float(kwargs.get("dropout", 0.25)),
    )
