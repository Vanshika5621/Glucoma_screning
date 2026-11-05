from __future__ import annotations

from typing import TYPE_CHECKING, Literal

Architecture = Literal["resnet50", "efficientnet_b0"]

if TYPE_CHECKING:
    from torch.nn import Module


def build_model(
    architecture: Architecture,
    pretrained: bool = False,
    num_classes: int = 2,
) -> Module:
    from torch import nn
    from torchvision import models

    if architecture == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes,
        )
        return model
    raise ValueError(f"Unsupported model architecture: {architecture}")
