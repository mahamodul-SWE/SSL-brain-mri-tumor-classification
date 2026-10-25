from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torchvision.models import resnet18


DEFAULT_CLASS_NAMES: list[str] = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor",
]

DISPLAY_NAMES: dict[str, str] = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "No Tumor",
}


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention used in the thesis notebook."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        channel_weights = self.excitation(self.pool(feature_map))
        return feature_map * channel_weights


class SEResNet18Encoder(nn.Module):
    """Stage-wise SE-ResNet18 encoder matching the training notebook."""

    def __init__(self, use_se: bool = True) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.use_se = use_se
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.se1 = SEBlock(64) if use_se else nn.Identity()
        self.se2 = SEBlock(128) if use_se else nn.Identity()
        self.se3 = SEBlock(256) if use_se else nn.Identity()
        self.se4 = SEBlock(512) if use_se else nn.Identity()

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512

    def forward_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = self.se1(self.layer1(features))
        features = self.se2(self.layer2(features))
        features = self.se3(self.layer3(features))
        features = self.se4(self.layer4(features))
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.forward_feature_map(images)
        return self.global_pool(feature_map).flatten(1)


class BrainTumorClassifier(nn.Module):
    """Four-class classifier matching the notebook checkpoint keys."""

    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.30,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SEResNet18Encoder(use_se=use_se)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(images))


def normalize_class_names(class_names: Sequence[str] | None) -> list[str]:
    """Return a safe class order, preserving the checkpoint order when valid."""
    if class_names is None:
        return DEFAULT_CLASS_NAMES.copy()

    normalized = [str(name).strip().lower().replace(" ", "") for name in class_names]
    expected = set(DEFAULT_CLASS_NAMES)
    if len(normalized) == 4 and set(normalized) == expected:
        return normalized
    return DEFAULT_CLASS_NAMES.copy()
