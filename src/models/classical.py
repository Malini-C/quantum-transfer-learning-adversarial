"""
Classical ResNet18 baseline for Ants vs Bees.

We use a pretrained ResNet18 and replace its final FC layer (originally 1000
classes) with a fresh linear layer for our 2-class problem. Two modes are
possible:

  - freeze_backbone=True : only the last FC layer is trained (fast, good when
                           data is tiny).
  - freeze_backbone=False: the whole network is fine-tuned (slower, usually
                           better accuracy on Ants vs Bees).

For our baseline we FINE-TUNE the whole network, because Ants vs Bees is
small and fine-tuning typically reaches ~93-95% validation accuracy.
"""

import torch.nn as nn
from torchvision import models


def build_classical_resnet18(num_classes: int = 2,
                             freeze_backbone: bool = False) -> nn.Module:
    """
    Build a ResNet18 classifier with a fresh final layer.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)   # always trainable
    return model