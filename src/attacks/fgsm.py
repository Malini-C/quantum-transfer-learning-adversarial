"""
FGSM (Fast Gradient Sign Method) adversarial attack.

Given a batch (x, y):
  1. Compute loss = CrossEntropy(model(x), y)
  2. Compute grad = d(loss) / d(x)      [requires_grad on x]
  3. x_adv = x + eps * sign(grad)
  4. Clamp x_adv to the valid normalised-image range

Note: The model must be in eval() mode for reproducible attacks.
"""

import torch
import torch.nn as nn

# Valid range after ImageNet normalisation
#   min = (0 - mean) / std ; max = (1 - mean) / std
# We compute this from the standard ImageNet stats used in config.py.
from config import IMAGENET_MEAN, IMAGENET_STD

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

X_MIN = ((0.0 - _MEAN) / _STD)      # per-channel minimum
X_MAX = ((1.0 - _MEAN) / _STD)      # per-channel maximum


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    criterion: nn.Module = None,
) -> torch.Tensor:
    """
    Produce adversarial examples for a batch.

    Args:
        model: the classifier (in eval mode)
        x:     clean input batch, shape (B, 3, H, W), already normalised
        y:     true labels, shape (B,)
        eps:   perturbation budget (in normalised pixel units)

    Returns:
        x_adv: adversarial batch, same shape as x
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)

    logits = model(x_adv)
    loss = criterion(logits, y)
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        perturb = eps * x_adv.grad.sign()
        x_adv = x + perturb
        # Clamp per channel to valid range
        x_min = X_MIN.to(x.device)
        x_max = X_MAX.to(x.device)
        x_adv = torch.max(torch.min(x_adv, x_max), x_min)

    return x_adv.detach()