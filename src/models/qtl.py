"""
Quantum Transfer Learning model (fully differentiable).

Architecture:
  ResNet18 (frozen)
    -> StandardScaler (frozen, PyTorch)
    -> PCA(4, whiten=True)  (frozen, PyTorch-equivalent of sklearn)
    -> scale to [-pi/2, pi/2]
    -> VQC(4 qubits, 2 layers, 32 params, trainable)
    -> 2 logits

Why PyTorch PCA?
  sklearn PCA uses NumPy, which breaks the autograd graph. That makes image-
  space adversarial attacks (FGSM) impossible because x.grad stays None.
  We therefore fit sklearn once to get the statistics, then bake them into
  frozen PyTorch buffers so the whole pipeline is differentiable end-to-end.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import SEED
from src.models.quantum import build_quantum_classifier, N_INPUTS


# ---------------------------------------------------------------
# Frozen ResNet18
# ---------------------------------------------------------------
def build_frozen_resnet18_feature_extractor() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


@torch.no_grad()
def extract_features(loader, model, device):
    model.eval()
    feats_list, labels_list = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        f = model(xb)
        feats_list.append(f.cpu().numpy())
        labels_list.append(yb.numpy())
    return np.concatenate(feats_list), np.concatenate(labels_list)


# ---------------------------------------------------------------
# sklearn pipeline (FIT ONLY). We extract stats; PyTorch applies them.
# ---------------------------------------------------------------
class FeaturePipeline:
    def __init__(self, n_components: int = N_INPUTS):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, whiten=True, random_state=SEED)
        self._fitted = False

    def fit(self, X_train_512: np.ndarray):
        Xs = self.scaler.fit_transform(X_train_512)
        self.pca.fit(Xs)
        self._fitted = True
        return self

    def to_torch_stats(self, device):
        """
        Returns (scaler_mu, scaler_sigma, pca_mean, pca_components, pca_sqrt_var).
        Shapes:
          scaler_mu:        (512,)
          scaler_sigma:     (512,)
          pca_mean:         (512,)   <-- sklearn PCA mean lives in SCALED space
          pca_components:   (4, 512)
          pca_sqrt_var:     (4,)
        """
        assert self._fitted
        mu           = torch.tensor(self.scaler.mean_,         dtype=torch.float32, device=device)
        sigma        = torch.tensor(self.scaler.scale_,        dtype=torch.float32, device=device)
        pca_mean     = torch.tensor(self.pca.mean_,            dtype=torch.float32, device=device)
        components   = torch.tensor(self.pca.components_,      dtype=torch.float32, device=device)
        sqrt_var     = torch.tensor(np.sqrt(self.pca.explained_variance_),
                                    dtype=torch.float32, device=device)
        return mu, sigma, pca_mean, components, sqrt_var

    def explained_variance(self):
        return self.pca.explained_variance_ratio_


# ---------------------------------------------------------------
# Differentiable classical -> quantum head
# ---------------------------------------------------------------
class QTLHead(nn.Module):
    """
    Frozen StandardScaler + frozen PCA(whiten) + trainable VQC.
    Fully differentiable.
    """

    def __init__(self, feature_pipeline: FeaturePipeline, device):
        super().__init__()
        self.q_layer = build_quantum_classifier()

        mu, sigma, pca_mean, components, sqrt_var = feature_pipeline.to_torch_stats(device)
        self.register_buffer("scaler_mu",       mu)               # (512,)
        self.register_buffer("scaler_sigma",    sigma)            # (512,)
        self.register_buffer("pca_mean",        pca_mean)         # (512,)
        self.register_buffer("pca_components",  components)       # (4, 512)
        self.register_buffer("pca_sqrt_var",    sqrt_var)         # (4,)
        self.scale_to_pi_half = float(np.pi / 2.0)

    def forward(self, feats_512: torch.Tensor) -> torch.Tensor:
        # 1. StandardScaler  (differentiable)
        x = (feats_512 - self.scaler_mu) / self.scaler_sigma             # (B, 512)

        # 2. PCA projection (differentiable)
        x_centered = x - self.pca_mean                                    # (B, 512)
        z = x_centered @ self.pca_components.T                            # (B, 4)

        # 3. Whitening: divide by sqrt(explained_variance)
        z = z / self.pca_sqrt_var                                         # (B, 4)

        # 4. Clip and scale to [-pi/2, pi/2]  (differentiable where not clipped)
        z = torch.clamp(z, -2.0, 2.0) / 2.0                               # [-1, 1]
        z = z * self.scale_to_pi_half                                     # [-pi/2, pi/2]

        # 5. VQC: per-sample forward
        logits = torch.stack([self.q_layer(z[i]) for i in range(z.size(0))])
        return logits


# ---------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------
class QTLModel(nn.Module):
    def __init__(self, feature_pipeline: FeaturePipeline, device):
        super().__init__()
        self.resnet = build_frozen_resnet18_feature_extractor()
        self.head = QTLHead(feature_pipeline, device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.resnet(images)
        return self.head(feats)