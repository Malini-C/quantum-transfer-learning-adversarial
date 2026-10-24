"""
Interactive demo: pick a validation image, show predictions of all 4 models
on clean and adversarial versions of the image.

Run:
  python -m src.demo
"""

import sys, random, joblib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, IMAGENET_MEAN, IMAGENET_STD, SEED
from src.data.dataset import get_dataloaders
from src.models.classical import build_classical_resnet18
from src.models.qtl import QTLModel
from src.attacks.fgsm import fgsm_attack


def unnormalize(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD ).view(3,1,1)
    return (t*std + mean).clamp(0,1).permute(1,2,0).cpu().numpy()


def load_all_models(device):
    # Classical
    classical = build_classical_resnet18(num_classes=2, freeze_backbone=False).to(device)
    ckpt = torch.load(RESULTS_DIR / "classical_resnet18.pth", map_location=device)
    classical.load_state_dict(ckpt["model_state"]); classical.eval()

    classical_adv = build_classical_resnet18(num_classes=2, freeze_backbone=False).to(device)
    ckpt = torch.load(RESULTS_DIR / "classical_resnet18_robust.pth", map_location=device)
    classical_adv.load_state_dict(ckpt["model_state"]); classical_adv.eval()

    # QTL
    pipe = joblib.load(RESULTS_DIR / "qtl_pipeline.joblib")
    qtl = QTLModel(pipe, device).to(device)
    ckpt = torch.load(RESULTS_DIR / "qtl_head.pth", map_location=device)
    qtl.head.q_layer.load_state_dict(ckpt["head_state"]); qtl.eval()

    qtl_adv = QTLModel(pipe, device).to(device)
    ckpt = torch.load(RESULTS_DIR / "qtl_head_robust.pth", map_location=device)
    qtl_adv.head.q_layer.load_state_dict(ckpt["head_state"]); qtl_adv.eval()

    return {"Classical": classical, "Classical+Adv": classical_adv,
            "QTL": qtl, "QTL+Adv": qtl_adv}


def predict(model, xb, device):
    with torch.no_grad():
        return model(xb.to(device)).argmax(1).cpu().numpy()


def make_adv(model, xb, yb, eps, device):
    return fgsm_attack(model, xb.to(device), yb.to(device),
                       eps=eps, criterion=nn.CrossEntropyLoss()).cpu()


def main():
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    device = torch.device("cpu")

    _, val_loader, classes = get_dataloaders()
    models = load_all_models(device)

    # grab one batch
    xb, yb = next(iter(val_loader))
    xb, yb = xb[:4], yb[:4]

    # --- figure 1: 4 images × 2 (clean / adv) × 4 models ---
    # Use QTL as the "attack target" - one common adversarial example
    eps = 4 / 255
    xb_adv = make_adv(models["QTL"], xb, yb, eps, device)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        axes[0, i].imshow(unnormalize(xb[i]))
        axes[0, i].set_title(f"Clean | true = {classes[yb[i].item()]}")
        axes[0, i].axis("off")
        axes[1, i].imshow(unnormalize(xb_adv[i]))
        axes[1, i].set_title(f"Adversarial (eps=4/255)")
        axes[1, i].axis("off")

    # overlay model predictions in the title of the clean row
    for name, model in models.items():
        pc = predict(model, xb, device)
        pa = predict(model, xb_adv, device)
        for i in range(4):
            axes[0, i].set_title(
                axes[0, i].get_title() +
                f"\n{name}: {classes[pc[i]]}", fontsize=9)

    plt.suptitle(f"Clean (top) vs FGSM-adversarial (bottom) images — eps=4/255")
    plt.tight_layout()
    out = RESULTS_DIR / "demo_predictions.png"
    plt.savefig(out, dpi=120); plt.close()
    print(f"Saved {out}")

    # --- console summary for each model ---
    print("\n=== Predictions per model ===")
    for name, model in models.items():
        pc = predict(model, xb, device)
        pa = predict(model, xb_adv, device)
        print(f"\n{name}:")
        for i in range(4):
            true = classes[yb[i].item()]
            print(f"  image {i}: true={true:5s}  clean→{classes[pc[i]]:5s}  "
                  f"adv→{classes[pa[i]]:5s}  {'✓' if pc[i]==yb[i].item() else '✗'}  "
                  f"{'✓' if pa[i]==yb[i].item() else '✗'}")


if __name__ == "__main__":
    main()