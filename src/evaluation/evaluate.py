"""
Evaluation harness for QTL model.

Evaluates:
  - clean validation accuracy
  - adversarial accuracy under FGSM for several eps values

Loads:
  - results/qtl_pipeline.joblib   (StandardScaler + PCA stats, fitted)
  - results/qtl_head.pth          (VQC weights)

Uses QTLModel (fully differentiable) so FGSM can backprop to the image.

Run (from project root):
  python -m src.evaluation.evaluate
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, SEED
from src.data.dataset import get_dataloaders
from src.models.qtl import QTLModel
from src.attacks.fgsm import fgsm_attack


# ---------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------
def evaluate_full_model(model, loader, device, eps=None):
    """
    Evaluate a model on a full DataLoader.
    If eps is provided, evaluate on FGSM-perturbed data.
    Returns (accuracy, mean_loss).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if eps is not None:
            xb = fgsm_attack(model, xb, yb, eps=eps, criterion=criterion)
        with torch.no_grad():
            logits = model(xb)
        loss = criterion(logits, yb)
        loss_sum += loss.item() * xb.size(0)
        correct  += (logits.argmax(1) == yb).sum().item()
        total    += xb.size(0)
    return correct / total, loss_sum / total


# ---------------------------------------------------------------
# Load the trained QTL model
# ---------------------------------------------------------------
def load_qtl_model(device):
    pipe = joblib.load(RESULTS_DIR / "qtl_pipeline.joblib")
    model = QTLModel(pipe, device).to(device)
    ckpt = torch.load(RESULTS_DIR / "qtl_head.pth", map_location=device)
    model.head.q_layer.load_state_dict(ckpt["head_state"])
    model.eval()
    return model


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    train_loader, val_loader, classes = get_dataloaders()
    model = load_qtl_model(device)
    print(f"[eval] Loaded QTL model and pipeline")

    # ---- 1. Clean evaluation ----
    clean_acc, clean_loss = evaluate_full_model(model, val_loader, device, eps=None)
    print(f"[eval] Clean val accuracy: {clean_acc:.4f}   (loss: {clean_loss:.4f})")

    # ---- 2. FGSM sweep ----
    eps_values = [1/255, 2/255, 4/255, 8/255]
    print(f"[eval] FGSM sweep over eps = {[f'{e:.4f}' for e in eps_values]}")

    results = [{"eps": 0.0, "accuracy": clean_acc, "loss": clean_loss}]
    for eps in eps_values:
        acc, loss = evaluate_full_model(model, val_loader, device, eps=eps)
        results.append({"eps": eps, "accuracy": acc, "loss": loss})
        print(f"[eval]   eps={eps:.4f}  acc={acc:.4f}  loss={loss:.4f}  "
              f"(drop vs clean: {clean_acc - acc:+.4f})")

    # Save CSV for later use in comparison table
    import csv
    csv_path = RESULTS_DIR / "m5_fgsm_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eps", "accuracy", "loss"])
        w.writeheader(); w.writerows(results)
    print(f"[eval] Saved FGSM sweep -> {csv_path.name}")

    # ---- 3. Save a plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    eps_x = [r["eps"] for r in results]
    ax.plot(eps_x, [r["accuracy"] for r in results], marker="o")
    ax.set_xlabel("FGSM epsilon (normalised pixel units)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("QTL under FGSM attack")
    ax.grid(True); ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "m5_fgsm_sweep.png", dpi=120)
    plt.close()
    print(f"[eval] Saved plot -> m5_fgsm_sweep.png")


if __name__ == "__main__":
    main()