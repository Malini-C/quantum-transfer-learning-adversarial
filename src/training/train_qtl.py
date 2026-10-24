"""
Milestone 4 (v2) - Train the QTL model with improved feature preprocessing.

Outputs:
  results/qtl_pipeline.joblib   -> fitted FeaturePipeline (scaler + PCA)
  results/qtl_head.pth          -> best VQC weights
  results/m4_qtl_log.csv
  results/m4_qtl_curves.png
"""

import sys, csv, time, joblib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, SEED
from src.data.dataset import get_dataloaders
from src.models.qtl import (
    build_frozen_resnet18_feature_extractor,
    extract_features,
    FeaturePipeline,
    QTLHead,
)
from src.models.quantum import N_LAYERS, N_QUBITS


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_epoch(head, X, y, optimizer, criterion, train: bool):
    if train:
        head.train()
        perm = torch.randperm(X.size(0)); X, y = X[perm], y[perm]
    else:
        head.eval()

    losses, correct, n = 0.0, 0, X.size(0)
    batch_size = 16
    with torch.set_grad_enabled(train):
        for i in range(0, n, batch_size):
            xb, yb = X[i:i+batch_size], y[i:i+batch_size]
            logits = head(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses  += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
    return losses / n, correct / n


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[qtl] Device: {device}")

    train_loader, val_loader, classes = get_dataloaders()
    print(f"[qtl] Classes: {classes}")

    print("[qtl] Extracting features ...")
    resnet = build_frozen_resnet18_feature_extractor().to(device)
    t0 = time.time()
    Xtr, ytr = extract_features(train_loader, resnet, device)
    Xva, yva = extract_features(val_loader,   resnet, device)
    print(f"[qtl] Features ready in {time.time()-t0:.1f}s")

    # ---- Fit FeaturePipeline on TRAIN ONLY ----
    fp = FeaturePipeline(n_components=4).fit(Xtr)
    evr = fp.explained_variance()
    print(f"[qtl] PCA (4 comps, whiten=True) explained variance per component: "
          f"{[f'{v:.3f}' for v in evr]}")
    print(f"[qtl] Cumulative variance kept: {evr.sum():.3f}")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32); yva_t = torch.tensor(yva, dtype=torch.long)

    head = QTLHead(fp, device).to(device)
    n_params = sum(p.numel() for p in head.q_layer.parameters() if p.requires_grad)
    print(f"[qtl] N_LAYERS={N_LAYERS}, N_QUBITS={N_QUBITS}, trainable params={n_params}")

    optimizer = Adam(head.q_layer.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    epochs = 25
    best_val_acc = 0.0
    best_head_path = RESULTS_DIR / "qtl_head.pth"
    log_rows = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(head, Xtr_t, ytr_t, optimizer, criterion, train=True)
        va_loss, va_acc = run_epoch(head, Xva_t, yva_t, optimizer, criterion, train=False)
        log_rows.append({"epoch": epoch,
                         "train_loss": tr_loss, "train_acc": tr_acc,
                         "val_loss":   va_loss, "val_acc":   va_acc})
        print(f"[epoch {epoch:02d}/{epochs}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"head_state": head.q_layer.state_dict(),
                        "val_acc": va_acc, "classes": classes}, best_head_path)
            print(f"[qtl]   New best val_acc={va_acc:.4f}")

    print(f"[qtl] Training done in {time.time()-t0:.1f}s. Best val_acc = {best_val_acc:.4f}")

    pipe_path = RESULTS_DIR / "qtl_pipeline.joblib"
    joblib.dump(fp, pipe_path)
    print(f"[qtl] Pipeline saved -> {pipe_path.name}")

    csv_path = RESULTS_DIR / "m4_qtl_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","train_acc","val_loss","val_acc"])
        w.writeheader(); w.writerows(log_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ex = [r["epoch"] for r in log_rows]
    ax1.plot(ex, [r["train_loss"] for r in log_rows], label="train")
    ax1.plot(ex, [r["val_loss"]   for r in log_rows], label="val")
    ax1.set_title("QTL Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(ex, [r["train_acc"] for r in log_rows], label="train")
    ax2.plot(ex, [r["val_acc"]   for r in log_rows], label="val")
    ax2.set_title("QTL Accuracy"); ax2.set_xlabel("epoch"); ax2.set_ylim(0, 1.05); ax2.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "m4_qtl_curves.png", dpi=120); plt.close()
    print(f"[qtl] Curves saved -> m4_qtl_curves.png")


if __name__ == "__main__":
    main()