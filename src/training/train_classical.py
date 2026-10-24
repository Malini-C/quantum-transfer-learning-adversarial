"""
Milestone 2 - Train the classical ResNet18 baseline on Ants vs Bees.

Outputs:
  results/classical_resnet18.pth      -> best model weights
  results/m2_classical_log.csv        -> per-epoch metrics
  results/m2_classical_curves.png     -> loss & accuracy curves

Run:
  python -m src.training.train_classical
"""

import sys
import csv
import time
from pathlib import Path

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
from src.models.classical import build_classical_resnet18


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += xb.size(0)

    return total_loss / total, correct / total


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    train_loader, val_loader, classes = get_dataloaders()
    print(f"[train] Classes: {classes}")
    print(f"[train] Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model = build_classical_resnet18(num_classes=len(classes), freeze_backbone=False).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Trainable parameters: {n_trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    epochs = 10
    log_rows = []
    best_val_acc = 0.0
    best_path = RESULTS_DIR / "classical_resnet18.pth"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        log_rows.append({"epoch": epoch,
                         "train_loss": tr_loss, "train_acc": tr_acc,
                         "val_loss":   va_loss, "val_acc":   va_acc})

        print(f"[epoch {epoch:02d}/{epochs}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model_state": model.state_dict(),
                        "classes": classes,
                        "val_acc": va_acc}, best_path)
            print(f"[train] Saved new best (val_acc={va_acc:.4f}) -> {best_path.name}")

    elapsed = time.time() - t0
    print(f"[train] Done in {elapsed/60:.1f} min. Best val_acc = {best_val_acc:.4f}")

    # Save CSV log
    csv_path = RESULTS_DIR / "m2_classical_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                               "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[train] Log saved -> {csv_path}")

    # Save curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_x = [r["epoch"] for r in log_rows]
    ax1.plot(epochs_x, [r["train_loss"] for r in log_rows], label="train")
    ax1.plot(epochs_x, [r["val_loss"]   for r in log_rows], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()

    ax2.plot(epochs_x, [r["train_acc"] for r in log_rows], label="train")
    ax2.plot(epochs_x, [r["val_acc"]   for r in log_rows], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.set_ylim(0, 1.05); ax2.legend()

    plt.tight_layout()
    curves_path = RESULTS_DIR / "m2_classical_curves.png"
    plt.savefig(curves_path, dpi=120)
    plt.close()
    print(f"[train] Curves saved -> {curves_path}")


if __name__ == "__main__":
    main()