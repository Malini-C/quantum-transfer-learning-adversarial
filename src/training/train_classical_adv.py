"""
Milestone 7 supporting script:
Classical ResNet18 (fine-tuned) + FGSM adversarial training at eps=2/255,
matching the setting used for the quantum robust model so the comparison is fair.

Outputs:
  results/classical_resnet18_robust.pth
  results/m7_classical_adv_log.csv

Run:
  python -m src.training.train_classical_adv
"""

import sys, csv, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, SEED
from src.data.dataset import get_dataloaders
from src.models.classical import build_classical_resnet18
from src.attacks.fgsm import fgsm_attack


EPS_TRAIN    = 2.0 / 255.0
ADV_FRACTION = 0.5
EPOCHS       = 10
LR           = 1e-4
BATCH_SIZE   = 16


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses, correct, seen = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        bsz = xb.size(0)
        n_adv = int(round(bsz * ADV_FRACTION))

        if n_adv > 0:
            xb_clean, yb_clean = xb[:bsz-n_adv], yb[:bsz-n_adv]
            xb_a, yb_a         = xb[bsz-n_adv:], yb[bsz-n_adv:]
            xb_adv = fgsm_attack(model, xb_a, yb_a, eps=EPS_TRAIN, criterion=criterion)
            xb_final = torch.cat([xb_clean, xb_adv], dim=0)
            yb_final = torch.cat([yb_clean, yb_a],     dim=0)
        else:
            xb_final, yb_final = xb, yb

        logits = model(xb_final)
        loss = criterion(logits, yb_final)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses  += loss.item() * xb_final.size(0)
        correct += (logits.argmax(1) == yb_final).sum().item()
        seen    += xb_final.size(0)
    return losses / seen, correct / seen


@torch.no_grad()
def eval_clean(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        losses  += criterion(logits, yb).item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total   += xb.size(0)
    return losses / total, correct / total


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cladv] Device: {device}, eps_train={EPS_TRAIN:.4f}, adv_fraction={ADV_FRACTION}")

    train_loader, val_loader, classes = get_dataloaders()
    model = build_classical_resnet18(num_classes=len(classes), freeze_backbone=False).to(device)
    print(f"[cladv] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    log_rows = []
    best_path = RESULTS_DIR / "classical_resnet18_robust.pth"
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_clean(model, val_loader, device)
        log_rows.append({"epoch": epoch,
                         "train_loss": tr_loss, "train_acc": tr_acc,
                         "val_loss":   va_loss, "val_acc":   va_acc})
        print(f"[cladv epoch {epoch:02d}/{EPOCHS}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model_state": model.state_dict(),
                        "classes": classes, "val_acc": va_acc,
                        "eps_train": EPS_TRAIN, "adv_fraction": ADV_FRACTION},
                       best_path)
            print(f"[cladv]   New best val_acc={va_acc:.4f}")

    print(f"[cladv] Done in {time.time()-t0:.1f}s. Best val_acc = {best_val:.4f}")

    csv_path = RESULTS_DIR / "m7_classical_adv_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","train_acc","val_loss","val_acc"])
        w.writeheader(); w.writerows(log_rows)
    print(f"[cladv] Log saved -> {csv_path.name}")


if __name__ == "__main__":
    main()