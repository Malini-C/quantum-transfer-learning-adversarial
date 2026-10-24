"""
Milestone 6 - Adversarial training of the QTL model.

For each training batch:
  1. Split batch into "clean" and "adv" halves (50/50 by default).
  2. Generate FGSM adversarial examples for the "adv" half using eps_train.
  3. Forward both halves through the model, concatenate logits, compute loss.
  4. Backprop and step.

Outputs:
  results/qtl_head_robust.pth
  results/m6_qtl_robust_log.csv
  results/m6_qtl_robust_curves.png

Run:
  python -m src.training.adversarial_training
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
from src.attacks.fgsm import fgsm_attack


# ---------------- Hyperparameters (documented) ----------------
EPS_TRAIN       = 2.0 / 255.0      # FGSM eps used DURING training
ADV_FRACTION    = 0.5              # fraction of each batch replaced by adv
EPOCHS          = 25
LR              = 0.1
BATCH_SIZE      = 16
# --------------------------------------------------------------


def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_epoch_adv(head, resnet, X, y, optimizer, criterion, device, train: bool):
    """
    Adversarial training epoch.

    For each batch of RAW IMAGES we:
      1. Take the clean image batch.
      2. Split into clean_half and adv_half.
      3. Generate FGSM adversarial examples from adv_half (using a temporary
         model that wraps resnet+head so fgsm_attack can backprop to images).
      4. Concatenate (clean_half, adv_half) images, forward through the full
         pipeline, compute loss, step.

    Because FGSM requires an end-to-end model (image -> logits), we build a
    small nn.Module wrapper inside this function.
    """
    class _Wrap(nn.Module):
        def __init__(self, resnet, head):
            super().__init__()
            self.resnet = resnet
            self.head = head
        def forward(self, x):
            return self.head(self.resnet(x))

    full_model = _Wrap(resnet, head)
    if train:
        head.train()
        perm = torch.randperm(X.size(0))
        X, y = X[perm], y[perm]
    else:
        head.eval()

    losses, correct, n = 0.0, 0, X.size(0)

    with torch.set_grad_enabled(train):
        for i in range(0, n, BATCH_SIZE):
            xb, yb = X[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE]
            bsz = xb.size(0)
            n_adv = int(round(bsz * ADV_FRACTION)) if train else 0

            if train and n_adv > 0:
                xb_clean = xb[:bsz-n_adv]
                yb_clean = yb[:bsz-n_adv]
                xb_to_attack = xb[bsz-n_adv:]
                yb_to_attack = yb[bsz-n_adv:]
                xb_adv = fgsm_attack(full_model, xb_to_attack, yb_to_attack,
                                     eps=EPS_TRAIN, criterion=criterion)
                xb_final = torch.cat([xb_clean, xb_adv], dim=0)
                yb_final = torch.cat([yb_clean, yb_to_attack], dim=0)
            else:
                xb_final = xb
                yb_final = yb

            logits = full_model(xb_final)
            loss = criterion(logits, yb_final)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses  += loss.item() * xb_final.size(0)
            correct += (logits.argmax(1) == yb_final).sum().item()
            n       += 0  # n was set to X.size(0); correct with xb_final size below

        # Recompute using averaged size for safety
        total_bs = n if n > 0 else 1
    # accuracy/loss computed against total clean+adv examples seen
    n_seen = 0
    return losses / X.size(0), correct / X.size(0)


def evaluate_clean_and_fgsm(head, resnet, Xtr, ytr, Xva, yva, device):
    """Return (clean_acc, fgsm_acc@2/255)."""
    class _Wrap(nn.Module):
        def __init__(self, resnet, head):
            super().__init__()
            self.resnet = resnet
            self.head = head
        def forward(self, x):
            return self.head(self.resnet(x))

    full_model = _Wrap(resnet, head).to(device)
    full_model.eval()
    criterion = nn.CrossEntropyLoss()

    def eval_loader(X, y, eps=None):
        total, correct = 0, 0
        Xt = torch.tensor(X, dtype=torch.float32)  # features, not images!
        yt = torch.tensor(y, dtype=torch.long)
        # We need IMAGES for FGSM, but we only cached FEATURES.
        # For validation of FGSM on cached features we cannot apply image attack.
        # => For M6 we evaluate robustness with the FEATURE-cached val set for
        #    CLEAN acc, and rely on M7 for full image-space robustness eval.
        raise RuntimeError("Use evaluate_full_model from src.evaluation.evaluate instead")

    # We fall back to clean evaluation using cached features only.
    total, correct = 0, 0
    with torch.no_grad():
        for i in range(0, Xva.shape[0], BATCH_SIZE):
            xb = torch.tensor(Xva[i:i+BATCH_SIZE], dtype=torch.float32).to(device)
            yb = torch.tensor(yva[i:i+BATCH_SIZE], dtype=torch.long).to(device)
            logits = head(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total   += yb.size(0)
    return correct / total, None


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[adv] Device: {device}")
    print(f"[adv] eps_train={EPS_TRAIN:.4f}, adv_fraction={ADV_FRACTION}, "
          f"epochs={EPOCHS}, lr={LR}")

    # --- 1. Load cached ResNet features (same as M4) ---
    train_loader, val_loader, classes = get_dataloaders()
    print("[adv] Extracting ResNet features (one pass)...")
    resnet = build_frozen_resnet18_feature_extractor().to(device)
    # We need RAW IMAGES for FGSM, so we ALSO keep a DataLoader that returns
    # images. We'll iterate over the DataLoader directly during training,
    # recomputing ResNet(features) each step. That is slower than caching, but
    # required for image-space FGSM.
    #
    # Compromise: use cached features for clean half, recompute features only
    # for adv half during the FGSM. That is what run_epoch_adv does.
    #
    # For clean-only fast evaluation use cached features.
    Xtr, ytr = extract_features(train_loader, resnet, device)
    Xva, yva = extract_features(val_loader,   resnet, device)
    print(f"[adv] Clean features cached: train={Xtr.shape}, val={Xva.shape}")

    # Fit pipeline on TRAIN features (identical to M4)
    fp = FeaturePipeline(n_components=4).fit(Xtr)
    evr = fp.explained_variance()
    print(f"[adv] PCA cumulative variance kept: {evr.sum():.3f}")

    # --- 2. Build head + optimizer ---
    head = QTLHead(fp, device).to(device)
    n_params = sum(p.numel() for p in head.q_layer.parameters() if p.requires_grad)
    print(f"[adv] N_LAYERS={N_LAYERS}, N_QUBITS={N_QUBITS}, trainable params={n_params}")

    optimizer = Adam(head.q_layer.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # --- 3. Adversarial training loop ---
    # We iterate the TRAIN DATALOADER directly (raw images) so FGSM can perturb
    # them. ResNet runs on each batch (frozen) inside forward, so this is a bit
    # slower than M4's cached-feature training but still fine for Ants vs Bees.
    best_clean = 0.0
    best_path  = RESULTS_DIR / "qtl_head_robust.pth"
    log_rows   = []
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        head.train()
        losses, correct, seen = 0.0, 0, 0

        # Wrap for FGSM
        class _Wrap(nn.Module):
            def __init__(self, resnet, head):
                super().__init__()
                self.resnet = resnet
                self.head = head
            def forward(self, x):
                return self.head(self.resnet(x))
        full_model = _Wrap(resnet, head).to(device)

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            bsz = xb.size(0)
            n_adv = int(round(bsz * ADV_FRACTION))

            if n_adv > 0:
                xb_clean = xb[:bsz-n_adv]; yb_clean = yb[:bsz-n_adv]
                xb_a     = xb[bsz-n_adv:]; yb_a     = yb[bsz-n_adv:]
                xb_adv   = fgsm_attack(full_model, xb_a, yb_a,
                                       eps=EPS_TRAIN, criterion=criterion)
                xb_final = torch.cat([xb_clean, xb_adv], dim=0)
                yb_final = torch.cat([yb_clean, yb_a],     dim=0)
            else:
                xb_final, yb_final = xb, yb

            logits = full_model(xb_final)
            loss = criterion(logits, yb_final)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            losses  += loss.item() * xb_final.size(0)
            correct += (logits.argmax(1) == yb_final).sum().item()
            seen    += xb_final.size(0)

        # --- Clean validation on cached features (fast) ---
        head.eval()
        va_correct, va_total, va_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            Xva_t = torch.tensor(Xva, dtype=torch.float32)
            yva_t = torch.tensor(yva, dtype=torch.long)
            for i in range(0, Xva_t.size(0), BATCH_SIZE):
                xb = Xva_t[i:i+BATCH_SIZE].to(device)
                yb = yva_t[i:i+BATCH_SIZE].to(device)
                logits = head(xb)
                l = criterion(logits, yb)
                va_loss_sum += l.item() * yb.size(0)
                va_correct  += (logits.argmax(1) == yb).sum().item()
                va_total    += yb.size(0)

        tr_loss = losses / seen; tr_acc = correct / seen
        va_loss = va_loss_sum / va_total; va_acc = va_correct / va_total

        log_rows.append({"epoch": epoch,
                         "train_loss": tr_loss, "train_acc": tr_acc,
                         "val_loss":   va_loss, "val_acc":   va_acc})
        print(f"[adv epoch {epoch:02d}/{EPOCHS}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_clean:
            best_clean = va_acc
            torch.save({"head_state": head.q_layer.state_dict(),
                        "val_acc": va_acc, "classes": classes,
                        "eps_train": EPS_TRAIN,
                        "adv_fraction": ADV_FRACTION}, best_path)
            print(f"[adv]   New best val_acc={va_acc:.4f}")

    print(f"[adv] Training done in {time.time()-t0:.1f}s. "
          f"Best val_acc = {best_clean:.4f}")

    # --- 4. Save CSV + curves ---
    csv_path = RESULTS_DIR / "m6_qtl_robust_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","train_acc","val_loss","val_acc"])
        w.writeheader(); w.writerows(log_rows)
    print(f"[adv] Log saved -> {csv_path.name}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ex = [r["epoch"] for r in log_rows]
    ax1.plot(ex, [r["train_loss"] for r in log_rows], label="train")
    ax1.plot(ex, [r["val_loss"]   for r in log_rows], label="val")
    ax1.set_title("Robust QTL Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(ex, [r["train_acc"] for r in log_rows], label="train")
    ax2.plot(ex, [r["val_acc"]   for r in log_rows], label="val")
    ax2.set_title("Robust QTL Accuracy"); ax2.set_xlabel("epoch"); ax2.set_ylim(0, 1.05); ax2.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "m6_qtl_robust_curves.png", dpi=120); plt.close()
    print(f"[adv] Curves saved -> m6_qtl_robust_curves.png")


if __name__ == "__main__":
    main()