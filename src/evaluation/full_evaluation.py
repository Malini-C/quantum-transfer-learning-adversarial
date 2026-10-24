"""
Milestone 7 - Full comparison evaluation.

Evaluates FOUR models:
  1. Classical ResNet18  (from M2)
  2. Classical ResNet18 + FGSM adv training  (from M7 step 1)
  3. QTL clean-trained   (from M4)
  4. QTL + FGSM adv training  (from M6)

For each, computes at eps in {0, 1, 2, 4, 8}/255:
  - accuracy
  - precision, recall, F1 (macro)
  - attack success rate (correct -> incorrect transitions)

Also produces:
  - confusion matrices at eps=2/255
  - robustness-vs-eps plot
  - results/m7_full_table.csv
  - results/m7_summary.md (a human-readable summary)

Run:
  python -m src.evaluation.full_evaluation
"""

import sys, csv, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, SEED
from src.data.dataset import get_dataloaders
from src.models.classical import build_classical_resnet18
from src.models.qtl import QTLModel
from src.attacks.fgsm import fgsm_attack


EPS_LIST = [0.0, 1/255, 2/255, 4/255, 8/255]


# ---------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------
def load_classical(device, ckpt_name, num_classes=2):
    model = build_classical_resnet18(num_classes=num_classes, freeze_backbone=False).to(device)
    ckpt = torch.load(RESULTS_DIR / ckpt_name, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_qtl(device, ckpt_name):
    pipe = joblib.load(RESULTS_DIR / "qtl_pipeline.joblib")
    model = QTLModel(pipe, device).to(device)
    ckpt = torch.load(RESULTS_DIR / ckpt_name, map_location=device)
    model.head.q_layer.load_state_dict(ckpt["head_state"])
    model.eval()
    return model


# ---------------------------------------------------------------
# Evaluation for one model at one eps
# ---------------------------------------------------------------
@torch.no_grad()
def predict_clean(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        p = model(xb).argmax(1).cpu().numpy()
        ys.append(yb.numpy()); ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)


def predict_adv(model, loader, device, eps):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    ys, ps = [], []
    for xb, yb in loader:
        xb, yb_dev = xb.to(device), yb.to(device)
        xb_adv = fgsm_attack(model, xb, yb_dev, eps=eps, criterion=criterion)
        with torch.no_grad():
            p = model(xb_adv).argmax(1).cpu().numpy()
        ys.append(yb.numpy()); ps.append(p)
    return np.concatenate(ys), np.concatenate(ps)


def compute_metrics(y_true, y_pred):
    acc = float((y_true == y_pred).mean())
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": float(prec),
            "recall": float(rec), "f1": float(f1)}


def compute_asr(y_true, y_clean_pred, y_adv_pred):
    """
    Attack Success Rate = fraction of originally-correct predictions that flip.
    """
    orig_correct = (y_clean_pred == y_true)
    n_correct = int(orig_correct.sum())
    if n_correct == 0:
        return 0.0
    flipped = (y_adv_pred != y_true) & orig_correct
    return float(flipped.sum() / n_correct)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[m7] Device: {device}")

    _, val_loader, classes = get_dataloaders()
    print(f"[m7] Classes: {classes}")

    # ---- Load all four models ----
    print("[m7] Loading models ...")
    models = {
        "Classical":         load_classical(device, "classical_resnet18.pth"),
        "Classical+Adv":     load_classical(device, "classical_resnet18_robust.pth"),
        "QTL":               load_qtl(device, "qtl_head.pth"),
        "QTL+Adv":           load_qtl(device, "qtl_head_robust.pth"),
    }

    # ---- Compute clean predictions once per model (needed for ASR) ----
    print("[m7] Clean predictions ...")
    clean_preds = {}
    for name, model in models.items():
        y_true, y_pred = predict_clean(model, val_loader, device)
        clean_preds[name] = (y_true, y_pred)
        m = compute_metrics(y_true, y_pred)
        print(f"[m7]   {name:14s}  clean acc={m['accuracy']:.4f}")

    # ---- Full per-eps sweep ----
    print("\n[m7] Full sweep over eps ...")
    rows = []
    for name, model in models.items():
        y_true_clean, y_pred_clean = clean_preds[name]
        for eps in EPS_LIST:
            if eps == 0.0:
                y_pred = y_pred_clean
            else:
                _, y_pred = predict_adv(model, val_loader, device, eps=eps)
            m = compute_metrics(y_true_clean, y_pred)
            asr = 0.0 if eps == 0.0 else compute_asr(y_true_clean, y_pred_clean, y_pred)
            row = {"model": name, "eps": eps, "accuracy": m["accuracy"],
                   "precision": m["precision"], "recall": m["recall"],
                   "f1": m["f1"], "asr": asr}
            rows.append(row)
            print(f"[m7]   {name:14s}  eps={eps:.4f}  "
                  f"acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  asr={asr:.4f}")

    # ---- Save CSV ----
    csv_path = RESULTS_DIR / "m7_full_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","eps","accuracy",
                                          "precision","recall","f1","asr"])
        w.writeheader(); w.writerows(rows)
    print(f"\n[m7] Table saved -> {csv_path.name}")

    # ---- Save human-readable summary ----
    md_path = RESULTS_DIR / "m7_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# M7 Results Summary\n\n")
        f.write("## Clean accuracy\n\n")
        f.write("| Model | Clean acc | F1 |\n|---|---|---|\n")
        for name in models:
            r = next(x for x in rows if x["model"] == name and x["eps"] == 0.0)
            f.write(f"| {name} | {r['accuracy']:.4f} | {r['f1']:.4f} |\n")
        f.write("\n## Robustness (accuracy under FGSM)\n\n")
        header = "| Model | " + " | ".join(f"eps={e:.4f}" for e in EPS_LIST) + " |\n"
        f.write(header)
        f.write("|" + "---|" * (len(EPS_LIST) + 1) + "\n")
        for name in models:
            accs = [next(x for x in rows if x["model"]==name and x["eps"]==e)["accuracy"]
                    for e in EPS_LIST]
            f.write(f"| {name} | " + " | ".join(f"{a:.4f}" for a in accs) + " |\n")
        f.write("\n## Attack Success Rate\n\n")
        f.write(header)
        f.write("|" + "---|" * (len(EPS_LIST) + 1) + "\n")
        for name in models:
            asrs = [next(x for x in rows if x["model"]==name and x["eps"]==e)["asr"]
                    for e in EPS_LIST]
            f.write(f"| {name} | " + " | ".join(f"{a:.4f}" for a in asrs) + " |\n")
    print(f"[m7] Summary saved -> {md_path.name}")

    # ---- Robustness plot (all 4 models) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"Classical": "tab:blue", "Classical+Adv": "tab:cyan",
              "QTL": "tab:red", "QTL+Adv": "tab:orange"}
    for name in models:
        accs = [next(x for x in rows if x["model"]==name and x["eps"]==e)["accuracy"]
                for e in EPS_LIST]
        ax.plot(EPS_LIST, accs, marker="o", label=name, color=colors[name])
    ax.set_xlabel("FGSM epsilon (normalised pixel units)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Robustness comparison")
    ax.set_ylim(0, 1.02); ax.grid(True); ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "m7_robustness_plot.png", dpi=130)
    plt.close()
    print(f"[m7] Robustness plot saved -> m7_robustness_plot.png")

    # ---- Confusion matrices at eps=2/255 for each model ----
    print("\n[m7] Confusion matrices at eps=2/255 ...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, model) in zip(axes.flat, models.items()):
        _, y_pred = predict_adv(model, val_loader, device, eps=2/255)
        y_true = clean_preds[name][0]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f"{name} @ eps=2/255")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "m7_confusion_matrices.png", dpi=130)
    plt.close()
    print(f"[m7] Confusion matrices saved -> m7_confusion_matrices.png")


if __name__ == "__main__":
    main()