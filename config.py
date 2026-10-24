"""
Central configuration for the QTL + Adversarial Robustness project.
Every module imports from here so that paths, seeds, and hyperparameters
are consistent across the whole project.
"""

from pathlib import Path

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Reproducibility ----------------
SEED = 42

# ---------------- Image / Dataloader ----------------
IMG_SIZE   = 224          # ResNet18 expects 224x224
BATCH_SIZE = 16           # small enough for a laptop
NUM_WORKERS = 2           # Windows: 0 if you get DataLoader errors

# ImageNet normalisation statistics
# (used because ResNet18 was trained on ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)