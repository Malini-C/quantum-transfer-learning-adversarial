# Quantum Transfer Learning with Adversarial Robustness

A hybrid quantum-classical image classifier with FGSM adversarial training, implemented with PyTorch, torchvision, and PennyLane.

Evaluated on the Ants vs Bees dataset.


## Results Summary

| Model | Clean Acc | FGSM at eps=2/255 | FGSM at eps=8/255 | Trainable Params |
|---|---|---|---|---|
| Classical ResNet18 | 92.81% | 54.90% | 23.53% | 11,177,538 |
| Classical ResNet18 + Adv | 90.85% | **79.74%** | **48.37%** | 11,177,538 |
| QTL (4 qubits, 2 layers) | **95.42%** | 46.41% | 16.34% | **32** |
| QTL + Adv | **96.08%** | 53.59% | 22.22% | **32** |

## Architecture

Image (224x224x3) -> ResNet18 (frozen) -> 512-d features -> StandardScaler + PCA(4, whiten) -> Scale to [-pi/2, pi/2] -> 4-qubit VQC (2 layers, 32 params) -> CrossEntropyLoss

## Reproducing the results

```bash
pip install -r requirements.txt

python -m src.training.train_classical
python -m src.training.train_qtl
python -m src.training.train_classical_adv
python -m src.training.adversarial_training
python -m src.evaluation.full_evaluation
python -m src.demo
```

## Key findings

- Clean-accuracy parity: QTL (32 quantum params) matches classical 11.2M-param ResNet18 on this dataset.
- QTL is more fragile under FGSM at every tested eps, likely due to extreme parameter reduction and the 4-dimensional PCA bottleneck.
- Adversarial training works for both models, but classical adversarial training improves robustness far more (+24.8 pp at eps=8/255) than quantum adversarial training (+5.9 pp).

## Honest scope

This project does not claim quantum advantage. It provides a reproducible benchmark for QML adversarial robustness on small datasets and 4-qubit registers.


- src/models/  -- ResNet18, VQC, QTL pipeline
- src/training/ -- training scripts for 4 models
- src/attacks/fgsm.py -- FGSM implementation
- src/evaluation/ -- evaluation and comparison
- results/ -- all figures, CSVs, PDF summary, and trained checkpoints
