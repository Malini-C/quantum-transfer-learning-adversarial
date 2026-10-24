"""
Milestone 3 verification:
  1. VQC produces 2 outputs for a given 4-feature input.
  2. Output depends on trainable parameters (they actually do something).
  3. Gradients flow through the TorchLayer (backprop works).
  4. Circuit diagram is saved to results/m3_circuit.png.
"""

import sys
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, SEED
from src.models.quantum import (
    _qnode, build_quantum_classifier,
    N_INPUTS, N_QUBITS, N_LAYERS,
)

torch.manual_seed(SEED)


def check_shapes():
    layer = build_quantum_classifier()
    x = torch.randn(N_INPUTS)
    out = layer(x)
    assert out.shape == (2,), f"Expected shape (2,), got {tuple(out.shape)}"
    print(f"[check 1] output shape OK: {tuple(out.shape)}, values = {out.detach().numpy()}")


def check_param_sensitivity():
    """Changing weights must change the output (proves params matter)."""
    layer = build_quantum_classifier()
    x = torch.randn(N_INPUTS)

    with torch.no_grad():
        out_a = layer(x).clone()
        # perturb every trainable weight
        for p in layer.parameters():
            p.add_(0.3)
        out_b = layer(x).clone()

    diff = (out_a - out_b).abs().max().item()
    assert diff > 1e-3, f"Outputs too similar after weight change: diff={diff}"
    print(f"[check 2] parameters influence output; max diff = {diff:.4f}")


def check_gradients():
    """Gradient must flow into trainable weights."""
    layer = build_quantum_classifier()
    x = torch.randn(N_INPUTS)
    target = torch.tensor([0.0, 1.0])       # arbitrary target logits

    out = layer(x)
    loss = ((out - target) ** 2).sum()
    loss.backward()

    params = list(layer.parameters())
    assert len(params) == 1, f"Expected 1 parameter tensor, got {len(params)}"
    grad = params[0].grad
    assert grad is not None, "No gradient was computed"
    assert grad.abs().sum().item() > 0, "Gradient is all zeros"
    print(f"[check 3] gradient OK; grad shape = {tuple(grad.shape)}, "
          f"sum |grad| = {grad.abs().sum().item():.6f}")


def save_circuit_diagram():
    """Draw the QNode and save as PNG for the report."""
    # qml.draw_mpl produces a matplotlib figure using a dummy weights tensor
    weights = torch.zeros(N_LAYERS, N_QUBITS, 4)
    inputs  = torch.zeros(N_INPUTS)

    fig, ax = qml.draw_mpl(_qnode, decimals=2, expansion_strategy="device")(inputs, weights)
    out_path = RESULTS_DIR / "m3_circuit.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[check 4] circuit diagram saved -> {out_path}")


def main():
    print(f"[test_quantum] N_QUBITS={N_QUBITS}, N_LAYERS={N_LAYERS}, N_INPUTS={N_INPUTS}")
    check_shapes()
    check_param_sensitivity()
    check_gradients()
    save_circuit_diagram()
    print("\nALL CHECKS PASSED - Milestone 3 is complete.")


if __name__ == "__main__":
    main()