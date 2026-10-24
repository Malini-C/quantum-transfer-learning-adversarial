"""
4-qubit variational quantum circuit for binary classification.

Design
------
Encoding:      RY(x_i) on qubit i
Variational:   RY(theta_i), RZ(theta_i), ring-CNOT, RY(theta_i), RZ(theta_i)
Measurement:   <Z> on qubits 0 and 1  -> 2 logits

The QNode is wrapped in a PennyLane TorchLayer so PyTorch can backpropagate
through it and treat it like any nn.Module layer.
"""

import pennylane as qml
import torch.nn as nn

N_QUBITS  = 4
N_INPUTS  = 4             # 4 classical features from PCA
N_OUTPUTS = 2             # 2 classes (ants, bees)
N_LAYERS  = 2             # number of variational layers; keep 1 for M3

# --- Device: CPU simulator, no quantum hardware needed ---
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def _qnode(inputs, weights):
    """
    inputs:  shape (N_INPUTS,)     -> 4 classical features
    weights: shape (N_LAYERS, N_QUBITS, 4) -> trainable angles
    """
    # 1. Encoding layer: put each input feature on its own qubit
    for i in range(N_INPUTS):
        qml.RY(inputs[i], wires=i)

    # 2. Variational layers
    for layer in range(N_LAYERS):
        # Rotation block 1
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)

        # Entanglement (ring CNOT)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

        # Rotation block 2
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 2], wires=i)
            qml.RZ(weights[layer, i, 3], wires=i)

    # 3. Measurement -> logits
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]


def build_quantum_classifier() -> nn.Module:
    """
    Wrap the QNode in a TorchLayer so it plugs into nn.Sequential.
    """
    weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 4)}
    return qml.qnn.TorchLayer(_qnode, weight_shapes)


if __name__ == "__main__":
    # Quick self-test
    layer = build_quantum_classifier()
    x = __import__("torch").randn(N_INPUTS)
    out = layer(x)
    print(f"[quantum] input shape : {x.shape}")
    print(f"[quantum] output shape: {out.shape}")
    print(f"[quantum] output value: {out.detach().numpy()}")