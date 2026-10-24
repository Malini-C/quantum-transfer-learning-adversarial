import sys
print("Python:", sys.version)

import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

import torchvision
print("torchvision:", torchvision.__version__)

import pennylane as qml
print("PennyLane:", qml.__version__)

# Minimal quantum circuit test
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

result = circuit(0.5)
print("Minimal QNode result:", result)

# Minimal TorchLayer test
weight_shapes = {"w": 1}
@qml.qnode(dev)
def qnode(inputs, w):
    qml.RX(inputs, wires=0)
    qml.RY(w, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
out = qlayer(torch.tensor(0.3))
print("TorchLayer output:", out)
print("VERIFICATION SUCCESSFUL")