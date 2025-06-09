from qudit.tools.metrics import Fidelity, negativity

import numpy as np


def test_channel():
    K0 = np.sqrt(0.5) * np.eye(2)
    K1 = np.sqrt(0.5) * np.array([[0, 1], [1, 0]])
    kraus_ops = [K0, K1]

    # Input state: |0⟩
    rho = np.array([[1, 0], [0, 0]], dtype=complex)

    output = Fidelity.channel(kraus_ops, rho)
    print("Quantum Channel Output (should be 0.5*|0><0| + 0.5*|1><1|):")
    print(np.round(output, 3))


def test_entanglement_fidelity():
    p = 0.25
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    kraus_ops = [
        np.sqrt(1 - 3 * p / 4) * I,
        np.sqrt(p / 4) * X,
        np.sqrt(p / 4) * Y,
        np.sqrt(p / 4) * Z,
    ]

    # Input state: |0⟩⟨0|
    rho = np.array([[1, 0], [0, 0]], dtype=complex)

    Fe = Fidelity.entanglement(rho, kraus_ops)
    print(f"Entanglement Fidelity: {Fe:.4f} (Expected: ~0.8125 for p=0.25)")


def test_fidelity():
    psi = np.array([1, 0], dtype=complex)  # |0⟩
    phi = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)  # (|0⟩ + |1⟩)/√2

    # f = Fidelity(psi, phi)
    f = Fidelity.default(psi, phi)
    print(f"Uhlmann Fidelity Test: {f:.4f} (Expected: ~0.7071)")


def test_negativity():
    d = 3
    psi = np.zeros((d, d), dtype=complex)
    for i in range(d):
        psi[i, i] = 1
    psi = psi.flatten() / np.sqrt(d)
    rho = np.outer(psi, psi.conj())
    N = negativity(rho, d, d)
    print(f"Qutrit Entangled State Negativity: {N:.4f} (Expected: > 0)")


if __name__ == "__main__":
    test_fidelity()
    test_channel()
    test_negativity()
    test_entanglement_fidelity()
