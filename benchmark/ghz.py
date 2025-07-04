import time
import numpy as np
from scipy.sparse import csr_matrix
from qudit.circuit import Circuit,cfn,Layer
from qudit.gates import Gategen
from qudit.index import Gate, VarGate,State,Basis
from qudit.utils import ID,Braket,Tensor,partial
from qudit.algebra import gellmann,dGellMann,Unity
import cirq
from qutip import *
from qutip import basis, tensor
from qutip.qip.operations import hadamard_transform, cnot
import matplotlib as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

REPEATS      = 100              # keep small for quick runs
QUBIT_SIZES  = [2, 3, 4, 5]       # points for the x‑axis


def Qudit(n_qubits):
    D = Gategen(n_qubits)
    C = Circuit(n_qubits)

    C.gate(D.H, dits=[0])
    for i in range(n_qubits - 1):
        C.gate(D.CX, dits=[i, i + 1])

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        _ = C.solve()
    t1 = time.perf_counter()
    return (t1 - t0) / REPEATS


def Cirq(n_qubits):
    qs   = cirq.LineQubit.range(n_qubits)
    circ = cirq.Circuit()
    circ.append(cirq.H(qs[0]))
    for i in range(n_qubits - 1):
        circ.append(cirq.CNOT(qs[i], qs[i + 1]))
    sim  = cirq.Simulator()

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        _ = sim.simulate(circ)
    t1 = time.perf_counter()
    return (t1 - t0) / REPEATS


def Qutip(n_qubits):
    H    = hadamard_transform(1)
    I    = qutip.qeye(2)
    ket0 = basis(2, 0)

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        state = tensor([ket0] * n_qubits)
        ops   = [H] + [I]*(n_qubits-1)
        state = tensor(ops) * state
        for i in range(n_qubits - 1):
            state = qutip.qip.operations.cnot(n_qubits, i, i+1) * state
    t1 = time.perf_counter()
    return (t1 - t0) / REPEATS


def Qiskit(n_qubits):
    """Benchmark using Qiskit Aer statevector simulator."""
    simulator = AerSimulator(method="statevector")

    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    tqc = transpile(qc, simulator)

    t0 = time.perf_counter()
    for _ in range(REPEATS):
        _ = simulator.run(tqc).result()
    t1 = time.perf_counter()
    return (t1 - t0) / REPEATS


times_qudit  = []
times_cirq   = []
times_qutip  = []
times_qiskit = []

print(f"Running {REPEATS} repetitions per point …\n")
for nq in QUBIT_SIZES:
    print(f"{nq} qubits:", end=" ", flush=True)
    t_qudit  = Qudit(nq)
    t_cirq   = Cirq(nq)
    t_qutip  = Qutip(nq)
    t_qiskit = Qiskit(nq)
    print(f"Qudit={t_qudit:.4f}s, Cirq={t_cirq:.4f}s, QuTiP={t_qutip:.4f}s, Qiskit={t_qiskit:.4f}s")

    times_qudit.append(t_qudit)
    times_cirq.append(t_cirq)
    times_qutip.append(t_qutip)
    times_qiskit.append(t_qiskit)


plt.figure(figsize=(8, 5))
plt.plot(QUBIT_SIZES, times_qudit,  marker="o", label="Qudit")
plt.plot(QUBIT_SIZES, times_cirq,   marker="s", label="Cirq")
plt.plot(QUBIT_SIZES, times_qutip,  marker="^", label="QuTiP")
plt.plot(QUBIT_SIZES, times_qiskit, marker="d", label="Qiskit")

plt.xlabel("Number of qubits in GHZ circuit")
plt.ylabel("Average simulation time per run (s)")
plt.title(f"GHZ benchmark – {REPEATS} repetitions per data point")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()