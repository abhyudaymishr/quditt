import time
import numpy as np
from scipy.sparse import csr_matrix
from qudit.circuit import Circuit,cfn,Layer
from qudit.gates import Gategen
from qudit.index import Gate, VarGate,State,Basis
from qudit.utils import ID,Braket,Tensor,partial
from qudit.algebra import gellmann,dGellMann,Unity
import cirq

import qutip
from qutip import basis, tensor
from qutip.qip.operations import hadamard_transform, cnot




REPEATS = 100

def Unity(d):
    return np.exp(2j * np.pi / d)

def benchmark_custom():
    D = Gategen(2)
    C = Circuit(2)
    C.gate(D.H, dits=[0])
    C.gate(D.CX, dits=[0, 1])

    start = time.perf_counter()
    for _ in range(REPEATS):
        _ = C.solve()
    end = time.perf_counter()

    return (end - start) / REPEATS


def benchmark_qiskit():
    from qiskit import QuantumCircuit, transpile, assemble
    from qiskit.providers.aer import Aer

    backend = Aer.get_backend("unitary_simulator")

    start = time.perf_counter()
    for _ in range(REPEATS):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        tqc = transpile(qc, backend)
        qobj = assemble(tqc)
        result = backend.run(qobj).result()
        _ = result.get_unitary()
    end = time.perf_counter()
    return (end - start) / REPEATS


def benchmark_cirq():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    simulator = cirq.Simulator()

    start = time.perf_counter()
    for _ in range(REPEATS):
        _ = simulator.simulate(circuit)
    end = time.perf_counter()
    return (end - start) / REPEATS
def benchmark_qutip():
    ket0 = basis(2, 0)
    H = hadamard_transform(1)
    I = qutip.qeye(2)
    CX = cnot()

    start = time.perf_counter()
    for _ in range(REPEATS):
        psi = tensor(ket0, ket0)
        psi = tensor(H, I) * psi
        bell = CX * psi
    end = time.perf_counter()

    return (end - start) / REPEATS


if __name__ == "__main__":
    print(f"Benchmarking over {REPEATS} runs...\n")
    print(f"Qudit     : {benchmark_custom():.6f} s")
   
    print(f"Cirq             : {benchmark_cirq():.6f} s")
    print(f"QuTiP            : {benchmark_qutip():.6f} s")
