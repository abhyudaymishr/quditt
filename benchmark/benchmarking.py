import sys
sys.path.append("..")

# from qutip_qip.operations import hadamard_transform, cnot
# from qiskit.quantum_info import Operator
from time import perf_counter as bench
# from qutip import basis, tensor, qeye
# from qiskit import QuantumCircuit
from qudit.circuit import Circuit
from qudit.gates import Gategen
# import matplotlib.pyplot as plt
# import numpy as np
# import cirq as CQ

REPEATS = 100
ms = 1e3

qutip_times = []
qiskit_times = []
cirq_times = []
custom_times = []

def benchmark_custom(n):
    G = Gategen(2)
    C = Circuit(n)
    C.gate(G.H, dits=[0])
    for i in range(n - 1):
        C.gate(G.CX, dits=[i, i + 1])
    start = bench()
    for _ in range(REPEATS):
        _ = C.solve()
    return (bench() - start) / REPEATS

def benchmark_qiskit(n):
    start = bench()
    for _ in range(REPEATS):
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        _ = Operator.from_circuit(qc)
    return (bench() - start) / REPEATS

def benchmark_cirq(n):
    q = CQ.LineQubit.range(n)
    ops = [CQ.H(q[0])] + [CQ.CNOT(q[i], q[i+1]) for i in range(n - 1)]
    circuit = CQ.Circuit(ops)
    sim = CQ.Simulator()
    start = bench()
    for _ in range(REPEATS):
        _ = sim.simulate(circuit)
    return (bench() - start) / REPEATS

def benchmark_qutip(n):
    ket0 = basis(2, 0)
    psi = tensor([ket0] * n)
    H = hadamard_transform(1)
    I = qeye(2)
    start = bench()
    for _ in range(REPEATS):
        state = psi
        ops = [I] * n
        ops[0] = H
        state = tensor(ops) * state
        for i in range(n - 1):
            CX = cnot(n, control=i, target=i + 1)
            state = CX * state
    return (bench() - start) / REPEATS

# n_range = range(2, 12)
n_range = range(8, 12)
for n in n_range:
    print(f"{n}/{len(n_range) + 2}")
    if n>=10: REPEATS = 10

    # t_qiskit = ms * benchmark_qiskit(n)
    # print(f"\tQiskit: {t_qiskit:.3f} ms")
    # t_cirq = ms * benchmark_cirq(n)
    # print(f"\tCirq: {t_cirq:.3f} ms")
    # t_qutip = ms * benchmark_qutip(n)
    # print(f"\tQuTiP: {t_qutip:.3f} ms")
    t_custom = ms * benchmark_custom(n)
    print(f"\tQudit: {t_custom:.3f} ms")

    # total time for each library
    # tot = t_custom + t_qiskit + t_cirq + t_qutip
    # print(f"\tTotal: {tot*REPEATS/60:.3f}")

    custom_times.append(t_custom)
    # qiskit_times.append(t_qiskit)
    # cirq_times.append(t_cirq)
    # qutip_times.append(t_qutip)


# custom_times = np.log(custom_times)
# qiskit_times = np.log(qiskit_times)
# cirq_times = np.log(cirq_times)
# qutip_times = np.log(qutip_times)

# # Plotting
# plt.plot(n_range, custom_times, label='Qudit', marker=".")
# plt.plot(n_range, qiskit_times, label='Qiskit', marker=".")
# plt.plot(n_range, cirq_times, label='Cirq', marker=".")
# plt.plot(n_range, qutip_times, label='QuTiP', marker=".")
# plt.xlabel("Number of Qubits (n)")
# plt.ylabel("Avg Time per Run (ms)")
# plt.title("GHZ Circuit Benchmark")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
