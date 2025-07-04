import time
from qiskit.circuit.library import QFT                 # Qiskit
from qiskit.quantum_info import Operator
import cirq                                            # Cirq
from qutip import *
from qudit.circuit import Circuit,cfn,Layer
from qudit.gates import Gategen
from qudit.index import Gate, VarGate,State,Basis
from qudit.utils import ID,Braket,Tensor,partial
from qudit.algebra import gellmann,dGellMann,Unity# QuTiP
from qudit.qft import *
d, n = 2, 4  # 4 qubits

# --- Our QFT ---
t0 = time.perf_counter()
_ = qft.gate(n, d)
print(f"Qudit QFT time     : {(time.perf_counter() - t0)*1e3:.3f} ms")

# --- Qiskit QFT ---
t0 = time.perf_counter()
qc = QFT(num_qubits=n, inverse=False, do_swaps=False)
_ = Operator(qc).data
print(f"Qiskit QFT time     : {(time.perf_counter() - t0)*1e3:.3f} ms")

# --- Cirq QFT ---
qubits = cirq.LineQubit.range(n)
t0 = time.perf_counter()
circuit = cirq.qft(*qubits, without_reverse=True)
_ = cirq.unitary(circuit)
print(f"Cirq QFT time       : {(time.perf_counter() - t0)*1e3:.3f} ms")

# --- QuTiP QFT ---
t0 = time.perf_counter()
_ = qft(n).full()
print(f"QuTiP QFT time      : {(time.perf_counter() - t0)*1e3:.3f} ms")
