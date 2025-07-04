from typing import List
import numpy as np
import math as ma
from math import pi
from numpy import exp
from scipy.sparse import csr_matrix
from qudit.circuit import Circuit,cfn,Layer
from qudit.gates import Gategen
from qudit.index import Gate, VarGate,State,Basis
from qudit.utils import ID,Braket,Tensor,partial
from qudit.algebra import gellmann,dGellMann,Unity



def _phase_diag(d: int, denom: int, name: str = "SUMP") -> "Gate":
    
    w = exp(2j * pi / (d ** denom))
    O = np.diag([w ** k for k in range(d)])
    return Gate(d, O, f"{name}_{denom}")

class qft:
    @staticmethod
    
    def gate(n: int, d: int) -> "Gate":
   
     if n < 1:
        raise ValueError("Need at least one qudit (n ≥ 1)")
     if d < 2:
        raise ValueError("Local dimension must be ≥ 2")

     N = d ** n
     w = np.exp(2j * np.pi / N)
     F = np.empty((N, N), dtype=complex)
     for j in range(N):
         F[j, :] = w ** (j * np.arange(N))
     F /= np.sqrt(N)
     return Gate(d, F, f"QFT_{n}_{d}")


    @staticmethod
    def circuit(n: int, d: int) -> "Circuit":
    
     if n < 1:
        raise ValueError("Need at least one qudit (n ≥ 1)")
     if d < 2:
        raise ValueError("Local dimension must be ≥ 2")

     G = Gategen(d)
     circ = Circuit(size=n)

    
     for t in range(n):
        
        circ.gate(G.H, [t])

        
        for c in range(t + 1, n):
            
            denom = (c - t + 1)
            R = _phase_diag(d, denom)
            CR = G.CU(R, rev=False)  
            circ.gate(CR, [c, t])

     return circ



"""""
if __name__ == "__main__":
   
    qft_gate = qft.gate(n, d)
    print("Dense QFT gate is unitary?", qft_gate.isUnitary())
    print("Span (qudits acted on):", qft_gate.span)

    # Circuit  -----------------------------------------------------------------
    qft_circ = qft.circuit(n, d)
    print("\nDecomposed circuit:\n")
    print(qft_circ.draw())""""
