import sys
sys.path.append("..")

from qudit.circuit import Circuit
from qudit.gates import Gategen

n = 10
G = Gategen(2)
C = Circuit(n)
C.gate(G.H, dits=[0])
for i in range(n - 1):
    C.gate(G.CX, dits=[i, i + 1])

_ = C.solve()
