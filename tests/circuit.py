import sys

sys.path.append("..")

from sympy import exp, SparseMatrix, Symbol
from qudit import Gategen, Circuit
import numpy as np

D = Gategen(2)
C = Circuit(4)


def everything():
    P = exp(1j * Symbol("p"))
    P = SparseMatrix([[1, 0], [0, P]])
    P = D.create(P, "P")

    for i in range(4):
        ip = (i + 1) % 4

        C.gate(D.CX, dits=[i, ip])
        C.gate(D.X, dits=[ip])
        C.gate(D.Y, dits=[i])
        C.gate(D.Z, dits=[ip])
        C.gate(D.H, dits=[i])
        C.barrier()

    C.gate(P, dits=[3])
    print(C.draw())

    sum = np.sum(C.solve())
    sum = np.abs(sum.subs("p", 0.5).n())
    print(sum)


if __name__ == "__main__":
    everything()
