import sys

sys.path.append("..")

import scipy.sparse as sp
import sympy as sym
from qudit import *
import numpy as np
import math as m

D = Gategen(2)


def everything():
    C = Circuit(5)

    P = sym.exp(1j * sym.Symbol("p"))
    P = sym.SparseMatrix([[1, 0], [0, P]])
    P = Gate(D.d, P, "P")

    for i in range(5):
        C.gate(D.H, dits=[i])
        C.gate(D.CX, dits=[i, (i + 1) % 5])
        C.gate(D.X, dits=[i])
        C.gate(D.Y, dits=[i])
        C.gate(D.Z, dits=[i])

    # C.barrier()
    C.gate(P, dits=[4])

    print(C.draw())

    sum = np.sum(C.solve())
    sum = np.abs( sum.subs("p", 0.5).n() )
    print(sum)


if __name__ == "__main__":
    everything()
