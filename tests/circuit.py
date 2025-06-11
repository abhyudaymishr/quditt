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

    C.gate(D.H, dits=[1])
    C.gate(D.CX, dits=[0, 1])
    C.gate(D.X, dits=[2])
    C.gate(D.Z, dits=[3])

    P = sym.exp(1j * sym.Symbol("p"))
    P = Gate(D.d, sym.SparseMatrix([[1, 0], [0, P]]), "P")
    C.gate(P, dits=[4])

    C.gate(D.X, dits=[0])
    C.gate(D.X, dits=[1])
    C.gate(D.I, dits=[2])
    C.gate(D.Z, dits=[3])
    C.gate(D.Z, dits=[4])

    C.gate(D.H, dits=[0])
    C.gate(D.X, dits=[2])
    C.gate(D.CX, dits=[1, 2])
    C.gate(D.H, dits=[0])

    C.gate(D.X, dits=[0])
    C.gate(D.X, dits=[1])
    C.gate(D.I, dits=[2])
    C.gate(D.Z, dits=[3])
    C.gate(D.Z, dits=[4])

    C.gate(D.I, dits=[0])
    C.gate(D.CX, dits=[2, 3])
    C.gate(D.CX, dits=[1, 4])

    print(C.draw())
    print("---" * 3)
    print(C.draw(output="penny"))

    print(f"Solved to: {C.solve().shape} sparse matrix")


if __name__ == "__main__":
    everything()
