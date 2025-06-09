import sys

sys.path.append("..")

import scipy.sparse as sp
import sympy as sym
from qudit import *
import numpy as np
import math as ma

def everything():
    D = Gategen(2)

    C = Circuit()
    C.layer(D.X, D.X, D.I, D.Z, D.Z)
    C.layer(D.H, D._, D.X, D.CX(1), D.H)
    C.barrier()
    C.layer(D.X, D.X, D.I, D.Z, D.Z)
    C.barrier()
    C.layer(D.I, D._, D.CX(3), D._, D.CX(1))

    print(C.draw())
    print("---" * 3)
    print(C.draw(output="penny"))

    print(f"Solved to: {C.solve().shape} sparse matrix")


def var_everything():
    D = Gategen(2)

    C = Circuit()
    p = sym.exp(1j * sym.Symbol("p"))
    P = Gate(D.d, sym.Matrix([[1, 0], [0, p]]), "P")

    C.layer(D.X, D.X, P, D.Z, D.Z)
    C.layer(D.H, D._, D.X, D.CX(1), D.H)
    C.barrier()
    C.layer(D.X, D.X, D.I, D.Z, D.Z)
    C.barrier()
    C.layer(D.I, D._, D.CX(3), D._, D.CX(1))

    print(C.draw())
    print("---" * 3)
    print(C.draw(output="penny"))

    print(f"Solved to: {C.solve().shape} sparse matrix")


if __name__ == "__main__":
    everything()
    var_everything()
