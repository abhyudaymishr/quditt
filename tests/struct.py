import sys

sys.path.append("..")
from qudit import Layer, Gategen, Circuit
import numpy as np
import scipy as sp


def test_render():
    D = Gategen(2)

    L1 = Layer(D.H, D.X, D.CX, D.H)
    L2 = Layer(D.I, D.CX, D.CX)
    L3 = Layer(D.X, D.X, D.I, D.Z, D.Z)
    C = Circuit(L1, L2, L3)

    print(C.draw())
    print("---" * 3)
    print(C.draw(output="penny"))


D = Gategen(2)

L1 = Layer(D.H, D.X, D.CX, D.H)
L2 = Layer(D.I, D.CX, D.CX)
L3 = Layer(D.X, D.X, D.I, D.Z, D.Z)
C = Circuit(L1, L2, L3)

# print(C.draw())
# print("---" * 3)
# print(C.draw(output="penny"))

print(f"Solved to: {C.solve().shape} sparse matrix")
