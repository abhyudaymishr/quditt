import sys

sys.path.append("..")

from qudit import Layer, Gategen, Circuit
import numpy as np

# import scipy as sp


D = Gategen(2)

C = Circuit()
C.layer(D.X, D.X, D.I, D.Z, D.Z)
C.layer(D.H, D.X, D.CX(1), D.H)
C.layer(D.I, D.CX(3), D.CX(1))

print(C)

# print(C.draw())
# print("---" * 3)
# print(C.draw(output="penny"))

# print(f"Solved to: {C.solve().shape} sparse matrix")
