import sys

sys.path.append("..")

from qudit.tools.entanglement import *
from qudit import *
import numpy as np

THETA = 1.5
D, r = 5, 2

Bits, Trits = Basis(2), Basis(3)
def Psi(i):
  A = Bits(0) ^ Trits( i )
  B = Bits(1) ^ Trits( i + 1 )

  return A * np.cos(THETA/2) + B * np.sin(THETA/2)


perp = Perp([Psi(i) for i in range(2)])
def system(X):
  qbit = Vec(X[1:5].reshape(2, 2).dot([1, 1j]))
  qtrit = Vec(X[5:11].reshape(3, 2).dot([1, 1j]))

  phi_rx = (X[0] * (qbit ^ qtrit)).norm()
  return Loss(phi_rx, perp)

res = rank(system, D, r, tries=2)
print(f'E_r(θ={THETA:.2f}) = {res:.4f}')
