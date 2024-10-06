from .algebra import w, dGellMann
from typing import List
from .qdits import Dit
import numpy as np
import math as ma

# gate as a wrapper around a numpy.ndarray
class Gate(np.ndarray):
  def __new__(cls, d: int, O: np.ndarray=None, name: str = None):
    if O is None:
      obj = np.zeros((d, d), dtype=complex).view(cls)
      obj.sz = 1
    else:
      obj = np.asarray(O, dtype=complex).view(cls)
      obj.sz = ma.log(len(O[0]), d)
    # endif

    obj.name = name if name else "Gate"
    obj.d = d

    return obj

  def __array_finalize__(self, obj):
    if obj is None: return
    self.d = getattr(obj, 'd', None)
    self.sz = getattr(obj, 'sz', None)
    self.name = getattr(obj, 'name', None)

  def is_unitary(self):
    return np.allclose(self @ self.conj().T, np.eye(self.shape[0]))

ck = 23
# special class to create "d" once and pass through all gates
# so G = DGate(d) -> G.X -> G.Z -> G.H -> ...
class DGate:
  def __init__(self, d: int):
    self.d = d

  @property
  def X(self):
    O = np.zeros((self.d, self.d))
    O[0, self.d - 1] = 1
    O[1:, 0:self.d - 1] = np.eye(self.d - 1)
    return Gate(self.d, O, "X")

  @property
  def CX(self):
    perm = self.X

    # Sum of X^k ⊗ |k><k|
    O = sum(
      np.kron(
        np.linalg.matrix_power(perm, k),
        Dit(self.d, k).density()
      ) for k in range(self.d)
    )

    return Gate(self.d**2, O, "CX")

  @property
  def Z(self):
    O = np.diag([w(self.d)**i for i in range(self.d)])
    return Gate(self.d, O, "Z")

  @property
  def H(self):
    O = np.zeros((self.d, self.d), dtype=complex)
    for j in range(self.d):
      for k in range(self.d):
        O[j, k] = w(self.d)**(j*k) / np.sqrt(self.d)

    return Gate(self.d, O, "H")

  def Rot(self, thetas: List[complex]):
    R = np.eye(self.d)
    for i, theta in enumerate(thetas):
      R = np.exp(-1j * theta * dGellMann(self.d)[i]) @ R

    return Gate(self.d, R, "Rot")

  @property
  def I(self):
    return Gate(self.d, np.eye(self.d), "I")

# def Layer(g1: Gate, g2: Gate) -> Gate:
#   return np.kron(g1, g2)
def Layer(*args: List[Gate]) -> Gate:
  op = args[0]
  for g in args[1:]:
    op = np.kron(g, op)

  return op
