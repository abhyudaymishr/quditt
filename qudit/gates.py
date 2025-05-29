from .algebra import Unity, dGellMann
from .index import Gate, Vec
from typing import List
import numpy as np


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
        O[1:, 0 : self.d - 1] = np.eye(self.d - 1)
        return Gate(self.d, O, "X")

    @property
    def CX(self):
        perm = self.X

        # Sum of X^k ⊗ |k><k|
        O = sum(
            np.kron(np.linalg.matrix_power(perm, k), Vec(self.d, k).density())
            for k in range(self.d)
        )

        return Gate(self.d**2, O, "CX")

    @property
    def Z(self):
        w = Unity(self.d)
        O = np.diag([w**i for i in range(self.d)])
        return Gate(self.d, O, "Z")

    @property
    def H(self):
        O = np.zeros((self.d, self.d), dtype=complex)
        w = Unity(self.d)
        for j in range(self.d):
            for k in range(self.d):
                O[j, k] = w**(j * k) / np.sqrt(self.d)

        return Gate(self.d, O, "H")

    def Rot(self, thetas: List[complex]):
        R = np.eye(self.d)
        for i, theta in enumerate(thetas):
            R = np.exp(-1j * theta * dGellMann(self.d)[i]) @ R

        return Gate(self.d, R, "Rot")

    @property
    def I(self):
        return Gate(self.d, np.eye(self.d), "I")
