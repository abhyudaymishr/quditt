from .algebra import Unity, dGellMann
from .index import Gate, Vec, Basis
from typing import List
import numpy as np


ck = 23


# special class to create "d" once and pass through all gates
# so G = DGate(d) -> G.X -> G.Z -> G.H -> ...
class Gategen:
    def __init__(self, d: int):
        self.d = d
        self.Ket = Basis(d)

    @property
    def X(self) -> Gate:
        O = np.zeros((self.d, self.d))
        O[0, self.d - 1] = 1
        O[1:, 0 : self.d - 1] = np.eye(self.d - 1)
        return Gate(self.d, O, "X")

    @property
    def CX(self) -> Gate:
        perm = self.X

        # Sum of X^k ⊗ |k><k|
        O = sum(
            np.kron(np.linalg.matrix_power(perm, k), self.Ket(k).density())
            for k in range(self.d)
        )

        return Gate(self.d, O, "CX")

    @property
    def Z(self) -> Gate:
        w = Unity(self.d)
        O = np.diag([w**i for i in range(self.d)])
        return Gate(self.d, O, "Z")

    @property
    def H(self) -> Gate:
        O = np.zeros((self.d, self.d), dtype=complex)
        w = Unity(self.d)
        for j in range(self.d):
            for k in range(self.d):
                O[j, k] = w ** (j * k) / np.sqrt(self.d)

        return Gate(self.d, O, "H")

    def Rot(self, thetas: List[complex]) -> Gate:
        R = np.eye(self.d)
        for i, theta in enumerate(thetas):
            R = np.exp(-1j * theta * dGellMann(self.d)[i]) @ R

        return Gate(self.d, R, "Rot")

    @property
    def I(self) -> Gate:
        return Gate(self.d, np.eye(self.d), "I")
