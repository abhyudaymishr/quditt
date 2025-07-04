from scipy.sparse import csr_matrix, kron
from .algebra import Unity, dGellMann
from typing import List, Union
from .index import Gate, Basis
from functools import cache
import numpy.linalg as LA
import numpy as np

ck = 21


class Swapper:
    def __init__(self, d: int, width: int, swap: Gate, I: Gate):
        self.d = d
        self.width = width
        self.swap = csr_matrix(swap)
        self.I = csr_matrix(I)

        swaps = []
        for i in range(self.width - 1):
            op = None
            for j in range(self.width - 1):
                gate = self.swap if j == i else self.I
                op = gate if op is None else kron(op, gate, format="csr")
            swaps.append(op)

        self.S = swaps

    @cache
    def get(self, a: int, b: int) -> csr_matrix:
        if a > b:
            a, b = b, a

        icheck = 0 <= a < self.width and 0 <= b < self.width and a != b
        assert icheck, f"Invalid indices a: ({a},{b}), in: {self.width}"

        prod = self.S[a]
        for i in range(a + 1, b):
            prod = self.S[i] @ prod

        return prod


class Gategen:
    d: int
    Ket: Basis
    swapper: Union[Swapper, None]

    def __init__(self, d: int):
        self.d = d
        self.Ket = Basis(d)
        self.swapper = None

    def create(self, O: np.ndarray = None, name: str = "U"):
        return Gate(self.d, O, name)

    @property
    def X(self) -> Gate:
        O = np.zeros((self.d, self.d))
        O[0, self.d - 1] = 1
        O[1:, 0 : self.d - 1] = np.eye(self.d - 1)
        return Gate(self.d, O, "X")

    @property
    def Y(self) -> Gate:
        O = np.zeros((self.d, self.d), dtype=complex)
        O[0, self.d - 1] = 1j
        O[1:, 0 : self.d - 1] = np.eye(self.d - 1)
        return Gate(self.d, O, "Y")

    @property
    def Z(self) -> Gate:
        w = Unity(self.d)
        O = np.diag([w**i for i in range(self.d)])
        return Gate(self.d, O, "Z")

    def CU(self, U: Gate, rev=False) -> Gate:
        """
        CU = Σ_k U^k ⊗ |k><k| (target, ctrl)
        CU = Σ_k |k><k| ⊗ U^k (ctrl, target)

        for everything else we insert I
        Eg: CU(1, 4) = Σ_k |k><k| ⊗ I ⊗ I ⊗ U^k
        """

        F = lambda k: [LA.matrix_power(U, k), self.Ket(k).density()]
        if rev:
            F = lambda k: [self.Ket(k).density(), LA.matrix_power(U, k)]

        gate = [np.kron(*F(k)) for k in range(self.d)]

        name = U.name if U.name else "U"
        gate = Gate(self.d, sum(gate), "C" + name)
        gate.span = 2

        return gate

    @property
    def CX(self) -> Gate:
        return self.CU(self.X, False)

    @property
    def CY(self) -> Gate:
        return self.CU(self.Y, False)

    @property
    def CZ(self) -> Gate:
        return self.CU(self.Z, False)

    @property
    def swap(self) -> Gate:
        cx, xc = self.CU(self.X, False), self.CU(self.X, True)
        return Gate(self.d, cx @ xc @ cx, "sw")

    def long_swap(self, a: int, b: int, width: int) -> Gate:
        sw = self.swap
        if a == b:
            return Gate(self.d, np.eye(self.d**width), f"SWAP({a}, {b})[{width}]")
        if a > b:
            a, b = b, a

        assert 0 <= a <= width, f"Invalid index a: {a}, width: {width}"
        assert 0 <= b <= width, f"Invalid index b: {b}, width: {width}"

        if self.swapper is None:
            self.swapper = Swapper(self.d, width, sw, self.I)

        mat = self.swapper.get(a, b)
        mat.name = f"SWAP({a}, {b})[{width}]"
        # mat = mat.todense()
        return mat

    @property
    def S(self):
        w = Unity(self.d)
        O = np.diag([w**j for j in range(self.d)])
        return Gate(self.d, O, "S")

    @property
    def T(self):
        w = Unity(self.d * 2)
        O = np.diag([w**j for j in range(self.d)])
        return Gate(self.d, O, "T")

    def P(self, theta: float):
        w = Unity(self.d * 2)
        O = np.diag([w**j for j in range(self.d)])
        return Gate(self.d, O, f"P({theta:.2f})")

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
