from typing import List, Union
import numpy.linalg as LA
import numpy as np
import math as ma

"""
  B = Basis(3) will create a qutrit basis
  so B("111") will return vec for |111>
  or B(1, 2, 0) will return vec for |120>
"""


class Basis:
    d: int = None
    span: int = None

    def __init__(self, d: int):
        self.d = d

    def __call__(self, *args: Union[List[int], str]) -> "Vec":
        if len(args) == 1 and isinstance(args[0], str):
            args = [int(i) for i in args[0]]

        basis = np.eye(self.d, dtype=np.complex128)
        prod = 1
        for ket in args:
            if ket < 0 or ket >= self.d:
                raise ValueError(f"Index {ket} out of bounds for dimension {self.d}")
            prod = np.kron(prod, basis[ket])

        return Vec(prod)


class Density(np.ndarray):
    def __new__(cls, d: Union[np.ndarray, List[List[Union[complex, float]]]]):
        obj = np.asarray(d, dtype=np.complex128).view(cls)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if obj.ndim != 2 or obj.shape[0] != obj.shape[1]:
            raise ValueError("Density matrix must be square")

    def __xor__(self, other: "Density") -> "Density":
        return Density(np.kron(self, other))

    @property
    def trace(self) -> float:
        return np.trace(self).real

    @property
    def d(self):
        return self.shape[0]

    def norm(self) -> "Density":
        return Density(self / self.trace)

    @property
    def H(self) -> "Density":
        return Density(self.conj().T)

    def proj(self) -> "Density":
        evals, evecs = LA.eig(self)
        matrix = sum(
            [
                Vec(evecs[:, i]).density()
                for i in range(len(evals))
                if np.abs(evals[i]) > 1e-8
            ]
        )

        return Density(matrix)

    def oproj(self) -> "Density":
        proj = self.proj()
        perp = np.eye(proj.shape[0]) - proj

        return Density(perp)


class Vec(np.ndarray):
    def __new__(cls, d: Union[np.ndarray, List[Union[complex, float]]]):
        obj = np.asarray(d, dtype=np.complex128).view(cls)
        obj /= np.linalg.norm(obj)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __xor__(self, other: "Vec") -> "Density":
        return Vec(np.kron(self, other))

    @property
    def d(self):
        return len(self)

    def density(self) -> np.ndarray:
        return Density(np.outer(self, self.conj().T))

    def norm(self) -> "Vec":
        return Vec(self / np.linalg.norm(self))

    @property
    def H(self) -> "Vec":
        return Vec(self.conj().T)


class Gate(np.ndarray):
    span: int
    d: int
    name: str = ""
    dits: List[int]

    def __new__(cls, d: int, O: np.ndarray = None, name: str = None):
        if O is None:
            obj = np.zeros((d, d), dtype=complex).view(cls)
            obj.span = 1
        else:
            obj = np.asarray(O, dtype=complex).view(cls)
            obj.span = int(ma.log(len(O[0]), d))
        # endif

        obj.name = name if name else f"Gate({d})"
        obj.d = d
        obj.dits = []

        return obj

    @property
    def H(self):
        return self.conj().T

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.d = getattr(obj, "d", 0)
        self.span = getattr(obj, "span", 0)
        self.name = getattr(obj, "name", "Gate")
        self.dits = getattr(obj, "dits", [])

    def is_unitary(self):
        return np.allclose(self @ self.H, np.eye(self.shape[0]))

    def is_hermitian(self):
        return np.allclose(self, self.H)


# <A|b@c@d@e...@n|B>
def braket(*args: np.ndarray) -> np.ndarray:
    if len(args) < 2:
        raise ValueError("At least two arguments are required for Bracket")

    args = list(args)
    args[-1] = args[-1].conj().T
    result = args[0]
    for arg in args[1:]:
        result = np.dot(result, arg)

    return result


# A ^ B ^ C ^ D ^ ... ^ N
def Tensor(*args: Union[Gate, Vec]) -> np.ndarray:
    if len(args) < 2:
        raise ValueError("At least two args needed")

    result = args[0]
    for arg in args[1:]:
        result = np.kron(result, arg)

    if result.ndim == 2:
        d = result.d
        return Gate(d, result, name="AggreGate")
        # since X, H, CNOT are not longer valid names
    else:
        return result
