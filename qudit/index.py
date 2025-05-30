from typing import List, Union
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


class Vec(np.ndarray):
    def __new__(cls, d: Union[np.ndarray, List[complex]]):
        obj = np.asarray(d, dtype=np.complex128).view(cls)
        obj /= np.linalg.norm(obj)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def d(self):
        return len(self)

    def density(self) -> np.ndarray:
        return np.outer(self, self.conj().T)

    def norm(self) -> "Vec":
        return Vec(self / np.linalg.norm(self))

    def H(self) -> "Vec":
        return Vec(self.conj().T)


class Gate(np.ndarray):
    span: int
    d: int
    name: str = ""

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

        return obj

    @property
    def H(self):
        return self.conj().T

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.d = getattr(obj, "d", None)
        self.span = getattr(obj, "span", None)
        self.name = getattr(obj, "name", None)

    def is_unitary(self):
        return np.allclose(self @ self.H, np.eye(self.shape[0]))

    def is_hermitian(self):
        return np.allclose(self, self.H)


def braket(*args: np.ndarray) -> np.ndarray:
    if len(args) < 2:
        raise ValueError("At least two arguments are required for Bracket")

    args[-1] = args[-1].conj().T
    result = args[0]
    for arg in args[1:]:
        result = np.dot(result, arg)

    return result


def Tensor(*args: Union[Gate, Vec]) -> np.ndarray:
    if len(args) < 2:
        raise ValueError("At least two args needed")

    result = args[0]
    for arg in args[1:]:
        result = np.kron(result, arg)

    return result
