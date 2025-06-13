from scipy.special._comb import _comb_int as nCr
from .utils import Error
import numpy as np


class GAD:
    @staticmethod
    def A(order: int, d: int, Y: float):
        obj = np.zeros((d, d), dtype=np.complex128)
        for r in range(order, d):
            obj[r - order][r] = np.sqrt(nCr(r, order) * (1 - Y) ** r - order * Y**order)

        return Error(d, obj, f"A{order}", {"Y": Y, "order": order})

    @staticmethod
    def R(order: int, d: int, Y: float):
        obj = np.zeros((d, d), dtype=np.complex128)
        for r in range(d - order):
            obj[r + order][r] = np.sqrt(
                nCr(d - r - 1, order) * (1 - Y) ** (d - r - order - 1) * Y**order
            )

        return Error(d, obj, f"R{order}", {"Y": Y, "order": order})


class Pauli:
    @staticmethod
    def X(d: int, Y: float):
        obj = np.zeros((d, d), dtype=np.complex128)
        for r in range(d):
            obj[r][(r + 1) % d] = np.sqrt(Y)
            obj[r][(r - 1) % d] = np.sqrt(1 - Y)

        return Error(d, obj, "X", {"Y": Y})

    @staticmethod
    def Y(d: int, Y: float):
        obj = np.zeros((d, d), dtype=np.complex128)
        for r in range(d):
            obj[r][(r + 1) % d] = np.sqrt(Y) * 1j
            obj[r][(r - 1) % d] = np.sqrt(1 - Y) * 1j

        return Error(d, obj, "Y", {"Y": Y})

    @staticmethod
    def Z(d: int, Y: float):
        obj = np.zeros((d, d), dtype=np.complex128)
        for r in range(d):
            obj[r][r] = np.sqrt(Y)
            obj[r][(r + 1) % d] = np.sqrt(1 - Y)

        return Error(d, obj, "Z", {"Y": Y})
