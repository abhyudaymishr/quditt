from scipy.linalg import fractional_matrix_power
from typing import List, Union
from .. import Dit, Psi, In
import numpy as np


class Fidelity:
    def __new__(rho: np.ndarray, sigma: np.ndarray) -> float:
        return Fidelity.default(rho, sigma)

    def default(rho: np.ndarray, sigma: np.ndarray) -> float:
        if rho.ndim == 1 and sigma.ndim == 1:
            return float(np.abs(np.vdot(rho, sigma)) ** 2)

        if rho.ndim == 1:
            rho = np.outer(rho, rho.conj())
        if sigma.ndim == 1:
            sigma = np.outer(sigma, sigma.conj())

        sqrt_rho = fractional_matrix_power(rho, 0.5)
        inner = sqrt_rho @ sigma @ sqrt_rho
        fidelity = np.trace(fractional_matrix_power(inner, 0.5))
        return float(np.real(fidelity))

    def channel(
        kraus: List[Union[np.ndarray, List[float]]], rho: np.ndarray
    ) -> np.ndarray:
        if rho.ndim == 1:
            rho = np.outer(rho, rho.conj())

        d_1, d_2 = kraus[0].shape
        if rho.shape != (d_2, d_2):
            raise ValueError(
                f"Incompatible shape: expected {(d_2, d_2)}, got {rho.shape}"
            )

        rho_out = np.zeros((d_1, d_1), dtype=complex)
        for K in kraus:
            rho_out += K @ rho @ K.conj().T

        return rho_out

    # TODO: is ndim enough? or do we need to check for square?
    def entanglement(rho: np.ndarray, kraus_ops: List[np.ndarray]) -> float:
        d = rho.shape[0]
        assert rho.shape == (d, d), "rho must be a square matrix"
        for K in kraus_ops:
            assert K.shape == (d, d), "Each Kraus operator must be of shape (d, d)"

        F_e = 0.0
        for K in kraus_ops:
            term = np.trace(rho @ K.conj().T @ K @ rho)
            F_e += np.real(term)

        return F_e


def partial_transpose(rho, dim_A, dim_B):

    rho = rho.reshape((dim_A, dim_B, dim_A, dim_B))
    rho_pt = np.transpose(rho, (0, 3, 2, 1))
    return rho_pt.reshape((dim_A * dim_B, dim_A * dim_B))


def negativity(rho, dim_A, dim_B):

    rho_pt = partial_transpose(rho, dim_A, dim_B)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return np.sum(np.abs(eigenvalues[eigenvalues < 0]))


class Entropy:
    def __new__(cls, *args):
        return Entropy.default(*args)

    @staticmethod
    def default(*args):
        pass

    @staticmethod
    def entanglement():
        pass

    @staticmethod
    def tsallis():
        pass

    @staticmethod
    def shannon():
        pass

    @staticmethod
    def renyi():
        pass

    @staticmethod
    def hartley():
        pass

    @staticmethod
    def neumann():
        pass

    @staticmethod
    def unified():
        pass
