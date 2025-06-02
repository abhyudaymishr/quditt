from scipy.optimize import minimize
import numpy.linalg as LA
import numpy as np
from . import GramSchmidt

dagger = lambda x: np.conj(x).T

ROUNDOFF_TOL = 1e-6


normalise = lambda phi: phi / np.linalg.norm(phi)
dagger = lambda x: np.conj(x).T


def Qdit(order, coeffs):
    if len(coeffs) == order * 2:
        # for [a,b,a,b] case
        psis = [coeffs[i] + 1j * coeffs[i + 1] for i in range(0, len(coeffs), 2)]
    else:
        # for [[a,b][a,b]] case
        psis = [c[0] + 1j * c[1] for c in coeffs]

    return normalise(np.array(psis))


def Projector(basis, gs=False):
    if gs:
        basis = GramSchmidt(basis)

    projector = np.array(
        [np.outer(basis[i], dagger(basis[i])) for i in range(len(basis))]
    )

    perp = np.eye(len(basis[0])) - sum(projector)
    return projector, perp


class State:
    Ket_0 = np.array([1, 0])
    Ket_1 = np.array([0, 1])
    Ket_p = normalise(np.array([1, 1]))
    Ket_m = normalise(np.array([1, -1]))
    Ket_i = normalise(np.array([1, 1j]))
    Ket_mi = normalise(np.array([1, -1j]))

    def create(state, basis_vec):
        return np.kron(state, basis_vec)

    def combine(states, coeffs):
        if len(states) != len(coeffs):
            raise ValueError("states and coeffs must have the same length")
        return normalise(sum([coeffs[i] * states[i] for i in range(len(states))]))

def Loss(phi_rx, projector):
    left = dagger(phi_rx)
    right = phi_rx

    prod = np.dot(np.dot(left, projector), right)

    return np.real_if_close(prod, tol=ROUNDOFF_TOL)


def minima(f, x0, **kwargs):
    if "method" not in kwargs:
        kwargs["method"] = "L-BFGS-B"

    if "tol" not in kwargs:
        kwargs["tol"] = ROUNDOFF_TOL

    if "tries" in kwargs:
        tries = kwargs["tries"]
        del kwargs["tries"]
    else:
        tries = 1

    minimas = np.ones(tries)
    for i in range(tries):
        try:
            minimas[i] = minimize(f, x0=x0, **kwargs).fun
        except Exception as e:
            pass

    return np.min(minimas)
