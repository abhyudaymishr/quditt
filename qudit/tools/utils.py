from numpy import linalg as LA
import numpy as np


class Space:
    @staticmethod
    def gramSchmidt(vectors: np.ndarray) -> np.ndarray:
        ortho = []
        for v in vectors:
            w = v - sum(np.dot(v, np.conj(u)) * u for u in ortho)
            if LA.norm(w) > 1e-8:
                ortho.append(w / LA.norm(w))

        return np.array(ortho)

    @staticmethod
    def schmidtDecompose(state: np.ndarray) -> list:
        U, D, V = LA.svd(state)
        dims = np.min(state.shape)

        return sorted(
            [(D[k], U[:, k], V.T[:, k]) for k in range(dims)],
            key=lambda dec: dec[0],
            reverse=True,
        )

    @staticmethod
    def schmidtRank(mat: np.ndarray) -> int:
        return LA.matrix_rank(mat)
