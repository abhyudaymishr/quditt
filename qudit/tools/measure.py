from scipy.linalg import logm
import numpy as np
from qudit.tools.metrics import Entropy


class Distance:
    @staticmethod
    def relative_entropy(
        rho: np.ndarray, sigma: np.ndarray, base: float = 2.0
    ) -> float:
        rho = Entropy.density_matrix(rho)
        sigma = Entropy.density_matrix(sigma)

        eps = 1e-12
        rho += eps * np.eye(rho.shape[0])
        sigma += eps * np.eye(sigma.shape[0])

        log_rho = logm(rho)
        log_sigma = logm(sigma)
        delta_log = log_rho - log_sigma

        result = np.trace(rho @ delta_log).real  #ensured
        return float(result / np.log(base))
