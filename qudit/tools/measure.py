from scipy.linalg import logm
import numpy as np
from qudit.tools.metrics import Entropy,Fidelity


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
    @staticmethod
    def bures(rho: np.ndarray, sigma: np.ndarray) -> float:
        
        rho = Entropy.density_matrix(rho)
        sigma = Entropy.density_matrix(sigma)
        
        bures_distance = np.sqrt(
            2 - 2*(Fidelity.default(rho, sigma))**0.5)
        
        return float(bures_distance)
    @staticmethod
    def jensen_shannon(
        rho: np.ndarray, sigma: np.ndarray, base: float = 2.0
    ) -> float:
        rho = Entropy.density_matrix(rho)
        sigma = Entropy.density_matrix(sigma)

        m = 0.5 * (rho + sigma)
        return 0.5 * (Distance.relative_entropy(rho, m, base) + Distance.relative_entropy(sigma, m, base))
    @staticmethod
    def trace_distance(
        rho: np.ndarray, sigma: np.ndarray
    ) -> float:
        rho = Entropy.density_matrix(rho)
        sigma = Entropy.density_matrix(sigma)

        return 0.5 * np.trace(np.abs(rho - sigma)).real
    @staticmethod
    def bhattacharyya(
        rho: np.ndarray, sigma: np.ndarray, base: float = 2.0
    ) -> float:
        rho = Entropy.density_matrix(rho)
        sigma = Entropy.density_matrix(sigma)

        
        return Fidelity.default(rho, sigma)