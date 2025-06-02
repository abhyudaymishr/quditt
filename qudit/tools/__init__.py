from .entanglement import *
from .metrics import *
from .tests import *


def GramSchmidt(vectors):
    ortho = []
    for v in vectors:
        w = v - sum(np.dot(v, np.conj(u)) * u for u in ortho)
        if LA.norm(w) > 1e-8:
            ortho.append(w / LA.norm(w))

    return np.array(ortho)


# schmidt_decomposition
def schmidt_decomposition():
    pass


# schmidt_rank = no of non-zero singular values
def schmidt_rank():
    pass


# entanglement_entropy = entropy of sq of schmidt coefficients
def entanglement_entropy():
    pass
