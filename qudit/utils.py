from sympy.physics.quantum import TensorProduct
from sympy import SparseMatrix as Matrix
from .index import Gate, State, VarGate
from typing import Union
from uuid import uuid4
import numpy as np


def ID() -> str:
    return str(uuid4()).split("-")[0]


def isVar(*args) -> bool:
    for arg in args:
        if isinstance(arg, (VarGate, Matrix)):
            return True
    return False


# <A|b@c@d@e...@n|B>
def Braket(*args: np.ndarray) -> np.ndarray:
    if len(args) < 2:
        raise ValueError("At least two arguments are required for Braket")

    args = list(args)
    args[-1] = args[-1].conj().T
    result = args[0]
    for arg in args[1:]:
        result = np.dot(result, arg)

    return result


# A ^ B ^ C ^ D ^ ... ^ N
def Tensor(*args: Union[Gate, State]) -> np.ndarray:
    if len(args) == 0:
        raise ValueError("At least one arg needed")
    if len(args) == 1:
        return args[0]

    names = [args[0].name] if isinstance(args[0], (Gate, VarGate)) else ["?"]
    result = args[-1]
    for arg in args[:-1]:
        if isinstance(arg, Gate):
            result = np.kron(result, arg)
            names.append(arg.name)
        elif isinstance(arg, VarGate):
            result = TensorProduct(result, arg)
            names.append(arg.name)
        else:
            result = np.kron(result, arg)
            names.append("?")

    if result.ndim == 2:
        d = result.d
        if isVar(*args):
            return VarGate(d, result, name=".".join(names))
        else:
            return Gate(d, result, name=".".join(names))
            # since X, H, CNOT are not longer valid names
    else:
        return result
