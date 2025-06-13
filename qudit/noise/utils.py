from ..index import Gate
from typing import Any
import numpy as np


class Params:
    list: dict[str, float]

    def __init__(self, params: dict[str, Any]):
        self.list = params

    def __getitem__(self, key: str) -> float:
        return self.list[key]

    def __setitem__(self, key: str, value: float):
        self.list[key] = value

    def __iter__(self):
        return iter(self.list.items())

    def __contains__(self, key: str) -> bool:
        return key in self.list


class Error(Gate):
    params: Params

    def __new__(cls, d: int, O: np.ndarray = None, name: str = "Err", params={}):
        obj = super().__new__(cls, d, O, name)
        obj.params = Params(params)
        return obj
