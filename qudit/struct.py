from typing import List, Union
from .index import Gate
import numpy as np

class Layer:
    def __init__(self, *args: List[Gate]):
        self.gates = list(args)
        span = 0
        for gate in args:
            if isinstance(gate, Gate):
                span += gate.span
            else:
                raise TypeError(f"Expected Gate, got {type(gate)}")
        self.span = span
        self.d = args[0].d

    def __repr__(self):
        return f"Layer({', '.join(gate.name for gate in self.gates)})"

    def __getitem__(self, index):
        return self.gates[index]

    def _setitem__(self, index, value):
        self.gates[index] = value

    def __len__(self):
        return len(self.gates)

    def __iter__(self):
        return iter(self.gates)

    def append(self, gate: Gate):
        self.gates.append(gate)


class Circuit:
    def __init__(self, *args: List[Union[Layer, np.ndarray]]):
        self.layers = list(args)
        self.isSparse = False

        if len(self.layers) > 0:
            span = self.layers[0].span
            for layer in self.layers:
                if layer.span != span:
                    raise ValueError(f"Expected {span}, got {layer.span}")

            self.span = span

    def solve(self) -> np.ndarray:
        if self.isSparse == False:
            self.make_sparse()
        assert not isinstance(self.layers[0], Layer)

        prod = self.layers[0]
        for m in self.layers[1:]:
            prod = m @ prod

        return prod

    def make_sparse(self):
        from scipy import sparse

        matrices = []
        for layer in self.layers:
            prod = 1
            for gate in layer:
                prod = np.kron(prod, gate)

            prod = sparse.csr_matrix(prod)
            matrices.append(prod)

        self.layers = matrices
        self.isSparse = True

    def _draw_penny(self):
        qudits = sum(gate.span for gate in self.layers[0])

        strings = ["─"] * qudits
        for l, layer in enumerate(self.layers):
            qctr = 0
            for gate in layer:
                if gate.span == 2:
                    strings[qctr] += f"╭●─"
                    strings[qctr + 1] += f"╰{gate.name}─"
                    qctr += 2
                else:
                    if gate.name == "I":
                        strings[qctr] += "──"
                    else:
                        strings[qctr] += f"{gate.name}─"
                    qctr += 1
            # endfor
            lmax = max(len(s) for s in strings)
            strings = [s.ljust(lmax, "─") for s in strings]
        # endfor

        return "\n".join(strings)

    def _draw_raw(self):
        p = "Circuit("
        for layer in self.layers:
          if self.isSparse:
            p += f"\n  Sparse({layer.shape}),"
          else:
            p += f"\n  {layer},"
        p += "\n)"
        return p

    def draw(self, output: str = "raw"):
        if output == "penny":
            if self.isSparse:
                raise ValueError("Cannot draw sparse circuit in pennylane format.")
            return self._draw_penny()
        elif output == "raw":
            return self._draw_raw()
        else:
            raise ValueError(f"Unknown output format: {output}")

    def __repr__(self):
        return self._draw_raw()

    def __getitem__(self, index):
        return self.layers[index]

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def append(self, layer: Union[Layer, np.ndarray]):
        self.layers.append(layer)
