from typing import List, Union, Callable
from dataclasses import dataclass
from .index import Gate, Tensor
from scipy import sparse as S
import numpy as np

@dataclass
class Door:
  dits: List[int]
  name: str
  span: int
  d: int

  def create(gate: Gate):
    return Door(
      name=gate.name,
      span=gate.span,
      dits=gate.dits,
      d=gate.d,
    )

class Layer:
    display: List[str]
    gates: List[Door]
    data: np.ndarray
    span: int
    d: int

    def __init__(self, *args: Union[Gate, Callable]):
        if len(args) == 0:
            raise ValueError("Layer must contain at least one gate.")

        gates = list(args)
        span = 0
        for g, gate in enumerate(args):
            if isinstance(gate, Gate):
                span += gate.span
            elif isinstance(gate, Callable):
                gate = gate(g)
                gates[g] = gate
                span += gate.span
            else:
                raise TypeError(f"Expected Gate, got {type(gate)}")

        display = []
        gate_data = []
        for g, gate in enumerate(gates):
            display.append(f"{gate.name}({gate.dits if gate.dits else g})")
            if len(gate.dits) == 0:
              gate.dits = [g]
            gate_data.append(Door.create(gate))

        self.span = span
        self.d = args[0].d
        self.display = display
        self.data = self.getMat(gates)
        self.gates = gate_data

    def getMat(self, in_gates) -> np.ndarray:
        sublayer = [[]]
        I = np.eye(self.d)
        for gate in in_gates:
            span = gate.span
            if span != 1:
                sublayer[0].append(I)

                if span == 0:
                    continue
                sublayer.append(gate)
            else:
                sublayer[0].append(gate)
        sublayer[0] = S.csr_matrix(Tensor(*sublayer[0]))

        # REMMEBER TO ADD ONE BECAUSE ENUM IS FROM [1:]
        for s, sub in enumerate(sublayer[1:]):
            dits = sub.dits
            [lq, mq] = [min(dits), max(dits)]

            isDone = False
            gates = []
            for i in range(self.span):
                if i < lq or i > mq:
                    gates.append(I)
                else:
                    if isDone:
                      continue
                    gates.append(sub)
                    isDone = True

            sublayer[s + 1] = S.csr_matrix(Tensor(*gates))

        prod = sublayer[0]
        for sub in sublayer[1:]:
          prod = prod @ sub

        return prod

    def __repr__(self):
        return f"Layer({', '.join(self.display)})"

    def __getitem__(self, index):
        return self.gates[index]

    def __iter__(self):
        return iter(self.gates)


class Circuit:
    layers: List[Layer]
    span: int
    d: int

    def __init__(self, *args: Layer):
        self.layers = list(args)

        if len(self.layers) > 0:
            span = self.layers[0].span
            for layer in self.layers:
                if layer.span != span:
                    raise ValueError(f"Expected {span}, got {layer.span}")

            self.span = span

    def solve(self) -> np.ndarray:
        prod = self.layers[0].data
        for m in self.layers[1:]:
            prod = m.data @ prod

        return prod

    def _draw_penny(self):
        qudits = self.layers[0].span

        strings = ["─"] * qudits
        for l, layer in enumerate(self.layers):
            qctr = 0
            for gate in layer:
                print(f"{gate=} {qctr=}")
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
            p += f"\n  {layer},"
        p += "\n)"
        return p

    def draw(self, output: str = "raw"):
        if output == "penny":
            return self._draw_penny()
        elif output == "raw":
            return self._draw_raw()
        else:
            raise ValueError(f"Unknown output format: {output}")

    def __repr__(self):
        return self._draw_raw()

    def __getitem__(self, index):
        return self.layers[index]

    def __iter__(self):
        return iter(self.layers)

    def layer(self, *args: Union[Gate, Callable]):
        layer = Layer(*args)
        self.layers.append(layer)
        return layer
