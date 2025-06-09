from typing import List, Union, Callable
from .utils import Tensor, isVar, ID
from dataclasses import dataclass
from .index import Gate, VarGate
from scipy import sparse as S
from sympy import Matrix
import numpy as np

BARRIER = "─|─"


@dataclass
class Frame:
    dits: List[int]
    name: str
    span: int
    d: int

    def create(gate: Gate):
        return Frame(
            name=gate.name,
            span=gate.span,
            dits=gate.dits,
            d=gate.d,
        )


class Layer:
    vqc: bool = False
    display: List[str]
    gates: List[Frame]
    data: np.ndarray
    span: int
    id: str
    d: int

    def __init__(self, *args: Union[Gate, Callable]):
        if len(args) == 0:
            raise ValueError("Layer must contain at least one gate.")

        gates = list(args)
        span = 0
        self.vqc = isVar(*args)
        self.id = ID()
        for g, gate in enumerate(args):
            if isinstance(gate, (Gate, VarGate)):
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
            gate_data.append(Frame.create(gate))

        self.span = span
        self.d = args[0].d
        self.display = display
        self.data = self.getMat(gates, not self.vqc)
        self.gates = gate_data

    def getMat(self, in_gates, compress=False) -> np.ndarray:
        sublayer = [[]]
        I = Gate(self.d, np.eye(self.d), "I")
        for gate in in_gates:
            span = gate.span
            if span != 1:
                sublayer[0].append(I)

                if span == 0:
                    continue
                sublayer.append(gate)
            else:
                sublayer[0].append(gate)
        sublayer[0] = Tensor(*sublayer[0])

        # REMMEBER TO ADD ONE BECAUSE ENUM IS FROM [1:]
        for s, sub in enumerate(sublayer[1:]):
            dits = sub.dits
            if len(dits) != sub.span:
                dits = [dits[0] + i for i in range(len(dits))]
            # endif

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
                # endif
            # endfor
            sublayer[s + 1] = Tensor(*gates)
        # endfor

        if compress == True:
            for i in range(len(sublayer)):
                sublayer[i] = S.csr_matrix(sublayer[i])

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


class cfn:
    def balance(strings: List[str]) -> List[str]:
        lmax = max(len(s) for s in strings)
        return [s.ljust(lmax, "─") for s in strings]

    def cx(strings: List[str], dits: List[int], name: str = "U") -> List[str]:
        ctrl, targ = dits
        name = name[1:] if name.startswith("C") else name

        if ctrl > targ:
            strings[targ] += f"╭{name}─"
            strings[ctrl] += "╰●─"
            scan = range(targ + 1, ctrl)
        else:
            strings[ctrl] += "╭●─"
            strings[targ] += f"╰{name}─"
            scan = range(ctrl + 1, targ)

        for i in scan:
            strings[i] += "│─"

        return strings


class Circuit:
    layers: List[Layer]
    vqc: bool = False
    span: int
    id: str
    d: int

    def __init__(self, *args: Layer):
        self.layers = list(args)
        self.d = -1
        self.span = -1
        self.id = ID()

        if len(self.layers) > 0:
            span = self.layers[0].span
            for layer in self.layers:
                if layer.vqc:
                    self.vqc = True
                if layer.span != span:
                    raise ValueError(f"Expected {span}, got {layer.span}")

            self.span = span
            self.d = self.layers[0].d

    def solve(self) -> np.ndarray:
        if self.vqc:
            for layer in self.layers:
                if not layer.vqc:
                    layer.data = layer.data.todense()

            prod = Matrix(self.layers[0].data)
            for layer in self.layers[1:]:
                prod = Matrix(layer.data) * prod
        else:
            prod = self.layers[0].data
            for m in self.layers[1:]:
                prod = m.data @ prod

        return prod

    def _draw_penny(self):
        qudits = self.layers[0].span

        strings = ["─"] * qudits
        for l, layer in enumerate(self.layers):
            qctr = 0
            for g, gate in enumerate(layer):
                if gate.span > 1:
                    strings = cfn.balance(strings)
                    strings = cfn.cx(strings, gate.dits, gate.name)
                    qctr += 2
                else:
                    if gate.name == "I" or gate.name == "_":
                        strings[g] += "──"
                    else:
                        strings[g] += f"{gate.name}─"
                    qctr += 1
            # endfor
            strings = cfn.balance(strings)
        # endfor

        return "\n".join(strings)

    def _draw_raw(self):
        p = "Circuit("
        for layer in self.layers:
            if layer[0].name == BARRIER:
                continue
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

    def __setitem__(self, index: int, value: Layer):
        if not isinstance(value, Layer):
            raise TypeError(f"Expected Layer, got {type(value)}")
        if index < 0 or index >= len(self.layers):
            raise IndexError(f"Expected index in [0, {len(self.layers) - 1}], got {index}")
        self.layers[index] = value
        self._refresh()

    def __iter__(self):
        return iter(self.layers)

    def layer(self, *args: Union[Gate, Callable]):
        layer = Layer(*args)
        self.layers.append(layer)
        self._refresh()
        return layer

    def _refresh(self):
      if self.d == -1:
          for layer in self.layers:
              if layer.d > 0:
                  self.d = layer.d
                  break

      if self.span == -1:
        span_sum = 0
        for layer in self.layers:
            if layer.span > 0:
                span_sum += layer.span

        if span_sum > 0:
          self.span = span_sum

      if self.vqc is False:
        for layer in self.layers:
            if layer.vqc:
                self.vqc = True
                break

      if not self.id:
          self.id = ID()


    def barrier(self):
        if len(self.layers) < 1:
            raise ValueError("Add at least 1 layer for a barrier")
        assert self.d > 0, "Dimension Unknown, add a layer first"
        assert self.span > 0, "Span Unknown, add a layer first"

        d = self.d
        args = [Gate(d, np.eye(d), BARRIER)] * self.span
        self.layer(*args)
