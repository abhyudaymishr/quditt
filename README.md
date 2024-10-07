<img src="./assets/icon.svg" width="75" height="75" align="right">

### Qudit
Simulations for Qudit systems

[![PyPI version](https://badge.fury.io/py/qudit.svg)](https://pypi.org/project/qudit/)

## Usage
After `pip install qudit`, you can use the package as follows:


### Usage
Two qubit example: bell state

```python
from qudit import *
D = DGate(2) # create gateset for 2-dits

# p = |00><00|
X0 = Psi(Dit(2, 0), Dit(2, 0)).density()

H_layer = Layer(D.H, D.I) # H x I

rho = UpU(H_layer, X0) # H x I p (H x I)^H
rho = UpU(D.CX, rho) # CX p (CX)^H
print(rho)

# Tr(|11><11| p)
prob = Tr(np.array(Psi(2, "11").density().dot(rho)))
print(prob) # 0.5
```

Three Qutrit example: 3-GHZ state = $\frac{1}{\sqrt{3}}(|000\rangle + |111\rangle + |222\rangle)$
```python
from qudit import *

P000 = Psi(Dit(3, 0), Dit(3, 0), Dit(3, 0))
P111 = Psi(Dit(3, 1), Dit(3, 1), Dit(3, 1))
P222 = Psi(Dit(3, 2), Dit(3, 2), Dit(3, 2))

GHZ = Dit(P000, P111, P222)

# alternative way to create GHZ state
P000 = Psi(3, "000")
P111 = Psi(3, "111")
P222 = Psi(3, "222")

GHZ = Dit(P000, P111, P222)

print(GHZ.density())
```