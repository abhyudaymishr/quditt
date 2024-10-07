from .qudit import *

def Qubit_Bell() -> Dit:
  D = DGate(2)
  X0 = Psi(Dit(2, 0), Dit(2, 0)).density()

  H_layer = Layer(D.H, D.I)

  rho = UpU(H_layer, X0)
  rho = UpU(D.CX, rho)
  print(rho)

  for i in range(2):
    for j in range(2):
      prob = Tr(np.array(Psi(Dit(2, i), Dit(2, j)).density().dot(rho)))
      print(f"Proj{i}{j} -> {prob:.2f}")

  return rho

Qubit_Bell()

# test with d=2,3 to get back paulis and gell-mann matrices
print("Pauli matrices")
print(dGellMann(2))
print("Gell-Mann matrices")
gm = dGellMann(3)