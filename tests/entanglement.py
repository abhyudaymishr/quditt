import sys

sys.path.append("..")

from qudit.tools import *

N = 10
R_MAX = 2**7
THETA = 1.5
d = 3 # dim of QuDit 2
D = 5 # Qubit + Qutrit
r = 2 # level of entanglement

Rn = 3
basis = np.eye(Rn)

def PSI(i):
  A = State.create(State.Ket_0, basis[ i % Rn ])
  B = State.create(State.Ket_1, basis[ (i+1) % Rn ])

  return State.combine([A, B], [np.cos(THETA/2), np.sin(THETA/2)])

subspace_basis = np.array([PSI(i) for i in range(2)])
# subspace_basis = GramSchmidt(subspace_basis)

proj, proj_perp = Projector(basis=subspace_basis)

size = (2*D + 1)*(r - 1)

def phi_ik(X):
  lmda = X[0]
  qubit = Qdit(2, X[1:5].reshape(2, 2))
  qutrit = Qdit(3, X[5:11].reshape(3, 2))

  return lmda, qubit, qutrit

def f(X):
  L, qbit, qtrit = phi_ik(X)
  PHI_rx = normalise(L * np.kron(qbit, qtrit))

  return Loss(PHI_rx, proj_perp)

res = minima(f, rand(R_MAX, size), tries=2)
print(f'E_r(θ={THETA:.2f}) = {res:.4f}')
