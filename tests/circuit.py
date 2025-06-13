import sys

sys.path.append("..")
from unittest import TestCase, main
from qudit import Gategen, Circuit
import numpy as np


class Circuits(TestCase):
    def test_bell(self):
        HCX = np.array(
            [[1, 1, 0.0, 0], [0.0, 0, 1, -1], [0.0, 0, 1, 1], [1, -1, 0.0, 0]]
        ) / np.sqrt(2)

        D, C = Gategen(2), Circuit(2)
        C.gate(D.H, dits=[0])
        C.gate(D.CX, dits=[0, 1])

        U = C.solve().todense()

        self.assertTrue(np.allclose(U, HCX, atol=1e-4))


if __name__ == "__main__":
    main()
