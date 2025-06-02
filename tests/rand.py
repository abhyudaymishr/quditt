import sys

sys.path.append("..")
import qudit.random as rand
import numpy as np

print(rand.unitary(5))
print("-" * 20)
print(rand.state(5))
