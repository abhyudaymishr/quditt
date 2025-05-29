import sys

sys.path.append("..")
from qudit import *


def gellMann():
    gm = dGellMann(3)
    count = 0
    for mat in gm:
        print(f"{count}" + "=" * 20)
        print(mat)
        count += 1
