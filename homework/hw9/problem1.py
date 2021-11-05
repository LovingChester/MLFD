import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
from data_preprocess import *

np.set_printoptions(precision=3, suppress=False, threshold=5)

# compute the legendre polynomial
def Leg_poly(x, poly, row):
    if poly == 0:
        return np.ones((row, 1))
    elif poly == 1:
        return x
    else:
        return ((2*poly-1)/poly) * x * Leg_poly(x, poly-1, row) - \
            ((poly-1)/poly) * Leg_poly(x, poly-2, row)

Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])
print(Dx_test)


