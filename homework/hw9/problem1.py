import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
from data_preprocess import *

np.set_printoptions(precision=3, suppress=False, threshold=5)

def Legendre_polynomials():

    return

Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])

print(Dx_test)
