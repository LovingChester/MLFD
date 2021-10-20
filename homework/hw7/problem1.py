import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math

np.set_printoptions(precision=3, suppress=False, threshold=5)

def linear_regression(Dx, Dy):
    inv = np.linalg.pinv(np.matmul(np.transpose(Dx), Dx))
    x_plus = np.matmul(inv, np.transpose(Dx))
    w = np.matmul(x_plus, Dy)
    return w


