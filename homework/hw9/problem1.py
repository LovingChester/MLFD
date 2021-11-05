import numpy as np
from data_preprocess import *

np.set_printoptions(precision=3, suppress=False, threshold=5)

# compute the legendre polynomial
def Leg_poly(x, poly):
    row = np.size(x, 0)
    if poly == 0:
        return np.ones((row, 1))
    elif poly == 1:
        return x
    else:
        return ((2*poly-1)/poly) * x * Leg_poly(x, poly-1) - \
            ((poly-1)/poly) * Leg_poly(x, poly-2)

# 8th order legendre polynomial transform
def poly_transform(x1, x2):
    row = np.size(x1, 0)
    Z = np.ones((row, 1))
    for poly in range(1,9):
        i = poly
        j = 0
        while(i >= 0):
            term = Leg_poly(x1, i) * Leg_poly(x2, j)
            Z = np.append(Z, term, axis=1)
            i -= 1
            j += 1
    return Z

if __name__ == '__main__':
    Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])
    print(Dx_train)

    # Z = np.append(Leg_poly(Dx_train[:, [0]], 0, 300), Leg_poly(Dx_train[:, [0]], 1, 300))
    Z_train = poly_transform(Dx_train[:,[0]], Dx_train[:,[1]])
    Z_test = poly_transform(Dx_test[:,[0]], Dx_test[:,[1]])
    print(np.size(Z_train,1))
