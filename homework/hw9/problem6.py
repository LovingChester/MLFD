import numpy as np
import matplotlib.pyplot as plt
from data_preprocess import *
from problem1 import *
from problem2 import *

Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])

Zx_train = poly_transform(Dx_train[:, [0]], Dx_train[:, [1]])
Zy_train = np.copy(Dy_train)

Zx_test = poly_transform(Dx_test[:, [0]], Dx_test[:, [1]])
Zy_test = np.copy(Dy_test)

# has regularization
w_reg = linear_regression(Zx_train, Zy_train, 0.01)
row = np.size(Zx_test, 0)
y_pred = np.matmul(Zx_test, w_reg)
one = np.ones((row, 1))
e = np.full((row, 1), np.e)

E_test = np.sum(np.log(one + e ** (-Zy_test * y_pred))) / row

print("{}".format(E_test))
