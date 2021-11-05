import numpy as np
import matplotlib.pyplot as plt
from data_preprocess import *
from problem1 import *
from problem2 import *

Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])

plt.xlabel("intensity")
plt.ylabel("symmetry")
plt.title("Has regularization: alpha = 2")
plot_points(Dx_train, Dy_train)

Zx_train = poly_transform(Dx_train[:,[0]], Dx_train[:,[1]])
Zy_train = np.copy(Dy_train)

# has regularization
w_reg = linear_regression(Zx_train, Zy_train, 2)

x1 = np.linspace(0, 1, num=100)
x2 = np.linspace(0, 1, num=100)
X1, X2 = np.meshgrid(x1, x2)
X = np.insert(X1.reshape(1, -1).reshape(10000, 1), [1], X2.reshape(1, -1).reshape(10000, 1), axis=1)
Z_X = poly_transform(X[:,[0]], X[:, [1]])
result = np.matmul(Z_X, w_reg)
result = np.reshape(result, np.shape(X1))
plt.contour(X1, X2, result, 1)
plt.show()
