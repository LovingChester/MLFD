import numpy as np
import matplotlib.pyplot as plt
from data_preprocess import *
from problem1 import *
from problem2 import *

# compute the hat matrix
def compute_hat(Dx, alpha):
    col = np.size(Dx, 1)
    inv = np.linalg.pinv(np.matmul(np.transpose(Dx), Dx) + alpha * np.identity(col))
    H = np.matmul(Dx, inv)
    H = np.matmul(H, np.transpose(Dx))
    return H

def compute_E_cv(Zx_train, Zy_train, alpha):
    w_reg = linear_regression(Zx_train, Zy_train, alpha)
    y_pred = np.matmul(Zx_train, w_reg)

    H = compute_hat(Zx_train, alpha)
    H_diag = np.diag(H).reshape(-1, 1)
    row = np.size(H_diag, 0)
    one = np.ones((row, 1))

    E_cv = np.sum(((y_pred - Zy_train) / (one - H_diag)) ** 2) / row
    return E_cv


def compute_E_test(Zx_train, Zy_train, Zx_test, Zy_test, alpha):
    row = np.size(Zx_test, 0)
    w_reg = linear_regression(Zx_train, Zy_train, alpha)
    y_pred = np.matmul(Zx_test, w_reg)
    E_test = np.sum((y_pred - Zy_test) ** 2) / row

    return E_test

Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])

Zx_train = poly_transform(Dx_train[:, [0]], Dx_train[:, [1]])
Zy_train = np.copy(Dy_train)

Zx_test = poly_transform(Dx_test[:, [0]], Dx_test[:, [1]])
Zy_test = np.copy(Dy_test)

#plt.axis([0, 1, 0, 0.2])
plt.title("E_cv and E_test vs lambda")
plt.xlabel("lambda")
plt.ylabel("error")

alphas = list(np.arange(0, 2.01, 0.01))
E_cvs = []
for alpha in alphas:
    E_cvs.append(compute_E_cv(Zx_train, Zy_train, alpha))
print(E_cvs.index(min(E_cvs)))
print(alphas[8])
plt.plot(alphas, E_cvs)

E_tests = []
for alpha in alphas:
    E_tests.append(compute_E_test(Zx_train, Zy_train, Zx_test, Zy_test, alpha))

plt.plot(alphas, E_tests)
plt.show()
