import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
from data_preprocess import *
from problem1 import *

def linear_regression(Dx, Dy, alpha):
    col = np.size(Dx, 1)
    inv = np.linalg.pinv(np.matmul(np.transpose(Dx), Dx) + alpha * np.identity(col))
    x_plus = np.matmul(inv, np.transpose(Dx))
    w = np.matmul(x_plus, Dy)
    return w

def plot_points(Dx, Dy):
    row = np.size(Dx, 0)
    for i in range(row):
        if Dy[i,0] == 1:
            plt.scatter(Dx[i, 0], Dx[i, 1], c='b', marker='o')
        else:
            plt.scatter(Dx[i, 0], Dx[i, 1], c='r', marker='x')

if __name__ == '__main__':
    Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])
    print(Dx_train)
    plt.xlabel("intensity")
    plt.ylabel("symmetry")
    plt.title("No regularization")
    plot_points(Dx_train, Dy_train)

    Zx_train = poly_transform(Dx_train[:,[0]], Dx_train[:,[1]])
    Zy_train = np.copy(Dy_train)

    # no regularization
    w_reg = linear_regression(Zx_train, Zy_train, 0)

    x1 = np.linspace(-1, 1, num=100)
    x2 = np.linspace(-1, 1, num=100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.insert(X1.reshape(1, -1).reshape(10000, 1), [1], X2.reshape(1, -1).reshape(10000, 1), axis=1)
    Z_X = poly_transform(X[:,[0]], X[:, [1]])
    result = np.matmul(Z_X, w_reg)
    result = np.reshape(result, np.shape(X1))
    plt.contour(X1, X2, result, 1)
    plt.show()
