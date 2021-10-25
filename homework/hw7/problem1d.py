from typing import final
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
from sklearn.preprocessing import PolynomialFeatures
from sympy.core import symbol
from problem1a import *
from sympy.solvers import solve
from sympy import Symbol
from sympy.abc import x, y

np.set_printoptions(precision=3, suppress=False, threshold=5)

def gather_data(filename):
    intensitys = []
    symmetrys = []
    numbers = []
    intensity_one = []
    intensity_five = []
    symmetry_one = []
    symmetry_five = []
    count = 0
    for items in open(filename):
        item_list = items.strip().split(' ')
        item_list = list(map(lambda x: float(x), item_list))
        number = item_list.pop(0)
        grayscale = np.array(item_list)
        grayscale = np.reshape(grayscale, (16, 16))
        intensity = average_intensity(grayscale)
        symmetry = symmetric_score(grayscale)
        if number == 1.0:
            symmetrys.append(symmetry)
            intensitys.append(intensity)
            intensity_one.append(intensity)
            symmetry_one.append(symmetry)
            numbers.append(1)
            count += 1
        elif number == 5.0:
            symmetrys.append(symmetry)
            intensitys.append(intensity)
            intensity_five.append(intensity)
            symmetry_five.append(symmetry)
            numbers.append(-1)
            count += 1

    if(filename.split('.')[1] == "train"):
        plt.title("ZipDigit train")
    else:
        plt.title("ZipDigit test")

    plt.xlabel("average intensity")
    plt.ylabel("symmetry")
    plt.plot(intensity_one, symmetry_one, 'bo')
    plt.plot(intensity_five, symmetry_five, 'rx')

    intensitys = np.array(intensitys)
    intensitys = intensitys.reshape(1, -1)
    intensitys = intensitys.reshape(count, 1)

    symmetrys = np.array(symmetrys)
    symmetrys = symmetrys.reshape(1, -1)
    symmetrys = symmetrys.reshape(count, -1)

    Dy = np.array(numbers)
    Dy = Dy.reshape(1, -1)
    Dy = Dy.reshape(count, 1)

    Dx = np.insert(intensitys, [1], symmetrys, axis=1)

    # perform transform feature
    poly = PolynomialFeatures(3)
    Dx = poly.fit_transform(Dx)

    if filename.split('.')[1] != 'train':
        return intensitys, Dx, Dy, count

    final_w = linear_regression(Dx, Dy)

    return Dx, Dy, final_w, count


if __name__ == '__main__':
    Dx, Dy, final_w, count = gather_data("ZipDigits.train")
    E_in = np.linalg.norm(np.matmul(Dx, final_w) - Dy) ** 2 / count
    print("E_in is: {:.3f}".format(E_in))
    
    x1 = np.linspace(0, 1, num=100)
    x2 = np.linspace(0, 1, num=100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.insert(X1.reshape(1, -1).reshape(10000, 1), [1], X2.reshape(1, -1).reshape(10000, 1), axis=1)
    poly = PolynomialFeatures(3)
    X_poly = poly.fit_transform(X)
    result = np.matmul(X_poly, final_w)
    result = np.reshape(result, np.shape(X1))
    print(result)
    plt.contour(X1, X2, result, 1)
    plt.show()

    intensitys, Dx, Dy, count = gather_data("ZipDigits.test")
    E_test = np.linalg.norm(np.matmul(Dx, final_w) - Dy) ** 2 / count
    print("E_test is: {:.3f}".format(E_test))
    plt.contour(X1, X2, result, 1)
    plt.show()

