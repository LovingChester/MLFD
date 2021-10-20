import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
from sklearn.preprocessing import PolynomialFeatures
from problem1a import *

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

    poly = PolynomialFeatures(3)
    Dx = poly.fit_transform(Dx)

    final_w = linear_regression(Dx, Dy)
    # new_x2 = np.array((-final_w[1]/final_w[2]) *
    #                   intensitys+(-final_w[0]/final_w[2]))
    # plt.plot(intensitys, new_x2, "c")

    return Dx, Dy, final_w, count


if __name__ == '__main__':
    Dx, Dy, final_w, count = gather_data("ZipDigits.train")
    print(final_w)

