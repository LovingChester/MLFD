import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math

np.set_printoptions(precision=3, suppress=False, threshold=5)

# linear regression for classification
def linear_regression(Dx, Dy):
    inv = np.linalg.pinv(np.matmul(np.transpose(Dx), Dx))
    x_plus = np.matmul(inv, np.transpose(Dx))
    w = np.matmul(x_plus, Dy)
    return w

'''
Compute the average intensity of the grayscale given
Input: grayscale pixel
return: the average intensity of the greyscale
'''
def average_intensity(grayscale):
    neg_ones = np.full((16, 16), -1, dtype=float)
    tmp = grayscale - neg_ones
    #print(tmp)
    non_zero = np.count_nonzero(tmp)
    return non_zero / 256

'''
Compute the symmetry of the grayscale given
Input: grayscale pixel
return: the symmetry of the greyscale
'''
def symmetric_score(grayscale):

    horizontal_flip = np.fliplr(grayscale)
    vertical_flip = np.flipud(grayscale)
    horizontal_diff = grayscale - horizontal_flip
    vertical_diff = grayscale - vertical_flip

    horizontal_zero = 256 - np.count_nonzero(horizontal_diff)
    vertical_zero = 256 - np.count_nonzero(vertical_diff)
    avg = (horizontal_zero + vertical_zero) / 2
    return avg / 256

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
    Dx = np.insert(Dx, 0, count*[1], axis=1)

    final_w = linear_regression(Dx, Dy)
    new_x2 = np.array((-final_w[1]/final_w[2])*intensitys+(-final_w[0]/final_w[2]))
    plt.plot(intensitys, new_x2, "c")
    
    return Dx, Dy, final_w, count

if __name__ == '__main__':
    gather_data("ZipDigits.train")
    plt.show()
    gather_data("ZipDigits.test")
    plt.show()
