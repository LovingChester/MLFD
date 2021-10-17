import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math

np.set_printoptions(precision=3, suppress=False, threshold=5)

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

intensity_one = []
intensity_five = []
symmetry_one = []
symmetry_five = []
for items in open("ZipDigits.train"):
    item_list = items.strip().split(' ')
    item_list = list(map(lambda x: float(x), item_list))
    number = item_list.pop(0)
    if number == 1.0:
        grayscale = np.array(item_list)
        grayscale = np.reshape(grayscale, (16, 16))
        intensity_one.append(average_intensity(grayscale))
        symmetry_one.append(symmetric_score(grayscale))
    elif number == 5.0:
        grayscale = np.array(item_list)
        grayscale = np.reshape(grayscale, (16, 16))
        intensity_five.append(average_intensity(grayscale))
        symmetry_five.append(symmetric_score(grayscale))

plt.title("ZipDigit train")
plt.xlabel("average intensity")
plt.ylabel("symmetry")
plt.plot(intensity_one, symmetry_one, 'bo')
plt.plot(intensity_five, symmetry_five, 'rx')
plt.show()

intensity_one.clear()
symmetry_one.clear()
intensity_five.clear()
symmetry_five.clear()

for items in open("ZipDigits.test"):
    item_list = items.strip().split(' ')
    item_list = list(map(lambda x: float(x), item_list))
    number = item_list.pop(0)
    if number == 1.0:
        grayscale = np.array(item_list)
        grayscale = np.reshape(grayscale, (16, 16))
        intensity_one.append(average_intensity(grayscale))
        symmetry_one.append(symmetric_score(grayscale))
    elif number == 5.0:
        grayscale = np.array(item_list)
        grayscale = np.reshape(grayscale, (16, 16))
        intensity_five.append(average_intensity(grayscale))
        symmetry_five.append(symmetric_score(grayscale))

plt.title("ZipDigit test")
plt.xlabel("average intensity")
plt.ylabel("symmetry")
plt.plot(intensity_one, symmetry_one, 'bo')
plt.plot(intensity_five, symmetry_five, 'rx')
plt.show()
