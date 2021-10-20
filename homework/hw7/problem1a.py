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


