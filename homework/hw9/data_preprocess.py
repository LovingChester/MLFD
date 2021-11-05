import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math

np.random.seed(10)

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

def gather_data(filenames):
    intensitys = []
    symmetrys = []
    numbers = []
    intensity_one = []
    intensity_other = []
    symmetry_one = []
    symmetry_other = []
    count = 0
    for filename in filenames:
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
            else:
                symmetrys.append(symmetry)
                intensitys.append(intensity)
                intensity_other.append(intensity)
                symmetry_other.append(symmetry)
                numbers.append(-1)
                count += 1
    
    intensitys = np.array(intensitys)
    intensitys = np.transpose(intensitys.reshape(1, -1))
    intensitys_max = np.max(np.abs(intensitys))
    intensitys = intensitys / intensitys_max

    symmetrys = np.array(symmetrys)
    symmetrys = np.transpose(symmetrys.reshape(1, -1))
    symmetrys_max = np.max(np.abs(symmetrys))
    symmetrys = symmetrys / symmetrys_max

    # intensity_one = np.array(intensity_one) / intensitys_max
    # intensity_other = np.array(intensity_other) / intensitys_max

    # symmetry_one = np.array(symmetry_one) / symmetrys_max
    # symmetry_other = np.array(symmetry_other) / symmetrys_max

    # plt.scatter(intensity_one, symmetry_one, c='b', marker='o')
    # plt.scatter(intensity_other, symmetry_other, c='r', marker='x')

    Dx = np.insert(intensitys, [1], symmetrys, axis=1)
    # Dx = np.insert(Dx, 0, count*[1], axis=1)

    Dy = np.array(numbers)
    Dy = np.transpose(Dy.reshape(1, -1))

    # randomly select 300 points for test
    random_indices = np.random.choice(count, size=300, replace=False)
    Dx_train = Dx[random_indices, :]
    Dy_train = Dy[random_indices, :]

    Dx_test = np.delete(Dx, random_indices, axis=0)
    Dy_test = np.delete(Dy, random_indices, axis=0)

    return Dx_train, Dy_train, Dx_test, Dy_test

if __name__ == '__main__':
    gather_data(["ZipDigits.train", "ZipDigits.test"])
    #plt.show()
