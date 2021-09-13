'''
This program will simulate experiment stated
on problem 1.10 from LFD
@author Houhua Li
'''

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import math


def prob_larger(x1, diff):
    count = 0
    x2 = []
    for i in range(len(x1)):
        for j in range(len(diff)):
            if diff[j] > x1[i]:
                count += 1
        x2.append(count/len(diff))
        count = 0
    return x2

n, p = 6, .5
max_res = []
np.random.seed(10)
for i in range(10000):
    result = np.random.binomial(n, p, 2)
    max_res.append(max(list(result / np.array([6]*2) - np.array([0.5]*2))))

#print(result / np.array([6]*2))

x1 = np.arange(0,2,0.01)

plt.axis([0, 1, 0, 2])
plt.title("problem 1.7")
x2 = list(map(lambda x: 4*math.e ** (-12*x ** 2), x1))
plt.plot(x1, np.array(x2))

x2 = prob_larger(list(x1), list(max_res))
plt.plot(x1, np.array(x2), 'm')

plt.show()
