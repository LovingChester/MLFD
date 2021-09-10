'''
This program will simulate experiment stated
on problem 1.10 from LFD
@author Houhua Li
'''
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import time
'''
Each entry corresponds to a coin's flip record
'''
exp_time = 100000

c_1 = []
c_rand = []
c_min = []
'''
create 1000 x 10 numpy array where 1000 represents
1000 coins and 10 represents 10 flip
'''
n, p = 10, .5
start = time.time()
np.random.seed(100)
for i in range(exp_time):
    flip_record = np.random.binomial(n, p, 1000)
    c_1.append(flip_record[0]/10)
    c_rand.append(rd.choice(flip_record)/10)
    c_min.append(np.amin(flip_record)/10)

n_bin = 11
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.axis([0, 1, 0, 25000])
ax1.set_title("c_1")
ax1.hist(np.array(c_1), bins=n_bin)
ax2.axis([0, 1, 0, 25000])
ax2.set_title("c_rand")
ax2.hist(np.array(c_rand), bins=n_bin)
ax3.axis([0, 1, 0, 25000])
ax3.set_title("c_min")
ax3.hist(np.array(c_min), bins=n_bin)
print(time.time()-start)
plt.show()
