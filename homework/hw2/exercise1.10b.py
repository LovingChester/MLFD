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
start = time.time()
np.random.seed(100)
for i in range(exp_time):
    flip_record = np.random.randint(0,2, size=(1000,10))
    flip_head = list(map(lambda x : np.count_nonzero(x), flip_record))
    c_1.append(flip_head[0])
    c_rand.append(rd.choice(flip_head))
    c_min.append(min(flip_head))

# print(flip_record)
# print(flip_head)
# print(np.count_nonzero(flip_record[0]))

print(c_1)
print(c_rand)
print(c_min)
#x = np.array([1,1,2,3,4,5,3,4,10])
n_bin = 11
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.axis([0, 10, 0, 25000])
ax1.hist(np.array(c_1), bins=n_bin)
ax2.axis([0, 10, 0, 25000])
ax2.hist(np.array(c_rand), bins=n_bin)
ax3.axis([0, 10, 0, 25000])
ax3.hist(np.array(c_min), bins=n_bin)
print(time.time()-start)
plt.show()
