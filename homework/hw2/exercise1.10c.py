'''
This program will simulate experiment stated
on problem 1.10 from LFD
@author Houhua Li
'''
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
'''
Each entry corresponds to a coin's flip record
'''

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

ax1.axis([0, 1, 0, 2])
ax1.set_title("v_1")
x1 = np.arange(0,2,0.01)
diff = np.abs(np.array(c_1)-np.array([0.5]*len(c_1)))
x2 = prob_larger(list(x1), list(diff))
ax1.plot(x1, np.array(x2), "m")
tmp_x2 = list(map(lambda x: 2*math.e ** (-2*x ** 2*10), x1))
ax1.plot(np.array(x1), np.array(tmp_x2))

ax2.axis([0, 1, 0, 2])
ax2.set_title("v_rand")
x1 = np.arange(0, 2, 0.01)
diff = np.abs(np.array(c_rand)-np.array([0.5]*len(c_rand)))
x2 = prob_larger(list(x1), list(diff))
ax2.plot(x1, np.array(x2), "m")
tmp_x2 = list(map(lambda x: 2*math.e ** (-2*x ** 2*10), x1))
ax2.plot(np.array(x1), np.array(tmp_x2))

ax3.axis([0, 1, 0, 2])
ax3.set_title("v_min")
x1 = np.arange(0, 2, 0.01)
diff = np.abs(np.array(c_min)-np.array([0.5]*len(c_min)))
x2 = prob_larger(list(x1), list(diff))
ax3.plot(x1, np.array(x2), "m")
tmp_x2 = list(map(lambda x: 2*math.e ** (-2*x ** 2*10), x1))
ax3.plot(np.array(x1), np.array(tmp_x2))

print(time.time()-start)
plt.show()
