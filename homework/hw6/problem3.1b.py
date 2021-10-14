import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math



rad = 10
thk = 5
sep = 5

np.set_printoptions(precision=3, suppress=False, threshold=5)

np.random.seed(10)
'''
generate random radius and angle
'''
rad_angle = np.random.uniform([rad, 0], [rad+thk+1, 2*math.pi], size=(2000, 2))
#print(rad_angle)

# define the center of the circle
center = np.random.uniform(5, 15, size=(1, 2))
#print(center[0])
radius = rad_angle[:, [0]]
angles = rad_angle[:, [1]]

x1 = radius * np.cos(angles)
x2 = radius * np.sin(angles)

x1 = x1 + center[0][0]
x2 = x2 + center[0][1]

'''
positive will store the data points which has +1
negative will store the data points which has -1
'''
pos_x = []
pos_y = []
neg_x = []
neg_y = []
Dy_pos = []
Dy_neg = []
for i in range(2000):
    if(float(angles[i]) > math.pi):
        pos_x.append(float(x1[i]) + rad + thk/2)
        pos_y.append(float(x2[i]) - sep)
        Dy_pos.append(1)
    else:
        neg_x.append(float(x1[i]))
        neg_y.append(float(x2[i]))
        Dy_neg.append(-1)

plt.plot(pos_x, pos_y, "bo")
plt.plot(neg_x, neg_y, "ro")

pos_x.extend(neg_x)
x = pos_x.copy()
pos_x = np.array(pos_x)
pos_x = pos_x.reshape(1, -1)
pos_x = pos_x.reshape(2000, 1)
pos_y.extend(neg_y)
pos_y = np.array(pos_y)
pos_y = pos_y.reshape(1, -1)
pos_y = pos_y.reshape(2000, 1)
Dy_pos.extend(Dy_neg)

Dx = np.insert(pos_x, [1], pos_y, axis=1)
Dx = np.insert(Dx, 0, 2000*[1], axis=1)
#print(Dx)

w = np.zeros(3)
