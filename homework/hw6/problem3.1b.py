import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math

np.set_printoptions(precision=3, suppress=False, threshold=5)

def linear_regression(Dx, Dy):
    inv = np.linalg.pinv(np.matmul(np.transpose(Dx), Dx))
    x_plus = np.matmul(inv, np.transpose(Dx))
    w = np.matmul(x_plus, Dy)
    return w

rad = 10
thk = 5
sep = 5

#np.random.seed(5)
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
Dy = []
x = []
y = []
for i in range(2000):
    if(float(angles[i]) > math.pi):
        pos_x.append(float(x1[i]) + rad + thk/2)
        pos_y.append(float(x2[i]) - sep)
        x.append(float(x1[i]) + rad + thk/2)
        y.append(float(x2[i]) - sep)
        Dy.append(1)
    else:
        neg_x.append(float(x1[i]))
        neg_y.append(float(x2[i]))
        x.append(float(x1[i]))
        y.append(float(x2[i]))
        Dy.append(-1)

plt.plot(pos_x, pos_y, "bo")
plt.plot(neg_x, neg_y, "ro")

#x = x1.copy()

x = np.array(x)
x = x.reshape(1,-1)
x = x.reshape(2000,1)

y = np.array(y)
y = y.reshape(1,-1)
y = y.reshape(2000,1)

Dy = np.array(Dy)
Dy = Dy.reshape(1,-1)
Dy = Dy.reshape(2000,1)

Dx = np.insert(x, [1], y, axis=1)
Dx = np.insert(Dx, 0, 2000*[1], axis=1)
print(Dy)

w = np.zeros(3)

final_w = linear_regression(Dx, Dy)
#print(compute_sum(Dx, Dy_pos, w))

new_x2 = np.array((-final_w[1]/final_w[2])*x+(-final_w[0]/final_w[2]))
plt.plot(x, new_x2, "c")
# plt.annotate("final hypothesis g", xy=(18, 11), xytext=(
#     22, 10), arrowprops=dict(facecolor="c"))
plt.show()
