import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math

np.set_printoptions(precision=3, suppress=False, threshold=5)

def compute_gradient(Dx, Dy, w):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    grad = np.zeros(col)
    for i in range(row):
        nume = np.sum(Dy[i]) * Dx[i]
        deno = 1 + math.e ** (np.sum(np.matmul(Dx[i], np.transpose(w))) * np.sum(Dy[i]))
        grad += nume / deno
        #print(result)
    return grad / (-1/row)


def logistic_regression(Dx, Dy, w):
    t = 0
    w_prev = None
    row = np.size(Dx, 0)
    while(True):
        g_t = compute_gradient(Dx, Dy, w)
        v_t = -g_t
        w_prev = w.copy()
        w = w + 0.1*v_t
        if np.linalg.norm(w-w_prev) <= 0.001:
            break
        #print(w)
        t += 1
    print("It is being updated for {} times".format(t))
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

Dx = np.insert(x, [1], y, axis=1)
Dx = np.insert(Dx, 0, 2000*[1], axis=1)
#print(Dx)

w = np.zeros(3)

final_w = logistic_regression(Dx, Dy, w)
#print(compute_sum(Dx, Dy_pos, w))

new_x2 = np.array((-final_w[1]/final_w[2])*x+(-final_w[0]/final_w[2]))
plt.plot(x, new_x2, "c")
# plt.annotate("final hypothesis g", xy=(18, 11), xytext=(
#     22, 10), arrowprops=dict(facecolor="c"))
plt.show()
