import numpy as np
import matplotlib.pyplot as plt

rad = 10
thk = 5
sep = 5

np.set_printoptions(precision=3, suppress=False, threshold=5)

np.random.seed(5)
'''
generate random radius and angle
'''
rad_angle = np.random.uniform([rad, 0], [rad+thk+1, 2*np.pi], size=(2000, 2))
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
#print(x1)
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
    if(float(angles[i]) > np.pi):
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

x = np.array(x)
x = x.reshape(1, -1)
x = x.reshape(2000, 1)

y = np.array(y)
y = y.reshape(1, -1)
y = y.reshape(2000, 1)

Dx = np.insert(x, [1], y, axis=1)

#plt.show()

# create test points
Dx_test = []
r = np.arange(-15, 40, 1)
for i in r:
    for j in r:
        Dx_test.append([i, j])

Dx_test = np.array(Dx_test)
row = np.size(Dx_test, 0)

# 1-NN
# plt.title("1-NN decision region")
# for i in range(row):
#     # store the distance
#     distances = []
#     for j in range(2000):
#         dist = np.linalg.norm(Dx[j] - Dx_test[i])
#         distances.append(dist)
#     index = distances.index(min(distances))
#     if Dy[index] == 1:
#         plt.scatter(Dx_test[i, 0], Dx_test[i, 1], c='#4169E1', marker='o')
#     else:
#         plt.scatter(Dx_test[i, 0], Dx_test[i, 1], c='#FF6347', marker='o')

# plt.show()

# 3-NN
plt.title("3-NN decision region")
for i in range(row):
    # store the distance
    distances = []
    for j in range(2000):
        dist = np.linalg.norm(Dx[j] - Dx_test[i])
        distances.append(dist)
    tmp_dist = distances.copy()
    distances.sort()
    distances = distances[: 3]
    total = 0
    for dist in distances:
        total += Dy[tmp_dist.index(dist)]
    res = np.sign(total)
    if res == 1:
        plt.scatter(Dx_test[i, 0], Dx_test[i, 1], c='#4169E1', marker='o')
    else:
        plt.scatter(Dx_test[i, 0], Dx_test[i, 1], c='#FF6347', marker='o')

plt.show()
