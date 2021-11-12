import numpy as np
import matplotlib.pyplot as plt

Dx = np.array([[1, 0], [0, 1], [0, -1], [-1, 0]\
    ,[0, 2], [0, -2], [-2, 0]])

Dy = [-1, -1, -1, -1, +1, +1, +1]

# create test points
Dx_test = []
r = np.arange(-3, 3, 0.1)
for i in r:
    for j in r:
        Dx_test.append([i, j])

Dx_test = np.array(Dx_test)
row = np.size(Dx_test, 0)

# 1-NN
plt.title("1-NN decision region")
for i in range(row):
    # store the distance
    distances = []
    for j in range(7):
        dist = np.linalg.norm(Dx[j] - Dx_test[i])
        distances.append(dist)
    index = distances.index(min(distances))
    if Dy[index] == 1:
        plt.scatter(Dx_test[i, 0], Dx_test[i, 1], c='b', marker='o')
    else:
        plt.scatter(Dx_test[i, 0], Dx_test[i, 1], c='r', marker='x')

plt.show()

# 3-NN
plt.title("3-NN decision region")
for i in range(row):
    # store the distance
    distances = []
    for j in range(7):
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
        plt.scatter(Dx_test[i, 0], Dx_test[i, 1], c='b', marker='o')
    else:
        plt.scatter(Dx_test[i, 0], Dx_test[i, 1], c='r', marker='x')

plt.show()
