import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=False, threshold=5)
np.random.seed(1314)

# generate data set
D = np.random.uniform([0, 0], [1, 1], size=(10000, 2))
#print(D)
# randomly select the first center
indexs = np.random.choice(10000, 1)
centers = D[indexs, :]

# compute rest of the centers
for i in range(9):
    row = np.size(centers, 0)
    total_dists = []
    for j in range(10000):
        distances = []
        for k in range(row):
            dist = np.linalg.norm(D[j] - centers[k])
            distances.append(dist)
        total_dists.append(min(distances))
    index = total_dists.index(max(total_dists))
    centers = np.append(centers, D[[index], :], axis=0)

# plot the unupdated center
# plt.plot(np.transpose(D)[0], np.transpose(D)[1], 'bo')
# plt.plot(np.transpose(centers)[0], np.transpose(centers)[1], 'rx')
# plt.show()

# create the query data
D_query = np.random.rand(10000, 2)

# Brute force to find NN
# store the NN for the ith point in D_query
# NN_point = []
# for i in range(10000):
#     distances = []
#     for j in range(10000):
#         dist = np.linalg.norm(D[j] - D_query[i])
#         distances.append(dist)
#     index = distances.index(min(distances))
#     NN_point.append(D[index])

# branch and bound to find NN
# store the center index and its correcponding cluster
center_cluster = dict()
for i in range(10):
    center_cluster[i] = []

for i in range(10000):
    distances = []
    for j in range(10):
        dist = np.linalg.norm(centers[j] - D[i])
        distances.append(dist)
    index = distances.index(min(distances))
    center_cluster[index].append(D[i])

# update centers
for i in range(10):
    center = sum(center_cluster[i]) / len(center_cluster[i])
    centers[i] = center

# plot the updated centers
# plt.plot(np.transpose(D)[0], np.transpose(D)[1], 'bo')
# plt.plot(np.transpose(centers)[0], np.transpose(centers)[1], 'rx')
# plt.show()
