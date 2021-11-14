import numpy as np
import matplotlib.pyplot as plt
import time

np.set_printoptions(precision=3, suppress=False, threshold=5)
#np.random.seed(13)

# generate data set
centers = np.random.normal([0.5, 0.5], 0.1, (10, 2))
#print(centers)

#center_cluster = dict()
D = np.random.normal(centers[0], 0.1, (1000, 2))
plt.plot(np.transpose(D)[0], np.transpose(D)[1], 'bo')
#center_cluster[0] = D
for i in range(1, 10):
    data = np.random.normal(centers[i], 0.1, (1000, 2))
    plt.plot(np.transpose(data)[0], np.transpose(data)[1], 'bo')
    #center_cluster[i] = data
    D = np.append(D, data, axis=0)

plt.plot(np.transpose(centers)[0], np.transpose(centers)[1], 'rx')
#plt.show()
# create the query data
D_query = np.random.normal([0.5, 0.5], 0.1, (10000, 2))

# Brute force to find NN
# store the NN for the ith point in D_query
start = time.time()
NN_point = []
for i in range(10000):
    distances = []
    for j in range(10000):
        dist = np.linalg.norm(D[j] - D_query[i])
        distances.append(dist)
    index = distances.index(min(distances))
    NN_point.append(D[index])
end = time.time()
print("Brute force time: {}".format(end-start))


# branch and bound to find NN
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
plt.plot(np.transpose(centers)[0], np.transpose(centers)[1], 'rx')
plt.show()
# compute the radius for each cluster
radius = []
for i in range(10):
    distances = []
    for j in range(len(center_cluster[i])):
        dist = np.linalg.norm(center_cluster[i][j] - centers[i])
        distances.append(dist)
    radius.append(max(distances))

start = time.time()
for i in range(10000):
    distances = []
    for j in range(10):
        dist = np.linalg.norm(centers[j] - D_query[i])
        distances.append(dist)
    index = distances.index(min(distances))
    for j in range(len(center_cluster[index])):
        is_NN = True
        for k in range(10):
            lhs = np.linalg.norm(center_cluster[index][j] - D_query[i])
            rhs = np.linalg.norm(centers[k] - D_query[i]) - radius[k]
            if k != index and lhs > rhs:
                is_NN = False
                break
        if is_NN:
            break
end = time.time()
print("Branch and bound time: {}".format(end-start))
