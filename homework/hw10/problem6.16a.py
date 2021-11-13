import numpy as np
import matplotlib.pyplot as plt

#np.set_printoptions(precision=3, suppress=False, threshold=5)
np.random.seed(1314)

# generate data set
D = np.random.uniform([0, 0], [1, 1], size=(10000, 2))

# randomly select the first center
indexs = np.random.choice(10000, 1)
centers = D[indexs, :]

#plt.plot(np.transpose(D)[0], np.transpose(D)[1], 'bo')

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

print(centers)

plt.plot(np.transpose(centers)[0], np.transpose(centers)[1], 'ro')
plt.show()


