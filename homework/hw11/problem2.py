from data_preprocess import *
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

np.random.seed(10)

def get_centers(K, Dx):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    indexs = np.random.choice(row, 1)
    centers = Dx[indexs, :]

    # compute rest of the centers
    for i in range(K-1):
        center_num = np.size(centers, 0)
        total_dists = []
        for j in range(10000):
            distances = []
            for k in range(center_num):
                dist = np.linalg.norm(Dx[j] - centers[k])
                distances.append(dist)
            total_dists.append(min(distances))
        index = total_dists.index(max(total_dists))
        centers = np.append(centers, Dx[[index], :], axis=0)
    
    center_cluster = dict()

    for i in range(10):
        center_cluster[i] = []

    for i in range(10000):
        distances = []
        for j in range(10):
            dist = np.linalg.norm(centers[j] - Dx[i])
            distances.append(dist)
        index = distances.index(min(distances))
        center_cluster[index].append(Dx[i])

    # update centers
    for i in range(10):
        center = sum(center_cluster[i]) / len(center_cluster[i])
        centers[i] = center

    return centers

def RBF(K, Dx, Dy):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    centers = get_centers(K, Dx)
    

    return

if __name__ == '__main__':
    Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])
