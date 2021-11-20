from data_preprocess import *
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

np.random.seed(10)
np.set_printoptions(precision=3, suppress=False, threshold=5)

def get_centers(K, Dx):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    indexs = np.random.choice(row, 1)
    centers = Dx[indexs, :]

    # compute rest of the centers
    for i in range(K-1):
        # center_num = np.size(centers, 0)
        # total_dists = []
        # for j in range(row):
        #     distances = []
        #     for k in range(center_num):
        #         dist = np.linalg.norm(Dx[j] - centers[k])
        #         distances.append(dist)
        #     total_dists.append(min(distances))
        dist_matrix = distance_matrix(Dx, centers)
        sorted_dist = np.sort(dist_matrix, axis=1)
        min_dist = sorted_dist[:, 0]
        index = np.argmax(min_dist)
        centers = np.append(centers, Dx[[index], :], axis=0)
    
    center_cluster = dict()

    for i in range(K):
        center_cluster[i] = []

    # for i in range(row):
    #     distances = []
    #     for j in range(K):
    #         dist = np.linalg.norm(centers[j] - Dx[i])
    #         distances.append(dist)
    #     index = distances.index(min(distances))
    #     center_cluster[index].append(Dx[i])
    
    dist_matrix = distance_matrix(Dx, centers)
    min_index = np.argmin(dist_matrix, axis=1)
    for i in range(row):
        center_cluster[min_index[i]].append(Dx[i])

    # update centers
    for i in range(K):
        center = sum(center_cluster[i]) / len(center_cluster[i])
        centers[i] = center

    return centers

def transform(K, Dx, centers):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    r = 2 / np.sqrt(K)
    Z = distance_matrix(Dx, centers) / r
    Z = np.insert(Z, 0, row*[1], axis=1)

    return Z

def RBF(K, Dx, Dy, centers):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    #centers = get_centers(K, Dx)
    r = 2 / np.sqrt(K)
    Z = distance_matrix(Dx, centers) / r
    Z = np.insert(Z, 0, row*[1], axis=1)

    # compute regression for classifiction
    inv = np.linalg.pinv(np.matmul(np.transpose(Z), Z))
    x_plus = np.matmul(inv, np.transpose(Z))
    w = np.matmul(x_plus, Dy)

    return w

def compute_E_cv(K, Dx, Dy, centers):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    total_e_cv = 0
    for i in range(row):
        Dx_cv = np.delete(Dx, i, axis=0)
        Dy_cv = np.delete(Dy, i, axis=0)
        w = RBF(K, Dx_cv, Dy_cv, centers)
        Z_Dx = transform(K, Dx[[i], :], centers)
        if np.count_nonzero(np.sign(np.matmul(Z_Dx, w)) - Dy[[i], :]) != 0:
            total_e_cv += 1

    return total_e_cv / row

def draw(K, Dx, Dy, centers):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    x1 = np.linspace(-1, 1, num=100)
    x2 = np.linspace(-1, 1, num=100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.insert(X1.reshape(1, -1).reshape(10000, 1),
                  [1], X2.reshape(1, -1).reshape(10000, 1), axis=1)
    Z_X = transform(K, X, centers)
    w = RBF(K, Dx_train, Dy_train, centers)
    result = np.matmul(Z_X, w)
    result = np.reshape(result, np.shape(X1))
    #plt.pcolormesh(X1, X2, result, cmap=ListedColormap(['#FFAAAA', '#AAFFAA']))
    plt.title("K = {}".format(K))
    plt.xlabel("intensity")
    plt.ylabel("symmetry")
    plt.contour(X1, X2, result, 0)

    for i in range(row):
        if Dy[i] == 1:
            plt.scatter(Dx[i, 0], Dx[i, 1], c='b', marker='o')
        else:
            plt.scatter(Dx[i, 0], Dx[i, 1], c='r', marker='x')

    return

if __name__ == '__main__':
    Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])

    K_s = np.arange(1, 60, 1)

    E_cvs = []
    for K in K_s:
        centers = get_centers(K, Dx_train)
        E_cv = compute_E_cv(K, Dx_train, Dy_train, centers)
        #print("K = {}, E_cv = {}".format(K, E_cv))
        E_cvs.append(E_cv)
    
    plt.title("E_cv vs K")
    plt.xlabel("K")
    plt.ylabel("E_cv")
    plt.plot(K_s, E_cvs, 'b')
    plt.show()

    K = 16
    centers = get_centers(K, Dx_train)
    draw(K, Dx_train, Dy_train, centers)
    plt.show()
