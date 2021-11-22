from data_preprocess import *
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

np.set_printoptions(precision=3, suppress=False, threshold=5)

def get_centers(K, Dx):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    #indexs = np.random.choice(row, K, False)
    indexs = np.random.choice(row, 1)
    centers = Dx[indexs, :]

    #compute rest of the centers
    for i in range(K-1):
        dist_matrix = distance_matrix(Dx, centers)
        sorted_dist = np.sort(dist_matrix, axis=1)
        min_dist = sorted_dist[:, 0]
        index = np.argmax(min_dist)
        centers = np.append(centers, Dx[[index], :], axis=0)
    
    center_cluster = dict()

    for i in range(K):
        center_cluster[i] = []
    
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
    Z = transform(K, Dx, centers)

    # compute regression for classifiction followed by pocket
    inv = np.linalg.pinv(np.matmul(np.transpose(Z), Z))
    x_plus = np.matmul(inv, np.transpose(Z))
    w = np.matmul(x_plus, Dy)

    t = 0
    while t < 100:
        # run PLA
        res = np.sign(np.matmul(Z, w))
        diff = res - Dy
        index = np.where(diff != 0)[0]
        #mis = select_missclassify(list(np.transpose(diff)[0]))
        if len(index) == 0: break
        mis = np.random.choice(index)
        tmp_w = w + Dy[mis][0] * np.transpose(Z[[mis], :])
        new_diff = np.sign(np.matmul(Z, tmp_w)) - Dy
        # if the new w can classify the point better, update it
        if np.count_nonzero(new_diff) < np.count_nonzero(diff):
            w = tmp_w
        t = t + 1

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

def compute_E_in_test(K, Dx, Dy, centers, w):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    Z_Dx = transform(K, Dx, centers)
    E = np.count_nonzero(np.sign(np.matmul(Z_Dx, w)) - Dy) / row
    return E

def draw(K, Dx, Dy, centers):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    x1 = np.linspace(-1, 1, num=100)
    x2 = np.linspace(-1, 1, num=100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.insert(X1.reshape(1, -1).reshape(10000, 1),
                  [1], X2.reshape(1, -1).reshape(10000, 1), axis=1)
    Z_X = transform(K, X, centers)
    w = RBF(K, Dx, Dy, centers)
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
    centers_K = []
    for K in K_s:
        centers = get_centers(K, Dx_train)
        E_cv = compute_E_cv(K, Dx_train, Dy_train, centers)
        #print("K = {}, E_cv = {}".format(K, E_cv))
        E_cvs.append(E_cv)
        centers_K.append(centers)
    
    plt.title("E_cv vs K")
    plt.xlabel("K")
    plt.ylabel("E_cv")
    plt.plot(K_s, E_cvs, 'b')
    plt.show()

    K = 16
    centers = centers_K[np.argwhere(K_s == K)[0][0]]
    draw(K, Dx_train, Dy_train, centers)
    plt.show()

    w = RBF(K, Dx_train, Dy_train, centers)
    E_in = compute_E_in_test(K, Dx_train, Dy_train, centers, w)
    print("E_in is {:.3f}".format(E_in))
    print("E_cv is {:.3f}".format(E_cvs[np.argwhere(K_s == K)[0][0]]))

    E_test = compute_E_in_test(K, Dx_test, Dy_test, centers, w)
    print("E_test is {:.5f}".format(E_test))
