from matplotlib.colors import ListedColormap
from data_preprocess import *
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

np.set_printoptions(precision=3, suppress=False, threshold=5)

def K_NN(K, Dx, Dy, D_test):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    test_row = np.size(D_test, 0)
    pred = []
    
    dist_matrix = distance_matrix(D_test, Dx)
    sorted_matrix = np.argsort(dist_matrix, axis=1)
    sorted_matrix = sorted_matrix[:, range(K)]
    for i in range(test_row):
        index = sorted_matrix[i, :]
        res = np.sign(np.sum(Dy[index, :]))
        pred.append(res)
    
    pred = np.array(pred).reshape(-1, 1)

    return pred

def compute_E_cv(K, Dx, Dy):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    total_e_cv = 0
    for i in range(row):
        Dx_cv = np.delete(Dx, i, axis=0)
        Dy_cv = np.delete(Dy, i, axis=0)
        pred = K_NN(K, Dx_cv, Dy_cv, Dx[[i], :])
        if np.count_nonzero(pred - Dy[[i], :]) != 0:
            total_e_cv += 1

    return total_e_cv / row

def compute_E_in_test(K, Dx, Dy):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    pred = K_NN(K, Dx, Dy, Dx)

    return np.count_nonzero(pred - Dy) / row

def draw(K, Dx, Dy):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    x1 = np.linspace(-1, 1, num=100)
    x2 = np.linspace(-1, 1, num=100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.insert(X1.reshape(1, -1).reshape(10000, 1),
                  [1], X2.reshape(1, -1).reshape(10000, 1), axis=1)
    result = K_NN(K, Dx, Dy, X)
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
    #row_train, col_train = np.size(Dx_train, 0), np.size(Dx_train, 1)

    K_s = np.arange(1, 300, 2)

    E_cvs = []
    for K in K_s:
        E_cv = compute_E_cv(K, Dx_train, Dy_train)
        E_cvs.append(E_cv)

    plt.title("E_cv vs K")
    plt.xlabel("K")
    plt.ylabel("E_cv")
    plt.plot(K_s, E_cvs, 'b')
    plt.show()
    
    K = 3
    draw(K, Dx_train, Dy_train)
    plt.show()

    E_in = compute_E_in_test(K, Dx_train, Dy_train)
    print("E_in is {:.3f}".format(E_in))
    print("E_cv is {:.3f}".format(E_cvs[np.argwhere(K_s==K)[0][0]]))

    E_test = compute_E_in_test(K, Dx_test, Dy_test)
    print("E_test is {:.5f}".format(E_test))
