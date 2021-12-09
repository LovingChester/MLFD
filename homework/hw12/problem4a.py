from numpy import double
from data_preprocess import *
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

np.set_printoptions(precision=3, suppress=False, threshold=5)

def get_kernel(D):
    K = np.matmul(D, np.transpose(D))
    K = (1 + K) ** 8
    return K

def SVM(Dx, Dy, C):
    K = get_kernel(Dx)
    Q = np.outer(Dy, np.transpose(Dy)) * K
    row, col = np.size(Q, 0), np.size(Q, 1)
    #print(np.count_nonzero(np.diag(Q) <= 0))
    # make it positive finite
    Q += 0.001 * np.identity(row)
    one = -np.ones(col)
    y = Dy.reshape((row,)).astype(float)
    alpha = solve_qp(P=Q, q=one, A=y, b=np.array([0.0]), lb=np.zeros(row), ub=C*np.ones(row))

    alpha[alpha < 0] = 0
    index = int(np.argwhere(alpha > 0)[0])

    # compute b
    b = np.sum(Dy * alpha.reshape(-1, 1) * K[:, [index]])
    b = Dy[index, 0] - b

    return alpha, b

def draw(Dx, Dy, alpha, b):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    x1 = np.linspace(-1, 1, num=50)
    x2 = np.linspace(-1, 1, num=50)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.insert(X1.reshape(1, -1).reshape(2500, 1),
                  [1], X2.reshape(1, -1).reshape(2500, 1), axis=1)
    
    K = np.matmul(Dx, np.transpose(X))
    K = (1 + K) ** 8
    result = []
    for i in range(2500):
        total = np.sum(Dy * alpha.reshape(-1, 1) * K[:, [i]])
        result.append(total + b)
    
    result = np.array(result)
    result = np.reshape(result, np.shape(X1))
    plt.contour(X1, X2, result, 0)

    for i in range(row):
        if Dy[i] == 1:
            plt.scatter(Dx[i, 0], Dx[i, 1], c='b', marker='o')
        else:
            plt.scatter(Dx[i, 0], Dx[i, 1], c='r', marker='x')

    return

if __name__ == '__main__':
    Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])

    C = 36
    alpha, b = SVM(Dx_train, Dy_train, C)
    print(alpha)
    print(np.count_nonzero(alpha > 0))
    print(b)

    plt.title("Optimal hyperplane C=" + str(C))
    draw(Dx_train, Dy_train, alpha, b)
    plt.show()
