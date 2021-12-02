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

    # compute b
    index = 0
    for i in range(row):
        if alpha[i] > 0:
            index = i
            break
    
    b = 0
    for i in range(row):
        if alpha[i] > 0:
            b += Dy[i, 0] * alpha[i] * K[i, index]
    
    b = Dy[index, 0] - b

    return alpha, b

def draw(Dx, Dy, alpha, b):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    x1 = np.linspace(-1, 1, num=50)
    x2 = np.linspace(-1, 1, num=50)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.insert(X1.reshape(1, -1).reshape(2500, 1),
                  [1], X2.reshape(1, -1).reshape(2500, 1), axis=1)
    
    result = []
    for i in range(2500):
        x = X[i, :]
        total = 0
        for j in range(300):
            if alpha[j] > 0:
                total += Dy[j, 0] * alpha[j] * (1 + np.dot(Dx[j], x)) ** 8
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

    print(get_kernel(Dx_train))

    alpha, b = SVM(Dx_train, Dy_train, 500)
    print(alpha)
    print(np.count_nonzero(alpha > 0))
    print(b)

    draw(Dx_train, Dy_train, alpha, b)
    plt.show()
