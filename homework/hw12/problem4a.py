from numpy import double
from data_preprocess import *
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

np.set_printoptions(precision=3, suppress=False, threshold=5)

C = 5

def get_kernel(D):
    K = np.matmul(D, np.transpose(D))
    K = (1 + K) ** 8
    return K

def SVM(Dx, Dy):
    K = get_kernel(Dx)
    Q = np.outer(Dy, np.transpose(Dy)) * K
    row, col = np.size(Q, 0), np.size(Q, 1)
    #print(np.count_nonzero(np.diag(Q) <= 0))
    # make it positive finite
    Q += 0.1 * np.identity(row)
    one = -np.ones(col)
    
    alpha = solve_qp(P=Q, q=one)
    print(alpha)
    return

if __name__ == '__main__':
    Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])

    print(get_kernel(Dx_train))

    # M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    # P = np.dot(np.transpose(M), M)  # this is a positive definite matrix
    # q = np.dot(np.array([3., 2., 3.]), M).reshape((3,))
    # x = solve_qp(P, q)
    # print("QP solution: x = {}".format(x))
    # print(Dy_train)

    # print(np.outer(Dy_train, np.transpose(Dy_train)))

    SVM(Dx_train, Dy_train)
