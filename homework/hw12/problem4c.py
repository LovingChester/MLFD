from problem4a import *

def compute_E_cv(Dx, Dy, C):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    total_e_cv = 0
    for i in range(row):
        Dx_cv = np.delete(Dx, i, axis=0)
        Dy_cv = np.delete(Dy, i, axis=0)
        alpha, b = SVM(Dx_cv, Dy_cv, C)
        x = Dx[i].reshape(-1, 1)
        total = 0
        K = np.matmul(Dx_cv, x)
        K = (1 + K) ** 8
        total = np.sum(Dy_cv * alpha.reshape(-1, 1) * K)
        total += b
        if Dy[i, 0] != np.sign(total):
            total_e_cv += 1
        
    return total_e_cv / row

if __name__ == '__main__':
    Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])

    Cs = range(1, 50)

    # E_cvs = []
    # for C in Cs:
    #     E_cv = compute_E_cv(Dx_train, Dy_train, C)
    #     print("iteration {}, E_cv: {}".format(C, E_cv))
    #     E_cvs.append(E_cv)
    
    # plt.plot(Cs, E_cvs)
    # plt.show()

    alpha, b = SVM(Dx_train, Dy_train, 36)
    K = np.matmul(Dx_train, np.transpose(Dx_test))
    K = (1 + K) ** 8
    result = []
    for i in range(np.size(Dx_test, 0)):
        total = np.sum(Dy_train * alpha.reshape(-1, 1) * K[:, [i]])
        result.append(total + b)
    
    result = np.sign(result).reshape(-1, 1)
    print(np.count_nonzero(result - Dy_test) / np.size(Dx_test, 0))
