from data_preprocess import *
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=False, threshold=5)

def get_kernel(D):
    K = np.matmul(D, np.transpose(D))
    K = (1 + K) ** 8
    return K

def SVM():

    return

if __name__ == '__main__':
    Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])

    print(get_kernel(Dx_train))
