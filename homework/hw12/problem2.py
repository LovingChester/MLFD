from data_preprocess import *
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=False, threshold=5)

def MLP_training(Dx, Dy, W_h, W_o):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    

    return

if __name__ == '__main__':
    Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])
    m = 10
    W_h = np.random.rand(2, m)
    W_o = np.random.rand(m, 1)

