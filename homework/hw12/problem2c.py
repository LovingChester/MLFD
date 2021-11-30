from data_preprocess import *
import matplotlib.pyplot as plt
import time

MAXITER = 10000

def compute_E_cv():

    return

# early stopping
def MLP_training(Dx_train, Dy_train, Dx_val, Dy_val, W_1, W_2, B_1, B_2):
    row, col = np.size(Dx_train, 0), np.size(Dx_train, 1)
    rate = np.random.rand()

    t = 0
    while t < MAXITER:
        E_in = 0
        G_1 = 0 * W_1
        G_2 = 0 * W_2
        Gb_1 = 0 * B_1
        Gb_2 = 0 * B_2
        for i in range(row):
            # forward propagation
            x_0 = Dx_train[[i], :]
            s_1 = np.matmul(np.transpose(W_1), np.transpose(x_0)) + B_1
            x_1 = np.tanh(s_1)
            s_2 = np.matmul(np.transpose(W_2), x_1) + B_2
            x_2 = 1 * s_2

            # backward propagation
            sens_2 = 2 * (x_2 - Dy_train[[i], :])
            sens_1 = (1 - x_1 * x_1) * np.matmul(W_2, sens_2)

            # compute current E_in
            E_in += (1 / (4*row)) * (float(x_2 - Dy_train[[i], :])) ** 2
            G_1 += (1 / (4*row)) * np.outer(np.transpose(x_0), np.transpose(sens_1))
            G_2 += (1 / (4*row)) * np.outer(x_1, np.transpose(sens_2))
            Gb_1 += (1 / (4*row)) * sens_1
            Gb_2 += (1 / (4*row)) * sens_2

        # compute validation

        if t % 10000 == 0:
            print("iteration: {}, E_in: {}".format(t, E_in))

        W_1_next = W_1 - rate * G_1
        W_2_next = W_2 - rate * G_2
        B_1_next = B_1 - rate * Gb_1
        B_2_next = B_2 - rate * Gb_2

        W_1 = W_1_next
        W_2 = W_2_next
        B_1 = B_1_next
        B_2 = B_2_next

        t += 1

    return W_1, W_2, B_1, B_2

if __name__ == '__main__':
    Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])
    m = 10
    B_1 = np.random.uniform(low=0.0, high=0.5, size=(m, 1))
    B_2 = np.random.uniform(low=0.0, high=0.5, size=(1, 1))
    W_1 = np.random.uniform(low=0.0, high=0.5, size=(2, m))
    W_2 = np.random.uniform(low=0.0, high=0.5, size=(m, 1))
    
    random_indices = np.random.choice(300, size=50, replace=False)
    Dx_val = Dx_train[random_indices, :]
    Dy_val = Dy_train[random_indices, :]

    Dx_train = np.delete(Dx_train, random_indices, axis=0)
    Dy_train = np.delete(Dy_train, random_indices, axis=0)
