from data_preprocess import *
import matplotlib.pyplot as plt
import time

#np.set_printoptions(precision=3, suppress=False, threshold=5)

MAXITER = 2000000
alpha = 1.1
beta = 0.8

# use variable Learning Rate Gradient Descent
# add weight decay
def MLP_training(Dx, Dy, W_1, W_2, B_1, B_2):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    rate = np.random.rand()
    #E_ins = []
    t = 0
    while t < MAXITER:
        E_in = 0
        G_1 = 0 * W_1
        G_2 = 0 * W_2
        Gb_1 = 0 * B_1
        Gb_2 = 0 * B_2
        for i in range(row):
            # forward propagation
            x_0 = Dx[[i], :]
            s_1 = np.matmul(np.transpose(W_1), np.transpose(x_0)) + B_1
            x_1 = np.tanh(s_1)
            s_2 = np.matmul(np.transpose(W_2), x_1) + B_2
            x_2 = 1 * s_2

            # backward propagation
            sens_2 = 2 * (x_2 - Dy[[i], :])
            sens_1 = (1 - x_1 * x_1) * np.matmul(W_2, sens_2)

            # compute current E_in
            E_in += (1 / (4*row)) * (float(x_2 - Dy[[i], :])) ** 2
            G_1 += (1 / (4*row)) * np.outer(np.transpose(x_0), np.transpose(sens_1))
            G_2 += (1 / (4*row)) * np.outer(x_1, np.transpose(sens_2))
            Gb_1 += (1 / (4*row)) * sens_1
            Gb_2 += (1 / (4*row)) * sens_2

        #E_ins.append(E_in)
        if t % 10000 == 0:
            print("iteration: {}, E_in: {}".format(t, E_in))

        # regularize using weight decay
        G_1 += ((2 * (0.01/row)) / row) * W_1
        G_2 += ((2 * (0.01/row)) / row) * W_2

        W_1_next = W_1 - rate * G_1
        W_2_next = W_2 - rate * G_2
        B_1_next = B_1 - rate * Gb_1
        B_2_next = B_2 - rate * Gb_2
        new_E_in = 0
        for i in range(row):
            # forward propagation
            x_0 = Dx[[i], :]
            s_1 = np.matmul(np.transpose(W_1_next), np.transpose(x_0)) + B_1_next
            x_1 = np.tanh(s_1)
            s_2 = np.matmul(np.transpose(W_2_next), x_1) + B_2_next
            x_2 = 1 * s_2

            # compute current E_in
            new_E_in += (1 / (4*row)) * (float(x_2 - Dy[[i], :])) ** 2
        
        if new_E_in < E_in:
            W_1 = W_1_next
            W_2 = W_2_next
            B_1 = B_1_next
            B_2 = B_2_next
            rate *= alpha
        else:
            rate *= beta

        t += 1 

    return W_1, W_2, B_1, B_2

def draw(Dx, Dy, W_1, W_2, B_1, B_2):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    x1 = np.linspace(-1, 1, num=100)
    x2 = np.linspace(-1, 1, num=100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.insert(X1.reshape(1, -1).reshape(10000, 1),
                  [1], X2.reshape(1, -1).reshape(10000, 1), axis=1)
    
    result = []
    for i in range(10000):
        x_0 = X[[i], :]
        s_1 = np.matmul(np.transpose(W_1), np.transpose(x_0)) + B_1
        x_1 = np.tanh(s_1)
        s_2 = np.matmul(np.transpose(W_2), x_1) + B_2
        x_2 = 1 * s_2
        result.append(float(x_2))

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
    m = 10
    B_1 = np.random.uniform(low=0.0, high=0.5, size=(m, 1))
    B_2 = np.random.uniform(low=0.0, high=0.5, size=(1, 1))
    W_1 = np.random.uniform(low=0.0, high=0.5, size=(2, m))
    W_2 = np.random.uniform(low=0.0, high=0.5, size=(m, 1))

    start = time.time()
    final_W_1, final_W_2, final_B_1, final_B_2 = MLP_training(
        Dx_train, Dy_train, W_1, W_2, B_1, B_2)
    print(final_W_1)
    print(final_W_2)
    draw(Dx_train, Dy_train, final_W_1, final_W_2, final_B_1, final_B_2)
    plt.show()
