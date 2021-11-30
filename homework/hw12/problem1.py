import numpy as np

def MLP_training(Dx, Dy, b_h, b_o, W_h, W_o, trans):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    t = 0
    g_o = 0
    g_h = 0
    G_o = 0 * W_o
    G_h = 0 * W_h
    while t < 1:
        # forward propagation
        for i in range(row):
            s = np.tanh(np.matmul(np.transpose(W_h), np.transpose(Dx[[i], :])) + b_h)
            o = 0
            if trans == "identity":
                o = np.matmul(np.transpose(W_o), s) + b_o
            elif trans == "tanh":
                o = np.tanh(np.matmul(np.transpose(W_o), s) + b_o)
            elif trans == "sign":
                o = np.sign(np.matmul(np.transpose(W_o), s) + b_o)
            
            # backward propagation
            sens_o = 0
            if trans == "identity":
                sens_o = 2 * (o - Dy)
            elif trans == "tanh":
                sens_o = 2 * (o - Dy) * (1 - o ** 2)
            elif trans == 'sign':
                sens_o = 2 * (o - Dy) * 0
        
            sens_h = (1 - s * s) * np.matmul(W_o, sens_o)

            g_o = np.outer(s, np.transpose(sens_o))
            g_h = np.outer(np.transpose(Dx[[i], :]), np.transpose(sens_h))
            G_o += (1 / (4*row)) * g_o
            G_h += (1 / (4*row)) * g_h

        t += 1

    return g_o, g_h, G_o, G_h


if __name__ == '__main__':
    m = 2
    Dx = np.array([[1, 2]])
    Dy = np.array([[1]])
    b_h = np.full((m, 1), 0.25)
    b_o = np.full((1, 1), 0.25)
    W_h = np.full((2, m), 0.25)
    W_o = np.full((m, 1), 0.25)

    g_o, g_h, G_o, G_h = MLP_training(Dx, Dy, b_h, b_o, W_h, W_o, "tanh")
    print(g_o)
    print(g_h)
    # print(G_o)
    # print(G_h)

    W_h_plus = np.array([[0.25, 0.25], [0.25, 0.25]])
    W_o_plus = np.array([[0.25], [0.25+0.0001]])

    s_plus = np.tanh(np.matmul(np.transpose(W_h_plus), np.transpose(Dx[[0], :])) + b_h)
    o_plus = np.tanh(np.matmul(np.transpose(W_o_plus), s_plus) + b_o)
    
    f_plus = (o_plus - Dy) ** 2

    W_h_minus = np.array([[0.25, 0.25], [0.25, 0.25]])
    W_o_minus = np.array([[0.25], [0.25-0.0001]])

    s_minus = np.tanh(np.matmul(np.transpose(W_h_minus), np.transpose(Dx[[0], :])) + b_h)
    o_minus = np.tanh(np.matmul(np.transpose(W_o_minus), s_minus) + b_o)

    f_minus = (o_minus - Dy) ** 2

    print((f_plus - f_minus) / (2 * 0.0001))
