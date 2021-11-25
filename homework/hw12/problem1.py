import numpy as np

def MLP_training(Dx, Dy, W_h, W_o, trans):
    row, col = np.size(Dx, 0), np.size(Dx, 1)
    t = 0
    G_o = 0 * W_o
    G_h = 0 * W_h
    while t < 1:
        # forward propagation
        for i in range(row):
            s = np.tanh(np.matmul(np.transpose(W_h), np.transpose(Dx[[i], :])))
            o = 0
            if trans == "identity":
                o = np.matmul(np.transpose(W_o), s)
            elif trans == "tanh":
                o = np.tanh(np.matmul(np.transpose(W_o), s))
            elif trans == "sign":
                o = np.sign(np.matmul(np.transpose(W_o), s))
            
        # backward propagation
        sens_o = 0
        if trans == "identity":
            sens_o = (1 / (2*row)) * (o - Dy)
        elif trans == "tanh":
            sens_o = (np.ones((np.size(o, 0), 1)) - o * o) * \
                (1 / (2*row)) * (o - Dy)
        elif trans == 'sign':
            sens_o = 0 * (1 / (2*row)) * (o - Dy)
        
        sens_h = (np.ones((np.size(s, 0), 1)) - s * s) * np.matmul(W_o, sens_o)

        g_o = np.outer(s, np.transpose(sens_o))
        g_h = np.outer(np.transpose(Dx[[i], :]), np.transpose(sens_h))
        G_o += g_o
        G_h += g_h

        t += 1

    return G_o, G_h

Dx = np.array([[1, 2]])
Dy = np.array([[1]])
W_h = np.array([[0.25, 0.25], [0.25, 0.25]])
W_o = np.array([[0.25], [0.25]])

G_o, G_h = MLP_training(Dx, Dy, W_h, W_o, "tanh")
print(G_o)
print(G_h)
