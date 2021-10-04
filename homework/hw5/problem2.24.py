import numpy as np
import matplotlib.pyplot as plt
import math

K = 10000

# generate K numbers of data sets, each contains two points
X = np.random.uniform(-1.0, 1.0, (K,2,1))

# compute average g(x) and plot avg g(x) and f(x)
avg_a = 0
avg_b = 0
bias = 0
var = 0
for i in range(K):
    x1, x2 = float(X[i][0]), float(X[i][1])
    avg_a += (x1+x2)
    avg_b += x1*x2
    bias += ((x1**2)**2 + (x1**2)**2) / 2
    

avg_a = avg_a / K
avg_b = avg_b / K

x = np.arange(-1, 1.01, 0.01)
y = np.array(x**2)
avg_g = np.array(avg_a*x - avg_b)
plt.plot(x, y, 'b')
plt.plot(x, avg_g, 'm')
plt.show()

# bias
print("bias: {:.2f}".format(bias/K))
