import numpy as np
import matplotlib.pyplot as plt
import math

K = 10000

# generate K numbers of data sets, each contains two points
X = np.random.uniform(-1.0, 1.0, (K,2,1))
test = np.random.uniform(-1.0, 1.0, (K,1))
# compute average g(x) and plot avg g(x) and f(x)
avg_a = 0
avg_b = 0
bias = 0
var_x_2 = 0
var_x = 0
var_b = 0
for i in range(K):
    x1, x2 = float(X[i][0]), float(X[i][1])
    avg_a += (x1+x2)
    avg_b += x1*x2
    bias += ((x1**2)**2 + (x2**2)**2) / 2
    var_x_2 += (x1+x2)**2
    var_x += 2*(x1+x2)*x1*x2
    var_b += x1**2*x2**2

avg_a = avg_a / K
avg_b = avg_b / K

avg_var_x_2 = var_x_2 / K
avg_var_x = var_x / K
avg_var_b = var_b / K

x = np.arange(-1, 1.01, 0.01)
y = np.array(x**2)
avg_g = np.array(avg_a*x - avg_b)
plt.plot(x, y, 'b')
plt.plot(x, avg_g, 'm')

var_sum = 0
for i in range(K):
    num = float(test[i])
    var_sum += avg_var_x_2 * num**2 - avg_var_x * num + avg_var_b

e_out_sum = 0
for i in range(K):
    num = float(test[i])
    e_out_sum += avg_var_x_2 * num**2 - avg_var_x * num + avg_var_b - 2*(avg_a*num - avg_b)*num**2 + num**4

plt.show()

# bias
print("bias: {:.2f}".format(bias/K))
print("Var: {:.2f}".format(var_sum/K))
print("E out: {:.2f}".format(e_out_sum/K))
print("E out: {:.2f} by adding bias and var".format(bias/K + var_sum/K))
