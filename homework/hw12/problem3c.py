import numpy as np
import matplotlib.pyplot as plt

plt.scatter(1, 0, c='b', marker='o')
plt.scatter(-1, 0, c='r', marker='x')
plt.plot(100*[0], np.linspace(-125, 125, num=100), 'b')

x1 = np.linspace(-5, 5, num=100)
x2 = x1 ** 3
plt.plot(x1, x2, 'm')

plt.show()
