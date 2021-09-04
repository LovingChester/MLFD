'''
This program will implement the PLA as states on
the problem 1.4
@author Houhua Li
'''
import numpy as np
import matplotlib.pyplot as plt

print("hello")

'''
plot the f: x1 + x2 + 2 = 0
'''
plt.axis([-10, 10, -10, 10])
plt.xlabel("x1")
plt.ylabel("x2")

x1 = np.arange(-10, 11)
x2 = np.array(-x1+2)
plt.plot(x1,x2)

plt.grid(True)

np.random.seed(12)
D = np.random.randint(-9, 10, size=(20,2))
print(np.transpose(D))
plt.plot(np.transpose(D)[0],np.transpose(D)[1], 'ro')

plt.show()

# np.random.seed(20)
# D = np.random.randint(-4, 5, size=(20,2))
# print(D)