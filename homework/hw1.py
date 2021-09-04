'''
This program will implement the PLA as states on
the problem 1.4
@author Houhua Li
'''
import numpy as np
import matplotlib.pyplot as plt

print("hello")
a = np.array([1,2,3,4])
print(a)

'''
plot the f: x1 + x2 + 2 = 0
'''
plt.axis([-4, 4, -4, 4])
plt.xlabel("x1")
plt.ylabel("x2")

x1 = np.arange(-4, 5)
x2 = np.array(-x1+2)
plt.plot(x1,x2)

plt.grid(True)
plt.show()
