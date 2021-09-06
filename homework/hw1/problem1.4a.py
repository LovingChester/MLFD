#!/usr/bin/env python
# coding: utf-8

# # Problem a

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import random as rd


# ## Generate target function and random data set

# In[2]:


plt.axis([-100, 100, -100, 100])
plt.xlabel("x1")
plt.ylabel("x2")

x1 = np.arange(-100, 100)
x2 = np.array(-x1+2)
plt.plot(x1,x2,"m")
plt.annotate("target function f", xy=(50, -40), xytext=(50, 0), arrowprops=dict(facecolor="m"))
plt.grid(True)

'''
generate the random data set
and plot the data set based on
its corresponding y
'''
np.random.seed(14)
Dx = np.random.randint(-100, 101, size=(20,2))
Dy = list(map(lambda x: x[0] + x[1] - 2, Dx))
Dy = np.sign(Dy)

'''
positive will store the data points which has +1
negative will store the data points which has -1
'''
positive = []
negative = []
for i in range(np.size(Dy,0)):
    if Dy[i] == 1:
        positive.append(list(Dx[i]))
    else:
        negative.append(list(Dx[i]))
# print(positive)
plt.plot(np.transpose(positive)[0], np.transpose(positive)[1], 'bo')
plt.plot(np.transpose(negative)[0], np.transpose(negative)[1], 'rx')

plt.show()

