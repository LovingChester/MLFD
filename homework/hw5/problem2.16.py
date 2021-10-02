import numpy as np
import matplotlib.pyplot as plt

plt.xlabel('x1')
plt.ylabel('x2')

x1 = np.arange(0, 3)
x2 = np.array(-x1+2)
plt.plot(x1, x2, 'm')
plt.annotate("+1", xy=(2,2))
plt.annotate("-1", xy=(0,0))
# plt.plot([0.75], [0.75], 'rx')
# plt.plot([1.5], [1.5], 'bo')
plt.show()
