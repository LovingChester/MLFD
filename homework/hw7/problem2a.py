import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math

def gradient_descent(x, y, rate):
    func_value = []
    for i in range(50):
        tmp_x = x - rate * (2 * x + 2 * math.sin(2 * math.pi * y) * 2 * math.pi * math.cos(2 * math.pi * x))
        tmp_y = y - rate * (4 * y + 2 * math.sin(2 * math.pi * x) * 2 * math.pi * math.cos(2 * math.pi * y))
        x = tmp_x
        y = tmp_y
        value = x**2 + 2 * y**2 + 2 * math.sin(2 * math.pi * x) * math.sin(2 * math.pi * y)
        func_value.append(value)

    return func_value


if __name__ == '__main__':
    func_value = gradient_descent(0.1, 0.1, 0.01)
    plt.title("learning rate = 0.01")
    plt.xlabel("iteration number")
    plt.ylabel("function value")
    plt.plot(range(50), func_value)
    plt.show()

    func_value = gradient_descent(0.1, 0.1, 0.1)
    plt.title("learning rate = 0.1")
    plt.xlabel("iteration number")
    plt.ylabel("function value")
    plt.plot(range(50), func_value)
    plt.show()
