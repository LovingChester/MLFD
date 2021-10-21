from problem2a import *

func_value = gradient_descent(0.1, 0.1, 0.01)
print(func_value[-1])
func_value = gradient_descent(1, 1, 0.01)
print(func_value[-1])
func_value = gradient_descent(-0.5, -0.5, 0.01)
print(func_value[-1])
func_value = gradient_descent(-1, -1, 0.01)
print(func_value[-1])
