import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
from problem1a import *


Dx, Dy, final_w, count = gather_data("ZipDigits.train")

E_in = np.linalg.norm(np.matmul(Dx, final_w) - Dy) ** 2 / count
