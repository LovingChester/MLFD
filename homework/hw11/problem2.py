from data_preprocess import *
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix



if __name__ == '__main__':
    Dx_train, Dy_train, Dx_test, Dy_test = gather_data(["ZipDigits.train", "ZipDigits.test"])
