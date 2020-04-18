import numpy as np

def sigmoidFunc(val):
    return 1. / (1. + np.exp(-val))