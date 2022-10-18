# Code from : https://albertauyeung.github.io/2017/04/23/python-matrix-factorization.html/

import numpy as np

def mse(actual, predicted):
    """
    A function to compute the total mean square error of the non zero ratings
    
    Arguments
        - actual (ndarray)   : actual user-item rating matrix
        - predicted (int)   : predicted user-item rating matrix
    """
    xs, ys = actual.nonzero()
    error = 0
    for x, y in zip(xs, ys):
        error += pow(actual[x, y] - predicted[x, y], 2)
    return error / len(xs)

def rmse(actual, predicted):
    """
    A function to compute the root mean square error of the non missing ratings
    
    Arguments
        - actual (ndarray)   : actual user-item rating matrix
        - predicted (int)   : predicted user-item rating matrix
    """
    indices = actual >= 0
    return np.sqrt(np.mean(np.square(actual[indices] - predicted[indices])))