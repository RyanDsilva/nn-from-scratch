import numpy as np


def MSE(y, yhat):
    return np.square(y-yhat).mean()


def MAE(y, yhat):
    return np.sum(np.abs(y-yhat))


def CrossEntropy(y, yhat):
    return -np.sum(y*np.log(yhat))
