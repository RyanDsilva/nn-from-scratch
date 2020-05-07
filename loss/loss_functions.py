import numpy as np


def MSE(y, yhat):
    return np.mean(np.power(y-yhat, 2))


def dMSE(y, yhat):
    return 2*(yhat-y)/y.size


def MAE(y, yhat):
    return np.sum(np.abs(y-yhat))


def dMAE(y, yhat):
    return 1 if y == yhat else -1
