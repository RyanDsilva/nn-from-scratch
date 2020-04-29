import numpy as np


def Linear(X):
    return X


def Sigmoid(X):
    res = 1 / (1 + np.exp(-X))
    return res


def Tanh(X):
    res = (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    return res


def ReLu(X):
    return np.maximum(0, X)


def Leaky_ReLu(X, factor=0.01):
    return np.maximum(factor*X, X)


def Softmax(X):
    exp = np.exp(X)
    exp_sum = np.sum(np.exp(X))
    res = exp/exp_sum
    return res


def GeLu(X):
    res = 0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) *
                                 (X + 0.044715 * np.power(X, 3))))
    return res
