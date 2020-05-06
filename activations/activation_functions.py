import numpy as np


def Linear(X, constant=1):
    return constant*X


def dLinear(constant=1):
    return constant


def Sigmoid(X):
    res = 1 / (1 + np.exp(-X))
    return res


def dSigmoid(X):
    return Sigmoid(X) * (1-Sigmoid(X))


def Tanh(X):
    return np.tanh(X)


def dTanh(X):
    return 1 - np.power(np.tanh(X), 2)


def ReLu(X):
    return np.maximum(0, X)


def dReLu(X):
    return 1 if X > 0 else 0


def Leaky_ReLu(X, factor=0.01):
    return np.maximum(factor*X, X)


def dLeaky_ReLu(X, factor=0.01):
    return 1 if X > 0 else factor


# TODO: d/dX

def Softmax(X):
    exp = np.exp(X)
    exp_sum = np.sum(np.exp(X))
    res = exp/exp_sum
    return res


def GeLu(X):
    res = 0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) *
                                 (X + 0.044715 * np.power(X, 3))))
    return res
