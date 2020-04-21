import numpy as np


def linear(X):
    return X


def sigmoid(X):
    res = 1 / (1 + np.exp(-X))
    return res


def tanh(X):
    res = (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    return res


def relu(X):
    return np.maximum(0, X)


def leaky_relu(X, factor=0.01):
    return np.maximum(factor*X, X)


def softmax(X):
    exp = np.exp(X)
    exp_sum = np.sum(np.exp(X))
    res = exp/exp_sum
    return res


def gelu(X):
    res = 0.5 * X * (1 + np.tanh(np.sqrt(2 / np.pi) *
                                 (X + 0.044715 * np.power(X, 3))))
    return res
