import numpy as np


def GradientDescent(w, grad, learning_rate):
    w_updated = w - learning_rate*grad
    return w_updated


def Momentum(w, grad, learning_rate, beta):
    pass


def RMSProp(w, grad, learning_rate, beta, epsilon):
    pass


def Adam(w, grad, learning_rate, beta1, beta2, epsilon):
    pass
