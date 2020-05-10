import numpy as np


def GradientDescent(w, b, dW, dB, learning_rate=0.01):
    """Implements Gradient Descent to find minima of cost function

    Parameters:
    - w (numpy array): weights matrix
    - b (numpy array): bias matrix
    - dW (numpy array): gradient of weights matrix wrt cost function
    - dB (numpy array): gradient of bias matrix wrt cost function
    - learning_rate (double): learning rate used to update weights

    Returns:
    - w_updated (numpy array): updated weights
    - b_updated (numpy array): updated bias

    """
    w_updated = w - learning_rate*dW
    b_updated = b - learning_rate*dB
    return w_updated, b_updated


def Momentum(w, b, dW, dB, learning_rate, beta):
    """Implements Gradient Descent with Momentum to find minima of cost function

    Parameters:
    - w (numpy array): weights matrix
    - b (numpy array): bias matrix
    - dW (numpy array): gradient of weights matrix wrt cost function
    - dB (numpy array): gradient of bias matrix wrt cost function
    - learning_rate (double): learning rate used to update weights
    - beta (double): 

    Returns:
    - w_updated (numpy array): updated weights
    - b_updated (numpy array): updated bias

    """
    pass


def RMSProp(w, b, dW, dB, learning_rate, beta, epsilon):
    pass


def Adam(w, b, dW, dB, learning_rate, beta1, beta2, epsilon):
    pass
