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


def Momentum(w, grad, learning_rate, beta, v):  # pass v as an initialized vector
    v = beta * v + (1 - beta) * grad
    w_prev = w  # keep track of parameters of the previous iteration
    w_updated = w - learning_rate * v
    print("Value of weights at this iteration ", w_updated)
    if w_prev == w_updated:
        return ("done", w_updated)
    else:
        # return new weights and the v vector to pass it back again for next iteration
        return v, w_updated


def RMSProp(w, grad, learning_rate, beta, epsilon):
    pass


def Adam(w, grad, learning_rate, beta1, beta2, epsilon):
    pass
