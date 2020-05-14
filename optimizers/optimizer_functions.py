import numpy as np


class GradientDescent:
    def minimize(self, w, b, dW, dB, vw, vb, learning_rate=0.01, beta=0.9):
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
        w_updated = w - learning_rate * dW
        b_updated = b - learning_rate * dB
        return w_updated, b_updated, vw, vb


class Momentum:
    def minimize(self, w, b, dW, dB, vw, vb, learning_rate=0.01, beta=0.9):
        """Implements Gradient Descent with Momentum to find minima of cost function

        Parameters:
        - w (numpy array): weights matrix
        - b (numpy array): bias matrix
        - dW (numpy array): gradient of weights matrix wrt cost function
        - dB (numpy array): gradient of bias matrix wrt cost function
        - learning_rate (double): learning rate used to update weights
        - beta (double):
        - vw (numpy array): holds the state of the optimizer for previous iteration (weights)
        - vb (numpy array): holds the state of the optimizer for previous iterations (biases)

        Returns:
        - w_updated (numpy array): updated weights
        - b_updated (numpy array): updated bias
        - vw (numpy array): updated state of the optimizer for current iteration (weights)
        - vb (numpy array): updated state of the optimizer for current iteration (biases)

        """

        vw = beta * vw + (1 - beta) * dW

        vb = beta * vb + (1 - beta) * dB

        w_updated = w - learning_rate * vw
        b_updated = b - learning_rate * vb

        return w_updated, b_updated, vw, vb


def RMSProp(w, b, dW, dB, learning_rate, beta, epsilon):
    pass


def Adam(w, b, dW, dB, learning_rate, beta1, beta2, epsilon):
    pass
