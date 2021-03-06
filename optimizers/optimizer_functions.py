import numpy as np


class GradientDescent:
    def minimize(self, w, b, dW, dB, vW, vB, learning_rate=0.01, beta=0.9):
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
        return w_updated, b_updated, vW, vB


class Momentum:
    def __init__(self, nestrov=False):
        self.nestrov = nestrov

    def minimize(self, w, b, dW, dB, vW, vB, learning_rate=0.01, beta=0.9):
        """Implements Gradient Descent with Momentum to find minima of cost function

        Parameters:
        - w (numpy array): weights matrix
        - b (numpy array): bias matrix
        - dW (numpy array): gradient of weights matrix wrt cost function
        - dB (numpy array): gradient of bias matrix wrt cost function
        - learning_rate (double): learning rate used to update weights
        - beta (double): Momentum term for smoothing
        - vW (numpy array): holds the state of the optimizer for previous iteration (weights)
        - vB (numpy array): holds the state of the optimizer for previous iterations (biases)

        Returns:
        - w_updated (numpy array): updated weights
        - b_updated (numpy array): updated bias
        - vW (numpy array): updated state of the optimizer for current iteration (weights)
        - vB (numpy array): updated state of the optimizer for current iteration (biases)

        """
        if (self.nestrov == False):
            vW = beta * vW + (1-beta) * dW
            vB = beta * vB + (1-beta) * dB
            w_updated = w - learning_rate * vW
            b_updated = b - learning_rate * vB

        elif (self.nestrov == True):

            vW_prev = vW
            vB_prev = vB
            vW = beta * vW + (1-beta) * dW
            vB = beta * vB + (1-beta) * dB
            # need to check code here
            w = w - learning_rate * ((beta * vW_prev) + ((1 - beta) * dW))
            b = b - learning_rate * ((beta * vB_prev) + ((1 - beta) * dB))
            w_updated = w
            b_updated = b

        return w_updated, b_updated, vW, vB


class RMSProp:
    def minimize(self, w, b, dW, dB, sdW, sdB, learning_rate=0.1, beta=0.999):
        sdW = beta * sdW + (1 - beta) * np.square(dW/2000)

        sdB = beta * sdB + (1 - beta) * np.square(dB/2000)
        w_updated = w - learning_rate*dW/2000*np.sqrt(sdW+ float(pow(10,-8)))
        b_updated = b - learning_rate*dB/2000*np.sqrt(sdB+ float(pow(10,-8)))

        return w_updated, b_updated, sdW, sdB


def Adam(w, b, dW, dB, learning_rate, beta1, beta2, epsilon):
    pass
