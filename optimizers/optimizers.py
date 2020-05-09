import numpy as np


def GradientDescent(w, grad, learning_rate):
    w_updated = w - learning_rate*grad
    return w_updated


def Momentum(w, grad, learning_rate, beta,v): #pass v as an initialized vector
    v = beta * v + (1 - beta) * grad
    w_prev = w #keep track of parameters of the previous iteration
    w_updated = w - learning_rate * v
    print("Value of weights at this iteration ", w_updated)
    if w_prev == w_updated:
        return ("done",w_updated)
    else:
        return v,w_updated #return new weights and the v vector to pass it back again for next iteration

        



def RMSProp(w, grad, learning_rate, beta, epsilon):
    pass


def Adam(w, grad, learning_rate, beta1, beta2, epsilon):
    pass
