import numpy as np


def MSE(y, yhat):
    return np.mean(np.power(y-yhat, 2))

def dMSE(y, yhat):
    return 2*(yhat-y)/y.size


def MAE(y, yhat):
    return np.sum(np.abs(y-yhat))

def dMAE(y, yhat):
    return 1 if y == yhat else -1



def kl_divergence(y, yhat):

    """
    measures the difference between two probability distributions
    over the same variable.

    Parameters:
    - y : Numpy array
    - yhat : Numpy array

    Returns:
    difference between two probability distribution.

     KL divergence can be calculated as
     the negative sum of probability of each event in P multiplied by
     the log of the probability of the event in Q over the probability of the event

    """
	return sum(y[i] * log2(y[i]/yhat[i]) for i in range(len(y)))

def entropy(y,factor=1e-15):
    """
      measures the performance of a classification model
      whose output is a probability value between 0 and 1

    Parameters:
    - y: Numpy array
    - factor: Optional (To ensure 0 is not returned).

    Returns:
     between 0 to 1
    """
	return -sum([y[i] * log2(y[i]+factor) for i in range(len(y))])


def cross_entropy(y,yhat,mode=None,factor=1e-15):
    """
    calculates loss among two probability vectors.

    Parameters:
    - y: Numpy array
    - yhat: numpy array
    - mode: Optional (mode= kl_divergence then calculate cross entropy using kl_divergence )
    - factor:  Optional (To ensure 0 is not returned).

    Returns:
     between 0 to 1 
    """
    if(mode=='Kl_diversion'):
        return entropy(y) + kl_divergence(y, yhat)
	return -sum([y[i]*log2(yhat[i]+factor) for i in range(len(y))])
