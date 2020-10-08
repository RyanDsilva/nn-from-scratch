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
	return sum(y[i] * log2(y[i]/yhat[i]) for i in range(len(y)))

def entropy(y,ets=1e-15):
	return -sum([y[i] * log2(y[i]+ets) for i in range(len(y))])


def cross_entropy(y,yhat,mode=None,,ets=1e-15):
    if(mode=='Kl_diversion'):
        return entropy(y) + kl_divergence(y, yhat)
	return -sum([y[i]*log2(yhat[i]+ets) for i in range(len(y))])
