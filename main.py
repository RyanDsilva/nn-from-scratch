import numpy as np
import time

import config
from core.network import Network
from core.dense import Dense
from core.activation_layer import Activation
from activations.activation_functions import Tanh, dTanh
from loss.loss_functions import MSE, dMSE
from optimizers.optimizer_functions import Momentum

from keras.datasets import mnist
from keras.utils import np_utils

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
x_train = x_train.astype("float32")
x_train /= 255
y_train = np_utils.to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
x_test = x_test.astype("float32")
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Model
nn = Network()
nn.add(Dense(28 * 28, 100))
nn.add(Activation(Tanh, dTanh))
nn.add(Dense(100, 50))
nn.add(Activation(Tanh, dTanh))
nn.add(Dense(50, 10))
nn.add(Activation(Tanh, dTanh))

# Training

nn.useLoss(MSE, dMSE)
nn.useOptimizer(Momentum(nestrov=True), learning_rate=config.learning_rate , beta = config.beta)
nn.fit(x_train[0:2000], y_train[0:2000], epochs=config.epochs)


# Prediction
out = nn.predict(x_test[0:2])
print("\nPredicted Values: ")
print(out, end="\n")
print("True Values: ")
print(y_test[0:2])
