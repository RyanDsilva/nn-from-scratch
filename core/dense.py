from .layer import Layer
import numpy as np


class Dense(Layer):
    # input_size = Number of Input Neurons
    # output_size = Number of Output Neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        dW = np.dot(self.input.T, output_error)
        dB = output_error
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * dB
        return input_error
