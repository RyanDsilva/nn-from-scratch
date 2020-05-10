from .layer import Layer
import numpy as np
import config


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

    def backward_propagation(self, output_error, optimizer_fn):
        input_error = np.dot(output_error, self.weights.T)
        dW = np.dot(self.input.T, output_error)
        dB = output_error
        w_updated, b_updated = optimizer_fn(
            self.weights, self.bias, dW, dB, config.learning_rate)
        self.weights = w_updated
        self.bias = b_updated
        return input_error
