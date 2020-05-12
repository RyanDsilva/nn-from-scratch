from .layer import Layer
import numpy as np
import config


class Dense(Layer):
    # input_size = Number of Input Neurons
    # output_size = Number of Output Neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.v_weights = np.zeros_like(self.weights) + 0.0
        self.v_bias = np.zeros_like(self.bias) + 0.0
        print(self.v_weights.shape)
        print(self.v_bias.shape)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, optimizer_fn, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        dW = np.dot(self.input.T, output_error)
        dB = output_error
        print("Gradient shape: ", dW.shape)
        w_updated, b_updated, vw_updated, vb_updated = optimizer_fn.Momentum(
            self.weights, self.bias, dW, dB, learning_rate, self.v_weights, self.v_bias
        )
        # w_updated, b_updated = optimizer_fn.GradientDescent(
        #     self.weights, self.bias, dW, dB, learning_rate)
        self.weights = w_updated
        print("Updated weight shape:", self.weights.shape)
        self.bias = b_updated
        self.v_weights = vw_updated
        self.v_bias = vb_updated
        return input_error
