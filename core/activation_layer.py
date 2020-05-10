from .layer import Layer


class Activation(Layer):
    def __init__(self, activation, dactivation):
        self.activation = activation
        self.dactivation = dactivation

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, optimizer_fn):
        return self.dactivation(self.input) * output_error
