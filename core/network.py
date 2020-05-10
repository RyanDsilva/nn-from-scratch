class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.dloss = None

    def add(self, layer):
        self.layers.append(layer)

    def useLoss(self, loss, dloss):
        self.loss = loss
        self.dloss = dloss

    def useOptimizer(self, optimizer, learning_rate):
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def predict(self, input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs):
        samples = len(x_train)
        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err += self.loss(y_train[j], output)
                error = self.dloss(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(
                        error, self.optimizer, self.learning_rate)
            err /= samples
            print('epoch %d/%d\terror=%f' % (i+1, epochs, err))
