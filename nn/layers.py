import numpy as np

class Neuron:
    def __init__(self, input_size, activation_fn):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.activation_fn = activation_fn

    def activate(self, x):
        z = np.dot(self.weights, x) + self.bias
        return self.activation_fn(z)


class Layer:
    def __init__(self, input_size, output_size, activation_fn):
        self.neurons = [Neuron(input_size, activation_fn) for _ in range(output_size)]
        self.activation_fn = activation_fn

    def forward(self, x):
        if self.activation_fn.__name__ == "softmax":
            z = np.array([np.dot(n.weights, x) + n.bias for n in self.neurons])
            return self.activation_fn(z)
        return np.array([n.activate(x) for n in self.neurons])
