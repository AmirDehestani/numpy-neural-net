import numpy as np
from nn.layers import Layer
from nn.activations import ActivationFunction
from nn.losses import cross_entropy, cross_entropy_derivative


class NeuralNetwork:
    def __init__(self, layer_sizes, hidden_activation, output_activation):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            act = output_activation if i == len(layer_sizes) - 1 else hidden_activation
            self.layers.append(Layer(layer_sizes[i - 1], layer_sizes[i], act))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, y, lr):
        activations = [x]
        zs = []

        for layer in self.layers:
            z = np.array(
                [np.dot(n.weights, activations[-1]) + n.bias for n in layer.neurons]
            )
            zs.append(z)
            activations.append(layer.forward(activations[-1]))

        delta = activations[-1] - y  # softmax + CE

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            a_prev = activations[i]

            grad_w = np.outer(delta, a_prev)
            grad_b = delta.copy()

            if i > 0:
                W = np.array([n.weights for n in layer.neurons])
                delta = W.T @ delta

                prev_layer = self.layers[i - 1]
                if prev_layer.activation_fn.__name__ == "relu":
                    delta *= zs[i - 1] > 0

            for j, neuron in enumerate(layer.neurons):
                neuron.weights -= lr * grad_w[j]
                neuron.bias -= lr * grad_b[j]

    def train(self, X, Y, epochs, lr):
        losses = []
        for epoch in range(epochs):
            total = 0
            for x, y in zip(X, Y):
                pred = self.forward(x)
                total += cross_entropy(pred, y)
                self.backward(x, y, lr)

            losses.append(total / len(X))
            print(f"Epoch {epoch}: loss={losses[-1]:.4f}")
        return losses

    def predict(self, X):
        return np.array([self.forward(x) for x in X])
