import numpy as np
from nn.layers import Layer
from nn.activations import ActivationFunction


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

    def backward(self, y, lr):
        delta = self.layers[-1].a - y

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            dW = np.outer(delta, layer.x)
            db = delta

            if i > 0:
                prev_layer = self.layers[i - 1]
                delta = layer.W.T @ delta
                if prev_layer.activation == ActivationFunction.relu:
                    delta *= ActivationFunction.relu_derivative(prev_layer.z)

            layer.W -= lr * dW
            layer.b -= lr * db

    def train(self, X, Y, epochs, lr):
        losses = []

        for epoch in range(epochs):
            total_loss = 0

            for x, y in zip(X, Y):
                pred = self.forward(x)
                total_loss += -np.sum(y * np.log(pred + 1e-10))
                self.backward(y, lr)

            avg_loss = total_loss / len(X)
            losses.append(avg_loss)
            print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        return losses

    def predict(self, X):
        return np.array([self.forward(x) for x in X])
