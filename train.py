from nn.network import NeuralNetwork
from nn.activations import ActivationFunction
from data.mnist import load_mnist
import numpy as np

X, y, y_oh = load_mnist()

nn = NeuralNetwork(
    layer_sizes=[784, 64, 10],
    hidden_activation=ActivationFunction.relu,
    output_activation=ActivationFunction.softmax,
)

losses = nn.train(X, y_oh, epochs=20, lr=5e-3)

preds = nn.predict(X)
acc = np.mean(np.argmax(preds, axis=1) == y)
print("Train accuracy:", acc)
