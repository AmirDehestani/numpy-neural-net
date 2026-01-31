import numpy as np


class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2 / input_dim)
        self.b = np.zeros(output_dim)
        self.activation = activation

        # cache for backprop
        self.x = None
        self.z = None
        self.a = None

    def forward(self, x):
        self.x = x
        self.z = self.W @ x + self.b
        self.a = self.activation(self.z)
        return self.a
