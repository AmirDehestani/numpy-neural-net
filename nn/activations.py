import numpy as np


class ActivationFunction:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    @staticmethod
    def softmax_derivative(x):
        s = ActivationFunction.softmax(x)
        return s * (1 - s)
