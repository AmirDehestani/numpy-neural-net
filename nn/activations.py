import numpy as np


class ActivationFunction:
    @staticmethod
    def relu(x):
        """
        Applies the ReLU activation function.

        :param x: Input array or scalar.
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """
        Computes the derivative of the ReLU activation function.

        :param x: Input array or scalar.
        """
        return (x > 0).astype(float)

    @staticmethod
    def softmax(x):
        """
        Applies the softmax activation function.

        :param x: Input array or scalar.
        """
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    @staticmethod
    def softmax_derivative(x):
        """
        Computes the derivative of the softmax activation function.

        :param x: Input array or scalar.
        """
        s = ActivationFunction.softmax(x)
        return s * (1 - s)
