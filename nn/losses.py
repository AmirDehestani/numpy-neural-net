import numpy as np


def cross_entropy(pred, target, eps=1e-10):
    """
    Calculates the cross-entropy loss.

    :param pred: Predicted probabilities.
    :param target: True labels.
    :param eps: Small value to avoid log(0).
    """
    return -np.sum(target * np.log(pred + eps))


def cross_entropy_derivative(pred, target):
    """
    Calculates the derivative of the cross-entropy loss.

    :param pred: Predicted probabilities.
    :param target: True labels.
    """
    return pred - target
