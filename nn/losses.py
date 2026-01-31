import numpy as np

def cross_entropy(pred, target):
    eps = 1e-10
    return -np.sum(target * np.log(pred + eps))

def cross_entropy_derivative(pred, target):
    return pred - target
