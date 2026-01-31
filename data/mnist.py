import torch
from torchvision import datasets, transforms
import numpy as np


def load_mnist(n_samples=12000):
    """
    Load MNIST dataset and preprocess it.

    :param n_samples: Number of samples to load from the dataset
    """
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        "./data/cache", train=True, download=True, transform=transform
    )

    X = dataset.data[:n_samples].float() / 255.0
    y = dataset.targets[:n_samples]

    X = torch.flatten(X, start_dim=1).numpy()
    y = y.numpy()

    y_onehot = np.zeros((y.size, y.max() + 1))
    y_onehot[np.arange(y.size), y] = 1

    return X, y, y_onehot
