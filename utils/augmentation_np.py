import itertools
import random

import numpy as np
import scipy.interpolate

def noise_transform_vectorized(X, sigma=0.05):
    """
    Adding random Gaussian noise with mean 0
    """
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise


def scaling_transform_vectorized(X, sigma=0.1):
    """
    Scaling by a random factor
    """
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1, X.shape[2]))
    return X * scaling_factor


def augment_list():
    l = [
        (noise_transform_vectorized),
        (scaling_transform_vectorized),
    ]
    return l

class RandAugment:
    def __init__(self, n):
        self.augment_list = augment_list()
        self.n = n

    def __call__(self, x):
        ops = random.choices(self.augment_list, k=self.n)

        for op in ops:
            x = op(x)
        return x


if __name__ == '__main__':
    randaugment = RandAugment(n=2)
    x = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    x = np.expand_dims(x, axis=0)
    x = randaugment(x)