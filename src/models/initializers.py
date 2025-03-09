import numpy as np

class Initializers:
    @staticmethod
    def zeros(shape):
        return np.zeros(shape)

    @staticmethod
    def ones(shape):
        return np.ones(shape)

    @staticmethod
    def random_uniform(shape, low=-0.1, high=0.1):
        return np.random.uniform(low, high, shape)

    @staticmethod
    def random_normal(shape, mean=0.0, std=0.1):
        return np.random.normal(mean, std, shape)

    @staticmethod
    def xavier(shape):
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def he(shape):
        std = np.sqrt(2 / shape[0])
        return np.random.normal(0, std, shape)
    