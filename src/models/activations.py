import numpy as np


class Activation:
    def __init__(self, name):
        self.name = name

    def forward(self, x, derivative=False):
        if self.name == 'linear':
            if derivative:
                return np.ones_like(x)
            return x
        elif self.name == 'relu':
            if derivative:
                return (x > 0).astype(float)
            return np.maximum(0, x)
        elif self.name == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            if derivative:
                return s * (1 - s)
            return s
        elif self.name == 'tanh':
            t = np.tanh(x)
            if derivative:
                return 1 - t**2
            return t
        elif self.name == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation function: {self.name}")


class ActivationFunctions:
    linear = Activation('linear')
    relu = Activation('relu')
    sigmoid = Activation('sigmoid')
    tanh = Activation('tanh')
    softmax = Activation('softmax')
