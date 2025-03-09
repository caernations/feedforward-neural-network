import numpy as np

class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, dvalues):
        return dvalues * (self.inputs > 0)

class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), np.argmax(y_true, axis=1)] -= 1
        self.dinputs /= samples
        return self.dinputs

class ActivationFunctions:
    relu = ReLU()
    softmax = Softmax()
    