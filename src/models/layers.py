import numpy as np
from models.initializers import Initializers

class DenseLayer:
    def __init__(self, input_size, output_size, activation, learning_rate=0.01):
        if activation == "relu":
            self.weights = Initializers.he((input_size, output_size))
        else:
            self.weights = Initializers.xavier((input_size, output_size))
            
        self.biases = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

        self.weights -= self.learning_rate * self.dweights
        self.biases -= self.learning_rate * self.dbiases
        return self.dinputs
    