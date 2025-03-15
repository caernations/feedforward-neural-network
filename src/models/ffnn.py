from models.layers import DenseLayer
from models.activations import ReLU, Softmax
from models.loss import CategoricalCrossEntropy
import numpy as np

class FeedForwardNN:
    def __init__(self, input_size, hidden_layers, output_size, activations, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate

        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(DenseLayer(
                layer_sizes[i], layer_sizes[i+1], activations[i], learning_rate))
            if activations[i] == "relu":
                self.layers.append(ReLU())
            elif activations[i] == "softmax":
                self.layers.append(Softmax())

        self.loss_function = CategoricalCrossEntropy()

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dvalues, y_true):
        for i in reversed(range(len(self.layers))):
            if isinstance(self.layers[i], Softmax):
                loss_grad = self.layers[i].backward(dvalues, y_true)
            else:
                loss_grad = self.layers[i].backward(loss_grad)

    def train(self, X_train, y_train, epochs=10):
        for epoch in range(epochs):
            predictions = self.forward(X_train)
            loss = self.loss_function.forward(predictions, y_train)
            self.backward(predictions, y_train)
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    def save_weights(self, filename):
        weights = []
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                weights.append({
                    'weights': layer.weights,
                    'biases': layer.biases
                })
        np.save(filename, weights)

    def load_weights(self, filename):
        weights = np.load(filename, allow_pickle=True)
        dense_idx = 0
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.weights = weights[dense_idx]['weights']
                layer.biases = weights[dense_idx]['biases']
                dense_idx += 1
