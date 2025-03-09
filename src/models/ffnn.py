import numpy as np
from src.models.layers import Layer
from src.models.activations import ActivationFunctions
from src.models.loss import LossFunctions
from src.models.optimizers import Optimizer

class FFNN:
    def __init__(self, input_size, hidden_layers, output_size, activations, loss_function, weight_init="random_uniform"):
        self.layers = []
        self.activations = activations
        self.loss_function = LossFunctions[loss_function]

        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activation=activations[i], weight_init=weight_init)
            self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y, learning_rate):
        output = self.forward(X)
        loss_grad = self.loss_function["derivative"](y, output)

        # backward pass
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32, learning_rate=0.01, verbose=1):
        n_samples = X_train.shape[0]
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                self.backward(X_batch, y_batch, learning_rate)

            # hitung loss
            train_loss = self.loss_function["function"](y_train, self.forward(X_train))
            history["train_loss"].append(train_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self.loss_function["function"](y_val, self.forward(X_val))
                history["val_loss"].append(val_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}" + (f" - Val Loss: {val_loss:.4f}" if val_loss else ""))

        return history

    def predict(self, X):
        return self.forward(X)

    def save_weights(self, file_path):
        weights = [layer.weights for layer in self.layers]
        biases = [layer.biases for layer in self.layers]
        np.savez(file_path, weights=weights, biases=biases)
        print(f"✅ Bobot berhasil disimpan di {file_path}")

    def load_weights(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        for i, layer in enumerate(self.layers):
            layer.weights = data["weights"][i]
            layer.biases = data["biases"][i]
        print(f"✅ Bobot berhasil dimuat dari {file_path}")

    def summary(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i+1}: {layer.input_size} -> {layer.output_size} | Activation: {layer.activation}")