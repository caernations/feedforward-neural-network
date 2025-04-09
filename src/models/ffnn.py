import numpy as np
from models.layers import DenseLayer
from models.activations import ActivationFunctions
from tqdm import tqdm
from models.loss import LossFunctions 
from models.loss import l1_regularization, l2_regularization

class FeedForwardNN:
    def __init__(self, input_size, hidden_layers, output_size, activations,
                 weight_init='xavier', weight_init_params=None, loss_function='categorical_crossentropy', lambda_l1=0.0, lambda_l2=0.0):
        """
        Initialize a Feedforward Neural Network

        Parameters:
        -----------
        input_size : int
            Size of the input layer
        hidden_layers : list
            List of sizes for each hidden layer
        output_size : int
            Size of the output layer
        activations : list
            List of activation functions for each layer
        weight_init : str
            Initialization method ('zero', 'uniform', 'normal', 'xavier', 'he')
        weight_init_params : dict
            Parameters for weight initialization (lower_bound, upper_bound, seed, mean, variance)
        loss_function : str
            Loss function to use ('mse', 'binary_crossentropy', 'categorical_crossentropy')
        """
        self.layers = []
        self.layer_outputs = []  
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2


        # Set up layers
        layer_sizes = [input_size] + hidden_layers + [output_size]

        # Initialize layers
        for i in range(len(layer_sizes) - 1):
            # Get activation function
            if activations[i] == 'linear':
                activation = ActivationFunctions.linear
            elif activations[i] == 'relu':
                activation = ActivationFunctions.relu
            elif activations[i] == 'sigmoid':
                activation = ActivationFunctions.sigmoid
            elif activations[i] == 'tanh':
                activation = ActivationFunctions.tanh
            elif activations[i] == 'softmax':
                activation = ActivationFunctions.softmax
            else:
                raise ValueError(
                    f"Unsupported activation function: {activations[i]}")

            # Create layer
            self.layers.append(DenseLayer(
                layer_sizes[i],
                layer_sizes[i+1],
                activation,
                loss_function,
                weight_init,
                weight_init_params,
            ))

        # Set loss function
        if loss_function == 'mse':
            self.loss_fn = LossFunctions['mse']
        elif loss_function == 'binary_crossentropy':
            self.loss_fn = LossFunctions['binary_crossentropy']
        elif loss_function == 'categorical_crossentropy':
            self.loss_fn = LossFunctions['categorical_crossentropy']
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

    def forward(self, X, training=True):
        self.layer_outputs = []

        # Input layer
        current_output = X
        self.layer_outputs.append(current_output)

        # Hidden layers and output layer
        for layer in self.layers:
            current_output = layer.forward(current_output)
            self.layer_outputs.append(current_output)

        return current_output


    def backward(self, y_pred, y_true):
        dA = self.loss_fn.backward(y_pred, y_true)

        for i in reversed(range(len(self.layers))):
            dA = self.layers[i].backward(dA)

        # Base loss
        base_loss = self.loss_fn.forward(y_pred, y_true)

        # Regularization loss
        reg_loss = 0
        if self.lambda_l1 > 0:
            reg_loss += l1_regularization(self.layers, self.lambda_l1)
        if self.lambda_l2 > 0:
            reg_loss += l2_regularization(self.layers, self.lambda_l2)

        return base_loss + reg_loss


    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32,
              learning_rate=0.01, epochs=100, verbose=1):
        """
        Train the neural network

        Parameters:
        -----------
        X_train : array-like
            Training input data
        y_train : array-like
            Training target data
        X_val : array-like, optional
            Validation input data
        y_val : array-like, optional
            Validation target data
        batch_size : int
            Size of mini-batches
        learning_rate : float
            Learning rate for gradient descent
        epochs : int
            Number of training epochs
        verbose : int
            Verbosity level (0: silent, 1: progress bar)

        Returns:
        --------
        dict
            Training history containing loss values
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }

        n_samples = X_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(epochs):
            epoch_loss = 0

            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            if verbose == 1:
                batch_iterator = tqdm(
                    range(n_batches), desc=f'Epoch {epoch+1}/{epochs}')
            else:
                batch_iterator = range(n_batches)

            # Mini-batch training
            for batch_idx in batch_iterator:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Backward pass
                batch_loss = self.backward(y_pred, y_batch)
                epoch_loss += batch_loss * (end_idx - start_idx)

                # Update weights
                self.update_weights(learning_rate)

            avg_train_loss = epoch_loss / n_samples
            history['train_loss'].append(avg_train_loss)

            # Validation
            if X_val is not None and y_val is not None:
                val_predictions = self.forward(X_val, training=False)
                val_loss = self.loss_fn.forward(val_predictions, y_val)
                history['val_loss'].append(val_loss)

                if verbose == 1:
                    print(
                        f' - train_loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f}')
            else:
                if verbose == 1:
                    print(f' - train_loss: {avg_train_loss:.4f}')

        return history

    def predict(self, X):
        return self.forward(X, training=False)

    def save(self, filename):
        loss_fn_name = None
        for name, func in self.loss_fn.__class__.__dict__.items():
            if func is self.loss_fn:
                loss_fn_name = name
                break

        model_data = {
            'weights': [layer.weights for layer in self.layers],
            'biases': [layer.biases for layer in self.layers],
            'activations': [layer.activation.name for layer in self.layers],
            'loss_fn': loss_fn_name or 'unknown'
        }
        np.save(filename, model_data)
        print(f"Model saved to {filename}")

    def load(self, filename):
        model_data = np.load(filename, allow_pickle=True).item()

        for i, layer in enumerate(self.layers):
            layer.weights = model_data['weights'][i]
            layer.biases = model_data['biases'][i]

        print(f"Model loaded from {filename}")
