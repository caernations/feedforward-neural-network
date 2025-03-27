import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        """Base forward method to be implemented by subclasses"""
        pass
    
    def backward(self, y_pred, y_true):
        """Base backward method to be implemented by subclasses"""
        pass

class MSE(Loss):
    def forward(self, y_pred, y_true):
        """
        Mean Squared Error loss function
        MSE = (1/n) * Σ(y_true - y_pred)²
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, y_pred, y_true):
        """
        Derivative of MSE with respect to y_pred
        dMSE/dy_pred = -2(y_true - y_pred)/n
        """
        samples = len(y_pred)
        self.dinputs = -2 * (y_true - y_pred) / samples
        return self.dinputs

class BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        """
        Binary Cross-Entropy loss function
        BCE = -(1/n) * Σ(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        """
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    
    def backward(self, y_pred, y_true):
        """
        Derivative of BCE with respect to y_pred
        dBCE/dy_pred = -(1/n) * ((1-y_true)/(1-y_pred) - y_true/y_pred)
        """
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.dinputs = ((1 - y_true) / (1 - y_pred_clipped) - y_true / y_pred_clipped) / samples
        return self.dinputs

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        """
        Categorical Cross-Entropy loss function
        CCE = -(1/n) * Σ Σ y_true * log(y_pred)
        """
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        correct_confidences = np.sum(y_true * np.log(y_pred_clipped), axis=1)
        return -np.mean(correct_confidences)
    
    def backward(self, y_pred, y_true):
        """
        Derivative of CCE with respect to y_pred
        When used with softmax, this simplifies to (y_pred - y_true)/n
        """
        samples = len(y_pred)
        self.dinputs = y_pred.copy()
        self.dinputs[range(samples), np.argmax(y_true, axis=1)] -= 1
        self.dinputs /= samples
        return self.dinputs

LossFunctions = {
    "mse": MSE(),
    "binary_crossentropy": BinaryCrossEntropy(),
    "categorical_crossentropy": CategoricalCrossEntropy()
}