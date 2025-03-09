import numpy as np

class CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) 
        correct_confidences = np.sum(y_true * np.log(y_pred_clipped), axis=1)
        return -np.mean(correct_confidences)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        y_true = y_true.astype(int)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), np.argmax(y_true, axis=1)] -= 1
        self.dinputs /= samples
        return self.dinputs

LossFunctions = {
    "categorical_crossentropy": CategoricalCrossEntropy()
}
