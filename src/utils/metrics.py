import numpy as np

def accuracy_score(y_true, y_pred):
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    
    return np.mean(y_true == y_pred)

def cross_entropy_loss(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss
