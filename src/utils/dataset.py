from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split

def load_mnist(normalize=True, one_hot=True):
    print("Downloading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, parser='pandas', as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    
    # Split original MNIST (60k train, 10k test)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    print(f"Original split - Train: {len(X_train)}, Test: {len(X_test)}")

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
    if one_hot:
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]

    return X_train, X_test, y_train, y_test