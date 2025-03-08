from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split

def load_mnist(test_split=0.2, normalize=True, one_hot=True):
    print("downloading")
    mnist = fetch_openml('mnist_784', version=1, parser='pandas', as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)  
    print(f"loaded successfully! Shape: {X.shape}, Labels: {y.shape}")

    if normalize:
        X = X / 255.0
    if one_hot:
        y = np.eye(10)[y] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    print(f"dataset displit menjadi {len(X_train)} training data sdan {len(X_test)} testing data.")

    return X_train, X_test, y_train, y_test
