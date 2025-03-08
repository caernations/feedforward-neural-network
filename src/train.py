from utils.dataset import load_mnist

X_train, X_test, y_train, y_test = load_mnist()
print(f"Data train: {X_train.shape}, Label train: {y_train.shape}")
print(f"Data test: {X_test.shape}, Label test: {y_test.shape}")
