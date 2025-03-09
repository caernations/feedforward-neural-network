import numpy as np
from utils.dataset import load_mnist
from models.ffnn import FeedForwardNN

X_train, X_test, y_train, y_test = load_mnist()

model = FeedForwardNN(
    input_size=784,
    hidden_layers=[256, 128, 64],
    output_size=10,
    activations=["relu", "relu", "relu", "softmax"],
    learning_rate=0.01  
)

model.train(X_train, y_train, epochs=200)

def evaluate(model, X_test, y_test):
    predictions = model.forward(X_test)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
    print(f"Accuracy: {accuracy * 100:.2f}%")

evaluate(model, X_test, y_test)
