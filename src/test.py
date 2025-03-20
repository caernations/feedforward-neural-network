import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import load_mnist
from models.ffnn import FeedForwardNN

def analyze_overfitting():
    X_train, X_test, y_train, y_test = load_mnist()

    # 80% training, 20% validation
    train_size = int(0.8 * len(X_train))
    X_train_split = X_train[:train_size]
    y_train_split = y_train[:train_size]
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]

    model = FeedForwardNN(
        input_size=784,
        hidden_layers=[256, 128, 64],
        output_size=10,
        activations=["relu", "relu", "relu", "softmax"]
    )

    # Train with validation data
    history = model.train(
        X_train_split, y_train_split, 
        X_val=X_val, y_val=y_val,
        learning_rate=0.01, 
        epochs=20, 
        verbose=1
    )

    # Visualize training progress
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    def compute_accuracy(X, y):
        predictions = model.forward(X)
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))

    train_accuracy = compute_accuracy(X_train_split, y_train_split)
    val_accuracy = compute_accuracy(X_val, y_val)
    test_accuracy = compute_accuracy(X_test, y_test)

    plt.subplot(1, 2, 2)
    accuracies = [train_accuracy, val_accuracy, test_accuracy]
    plt.bar(['Train', 'Validation', 'Test'], accuracies)
    plt.title('Model Accuracies')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    for i, acc in enumerate(accuracies):
        plt.text(i, acc, f'{acc:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    # Print detailed accuracy information
    print("\nOverfitting Analysis:")
    print(f"Training Accuracy:   {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy:       {test_accuracy * 100:.2f}%")

    # Detect potential overfitting
    if train_accuracy > val_accuracy + 0.1:
        print("\n⚠️ Potential Overfitting Detected!")
        print("Training accuracy is significantly higher than validation accuracy.")
    elif val_accuracy > train_accuracy + 0.1:
        print("\n⚠️ Potential Underfitting Detected!")
        print("Validation accuracy is significantly higher than training accuracy.")
    else:
        print("\n✅ No Strong Signs of Overfitting")
        print("Model performance seems consistent across training and validation sets.")

if __name__ == "__main__":
    analyze_overfitting()