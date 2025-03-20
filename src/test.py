import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import load_mnist
from models.ffnn import FeedForwardNN
from configs.config_loader import load_config

def analyze_overfitting():
    # Load configuration
    config = load_config()

    # Load MNIST dataset
    X_train, X_test, y_train, y_test = load_mnist()

    # Calculate validation split based on config
    val_size = int(len(X_train) * config['data']['validation_split'])
    X_train_split = X_train[val_size:]
    y_train_split = y_train[val_size:]
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]

    # Create model using configuration
    model = FeedForwardNN(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        output_size=config['model']['output_size'],
        activations=config['model']['activations'],
        weight_init=config['model']['weight_init'],
        weight_init_params=config.get('weight_init_params', None),
        loss_function=config['model']['loss_function']
    )

    # Train with validation data using config parameters
    history = model.train(
        X_train_split, y_train_split, 
        X_val=X_val, y_val=y_val,
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'], 
        epochs=config['training']['epochs'], 
        verbose=config['training']['verbose']
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

    if 'save' in config and 'model_path' in config['save']:
        model.save(config['save']['model_path'])

if __name__ == "__main__":
    analyze_overfitting()