import numpy as np
from utils.dataset import load_mnist
from models.ffnn import FeedForwardNN
from configs.config_loader import load_config

def main():
    # Load configuration
    config = load_config()

    # Load MNIST dataset
    X_train, X_test, y_train, y_test = load_mnist()

    # Calculate validation split
    val_size = int(len(X_train) * config['data']['validation_split'])
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]

    # Create model with configuration
    model = FeedForwardNN(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        output_size=config['model']['output_size'],
        activations=config['model']['activations'],
        weight_init=config['model']['weight_init'],
        weight_init_params=config['weight_init_params'],
        loss_function=config['model']['loss_function']
    )

    # Train the model
    history = model.train(
        X_train, y_train, 
        X_val=X_val, y_val=y_val,
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        epochs=config['training']['epochs'],
        verbose=config['training']['verbose']
    )

    # Evaluate the model
    def evaluate(model, X_test, y_test):
        predictions = model.forward(X_test)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))
        print(f"Accuracy: {accuracy * 100:.2f}%")

    evaluate(model, X_test, y_test)

    # Save the model
    model.save(config['save']['model_path'])

if __name__ == "__main__":
    main()