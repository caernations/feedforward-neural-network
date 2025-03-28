import numpy as np
import matplotlib.pyplot as plt
from configs.config_loader import load_config
from models.ffnn import FeedForwardNN
from utils.visualization import NetworkVisualizer
from utils.dataset import load_mnist

def main():
    config = load_config()

    print("\nMemuat dataset MNIST...")
    X_train, X_test, y_train, y_test = load_mnist()

    val_size = int(len(X_train) * config['data']['validation_split'])  
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:] 
    y_train = y_train[val_size:]

    print("\nMembangun model FFNN...")
    model = FeedForwardNN(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        output_size=config['model']['output_size'],
        activations=config['model']['activations'],
        weight_init=config['model']['weight_init'],
        weight_init_params=config.get('weight_init_params', None),
        loss_function=config['model']['loss_function']
    )

    # Training model
    print("\nMemulai proses training...")
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        epochs=config['training']['epochs'],
        verbose=config['training']['verbose']
    )

    model.save("saved_model.npy")
    print("\nModel telah disimpan sebagai 'saved_model.npy'")

    print("\nMemuat model untuk testing...")
    loaded_model = FeedForwardNN(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        output_size=config['model']['output_size'],
        activations=config['model']['activations'],
        weight_init=config['model']['weight_init'],
        weight_init_params=config.get('weight_init_params', None),
        loss_function=config['model']['loss_function']
    )
    loaded_model.load("saved_model.npy")

    # Prepare a sample batch for gradient visualization
    print("\nMenyiapkan batch data untuk visualisasi gradien...")
    batch_size = 32
    random_indices = np.random.choice(len(X_train), batch_size)
    X_batch = X_train[random_indices]
    y_batch = y_train[random_indices]

    # Create visualizer
    visualizer = NetworkVisualizer(loaded_model)

    print("\nVisualisasi struktur jaringan dengan bobot dan gradien:")
    visualizer.visualize_network_structure(
        max_neurons_per_layer=5,
        max_connections_per_neuron=3,
        show_gradients=True,
        generate_gradients=True,
        X_batch=X_batch,
        y_batch=y_batch
    )

    # Visualize weight distributions
    print("\nVisualisasi distribusi bobot:")
    visualizer.visualize_weight_distribution()
    
    # Visualize weight distributions for selected layers
    print("\nVisualisasi distribusi bobot layer tertentu:")
    visualizer.visualize_weight_distribution([0, 1])
    
    # Visualize gradient distributions
    print("\nVisualisasi distribusi gradien bobot:")
    visualizer.visualize_gradient_distribution()

    def compute_accuracy(model, X, y):
        predictions = model.forward(X, training=False)
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))

    print("\nEvaluasi model:")
    train_acc = compute_accuracy(loaded_model, X_train, y_train)
    val_acc = compute_accuracy(loaded_model, X_val, y_val)
    test_acc = compute_accuracy(loaded_model, X_test, y_test)

    plt.figure(figsize=(10, 6))
    accuracies = [train_acc, val_acc, test_acc]
    plt.bar(['Train', 'Validation', 'Test'], accuracies, color=['blue', 'orange', 'green'])
    plt.title('Perbandingan Akurasi Model')
    plt.ylabel('Akurasi')
    plt.ylim(0, 1)
    
    for i, acc in enumerate(accuracies):
        plt.text(i, acc, f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

    print("\nAnalisis Overfitting:")
    print(f"Akurasi Training:   {train_acc * 100:.2f}%")
    print(f"Akurasi Validasi:   {val_acc * 100:.2f}%")
    print(f"Akurasi Test:       {test_acc * 100:.2f}%")

    if train_acc > val_acc + 0.1:
        print("\n⚠️ Terdeteksi Potensi Overfitting!")
    elif val_acc > train_acc + 0.1:
        print("\n⚠️ Terdeteksi Potensi Underfitting!")
    else:
        print("\n✅ Model menunjukkan performa yang konsisten")

if __name__ == "__main__":
    main()