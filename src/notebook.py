import numpy as np
import matplotlib.pyplot as plt
import copy
from models.ffnn import FeedForwardNN
from utils.dataset import load_mnist
from configs.config_loader import load_config
from utils.visualization import NetworkVisualizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X_train, X_test, y_train, y_test = load_mnist()
val_size = int(len(X_train) * 0.2)
X_val, y_val = X_train[:val_size], y_train[:val_size]
X_train, y_train = X_train[val_size:], y_train[val_size:]

def compute_accuracy(model, X, y):
    predictions = model.forward(X, training=False)
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))

def train_and_evaluate(config):
    model = FeedForwardNN(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        output_size=config['model']['output_size'],
        activations=config['model']['activations'],
        weight_init=config['model']['weight_init'],
        loss_function=config['model']['loss_function']
    )
    
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        epochs=config['training']['epochs']
    )
    
    train_acc = compute_accuracy(model, X_train, y_train)
    val_acc = compute_accuracy(model, X_val, y_val)
    test_acc = compute_accuracy(model, X_test, y_test)
    
    return model, history, (train_acc, val_acc, test_acc)

# Analisis Depth dan Width
base_config = load_config()
width_configs = [
    {'hidden_layers': [64, 64], 'name': 'Width 64x2'},
    {'hidden_layers': [128, 128], 'name': 'Width 128x2'},
    {'hidden_layers': [256, 256], 'name': 'Width 256x2'}
]

width_results = {}
for cfg in width_configs:
    modified_config = copy.deepcopy(base_config)
    modified_config['model']['hidden_layers'] = cfg['hidden_layers']
    model, history, accs = train_and_evaluate(modified_config)
    width_results[cfg['name']] = {'history': history, 'accs': accs}

depth_configs = [
    {'hidden_layers': [128], 'name': 'Depth 1x128'},
    {'hidden_layers': [128, 128], 'name': 'Depth 2x128'},
    {'hidden_layers': [128, 128, 128], 'name': 'Depth 3x128'}
]

depth_results = {}
for cfg in depth_configs:
    modified_config = copy.deepcopy(base_config)
    modified_config['model']['hidden_layers'] = cfg['hidden_layers']
    model, history, accs = train_and_evaluate(modified_config)
    depth_results[cfg['name']] = {'history': history, 'accs': accs}

# Plot hasil
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
for name, res in width_results.items():
    plt.plot(res['history']['train_loss'], label=name)
plt.title('Training Loss (Variasi Width)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
for name, res in depth_results.items():
    plt.plot(res['history']['train_loss'], label=name)
plt.title('Training Loss (Variasi Depth)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Analisis Fungsi Aktivasi
base_config = load_config()
activation_configs = [
    {'activation': 'sigmoid', 'name': 'Sigmoid'},
    {'activation': 'tanh', 'name': 'Tanh'},
    {'activation': 'relu', 'name': 'ReLU'},
    {'activation': 'linear', 'name': 'Linear'},
    {'activation': 'softmax', 'name': 'Softmax'}
]
activation_results = {}
for cfg in activation_configs:
    modified_config = copy.deepcopy(base_config)
    modified_config['model']['activations'] = [cfg['activation']] * len(modified_config['model']['hidden_layers']) + ['softmax']
    model, history, accs = train_and_evaluate(modified_config)
    activation_results[cfg['name']] = {'history': history, 'accs': accs, 'model': model}

plt.figure(figsize=(10, 5))
for name, res in activation_results.items():
    plt.plot(res['history']['train_loss'], label=name)
plt.title('Training Loss (Fungsi Aktivasi)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Analisis Learning Rate
lr_configs = [
    {'lr': 0.01, 'name': 'LR 0.01'},
    {'lr': 0.01, 'name': 'LR 0.05'},
    {'lr': 0.1, 'name': 'LR 0.1'},
    {'lr': 0.5, 'name': 'LR 0.5'}
]
lr_results = {}
for cfg in lr_configs:
    modified_config = copy.deepcopy(base_config)
    modified_config['training']['learning_rate'] = cfg['lr']
    model, history, accs = train_and_evaluate(modified_config)
    lr_results[cfg['name']] = {'history': history, 'accs': accs, 'model': model}

plt.figure(figsize=(10, 5))
for name, res in lr_results.items():
    plt.plot(res['history']['train_loss'], label=name)
plt.title('Training Loss (Learning Rate)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Analisis Inisialisasi Bobot
init_configs = [
    {'init': 'xavier', 'name': 'Xavier'},
    {'init': 'zero', 'name': 'Zero'},
    {'init': 'uniform', 'name': 'Uniform'},
    {'init': 'he', 'name': 'He'},
    {'init': 'normal', 'name': 'Normal'},
]
init_results = {}
for cfg in init_configs:
    modified_config = copy.deepcopy(base_config)
    modified_config['model']['weight_init'] = cfg['init']
    model, history, accs = train_and_evaluate(modified_config)
    init_results[cfg['name']] = {'history': history, 'accs': accs, 'model': model}

plt.figure(figsize=(10, 5))
for name, res in init_results.items():
    plt.plot(res['history']['train_loss'], label=name)
plt.title('Training Loss (Inisialisasi Bobot)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Perbandingan dengan Sklearn MLP
base_config = load_config()
ffnn_model, _, (_, _, ffnn_test_acc) = train_and_evaluate(base_config)
mlp = MLPClassifier(
    hidden_layer_sizes=base_config['model']['hidden_layers'],
    activation='relu',
    learning_rate_init=base_config['training']['learning_rate'],
    max_iter=base_config['training']['epochs'],
    batch_size=base_config['training']['batch_size']
)
mlp.fit(X_train, np.argmax(y_train, axis=1))
sklearn_test_acc = mlp.score(X_test, np.argmax(y_test, axis=1))
print(f"Test Accuracy:\nFFNN: {ffnn_test_acc:.4f}\nSklearn: {sklearn_test_acc:.4f}")
