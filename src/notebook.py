import numpy as np
import matplotlib.pyplot as plt
import copy
from models.ffnn import FeedForwardNN
from utils.dataset import load_mnist
from configs.config_loader import load_config
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load dataset
X_train, X_test, y_train, y_test = load_mnist()
val_size = int(len(X_train) * 0.2)
X_val, y_val = X_train[:val_size], y_train[:val_size]
X_train, y_train = X_train[val_size:], y_train[val_size:]

def compute_accuracy(model, X, y):
    predictions = model.forward(X, training=False)
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))

def train_and_evaluate(config):
    # Create model with base parameters
    model = FeedForwardNN(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        output_size=config['model']['output_size'],
        activations=config['model']['activations'],
        weight_init=config['model']['weight_init'],
        weight_init_params=config.get('weight_init_params', None),
        loss_function=config['model']['loss_function']
    )
    
    # Apply regularization if implemented in the model
    if hasattr(model, 'set_regularization'):
        if 'regularization' in config['model'] and config['model']['regularization']:
            if config['model']['regularization'] == 'l1':
                lambda_val = config['model'].get('lambda_l1', 0.0)
                model.set_regularization('l1', lambda_val)
            elif config['model']['regularization'] == 'l2':
                lambda_val = config['model'].get('lambda_l2', 0.0)
                model.set_regularization('l2', lambda_val)
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        epochs=config['training']['epochs'],
        verbose=config['training']['verbose']
    )
    
    # Evaluate model
    train_acc = compute_accuracy(model, X_train, y_train)
    val_acc = compute_accuracy(model, X_val, y_val)
    test_acc = compute_accuracy(model, X_test, y_test)
    
    return model, history, (train_acc, val_acc, test_acc)

def summarize_results(results_dict, title):
    """Visualize accuracy results for each experiment"""
    names = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    
    for name, res in results_dict.items():
        train_acc, val_acc, test_acc = res['accs']
        names.append(name)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

    plt.figure(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.25
    
    plt.bar(x - width, train_accuracies, width, label='Train', color='skyblue')
    plt.bar(x, val_accuracies, width, label='Validation', color='lightgreen')
    plt.bar(x + width, test_accuracies, width, label='Test', color='salmon')
    
    plt.xlabel('Configuration')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(x, names, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print detailed results in table format
    print("\nDetailed results:")
    print(f"{'Configuration':<20} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10}")
    print("-" * 50)
    for i, name in enumerate(names):
        print(f"{name:<20} {train_accuracies[i]:.4f}     {val_accuracies[i]:.4f}     {test_accuracies[i]:.4f}")

def plot_loss_histories(results_dict, title):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 6))
    
    for name, res in results_dict.items():
        plt.plot(res['history']['train_loss'], label=f"{name} - Train")
        if 'val_loss' in res['history']:
            plt.plot(res['history']['val_loss'], linestyle='--', label=f"{name} - Val")

    plt.title(f'Training Loss vs Validation Loss ({title})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def visualize_weight_distribution(model, layer_idx, title):
    """Visualize weight distribution for a specific layer"""
    if layer_idx < len(model.layers):
        weights = model.layers[layer_idx].weights.flatten()
        
        plt.figure(figsize=(10, 5))
        plt.hist(weights, bins=50, alpha=0.7, color='blue')
        plt.title(f"{title} - Layer {layer_idx+1} Weight Distribution")
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"Weight stats for {title} - Layer {layer_idx+1}:")
        print(f"  Mean: {np.mean(weights):.6f}")
        print(f"  Std Dev: {np.std(weights):.6f}")
        print(f"  Min: {np.min(weights):.6f}")
        print(f"  Max: {np.max(weights):.6f}")
    else:
        print(f"Layer index {layer_idx} out of range for model with {len(model.layers)} layers")

def visualize_gradient_distribution(model, X_batch, y_batch, layer_idx, title):
    """Generate and visualize gradient distribution for a specific layer"""
    # Forward pass
    y_pred = model.forward(X_batch, training=True)
    
    # Backward pass to generate gradients
    model.backward(y_pred, y_batch)
    
    # Get gradients if available
    if hasattr(model.layers[layer_idx], 'dW'):
        gradients = model.layers[layer_idx].dW.flatten()
        
        plt.figure(figsize=(10, 5))
        plt.hist(gradients, bins=50, alpha=0.7, color='green')
        plt.title(f"{title} - Layer {layer_idx+1} Gradient Distribution")
        plt.xlabel('Gradient Value')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"Gradient stats for {title} - Layer {layer_idx+1}:")
        print(f"  Mean: {np.mean(gradients):.6f}")
        print(f"  Std Dev: {np.std(gradients):.6f}")
        print(f"  Min: {np.min(gradients):.6f}")
        print(f"  Max: {np.max(gradients):.6f}")
    else:
        print(f"No gradient (dW) attribute found for layer {layer_idx+1}")

def visualize_model_distributions(results_dict, title):
    """Visualize weight and gradient distributions for all models in results"""
    X_batch = X_train[:32]  # Small batch for generating gradients
    y_batch = y_train[:32]
    
    for name, res in results_dict.items():
        if 'model' in res:
            model = res['model']
            print(f"\n===== {title} - {name} =====")
            
            # Visualize weights for each layer
            for i in range(len(model.layers)):
                visualize_weight_distribution(model, i, f"{name}")
            
            # Visualize gradients for each layer
            for i in range(len(model.layers)):
                visualize_gradient_distribution(model, X_batch, y_batch, i, f"{name}")

# Load the base configuration
base_config = {
    'model': {
        'input_size': 784,
        'hidden_layers': [256, 128, 64],
        'output_size': 10,
        'activations': ["relu", "relu", "relu", "softmax"],
        'weight_init': "xavier",
        'loss_function': "categorical_crossentropy",
        'lambda_l1': 0.0,
        'lambda_l2': 1e-5
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.01,
        'epochs': 10,
        'verbose': 1
    },
    'weight_init_params': {
        'lower_bound': -0.1,
        'upper_bound': 0.1,
        'seed': 42,
        'mean': 0,
        'variance': 0.1
    },
    'data': {
        'validation_split': 0.2
    },
    'save': {
        'model_path': "models/saved_model.npy",
        'save_frequency': 10
    }
}

# 1. Experiment: Depth and Width
print("\n===== EXPERIMENT: DEPTH AND WIDTH =====")

# Width experiment (fixed 2-layer depth)
width_configs = [
    {'hidden_layers': [64, 64], 'name': 'Width 64x2'},
    {'hidden_layers': [128, 128], 'name': 'Width 128x2'},
    {'hidden_layers': [256, 256], 'name': 'Width 256x2'}
]

width_results = {}
for cfg in width_configs:
    print(f"\nTraining model with {cfg['name']}...")
    modified_config = copy.deepcopy(base_config)
    modified_config['model']['hidden_layers'] = cfg['hidden_layers']
    modified_config['model']['activations'] = ['relu'] * len(cfg['hidden_layers']) + ['softmax']
    model, history, accs = train_and_evaluate(modified_config)
    width_results[cfg['name']] = {'history': history, 'accs': accs, 'model': model}

# Depth experiment (fixed 128 neurons per layer)
depth_configs = [
    {'hidden_layers': [128], 'name': 'Depth 1x128'},
    {'hidden_layers': [128, 128], 'name': 'Depth 2x128'},
    {'hidden_layers': [128, 128, 128], 'name': 'Depth 3x128'}
]

depth_results = {}
for cfg in depth_configs:
    print(f"\nTraining model with {cfg['name']}...")
    modified_config = copy.deepcopy(base_config)
    modified_config['model']['hidden_layers'] = cfg['hidden_layers']
    modified_config['model']['activations'] = ['relu'] * len(cfg['hidden_layers']) + ['softmax']
    model, history, accs = train_and_evaluate(modified_config)
    depth_results[cfg['name']] = {'history': history, 'accs': accs, 'model': model}

# Plot results
plot_loss_histories(width_results, "Width Variation")
summarize_results(width_results, 'Accuracy Comparison (Width Variation)')

plot_loss_histories(depth_results, "Depth Variation")
summarize_results(depth_results, 'Accuracy Comparison (Depth Variation)')

# 2. Experiment: Activation Functions
print("\n===== EXPERIMENT: ACTIVATION FUNCTIONS =====")

# Test different activation functions for hidden layers
activation_configs = [
    {'activation': 'sigmoid', 'name': 'Sigmoid'},
    {'activation': 'tanh', 'name': 'Tanh'},
    {'activation': 'relu', 'name': 'ReLU'},
    {'activation': 'linear', 'name': 'Linear'}
]

activation_results = {}
for cfg in activation_configs:
    print(f"\nTraining model with {cfg['name']} activation...")
    modified_config = copy.deepcopy(base_config)
    # Apply this activation to all hidden layers, keep softmax for output
    hidden_layer_count = len(modified_config['model']['hidden_layers'])
    modified_config['model']['activations'] = [cfg['activation']] * hidden_layer_count + ['softmax']
    model, history, accs = train_and_evaluate(modified_config)
    activation_results[cfg['name']] = {'history': history, 'accs': accs, 'model': model}

plot_loss_histories(activation_results, "Activation Functions")
summarize_results(activation_results, 'Accuracy Comparison (Activation Functions)')
visualize_model_distributions(activation_results, "Activation Functions")

# 3. Experiment: Learning Rate
print("\n===== EXPERIMENT: LEARNING RATE =====")

lr_configs = [
    {'lr': 0.001, 'name': 'LR 0.001'},
    {'lr': 0.01, 'name': 'LR 0.01'},
    {'lr': 0.1, 'name': 'LR 0.1'}
]

lr_results = {}
for cfg in lr_configs:
    print(f"\nTraining model with {cfg['name']}...")
    modified_config = copy.deepcopy(base_config)
    modified_config['training']['learning_rate'] = cfg['lr']
    model, history, accs = train_and_evaluate(modified_config)
    lr_results[cfg['name']] = {'history': history, 'accs': accs, 'model': model}

plot_loss_histories(lr_results, "Learning Rate")
summarize_results(lr_results, 'Accuracy Comparison (Learning Rate)')
visualize_model_distributions(lr_results, "Learning Rate")

# 4. Experiment: Weight Initialization
print("\n===== EXPERIMENT: WEIGHT INITIALIZATION =====")

init_configs = [
    {'init': 'xavier', 'name': 'Xavier'},
    {'init': 'zero', 'name': 'Zero'},
    {'init': 'uniform', 'name': 'Uniform'},
    {'init': 'he', 'name': 'He'},
    {'init': 'normal', 'name': 'Normal'}
]

init_results = {}
for cfg in init_configs:
    print(f"\nTraining model with {cfg['name']} initialization...")
    modified_config = copy.deepcopy(base_config)
    modified_config['model']['weight_init'] = cfg['init']
    model, history, accs = train_and_evaluate(modified_config)
    init_results[cfg['name']] = {'history': history, 'accs': accs, 'model': model}

plot_loss_histories(init_results, "Weight Initialization")
summarize_results(init_results, 'Accuracy Comparison (Weight Initialization)')
visualize_model_distributions(init_results, "Weight Initialization")

# 5. Experiment: Regularization
print("\n===== EXPERIMENT: REGULARIZATION =====")

reg_configs = [
    {'name': 'No Regularization', 'lambda_l1': 0.0, 'lambda_l2': 0.0},
    {'name': 'L1 Regularization', 'lambda_l1': 0.0001, 'lambda_l2': 0.0},
    {'name': 'L2 Regularization', 'lambda_l1': 0.0, 'lambda_l2': 0.0001}
]

reg_results = {}
for cfg in reg_configs:
    print(f"\nTraining model with {cfg['name']}...")
    
    # Create model with base parameters from config
    model = FeedForwardNN(
        input_size=base_config['model']['input_size'],
        hidden_layers=base_config['model']['hidden_layers'],
        output_size=base_config['model']['output_size'],
        activations=base_config['model']['activations'],
        weight_init=base_config['model']['weight_init'],
        weight_init_params=base_config.get('weight_init_params', None),
        loss_function=base_config['model']['loss_function']
    )
    
    # Set regularization directly on the model
    if hasattr(model, 'set_regularization'):
        if cfg['lambda_l1'] > 0:
            model.set_regularization('l1', cfg['lambda_l1'])
        elif cfg['lambda_l2'] > 0:
            model.set_regularization('l2', cfg['lambda_l2'])
    
    # Train model
    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        batch_size=base_config['training']['batch_size'],
        learning_rate=base_config['training']['learning_rate'],
        epochs=base_config['training']['epochs'],
        verbose=base_config['training']['verbose']
    )
    
    # Evaluate model
    train_acc = compute_accuracy(model, X_train, y_train)
    val_acc = compute_accuracy(model, X_val, y_val)
    test_acc = compute_accuracy(model, X_test, y_test)
    
    reg_results[cfg['name']] = {
        'history': history, 
        'accs': (train_acc, val_acc, test_acc), 
        'model': model
    }

plot_loss_histories(reg_results, "Regularization")
summarize_results(reg_results, 'Accuracy Comparison (Regularization)')
visualize_model_distributions(reg_results, "Regularization")

# 6. Comparison with Sklearn MLP
print("\n===== COMPARISON WITH SKLEARN MLP =====")

# Train our custom FFNN model with the base configuration
print("\nTraining custom FFNN model...")
ffnn_model, ffnn_history, (ffnn_train_acc, ffnn_val_acc, ffnn_test_acc) = train_and_evaluate(base_config)

# Train sklearn's MLPClassifier with similar parameters
print("\nTraining sklearn MLP model...")
mlp = MLPClassifier(
    hidden_layer_sizes=tuple(base_config['model']['hidden_layers']),
    activation='relu',
    solver='adam',
    alpha=base_config['model'].get('lambda_l2', 0.0001),  # L2 regularization
    learning_rate_init=base_config['training']['learning_rate'],
    max_iter=base_config['training']['epochs'],
    batch_size=base_config['training']['batch_size'],
    random_state=42
)

mlp.fit(X_train, np.argmax(y_train, axis=1))
sklearn_train_acc = mlp.score(X_train, np.argmax(y_train, axis=1))
sklearn_val_acc = mlp.score(X_val, np.argmax(y_val, axis=1))
sklearn_test_acc = mlp.score(X_test, np.argmax(y_test, axis=1))

# Compare results
print(f"\nAccuracy Comparison:")
print(f"                  Train       Validation  Test")
print(f"Custom FFNN:      {ffnn_train_acc:.4f}    {ffnn_val_acc:.4f}    {ffnn_test_acc:.4f}")
print(f"Sklearn MLP:      {sklearn_train_acc:.4f}    {sklearn_val_acc:.4f}    {sklearn_test_acc:.4f}")

# Visualize comparison
plt.figure(figsize=(10, 6))
models = ['Custom FFNN', 'Sklearn MLP']
train_accs = [ffnn_train_acc, sklearn_train_acc]
val_accs = [ffnn_val_acc, sklearn_val_acc]
test_accs = [ffnn_test_acc, sklearn_test_acc]

x = np.arange(len(models))
width = 0.25

plt.bar(x - width, train_accs, width, label='Train', color='skyblue')
plt.bar(x, val_accs, width, label='Validation', color='lightgreen')
plt.bar(x + width, test_accs, width, label='Test', color='salmon')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Custom FFNN vs Sklearn MLP')
plt.xticks(x, models)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\n===== ALL EXPERIMENTS COMPLETED =====")