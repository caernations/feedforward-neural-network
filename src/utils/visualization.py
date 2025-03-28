import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

class NetworkVisualizer:
    def __init__(self, model):
        self.model = model
    
    def visualize_network_structure(self, max_neurons_per_layer=5, max_connections_per_neuron=3, 
                                    show_gradients=True, generate_gradients=False, X_batch=None, y_batch=None):
        """
        Menampilkan struktur jaringan beserta bobot dan gradien bobot tiap neuron dalam bentuk graf.
        """
        if generate_gradients and X_batch is not None and y_batch is not None:
            y_pred = self.model.forward(X_batch)
            loss = self.model.backward(y_pred, y_batch)
            print(f"Generated fresh gradients with loss: {loss}")
        
        layer_sizes = []
        for i, layer in enumerate(self.model.layers):
            if i == 0:
                layer_sizes.append(layer.input_size)
                layer_sizes.append(layer.output_size)
            else:
                layer_sizes.append(layer.output_size)
        
        if show_gradients:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            axes = [ax1, ax2]
            titles = ["Struktur Jaringan dengan Bobot", "Struktur Jaringan dengan Gradien Bobot"]
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            axes = [ax]
            titles = ["Struktur Jaringan dengan Bobot"]
        
        legend_elements = []
        
        for ax_idx, ax in enumerate(axes):
            ax.set_title(titles[ax_idx], fontsize=14)
            layer_x = np.linspace(0.1, 0.9, len(layer_sizes))
            plotted_neurons = {}
            
            for layer_idx, size in enumerate(layer_sizes):
                n_neurons = min(size, max_neurons_per_layer)
                indices = np.linspace(0, size-1, n_neurons).astype(int) if size > n_neurons else np.arange(size)
                y_positions = np.linspace(0.2, 0.8, n_neurons)

                if layer_idx == 0:
                    color = 'lightblue'  # Input layer
                elif layer_idx == len(layer_sizes) - 1:
                    color = 'salmon'     # Output layer
                else:
                    color = 'lightgreen' # Hidden layers
                
                ax.scatter([layer_x[layer_idx]] * n_neurons, y_positions, s=100, color=color, zorder=3)
                
                for i, idx in enumerate(indices):
                    plotted_neurons[(layer_idx, idx)] = (layer_x[layer_idx], y_positions[i])
                
                if size > n_neurons:
                    ax.text(layer_x[layer_idx], 0.1, f"Total: {size}", ha='center', fontsize=9)
            
            for layer_idx in range(len(self.model.layers)):
                layer = self.model.layers[layer_idx]
                
                weights = layer.weights
                
                if ax_idx == 0 or not show_gradients: 
                    connection_values = weights
                    positive_color = 'blue'
                    negative_color = 'red'
                    if ax_idx == 0:
                        legend_elements = [
                            Line2D([0], [0], color='blue', lw=2, label='Bobot Positif'),
                            Line2D([0], [0], color='red', lw=2, label='Bobot Negatif')
                        ]
                else: 
                    if hasattr(layer, 'dweights') and layer.dweights is not None:
                        connection_values = layer.dweights
                        positive_color = 'green'
                        negative_color = 'orange'
                        legend_elements = [
                            Line2D([0], [0], color='green', lw=2, label='Gradien Positif'),
                            Line2D([0], [0], color='orange', lw=2, label='Gradien Negatif')
                        ]
                    else:
                        continue
                
                from_neurons = [key[1] for key in plotted_neurons.keys() if key[0] == layer_idx]
                to_neurons = [key[1] for key in plotted_neurons.keys() if key[0] == layer_idx + 1]
                
                from_neurons = from_neurons[:max_neurons_per_layer]
                to_neurons = to_neurons[:max_neurons_per_layer]
                
                for i in from_neurons:
                    if i < layer.input_size:
                        # Select top connections by magnitude if there are too many
                        if len(to_neurons) > max_connections_per_neuron and len(to_neurons) > 0:
                            valid_to_neurons = [j for j in to_neurons if j < layer.output_size]
                            if valid_to_neurons:
                                connection_magnitudes = np.abs([connection_values[i, j] for j in valid_to_neurons])
                                if len(connection_magnitudes) > 0:
                                    # Select top connections
                                    selected_indices = np.argsort(connection_magnitudes)[-min(max_connections_per_neuron, len(connection_magnitudes)):]
                                    selected_to = [valid_to_neurons[j] for j in selected_indices if j < len(valid_to_neurons)]
                                else:
                                    selected_to = []
                            else:
                                selected_to = []
                        else:
                            selected_to = [j for j in to_neurons if j < layer.output_size]
                        
                        # Draw each selected connection
                        for j in selected_to:
                            if (layer_idx, i) in plotted_neurons and (layer_idx+1, j) in plotted_neurons:
                                if j < layer.output_size:  
                                    value = connection_values[i, j]
                                    color = negative_color if value < 0 else positive_color
                                    linewidth = min(abs(value) * 5 + 0.5, 3.0)
                                    
                                    x1, y1 = plotted_neurons[(layer_idx, i)]
                                    x2, y2 = plotted_neurons[(layer_idx+1, j)]
                                    
                                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.6, zorder=1)
                                    
            for i, label_idx in enumerate(range(len(layer_sizes))):
                if label_idx == 0:
                    label = "Input"
                elif label_idx == len(layer_sizes) - 1:
                    label = "Output"
                else:
                    label = f"Hidden {label_idx}"
                
                ax.text(layer_x[i], 0.95, label, ha='center', fontsize=10)
            
            legend_elements.extend([
                Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', label='Input Layer', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', label='Hidden Layer', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', label='Output Layer', markersize=10)
            ])
            
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=9)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        if show_gradients and any(hasattr(layer, 'dweights') and not np.all(np.isclose(layer.dweights, 0)) 
                                 for layer in self.model.layers):
            return True
        else:
            return False
    
    def visualize_weight_distribution(self, layers_to_plot=None, bins=30, sample_size=1000):
        """
        Menampilkan distribusi bobot dari tiap layer yang dipilih pada epoch terakhir
        """
        if layers_to_plot is None:
            layers_to_plot = range(len(self.model.layers))
        
        n_layers = len(layers_to_plot)
        fig, axes = plt.subplots(n_layers, 1, figsize=(10, 3*n_layers))
        
        if n_layers == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layers_to_plot):
            if layer_idx < len(self.model.layers):
                layer = self.model.layers[layer_idx]
                
                all_weights = layer.weights.flatten()
                
                if len(all_weights) > sample_size:
                    np.random.seed(42)  
                    indices = np.random.choice(len(all_weights), sample_size, replace=False)
                    sampled_weights = all_weights[indices]
                else:
                    sampled_weights = all_weights
                
                mean_weight = np.mean(all_weights)
                std_weight = np.std(all_weights)
                
                axes[i].hist(sampled_weights, bins=bins, alpha=0.7, color='skyblue', density=True)
                
                x = np.linspace(min(sampled_weights), max(sampled_weights), 100)
                axes[i].plot(x, 1/(std_weight * np.sqrt(2 * np.pi)) * 
                         np.exp(-(x - mean_weight)**2 / (2 * std_weight**2)), 
                         color='blue', lw=2)
                
                axes[i].set_title(f'Distribusi Bobot Layer {layer_idx}')
                axes[i].set_xlabel('Nilai Bobot')
                axes[i].set_ylabel('Frekuensi')
                
                stats_text = f'Mean: {mean_weight:.4f}, Std: {std_weight:.4f}'
                axes[i].text(0.02, 0.95, stats_text, transform=axes[i].transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                
                if hasattr(layer, 'biases') and layer.biases is not None:
                    bias_values = layer.biases.flatten()
                    if len(bias_values) > 0:
                        mean_bias = np.mean(bias_values)
                        axes[i].axvline(x=mean_bias, color='g', linestyle='--',
                                     label=f'Mean Bias: {mean_bias:.4f}')
                        axes[i].legend(loc='upper right')
            else:
                axes[i].text(0.5, 0.5, f'Layer {layer_idx} tidak ditemukan', 
                          horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.show()

    def visualize_gradient_distribution(self, layers_to_plot=None, bins=30, sample_size=1000, gradient_snapshot_idx=None):
        """
        Menampilkan distribusi gradien dari tiap layer yang dipilih pada epoch terakhir
        """
        if layers_to_plot is None:
            layers_to_plot = range(len(self.model.layers))
        
        n_layers = len(layers_to_plot)
        fig, axes = plt.subplots(n_layers, 1, figsize=(10, 3*n_layers))
        
        if n_layers == 1:
            axes = [axes]
        
        gradients_to_use = None
        if (gradient_snapshot_idx is not None and 
            hasattr(self.model, 'gradient_history') and 
            gradient_snapshot_idx < len(self.model.gradient_history)):
            gradients_to_use = self.model.gradient_history[gradient_snapshot_idx]['gradients']
            epoch_info = f" (Epoch {self.model.gradient_history[gradient_snapshot_idx]['epoch']}, Batch {self.model.gradient_history[gradient_snapshot_idx]['batch']})"
        else:
            epoch_info = " (Current)"
        
        for i, layer_idx in enumerate(layers_to_plot):
            if layer_idx < len(self.model.layers):
                # Determine which gradients to visualize
                if gradients_to_use is not None:
                    all_gradients = gradients_to_use[layer_idx]['dweights'].flatten()
                    bias_gradients = gradients_to_use[layer_idx]['dbiases'].flatten() if 'dbiases' in gradients_to_use[layer_idx] else None
                else:
                    layer = self.model.layers[layer_idx]
                    if hasattr(layer, 'dweights') and layer.dweights is not None:
                        all_gradients = layer.dweights.flatten()
                        bias_gradients = layer.dbiases.flatten() if hasattr(layer, 'dbiases') else None
                    else:
                        all_gradients = None
                        bias_gradients = None
                
                if all_gradients is not None and len(all_gradients) > 0:
                    if len(all_gradients) > sample_size:
                        np.random.seed(42)  # For reproducibility
                        indices = np.random.choice(len(all_gradients), sample_size, replace=False)
                        sampled_gradients = all_gradients[indices]
                    else:
                        sampled_gradients = all_gradients
                    
                    # Calculate statistics
                    mean_grad = np.mean(all_gradients)
                    std_grad = np.std(all_gradients)
                    
                    # Handle non-zero gradients
                    if not np.all(sampled_gradients == 0) and std_grad > 1e-10:
                        axes[i].hist(sampled_gradients, bins=bins, alpha=0.7, color='skyblue', density=True)
                        
                        # Add normal distribution curve
                        x = np.linspace(min(sampled_gradients), max(sampled_gradients), 100)
                        axes[i].plot(x, 1/(std_grad * np.sqrt(2 * np.pi)) * 
                                np.exp(-(x - mean_grad)**2 / (2 * std_grad**2)), 
                                color='red', lw=2)
                        
                        axes[i].set_title(f'Distribusi Gradien Bobot Layer {layer_idx}{epoch_info}')
                        axes[i].set_xlabel('Nilai Gradien')
                        axes[i].set_ylabel('Frekuensi')
                        
                        stats_text = f'Mean: {mean_grad:.6f}, Std: {std_grad:.6f}'
                        axes[i].text(0.02, 0.95, stats_text, transform=axes[i].transAxes,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                        
                        if bias_gradients is not None and len(bias_gradients) > 0:
                            mean_bias_grad = np.mean(bias_gradients)
                            axes[i].axvline(x=mean_bias_grad, color='r', linestyle='--',
                                        label=f'Mean Bias Grad: {mean_bias_grad:.6f}')
                            axes[i].legend(loc='upper right')
                    else:
                        axes[i].set_title(f'Distribusi Gradien Bobot Layer {layer_idx}{epoch_info}')
                        axes[i].set_xlabel('Nilai Gradien')
                        axes[i].set_ylabel('Frekuensi')
                        
                        if np.all(sampled_gradients == 0):
                            axes[i].hist([0] * 30, bins=bins, alpha=0.7, color='skyblue', density=True)
                            axes[i].text(0.5, 0.5, 'Semua gradien bernilai 0', 
                                    horizontalalignment='center', verticalalignment='center',
                                    transform=axes[i].transAxes, fontsize=12,
                                    bbox=dict(facecolor='white', alpha=0.8))
                        else:
                            axes[i].hist(sampled_gradients, bins=bins, alpha=0.7, color='skyblue', density=True)
                            axes[i].text(0.5, 0.5, f'Gradien sangat kecil (std ≈ 0)', 
                                    horizontalalignment='center', verticalalignment='center',
                                    transform=axes[i].transAxes, fontsize=12,
                                    bbox=dict(facecolor='white', alpha=0.8))
                        
                        stats_text = f'Mean: {mean_grad:.6f}, Std: {std_grad:.6f}'
                        axes[i].text(0.02, 0.95, stats_text, transform=axes[i].transAxes,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                        
                        # Add bias gradient info if available
                        if bias_gradients is not None and len(bias_gradients) > 0:
                            mean_bias_grad = np.mean(bias_gradients)
                            axes[i].axvline(x=mean_bias_grad, color='r', linestyle='--',
                                        label=f'Mean Bias Grad: {mean_bias_grad:.6f}')
                            axes[i].legend(loc='upper right')
                else:
                    axes[i].set_title(f'Layer {layer_idx}')
                    axes[i].text(0.5, 0.5, f'Layer {layer_idx} tidak memiliki gradien bobot (dweights)', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=axes[i].transAxes, fontsize=12)
            else:
                axes[i].set_title(f'Layer {layer_idx}')
                axes[i].text(0.5, 0.5, f'Layer {layer_idx} tidak ditemukan', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[i].transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.show()