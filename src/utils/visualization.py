import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns

class NetworkVisualizer:
    """
    Kelas untuk memvisualisasikan jaringan saraf tiruan (neural network)
    """
    
    def __init__(self, model):
        """
        Inisialisasi visualizer dengan model neural network yang sudah dilatih
        
        Parameters:
        model : objek model neural network yang sudah selesai training
        """
        self.model = model
    
    def visualize_network_structure(self, max_neurons_per_layer=5, max_connections_per_neuron=3):
        """
        Menampilkan struktur jaringan beserta bobot dan gradien bobot tiap neuron dalam bentuk graf
        pada epoch terakhir (setelah training selesai). Menggunakan pendekatan sampling untuk
        mengurangi waktu loading.
        
        Parameters:
        max_neurons_per_layer : int, jumlah maksimum neuron yang ditampilkan per layer
        max_connections_per_neuron : int, jumlah maksimum koneksi per neuron
        """
        # Mendapatkan informasi arsitektur dari model
        layer_sizes = []
        for i, layer in enumerate(self.model.layers):
            if i == 0:
                layer_sizes.append(layer.input_size)
                layer_sizes.append(layer.output_size)
            else:
                layer_sizes.append(layer.output_size)
        
        # Buat visualisasi sederhana tanpa NetworkX
        plt.figure(figsize=(10, 6))
        
        # Tentukan posisi x untuk setiap layer
        layer_x = np.linspace(0, 1, len(layer_sizes))
        
        # Untuk menyimpan neuron yang sudah diplot
        plotted_neurons = {}
        
        # Plot neurons untuk setiap layer
        for layer_idx, size in enumerate(layer_sizes):
            # Pilih subset neuron jika ukuran layer terlalu besar
            n_neurons = min(size, max_neurons_per_layer)
            indices = np.linspace(0, size-1, n_neurons).astype(int) if size > n_neurons else np.arange(size)
            
            # Tentukan posisi y untuk neuron
            y_positions = np.linspace(0.2, 0.8, n_neurons)
            
            # Warna berdasarkan jenis layer
            if layer_idx == 0:
                color = 'lightblue'  # Input layer
            elif layer_idx == len(layer_sizes) - 1:
                color = 'salmon'     # Output layer
            else:
                color = 'lightgreen' # Hidden layer
            
            # Plot nodes
            plt.scatter([layer_x[layer_idx]] * n_neurons, y_positions, s=100, color=color, zorder=3)
            
            # Simpan posisi neuron yang diplot
            for i, idx in enumerate(indices):
                plotted_neurons[(layer_idx, idx)] = (layer_x[layer_idx], y_positions[i])
            
            # Tampilkan total neuron jika ada sampling
            if size > n_neurons:
                plt.text(layer_x[layer_idx], 0.1, f"Total: {size}", ha='center', fontsize=9)
        
        # Plot koneksi antar layer
        for layer_idx in range(len(self.model.layers)):
            layer = self.model.layers[layer_idx]
            weights = layer.weights  # (input_size, output_size)
            
            # Dapatkan neuron yang diplot di layer ini dan layer berikutnya
            from_neurons = [key[1] for key in plotted_neurons.keys() if key[0] == layer_idx]
            to_neurons = [key[1] for key in plotted_neurons.keys() if key[0] == layer_idx + 1]
            
            # Batasi jumlah koneksi untuk performa
            from_neurons = from_neurons[:max_neurons_per_layer]
            to_neurons = to_neurons[:max_neurons_per_layer]
            
            # Plot subset koneksi
            for i in from_neurons:
                # Pastikan indeks valid
                if i < layer.input_size:
                    # Pilih subset koneksi neuron ini
                    if len(to_neurons) > max_connections_per_neuron and len(to_neurons) > 0:
                        # Validasi indeks
                        valid_to_neurons = [j for j in to_neurons if j < layer.output_size]
                        if valid_to_neurons:
                            # Pilih koneksi dengan bobot tertinggi (absolut)
                            weights_row = np.abs([weights[i, j] for j in valid_to_neurons])
                            # Urutkan dan pilih n tertinggi
                            if len(weights_row) > 0:
                                selected_indices = np.argsort(weights_row)[-min(max_connections_per_neuron, len(weights_row)):]
                                selected_to = [valid_to_neurons[j] for j in selected_indices if j < len(valid_to_neurons)]
                            else:
                                selected_to = []
                        else:
                            selected_to = []
                    else:
                        selected_to = [j for j in to_neurons if j < layer.output_size]
                    
                    for j in selected_to:
                        if (layer_idx, i) in plotted_neurons and (layer_idx+1, j) in plotted_neurons:
                            if j < layer.output_size:  # Pastikan indeks valid
                                weight = weights[i, j]
                                
                                # Warna dan ketebalan berdasarkan bobot
                                color = 'red' if weight < 0 else 'blue'
                                linewidth = min(abs(weight) * 2, 2.5)
                                
                                # Plot koneksi
                                x1, y1 = plotted_neurons[(layer_idx, i)]
                                x2, y2 = plotted_neurons[(layer_idx+1, j)]
                                plt.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.6, zorder=1)
        
        # Tambahkan label layer
        for i, label_idx in enumerate(range(len(layer_sizes))):
            if label_idx == 0:
                label = "Input"
            elif label_idx == len(layer_sizes) - 1:
                label = "Output"
            else:
                label = f"Hidden {label_idx}"
            
            plt.text(layer_x[i], 0.95, label, ha='center', fontsize=10)
        
        # Tambahkan legenda untuk bobot
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Bobot Positif'),
            Line2D([0], [0], color='red', lw=2, label='Bobot Negatif'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', label='Input Layer', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', label='Hidden Layer', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', label='Output Layer', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=9)
        
        plt.title("Struktur Jaringan Neural dengan Sampel Bobot")
        plt.xlim(-0.05, 1.05)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_weight_distribution(self, layers_to_plot=None, bins=30, sample_size=1000):
        """
        Menampilkan distribusi bobot dari tiap layer yang dipilih pada epoch terakhir
        
        Parameters:
        layers_to_plot : list of int, layer indices yang akan ditampilkan distribusinya
                        None untuk menampilkan semua layer
        bins : int, jumlah bins untuk histogram
        sample_size : int, jumlah sampel maksimum untuk diplot (untuk mempercepat visualisasi)
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
                
                # Plot bobot
                all_weights = layer.weights.flatten()
                
                # Jika data terlalu besar, ambil sampel acak
                if len(all_weights) > sample_size:
                    np.random.seed(42)  # Untuk reprodusibilitas
                    indices = np.random.choice(len(all_weights), sample_size, replace=False)
                    sampled_weights = all_weights[indices]
                else:
                    sampled_weights = all_weights
                
                # Hitung statistik dari semua data
                mean_weight = np.mean(all_weights)
                std_weight = np.std(all_weights)
                
                # Plot histogram sederhana bukannya sns.histplot untuk kecepatan
                axes[i].hist(sampled_weights, bins=bins, alpha=0.7, color='skyblue', density=True)
                
                # Plot normal distribution fit
                x = np.linspace(min(sampled_weights), max(sampled_weights), 100)
                axes[i].plot(x, 1/(std_weight * np.sqrt(2 * np.pi)) * 
                         np.exp(-(x - mean_weight)**2 / (2 * std_weight**2)), 
                         color='blue', lw=2)
                
                # Tambahkan judul dan label
                axes[i].set_title(f'Distribusi Bobot Layer {layer_idx}')
                axes[i].set_xlabel('Nilai Bobot')
                axes[i].set_ylabel('Frekuensi')
                
                # Tambahkan statistik dasar
                stats_text = f'Mean: {mean_weight:.4f}, Std: {std_weight:.4f}'
                axes[i].text(0.02, 0.95, stats_text, transform=axes[i].transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                
                # Plot bias jika tersedia (tanpa visualisasi tambahan untuk kecepatan)
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

    def visualize_gradient_distribution(self, layers_to_plot=None, bins=30, sample_size=1000):
        """
        Menampilkan distribusi gradien bobot dari tiap layer yang dipilih pada epoch terakhir
        
        Parameters:
        layers_to_plot : list of int, layer indices yang akan ditampilkan distribusinya
                        None untuk menampilkan semua layer
        bins : int, jumlah bins untuk histogram
        sample_size : int, jumlah sampel maksimum untuk diplot (untuk mempercepat visualisasi)
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
                
                # Memeriksa atribut dweights di DenseLayer
                if hasattr(layer, 'dweights') and layer.dweights is not None:
                    # Ambil sampel untuk mempercepat
                    gradients = layer.dweights
                    all_gradients = gradients.flatten()
                    
                    # Jika data terlalu besar, ambil sampel acak
                    if len(all_gradients) > sample_size:
                        np.random.seed(42)  # Untuk reprodusibilitas
                        indices = np.random.choice(len(all_gradients), sample_size, replace=False)
                        sampled_gradients = all_gradients[indices]
                    else:
                        sampled_gradients = all_gradients
                    
                    # Hitung statistik dari semua data
                    mean_grad = np.mean(all_gradients)
                    std_grad = np.std(all_gradients)
                    
                    # Plot histogram sederhana bukannya sns.histplot untuk kecepatan
                    axes[i].hist(sampled_gradients, bins=bins, alpha=0.7, color='skyblue', density=True)
                    
                    # Plot normal distribution fit
                    x = np.linspace(min(sampled_gradients), max(sampled_gradients), 100)
                    axes[i].plot(x, 1/(std_grad * np.sqrt(2 * np.pi)) * 
                             np.exp(-(x - mean_grad)**2 / (2 * std_grad**2)), 
                             color='red', lw=2)
                    
                    # Tambahkan judul dan label
                    axes[i].set_title(f'Distribusi Gradien Bobot Layer {layer_idx}')
                    axes[i].set_xlabel('Nilai Gradien')
                    axes[i].set_ylabel('Frekuensi')
                    
                    # Tambahkan statistik dasar
                    stats_text = f'Mean: {mean_grad:.4f}, Std: {std_grad:.4f}'
                    axes[i].text(0.02, 0.95, stats_text, transform=axes[i].transAxes,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                    
                    # Tambahkan informasi bias gradient tanpa visualisasi tambahan
                    if hasattr(layer, 'dbiases') and layer.dbiases is not None:
                        bias_gradients = layer.dbiases.flatten()
                        if len(bias_gradients) > 0:
                            mean_bias = np.mean(bias_gradients)
                            axes[i].axvline(x=mean_bias, color='r', linestyle='--',
                                         label=f'Mean Bias Grad: {mean_bias:.4f}')
                            axes[i].legend(loc='upper right')
                else:
                    axes[i].text(0.5, 0.5, f'Layer {layer_idx} tidak memiliki gradien bobot (dweights)', 
                              horizontalalignment='center', verticalalignment='center')
            else:
                axes[i].text(0.5, 0.5, f'Layer {layer_idx} tidak ditemukan', 
                          horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.show()