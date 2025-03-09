import matplotlib.pyplot as plt
import numpy as np

def plot_weights(weights, title="Visualisasi Bobot"):
    num_weights = weights.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_weights)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        if i < num_weights:
            ax.imshow(weights[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    
    plt.suptitle(title)
    plt.show()

def plot_loss_curve(loss_history):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Kurva Loss selama Training')
    plt.legend()
    plt.grid()
    plt.show()
    