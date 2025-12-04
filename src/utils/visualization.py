"""
Visualization utilities for machine learning results.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib.colors import ListedColormap


def plot_2d_data(X, y=None, title='Data Visualization', figsize=(8, 6), 
                 cmap='viridis', save_path=None):
    """
    Plot 2D data points with optional labels.
    
    Args:
        X (np.ndarray or torch.Tensor): 2D data points (n_samples, 2)
        y (np.ndarray or torch.Tensor): Labels (optional)
        title (str): Plot title
        figsize (tuple): Figure size
        cmap (str): Colormap name
        save_path (str): Path to save figure (optional)
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    if y is not None:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=0.6, edgecolors='k')
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(X[:, 0], X[:, 1], alpha=0.6, edgecolors='k')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_dimensionality_reduction(X_original, X_reduced, y=None, method='PCA',
                                  figsize=(15, 5), save_path=None):
    """
    Plot original data and reduced dimensionality data side by side.
    
    Args:
        X_original (np.ndarray or torch.Tensor): Original high-dimensional data
        X_reduced (np.ndarray or torch.Tensor): Reduced dimensionality data (n_samples, 2)
        y (np.ndarray or torch.Tensor): Labels (optional)
        method (str): Dimensionality reduction method name
        figsize (tuple): Figure size
        save_path (str): Path to save figure (optional)
    """
    if isinstance(X_original, torch.Tensor):
        X_original = X_original.cpu().numpy()
    if isinstance(X_reduced, torch.Tensor):
        X_reduced = X_reduced.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot first two dimensions of original data
    if y is not None:
        axes[0].scatter(X_original[:, 0], X_original[:, 1], c=y, 
                       cmap='viridis', alpha=0.6, edgecolors='k')
    else:
        axes[0].scatter(X_original[:, 0], X_original[:, 1], 
                       alpha=0.6, edgecolors='k')
    axes[0].set_xlabel('Original Feature 1')
    axes[0].set_ylabel('Original Feature 2')
    axes[0].set_title('Original Data (First 2 Dimensions)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot reduced data
    if y is not None:
        scatter = axes[1].scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
                                 cmap='viridis', alpha=0.6, edgecolors='k')
        plt.colorbar(scatter, ax=axes[1], label='Label')
    else:
        axes[1].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                       alpha=0.6, edgecolors='k')
    axes[1].set_xlabel(f'{method} Component 1')
    axes[1].set_ylabel(f'{method} Component 2')
    axes[1].set_title(f'{method} Reduced Data')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(train_losses, val_losses=None, title='Training Curves',
                         figsize=(10, 5), save_path=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch (optional)
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str): Path to save figure (optional)
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_image_grid(images, n_rows=4, n_cols=8, figsize=(12, 6), 
                   title='Image Grid', save_path=None):
    """
    Plot a grid of images.
    
    Args:
        images (torch.Tensor): Batch of images (B, C, H, W)
        n_rows (int): Number of rows
        n_cols (int): Number of columns
        figsize (tuple): Figure size
        title (str): Plot title
        save_path (str): Path to save figure (optional)
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu()
    
    # Select subset of images
    n_images = min(n_rows * n_cols, images.shape[0])
    images = images[:n_images]
    
    # Create grid
    grid = torchvision.utils.make_grid(images, nrow=n_cols, normalize=True, padding=2)
    grid = grid.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.axis('off')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_reconstruction(original, reconstructed, n_images=8, figsize=(15, 4),
                       save_path=None):
    """
    Plot original and reconstructed images side by side.
    
    Args:
        original (torch.Tensor): Original images (B, C, H, W)
        reconstructed (torch.Tensor): Reconstructed images (B, C, H, W)
        n_images (int): Number of images to display
        figsize (tuple): Figure size
        save_path (str): Path to save figure (optional)
    """
    if isinstance(original, torch.Tensor):
        original = original.cpu()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu()
    
    n_images = min(n_images, original.shape[0])
    
    fig, axes = plt.subplots(2, n_images, figsize=figsize)
    
    for i in range(n_images):
        # Original images
        img_orig = original[i].permute(1, 2, 0).numpy()
        if img_orig.shape[2] == 1:
            img_orig = img_orig.squeeze()
            axes[0, i].imshow(img_orig, cmap='gray')
        else:
            img_orig = (img_orig - img_orig.min()) / (img_orig.max() - img_orig.min())
            axes[0, i].imshow(img_orig)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstructed images
        img_recon = reconstructed[i].permute(1, 2, 0).numpy()
        if img_recon.shape[2] == 1:
            img_recon = img_recon.squeeze()
            axes[1, i].imshow(img_recon, cmap='gray')
        else:
            img_recon = (img_recon - img_recon.min()) / (img_recon.max() - img_recon.min())
            axes[1, i].imshow(img_recon)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(8, 6),
                         save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.ndarray or torch.Tensor): True labels
        y_pred (np.ndarray or torch.Tensor): Predicted labels
        class_names (list): Names of classes (optional)
        figsize (tuple): Figure size
        save_path (str): Path to save figure (optional)
    """
    from sklearn.metrics import confusion_matrix
    
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
