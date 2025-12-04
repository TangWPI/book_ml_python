"""
Utility functions for machine learning algorithms.
"""

import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def get_device():
    """
    Get the device (CPU or CUDA) for PyTorch computations.
    
    Returns:
        torch.device: Device object
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch (int): Current epoch
        loss (float): Current loss
        filepath (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath (str): Path to checkpoint file
        
    Returns:
        tuple: (epoch, loss)
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss
