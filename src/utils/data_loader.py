"""
Utility module for data loading and preprocessing.
Provides functions to load MNIST, CIFAR-10, and generate synthetic Gaussian data.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def load_mnist(batch_size=64, train=True, download=True, data_dir='./data'):
    """
    Load MNIST dataset.
    
    Args:
        batch_size (int): Batch size for data loader
        train (bool): Whether to load training or test data
        download (bool): Whether to download the dataset if not present
        data_dir (str): Directory to store/load data
        
    Returns:
        DataLoader: PyTorch DataLoader for MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2
    )
    
    return loader


def load_cifar10(batch_size=64, train=True, download=True, data_dir='./data'):
    """
    Load CIFAR-10 dataset.
    
    Args:
        batch_size (int): Batch size for data loader
        train (bool): Whether to load training or test data
        download (bool): Whether to download the dataset if not present
        data_dir (str): Directory to store/load data
        
    Returns:
        DataLoader: PyTorch DataLoader for CIFAR-10 dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2
    )
    
    return loader


def generate_gaussian_data(n_samples=1000, n_features=2, n_clusters=3, 
                          cluster_std=1.0, random_state=None):
    """
    Generate synthetic Gaussian data for clustering and classification tasks.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features
        n_clusters (int): Number of clusters/classes
        cluster_std (float): Standard deviation of clusters
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X, y) where X is data tensor and y is labels tensor
    """
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    samples_per_cluster = n_samples // n_clusters
    X_list = []
    y_list = []
    
    for i in range(n_clusters):
        # Generate cluster centers randomly
        center = np.random.randn(n_features) * 5
        # Generate samples around center
        X_cluster = np.random.randn(samples_per_cluster, n_features) * cluster_std + center
        y_cluster = np.full(samples_per_cluster, i)
        
        X_list.append(X_cluster)
        y_list.append(y_cluster)
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return torch.FloatTensor(X), torch.LongTensor(y)


def generate_gaussian_mixture(n_samples=1000, n_features=2, means=None, 
                              covariances=None, weights=None, random_state=None):
    """
    Generate data from a Gaussian mixture model.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features
        means (list): List of mean vectors for each component
        covariances (list): List of covariance matrices for each component
        weights (list): Mixture weights (must sum to 1)
        random_state (int): Random seed for reproducibility
        
    Returns:
        torch.Tensor: Generated samples
    """
    if random_state is not None:
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    # Default parameters
    if means is None:
        n_components = 3
        means = [np.random.randn(n_features) * 3 for _ in range(n_components)]
    else:
        n_components = len(means)
    
    if covariances is None:
        covariances = [np.eye(n_features) for _ in range(n_components)]
    
    if weights is None:
        weights = np.ones(n_components) / n_components
    
    weights = np.array(weights)
    
    # Sample component assignments
    component_samples = np.random.choice(n_components, size=n_samples, p=weights)
    
    X = np.zeros((n_samples, n_features))
    for i in range(n_components):
        mask = component_samples == i
        n_samples_i = np.sum(mask)
        if n_samples_i > 0:
            X[mask] = np.random.multivariate_normal(
                means[i], covariances[i], size=n_samples_i
            )
    
    return torch.FloatTensor(X)


def create_dataloader(X, y=None, batch_size=64, shuffle=True):
    """
    Create a PyTorch DataLoader from tensors.
    
    Args:
        X (torch.Tensor): Input data
        y (torch.Tensor): Labels (optional)
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    if y is not None:
        dataset = TensorDataset(X, y)
    else:
        dataset = TensorDataset(X)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
