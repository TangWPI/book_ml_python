"""
Principal Component Analysis (PCA) implementation from scratch using PyTorch.
"""

import torch
import torch.nn as nn


class PCA:
    """
    Principal Component Analysis for dimensionality reduction.
    
    PCA finds orthogonal directions of maximum variance in the data.
    """
    
    def __init__(self, n_components=2):
        """
        Initialize PCA.
        
        Args:
            n_components (int): Number of principal components to keep
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None
    
    def fit(self, X):
        """
        Fit PCA model to data.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
        """
        # Center the data
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        
        # Calculate explained variance ratio
        total_variance = torch.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance
        
        return self
    
    def transform(self, X):
        """
        Transform data to principal component space.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Transformed data of shape (n_samples, n_components)
        """
        X_centered = X - self.mean
        return X_centered @ self.components
    
    def fit_transform(self, X):
        """
        Fit PCA model and transform data.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Transformed data of shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Transform data back to original space.
        
        Args:
            X_transformed (torch.Tensor): Transformed data of shape (n_samples, n_components)
            
        Returns:
            torch.Tensor: Reconstructed data of shape (n_samples, n_features)
        """
        return X_transformed @ self.components.T + self.mean
    
    def get_explained_variance_ratio(self):
        """
        Get the explained variance ratio for each component.
        
        Returns:
            torch.Tensor: Explained variance ratio
        """
        return self.explained_variance_ratio
