"""
Independent Component Analysis (ICA) implementation from scratch using PyTorch.
"""

import torch
import torch.nn as nn


class ICA:
    """
    Independent Component Analysis for blind source separation.
    
    ICA finds statistically independent components in the data.
    Uses FastICA algorithm with hyperbolic tangent as the non-linear function.
    """
    
    def __init__(self, n_components=2, max_iter=200, tol=1e-4, random_state=None):
        """
        Initialize ICA.
        
        Args:
            n_components (int): Number of independent components
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
            random_state (int): Random seed for reproducibility
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.mean = None
        self.whitening_matrix = None
        self.unmixing_matrix = None
        self.mixing_matrix = None
    
    def _center(self, X):
        """Center the data."""
        self.mean = torch.mean(X, dim=0)
        return X - self.mean
    
    def _whiten(self, X):
        """Whiten the data using eigenvalue decomposition."""
        # Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = (X.T @ X) / n_samples
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_components
        eigenvalues = eigenvalues[:self.n_components]
        eigenvectors = eigenvectors[:, :self.n_components]
        
        # Compute whitening matrix
        self.whitening_matrix = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues))
        
        return X @ self.whitening_matrix
    
    def _g(self, x):
        """Non-linear function (hyperbolic tangent)."""
        return torch.tanh(x)
    
    def _g_prime(self, x):
        """Derivative of non-linear function."""
        return 1 - torch.tanh(x) ** 2
    
    def fit(self, X):
        """
        Fit ICA model to data.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        # Center and whiten the data
        X_centered = self._center(X)
        X_whitened = self._whiten(X_centered)
        
        n_samples = X_whitened.shape[0]
        
        # Initialize unmixing matrix randomly
        W = torch.randn(self.n_components, self.n_components)
        
        # Orthogonalize initial matrix
        U, _, V = torch.linalg.svd(W)
        W = U @ V.T
        
        # FastICA algorithm
        for iteration in range(self.max_iter):
            W_old = W.clone()
            
            # Update each component
            for i in range(self.n_components):
                w = W[i, :].unsqueeze(1)  # (n_components, 1)
                
                # Compute projection
                wx = X_whitened @ w  # (n_samples, 1)
                
                # Update rule
                g_wx = self._g(wx)
                g_prime_wx = self._g_prime(wx)
                
                w_new = (X_whitened.T @ g_wx) / n_samples - torch.mean(g_prime_wx) * w
                
                # Orthogonalization (Gram-Schmidt)
                for j in range(i):
                    w_new = w_new - (w_new.T @ W[j, :].unsqueeze(1)) * W[j, :].unsqueeze(1)
                
                # Normalize
                w_new = w_new / torch.norm(w_new)
                
                W[i, :] = w_new.squeeze()
            
            # Check convergence
            if torch.max(torch.abs(torch.abs(torch.diag(W @ W_old.T)) - 1)) < self.tol:
                break
        
        self.unmixing_matrix = W
        self.mixing_matrix = torch.linalg.pinv(W @ self.whitening_matrix.T)
        
        return self
    
    def transform(self, X):
        """
        Transform data to independent component space.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Independent components of shape (n_samples, n_components)
        """
        X_centered = X - self.mean
        X_whitened = X_centered @ self.whitening_matrix
        return X_whitened @ self.unmixing_matrix.T
    
    def fit_transform(self, X):
        """
        Fit ICA model and transform data.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Independent components of shape (n_samples, n_components)
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
        return X_transformed @ self.mixing_matrix.T + self.mean
