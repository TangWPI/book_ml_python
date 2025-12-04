"""
K-Means clustering implementation from scratch using PyTorch.
"""

import torch


class KMeans:
    """
    K-Means clustering algorithm.
    
    Groups data into K clusters by iteratively assigning points to nearest centroids
    and updating centroids based on cluster assignments.
    """
    
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        """
        Initialize K-Means.
        
        Args:
            n_clusters (int): Number of clusters
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
            random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia = None
    
    def _initialize_centroids(self, X):
        """
        Initialize centroids using K-Means++ algorithm.
        
        Args:
            X (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Initial centroids
        """
        n_samples = X.shape[0]
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        
        # Choose first centroid randomly
        centroids = [X[torch.randint(0, n_samples, (1,))]]
        
        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Calculate distances to nearest centroid
            distances = torch.stack([
                torch.sum((X - c) ** 2, dim=1) for c in centroids
            ])
            min_distances = torch.min(distances, dim=0)[0]
            
            # Choose next centroid with probability proportional to distance squared
            probabilities = min_distances / torch.sum(min_distances)
            next_centroid_idx = torch.multinomial(probabilities, 1)
            centroids.append(X[next_centroid_idx])
        
        return torch.cat(centroids, dim=0)
    
    def _assign_clusters(self, X, centroids):
        """
        Assign each point to nearest centroid.
        
        Args:
            X (torch.Tensor): Input data
            centroids (torch.Tensor): Current centroids
            
        Returns:
            torch.Tensor: Cluster assignments
        """
        # Calculate distances to all centroids
        distances = torch.cdist(X, centroids)
        # Assign to nearest centroid
        labels = torch.argmin(distances, dim=1)
        return labels
    
    def _update_centroids(self, X, labels):
        """
        Update centroids as mean of assigned points.
        
        Args:
            X (torch.Tensor): Input data
            labels (torch.Tensor): Current cluster assignments
            
        Returns:
            torch.Tensor: New centroids
        """
        centroids = torch.zeros(self.n_clusters, X.shape[1], device=X.device, dtype=X.dtype)
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = torch.mean(cluster_points, dim=0)
            else:
                # If cluster is empty, reinitialize randomly
                random_idx = torch.randint(0, X.shape[0], size=()).item()
                centroids[k] = X[random_idx]
        
        return centroids
    
    def _calculate_inertia(self, X, labels, centroids):
        """
        Calculate sum of squared distances to centroids.
        
        Args:
            X (torch.Tensor): Input data
            labels (torch.Tensor): Cluster assignments
            centroids (torch.Tensor): Centroids
            
        Returns:
            float: Inertia value
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += torch.sum((cluster_points - centroids[k]) ** 2).item()
        return inertia
    
    def fit(self, X):
        """
        Fit K-Means to data.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X, self.centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = torch.sum(torch.abs(new_centroids - self.centroids))
            
            self.centroids = new_centroids
            self.labels = labels
            
            if centroid_shift < self.tol:
                break
        
        # Calculate final inertia
        self.inertia = self._calculate_inertia(X, self.labels, self.centroids)
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Predicted cluster labels
        """
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X):
        """
        Fit K-Means and return cluster labels.
        
        Args:
            X (torch.Tensor): Input data of shape (n_samples, n_features)
            
        Returns:
            torch.Tensor: Cluster labels
        """
        self.fit(X)
        return self.labels
    
    def get_centroids(self):
        """
        Get cluster centroids.
        
        Returns:
            torch.Tensor: Centroids
        """
        return self.centroids
    
    def get_inertia(self):
        """
        Get inertia (sum of squared distances to centroids).
        
        Returns:
            float: Inertia value
        """
        return self.inertia
