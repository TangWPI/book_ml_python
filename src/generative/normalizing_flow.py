"""
Normalizing Flow implementation using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math


class RealNVP(nn.Module):
    """
    Real-valued Non-Volume Preserving (RealNVP) normalizing flow.
    
    RealNVP learns an invertible transformation between data and a simple 
    base distribution (e.g., Gaussian), enabling both density estimation and sampling.
    """
    
    def __init__(self, input_dim, n_flows=4, hidden_dim=64):
        """
        Initialize RealNVP.
        
        Args:
            input_dim (int): Dimension of input data
            n_flows (int): Number of flow layers
            hidden_dim (int): Hidden dimension for coupling layers
        """
        super(RealNVP, self).__init__()
        
        self.input_dim = input_dim
        self.n_flows = n_flows
        
        # Create flow layers
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(CouplingLayer(input_dim, hidden_dim, mask_type='even' if i % 2 == 0 else 'odd'))
    
    def forward(self, x):
        """
        Forward pass (data to latent).
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            tuple: (latent, log_det_jacobian)
        """
        log_det_jacobian = 0
        z = x
        
        for flow in self.flows:
            z, ldj = flow.forward(z)
            log_det_jacobian += ldj
        
        return z, log_det_jacobian
    
    def inverse(self, z):
        """
        Inverse pass (latent to data).
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Generated data
        """
        x = z
        
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        
        return x
    
    def sample(self, n_samples, device='cpu'):
        """
        Generate samples from the model.
        
        Args:
            n_samples (int): Number of samples to generate
            device (str): Device to generate samples on
            
        Returns:
            torch.Tensor: Generated samples
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.input_dim).to(device)
            x = self.inverse(z)
        return x
    
    def log_prob(self, x):
        """
        Calculate log probability of data.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Log probabilities
        """
        z, log_det_jacobian = self.forward(x)
        
        # Log probability of base distribution (standard Gaussian)
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi))
        log_pz = torch.sum(log_pz, dim=1)
        
        # Apply change of variables formula
        log_px = log_pz + log_det_jacobian
        
        return log_px


class CouplingLayer(nn.Module):
    """
    Affine coupling layer for RealNVP.
    """
    
    def __init__(self, input_dim, hidden_dim=64, mask_type='even'):
        """
        Initialize coupling layer.
        
        Args:
            input_dim (int): Dimension of input
            hidden_dim (int): Hidden dimension
            mask_type (str): Type of mask ('even' or 'odd')
        """
        super(CouplingLayer, self).__init__()
        
        self.input_dim = input_dim
        self.mask_type = mask_type
        
        # Create mask
        self.register_buffer('mask', self._create_mask(input_dim, mask_type))
        
        # Scale and translation networks
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def _create_mask(self, dim, mask_type):
        """Create binary mask."""
        mask = torch.zeros(dim)
        if mask_type == 'even':
            mask[::2] = 1
        else:
            mask[1::2] = 1
        return mask
    
    def forward(self, x):
        """
        Forward transformation.
        
        Args:
            x (torch.Tensor): Input
            
        Returns:
            tuple: (output, log_det_jacobian)
        """
        x_masked = x * self.mask
        
        s = self.scale_net(x_masked)
        t = self.translate_net(x_masked)
        
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        
        y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
        
        log_det_jacobian = torch.sum(s, dim=1)
        
        return y, log_det_jacobian
    
    def inverse(self, y):
        """
        Inverse transformation.
        
        Args:
            y (torch.Tensor): Input
            
        Returns:
            torch.Tensor: Output
        """
        y_masked = y * self.mask
        
        s = self.scale_net(y_masked)
        t = self.translate_net(y_masked)
        
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        
        x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
        
        return x


class NormalizingFlowTrainer:
    """
    Trainer class for Normalizing Flow.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            model (RealNVP): Normalizing flow model
            device (str): Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, n_epochs=50, learning_rate=1e-3,
              val_loader=None, verbose=True):
        """
        Train the normalizing flow.
        
        Args:
            train_loader (DataLoader): Training data loader
            n_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            val_loader (DataLoader): Validation data loader (optional)
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Training history
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}') if verbose else train_loader
            
            for batch in iterator:
                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                else:
                    data = batch
                
                data = data.to(self.device)
                
                # Flatten if needed
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                
                # Calculate negative log likelihood
                log_prob = self.model.log_prob(data)
                loss = -torch.mean(log_prob)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                
                if verbose:
                    print(f'Epoch {epoch+1}/{n_epochs} - '
                          f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                if verbose:
                    print(f'Epoch {epoch+1}/{n_epochs} - '
                          f'Train Loss: {train_loss:.6f}')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses if val_loader else None
        }
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on given data.
        
        Args:
            data_loader (DataLoader): Data loader
            
        Returns:
            float: Average loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                else:
                    data = batch
                
                data = data.to(self.device)
                
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                log_prob = self.model.log_prob(data)
                loss = -torch.mean(log_prob)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
