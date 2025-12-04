"""
Variational Autoencoder (VAE) implementation using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class VAE(nn.Module):
    """
    Variational Autoencoder for generative modeling.
    
    VAE learns to encode data into a probabilistic latent space and 
    generate new samples by sampling from this latent distribution.
    """
    
    def __init__(self, input_dim, latent_dim, hidden_dims=None):
        """
        Initialize VAE.
        
        Args:
            input_dim (int): Dimension of input data
            latent_dim (int): Dimension of latent space
            hidden_dims (list): List of hidden layer dimensions
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        decoder_layers.extend([
            nn.Linear(latent_dim, prev_dim),
            nn.ReLU(),
            nn.BatchNorm1d(prev_dim)
        ])
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i-1]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[i-1])
            ])
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming normalized input [0, 1]
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            tuple: (mu, logvar) - mean and log variance of latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from latent distribution.
        
        Args:
            mu (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log variance of latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        Decode latent vector to reconstruction.
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Reconstructed data
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through VAE.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            tuple: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def sample(self, n_samples, device='cpu'):
        """
        Generate new samples from the latent space.
        
        Args:
            n_samples (int): Number of samples to generate
            device (str): Device to generate samples on
            
        Returns:
            torch.Tensor: Generated samples
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device)
            samples = self.decode(z)
        return samples


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function (ELBO).
    
    Args:
        recon_x (torch.Tensor): Reconstructed data
        x (torch.Tensor): Original data
        mu (torch.Tensor): Mean of latent distribution
        logvar (torch.Tensor): Log variance of latent distribution
        beta (float): Weight for KL divergence term
        
    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


class VAETrainer:
    """
    Trainer class for VAE.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            model (VAE): VAE model
            device (str): Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, n_epochs=50, learning_rate=1e-3,
              val_loader=None, beta=1.0, verbose=True):
        """
        Train the VAE.
        
        Args:
            train_loader (DataLoader): Training data loader
            n_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            val_loader (DataLoader): Validation data loader (optional)
            beta (float): Weight for KL divergence term (beta-VAE)
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Training history
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            
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
                
                # Normalize to [0, 1] if not already
                data_min = data.min()
                data_max = data.max()
                if data_min < 0 or data_max > 1:
                    # Add epsilon to prevent division by zero
                    data = (data - data_min) / (data_max - data_min + 1e-8)
                
                optimizer.zero_grad()
                recon, mu, logvar = self.model(data)
                loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
            
            train_loss /= len(train_loader.dataset)
            train_recon_loss /= len(train_loader.dataset)
            train_kl_loss /= len(train_loader.dataset)
            
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, beta)
                self.val_losses.append(val_loss)
                
                if verbose:
                    print(f'Epoch {epoch+1}/{n_epochs} - '
                          f'Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f}) - '
                          f'Val Loss: {val_loss:.4f}')
            else:
                if verbose:
                    print(f'Epoch {epoch+1}/{n_epochs} - '
                          f'Train Loss: {train_loss:.4f} (Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f})')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses if val_loader else None
        }
    
    def evaluate(self, data_loader, beta=1.0):
        """
        Evaluate the VAE on given data.
        
        Args:
            data_loader (DataLoader): Data loader
            beta (float): Weight for KL divergence term
            
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
                
                data_min = data.min()
                data_max = data.max()
                if data_min < 0 or data_max > 1:
                    data = (data - data_min) / (data_max - data_min + 1e-8)
                
                recon, mu, logvar = self.model(data)
                loss, _, _ = vae_loss(recon, data, mu, logvar, beta)
                total_loss += loss.item()
        
        return total_loss / len(data_loader.dataset)
