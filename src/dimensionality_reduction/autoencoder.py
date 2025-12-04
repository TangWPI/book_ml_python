"""
Autoencoder implementation for dimensionality reduction using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Autoencoder(nn.Module):
    """
    Basic Autoencoder for dimensionality reduction.
    
    The autoencoder learns to compress data into a lower-dimensional latent space
    and then reconstruct the original data from this compressed representation.
    """
    
    def __init__(self, input_dim, latent_dim, hidden_dims=None):
        """
        Initialize Autoencoder.
        
        Args:
            input_dim (int): Dimension of input data
            latent_dim (int): Dimension of latent space (bottleneck)
            hidden_dims (list): List of hidden layer dimensions (encoder)
        """
        super(Autoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
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
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror of encoder)
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, prev_dim))
        decoder_layers.extend([nn.ReLU(), nn.BatchNorm1d(prev_dim)])
        
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i-1]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[i-1])
            ])
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """
        Encode input to latent space.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent representation to reconstruction.
        
        Args:
            z (torch.Tensor): Latent representation
            
        Returns:
            torch.Tensor: Reconstructed data
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through autoencoder.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            tuple: (reconstruction, latent_representation)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoencoderTrainer:
    """
    Trainer class for Autoencoder.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            model (Autoencoder): Autoencoder model
            device (str): Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, n_epochs=50, learning_rate=1e-3, 
              val_loader=None, verbose=True):
        """
        Train the autoencoder.
        
        Args:
            train_loader (DataLoader): Training data loader
            n_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            val_loader (DataLoader): Validation data loader (optional)
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Training history
        """
        criterion = nn.MSELoss()
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
                recon, _ = self.model(data)
                loss = criterion(recon, data)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        if isinstance(batch, (tuple, list)):
                            data = batch[0]
                        else:
                            data = batch
                        
                        data = data.to(self.device)
                        
                        if len(data.shape) > 2:
                            data = data.view(data.size(0), -1)
                        
                        recon, _ = self.model(data)
                        loss = criterion(recon, data)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                self.val_losses.append(val_loss)
                
                if verbose:
                    print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                if verbose:
                    print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.6f}')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses if val_loader else None
        }
    
    def transform(self, data_loader):
        """
        Transform data to latent space.
        
        Args:
            data_loader (DataLoader): Data loader
            
        Returns:
            torch.Tensor: Latent representations
        """
        self.model.eval()
        latent_representations = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    data = batch[0]
                else:
                    data = batch
                
                data = data.to(self.device)
                
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                _, z = self.model(data)
                latent_representations.append(z.cpu())
        
        return torch.cat(latent_representations, dim=0)
