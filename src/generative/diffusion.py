"""
Denoising Diffusion Probabilistic Model (DDPM) implementation using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math


class DiffusionModel(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM).
    
    DDPM learns to generate data by gradually denoising samples from Gaussian noise.
    """
    
    def __init__(self, input_dim, hidden_dims=None, n_timesteps=1000):
        """
        Initialize Diffusion Model.
        
        Args:
            input_dim (int): Dimension of input data
            hidden_dims (list): List of hidden layer dimensions
            n_timesteps (int): Number of diffusion timesteps
        """
        super(DiffusionModel, self).__init__()
        
        self.input_dim = input_dim
        self.n_timesteps = n_timesteps
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        
        # Noise prediction network
        layers = []
        
        # Time embedding
        self.time_embed_dim = 64
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # Main network
        prev_dim = input_dim + self.time_embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.SiLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Define beta schedule (variance schedule)
        self.register_buffer('betas', self._cosine_beta_schedule(n_timesteps))
        
        # Pre-compute alphas and other useful quantities
        alphas = 1.0 - self.betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', 
                            torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]]))
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine beta schedule for diffusion.
        
        Args:
            timesteps (int): Number of timesteps
            s (float): Small offset
            
        Returns:
            torch.Tensor: Beta schedule
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _get_time_embedding(self, t, batch_size):
        """
        Get time embedding for timestep t.
        
        Args:
            t (torch.Tensor): Timesteps
            batch_size (int): Batch size
            
        Returns:
            torch.Tensor: Time embeddings
        """
        # Normalize timesteps to [0, 1]
        t_normalized = t.float() / self.n_timesteps
        t_normalized = t_normalized.view(-1, 1)
        return self.time_mlp(t_normalized)
    
    def forward(self, x, t):
        """
        Predict noise at timestep t.
        
        Args:
            x (torch.Tensor): Noisy data at timestep t
            t (torch.Tensor): Timesteps
            
        Returns:
            torch.Tensor: Predicted noise
        """
        batch_size = x.shape[0]
        
        # Get time embedding
        t_emb = self._get_time_embedding(t, batch_size)
        
        # Concatenate with input
        x_with_time = torch.cat([x, t_emb], dim=1)
        
        # Predict noise
        return self.network(x_with_time)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: add noise to data.
        
        Args:
            x_0 (torch.Tensor): Original data
            t (torch.Tensor): Timesteps
            noise (torch.Tensor): Noise to add (optional)
            
        Returns:
            torch.Tensor: Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self, x, t):
        """
        Single step of reverse diffusion process.
        
        Args:
            x (torch.Tensor): Noisy data at timestep t
            t (torch.Tensor): Timestep
            
        Returns:
            torch.Tensor: Denoised data at timestep t-1
        """
        batch_size = x.shape[0]
        
        # Predict noise
        predicted_noise = self.forward(x, t)
        
        # Get coefficients
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1)
        betas_t = self.betas[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(self, n_samples, device='cpu'):
        """
        Generate samples using reverse diffusion process.
        
        Args:
            n_samples (int): Number of samples to generate
            device (str): Device to generate samples on
            
        Returns:
            torch.Tensor: Generated samples
        """
        # Start from random noise
        x = torch.randn(n_samples, self.input_dim).to(device)
        
        # Iteratively denoise
        for t in reversed(range(self.n_timesteps)):
            t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
            x = self.p_sample(x, t_batch)
        
        return x


class DiffusionTrainer:
    """
    Trainer class for Diffusion Model.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            model (DiffusionModel): Diffusion model
            device (str): Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, n_epochs=50, learning_rate=1e-3,
              val_loader=None, verbose=True):
        """
        Train the diffusion model.
        
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
        criterion = nn.MSELoss()
        
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
                
                batch_size = data.shape[0]
                
                # Sample random timesteps
                t = torch.randint(0, self.model.n_timesteps, (batch_size,), device=self.device)
                
                # Sample noise
                noise = torch.randn_like(data)
                
                # Add noise to data
                x_noisy = self.model.q_sample(data, t, noise)
                
                # Predict noise
                optimizer.zero_grad()
                predicted_noise = self.model(x_noisy, t)
                
                # Calculate loss
                loss = criterion(predicted_noise, noise)
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
        criterion = nn.MSELoss()
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
                
                batch_size = data.shape[0]
                
                # Sample random timesteps
                t = torch.randint(0, self.model.n_timesteps, (batch_size,), device=self.device)
                
                # Sample noise
                noise = torch.randn_like(data)
                
                # Add noise to data
                x_noisy = self.model.q_sample(data, t, noise)
                
                # Predict noise
                predicted_noise = self.model(x_noisy, t)
                
                # Calculate loss
                loss = criterion(predicted_noise, noise)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
