"""
Transformer implementation for text translation using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer for transformers.
    
    Adds positional information to token embeddings using sinusoidal functions.
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Dimension of model embeddings
            max_len (int): Maximum sequence length
            dropout (float): Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x (torch.Tensor): Input embeddings (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Input with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model (int): Dimension of model
            n_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            Q (torch.Tensor): Query tensor
            K (torch.Tensor): Key tensor
            V (torch.Tensor): Value tensor
            mask (torch.Tensor): Attention mask (optional)
            
        Returns:
            tuple: (attention output, attention weights)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass through multi-head attention.
        
        Args:
            query (torch.Tensor): Query tensor
            key (torch.Tensor): Key tensor
            value (torch.Tensor): Value tensor
            mask (torch.Tensor): Attention mask (optional)
            
        Returns:
            torch.Tensor: Attention output
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape to (batch_size, n_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize feed-forward network.
        
        Args:
            d_model (int): Dimension of model
            d_ff (int): Dimension of feed-forward layer
            dropout (float): Dropout probability
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass through feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single encoder layer for transformer.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Initialize encoder layer.
        
        Args:
            d_model (int): Dimension of model
            n_heads (int): Number of attention heads
            d_ff (int): Dimension of feed-forward layer
            dropout (float): Dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass through encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor): Attention mask (optional)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Single decoder layer for transformer.
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Initialize decoder layer.
        
        Args:
            d_model (int): Dimension of model
            n_heads (int): Number of attention heads
            d_ff (int): Dimension of feed-forward layer
            dropout (float): Dropout probability
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass through decoder layer.
        
        Args:
            x (torch.Tensor): Input tensor
            encoder_output (torch.Tensor): Encoder output
            src_mask (torch.Tensor): Source attention mask (optional)
            tgt_mask (torch.Tensor): Target attention mask (optional)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Self-attention with residual connection and layer norm
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        
        # Cross-attention with residual connection and layer norm
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks like translation.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_encoder_layers=6, n_decoder_layers=6, d_ff=2048, 
                 max_len=5000, dropout=0.1, pad_idx=0):
        """
        Initialize Transformer model.
        
        Args:
            src_vocab_size (int): Source vocabulary size
            tgt_vocab_size (int): Target vocabulary size
            d_model (int): Dimension of model embeddings
            n_heads (int): Number of attention heads
            n_encoder_layers (int): Number of encoder layers
            n_decoder_layers (int): Number of decoder layers
            d_ff (int): Dimension of feed-forward layer
            max_len (int): Maximum sequence length
            dropout (float): Dropout probability
            pad_idx (int): Padding token index
        """
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Cache for look-ahead masks
        self._look_ahead_mask_cache = {}
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq):
        """
        Create padding mask for attention.
        
        Args:
            seq (torch.Tensor): Input sequence
            
        Returns:
            torch.Tensor: Padding mask
        """
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size, device='cpu'):
        """
        Create look-ahead mask for decoder self-attention.
        Uses caching to avoid recreating masks for common sequence lengths.
        
        Args:
            size (int): Sequence length
            device (str): Device to create mask on
            
        Returns:
            torch.Tensor: Look-ahead mask
        """
        # Check cache
        cache_key = (size, device)
        if cache_key in self._look_ahead_mask_cache:
            return self._look_ahead_mask_cache[cache_key]
        
        # Create new mask
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask == 0
        
        # Cache it (limit cache size to prevent memory issues)
        if len(self._look_ahead_mask_cache) < 100:
            self._look_ahead_mask_cache[cache_key] = mask
        
        return mask
    
    def encode(self, src, src_mask=None):
        """
        Encode source sequence.
        
        Args:
            src (torch.Tensor): Source sequence
            src_mask (torch.Tensor): Source mask (optional)
            
        Returns:
            torch.Tensor: Encoded sequence
        """
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decode target sequence.
        
        Args:
            tgt (torch.Tensor): Target sequence
            encoder_output (torch.Tensor): Encoder output
            src_mask (torch.Tensor): Source mask (optional)
            tgt_mask (torch.Tensor): Target mask (optional)
            
        Returns:
            torch.Tensor: Decoded sequence
        """
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt):
        """
        Forward pass through transformer.
        
        Args:
            src (torch.Tensor): Source sequence (batch_size, src_len)
            tgt (torch.Tensor): Target sequence (batch_size, tgt_len)
            
        Returns:
            torch.Tensor: Output logits (batch_size, tgt_len, tgt_vocab_size)
        """
        # Create masks
        src_mask = self.create_padding_mask(src)
        tgt_mask = self.create_padding_mask(tgt)
        
        # Create look-ahead mask for target (using caching)
        tgt_len = tgt.size(1)
        device = str(tgt.device)
        look_ahead_mask = self.create_look_ahead_mask(tgt_len, device)
        tgt_mask = tgt_mask & look_ahead_mask.unsqueeze(0)
        
        # Encode and decode
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output
    
    def translate(self, src, max_len=50, sos_idx=1, eos_idx=2, device='cpu'):
        """
        Translate a source sequence to target sequence using greedy decoding.
        
        Args:
            src (torch.Tensor): Source sequence (1, src_len)
            max_len (int): Maximum generation length
            sos_idx (int): Start-of-sequence token index
            eos_idx (int): End-of-sequence token index
            device (str): Device to use
            
        Returns:
            torch.Tensor: Generated target sequence
        """
        self.eval()
        with torch.no_grad():
            src = src.to(device)
            src_mask = self.create_padding_mask(src)
            encoder_output = self.encode(src, src_mask)
            
            # Start with SOS token
            tgt = torch.LongTensor([[sos_idx]]).to(device)
            
            for _ in range(max_len):
                tgt_mask = self.create_padding_mask(tgt)
                look_ahead_mask = self.create_look_ahead_mask(tgt.size(1), device)
                tgt_mask = tgt_mask & look_ahead_mask.unsqueeze(0)
                
                decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
                output = self.output_projection(decoder_output)
                
                # Get next token (greedy)
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == eos_idx:
                    break
            
            return tgt


class TransformerTrainer:
    """
    Trainer class for Transformer models.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            model (Transformer): Transformer model
            device (str): Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader, n_epochs=10, learning_rate=1e-4,
              val_loader=None, verbose=True, label_smoothing=0.1,
              betas=(0.9, 0.999), eps=1e-8):
        """
        Train the transformer.
        
        Args:
            train_loader (DataLoader): Training data loader
            n_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            val_loader (DataLoader): Validation data loader (optional)
            verbose (bool): Whether to print progress
            label_smoothing (float): Label smoothing factor
            betas (tuple): Adam optimizer betas. Default (0.9, 0.999) is PyTorch default.
                          Original Transformer paper used (0.9, 0.98).
            eps (float): Adam optimizer epsilon. Default 1e-8 is PyTorch default.
                        Original Transformer paper used 1e-9.
            
        Returns:
            dict: Training history
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=betas, eps=eps)
        criterion = nn.CrossEntropyLoss(ignore_index=self.model.pad_idx, label_smoothing=label_smoothing)
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}') if verbose else train_loader
            
            for batch in iterator:
                src, tgt = batch
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                # Target input and output (shifted by 1)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(src, tgt_input)
                
                # Calculate loss
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                
                if verbose:
                    print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
            else:
                if verbose:
                    print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses if val_loader else None
        }
    
    def evaluate(self, data_loader):
        """
        Evaluate the transformer on given data.
        
        Args:
            data_loader (DataLoader): Data loader
            
        Returns:
            float: Average loss
        """
        self.model.eval()
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss(ignore_index=self.model.pad_idx)
        
        with torch.no_grad():
            for batch in data_loader:
                src, tgt = batch
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = self.model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
