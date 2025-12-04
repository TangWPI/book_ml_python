"""
Neural Network classifier implementation using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class NeuralNetworkClassifier(nn.Module):
    """
    Multi-layer Perceptron (MLP) for classification.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout_rate=0.2):
        """
        Initialize Neural Network Classifier.
        
        Args:
            input_dim (int): Dimension of input features
            output_dim (int): Number of classes
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout rate for regularization
        """
        super(NeuralNetworkClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Logits (raw predictions before softmax)
        """
        return self.network(x)
    
    def predict_proba(self, x):
        """
        Get probability predictions.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Class probabilities
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x):
        """
        Get class predictions.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Predicted class labels
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)


class NNClassifierTrainer:
    """
    Trainer class for Neural Network Classifier.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            model (NeuralNetworkClassifier): Neural network model
            device (str): Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train(self, train_loader, n_epochs=50, learning_rate=1e-3,
              val_loader=None, verbose=True):
        """
        Train the neural network classifier.
        
        Args:
            train_loader (DataLoader): Training data loader
            n_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            val_loader (DataLoader): Validation data loader (optional)
            verbose (bool): Whether to print progress
            
        Returns:
            dict: Training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}') if verbose else train_loader
            
            for data, targets in iterator:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Flatten if needed
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * correct / total
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                if verbose:
                    print(f'Epoch {epoch+1}/{n_epochs} - '
                          f'Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}% - '
                          f'Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%')
            else:
                if verbose:
                    print(f'Epoch {epoch+1}/{n_epochs} - '
                          f'Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%')
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses if val_loader else None,
            'val_accuracies': self.val_accuracies if val_loader else None
        }
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on given data.
        
        Args:
            data_loader (DataLoader): Data loader
            
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def predict(self, data_loader):
        """
        Get predictions for given data.
        
        Args:
            data_loader (DataLoader): Data loader
            
        Returns:
            tuple: (predictions, true_labels)
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.append(predicted.cpu())
                all_labels.append(targets)
        
        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels)
        
        return predictions, labels
