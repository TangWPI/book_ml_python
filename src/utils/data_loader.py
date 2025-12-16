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


def load_translation_data(batch_size=32, max_len=50, train_split=0.8, data_dir='./data'):
    """
    Load a small English-French translation dataset.
    
    This function provides a small curated dataset of common English-French 
    phrase pairs for demonstration and educational purposes.
    
    Args:
        batch_size (int): Batch size for data loader
        max_len (int): Maximum sequence length
        train_split (float): Fraction of data to use for training
        data_dir (str): Directory to store vocabulary (not used for built-in data)
        
    Returns:
        tuple: (train_loader, val_loader, src_vocab, tgt_vocab)
            where src_vocab and tgt_vocab are dictionaries with:
            - 'word2idx': word to index mapping
            - 'idx2word': index to word mapping
            - 'vocab_size': vocabulary size
    """
    # Small curated English-French translation pairs
    translation_pairs = [
        ("hello", "bonjour"),
        ("goodbye", "au revoir"),
        ("thank you", "merci"),
        ("please", "s'il vous plaît"),
        ("yes", "oui"),
        ("no", "non"),
        ("good morning", "bonjour"),  # Note: "bonjour" can mean both "hello" and "good morning"
        ("good night", "bonne nuit"),
        ("how are you", "comment allez vous"),
        ("i am fine", "je vais bien"),
        ("what is your name", "comment vous appelez vous"),
        ("my name is", "je m'appelle"),
        ("nice to meet you", "enchanté"),
        ("excuse me", "excusez moi"),
        ("i love you", "je t'aime"),
        ("welcome", "bienvenue"),
        ("see you later", "à plus tard"),
        ("i am sorry", "je suis désolé"),
        ("where is", "où est"),
        ("how much", "combien"),
        ("i do not understand", "je ne comprends pas"),
        ("do you speak english", "parlez vous anglais"),
        ("i speak french", "je parle français"),
        ("good luck", "bonne chance"),
        ("happy birthday", "joyeux anniversaire"),
        ("merry christmas", "joyeux noël"),
        ("happy new year", "bonne année"),
        ("congratulations", "félicitations"),
        ("i am hungry", "j'ai faim"),
        ("i am thirsty", "j'ai soif"),
        ("the weather is nice", "il fait beau"),
        ("it is cold", "il fait froid"),
        ("it is hot", "il fait chaud"),
        ("i like", "j'aime"),
        ("i do not like", "je n'aime pas"),
        ("very good", "très bien"),
        ("not bad", "pas mal"),
        ("help me", "aidez moi"),
        ("i need", "j'ai besoin"),
        ("i want", "je veux"),
        ("i can", "je peux"),
        ("i cannot", "je ne peux pas"),
        ("i know", "je sais"),
        ("i do not know", "je ne sais pas"),
        ("where are you from", "d'où venez vous"),
        ("i am from", "je viens de"),
        ("what time is it", "quelle heure est il"),
        ("today", "aujourd'hui"),
        ("tomorrow", "demain"),
        ("yesterday", "hier"),
    ]
    
    # Build vocabularies
    src_words = set()
    tgt_words = set()
    
    for src, tgt in translation_pairs:
        src_words.update(src.split())
        tgt_words.update(tgt.split())
    
    # Add special tokens
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
    
    src_word2idx = {word: idx for idx, word in enumerate(special_tokens)}
    src_word2idx.update({word: idx + len(special_tokens) for idx, word in enumerate(sorted(src_words))})
    src_idx2word = {idx: word for word, idx in src_word2idx.items()}
    
    tgt_word2idx = {word: idx for idx, word in enumerate(special_tokens)}
    tgt_word2idx.update({word: idx + len(special_tokens) for idx, word in enumerate(sorted(tgt_words))})
    tgt_idx2word = {idx: word for word, idx in tgt_word2idx.items()}
    
    src_vocab = {
        'word2idx': src_word2idx,
        'idx2word': src_idx2word,
        'vocab_size': len(src_word2idx)
    }
    
    tgt_vocab = {
        'word2idx': tgt_word2idx,
        'idx2word': tgt_idx2word,
        'vocab_size': len(tgt_word2idx)
    }
    
    # Convert pairs to indices
    def sentence_to_indices(sentence, word2idx, max_len):
        words = sentence.split()
        indices = [word2idx['<sos>']] + [word2idx.get(word, word2idx['<unk>']) for word in words] + [word2idx['<eos>']]
        # Pad or truncate
        if len(indices) > max_len:
            indices = indices[:max_len-1] + [word2idx['<eos>']]
        else:
            indices += [word2idx['<pad>']] * (max_len - len(indices))
        return indices
    
    src_sequences = []
    tgt_sequences = []
    
    for src, tgt in translation_pairs:
        src_seq = sentence_to_indices(src, src_word2idx, max_len)
        tgt_seq = sentence_to_indices(tgt, tgt_word2idx, max_len)
        src_sequences.append(src_seq)
        tgt_sequences.append(tgt_seq)
    
    # Convert to tensors
    src_tensor = torch.LongTensor(src_sequences)
    tgt_tensor = torch.LongTensor(tgt_sequences)
    
    # Split into train and validation
    n_train = int(len(src_sequences) * train_split)
    indices = torch.randperm(len(src_sequences))
    
    train_src = src_tensor[indices[:n_train]]
    train_tgt = tgt_tensor[indices[:n_train]]
    val_src = src_tensor[indices[n_train:]]
    val_tgt = tgt_tensor[indices[n_train:]]
    
    # Create data loaders
    train_dataset = TensorDataset(train_src, train_tgt)
    val_dataset = TensorDataset(val_src, val_tgt)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, src_vocab, tgt_vocab
