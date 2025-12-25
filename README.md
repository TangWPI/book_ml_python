# Information Theory for Modern Machine Learning: Python Practice

This repository contains Python implementations of various **machine learning algorithms from scratch**, designed as companion code for the textbook "**Information Theory for Modern Machine Learning: From Theory to Python Practice**." [Accessible in Amazon](https://www.amazon.com/Information-Theory-Modern-Machine-Learning-ebook/dp/B0G873VY8M) 

## Overview

All algorithms are implemented from scratch using PyTorch and demonstrated in Jupyter notebooks. The implementations cover:

- **Dimensionality Reduction**: PCA, ICA, Autoencoder
- **Classification**: Neural Networks (Multi-layer Perceptron)
- **Clustering**: K-Means
- **Generative Learning**: VAE, Normalizing Flow (RealNVP), Diffusion Models (DDPM), Transformer for Text Translation

Both synthetic (Gaussian) and real-world datasets (MNIST, CIFAR-10) are used throughout, along with a small English-French translation dataset for the transformer model.

## Installation

```bash
# Clone the repository
git clone https://github.com/TangWPI/book_ml_python.git
cd book_ml_python

# Install dependencies (virtual environment recommended, e.g., conda)
pip install -r requirements.txt
```

## Repository Structure

```
book_information_theory_python/
├── src/
│   ├── dimensionality_reduction/
│   │   ├── pca.py                 # Principal Component Analysis
│   │   ├── ica.py                 # Independent Component Analysis
│   │   └── autoencoder.py         # Autoencoder
│   ├── classification/
│   │   └── neural_network.py      # Neural Network Classifier
│   ├── clustering/
│   │   └── kmeans.py              # K-Means Clustering
│   ├── generative/
│   │   ├── vae.py                 # Variational Autoencoder
│   │   ├── normalizing_flow.py    # RealNVP Normalizing Flow
│   │   ├── diffusion.py           # Diffusion Model (DDPM)
│   │   └── transformer.py         # Transformer for Text Translation
│   └── utils/
│       ├── data_loader.py         # Data loading utilities
│       ├── visualization.py       # Visualization functions
│       └── helpers.py             # Helper functions
├── notebooks/
│   ├── dimensionality_reduction/
│   ├── classification/
│   ├── clustering/
│   └── generative/
├── requirements.txt
└── README.md
```

## Quick Start

### Dimensionality Reduction

**PCA Example:**
```python
from src.dimensionality_reduction import PCA
from src.utils import generate_gaussian_data, plot_dimensionality_reduction

# Generate synthetic data
X, y = generate_gaussian_data(n_samples=1000, n_features=10, n_clusters=3)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Visualize
plot_dimensionality_reduction(X, X_reduced, y, method='PCA')
```

**ICA Example:**
```python
from src.dimensionality_reduction import ICA

# Apply ICA
ica = ICA(n_components=2)
X_independent = ica.fit_transform(X)

# Visualize
plot_dimensionality_reduction(X, X_independent, y, method='ICA')
```

**Autoencoder Example:**
```python
from src.dimensionality_reduction import Autoencoder, AutoencoderTrainer
from src.utils import load_mnist, get_device, plot_training_curves

# Load data
train_loader = load_mnist(batch_size=128, train=True)

# Create and train autoencoder
device = get_device()
model = Autoencoder(input_dim=784, latent_dim=32)
trainer = AutoencoderTrainer(model, device=device)
history = trainer.train(train_loader, n_epochs=20)

# Visualize
plot_training_curves(history["train_losses"], val_losses=None, title='Training Curves')
```

### Classification

**Neural Network Classifier Example:**
```python
from src.classification import NeuralNetworkClassifier, NNClassifierTrainer
from src.utils import load_mnist, get_device

# Load data
train_loader = load_mnist(batch_size=128, train=True)
test_loader = load_mnist(batch_size=128, train=False)

# Create and train classifier
device = get_device()
model = NeuralNetworkClassifier(input_dim=784, output_dim=10, hidden_dims=[256, 128])
trainer = NNClassifierTrainer(model, device=device)
history = trainer.train(train_loader, n_epochs=20, val_loader=test_loader)

# Evaluate
test_loss, test_acc = trainer.evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.2f}%")
```

### Clustering

**K-Means Example:**
```python
from src.clustering import KMeans
from src.utils import generate_gaussian_data, plot_2d_data

# Generate data
X, _ = generate_gaussian_data(n_samples=500, n_features=2, n_clusters=4)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

# Visualize
plot_2d_data(X, labels, title='K-Means Clustering')
```

### Generative Learning

**VAE Example:**
```python
from src.generative import VAE, VAETrainer
from src.utils import load_mnist, get_device, plot_image_grid

# Load data
train_loader = load_mnist(batch_size=128, train=True)

# Create and train VAE
device = get_device()
model = VAE(input_dim=784, latent_dim=20)
trainer = VAETrainer(model, device=device)
history = trainer.train(train_loader, n_epochs=20)

# Generate samples
samples = model.sample(n_samples=64, device=device)

# Visualize
plot_image_grid(samples, n_rows=4, n_cols=8, figsize=(12, 6), title='Image Grid')
```

**Normalizing Flow Example with Gaussian Mixture:**
```python
from src.generative import RealNVP, NormalizingFlowTrainer
from src.utils import generate_gaussian_mixture, create_dataloader, plot_2d_data

# Generate data
X = generate_gaussian_mixture(n_samples=1000, n_features=2)
train_loader = create_dataloader(X, batch_size=128)

# Create and train flow
device = get_device()
model = RealNVP(input_dim=2, n_flows=8)
trainer = NormalizingFlowTrainer(model, device=device)
history = trainer.train(train_loader, n_epochs=50)

# Generate samples
samples = model.sample(n_samples=100, device=device)

# Visualize
plot_2d_data(samples, title='Sampled Data Visualization')
```

**Normalizing Flow Example with MNIST:**
```python
from src.generative import RealNVP, NormalizingFlowTrainer
from src.utils import generate_gaussian_mixture, create_dataloader, plot_image_grid

# Load MNIST dataset
train_loader = load_mnist(batch_size=128)

# Create and train flow
device = get_device()
model = RealNVP(input_dim=784, n_flows=100)
trainer = NormalizingFlowTrainer(model, device=device)
history = trainer.train(train_loader, n_epochs=50)

# Generate samples
samples = model.sample(n_samples=100, device=device)

# Visualize
plot_image_grid(samples, n_rows=4, n_cols=8, figsize=(12, 6), title='Image Grid')
```

**Diffusion Model Example:**
```python
from src.generative import DiffusionModel, DiffusionTrainer
from src.utils import load_mnist, get_device, plot_image_grid

# Load data
train_loader = load_mnist(batch_size=128, train=True)

# Create and train diffusion model
device = get_device()
model = DiffusionModel(input_dim=784, n_timesteps=1000)
trainer = DiffusionTrainer(model, device=device)
history = trainer.train(train_loader, n_epochs=20)

# Generate samples
samples = model.sample(n_samples=64, device=device)

# Visualize
plot_image_grid(samples, n_rows=4, n_cols=8, figsize=(12, 6), title='Image Grid')
```

**Transformer for Text Translation Example:**
```python
from src.generative import Transformer, TransformerTrainer
from src.utils import load_translation_data, get_device, save_checkpoint, load_checkpoint

# Load English-French translation data
train_loader, val_loader, src_vocab, tgt_vocab = load_translation_data(batch_size=32)

# Create transformer model
device = get_device()
model = Transformer(
    src_vocab_size=src_vocab['vocab_size'],
    tgt_vocab_size=tgt_vocab['vocab_size'],
    d_model=256,
    n_heads=8,
    n_encoder_layers=3,
    n_decoder_layers=3,
    d_ff=512,
    max_len=50,
    dropout=0.1
)

# Train the model
trainer = TransformerTrainer(model, device=device)
history = trainer.train(train_loader, n_epochs=50, learning_rate=1e-4, val_loader=val_loader)

# Save trained model
save_checkpoint(model, 'transformer_translation.pth')

# Load pre-trained model for testing
model_loaded = Transformer(
    src_vocab_size=src_vocab['vocab_size'],
    tgt_vocab_size=tgt_vocab['vocab_size'],
    d_model=256,
    n_heads=8,
    n_encoder_layers=3,
    n_decoder_layers=3
)
load_checkpoint(model_loaded, 'transformer_translation.pth')

# Translate a sentence
import torch
def translate_sentence(sentence, model, src_vocab, tgt_vocab, device='cpu'):
    """Translate an English sentence to French."""
    model.eval()
    # Tokenize and convert to indices
    words = sentence.lower().split()
    src_indices = [src_vocab['word2idx']['<sos>']] + \
                  [src_vocab['word2idx'].get(w, src_vocab['word2idx']['<unk>']) for w in words] + \
                  [src_vocab['word2idx']['<eos>']]
    src_tensor = torch.LongTensor([src_indices]).to(device)
    
    # Translate
    output = model.translate(src_tensor, max_len=50, device=device)
    
    # Convert indices back to words
    translated_words = [tgt_vocab['idx2word'][idx.item()] for idx in output[0]]
    # Remove special tokens
    translated_words = [w for w in translated_words if w not in ['<sos>', '<eos>', '<pad>']]
    
    return ' '.join(translated_words)

# Example translation
translation = translate_sentence("hello", model_loaded, src_vocab, tgt_vocab, device=device)
print(f"Translation: {translation}")
```

## Jupyter Notebooks

Detailed tutorials and experiments are available in the `notebooks/` directory:

- `notebooks/dimensionality_reduction/`: PCA, ICA, and Autoencoder tutorials
- `notebooks/classification/`: Neural network classification examples
- `notebooks/clustering/`: K-Means clustering demonstrations
- `notebooks/generative/`: VAE, Normalizing Flow, and Diffusion model tutorials

## Features

- **Pure PyTorch Implementations**: All algorithms implemented from scratch using PyTorch
- **Educational Focus**: Clear, well-documented code for learning
- **Comprehensive Examples**: Both synthetic and real-world datasets
- **Visualization Tools**: Built-in functions for visualizing results
- **Modular Design**: Easy to use and extend

## Algorithms Implemented

### Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Linear dimensionality reduction via eigenvalue decomposition
- **ICA (Independent Component Analysis)**: Blind source separation using FastICA algorithm
- **Autoencoder**: Non-linear dimensionality reduction using neural networks

### Classification
- **Neural Network Classifier**: Multi-layer perceptron with customizable architecture

### Clustering
- **K-Means**: Iterative clustering using K-Means++ initialization

### Generative Learning
- **VAE (Variational Autoencoder)**: Probabilistic encoder-decoder with latent sampling
- **Normalizing Flow (RealNVP)**: Invertible transformations for density estimation
- **Diffusion Model (DDPM)**: Denoising diffusion probabilistic model
- **Transformer**: Sequence-to-sequence model with multi-head attention for text translation

## Datasets

The repository supports the following datasets:

- **Synthetic Gaussian Data**: Customizable multi-cluster Gaussian data
- **Translation Data**: Small curated English-French phrase pairs for translation tasks
- **MNIST**: Handwritten digits (28x28 grayscale images)
- **CIFAR-10**: Natural images (32x32 RGB images, 10 classes)

## Requirements

- Python >= 3.8
- PyTorch >= 2.6.0 (recommended for security fixes)
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- Jupyter >= 1.0.0

See `requirements.txt` for complete list.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is intended for educational purposes as companion code to the textbook "Information Theory for Machine Learning: From Theory to Python Practice."

## Citation

If you use this code in your research or projects, please cite:

```
Tang, B. (2025). Information Theory for Modern Machine Learning: From Theory to Python Practice.
```

## About the Author

Dr. Tang is an Associate Professor at Worcester Polytechnic Institute with extensive research and teaching experience at the intersection of information theory and machine learning. His work focuses on statistical machine learning, knowledge discovery, and the theoretical foundations that enable intelligent systems to interpret and act on data.

## Contact

For questions or feedback, please open an issue on GitHub.
