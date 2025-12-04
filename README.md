# Information Theory for Machine Learning: Python Practice

This repository contains Python implementations and Jupyter notebooks for the textbook **"Information Theory for Machine Learning: From Theory to Python Practice"**.

## Overview

This project implements various machine learning algorithms from scratch using PyTorch, covering:

- **Dimensionality Reduction**: PCA, ICA, Autoencoder
- **Classification**: Neural Networks
- **Clustering**: Various clustering algorithms
- **Generative Learning**: VAE, Normalizing Flow, Diffusion Models

All implementations include examples with both synthetic data (Gaussian distributions) and real-world datasets (MNIST, CIFAR-10).

## Project Structure

```
book_information_theory_python/
├── dimensionality_reduction/    # Dimensionality reduction techniques
│   ├── pca/                    # Principal Component Analysis
│   ├── ica/                    # Independent Component Analysis
│   └── autoencoder/            # Autoencoder implementations
├── classification/              # Classification algorithms
│   └── neural_networks/        # Neural network classifiers
├── clustering/                  # Clustering algorithms
├── generative_models/           # Generative learning methods
│   ├── vae/                    # Variational Autoencoders
│   ├── normalizing_flow/       # Normalizing Flow models
│   └── diffusion/              # Diffusion models
├── data/                        # Data management
│   ├── synthetic/              # Synthetic data generation
│   └── real_world/             # Real-world datasets (MNIST, CIFAR-10)
├── notebooks/                   # Jupyter notebooks with examples
└── utils/                       # Shared utility functions
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Jupyter Notebook
- NumPy
- Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/TangWPI/book_information_theory_python.git
cd book_information_theory_python

# Install dependencies (when requirements.txt is available)
pip install -r requirements.txt
```

## Usage

Each subdirectory contains implementations and examples. Jupyter notebooks in the `notebooks/` folder provide interactive demonstrations of the theory and implementations.

## Contributing

This repository is designed for educational purposes as a companion to the textbook. Contributions and suggestions are welcome.

## License

See LICENSE file for details.

## Author

Tang, WPI
