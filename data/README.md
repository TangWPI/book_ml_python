# Data

This folder contains data management utilities and datasets.

## Structure

- `synthetic/`: Synthetic data generation (e.g., Gaussian distributions)
- `real_world/`: Real-world datasets (e.g., MNIST, CIFAR-10)

## Note

Large datasets should not be committed to the repository. Instead:
- Use data loading scripts that download datasets automatically
- Store only small sample datasets if necessary
- Add data files to `.gitignore`
