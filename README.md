# AutoGNN: Neural Architecture Search for Graph Neural Networks

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.3+-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A comprehensive Neural Architecture Search (NAS) framework for Graph Neural Networks, designed to automatically discover optimal GNN architectures for node classification tasks.

## 🎯 Overview

AutoGNN implements an automated search system that evaluates multiple Graph Neural Network architectures and hyperparameter configurations to find the best-performing model for graph-based learning tasks. The framework supports popular GNN variants and provides detailed performance analysis.

### Key Features

- **Automated Architecture Search**: Intelligent exploration of GNN model space
- **Multiple GNN Architectures**: GCN, GAT, GraphSAGE, and GIN implementations  
- **Comprehensive Evaluation**: Validation-based model selection with test set evaluation
- **Multi-Dataset Support**: Compatible with Cora, CiteSeer, and PubMed datasets
- **Performance Analytics**: Detailed comparison and hyperparameter sensitivity analysis
- **Reproducible Results**: Fixed random seeds for consistent experiments

## 🏗️ Architecture

The framework consists of four main components:

```
AutoGNN/
├── main.py                 # Main execution pipeline
├── models/
│   └── gnn_models.py      # GNN architecture implementations
├── search/
│   └── search_algorithm.py # NAS algorithm implementation
├── utils.py               # Training and evaluation utilities
└── data/                  # Datasets (auto-downloaded)
```

### Supported Models

| Model | Description | Key Features |
|-------|-------------|--------------|
| **GCN** | Graph Convolutional Network | Spectral convolution, efficient |
| **GAT** | Graph Attention Network | Multi-head attention, interpretable |
| **GraphSAGE** | Sample and Aggregate | Inductive learning, scalable |
| **GIN** | Graph Isomorphism Network | Theoretical guarantees, powerful |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SukT03/AutoGNN_Project.git
   cd AutoGNN_Project
   ```

2. **Set up environment**
   ```bash
   conda create -n autognn python=3.10 -y
   conda activate autognn
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio
   pip install torch-geometric
   pip install numpy pandas matplotlib scikit-learn
   ```

### Usage

**Basic execution:**
```bash
python main.py
```

**Dataset selection:**
Edit `main.py` line 57 to choose dataset:
```python
dataset_name = 'Cora'      # Options: 'Cora', 'CiteSeer', 'PubMed'
```

## 📊 Results

### Benchmark Performance

| Dataset | Best Model | Test Accuracy | Validation Accuracy |
|---------|------------|---------------|-------------------|
| Cora | GAT (16 hidden) | 79.7% | 79.6% |
| CiteSeer | GCN (32 hidden) | 71.2% | 70.8% |
| PubMed | GraphSAGE (64 hidden) | 78.9% | 78.5% |

### Search Efficiency

- **Search Space**: 4 architectures × 4 hidden dimensions × 3 dropout rates = 48 configurations
- **Search Strategy**: Random search with early stopping
- **Typical Runtime**: 5-10 minutes on CPU, 2-3 minutes on GPU
- **Convergence**: Usually finds optimal architecture within 9 trials

## 🔧 Configuration

### Hyperparameter Search Space

```python
MODEL_TYPES = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
HIDDEN_DIMS = [16, 32, 64, 128]
DROPOUT_RATES = [0.3, 0.5, 0.7]
```

### Training Parameters

- **Epochs**: 200 (search), 100 (comparison)
- **Learning Rate**: 0.01
- **Weight Decay**: 5e-4
- **Optimizer**: Adam

## 📈 Advanced Features

### Model Comparison
The framework automatically runs a comprehensive comparison of all architectures with fixed hyperparameters for fair evaluation.

### Hyperparameter Sensitivity Analysis
Built-in utilities for testing learning rate sensitivity and architecture-specific optimizations.

### Extensibility
Easy to add new GNN architectures by extending the base classes in `models/gnn_models.py`.

## 🛠️ Technical Details

### Search Algorithm
- **Strategy**: Random search with validation-based selection
- **Evaluation**: 5-fold cross-validation on validation set
- **Selection**: Best validation accuracy with test set final evaluation
- **Reproducibility**: Fixed random seeds (42) for consistent results

### Model Implementation
- **Framework**: PyTorch Geometric
- **Device Support**: Automatic GPU detection with CPU fallback
- **Memory Efficiency**: Batch processing for large graphs
- **Gradient Flow**: Proper dropout and activation placement

## 📁 Project Structure

```
AutoGNN_Project/
├── main.py                    # 🚀 Main execution pipeline
├── utils.py                   # 🔧 Training and evaluation utilities  
├── requirements.txt           # 📦 Project dependencies
├── README.md                  # 📖 This file
├── models/
│   └── gnn_models.py         # 🧠 GNN architecture definitions
├── search/
│   └── search_algorithm.py   # 🔍 Neural Architecture Search implementation
└── data/                     # 📊 Datasets (auto-created)
    ├── Cora/
    ├── CiteSeer/
    └── PubMed/
```
<img width="570" height="1068" alt="Screenshot 2025-08-07 at 1 54 41 AM" src="https://github.com/user-attachments/assets/971292be-cdbf-49c7-b7eb-2dd0b3f56273" />


## 🎓 Academic Context

This implementation demonstrates key concepts in:

- **Graph Neural Networks**: Modern architectures for graph-structured data
- **Neural Architecture Search**: Automated machine learning for architecture design  
- **Node Classification**: Semi-supervised learning on citation networks
- **Hyperparameter Optimization**: Systematic exploration of model configurations

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Implement evolutionary search algorithms
- [ ] Add support for graph-level tasks
- [ ] Include more recent GNN architectures (GraphTransformer, etc.)
- [ ] Bayesian optimization for hyperparameter search
- [ ] Multi-GPU training support



**⭐ If you find this project helpful, please consider giving it a star!**
