"""
Utility functions for training and evaluating GNN models
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4, verbose=True):
    """
    Train a GNN model
    
    Args:
        model: The GNN model to train
        data: The graph data
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        verbose: Whether to print training progress
    
    Returns:
        The trained model
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if verbose and epoch % 50 == 0:
            val_acc = validate_model(model, data)
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return model

def validate_model(model, data):
    """
    Validate a model on the validation set
    
    Args:
        model: The trained model
        data: The graph data
    
    Returns:
        Validation accuracy
    """
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        val_acc = int((pred[data.val_mask] == data.y[data.val_mask]).sum()) / int(data.val_mask.sum())
    return val_acc

def test_model(model, data):
    """
    Test a model on the test set
    
    Args:
        model: The trained model
        data: The graph data
    
    Returns:
        Test accuracy
    """
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        test_acc = int((pred[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())
    return test_acc

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, data):
    """Print a summary of the model"""
    num_params = count_parameters(model)
    print(f"Model Parameters: {num_params:,}")
    print(f"Input Features: {data.num_features}")
    print(f"Output Classes: {data.y.max().item() + 1}")

def test_learning_rates(model_class, data, dataset, learning_rates=[0.001, 0.01, 0.1]):
    """Test different learning rates for a specific model"""
    print(f"\n🧪 Learning Rate Test for {model_class.__name__}")
    print("-" * 40)
    
    results = []
    
    for lr in learning_rates:
        print(f"Testing LR: {lr}")
        
        # Create fresh model
        model = model_class(dataset.num_features, 32, dataset.num_classes, 0.5)
        
        # Train with specific LR
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        model.train()
        
        for epoch in range(150):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
        
        # Test
        test_acc = test_model(model, data)
        results.append((lr, test_acc))
        print(f"  → Test Acc: {test_acc:.4f}")
    
    # Show best
    best_lr, best_acc = max(results, key=lambda x: x[1])
    print(f"🏆 Best LR: {best_lr} (Acc: {best_acc:.4f})")
    
    return results

def compare_model_sizes(dataset):
    """Compare parameter counts of different models"""
    from models.gnn_models import create_model, MODEL_TYPES
    
    print("\n📊 Model Size Comparison")
    print("-" * 30)
    
    hidden_dim = 32
    for model_type in MODEL_TYPES:
        model = create_model(model_type, dataset.num_features, hidden_dim, dataset.num_classes)
        params = count_parameters(model)
        print(f"{model_type:12s}: {params:,} parameters")

def get_dataset_info(dataset):
    """Get comprehensive dataset information"""
    data = dataset[0]
    
    info = {
        'name': dataset.__class__.__name__,
        'num_nodes': data.num_nodes,
        'num_edges': data.num_edges,
        'num_features': data.num_features,
        'num_classes': dataset.num_classes,
        'train_nodes': int(data.train_mask.sum()),
        'val_nodes': int(data.val_mask.sum()),
        'test_nodes': int(data.test_mask.sum()),
        'edge_density': data.num_edges / (data.num_nodes * (data.num_nodes - 1))
    }
    
    return info

def print_dataset_stats(dataset):
    """Print detailed dataset statistics"""
    info = get_dataset_info(dataset)
    
    print(f"\n📈 Dataset Statistics")
    print("-" * 25)
    print(f"Dataset: {info['name']}")
    print(f"Nodes: {info['num_nodes']:,}")
    print(f"Edges: {info['num_edges']:,}")
    print(f"Features: {info['num_features']:,}")
    print(f"Classes: {info['num_classes']:,}")
    print(f"Train/Val/Test: {info['train_nodes']}/{info['val_nodes']}/{info['test_nodes']}")
    print(f"Edge Density: {info['edge_density']:.6f}")