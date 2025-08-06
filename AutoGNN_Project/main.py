"""
AutoGNN Project - Main Entry Point
A simple Neural Architecture Search for Graph Neural Networks
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Import our custom modules
from models.gnn_models import GCN, GAT, GraphSAGE
from search.search_algorithm import SimpleNAS
from utils import train_model, test_model, set_seed

def quick_model_comparison(dataset):
    """Compare all models with fixed hyperparameters"""
    print("\n🚀 BONUS: Quick Model Comparison")
    print("="*50)
    
    data = dataset[0]
    hidden_dim = 32
    dropout = 0.5
    epochs = 100
    
    print(f"Testing all models with fixed params (hidden={hidden_dim}, dropout={dropout})")
    print("-" * 50)
    
    results = []
    
    # Import here to avoid circular imports
    from models.gnn_models import create_model, MODEL_TYPES
    
    for model_type in MODEL_TYPES:
        print(f"🧠 Testing {model_type}...")
        
        # Create and train model
        model = create_model(model_type, dataset.num_features, hidden_dim, dataset.num_classes, dropout)
        trained_model = train_model(model, data, epochs=epochs, verbose=False)
        
        # Test model
        test_acc = test_model(trained_model, data)
        results.append((model_type, test_acc))
        
        print(f"   {model_type}: {test_acc:.4f}")
    
    print("-" * 50)
    print("🏆 MODEL RANKING:")
    
    # Sort by accuracy
    results.sort(key=lambda x: x[1], reverse=True)
    for i, (model_type, acc) in enumerate(results, 1):
        print(f"{i}. {model_type}: {acc:.4f}")
    
    return results

def main():
    print("🚀 Starting AutoGNN Project")
    print("="*50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # CHOOSE YOUR DATASET HERE! 
    dataset_name = 'Cora'  # Options: 'Cora', 'CiteSeer', 'PubMed'
    
    # Step 1: Load the dataset
    print(f"📊 Loading {dataset_name} dataset...")
    dataset = Planetoid(root=f'data/{dataset_name}', name=dataset_name, transform=NormalizeFeatures())
    data = dataset[0]
    
    print(f"Dataset: {dataset}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print()
    
    # Step 2: Initialize the search algorithm
    print("🔍 Initializing Neural Architecture Search...")
    nas = SimpleNAS(
        dataset=dataset,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Step 3: Run the search
    print("🧠 Running AutoGNN search...")
    best_model, best_config, best_val_acc = nas.search()
    
    # Step 4: Test the best model
    print("\n🎯 Testing best model...")
    test_acc = test_model(best_model, data)
    
    # Step 5: Results
    print("\n" + "="*50)
    print("🏆 FINAL RESULTS")
    print("="*50)
    print(f"Best Architecture: {best_config['model_type']}")
    print(f"Best Hidden Dimensions: {best_config['hidden_dim']}")
    print(f"Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("="*50)
    
    # BONUS: Run quick comparison of all models
    comparison_results = quick_model_comparison(dataset)

if __name__ == "__main__":
    main()