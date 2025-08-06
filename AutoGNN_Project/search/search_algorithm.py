"""
Simple Neural Architecture Search for GNNs
Implements a basic random search strategy
"""

import torch
import random
import numpy as np
from models.gnn_models import create_model, MODEL_TYPES, HIDDEN_DIMS, DROPOUT_RATES
from utils import train_model, validate_model

class SimpleNAS:
    """Simple Neural Architecture Search using random search"""
    
    def __init__(self, dataset, device, num_trials=9):
        self.dataset = dataset
        self.device = device
        self.num_trials = num_trials
        self.data = dataset[0].to(device)
        
        # Search results
        self.results = []
        
    def generate_random_config(self):
        """Generate a random architecture configuration"""
        config = {
            'model_type': random.choice(MODEL_TYPES),
            'hidden_dim': random.choice(HIDDEN_DIMS),
            'dropout': random.choice(DROPOUT_RATES)
        }
        return config
    
    def evaluate_config(self, config, trial_num):
        """Evaluate a single architecture configuration"""
        print(f"Trial {trial_num + 1}/{self.num_trials}: Testing {config['model_type']} "
              f"(hidden={config['hidden_dim']}, dropout={config['dropout']})")
        
        # Create model
        model = create_model(
            model_type=config['model_type'],
            num_features=self.dataset.num_features,
            hidden_dim=config['hidden_dim'],
            num_classes=self.dataset.num_classes,
            dropout=config['dropout']
        ).to(self.device)
        
        # Train the model
        trained_model = train_model(model, self.data, epochs=200, verbose=False)
        
        # Validate the model
        val_acc = validate_model(trained_model, self.data)
        
        print(f"  → Validation Accuracy: {val_acc:.4f}")
        
        return trained_model, val_acc
    
    def search(self):
        """Run the neural architecture search"""
        print(f"🔍 Starting search with {self.num_trials} trials...")
        print("-" * 50)
        
        best_model = None
        best_config = None
        best_val_acc = 0.0
        
        for trial in range(self.num_trials):
            # Generate random configuration
            config = self.generate_random_config()
            
            # Evaluate configuration
            model, val_acc = self.evaluate_config(config, trial)
            
            # Store results
            self.results.append({
                'config': config.copy(),
                'val_acc': val_acc,
                'model': model
            })
            
            # Update best model if necessary
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = config.copy()
                best_model = model
                print(f"  🎯 New best model! Validation Accuracy: {val_acc:.4f}")
        
        print("-" * 50)
        print(f"🏆 Search completed!")
        print(f"Best configuration: {best_config}")
        
        return best_model, best_config, best_val_acc
    
    def get_search_summary(self):
        """Get a summary of all search results"""
        if not self.results:
            return "No search results available"
        
        summary = "Search Results Summary:\n"
        summary += "=" * 50 + "\n"
        
        # Sort by validation accuracy
        sorted_results = sorted(self.results, key=lambda x: x['val_acc'], reverse=True)
        
        for i, result in enumerate(sorted_results[:5]):  # Top 5 results
            config = result['config']
            val_acc = result['val_acc']
            summary += f"{i+1}. {config['model_type']} (h={config['hidden_dim']}, "
            summary += f"d={config['dropout']}) - Val Acc: {val_acc:.4f}\n"
        
        return summary