"""
GNN Model Architectures
Defines different Graph Neural Network models for AutoGNN search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Sequential, Linear, ReLU

class GCN(nn.Module):
    """Graph Convolutional Network"""
    
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    """Graph Attention Network"""
    
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5, heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class GraphSAGE(nn.Module):
    """GraphSAGE Network"""
    
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First SAGE layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second SAGE layer
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class GIN(nn.Module):
    """Graph Isomorphism Network"""
    
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GIN, self).__init__()
        
        # GIN layers with MLPs
        self.conv1 = GINConv(Sequential(Linear(num_features, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, num_classes)))
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GIN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GIN layer
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

def create_model(model_type, num_features, hidden_dim, num_classes, dropout=0.5):
    """
    Factory function to create different GNN models
    
    Args:
        model_type (str): 'GCN', 'GAT', 'GraphSAGE', or 'GIN'
        num_features (int): Number of input features
        hidden_dim (int): Hidden layer dimension
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
    
    Returns:
        torch.nn.Module: The created model
    """
    if model_type == 'GCN':
        return GCN(num_features, hidden_dim, num_classes, dropout)
    elif model_type == 'GAT':
        return GAT(num_features, hidden_dim, num_classes, dropout)
    elif model_type == 'GraphSAGE':
        return GraphSAGE(num_features, hidden_dim, num_classes, dropout)
    elif model_type == 'GIN':
        return GIN(num_features, hidden_dim, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Available model types for search
MODEL_TYPES = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
HIDDEN_DIMS = [16, 32, 64, 128]
DROPOUT_RATES = [0.3, 0.5, 0.7]