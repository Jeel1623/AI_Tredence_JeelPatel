import torch
import torch.nn as nn
from typing import List, Optional
from .prunable_layer import PrunableLinear

class SelfPruningNet(nn.Module):
    """
    A foundational feed-forward network utilizing the self-pruning architecture.
    
    Integrates the custom PrunableLinear wrappers sequentially against standard 
    regularization invariants (1D Batch Normalization & Dropout) to stabilize 
    internal covariant shifts inherently induced when gates sever weights 
    during active gradient descent.
    
    Args:
        input_size (int): Expected dimension of the flattened input tensor.
        hidden_sizes (List[int]): Dimensions for the sequential hidden representations.
        num_classes (int): Cardinality of the output classification domain.
        dropout_p (float): Dropout probability applied post-activation.
    """
    def __init__(self, input_size: int = 3072, hidden_sizes: List[int] = [1024, 512, 256], 
                 num_classes: int = 10, dropout_p: float = 0.3) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList()
        in_features = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(PrunableLinear(in_features, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout_p))
            in_features = hidden_size
            
        self.output = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class BaselineNet(nn.Module):
    """
    An architecturally identical map to SelfPruningNet engineered using strictly 
    standard non-pruning `nn.Linear` layers. Employed exclusively as the 
    control baseline for evaluation generalization.
    """
    def __init__(self, input_size: int = 3072, hidden_sizes: List[int] = [1024, 512, 256], 
                 num_classes: int = 10, dropout_p: float = 0.3) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList()
        in_features = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_features, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout_p))
            in_features = hidden_size
            
        self.output = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
