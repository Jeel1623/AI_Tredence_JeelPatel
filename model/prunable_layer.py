import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    """
    A linear layer that actively sparsifies its computational graph during training.
    
    This layer parameterizes dense weights alongside corresponding continuous gate 
    scores. Through a temperature-scaled Sigmoid activation, the gates are coerced 
    toward binary states (0 or 1) by an L1 subgradient penalty, structurally 
    severing unneeded parameters without discontinuous post-training pruning.
    
    Args:
        in_features (int): Number of input feature representations.
        out_features (int): Number of output feature representations.
        bias (bool, optional): Whether to allocate a learnable bias. Defaults to True.
        temperature (float, optional): Scaling factor applied sequentially before 
            the Sigmoid to accelerate convergence to bounds. Defaults to 5.0.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, temperature: float = 5.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        
        # Primary dense capacity
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Continuous gating scores driving the structural capacity bounds
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """
        Initializes parameter tensors using Kaiming Uniform for weights to 
        preserve variance through non-linearities, and zero-mean Gaussian 
        for gates to enforce neutral initial capacity (Sigmoid(0) = 0.5).
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            
        # Initialization injected with minimal scalar noise to break symmetry
        # At temperature 5.0, mean 0 computes an neutral 0.5 survival probability.
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the sparse linear formulation.
        
        Args:
            x (torch.Tensor): Continuous activation input of shape (Batch, ..., in_features).
            
        Returns:
            torch.Tensor: Sparsified activation output of shape (Batch, ..., out_features).
        """
        # The sequential temperature scalar amplifies the subgradient step size, 
        # accelerating the gate convergence toward definitive binary cutoffs.
        gates = torch.sigmoid(self.gate_scores * self.temperature)
        
        # Soft-mask application executing the structural pruning dynamically.
        pruned_weight = self.weight * gates
        
        return F.linear(x, pruned_weight, self.bias)
