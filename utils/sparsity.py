import torch
from typing import Dict
from model.prunable_layer import PrunableLinear

def calculate_sparsity_loss(model: torch.nn.Module) -> torch.Tensor:
    """
    Calculates the normalized L1-style sparsity loss based on gate activations.
    
    The loss summates the sigmoid activations of the gate scores across all 
    PrunableLinear layers. It is strictly normalized by the total number of
    gate parameters to prevent mathematical overflow against the primary 
    task loss (e.g., CrossEntropy).
    
    Args:
        model (torch.nn.Module): The network containing PrunableLinear layers.
        
    Returns:
        torch.Tensor: The normalized scalar sparsity loss.
    """
    sparsity_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    total_params = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores * getattr(module, 'temperature', 1.0))
            sparsity_loss += torch.sum(gates)
            total_params += gates.numel()
    
    if total_params > 0:
        sparsity_loss = sparsity_loss / total_params
        
    return sparsity_loss

def get_sparsity_metrics(model: torch.nn.Module, threshold: float = 0.05) -> Dict[str, float]:
    """
    Computes structural sparsity metrics including the effective parameter reduction.
    
    Any gate activation evaluating strictly below the parameterized threshold 
    is considered physically severed from the continuous computational graph.
    
    Args:
        model (torch.nn.Module): The network subject to evaluation.
        threshold (float): Gate activation ceiling to classify a parameter as "dead".
        
    Returns:
        dict: A dictionary mapping metric names to their numeric values.
    """
    total_gates = 0
    pruned_gates = 0
    
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            # Compute static gates (no gradients needed for metrics)
            with torch.no_grad():
                gates = torch.sigmoid(module.gate_scores * getattr(module, 'temperature', 1.0))
                total_gates += gates.numel()
                pruned_gates += torch.sum(gates < threshold).item()
            
    sparsity_percentage = (pruned_gates / total_gates) * 100 if total_gates > 0 else 0.0
    active_percentage = 100.0 - sparsity_percentage
    
    return {
        'total_params': total_gates,
        'pruned_params': pruned_gates,
        'sparsity_percentage': sparsity_percentage,
        'active_percentage': active_percentage,
        'effective_reduction_ratio': float(total_gates) / float(total_gates - pruned_gates + 1e-8) if total_gates > 0 else 1.0
    }
    
def get_all_gate_values(model: torch.nn.Module) -> torch.Tensor:
    """
    Aggregates a detached, flattened 1D tensor of all gate activations post-sigmoid.
    
    Args:
        model (torch.nn.Module): The network containing PrunableLinear layers.
        
    Returns:
        torch.Tensor: 1D tensor of all activation boundaries.
    """
    all_gates = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            with torch.no_grad():
                gates = torch.sigmoid(module.gate_scores * getattr(module, 'temperature', 1.0)).cpu().view(-1)
                all_gates.append(gates)
    
    if all_gates:
        return torch.cat(all_gates)
    return torch.tensor([])
