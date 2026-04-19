import torch

class AverageMeter:
    """
    Computes and stores the average and current value of a metric.
    Typical usage: tracking loss and accuracy over epoch batches.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Updates the running average given a new value and its batch size count.
        
        Args:
            val (float): The current metric value (e.g., loss.item()).
            n (int): The batch size or number of elements the value represents.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes the top-1 classification accuracy.
    
    Args:
        outputs (torch.Tensor): Logits or probabilities from the model of shape (N, C).
        targets (torch.Tensor): Ground truth integer class labels of shape (N,).
        
    Returns:
        float: Accuracy percentage (0.0 to 100.0).
    """
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    return (correct / total) * 100.0
