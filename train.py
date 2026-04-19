import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm

from model.network import SelfPruningNet, BaselineNet
from utils.sparsity import calculate_sparsity_loss, get_sparsity_metrics
from utils.metrics import AverageMeter, calculate_accuracy

# Setup execution device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def set_seed(seed: int = 42) -> None:
    """Fixes the random seeds globally for execution reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, criterion, optimizer, lambda_sparsity, epoch, is_prunable=True):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch:02d}", position=0, leave=True, ncols=100)
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        ce_loss = criterion(outputs, targets)
        
        # Integrate structurally sparse subspace regularization
        if is_prunable and lambda_sparsity > 0:
            sparsity_loss = calculate_sparsity_loss(model)
            total_loss = ce_loss + lambda_sparsity * sparsity_loss
        else:
            total_loss = ce_loss
            
        total_loss.backward()
        optimizer.step()
        
        acc = calculate_accuracy(outputs, targets)
        loss_meter.update(total_loss.item(), inputs.size(0))
        acc_meter.update(acc, inputs.size(0))
        
        pbar.update(1)
        pbar.set_postfix({'Loss': f"{loss_meter.avg:.4f}", 'Acc': f"{acc_meter.avg:.2f}%"})
        
    pbar.close()
    return loss_meter.avg, acc_meter.avg

def evaluate(model, dataloader, criterion):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            ce_loss = criterion(outputs, targets)
            
            acc = calculate_accuracy(outputs, targets)
            loss_meter.update(ce_loss.item(), inputs.size(0))
            acc_meter.update(acc, inputs.size(0))
            
    return loss_meter.avg, acc_meter.avg

def execute_pipeline(model, trainloader, testloader, criterion, args, config_name, lamba=0.0, is_prunable=True):
    print(f"\n{'-'*60}")
    print(f"🚀 Initializing Optimization: {config_name}")
    print(f"{'-'*60}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, lamba, epoch, is_prunable)
        val_loss, val_acc = evaluate(model, testloader, criterion)
        scheduler.step()
        
        if is_prunable:
            metrics = get_sparsity_metrics(model)
            spa = metrics['sparsity_percentage']
            red = metrics['effective_reduction_ratio']
            print(f"   [Valid] Loss: {val_loss:.4f} | Acc: {val_acc:5.2f}% | Dead Params: {spa:5.2f}% | Par Reduction: {red:.2f}x")
        else:
            print(f"   [Valid] Loss: {val_loss:.4f} | Acc: {val_acc:5.2f}% | Dead Params:   N/A  | Par Reduction: 1.00x")
            
    print(f"\nEvaluating generalizable characteristics for {config_name}...")
    final_val_loss, final_val_acc = evaluate(model, testloader, criterion)
    
    if is_prunable:
        metrics = get_sparsity_metrics(model)
        final_sparsity = metrics['sparsity_percentage']
        print(f"✨ Final Outcome -> Accuracy: {final_val_acc:.2f}%, Structural Sparsity: {final_sparsity:.2f}%")
        return final_val_acc, final_sparsity
    else:
        print(f"✨ Final Outcome -> Accuracy: {final_val_acc:.2f}%")
        return final_val_acc, 0.0

def main():
    parser = argparse.ArgumentParser(description="Self-Pruning Neural Network Regularizer")
    parser.add_argument('--batch-size', type=int, default=256, help='Input batch capacity.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Adam learning rate.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epoch bounds.')
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.001, 0.1, 1.0], help='L1 Sparse regularization vector.')
    parser.add_argument('--seed', type=int, default=42, help='Reproducibility initialization seed.')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs('experiments/models', exist_ok=True)
    
    # Load dataset representation bounds
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    print("Mounting CIFAR-10 evaluation space...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    criterion = nn.CrossEntropyLoss()
    results = []
    
    # 1. Establish Generalization Baseline Control
    baseline_model = BaselineNet().to(DEVICE)
    base_acc, _ = execute_pipeline(baseline_model, trainloader, testloader, criterion, args, "BaselineNet (Control Graph)", is_prunable=False)
    results.append({'model': 'Baseline', 'lambda': 0.0, 'accuracy': base_acc, 'sparsity': 0.0})
    
    # 2. Iterate Pruned Graphs mapped over sparsity limits
    for lamba in args.lambdas:
        model = SelfPruningNet().to(DEVICE)
        acc, spa = execute_pipeline(model, trainloader, testloader, criterion, args, f"SelfPruningNet (Penalty={lamba})", lamba=lamba, is_prunable=True)
        results.append({'model': 'SelfPruningNet', 'lambda': lamba, 'accuracy': acc, 'sparsity': spa})
        torch.save(model.state_dict(), f"experiments/models/model_lambda_{lamba}.pth")
        
    print(f"\n{'-'*60}")
    print("Optimization Completed. Aggregating results...")
    
    df = pd.DataFrame(results)
    df.to_csv('experiments/results.csv', index=False)
    print("Result vectors successfully flushed to experiments/results.csv")

if __name__ == '__main__':
    main()
