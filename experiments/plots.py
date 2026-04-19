import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Bind core application into path for execution from anywhere
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.network import SelfPruningNet
from utils.sparsity import get_all_gate_values

def set_style():
    """Sets a professional, rigorously academic aesthetic for the visualizations."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.6,
        'grid.linestyle': '--'
    })

def plot_gate_distributions(lambdas=[0.001, 0.1, 1.0], models_dir='experiments/models', output_dir='experiments'):
    set_style()
    fig, axes = plt.subplots(1, len(lambdas), figsize=(18, 5), sharey=True)
    fig.suptitle('Continuous Pruning Gate Distributions across Sparsity Penalties ($\lambda$)', 
                 fontsize=18, y=1.05, fontweight='bold')
    
    if len(lambdas) == 1: axes = [axes]  # Normalize axes list
    
    colors = ['#1f77b4', '#ff7f0e', '#d62728'] # Standard discrete mapping
    
    for i, (lam, ax) in enumerate(zip(lambdas, axes)):
        model_path = os.path.join(models_dir, f'model_lambda_{lam}.pth')
        if not os.path.exists(model_path):
            print(f"[Warning] Model artifacts for lambda={lam} missing. Skipping.")
            continue
            
        model = SelfPruningNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        
        gates = get_all_gate_values(model)
        if gates.numel() == 0:
            print(f"[Warning] No structured gate constraints isolated in lambda={lam}.")
            continue
            
        gates_np = gates.numpy()
        
        # Calculate localized spike density to annotate
        dead_gates = np.sum(gates_np < 0.05)
        total_gates = len(gates_np)
        sparsified_pct = (dead_gates / total_gates) * 100 if total_gates > 0 else 0
            
        sns.histplot(gates_np, bins=60, binrange=(0, 1), 
                     color=colors[i % len(colors)], edgecolor='black', ax=ax, alpha=0.8)
        
        ax.set_title(f"$\lambda$ = {lam} | Sparsity: {sparsified_pct:.1f}%", fontweight='bold')
        ax.set_xlabel('Gate Activation Output ($\sigma$)')
        
        if i == 0:
            ax.set_ylabel('Parameter Count Density')
            
        # Draw explicit boundary marker to clarify the mathematical subgradient threshold
        ax.axvline(x=0.05, color='black', linestyle='-.', lw=1.5, alpha=0.7)
        ax.text(0.06, ax.get_ylim()[1]*0.80, f'Severed\n< 0.05', color='black', alpha=0.8, fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'gate_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated high-fidelity visualization: {plot_path}")
    plt.close()

if __name__ == '__main__':
    plot_gate_distributions()
