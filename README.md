# Structurally Sparse Neural Networks ✂️🧠

A PyTorch-based Deep Learning Framework employing dynamic structural self-pruning. This architecture parameterizes fully continuous gating mechanisms in sequence with dense weights, actively training the network to mathematically sparsify its own computational graph during backpropagation via subgradient optimization.

## 🚀 Project Overview

Developed to evaluate active regularization techniques, this architecture demonstrates:
*   **Active Pruning:** A custom `PrunableLinear` module utilizing learned gating parameters, replacing disruptive post-training weight surgery.
*   **L1 Regularization:** Continuous gates are pushed precisely to zero utilizing a stable mathematical $L_1$ scalar bounding algorithm.
*   **Quantitative Generalization:** High sparsity extraction while preserving the latent topological capability.

---

## 📁 Architecture Structure

```text
├── model/
│   ├── network.py         # BaselineNet and SelfPruningNet topologies
│   └── prunable_layer.py  # Foundational continuous gating mechanism
├── utils/
│   ├── metrics.py         # State tracking and computation logic
│   └── sparsity.py        # Effective parameter reduction diagnostics
├── experiments/
│   ├── plots.py           # Seaborn visualization scripts
│   └── (results)          # Dynamic output tables and checkpoints
└── train.py               # Configurable optimization pipeline
```

---

## ⚡ Empirical Pipeline Results

Trained executing a full capacity configuration bounds on **CIFAR-10**, the model evaluated the parameter efficiency vs accuracy Pareto frontier:

| Architecture Variant | Sparsity Penalty ($\lambda$) | Validation Accuracy | Dead Parameters (< 0.05) | Effective Parameter Reduction |
|----------------------|------------------------------|---------------------|--------------------------|-------------------------------|
| Baseline Component   | - (Control)                  | ~71.5%              | -                        | -                             |
| Sparse Execution     | $0.001$                        | 71.5%               | 0.00%                    | 1.00x                         |
| Sparse Execution     | $0.1$                          | 70.8%               | 6.20%                    | 1.06x                         |
| Sparse Execution     | $1.0$                          | **70.3%**           | **49.46%**               | **~1.98x**                    |

> **Key Takeaway:** The subgradient effectively severes ~50% of the active dense parameter matrix symmetrically across the graph with a negligible 1.2% classification accuracy deviation.

---

## 🛠️ Execution & Setup

The framework is strictly deterministic with parameterized arguments allowing robust configuration. 

### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy pandas matplotlib seaborn tqdm
```

### 2. Run Optimization Pipeline

To iterate over the predefined sparse boundaries ($0.001, 0.1, 1.0$) alongside the Dense Baseline control graph:

```bash
python train.py --epochs 25 --batch-size 256 --lr 0.005 --lambdas 0.001 0.1 1.0
```

*Results instantly pipe out to continuous CLI progress bars and flush into `experiments/results.csv` on termination.*

### 3. Generate Analytical Visualizations

To render the parameter frequency and subgradient saturation boundary:

```bash
python experiments/plots.py
```
*Outputs rigorous distributions specifically optimized to `experiments/gate_distribution.png`.*