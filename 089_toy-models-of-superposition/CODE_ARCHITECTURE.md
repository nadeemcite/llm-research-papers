---
title: "Toy Superposition: Code architecture"
parent: "089 Toy Superposition"
nav_order: 1
---
# Code Architecture -- Toy Superposition Notebook

## Overview

The notebook implements the toy model $x' = \text{ReLU}(W^T W x + b)$ from Elhage et al. (2022) in PyTorch, trains it on synthetic sparse features, sweeps sparsity to demonstrate the phase change from PCA-like behavior to superposition, visualizes geometric structures (triangle, pentagon), and measures adversarial vulnerability. On Kaggle GPU (T4), training completes in about 8 seconds for a single model and the full sweep in about 14 seconds. The final result shows 2 of 5 features represented at S=0.0, all 5 at S=0.95–0.99, and a near-pentagonal geometry at S=0.99.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `SEED = 42` — global random seed for reproducibility
- `DEVICE` — CUDA if available, else CPU (Kaggle run used cuda)
- Imports: torch, numpy, matplotlib, torch.nn, torch.optim

### 2. Synthetic Sparse Features

**Function: `generate_sparse_inputs(n_features, batch_size, sparsity, seed)`** — generates a `(batch_size, n_features)` tensor where each entry is 0 with probability `sparsity` and uniform$[0,1]$ otherwise. The `seed` parameter allows deterministic test data.

### 3. Toy Model

**Class: `ToyModel`** (12 parameters for 5 features, 2 hidden)
- `W`: $n \times m$ parameter matrix (feature embedding directions)
- `b`: $n$-dimensional bias vector
- `forward(x)`: computes $h = xW$ (bottleneck), then $\text{ReLU}(hW^T + b)$ (reconstruction)
- `get_feature_directions()`: returns rows of W as feature direction vectors in hidden space

**Data flow:**
```
Input: (batch, n_features)
  -> x @ W -> (batch, n_hidden)        [bottleneck: compress to m dims]
  -> h @ W^T -> (batch, n_features)    [expand back to n dims]
  -> ReLU(out + b) -> (batch, n_features)  [nonlinear filtering]
```

### 4. Importance-Weighted MSE Loss

**Function: `importance_weighted_loss(model, x, importances)`** — computes $\frac{1}{n}\sum_i I_i (x_i - x_i')^2$ where $I_i = 0.8^i$.

**Function: `train_model(n_features, n_hidden, sparsity, n_steps, lr, batch_size)`** — trains a ToyModel for `n_steps` gradient steps (each step draws a fresh random batch), returns model and loss history.

### 5. Training a Single Model

- **Configuration:** `N_FEATURES=5`, `N_HIDDEN=2`, `SPARSITY=0.99`, `N_STEPS=1000`, `LR=0.01`
- **Optimizer:** Adam, lr=0.01
- Training time on Kaggle T4: 8.3s
- Initial loss: 0.002085, Final loss: 0.000104

### 6. Feature Directions Visualization

Two subplots: (1) training loss on log scale, (2) 2D arrow plot of feature directions $W_i$ with unit circle reference. Prints feature norms and cosine similarity matrix to quantify interference.

### 7. Phase Change Sweep

Trains 10 separate models at sparsity values `[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.999]` with `N_STEPS_SWEEP=800` each. Sweep time on Kaggle T4: 14.4s.

Results from the Kaggle run:
- S=0.000: 2 features represented (norms > 0.1)
- S=0.300: 3 features
- S=0.700: 4 features
- S=0.950: 5 features (all in superposition)
- S=0.990: 5 features
- S=0.999: 3 features (least important collapse — artifact of finite training)

### 8. Phase Change Plot

Plots feature norms vs sparsity. The phase change is visible as features 2–4 go from near-zero norm to nonzero norm as sparsity increases.

### 9. Geometric Structure

Trains 4 models with different feature counts (3, 4, 5, 7) in 2D at S=0.99, each for 1000 steps. Visualizes feature directions to show geometric arrangements.

Final losses from the Kaggle run:
- Triangle (3 features): 0.000000
- 4 features: 0.000017
- Pentagon (5 features): 0.000104
- 7 features: 0.000452

### 10. Adversarial Vulnerability

**Function: `compute_adversarial_vulnerability(model, n_features, sparsity, attack_norm)`** — computes the analytic optimal L2 attack along $(W W^T)_i$ directions (following the paper). Measures max loss increase from a perturbation of norm 0.1.

Results from the Kaggle run:
- S=0.00: max loss increase 0.001996
- S=0.50: 0.001507
- S=0.70: 0.001587
- S=0.90: 0.001313
- S=0.95: 0.000470
- S=0.99: 0.000115

### 11. Features per Dimension

Prints a table of features represented, features per dimension, and superposition state across the sparsity sweep.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_sparse_inputs` | Generate synthetic sparse feature vectors for training |
| `ToyModel` | The ReLU bottleneck model: $x' = \text{ReLU}(W^TWx + b)$ |
| `importance_weighted_loss` | Importance-weighted MSE loss |
| `train_model` | Train a ToyModel for n_steps gradient steps |
| `compute_adversarial_vulnerability` | Measure vulnerability to analytic L2 adversarial attacks |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Feature count | Up to 80 features, 20 hidden | 5–7 features, 2 hidden |
| Model variants | ReLU output, ReLU hidden layer, abs(x) computation | ReLU output only |
| Importance schedule | $0.8^i$, $0.9^i$, uniform | $0.8^i$ only |
| Training | Custom optimization, 1000 models per data point | Single seed, 800–1000 steps |
| Adversarial attacks | Gradient + analytic, multiple norms | Analytic only, single norm |
| Privileged basis | Full analysis of monosemantic vs polysemantic neurons | Not included |
