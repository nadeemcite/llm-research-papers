---
title: "Superposition Scaling: Code architecture"
parent: "037 Superposition Scaling"
nav_order: 1
---
# Code Architecture — Superposition Scaling Notebook

## Overview

The notebook implements Anthropic's toy model of superposition as a single-layer autoencoder, trains 30 models (2 data exponents × 3 weight decays × 5 model widths), and fits the loss-vs-width power law to extract the model exponent α_m. The result reproduces the paper's finding: strong superposition (wd=-1) yields α_m ≈ 0.98–1.15 (mean 0.995, close to 1), while weak superposition (wd=+1) yields α_m ≈ 0.13–0.63 (mean 0.382, much smaller and data-dependent).

## Section-by-Section Breakdown

### 1. Setup & Imports

Sets random seeds (SEED=42) and defines the key constants:
- `N_FEATURES = 1000` — number of features (data dimension n; paper uses 10240 for large models, 1000 for small)
- `M_VALUES = [10, 20, 40, 60, 100]` — model widths to sweep (paper sweeps up to 1024)
- `N_STEPS = 5000` — training steps per model (paper uses 20000 for small, 40000 for large)
- `BATCH_SIZE = 2048` — batch size (same as paper)
- `LR = 1e-2` — base learning rate (paper uses 1e-2)
- `WARMUP_STEPS = 500` — warmup for cosine schedule
- `WEIGHT_DECAYS = [-1.0, 0.0, 1.0]` — from strong to weak superposition (paper uses 10 values)
- `ALPHAS = [0.0, 1.0]` — data exponents: flat and Zipf (paper sweeps 9 values from 0 to 1)

### 2. Data

**Function: `make_probabilities(n, alpha)`** — returns a tensor of feature frequencies p_i ∝ 1/i^{1+α}, normalized to sum=1. Shape: (n,).

**Function: `sample_batch(batch_size, n, prob, device)`** — samples sparse feature vectors. x_i = u_i · v_i where u ~ Bernoulli(p), v ~ U(0,2). Shape: (batch_size, n).

Data is generated on-the-fly at each training step (i.i.d.), matching the paper's approach. No pre-generated dataset is stored.

### 3. Model

**Class: `FeatureRecovery`** (parameter count: n×m + n)
- `W`: nn.Parameter of shape (n, m) — weight matrix; row i is the representation of feature i
- `b`: nn.Parameter of shape (n,) — bias vector
- Forward: y = ReLU(x @ W @ W^T + b), input (batch, n) → output (batch, n)

This is identical to the paper's toy model (Equation in Section 2, based on Elhage et al., 2022). The key feature is that W @ W^T computes the Gram matrix of feature representations, so the reconstruction of feature i depends on how much feature i's representation overlaps with all active features.

**Data flow:**
```
Input: (batch, n) = (2048, 1000)
  -> x @ W: (batch, m) = (2048, m)   [encode to hidden space]
  -> (x @ W) @ W^T: (batch, n)       [decode back]
  -> + b: (batch, n)                  [add bias]
  -> ReLU: (batch, n)                 [nonlinearity for error correction]
Output: (batch, n) = (2048, 1000)
```

### 4. Custom AdamW Optimizer

**Class: `CustomAdamW`** — AdamW with decoupled weight decay/growth applied per-row.
- For γ ≥ 0: p ← p × (1 - γ·lr) — standard weight decay, shrinks all weights
- For γ < 0: p ← p + lr·γ·p·(1/\|\|p_i\|\| - 1) — weight growth toward unit row norm, encouraging superposition

This implements Equation 2 from the paper. The weight decay/growth is applied to W only (b has weight_decay=0).

### 5. Learning Rate Schedule

**Function: `get_lr(step, lr, n_steps, warmup_steps)`** — linear warmup for 500 steps, then cosine decay to 5% of base lr. Matches the paper's Appendix B.2 settings.

### 6. Training Loop

**Function: `train_one_model(n, m, alpha, weight_decay, n_steps, batch_size, lr, device)`**
- Creates model, optimizer, and probability vector
- For each step: sample fresh batch, update LR, forward, MSE loss, backward, step
- After training: evaluate on 4×batch fresh data, compute φ_{1/2} (fraction of rows with norm > 0.5)
- Returns: final_loss, phi, losses (full curve), W matrix

### 7. Run Experiments

Sweeps all 60 combinations and prints progress. Each model takes ~30s on GPU, ~2-3 min on CPU.

### 8. Scaling Analysis

**Function: `fit_power_law(ms, losses)`** — fits L = C/m^{α_m} via log-linear regression (np.polyfit on log-log). Returns α_m and C.

For each (alpha, wd) pair, fits the 6 width points and classifies the regime: strong superposition if φ(m=100) > 0.3, weak otherwise.

### 9–13. Plots

- **Loss vs width** (log-log): scaling curves with fitted power-law lines for each (alpha, wd)
- **φ vs weight decay**: reproduces Figure 3b, showing weight decay controls superposition
- **Row norms**: reproduces Figure 3a, showing bimodal (weak) vs near-uniform (strong) norms
- **α_m bar chart**: visual comparison of exponents across regimes
- **Training curves**: loss over steps for m=100, showing convergence differences

## Key Functions

| Function / Class | Purpose |
|---|---|
| `make_probabilities` | Generate feature frequency vector p_i ∝ 1/i^{1+α} |
| `sample_batch` | Sample i.i.d. sparse feature vectors x_i = Bernoulli(p_i) · U(0,2) |
| `FeatureRecovery` | Toy autoencoder model: ReLU(W W^T x + b) |
| `CustomAdamW` | AdamW with decoupled weight decay (γ≥0) or growth (γ<0) |
| `get_lr` | Cosine LR schedule with linear warmup |
| `train_one_model` | Train one model, return loss, φ, loss curve, and W |
| `fit_power_law` | Fit L ∝ 1/m^{α_m} via log-linear regression |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Data dimension n | 10240 (large) / 1000 (small) | 1000 (small setting) |
| Model widths m | up to 1024 (large) / 10–100 (small) | [10, 20, 40, 60, 100] |
| Training steps | 40000 (large) / 20000 (small) | 5000 (reduced for Kaggle timeout) |
| Weight decay values | 10 (linspace -1 to +1) | 3 ([-1.0, 0.0, 1.0]) |
| Data exponents α | 9 (linspace 0 to 1) | 2 (0.0 and 1.0) |
| LLM analysis | OPT, Pythia, Qwen2.5 evaluation | Not included (toy model only) |
| Loss function | MSE (toy) / cross-entropy (LLM) | MSE (toy only) |
| Feature distributions | power, exponential, linear, exponential1 | power-law only (varying α) |
