---
title: "S4: Code architecture"
parent: "016 S4"
nav_order: 1
---
# Code Architecture — S4 Notebook

## Overview

The notebook implements a **diagonal State Space Model (SSM)** layer with HiPPO-LegS initialization in PyTorch, and compares it against a GRU and a Transformer on a synthetic long-range retrieval task at sequence length 512. The diagonal SSM is a simplification of S4's NPLR parameterization (as used in S4D/S5). The result is measured by classification accuracy on a held-out validation set.

## Section-by-Section Breakdown

### 1. Imports and Configuration

Sets random seeds and defines the key constants:
- `SEQ_LEN = 512` — sequence length (paper: up to 16,384 for Path-X)
- `D_MODEL = 64` — hidden dimension (paper: 128-512)
- `N_STATE = 64` — SSM state size N (paper: N=64 for LRA, same)
- `N_LAYERS = 2` — number of layers (paper: 4-6)
- `EPOCHS = 10`, `LR = 1e-3`, `BATCH_SIZE = 32`

### 2. Data — Long-Range Retrieval Task

**Function: `generate_batch`** — generates (batch, seq_len) integer tensors. Position 0 is a query digit (1-9), the same digit appears once in the second half, and the target is the query digit. Returns x (batch, seq_len) and y (batch,).

### 3. HiPPO Matrix Initialization

**Function: `make_hippo_legs(N)`** — returns the N x N HiPPO-LegS matrix: A[n,k] = -sqrt(2n+1) if n=k, sqrt(2k+1) if n > k, 0 if n < k. Visualized as a heatmap.

### 4. SSM Discretization

**Function: `discretize_bilinear(A, B, delta)`** — bilinear (Tustin) discretization: A_bar = (I - Delta/2 A)^{-1}(I + Delta/2 A), B_bar = (I - Delta/2 A)^{-1} Delta B.

**Function: `compute_conv_kernel(A_bar, B_bar, C, L)`** — computes the convolution kernel K = [CB, CAB, ..., CA^{L-1}B] by iteratively applying A_bar.

### 5. Diagonal SSM Layer

**Class: `DiagSSMLayer`** (about 50K parameters total across the model)
- Initializes A from HiPPO-LegS eigenvalues (complex), stored as log(real) + imag
- Learns B, C (complex, stored as real+imag parts), delta (log-space), D (skip connection)
- `get_A_bar()`: bilinear discretization of diagonal eigenvalues
- `get_B_bar(delta)`: discretized B
- `compute_kernel(L)`: K[i] = sum_n C[n] * A_bar[n]^i * B_bar[n] (element-wise, diagonal)
- `forward(x)`: projects input, computes kernel, applies FFT-based depthwise convolution, adds skip, GELU

**Class: `S4Block`** — residual block: LayerNorm -> DiagSSMLayer -> Dropout -> residual add

**Data flow:**
```
Input: (batch, 512) int tokens
  -> embed -> (batch, 1024, 64)
  -> S4Block x2 -> (batch, 1024, 64)
  -> mean pool -> (batch, 64)
  -> Linear -> (batch, 10) logits
```

### 6. Baseline Models

**Class: `RNNModel`** — GRU with 2 layers, same hidden dim. Embedding -> GRU -> mean pool -> Linear head.

**Class: `TransformerModel`** — 2-layer Transformer encoder with 4 heads, positional embeddings, GELU activation. Embedding -> Transformer blocks -> mean pool -> Linear head.

### 7. Training

**Function: `train_model`** — AdamW optimizer (lr=1e-3, weight_decay=0.01), cross-entropy loss, gradient clipping at 1.0. Logs train loss, train acc, val acc every 3 epochs. All three models trained identically for 10 epochs.

### 8. Training Curves

Two subplots: validation accuracy and training loss over epochs for S4, GRU, and Transformer.

### 9. Final Comparison

Prints a table with model name, validation accuracy, and parameter count. Includes random baseline (1/10 = 0.1).

### 10. Bar Chart

Bar chart comparing final validation accuracy across the three models with a random baseline line.

### 11. Recurrent Mode Demo

**Function: `s4_recurrent_demo`** — processes one token at a time through the first SSM block, maintaining the 64-dim complex state. Demonstrates O(N) per-step inference (vs O(L^2) for attention).

## Key Functions

| Function / Class | Purpose |
|---|---|
| `make_hippo_legs(N)` | Create the HiPPO-LegS initialization matrix |
| `discretize_bilinear(A, B, delta)` | Bilinear (ZOH) discretization of continuous SSM |
| `compute_conv_kernel(A_bar, B_bar, C, L)` | Compute SSM convolution kernel K = [CB, CAB, ...] |
| `DiagSSMLayer` | Diagonal SSM layer with learnable eigenvalues, FFT-based convolution |
| `S4Block` | Residual block wrapping the SSM layer with LayerNorm and dropout |
| `S4Model` | Full model: embedding + S4 blocks + mean pool + classification head |
| `RNNModel` | GRU baseline with matching architecture |
| `TransformerModel` | Transformer baseline with matching architecture |
| `train_model` | Training loop with AdamW, cross-entropy, gradient clipping |
| `s4_recurrent_demo` | Demonstrate recurrent (streaming) inference mode |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| A parameterization | NPLR (Normal Plus Low-Rank) with Cauchy kernel | Diagonal (eigenvalues of HiPPO, as in S4D/S5) |
| Kernel computation | Cauchy kernel via FFT, O((N+L) log^2(N+L)) | Direct power iteration + FFT convolution |
| Dataset | LRA (6 tasks, 1K-16K steps), WikiText-103, sCIFAR-10 | Synthetic retrieval task, L=512, 2000 examples |
| Model size | 4-6 layers, 128-512 dim, millions of params | 2 layers, 64 dim, ~50K params |
| Training | 50-200 epochs, plateau LR scheduler, augmentation | 10 epochs, constant lr=1e-3, no augmentation |
| Normalization | BatchNorm or LayerNorm (ablated) | LayerNorm |
