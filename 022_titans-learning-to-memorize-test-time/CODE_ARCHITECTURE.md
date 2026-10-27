---
title: "Titans: Code architecture"
parent: "022 Titans"
nav_order: 1
---
# Code Architecture — Titans Notebook

## Overview

The notebook implements the Titans neural long-term memory (LMM) module from scratch in PyTorch, including the surprise-based gradient update with momentum and weight decay. It compares LMM against plain linear attention (matrix-valued memory without surprise/momentum) and full causal attention on a synthetic multi-query associative recall task. The result is measured by validation accuracy: LMM 0.97, linear attention 0.26, full attention 1.00.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `D_MODEL = 64` — hidden dimension of the memory module and attention heads
- `VOCAB_SIZE = 16` — number of distinct tokens in the synthetic task
- `SEQ_LEN = 64` — sequence length per training example
- `N_KV_PAIRS = 4` — number of key-value associations to recall
- `EPOCHS = 20`, `LR = 1e-3`, `BATCH_SIZE = 32`
- `N_TRAIN = 2000`, `N_VAL = 500`

### 2. Data — Synthetic Multi-Query Associative Recall

**Function: `generate_data`** — generates (input, target) pairs where the input sequence contains key-value pairs followed by queries. Returns tensors of shape (batch, seq_len) for inputs and (batch, n_queries) for targets. The task tests whether the model can store key-value associations and retrieve them when queried, which is the core capability the paper evaluates with NIAH and BABILong benchmarks.

### 3. Model — Neural Long-Term Memory (LMM)

**Class: `NeuralLongTermMemory`** (about 16K parameters)

The memory module is an MLP with configurable depth (1-2 layers). At each time step t:
1. Project input to key k_t = x_t W_K and value v_t = x_t W_V
2. Compute associative memory loss: ℓ = \|\|M_{t-1}(k_t) - v_t\|\|^2
3. Compute gradient ∇ℓ (the "surprise")
4. Update momentum: S_t = η_t S_{t-1} - θ_t ∇ℓ
5. Update memory: M_t = (1 - α_t) M_{t-1} + S_t
6. Retrieve: y_t = M_t(q_t) where q_t = x_t W_Q

The forgetting gate α_t, surprise decay η_t, and learning rate θ_t are computed from the input via small linear layers, making them data-dependent as in the paper.

**Class: `LinearAttentionBaseline`** — matrix-valued memory M_t = M_{t-1} + k_t v_t^T, y_t = q_t M_t. No surprise metric, no momentum, no forgetting gate. This is the linear Transformer recurrence (Eq. 4-5 in the paper).

**Class: `FullAttention`** — standard causal multi-head attention with the full sequence.

**Data flow (LMM):**
```
Input: (batch, seq_len, d_model)
  -> for each token t:
       k_t, v_t, q_t = projections(x_t)           -> (batch, d_model)
       loss = ||M(k_t) - v_t||^2                    -> scalar
       grad = autograd.grad(loss, M.parameters())   -> list of gradients
       S_t = η_t * S_{t-1} - θ_t * grad             -> momentum update
       M_t = (1 - α_t) * M_{t-1} + S_t              -> memory update
       y_t = M_t(q_t)                                -> (batch, d_model)
  -> Output: (batch, seq_len, d_model)
```

### 4. Loss Function

Cross-entropy loss on the recalled token predictions. The model receives a sequence with key-value pairs and queries, and must predict the correct value token for each query. Loss is computed only on query positions.

### 5. Training Loop

- AdamW optimizer, lr=1e-3, 20 epochs, batch size 32
- Three models trained in parallel: LMM, Linear Attention, Full Attention
- Logs validation accuracy every 5 epochs
- All three models use the same data, optimizer settings, and random seed for fair comparison

### 6. Results and Plots

- **Training loss curves:** Shows convergence of all three models over 20 epochs
- **Validation accuracy bar chart:** Final accuracy comparison — LMM 0.97, Linear Attn 0.26, Full Attn 1.00
- **Memory module weight evolution:** Shows how the LMM weights change during a single sequence (test-time learning), demonstrating the surprise mechanism in action

## Key Functions

| Function / Class | Purpose |
|---|---|
| `NeuralLongTermMemory` | The Titans neural memory module with surprise-based gradient updates, momentum, and weight decay |
| `LinearAttentionBaseline` | Plain linear attention (matrix-valued memory) without surprise or forgetting |
| `FullAttention` | Standard causal multi-head attention as the upper-bound baseline |
| `generate_data` | Creates synthetic multi-query associative recall sequences |
| `train_model` | Trains a model and returns loss history and validation accuracy |
| `plot_results` | Generates training curves and accuracy comparison plots |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Memory depth | 1-4 layer MLP, up to d_model=4096 | 1-2 layer MLP, d_model=64 |
| Dataset | FineWeb-Edu, 15-30B tokens | Synthetic recall task, 2000 examples |
| Parallel training | Chunkwise matmul + associative scan | Sequential loop for clarity |
| Architecture | MAC/MAG/MAL with SWA + persistent memory | LMM module only |
| Scale | 170M-760M parameters | ~16K parameters |
| Optimizer | AdamW, lr 4e-4 (paper), cosine annealing, batch 0.5M tokens | AdamW, lr 1e-3, batch 32 (simplified) |
