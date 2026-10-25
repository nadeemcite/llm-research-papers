---
title: "Mamba-2: Code architecture"
parent: "018 Mamba-2"
nav_order: 1
---
# Code Architecture — Mamba-2 Notebook

## Overview

The notebook implements the core **Structured State Space Duality (SSD)** layer from Listing 1 of the Mamba-2 paper in pure PyTorch, using only `einsum` operations. It implements both the quadratic (attention-like) form and the block-decomposed linear/recurrent form, verifies they produce identical outputs, and compares the SSD layer against standard linear attention on a synthetic multi-query associative recall task. The result is measured by classification accuracy on a held-out validation set.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `SEQ_LEN = 256` — sequence length (paper: up to 16K for speed benchmarks)
- `D_MODEL = 64` — model dimension (paper: 2560 for 2.7B)
- `D_STATE = 64` — state expansion N (paper: 64–256)
- `D_HEAD = 16` — head dimension P (paper: 64–128)
- `N_HEADS = 4` — number of heads (paper: D_model / P, grows with model size)
- `CHUNK_SIZE = 64` — block size Q for the chunked SSD algorithm (paper: 64–128)
- `EPOCHS = 15`, `LR = 1e-3`, `BATCH_SIZE = 64`

### 2. Data — Multi-Query Associative Recall (MQAR)

**Function: `generate_mqar_batch`** — generates (batch, seq_len) integer tensors. Random key-value pairs (from a 16-token vocab) are scattered through the sequence, then query tokens appear that must be completed with the matching value. Returns x (batch, seq_len) and y (batch, seq_len) where y is the shifted correct completions.

This task tests the model's ability to memorize multiple key-value associations in its recurrent state and retrieve them on demand — exactly the capability the paper highlights in Section 9.1.

### 3. SSD — Quadratic Form (Attention-like)

**Function: `segsum`** — computes the segment sum (cumulative sum with causal masking) used for the 1-semiseparable mask L. Returns a lower-triangular matrix where L[i,j] = sum(a[j+1:i]) for i > j, and -inf above the diagonal.

**Function: `ssd_quadratic(X, A, B, C)`** — computes the SSD output using the quadratic (attention-like) form from Eq. 16:
- Compute the 1-SS mask L = exp(segsum(A))
- Compute the Gram matrix G = C @ B^T (contract over state dim N)
- The SSD matrix M = L * G (element-wise / Hadamard product)
- Output Y = M @ X

This is O(T²N) and demonstrates that the SSD layer is a form of masked attention where the mask is input-dependent (semiseparable) rather than a fixed causal mask.

### 4. SSD — Block-Decomposed Form (Listing 1)

**Function: `ssd_chunked(X, A, B, C, block_len)`** — implements the four-step block-decomposed SSD algorithm from Section 6 and Listing 1:

1. **Diagonal blocks (intra-chunk):** For each chunk of length Q, compute the output using the quadratic form within the chunk. This uses matmuls and runs in parallel across chunks.
2. **Right factors (B terms):** Compute per-chunk states by accumulating B * decay * X within each chunk.
3. **Center factors (A recurrence):** Propagate states between chunk boundaries using a smaller recurrence (the inter-chunk SSM scan). This is the only sequential part.
4. **Left factors (C terms):** Convert propagated states back to outputs using C and within-chunk decay.

This is O(TN²) overall with work dominated by matrix multiplications, matching Theorem 6.1.

### 5. Duality Verification

Tests that the quadratic and chunked forms produce identical outputs on random inputs. Computes the max absolute difference between the two forms. The result should be < 1e-5, confirming the theoretical duality.

### 6. SSD Layer (Learnable)

**Class: `SSDLayer`** (about 30K parameters per layer)
- Projects input to X (values), B (expansion), C (contraction), and computes scalar A (decay) via a learned projection + softplus
- A is initialized to negative values (stable decay)
- `forward(x)`: uses `ssd_chunked` for training (efficient), produces Y of same shape as input

**Class: `Mamba2Block`** — residual block: LayerNorm → SSDLayer → GELU → Dropout → residual add

**Class: `Mamba2Model`** — embedding + 2 × Mamba2Block + output head

**Data flow:**
```
Input: (batch, 256) int tokens
  -> embed -> (batch, 256, 64)
  -> Mamba2Block x2 -> (batch, 256, 64)
  -> Linear -> (batch, 256, 16) logits per position
```

### 7. Linear Attention Baseline

**Class: `LinearAttentionLayer`** — standard causal linear attention with a fixed cumsum mask (no input-dependent decay). Uses the same Q/K/V projections but with L = lower-triangular ones (standard causal mask). This is the special case of SSD where A_t = 0 (no decay/forgotting).

### 8. Training

**Function: `train_model`** — AdamW optimizer (lr=1e-3, weight_decay=0.01), cross-entropy loss on all positions, gradient clipping at 1.0. Logs train loss and val accuracy every 3 epochs. Both models trained identically for 15 epochs.

### 9. Training Curves

Two subplots: validation accuracy and training loss over epochs for SSD (Mamba-2) and Linear Attention.

### 10. Final Comparison

Prints a table with model name, validation accuracy, and parameter count. Bar chart comparing final validation accuracy.

### 11. Duality Check Visualization

Bar chart showing the max absolute difference between quadratic and chunked SSD forms, confirming numerical equivalence.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `segsum(x)` | Compute segment sum for 1-semiseparable mask (cumsum with causal masking) |
| `ssd_quadratic(X, A, B, C)` | SSD via the quadratic (attention-like) form — O(T²N) |
| `ssd_chunked(X, A, B, C, block_len)` | SSD via block decomposition (Listing 1) — O(TN²), matmul-dominated |
| `SSDLayer` | Learnable SSD layer with input-dependent A, B, C projections |
| `Mamba2Block` | Residual block wrapping SSDLayer with LayerNorm and GELU |
| `Mamba2Model` | Full model: embedding + SSD blocks + output head |
| `LinearAttentionLayer` | Standard causal linear attention baseline (SSD with A=0) |
| `generate_mqar_batch` | Generate synthetic multi-query associative recall data |
| `train_model` | Training loop with AdamW, cross-entropy, gradient clipping |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 48+ layers, 2560 dim, 2.7B params | 2 layers, 64 dim, ~100K params |
| SSD computation | Custom CUDA/Triton kernel | Pure PyTorch einsum (Listing 1) |
| Dataset | Pile (300B tokens), MQAR | Synthetic MQAR, 4000 examples, L=256 |
| State size N | 64–256 | 64 |
| Training | Chinchilla-optimal, cosine decay | 15 epochs, constant lr=1e-3 |
| Block design | Parallel projections + extra norm | Sequential projections, no extra norm |
| Head pattern | Multi-value attention (MVA) | Standard multi-head (MHA-like) |
| Systems | Tensor parallel, sequence parallel | Single GPU, no parallelism |
