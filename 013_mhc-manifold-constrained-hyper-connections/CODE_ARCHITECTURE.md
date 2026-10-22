---
title: "Hyper-Connections: Code architecture"
parent: "013 Hyper-Connections"
nav_order: 1
---
# Code Architecture — Hyper-Connections (mHC) Notebook

## Overview

The notebook implements three transformer variants from scratch in PyTorch: a standard residual baseline, Hyper-Connections (HC), and Manifold-Constrained Hyper-Connections (mHC). All three are trained on Tiny Shakespeare (character-level, vocab 65) for 15 epochs. The key measurement is signal norm propagation with depth — how the L2 norm of the residual stream changes as it passes through each layer. mHC's doubly stochastic constraint keeps norms stable (max/min ratio 1.50), while HC's unconstrained mixing drifts more (ratio 8.86).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `DIM = 128` — hidden dimension C (paper uses much larger, e.g. 4096+)
- `N_LAYERS = 6` — number of transformer layers (paper uses 60+)
- `N_HEADS = 4` — attention heads
- `EXPANSION = 4` — expansion rate n for HC/mHC (same as paper)
- `SINKHORN_ITERS = 20` — Sinkhorn-Knopp iterations (same as paper, Section 4.2)
- `EPOCHS = 15`, `LR = 3e-4`, `BATCH_SIZE = 32`

### 2. Data

**Tiny Shakespeare** — character-level tokenization, ~1 MB of text, vocab size 65. Downloaded via URL, split into train (90%) and validation (10%). Sequences of 128 tokens are sampled.

### 3. Model

**Class: `HyperConnection`** — implements the HC/mHC residual connection pattern:
- `H_pre`: learnable projection C -> n, reads from the nC-dim stream into a C-dim layer input
- `H_post`: learnable projection C -> n, writes the layer output back to the stream
- `H_res`: learnable matrix n x n, mixes the n streams (in mHC, projected to doubly stochastic via Sinkhorn-Knopp)
- Dynamic + static coefficients: alpha * tanh(theta * RMSNorm(x)) + b, matching the paper's Eq. 5
- In mHC mode, `sinkhorn_knopp_batch()` projects H_res to the Birkhoff polytope before use

**Class: `TransformerBlock`** — standard pre-norm transformer block (RMSNorm + multi-head attention + RMSNorm + FFN with SwiGLU), with the residual connection delegated to a `HyperConnection` or standard skip.

**Class: `TinyTransformer`** — token embedding, N_LAYERS blocks, RMSNorm, output projection. About 1,001,350 params (baseline) / 1,019,938 params (HC/mHC).

**Data flow:**
```
Input: (batch, seq_len) 
  -> embedding -> (batch, seq_len, C)
  -> [HC: expand to (batch, seq, n*C), mix streams, read C, attend, write back, mix]
  -> [mHC: same but H_res projected to doubly stochastic]
  -> [baseline: standard x + F(x)]
  -> output projection -> (batch, seq_len, vocab)
```

### 4. Sinkhorn-Knopp Algorithm

**Function: `sinkhorn_knopp_batch(M, n_iters=20)`** — takes a batch of n x n matrices, applies exp() to make all elements positive, then iteratively normalizes rows and columns to sum to 1 (vectorized across the batch dimension). After 20 iterations, the result is approximately doubly stochastic. This is the core innovation of mHC (Section 4.2).

### 5. Loss Function

Standard next-token prediction: cross-entropy loss between predicted logits and true next token.

### 6. Training Loop

- AdamW optimizer, lr 3e-4, weight decay 0.1, betas (0.9, 0.95)
- 15 epochs, 50 steps per epoch, batch 32
- Logs training loss per epoch and validation perplexity
- All three models (baseline, HC, mHC) trained with identical settings

### 7. Signal Norm Analysis

After training, a forward pass is run through each model with a fixed input. At each layer, the L2 norm of the residual stream is recorded. Results:
- Baseline: norms range 2.6-11.1, max/min ratio 4.26
- HC: norms range 1.2-11.0, max/min ratio 8.86 (most drift)
- mHC: norms range 10.2-15.3, max/min ratio 1.50 (most stable)

### 8. Training Curves and Results

- Training loss curves for all three models plotted together
- Validation perplexity comparison: Baseline 9.60, HC 8.71, mHC 10.00
- Signal norm vs depth plot for all three models
- Doubly stochastic verification: mHC H_res row/col sums = 1.000000, spectral norm = 1.000000

## Key Functions

| Function / Class | Purpose |
|---|---|
| `sinkhorn_knopp_batch` | Vectorized projection of matrices onto the Birkhoff polytope (doubly stochastic) via iterative row/column normalization |
| `HyperConnection` | Implements the HC/mHC residual connection with learnable pre/post/res mappings |
| `TransformerBlock` | Pre-norm transformer block with configurable residual connection |
| `TinyTransformer` | Full transformer with embedding, layers, and output head |
| `RMSNorm` | Root mean square normalization (used in the paper's HC formulation) |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 27B MoE (DeepSeek-V3 inspired) | ~1.0M params, 6 layers, 128 dim |
| Dataset | Large-scale pretraining (trillions of tokens) | Tiny Shakespeare (~1 MB, char-level) |
| Architecture | MoE with MLA, DualPipe, pipeline parallelism | Dense transformer with standard MHA |
| Infrastructure | Fused TileLang kernels, recomputation, communication overlap | Pure PyTorch, no kernel fusion |
| Expansion rate n | 4 | 4 (same) |
| Sinkhorn iterations | 20 | 20 (same) |
| Training | Full-scale pretraining | 15 epochs, 50 steps/epoch, lr 3e-4 |
