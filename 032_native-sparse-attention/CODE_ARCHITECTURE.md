---
title: "NSA: Code architecture"
parent: "032 NSA"
nav_order: 1
---
# Code Architecture — NSA Notebook

## Overview

The notebook trains two 4-layer transformers on character-level Wikitext-2: one with standard Full Attention and one with Native Sparse Attention (NSA). The NSA model (~1,238,924 params) achieves validation perplexity of 1.02, while the Full Attention model (~794,880 params) reaches 4.01. The perplexity gap is largely explained by NSA having 55.9% more parameters (three independent K/V projection sets, a compression MLP, and a gate MLP).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `EPOCHS = 15`, `LR = 1e-3`, `BATCH_SIZE = 64`
- `SEQ_LEN = 256`, `N_LAYERS = 4`, `D_MODEL = 128`, `N_HEADS = 4`, `N_GROUPS = 2`
- `COMPRESS_BLOCK = 16` (l), `COMPRESS_STRIDE = 8` (d), `SELECT_BLOCK = 16` (l'), `TOP_N_BLOCKS = 8` (n), `WINDOW_SIZE = 64` (w)
- `DEVICE = cuda` (on Kaggle) / `cpu` (local fallback)

### 2. Data

Downloads character-level Wikitext-2 (~2M train chars, ~216K val chars, vocab ~100 chars) from the PyTorch examples repository. Uses `get_batch(data, batch_size, seq_len, device)` to sample random windows, returning (x, y) tensors of shape (batch, seq_len).

### 3. Model

**Class: `FullAttention`** (~part of 794,880-param model)
- Standard causal GQA: Q/K/V projections, `scaled_dot_product_attention` with `is_causal=True`, output projection.
- K/V projected to `N_GROUPS * d_head` (2 × 32 = 64 dims), repeated across heads.

**Class: `NSAAttention`** (~part of 1,238,924-param model)
- Shared Q projection, independent K/V projections for each of 3 branches (6 Linear layers vs 2 in FullAttention).
- **Compression MLP:** `nn.Sequential(Linear(l*d_head, d_model), GELU(), Linear(d_model, d_head))` — maps a block of l keys to one compressed key. Values are mean-pooled.
- **Gate MLP:** `nn.Sequential(Linear(d_model, d_model//2), GELU(), Linear(d_model//2, 3))` — produces 3 gate values via sigmoid.
- **Output projection:** same as FullAttention.

**Class: `TinyGPT`** — token + positional embeddings, N_LAYERS TransformerBlocks (LayerNorm → attention → residual → LayerNorm → GELU FFN → residual), final LayerNorm, weight-tied head.

**Data flow (NSA forward):**
```
Input: (B, T) int64 token IDs
  -> embeddings -> (B, T, 128)
  -> Q projection -> (B, H, T, D)
  -> Branch 1 (Compression):
       K,V -> (B, G, T, D) -> unfold to blocks (B, G, n_cmp, l, D)
       -> compress MLP -> (B, G, n_cmp, D) -> repeat to H
       -> causal attention -> (B, H, T, D)
  -> Branch 2 (Selection):
       importance from compression scores -> top-n block indices
       -> block-sparse mask (B, G, T, T) -> repeat to H
       -> masked attention on full K/V -> (B, H, T, D)
  -> Branch 3 (Window):
       window mask (|i-j| < w, causal)
       -> masked attention on full K/V -> (B, H, T, D)
  -> Gate: sigmoid(MLP(x)) -> (B, 3, T, 1)
  -> weighted sum of 3 branches -> (B, H, T, D)
  -> reshape -> (B, T, 128)
  -> output projection -> (B, T, 128)
  -> 4x TransformerBlock -> (B, T, 128)
  -> head -> (B, T, vocab_size)
```

### 4. Loss Function

Standard cross-entropy loss on next-token prediction. Both models use AdamW (lr 1e-3).

### 5. Training Loop

- 15 epochs, 200 batches per epoch, batch_size 64, seq_len 256
- Validation every epoch (50 batches), logs train/val loss and val perplexity
- Gradient clipping (max norm 1.0)
- Both models trained sequentially with identical hyperparameters

### 6. Training Curves

Three-panel figure: training loss, validation loss, and validation perplexity for both models.

### 7. Final Evaluation

Evaluates both models on 100 validation batches. Prints final val loss, val perplexity, perplexity gap, and parameter overhead.

### 8. Attention Sparsity Analysis

- Computes the theoretical sparsity ratio: (n_cmp + n*lp + w) / T
- Visualizes compression branch attention weights as a heatmap (query position × compressed block index)
- Visualizes gate weights as a stacked area chart across sequence positions
- Prints average gate weights for each branch

## Key Functions

| Function / Class | Purpose |
|---|---|
| `FullAttention` | Standard causal GQA attention |
| `NSAAttention` | Three-branch sparse attention with compression, selection, and window |
| `NSAAttention._compress` | Block compression via MLP for keys, mean-pool for values |
| `NSAAttention._select_topn` | Top-n block selection via importance scores from compression |
| `TransformerBlock` | LayerNorm + attention + FFN with residual connections |
| `TinyGPT` | Full GPT model with embedding, blocks, and weight-tied head |
| `get_batch` | Sample random training windows from tokenised text |
| `train_model` | Training loop with logging |
| `eval_perplexity` | Evaluate validation perplexity |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | 27B MoE (3B active), 30 layers, d=2560 | TinyGPT: 4 layers, d_model 128, ~795K–1.24M params |
| Attention | GQA (4 groups, 64 heads) | GQA (2 groups, 4 heads) |
| Dataset | 270B tokens of 8k texts, then 32k with YaRN | Character-level Wikitext-2 (~2M chars) |
| NSA params | l=32, d=16, l'=64, n=16, w=512 | l=16, d=8, l'=16, n=8, w=64 |
| Kernels | Custom Triton sparse attention kernels | Pure PyTorch (masked softmax) |
| Speedup measurement | 11.6× decoding, 9.0× forward, 6.0× backward on A100 | Not measured (simulation overhead at small scale) |
| Benchmarks | MMLU, LongBench, AIME 24, needle-in-a-haystack | Validation perplexity only |
| Training | Large-scale pretraining to convergence | 15 epochs, AdamW lr 1e-3 |
| K/V per branch | Independent projections (6 total) | Same (6 total), causing 55.9% param overhead |
| Selection | Group-consistent top-n via summed scores (Eq. 10) | Same, plus nan_to_num for early positions |
| Compression | Learnable MLP with intra-block position encoding | Learnable MLP without position encoding |
| Values | Learnable compression | Mean-pool (simplification) |
