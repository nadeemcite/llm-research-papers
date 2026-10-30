---
title: "Scaling Laws: Code architecture"
parent: "034 Scaling Laws"
nav_order: 1
---
# Code Architecture — Scaling Laws Notebook

## Overview

The notebook trains 6 decoder-only Transformers of increasing size (25,472 to 793,344 non-embedding parameters) on character-level WikiText-2, then fits the power law *L*(*N*) = (*N*\_c/*N*)^α\_N to the final validation losses. The fit achieves R² = 0.9944 with α\_N = 0.1367. It also varies dataset size *D* for the smallest and largest models to reproduce the overfitting relationship *L*(*N*,*D*) (Eq. 1.5), and fits the compute scaling *L*(*C*) using the estimate *C* ≈ 6*NBS*.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `SEQ_LEN = 128` — context length in characters (paper: 1024 tokens)
- `BATCH_SIZE = 64` — sequences per batch (paper: 512)
- `STEPS = 2000` — training steps per model (paper: 250,000)
- `LR = 3e-3` — peak learning rate with cosine decay
- `WARMUP_STEPS = 200` — linear warmup (paper: 3000)
- `DEVICE` — CUDA if available, else CPU

### 2. Data

Downloads WikiText-2 raw text files from GitHub to `/tmp/wikitext2/`, builds a character-level vocabulary (283 characters), and encodes train and validation text as integer tensors.

**Shapes:**
- `train_data`: `(10,780,437,)` — 1D tensor of character IDs
- `valid_data`: `(1,120,192,)` — 1D tensor of character IDs
- `get_batch` returns `x: (64, 128)`, `y: (64, 128)` — input and next-token targets

### 3. Model

**Class: `TinyTransformer`** — decoder-only Transformer using `nn.TransformerEncoderLayer` with `norm_first=True` and a causal mask.

Architecture per config: token embedding + positional embedding → *n*\_layers Transformer blocks (multi-head attention + GELU feed-forward) → LayerNorm → linear head (weight-tied to token embedding).

Parameter count (non-embedding): *N* ≈ 12 × *n*\_layers × *d*\_model² (matching the paper's Table 1 convention).

6 configurations:
| d_model | layers | heads | d_ff | N (actual) |
|---------|--------|-------|------|------------|
| 32 | 2 | 4 | 128 | 25,472 |
| 48 | 2 | 4 | 192 | 56,640 |
| 64 | 3 | 4 | 256 | 150,080 |
| 80 | 3 | 4 | 320 | 233,680 |
| 96 | 4 | 4 | 384 | 447,552 |
| 128 | 4 | 4 | 512 | 793,344 |

**Data flow:**
```
Input: (64, 128) integer indices
  -> token_emb + pos_emb -> (64, 128, d_model)
  -> n_layers × [causal self-attn + FFN] -> (64, 128, d_model)
  -> LayerNorm -> Linear -> (64, 128, vocab_size)
  -> cross_entropy with targets -> scalar loss
```

### 4. Loss

Cross-entropy loss averaged over all positions in the context, matching the paper's *L* (loss in nats). The `compute_loss` function reshapes logits to `(B*T, vocab)` and targets to `(B*T,)`.

### 5. Training Loop

- **Optimizer:** Adam, peak LR 3e-3
- **Schedule:** 200-step linear warmup → cosine decay to 0 (paper: 3000-step warmup → cosine decay)
- **Steps:** 2000 per model (paper: 250,000)
- **Gradient clipping:** max norm 1.0
- **Logging:** validation loss every 500 steps on 30 random batches; final evaluation on 50 batches
- `train_model(config, steps, data_limit)` returns `(n_params, final_val_loss, val_history)`

### 6. Experiment 1 — L(N) Scaling

Trains all 6 configs on the full training data, records final validation loss for each. Fits *L*(*N*) = (*N*\_c/*N*)^α\_N via linear regression in log-log space (`np.polyfit`). Reports α\_N = 0.1367, *N*\_c = 8.52×10⁶, and R² = 0.9944. The paper's values (α\_N ≈ 0.076, *N*\_c ≈ 8.8×10¹³) are printed for comparison; our values differ because of the ~10⁶× scale gap and different tokenisation.

Final losses: 2.2256 → 1.9621 → 1.7686 → 1.6158 → 1.4785 → 1.3976 (perplexity 9.26 → 4.05).

### 7. Fitting the Power Law

Extracts *N* and *L* arrays, performs `np.polyfit(log(N), log(L), 1)`, and computes:
- α\_N = −slope = 0.1367
- *N*\_c = exp(intercept / α\_N) = 8.52×10⁶
- R² = 0.9944 from residuals
- Per-model prediction error: under 2% for all 6 models

### 8. Scaling Law Plots

Two subplots: (left) loss vs *N* on log-log axes with the power-law fit overlaid; (right) validation loss curves during training for all 6 models.

### 9. Experiment 2 — L(N,D) Overfitting

Trains the smallest (CONFIGS[0], 25K params) and largest (CONFIGS[-1], 793K params) models on 5 dataset sizes: 20K, 50K, 100K, 200K, and full (10.8M) characters.

Key results:
- At D = 20K: large model val loss = 8.87 (train loss 0.07 — severe overfitting), small model val loss = 2.46. Overfitting gap = 6.41.
- At D = full: large model val loss = 1.44, small model val loss = 2.22. Gap = −0.78 (large model now better).
- The gap shrinks and reverses as *D* grows, matching Equation (1.5).

### 10. Overfitting Plot

Two subplots: (left) loss vs *D* for small and large models on a semilog x-axis; (right) the overfitting gap vs *D*, showing it shrinks from +6.41 to −0.78.

### 11. Experiment 3 — L(C) Compute Scaling

Estimates compute as *C* = 6 × *N* × batch\_tokens × STEPS for each model, then fits *L*(*C*) = (*C*\_c/*C*)^α\_C. Reports α\_C = 0.1367 (paper: ~0.050), *C*\_c = 8.38×10¹⁴, R² = 0.9944.

### 12. Sample Efficiency

Overlays all 6 validation loss curves on one plot. Larger models reach lower validation loss in fewer steps, reproducing the paper's Figure 2 observation.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `TinyTransformer` | Decoder-only Transformer with causal attention, weight-tied head |
| `count_non_embedding_params` | Counts parameters excluding token + positional embeddings (paper's *N* convention) |
| `get_batch` | Samples a random batch of `(B, T)` input/target sequences from the data tensor |
| `compute_loss` | Cross-entropy loss over all context positions |
| `evaluate` | Average validation loss over *n* random batches |
| `get_lr` | Linear warmup then cosine decay learning rate schedule |
| `train_model` | Full training loop for one model config; returns params, final loss, history |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Dataset | WebText2 (22.9B BPE tokens, 50,257 vocab) | WikiText-2 char-level (10.8M chars, 283 vocab) |
| Model sizes | 768 to 1.5B non-embedding parameters | 25,472 to 793,344 non-embedding parameters |
| Training | 250K steps, batch 512×1024 tokens | 2000 steps, batch 64×128 chars |
| Optimizer | Adam, 3000-step warmup, cosine decay | Adam, 200-step warmup, cosine decay |
| Context length | 1024 tokens | 128 characters |
| Experiments | L(N), L(D), L(C), L(N,D), L(N,S), critical batch size, LSTM comparison | L(N), L(N,D), L(C), sample efficiency |
| Compute measurement | PF-days on dedicated hardware | Estimated FLOPs (6NBS) |
