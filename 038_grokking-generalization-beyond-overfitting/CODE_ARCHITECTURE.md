---
title: "Grokking: Code architecture"
parent: "038 Grokking"
nav_order: 1
---
# Code Architecture — Grokking Notebook

## Overview

The notebook trains a 2-layer decoder-only Transformer (~409K non-embedding parameters) on modular addition
(a + b mod 97), using 8% of all 9,409 possible equations as the training set. It reproduces the grokking
phenomenon: training accuracy reaches ~100% within ~20 steps, validation accuracy plateaus at ~98%
for ~15 more steps, then jumps to ~100%. Results are measured by train/val accuracy and cross-entropy
loss logged every 5 steps (early) / 50 steps (later), plus a t-SNE visualization of the learned token
embeddings.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds (SEED=42) and defines the key constants:
- `P = 97` — prime modulus (same as paper)
- `TRAIN_FRAC = 0.08` — fraction of all p² equations for training (very low, near-critical → clear grokking gap)
- `D_MODEL = 128`, `N_LAYERS = 2`, `N_HEADS = 4` — Transformer architecture (same as paper Appendix A.1.2)
- `EPOCHS = 3000` — optimization steps (paper used 10⁵; 3000 is enough to grok at 8% without weight decay)
- `LR = 1e-3`, `WEIGHT_DECAY = 0.0`, `WARMUP = 10`, `BATCH_SIZE = 512` — optimizer settings (weight decay set to 0 to make the grokking gap more visible; the paper found wd accelerates grokking)

### 2. Data — Modular Addition

Generates all p² = 9,409 equations of the form `a + b ≡ c (mod 97)`, each encoded as a 5-token sequence:
`<a> <op> <b> <=> <c>`. Vocabulary: tokens 0–96 are residues, 97 is '+', 98 is '=' (vocab size 99).
Each operand is an abstract symbol — the model sees no integer values. Random 8/92 train/val split.

**Shapes:** `equations` (9409, 5) int64 → `train_data` (752, 5), `val_data` (8657, 5) on DEVICE.

### 3. Model — GrokkingTransformer

**Class: `GrokkingTransformer`** (~423K total params, ~409K non-embedding)
- `token_emb`: nn.Embedding(99, 128) — token embeddings for residues + operators
- `pos_emb`: nn.Embedding(5, 128) — positional embeddings (seq len = 5)
- `transformer`: nn.TransformerEncoder with 2 layers, 4 heads, dim_feedforward=512, GELU activation,
  pre-norm (norm_first=True), dropout=0.0
- `ln_f`: nn.LayerNorm(128) — final layer norm
- `head`: nn.Linear(128, 99, bias=False) — output projection to vocab
- `causal_mask`: 5×5 upper-triangular -inf mask (registered buffer)

Same architecture as the paper (Appendix A.1.2): 2 layers, width 128, 4 heads, ~4×10⁵ non-embedding
params (matching the paper's ~4×10⁵; our vocab is 99 tokens for a single operation).

**Data flow:**
```
Input: (B, 5) int64
  -> token_emb + pos_emb -> (B, 5, 128)
  -> TransformerEncoder (causal mask) -> (B, 5, 128)
  -> LayerNorm -> (B, 5, 128)
  -> Linear -> (B, 5, 99)
Output: logits (B, 5, 99)
```

### 4. Loss & Accuracy

Cross-entropy loss computed only on the answer token at position 4:
`loss = CE(logits[:, 4, :], targets[:, 4])`. Accuracy is argmax over vocab at position 4 compared
to the true residue. Chance level is 1/97 ≈ 1.03% (97 valid answer residues).

### 5. Training Loop

- **Optimizer:** AdamW (lr=1e-3, weight_decay=0.0, betas=(0.9, 0.98)) — same as paper except wd=0
- **Scheduler:** Linear warmup over first 10 steps, then constant lr
- **Batching:** Random minibatch of 512 from train_data each step
- **Logging:** Every 5 steps for the first 100 steps (to capture the grokking transition), then every 50 steps
- **Total:** 3,000 steps

### 6. Results — Grokking Plots

Two side-by-side plots:
- **Left:** Train vs. validation accuracy over optimization steps, with chance level marked. Shows
  train accuracy rising to ~100% within ~20 steps, val accuracy plateauing at ~98% then jumping to ~100% at ~step 35.
- **Right:** Train vs. validation loss. Shows the double descent in validation loss — it rises
  during overfitting, then falls sharply when the model groks.

Also prints: step where train/val accuracy first exceeds 99%, grokking ratio (val grok step /
train grok step), and final metrics.

### 7. Embedding Visualization

Extracts token embeddings for residues 0–96 (shape (97, 128)) and projects to 2D with t-SNE
(sklearn.manifold.TSNE, perplexity=15). Plots with color = residue value. If the model has grokked,
the embeddings should show structure reflecting the cyclic group Z/97Z.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `GrokkingTransformer` | 2-layer decoder-only Transformer for modular arithmetic |
| `compute_loss` | Cross-entropy loss on the answer token (position 4) only |
| `compute_accuracy` | Argmax accuracy on the answer token only |
| `evaluate` | Full-dataset evaluation: returns average loss and accuracy |
| `lr_lambda` | Linear warmup schedule for the first 10 steps |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Operations | 12+ (addition, subtraction, division, S₅ composition, polynomials, etc.) | Modular addition only (mod 97) |
| Training steps | 10⁵ (extended to 10⁶ for some experiments) | 3,000 |
| Train fraction | Explored 25–95% | Fixed at 8% (near-critical for visible grokking) |
| Ablations | Full-batch, SGD, dropout, weight decay variants, gradient noise, lr sweep | None (weight decay=0 only) |
| Data efficiency curves | 7 seeds per fraction, median steps to 99% val acc | Not measured |
| Outlier experiments | k ∈ [0, 10, 100, 1000, 2000, 3000] random-label outliers | Not included |
| Generalization measures | Sharpness φ, Spearman correlation | Not included |
| Embedding viz | t-SNE of output layer weights | t-SNE of token embeddings |
