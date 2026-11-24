---
title: "Induction Heads: Code architecture"
parent: "088 Induction Heads"
nav_order: 1
---
# Code Architecture -- Induction Heads Notebook

## Overview

The notebook implements a 2-layer attention-only transformer (67,584 parameters) from scratch in PyTorch, trains it on synthetic repeated random sequences (16-token vocabulary, 128 tokens per sequence), detects induction heads via prefix matching scores, and demonstrates that ablating induction heads destroys in-context learning. The ICL score goes from 2.84 (early loss 2.84, late loss 0.003) to -0.50 after ablation, while control ablation preserves ICL at 3.04.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `VOCAB_SIZE = 16`, `SEQ_LEN = 128` (half: 64), `N_LAYERS = 2`, `N_HEADS = 4`, `D_MODEL = 64`, `D_HEAD = 16`
- `EPOCHS = 100`, `BATCH_SIZE = 128`, `LR = 1e-3`
- `N_TRAIN_SEQS = 4096`, `N_TEST_SEQS = 512`

### 2. Data

**Function: `gen_repeated_sequences`** -- generates sequences where the first half is random and the second half is an exact copy. Returns `(n, seq_len)` tensor of token indices. Training data uses seed=42, test data seed=123, induction detection data seed=999, ensuring no overlap.

### 3. Model

**Class: `AttentionOnlyTransformer`** (67,584 parameters)
- `AttentionHead`: single attention head with W_Q, W_K, W_V, W_O linear projections (no bias). Causal masking. Returns optional attention weights.
- `AttentionLayer`: multi-head attention (sum of heads, no MLP). Returns optional list of per-head attention maps.
- `AttentionOnlyTransformer`: token embedding + learned positional embedding (max 512 positions), 2 attention layers with residual connections, unembed linear layer.

**Data flow:**
```
Input: (batch, 128) token indices
  -> embed + pos_embed -> (batch, 128, 64)
  -> Layer 0: 4-head attention + residual -> (batch, 128, 64)
  -> Layer 1: 4-head attention + residual -> (batch, 128, 64)
  -> unembed -> (batch, 128, 16) logits
```

### 4. Loss

Cross-entropy loss for next-token prediction: logits at positions 0..127 predict tokens 1..128. ICL score = loss at position 8 minus loss at position 100.

### 5. Training Loop

- Optimizer: Adam, lr=1e-3
- 100 epochs, batch size 128
- Logs loss and ICL score every 20 epochs
- Training time: ~109 seconds on CPU

### 6. Training Curves

Two subplots: training loss over epochs (with random baseline at log(16) = 2.77) and ICL score over epochs (with zero baseline).

### 7. Induction Head Detection

**Function: `compute_prefix_matching_score`** -- on repeated sequences, measures how much each attention head attends to position i-L+1 (the token after the matching prefix token) when processing position i in the second half. Returns `(n_layers, n_heads)` score array.

Threshold: max(0.05, 1/128 * 5) = 0.05. Layer 1 heads L1H0 (0.95) and L1H1 (0.18) exceed threshold; layer 0 heads do not.

### 8. Attention Pattern Visualization

Heatmaps of attention weights for all 8 heads on a single repeated sequence. Induction heads show a characteristic off-diagonal striped pattern.

### 9. Ablation

**Function: `model_forward_with_ablation`** -- forward pass that skips specified heads (zeroed output). Tests three conditions: no ablation, induction heads ablated, control heads ablated.

### 10. Per-Token Loss Plot

Per-position loss for all three conditions, with vertical lines marking the repeat boundary (idx 64), early token (idx 8), and late token (idx 100).

### 11. Prefix Matching Scores Bar Chart

Bar chart of prefix matching scores for all 8 heads, color-coded by induction status, with threshold and baseline reference lines.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `gen_repeated_sequences` | Generate synthetic repeated random sequences for training/testing |
| `AttentionHead` | Single causal attention head with QKV projections |
| `AttentionLayer` | Multi-head attention layer (sum of heads, no MLP) |
| `AttentionOnlyTransformer` | 2-layer attention-only transformer with residual connections |
| `compute_prefix_matching_score` | Detect induction heads via attention to correct induction positions |
| `model_forward_with_ablation` | Forward pass with specified heads zeroed out |
| `compute_icl_score_ablated` | ICL score with optional head ablation |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Data | Natural language (billions of tokens) | Synthetic repeated random sequences (4096 sequences, 16-token vocab) |
| Model size | 34 models, 1 layer to full-scale with MLPs | Single 2-layer attention-only model, 67K parameters |
| Training schedule | Billions of tokens, learning rate schedules | 100 epochs, fixed lr 1e-3 |
| Evidence | 6 arguments (co-occurrence, co-perturbation, ablation, generality, mechanism, continuity) | 2 of 6: prefix matching detection (related to Arg 1/4) + ablation (Arg 3) |
| Analysis tools | Eigenvalue analysis, PCA of per-token losses, model analysis table | Prefix matching score + ablation only |
