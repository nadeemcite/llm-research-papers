---
title: "Gated Attention: Code architecture"
parent: "012 Gated Attention"
nav_order: 1
---
# Code Architecture — Gated Attention Notebook

## Overview

The notebook implements a standard multi-head attention layer and a gated attention layer (with a head-specific sigmoid gate after SDPA) from scratch in PyTorch. Both are embedded in small transformer language models and trained on Tiny Shakespeare (character-level). The notebook measures validation perplexity, analyzes the attention sink (first-token attention proportion), and plots the gating score distribution. At this toy scale, the attention sink phenomenon does not emerge, consistent with the paper's finding that it requires large-scale training.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `D_MODEL = 128` — hidden dimension
- `N_HEADS = 4` — number of attention heads
- `N_LAYERS = 4` — transformer layers
- `D_FF = 512` — feed-forward dimension
- `SEQ_LEN = 128` — context length
- `EPOCHS = 12`, `STEPS_PER_EPOCH = 50`, `LR = 3e-4`, `BATCH_SIZE = 32`

### 2. Data — Tiny Shakespeare

Downloads Tiny Shakespeare (~1 MB, 65 unique characters). Prepares train/val splits (90/10). Creates a character-level tokenizer (char-to-index and index-to-char mappings).

### 3. Model Architecture

**Class: `MultiHeadAttention`** — standard multi-head attention with Q, K, V projections, scaled dot-product attention, and output projection. Uses causal masking. About 0.20M parameters per layer.

**Class: `GatedMultiHeadAttention`** — identical to `MultiHeadAttention` but adds a gate: after computing SDPA output per head, multiplies by `sigmoid(X @ W_gate)` where W_gate maps from d_model to n_heads * d_head. This is the paper's Eq. 5: `Y' = Y ⊙ σ(XWθ)`. About 0.21M parameters per layer.

**Class: `TransformerBlock`** — pre-RMSNorm + attention + SwiGLU FFN + residual connections.

**Class: `TransformerLM`** — token embedding + N transformer blocks + RMSNorm + output head with weight tying.

**Data flow:**
```
Input: (batch, seq_len)
  -> embedding -> (batch, seq_len, d_model)
  -> N × [RMSNorm -> Attention -> residual -> RMSNorm -> SwiGLU -> residual]
  -> RMSNorm -> output projection -> (batch, seq_len, vocab_size)
```

### 4. Training Loop

Both models train with identical hyperparameters. Cross-entropy loss on next-token prediction. AdamW optimizer (lr=3e-4, wd=0.1, betas=0.9/0.95). 12 epochs, 50 steps per epoch. Logs train/val loss per epoch.

### 5. Attention Sink Analysis

Extracts attention weights from all layers/heads on validation samples. Computes the proportion of attention allocated to the first token (the "attention sink" metric from the paper's Figure 2). Compares baseline vs gated model.

### 6. Gating Score Distribution

Extracts gating scores from the gated model on validation samples. Plots a histogram of all gate values across all layers and heads. Computes the mean gating score and sparsity metrics.

### 7. Results and Plots

- **Training curves:** Train/val loss and perplexity for both models over 12 epochs
- **Attention sink bar chart:** First-token attention proportion per layer, baseline vs gated
- **Gating score histogram:** Distribution of sigmoid gate values

## Key Functions

| Function / Class | Purpose |
|---|---|
| `MultiHeadAttention` | Standard multi-head attention without gating |
| `GatedMultiHeadAttention` | Multi-head attention with head-specific sigmoid gate after SDPA |
| `TransformerBlock` | Single transformer layer (pre-norm, attention, SwiGLU FFN) |
| `TransformerLM` | Full transformer language model |
| `train_model` | Training loop logging train/val loss per epoch |
| `analyze_attention_sink` | Extract first-token attention proportion across layers |
| `extract_gate_scores` | Extract and return all gating scores for distribution analysis |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 15B MoE / 1.7B dense | ~1M parameters (4 layers, 128 dim) |
| Dataset | 3.5T tokens, multilingual | Tiny Shakespeare (~1 MB, 65 chars) |
| Attention | GQA, 32 query heads, 4 KV heads | Standard MHA, 4 heads |
| Gate granularity | Elementwise (n × q × dk) | Elementwise (n × n_heads × d_head) |
| Context extension | YaRN to 128K | None (128-token context) |
| Training | Full-scale pretraining | 12 epochs, 50 steps/epoch |
