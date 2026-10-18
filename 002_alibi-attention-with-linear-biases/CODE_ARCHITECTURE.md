---
title: "ALiBi: Code architecture"
parent: "002 ALiBi"
nav_order: 1
---
# Code Architecture — ALiBi Notebook

## Overview

The notebook implements a small character-level Transformer (nanoGPT-style) in PyTorch with two position-encoding variants — sinusoidal (Vaswani et al., 2017) and ALiBi (Press et al., 2021) — and compares their length-extrapolation ability. Both models are trained on 2048 synthetic counting sequences of length 256 for 40 epochs, then evaluated on length-256 and length-1024 test sequences. ALiBi maintains about 99.89% next-token accuracy at 4× the training length, while Sinusoidal drops to about 71.75% (62.85% on the unseen positions 256–1023 alone).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds (SEED=42) and defines the key constants (copied from the code cell):
- `VOCAB_SIZE = 20`, `EMBED_DIM = 128`, `NUM_HEADS = 4`, `NUM_LAYERS = 4`
- `TRAIN_LEN = 256`, `EVAL_LEN = 1024` (4× training length for extrapolation test)
- `EPOCHS = 40`, `LR = 3e-4`, `BATCH_SIZE = 128`
- `N_TRAIN = 2048`, `N_TEST = 256`, `EVAL_BATCH = 32`
- `DEVICE = cuda` (on Kaggle GPU) or `cpu`

### 2. Data

**Function: `make_sequences(n, length, seed)`** — generates n sequences where token at position t = (a·t + b) % VOCAB_SIZE, with each sequence getting a random slope a and offset b. Returns `np.ndarray` of shape (n, length) with int64 tokens.

Three datasets: `train_data` (2048 × 256), `test_short` (256 × 256, same length as train), `test_long` (256 × 1024, 4× training length). A plot of three example sequences (first 80 tokens) is displayed.

### 3. Model

**Function: `get_alibi_slopes(n_heads)`** — returns a tensor of n_heads slopes forming a geometric sequence starting at $2^{-8/n}$ with ratio $2^{-8/n}$. For 4 heads: $[0.25, 0.0625, 0.015625, 0.00390625]$.

**Function: `build_alibi_mask(seq_len, n_heads, device)`** — builds a (1, n_heads, seq_len, seq_len) tensor combining the ALiBi bias (slope × distance) with the causal mask (-inf for future positions). This replaces the standard causal mask in attention.

**Function: `sinusoidal_pe(seq_len, embed_dim, device)`** — standard sinusoidal position embeddings (sin/cos of varying frequencies), shape (seq_len, embed_dim).

**Class: `CausalSelfAttention`** — multi-head causal self-attention with QKV projection. If `use_alibi=True`, passes the pre-built ALiBi mask to `F.scaled_dot_product_attention`; otherwise uses `is_causal=True`. Uses no bias in QKV/proj.

**Class: `Block`** — pre-norm Transformer decoder block: `x + attn(ln1(x))`, then `x + mlp(ln2(x))`. MLP uses GELU with 4× expansion.

**Class: `NanoGPT`** (933,120 params for Sinusoidal, 793,856 for ALiBi) — token embedding + optional sinusoidal position embedding (Sinusoidal only) + 4 Blocks + LayerNorm + linear head. ALiBi variant has no position embedding but builds the ALiBi mask on the fly in forward(). Weight tying between token embedding and output head.

**Data flow:**
```
Input: (B, T) integer tokens
  -> tok_emb: (B, T, 128)
  -> [+ pos_emb if Sinusoidal] -> (B, T, 128)
  -> 4x Block: (B, T, 128) -> (B, T, 128)
     Block: attn(B, 4_heads, T, 32) [with ALiBi mask or causal mask] -> proj -> mlp(512) -> (B, T, 128)
  -> ln_f -> head: (B, T, 20) [weight-tied with tok_emb]
```

### 4. Loss Function

Cross-entropy loss for next-token prediction: logits[:, :-1] predict targets[:, 1:]. The loss is `F.cross_entropy` over the 20-token vocabulary. Both models use the same loss.

### 5. Training Loop

- Optimizer: AdamW, lr=3e-4 (same for both models)
- 40 epochs, batch size 128, gradient clipping at norm 1.0
- Per epoch: shuffle training data, mini-batch over all 2048 sequences
- Logs training loss at epochs 1, 10, 20, 30, 40
- Sinusoidal training time: ~58s on Kaggle T4 GPU; ALiBi: ~74s on Kaggle T4 GPU

### 6. Results and Plots

- **Training curves plot:** both models converge, with ALiBi reaching lower final loss (0.013 vs 0.047). ALiBi converges faster, reaching loss 0.039 by epoch 20 vs sinusoidal's 1.04.
- **Extrapolation bar chart:** side-by-side accuracy and loss in three groups: length 256, length 1024, and length 1024 restricted to target positions 256-1023. ALiBi bars stay near 100% (99.59%, 99.89%, 100.00%); Sinusoidal drops to 71.75% at length 1024 and 62.85% on positions 256-1023.
- **ALiBi bias visualization:** heatmap of the ALiBi bias for each of the 4 heads over a 64×64 window, plus a line plot of bias vs distance for each head. Head 1 (slope=0.25) penalizes distant tokens most aggressively; head 4 (slope=0.0039) has the gentlest penalty.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `get_alibi_slopes` | Compute the geometric sequence of head slopes for ALiBi |
| `build_alibi_mask` | Build the (1, n_heads, L, L) ALiBi bias + causal mask tensor |
| `sinusoidal_pe` | Generate standard sinusoidal position embeddings |
| `CausalSelfAttention` | Multi-head causal self-attention with optional ALiBi mask |
| `NanoGPT` | Full model: embedding + N Transformer blocks + output head, with Sinusoidal PE or ALiBi |
| `make_sequences` | Generate synthetic counting sequences for training and evaluation |
| `train_model` | Training loop with mini-batch SGD, loss logging, gradient clipping |
| `evaluate_model` | Compute average cross-entropy loss, accuracy, and perplexity on a dataset |
| `evaluate_extrapolation_positions` | Evaluate accuracy only on target positions beyond the training length |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 16-layer / 25-layer, 1.3B params | 4-layer, 4-head, 128-dim (~800K params) |
| Dataset | WikiText-103 (103M tokens) / CC100+RoBERTa (461 GB) | 2048 synthetic counting sequences (20-token vocab) |
| Training | Multi-stage pretraining on WikiText-103 / 50k updates on 128 V100 GPUs (1.3B model) | 40 epochs, AdamW, lr 3e-4, CPU or single GPU |
| Extrapolation ratio | 2× (1024 → 2048) | 4× (256 → 1024) |
| Task | Language modeling (next-token prediction, 267K vocab) | Next-token prediction (20-token vocab) |
| Position encoding | ALiBi vs sinusoidal, rotary, T5 bias | ALiBi vs sinusoidal only |
