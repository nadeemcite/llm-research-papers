---
title: "GQA: Code architecture"
parent: "005 GQA"
nav_order: 1
---
# Code Architecture — GQA Notebook

## Overview

The notebook implements three attention variants — Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA) — each in a complete tiny Transformer decoder, trained from scratch on synthetic character-level text. The result is measured by validation loss and KV-cache tensor size at a fixed sequence length.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `EPOCHS = 30`, `LR = 3e-4`, `BATCH_SIZE = 128`
- `N_LAYERS = 4`, `D_MODEL = 128`, `N_HEADS = 8`, `SEQ_LEN = 64`
- `N_VOCAB = 32` (26 letters + space + apostrophe + digits 0-4 + extra char for template slots)
- `DEVICE = cuda if available else cpu`

### 2. Data

**Function: `generate_synthetic_data()`** — Creates ~8,000 short sentences from a fixed set of templates (e.g., "the cat sat on the mat", "a dog ran in the park"). Returns a list of strings. Split into 90% train / 10% validation.

**Class: `CharDataset`** — Converts strings to character-level token IDs. Each sample is SEQ_LEN characters; targets are shifted by one. Returns tensors of shape `(SEQ_LEN,)` and `(SEQ_LEN,)`.

### 3. Model

**Class: `MultiHeadAttention`** (MHA, 8 KV heads, ~1.07M total params in the full model)
- 8 query heads, 8 key heads, 8 value heads — standard attention.
- Q, K, V projections: `(D_MODEL, D_MODEL)` each, output reshaped to `(batch, N_HEADS, seq, head_dim)`.

**Class: `MultiQueryAttention`** (MQA, 1 KV head, ~0.95M total params)
- 8 query heads, 1 key head, 1 value head.
- Q projection: `(D_MODEL, D_MODEL)`. K, V projections: `(D_MODEL, head_dim)` — much smaller.
- K, V are broadcast across all query heads during attention.

**Class: `GroupedQueryAttention`** (GQA-2, 2 KV heads, ~0.97M total params)
- 8 query heads divided into 2 groups of 4. Each group shares 1 key and 1 value head.
- Q projection: `(D_MODEL, D_MODEL)`. K, V projections: `(D_MODEL, 2 * head_dim)`.
- K, V are repeated within each group: head i uses KV head `i // 4`.

**Class: `TransformerBlock`** — Standard pre-norm Transformer decoder block: attention → residual → RMSNorm → MLP (SwiGLU) → residual → RMSNorm.

**Class: `TinyTransformer`** — N_LAYERS Transformer blocks + token embedding + positional embedding (learned) + output projection. About 0.95–1.07M parameters depending on attention variant.

**Data flow:**
```
Input: (batch, seq)
  -> embedding: (batch, seq, D_MODEL)
  -> + positional: (batch, seq, D_MODEL)
  -> 4× TransformerBlock:
       -> attention: (batch, seq, D_MODEL)
       -> MLP: (batch, seq, D_MODEL)
  -> output proj: (batch, seq, N_VOCAB)
```

### 4. Loss Function

Standard cross-entropy loss for next-token prediction:
`L = -sum_t log P(x_{t+1} \| x_{1:t})` averaged over batch and sequence positions.

### 5. Training Loop

- Optimizer: AdamW, lr 3e-4, weight_decay 0.01
- 30 epochs, batch 128, sequence length 64
- Loss logged per epoch; printed every 5 epochs
- Separate model trained for each attention variant (MHA, MQA, GQA-2)

### 6. KV-Cache Measurement

After training, the KV cache size is computed for each variant at SEQ_LEN=64:
- `kv_cache_size = n_kv_heads * seq_len * head_dim * 2 (K + V) * batch_size`
- Printed as `KV cache (MHA): X elements`, etc.
- A bar chart compares the three variants.

### 7. Results and Plots

- Training loss curves for all three variants overlaid in one plot
- Validation loss comparison bar chart
- KV cache size comparison bar chart
- Print: `Val loss (MHA): X`, `Val loss (MQA): X`, `Val loss (GQA-2): X`
- Print: `KV cache reduction (MQA vs MHA): Xx`, `KV cache reduction (GQA vs MHA): Xx`

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_synthetic_data` | Creates ~8K template sentences for character-level training |
| `CharDataset` | Converts strings to fixed-length char token tensors with shifted targets |
| `MultiHeadAttention` | Standard MHA with 8 Q/K/V heads |
| `MultiQueryAttention` | MQA with 8 query heads, 1 shared K/V head |
| `GroupedQueryAttention` | GQA with 8 query heads, 2 shared K/V heads (groups of 4) |
| `TransformerBlock` | Pre-norm decoder block: attention + SwiGLU MLP + residuals + RMSNorm |
| `TinyTransformer` | Full model: embedding + N_LAYERS blocks + output projection |
| `train_model` | Trains one variant for EPOCHS epochs, returns model and loss history |
| `measure_kv_cache` | Computes KV cache tensor size for a given attention variant |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | T5-XXL, 11B params, 128 heads | 4 layers, 128-dim, 8 heads, ~1.07M params |
| Data | C4 (750GB), then fine-tuned on CNN/DailyMail, WMT, TriviaQA | ~8K synthetic template sentences, char-level |
| Training | Uptraining from MHA checkpoint (5% of original compute) with Adafactor | Trained from scratch with AdamW for 30 epochs |
| Evaluation | ROUGE/BLEU/F1 on 7 benchmarks + TPUv4 wall-clock time | Validation cross-entropy loss + KV cache tensor size |
| Attention variants | MHA, MQA, GQA-8 (uptrained) | MHA-8, MQA-1, GQA-2 (from scratch) |
| KV cache measurement | Inference time per sample on TPUv4 | Tensor element count at fixed seq_len |
