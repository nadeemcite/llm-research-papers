---
title: "LLM.int8(): Code architecture"
parent: "028 LLM.int8()"
nav_order: 1
---
# Code Architecture — LLM.int8() Notebook

## Overview

The notebook trains a tiny 6-layer transformer (2M params) on character-level Wikitext-2, then implements three int8 quantization methods (absmax, vector-wise, LLM.int8() mixed-precision) and compares their matmul error and end-to-end perplexity against the fp16 baseline. The result shows LLM.int8() achieves 5.5x lower matmul error than absmax (mean 0.009 vs 0.055), all methods produce perplexity 1.02 (model too small for differences to manifest), and exactly 50% memory savings from fp16 to int8.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `EPOCHS = 15`, `LR = 5e-4`, `BATCH_SIZE = 32`
- `SEQ_LEN = 128`, `N_LAYERS = 6`, `D_MODEL = 256`, `N_HEADS = 4`
- `OUTLIER_THRESHOLD = 6.0` — magnitude threshold for outlier detection (paper's value)
- `DEVICE = cuda` (on Kaggle) / `cpu` (local fallback)

### 2. Data

**Function: `get_batch(data, batch_size, seq_len, device)`** — samples random windows from the tokenised text, returns (x, y) tensors of shape (batch, seq_len). Uses character-level tokenisation on Wikitext-2 (~2M train tokens, ~240K valid tokens, vocab ~100 chars).

### 3. Model

**Class: `TinyGPT`** (~4.8M parameters)
- Token + positional embeddings (max_seq_len = 129)
- 6 × `TransformerBlock`: LayerNorm → MultiHeadAttention → residual → LayerNorm → FeedForward (d_model → 4*d_model → d_model with GELU) → residual
- LayerNorm → Linear head (weight-tied with token embedding)
- 3 outlier dimensions injected by scaling specific weight columns 8x at init, mimicking emergent outliers

**Data flow:**
```
Input: (batch, seq_len) int64 token IDs
  -> embeddings -> (batch, seq_len, 256)
  -> 6x TransformerBlock -> (batch, seq_len, 256)
  -> LayerNorm -> head -> (batch, seq_len, vocab_size)
```

### 4. Loss Function

Standard cross-entropy loss on next-token prediction. Optimizer: AdamW (lr 5e-4, weight decay 0.01, grad clip 1.0).

### 5. Training Loop

- 15 epochs, 200 batches per epoch, batch_size 32, seq_len 128
- Validation every epoch (50 batches), logs train/val loss and val perplexity
- Reports final validation perplexity

### 6. Training Curves

Plots training & validation loss (left) and validation perplexity on log scale (right).

### 7. Outlier Feature Analysis

Collects hidden states from 100 batches (128 × 256 = 32K tokens worth), computes per-dimension statistics (max, median, std). Reports the number of dimensions where max > 6 × median, then plots:
1. Per-dimension max absolute value (bar chart showing outlier columns)
2. Value distribution histogram for outlier vs normal dimensions

### 8. Int8 Quantization Implementations

**Function: `absmax_quantize(x)`** — tensor-level absmax scaling. Returns (int8 tensor, scale).
**Function: `vectorwise_quantize_weights(W)`** — per-row absmax for weight matrix W (out, in). Returns (int8_W, row_scales).
**Function: `vectorwise_quantize_hidden(h)`** — per-column absmax for hidden states h (batch, in). Returns (int8_h, col_scales).
**Function: `matmul_absmax(W, h)`** — full matmul with absmax quantization on both operands.
**Function: `matmul_vectorwise(W, h)`** — matmul with vector-wise (per-row/per-column) quantization.
**Function: `matmul_llm_int8(W, h, threshold)`** — LLM.int8() mixed-precision: detects outlier dims in h, splits W and h into outlier (fp16) and normal (int8) parts, computes each separately, sums results.

**Data flow (LLM.int8()):**
```
h (batch, in) -> detect outlier dims -> outlier_mask (in,)
W[:, outlier] -> fp16 matmul -> (out, batch)
W[:, normal]  -> int8 vector-wise matmul -> (out, batch)
sum both -> (out, batch)
```

### 9. Quantization Accuracy Comparison

Extracts a real weight matrix (ff[0], shape 1024×256) and hidden states from the trained model. Computes ground truth (fp32) matmul, then each quantized variant. Reports mean/median/p99 relative error for FP16, absmax int8, vector-wise int8, and LLM.int8(). Plots error box plot and per-dimension error comparison.

### 10. End-to-End Perplexity Comparison

**Function: `quantized_forward(model, input_ids, method)`** — runs the full transformer forward pass with feed-forward linear layers replaced by the specified quantization method. Attention layers stay in fp32 for simplicity. Evaluates validation perplexity for fp16, absmax, vectorwise, and llm_int8 methods over 30 batches. Plots bar chart comparison.

### 11. Memory Savings Analysis

Computes total weight memory for fp32 (4 bytes), fp16 (2 bytes), and int8 (1 byte). Reports the 4x savings vs fp32 and 2x savings vs fp16, matching the paper's claim.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `absmax_quantize` | Tensor-level int8 quantization with single scale factor (baseline) |
| `vectorwise_quantize_weights` | Per-row int8 quantization of weight matrix |
| `vectorwise_quantize_hidden` | Per-column int8 quantization of hidden states |
| `matmul_absmax` | Full matmul using absmax quantization on both operands |
| `matmul_vectorwise` | Matmul using vector-wise (per-row/per-column) quantization |
| `matmul_llm_int8` | LLM.int8() mixed-precision: fp16 for outliers, int8 for normal dims |
| `quantized_forward` | Full transformer forward pass with quantized feed-forward layers |
| `TinyGPT` | 6-layer GPT-style transformer with injected outlier features |
| `get_batch` | Sample random training windows from tokenised text |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | BLOOM-176B, OPT-175B/6.7B/13B/30B | TinyGPT: 6 layers, d_model 256, ~2M params |
| Outlier features | Emerge naturally above ~6B params | Injected via 8x weight scaling at init |
| Dataset | Web text (Pile, C4) | Character-level Wikitext-2 (~2M tokens) |
| Int8 kernels | Custom CUDA int8 tensor core kernels | PyTorch float simulation (no int8 hardware) |
| Benchmarks | LAMBADA, PIQA, HellaSwag, ARC, etc. | Validation perplexity only |
| Speed measurement | 15-23% slowdown vs fp16 | Not measured (pure-Python int8 is slower) |
| Optimizer / epochs | Large-scale pretraining | 15 epochs, AdamW lr 5e-4 |
