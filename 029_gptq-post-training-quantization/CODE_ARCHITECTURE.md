---
title: "GPTQ: Code architecture"
parent: "029 GPTQ"
nav_order: 1
---
# Code Architecture — GPTQ Notebook

## Overview

The notebook trains a tiny 4-layer transformer (846K params) on character-level Wikitext-2, then implements layer-wise GPTQ quantization with a Cholesky-based inverse Hessian and compares 4-bit perplexity and reconstruction error against round-to-nearest (RTN). The result shows GPTQ achieves perplexity of 4.14 (+0.33 vs fp16's 3.81) while RTN degrades to 5.54 (+1.72), and average layer reconstruction error of 0.024 vs RTN's 0.169 (7.1x improvement).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `EPOCHS = 20`, `LR = 1e-3`, `BATCH_SIZE = 64`
- `SEQ_LEN = 128`, `N_LAYERS = 4`, `D_MODEL = 128`, `N_HEADS = 4`
- `N_BITS = 4`, `BLOCK_SIZE = 128`, `DAMPING = 0.01`
- `DEVICE = cuda` (on Kaggle) / `cpu` (local fallback)

### 2. Data

**Function: `get_batch(data, batch_size, seq_len, device)`** — samples random windows from the tokenised text, returns (x, y) tensors of shape (batch, seq_len). Uses character-level tokenisation on Wikitext-2 (~2M train tokens, ~240K valid tokens, vocab ~100 chars).

### 3. Model

**Class: `TinyGPT`** (~846K parameters)
- Token + positional embeddings (max_seq_len = 129)
- 4 × `TransformerBlock`: LayerNorm → MultiHeadAttention → residual → LayerNorm → FeedForward (d_model → 4*d_model → d_model with GELU) → residual
- LayerNorm → Linear head (weight-tied with token embedding)

**Data flow:**
```
Input: (batch, seq_len) int64 token IDs
  -> embeddings -> (batch, seq_len, 128)
  -> 4x TransformerBlock -> (batch, seq_len, 128)
  -> LayerNorm -> head -> (batch, seq_len, vocab_size)
```

### 4. Loss Function

Standard cross-entropy loss on next-token prediction. Optimizer: AdamW (lr 1e-3, weight decay 0.01, grad clip 1.0).

### 5. Training Loop

- 20 epochs, 200 batches per epoch, batch_size 64, seq_len 128
- Validation every epoch (50 batches), logs train/val loss and val perplexity
- Reports final validation perplexity

### 6. Training Curves

Plots training & validation loss (left) and validation perplexity on log scale (right).

### 7. Quantization Utilities

**Function: `quantize_value(x, n_bits)`** — uniform symmetric quantization to n_bits. Returns quantized tensor. Scale = max(\|x\|) / (2^(n_bits-1) - 1).

**Function: `quantize_weight_rtn(W, n_bits)`** — per-row round-to-nearest quantization. Quantizes each row of W independently to the n_bits grid. Returns quantized weight matrix.

**Function: `quantize_layer_gptq(W, X, n_bits, block_size, damping)`** — the core GPTQ algorithm. Takes weight matrix W (out, in), input activations X (in, m), quantizes W column by column using the inverse Hessian of the reconstruction error.
- Computes H = 2 * X @ X^T + damping * I
- Computes H^{-1} via Cholesky: H_inv = cholesky(H)^{-T}
- For each block of B columns: quantize each column, compute error, update remaining weights in block
- Returns quantized weight matrix

**Data flow (GPTQ):**
```
W (out, in), X (in, m)
  -> H = 2*X*X^T + λI -> (in, in)
  -> Cholesky(H)^{-T} -> H_inv (in, in)
  -> for each block of B columns:
       quantize column j -> Q[:,j]
       error = (W[:,j] - Q[:,j]) / H_inv[j,j]
       W[:, j:block_end] -= error * H_inv[j, j:block_end]
  -> quantized W (out, in)
```

### 8. Layer-wise Quantization Application

**Function: `apply_quantization(model, method, calibration_data, n_bits)`** — iterates over all Linear layers in the model. For each layer, collects calibration activations by running calibration data through the model up to that layer, then applies either RTN or GPTQ quantization. Replaces the layer's weight with the quantized version (stored as float for simulation).

### 9. Perplexity Evaluation with Quantized Weights

**Function: `evaluate_perplexity(model, data, method, n_batches)`** — runs the full transformer forward pass with quantized linear layers, computes cross-entropy loss, converts to perplexity. Evaluates fp16, RTN 4-bit, and GPTQ 4-bit.

### 10. Reconstruction Error Analysis

Extracts a real weight matrix and calibration activations from the trained model. Computes the fp16 layer output (W @ X), then compares with RTN-quantized and GPTQ-quantized outputs. Reports mean squared error for each method. Plots per-column reconstruction error comparison.

### 11. Results Summary

Bar charts comparing perplexity across methods and reconstruction error across methods. Prints all key numbers.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `TinyGPT` | 4-layer GPT-style transformer (~846K params) |
| `get_batch` | Sample random training windows from tokenised text |
| `quantize_value` | Uniform symmetric quantization to n_bits |
| `quantize_weight_rtn` | Per-row round-to-nearest quantization |
| `quantize_layer_gptq` | Core GPTQ algorithm: column-wise quantization with Cholesky inverse Hessian |
| `apply_quantization` | Apply RTN or GPTQ to all Linear layers using calibration data |
| `evaluate_perplexity` | Evaluate validation perplexity with quantized weights |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | OPT-175B, BLOOM-176B | TinyGPT: 4 layers, d_model 128, ~846K params |
| Calibration data | 128 × 2048-token C4 segments | 128 batches × 128-token Wikitext-2 windows |
| Quantization bits | 2, 3, 4 bits | 4 bits only |
| Quantization grid | Per-row asymmetric min-max | Per-row symmetric absmax |
| GPU kernels | Custom CUDA dequantization kernels | Float32 simulation (no int4 hardware) |
| Benchmarks | WikiText2, PTB, C4, LAMBADA, PIQA, ARC | Validation perplexity only |
| Speedup measurement | 3.25x on A100, 4.5x on A6000 | Not measured (simulation is slower) |
| Optimizer / epochs | Large-scale pretraining | 20 epochs, AdamW lr 1e-3 |
