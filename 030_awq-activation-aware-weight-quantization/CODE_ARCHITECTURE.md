---
title: "AWQ: Code architecture"
parent: "030 AWQ"
nav_order: 1
---
# Code Architecture — AWQ Notebook

## Overview

The notebook trains a tiny 4-layer transformer (844K params) on character-level Wikitext-2, then implements three quantization methods — RTN (round-to-nearest), GPTQ (second-order error compensation), and AWQ (activation-aware scaling) — and compares 4-bit perplexity and reconstruction error. AWQ achieves perplexity of 4.40 (+0.58 vs fp16's 3.83) while RTN degrades to 4.45 (+0.62), and GPTQ achieves 4.04 (+0.21). Average layer reconstruction MSE is 0.041 for AWQ vs 0.047 for RTN (1.2x improvement). GPTQ outperforms AWQ on this tiny model; AWQ's advantage is generalization at large scale.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `EPOCHS = 20`, `LR = 1e-3`, `BATCH_SIZE = 64`
- `SEQ_LEN = 128`, `N_LAYERS = 4`, `D_MODEL = 128`, `N_HEADS = 4`
- `N_BITS = 4`, `GROUP_SIZE = 64`
- `N_CALIBRATION_BATCHES = 32`, `N_EVAL_BATCHES = 50`
- `DEVICE = cuda` (on Kaggle) / `cpu` (local fallback)

### 2. Data

**Function: `get_batch(data, batch_size, seq_len, device)`** — samples random windows from the tokenised text, returns (x, y) tensors of shape (batch, seq_len). Uses character-level tokenisation on Wikitext-2 (~2M train tokens, ~240K valid tokens, vocab ~100 chars).

### 3. Model

**Class: `TinyGPT`** (~846K parameters)
- Token + positional embeddings (max_seq_len = 129)
- 4 × `TransformerBlock`: LayerNorm → explicit Q/K/V/O projections → causal self-attention → residual → LayerNorm → FeedForward (d_model → 4*d_model → d_model with GELU) → residual
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

Three quantization methods are implemented:

**Function: `quantize_group_rtn(W, n_bits, group_size=64)`** — group-wise round-to-nearest quantization. For each group of 64 consecutive input channels, computes absmax scale and rounds to the n_bits grid. Returns quantized weight matrix.

**Function: `quantize_layer_gptq(W, X, n_bits, group_size=64, damping=0.01)`** — the GPTQ algorithm (from paper 029). Computes Hessian H = 2XX^T + λI, inverts via Cholesky, quantizes columns one at a time with error compensation. Returns quantized weight matrix.

**Function: `awq_search_scale(W, X, n_bits, group_size=64, alpha_grid=None)`** — the core AWQ algorithm. Identifies salient channels via activation magnitudes, searches for optimal per-channel scaling factor.
- Computes s_X = average activation magnitude per input channel (in_features,)
- For each alpha in grid {0, 0.25, 0.5, 0.75, 1.0}:
  - Scale: s = s_X^alpha (in_features,)
  - Scale weights: W' = W * s (per column)
  - Quantize W' with group-wise RTN
  - Inverse scale: Q = Q(W') / s
  - Compute reconstruction MSE: \|W@X - Q@X\|^2
- Returns quantized weight with best alpha, the scale vector, and best alpha value


**Data flow (AWQ):**
```
W (out, in), X (in, m)
  -> s_X = mean(|X|, dim=1) -> (in,)
  -> for each alpha in {0, 0.25, 0.5, 0.75, 1.0}:
       s = s_X^alpha -> (in,)
       W_scaled = W * s -> (out, in)
       Q_scaled = group_RTN(W_scaled) -> (out, in)
       Q = Q_scaled / s -> (out, in)
       loss = MSE(Q @ X, W @ X)
  -> return Q with lowest loss, best alpha
```

### 8. Collect Calibration Data

**Function: `collect_calibration_activations(model, data, n_batches, device)`** — registers forward hooks on all Linear layers, runs 128 calibration batches through the model, collects input activations for all Linear layers except the head. Returns dict mapping layer name to (in_features, n_samples) tensor.

### 9. Apply Quantization and Evaluate

Three wrapper functions apply quantization to all Linear layers except the head (tied embeddings):
- `apply_rtn_quantization` — group-wise RTN
- `apply_gptq_quantization` — GPTQ with calibration activations
- `apply_awq_quantization` — AWQ with grid search, returns model and per-layer alpha values

### 10. Perplexity Evaluation

**Function: `compute_perplexity(model, data, n_batches, device)`** — runs the full transformer forward pass with quantized linear layers, computes cross-entropy loss, converts to perplexity. Evaluates fp16, RTN 4-bit, GPTQ 4-bit, and AWQ 4-bit.

### 11. Reconstruction Error Analysis

**Function: `compute_reconstruction_error(model_orig, model_quant, calibration_activations)`** — for each Linear layer, computes MSE between fp16 output (W @ X) and quantized output (Q @ X). Reports per-layer and average for all three methods.

### 12. Visualization

Two-panel figure: (left) perplexity bar chart for FP16, RTN, GPTQ, AWQ; (right) per-layer reconstruction MSE grouped bar chart for all three methods.

### 13. Salient Channel Analysis

Visualizes activation magnitude distribution for a representative layer. Shows that a small fraction of channels (top 1%) account for a disproportionate share of total activation magnitude, explaining why AWQ's activation-aware scaling is effective.

### 14. Weight Distribution Analysis

Histograms of weight values for one representative layer under FP16, RTN, GPTQ, and AWQ, showing how each method affects the weight distribution.

### 15. Results Summary

Prints all key numbers: perplexity for each method, perplexity increase, reconstruction MSE, and AWQ alpha values.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `TinyGPT` | 4-layer GPT-style transformer (~844K params) |
| `get_batch` | Sample random training windows from tokenised text |
| `quantize_group_rtn` | Group-wise round-to-nearest quantization to n_bits |
| `quantize_layer_gptq` | GPTQ algorithm: column-wise quantization with Cholesky inverse Hessian |
| `awq_search_scale` | Core AWQ: grid search over alpha for per-channel activation-aware scaling |
| `collect_calibration_activations` | Collect input activations for all Linear layers via forward hooks |
| `apply_rtn_quantization` | Apply group-wise RTN to all Linear layers |
| `apply_gptq_quantization` | Apply GPTQ to all Linear layers with calibration data |
| `apply_awq_quantization` | Apply AWQ to all Linear layers, return per-layer alpha |
| `compute_perplexity` | Evaluate validation perplexity with quantized weights |
| `compute_reconstruction_error` | Compute MSE between fp16 and quantized layer outputs |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | OPT-6.7B, LLaMA-7B, Llama-2-70B | TinyGPT: 4 layers, d_model 128, ~844K params |
| Calibration data | 128 sequences from Pile | 128 batches × 128-token Wikitext-2 windows |
| Quantization bits | INT4, INT3, INT2 | INT4 only |
| Quantization grid | Group-wise asymmetric | Group-wise symmetric absmax |
| GPU kernels | Custom CUDA dequantization + TinyChat | Float32 simulation (no int4 hardware) |
| Benchmarks | WikiText2, PTB, MBPP, GSM8K, Vicuna, VLMs | Validation perplexity only |
| Speedup measurement | 2.7–3.9x on 4090, up to 1.7x over llama.cpp | Not measured (simulation is slower) |
| Alpha search | Grid over [0, 1] | Grid over {0, 0.25, 0.5, 0.75, 1.0} |
| AWQ + GPTQ combined | Table 10 (INT2 results) | Implemented but not evaluated in main results |
| Optimizer / epochs | Large-scale pretraining | 20 epochs, AdamW lr 1e-3 |
