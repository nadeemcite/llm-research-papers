---
title: "BitNet b1.58: Code architecture"
parent: "031 BitNet b1.58"
nav_order: 1
---
# Code Architecture — BitNet b1.58 Notebook

## Overview

The notebook implements a tiny 4-layer LLaMA-like transformer where every nn.Linear is replaced by a BitLinear layer with ternary {-1, 0, +1} weights. It trains the ternary model from scratch on word-level Wikitext-2 and compares validation perplexity against an equivalent FP16 model with the same architecture. The ternary model achieves validation perplexity of 25.05 vs the FP16 baseline's 3.72 — a significant gap expected at this tiny scale (~709K params, the paper shows matching at 3B+). The linear-layer weights use 10.1× less memory (1.58 bits vs 16 bits per weight).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `EPOCHS = 20`, `LR = 1e-3`, `BATCH_SIZE = 64`
- `SEQ_LEN = 128`, `N_LAYERS = 4`, `D_MODEL = 128`, `N_HEADS = 4`
- `DEVICE = cuda` (on Kaggle) / `cpu` (local fallback)

### 2. Data

**Function: `get_batch(data, batch_size, seq_len, device)`** — samples random windows from the word-tokenised Wikitext-2 text, returns (x, y) tensors of shape (batch, seq_len). Uses word-level tokenisation (vocab ~283 words). ~10.8M train tokens, ~1.1M valid tokens.

### 3. BitLinear Layer

**Class: `BitLinear`** — the core component replacing nn.Linear. During the forward pass:
1. Quantize weights to ternary {-1, 0, +1} using absmean quantization: W̃ = RoundClip(W / (γ + ε), -1, +1) where γ = mean(\|W\|)
2. Quantize activations to 8-bit per token: scale to [-128, 127], round
3. Compute matmul with quantized weights and activations (simulated in float32)
4. Dequantize output back to float

**Weight quantization function: `absmean_quantize(W)`**
- Computes γ = mean(\|W\|) over all elements
- Scales: W_scaled = W / (γ + ε)
- Rounds and clips to {-1, 0, +1}
- Returns ternary weight tensor

**Activation quantization function: `activation_quantize(x, bits=8)`**
- Computes per-token absmax (max absolute value per row)
- Scales to [-Qb, Qb] where Qb = 2^(bits-1) = 128
- Rounds to nearest integer
- Returns quantized activation tensor

**Data flow (BitLinear):**
```
Input: x (batch, in_features)
  -> activation_quantize(x) -> x_q (batch, in_features)
  -> absmean_quantize(W) -> W_q (out_features, in_features) with values in {-1, 0, 1}
  -> output = x_q @ W_q.T -> (batch, out_features)
```

### 4. Model

**Class: `TinyBitNet`** — LLaMA-like transformer with BitLinear layers:
- Token + positional embeddings (max_seq_len = SEQ_LEN + 1)
- 4 × `TransformerBlock`: RMSNorm → BitLinear Q/K/V → RoPE → causal attention → BitLinear O → residual → RMSNorm → SwiGLU FFN (BitLinear up, BitLinear down) → residual
- RMSNorm → BitLinear head (weight-tied with token embedding)

**Class: `TinyLLaMA`** — identical architecture but with standard nn.Linear instead of BitLinear, for comparison.

**Data flow:**
```
Input: (batch, seq_len) int64 token IDs
  -> embeddings -> (batch, seq_len, 128)
  -> 4x TransformerBlock (with BitLinear) -> (batch, seq_len, 128)
  -> RMSNorm -> head -> (batch, seq_len, vocab_size)
```

### 5. Loss Function

Standard cross-entropy loss on next-token prediction. Optimizer: AdamW (lr 1e-3, weight decay 0.01, grad clip 1.0).

### 6. Training Loop

- 20 epochs, 200 batches per epoch, batch_size 64, seq_len 128
- Validation every epoch (50 batches), logs train/val loss and val perplexity
- Trains both BitNet (ternary) and FP16 models with identical hyperparameters
- Reports final validation perplexity for both

### 7. Training Curves

Plots training & validation loss (left) and validation perplexity (right) for both the ternary BitNet model and the FP16 baseline on the same axes for comparison.

### 8. Weight Distribution Analysis

Visualizes the weight distribution of a representative BitLinear layer: (left) the original FP16 weights, (right) the ternary quantized weights showing the {-1, 0, +1} distribution with the majority at 0.

### 9. Memory Comparison

Computes and compares the theoretical memory footprint of linear-layer weights for the ternary model (1.58 bits/weight) vs the FP16 model (16 bits/weight), showing the ~10× reduction.

### 10. Results Summary

Prints all key numbers: perplexity for both models, perplexity difference, memory savings ratio, and the fraction of weights at each ternary value {-1, 0, +1}.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `BitLinear` | Linear layer with ternary {-1, 0, +1} weights via absmean quantization and 8-bit activations |
| `absmean_quantize` | Quantize a weight matrix to ternary values using the absmean function (Eq. 1–3) |
| `activation_quantize` | Per-token activation quantization to 8-bit integers (no zero-point) |
| `TinyBitNet` | 4-layer LLaMA-like transformer using BitLinear (~709K params) |
| `TinyLLaMA` | 4-layer LLaMA-like transformer with standard nn.Linear for comparison |
| `RMSNorm` | Root mean square layer normalization (as in the paper) |
| `get_batch` | Sample random training windows from tokenised text |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | 700M – 3.9B parameters | TinyBitNet: 4 layers, d_model 128, ~709K params |
| Training data | 100B tokens of RedPajama | ~10.8M tokens of word-level Wikitext-2 |
| Training tokens | 100B – 2T | 20 epochs × 100 batches × 64 × 128 ≈ 164M tokens total |
| Weight quantization | True ternary with custom kernels | Simulated in float32 (no int8/ternary hardware) |
| Activation quantization | INT8 with custom kernels | Simulated in float32 |
| Architecture | Full LLaMA scale | 4-layer toy with same components (RMSNorm, SwiGLU, RoPE) |
| Benchmarks | WikiText2, C4, ARC, HellaSwag, BoolQ, PIQA, OpenBookQA, Winogrande | Validation perplexity only |
| Memory/latency/energy | Measured with FasterTransformer + 2-bit Ladder kernels | Theoretical memory comparison only |
| Optimizer / epochs | Large-scale pretraining schedule | 20 epochs, AdamW lr 1e-3 |
