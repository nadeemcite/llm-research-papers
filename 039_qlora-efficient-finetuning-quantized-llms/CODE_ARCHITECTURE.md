---
title: "QLoRA: Code architecture"
parent: "039 QLoRA"
nav_order: 1
---
# Code Architecture — QLoRA Notebook

## Overview

The notebook implements QLoRA's three core innovations from scratch in PyTorch: (1) the NF4 data type (16 quantile values of N(0,1)), (2) block-wise NF4 quantization with Double Quantization, and (3) LoRA fine-tuning on a frozen 4-bit quantized model. It pre-trains a tiny 4-layer GPT-style transformer (~3.3M params) on character-level Wikitext-2, quantizes it to 4-bit NF4, fine-tunes with LoRA rank 8 for 10 epochs, and compares against full 16-bit fine-tuning. QLoRA achieves validation perplexity of 3.82 vs. full fine-tuning's 3.64, using only 7.6% trainable parameters and 11.5% of the memory.

## Section-by-Section Breakdown

### 1. Setup & Imports

Sets random seeds (SEED=42) and defines all hyperparameters in one cell:
- `EPOCHS = 15` — base model pre-training epochs
- `QLORA_EPOCHS = 10` — QLoRA fine-tuning epochs
- `FULL_FT_EPOCHS = 10` — full 16-bit fine-tuning epochs (comparison)
- `BATCH_SIZE = 64`, `SEQ_LEN = 128`, `LR = 1e-3`
- `LORA_R = 8` — LoRA rank (paper used 64; scaled down for tiny model)
- `LORA_ALPHA = 16` — LoRA scaling factor
- `N_BITS = 4` — NF4 quantization bits
- `BLOCK_SIZE = 64` — block size for quantization (same as paper)
- `DQ_BLOCK_SIZE = 256` — block size for double quantization (same as paper)

### 2. Data

Downloads Wikitext-2 (train + valid) from the pytorch/examples GitHub repo to `/tmp/wikitext2`. Tokenizes at character level, building a vocabulary of 283 characters. Train: 10,780,437 tokens, Valid: 1,120,192 tokens.

**Function: `get_batch(data, batch_size, seq_len, device)`** — samples random offset positions, returns input/target tensors of shape `(batch_size, seq_len)` where target is shifted by 1 position.

### 3. Base Model

**Class: `TinyGPT`** (3,260,928 parameters)
- 4-layer decoder-only transformer, d_model=256, 4 heads, d_ff=1024, max_seq_len=129
- Explicit Wq/Wk/Wv/Wo projections per layer (not fused QKV)
- Causal self-attention with pre-LayerNorm, GELU feed-forward
- Token + positional embeddings, weight tying (head.weight = tok_emb.weight)

**Class: `TransformerBlock`** — one transformer layer with:
- LayerNorm → Q/K/V projections → scaled dot-product attention (causal mask) → output projection → residual
- LayerNorm → FC1 (GELU) → FC2 → residual

**Data flow:**
```
Input: (batch=64, seq_len=128)
  -> tok_emb + pos_emb -> (64, 128, 256)
  -> 4x TransformerBlock -> (64, 128, 256)
  -> LayerNorm -> head -> (64, 128, 283)
```

### 4. Loss Function

Standard cross-entropy loss on next-token prediction: `nn.CrossEntropyLoss()`.
Optimizer: AdamW (lr=1e-3, weight_decay=0.01).

**Function: `compute_perplexity(model, data, n_batches, device)`** — evaluates model on n_batches of random samples, returns `exp(average_loss)`.

### 5. Pre-training the Base Model

Trains the base model from scratch for 15 epochs (200 batches/epoch). Logs loss and validation perplexity every 5 epochs. Gradient clipping at 1.0. Pre-training reaches val perplexity ~4.04.

### 6. Pre-training Curves

Two subplots: training loss (decreasing) and validation perplexity (decreasing) over 15 epochs.

### 7. NF4 Quantization

**Function: `compute_nf4_quantiles(n_bits=4)`** — returns the 16 NF4 values as a tensor. These are the quantile midpoints of N(0,1) normalized to [-1, 1], matching the bitsandbytes reference implementation. Verified against theoretical quantiles computed from `scipy.stats.norm.ppf`.

Output: 16 values from -1.0 to 1.0, confirmed to match theoretical quantile midpoints.

### 8. Block-wise NF4 Quantization

**Function: `nf4_quantize_tensor(W, nf4_values, block_size=64)`** — flattens weight tensor, pads to block_size multiples, computes per-block absmax scale, normalizes each block to [-1, 1], finds nearest NF4 bin index. Returns `(quantized_indices, scales)`.

**Function: `nf4_dequantize_tensor(...)`** — reverse: lookup NF4 values by index, multiply by per-block scale. Returns dequantized FP32 tensor.

Test on 128×128 random matrix: MSE 0.008549, range preserved exactly.

### 9. Double Quantization

**Function: `double_quantize_scales(scales_fp32, dq_block_size=256)`** — subtracts mean, quantizes centered FP32 scales to int8 in blocks of 256 with per-block FP32 second-level scale. Returns `(dq_indices, dq_scales, mean)`.

**Function: `double_dequantize_scales(...)`** — reverses: multiply int8 indices by second-level scales, add mean.

Test: 256 FP32 scales (8192 bits) → 256 int8 (2048 bits) + 1 FP32 (32 bits). Scale recovery MSE: 0.0000085. Overhead: 0.127 bits/param vs. 0.500 without DQ.

### 10. Apply NF4 to Pre-trained Model

**Function: `quantize_model_nf4(model, ...)`** — deep-copies the model, quantizes all 2D weight parameters (excluding embeddings/head) to NF4 with Double Quantization, replaces weights with dequantized approximations. Returns `(model_q, quant_data)`.

Reports per-layer MSE and relative error. Overall relative quantization error ~0.009.

### 11. LoRA Adapters

**Class: `LoRALinear`** — wraps a frozen Linear layer with trainable LoRA adapters:
- Base weight: `requires_grad=False` (frozen)
- LoRA A: `(r, in_features)` initialized with small random values
- LoRA B: `(out_features, r)` initialized to zeros
- Forward: `base(x) + (x @ A^T) @ B^T * (alpha/r)`

**Function: `add_lora_to_model(model, r, alpha)`** — replaces all 6 linear layers in each transformer block (Wq, Wk, Wv, Wo, FC1, FC2) with LoRA-wrapped versions.

Result: 3,408,384 total params, 257,536 trainable (7.6%), 3,150,848 frozen (92.4%).

### 12. QLoRA Fine-tuning

Fine-tunes the 4-bit quantized model with LoRA adapters for 10 epochs (200 batches/epoch). Only LoRA parameters are optimized. Reaches val perplexity 3.82.

### 13. Full 16-bit Fine-tuning (Baseline)

Deep-copies the original FP model, trains all parameters for 10 epochs. Reaches val perplexity 3.64.

### 14. QLoRA vs Full Fine-tuning Comparison

Two subplots comparing training loss and validation perplexity between QLoRA and full fine-tuning. QLoRA tracks full fine-tuning closely, with a small gap (0.18 perplexity points).

### 15. Memory Footprint Analysis

**Function: `estimate_memory(total_params, trainable_params, ...)`** — computes theoretical memory for weights (frozen at n_bits, trainable at 16-bit), quantization overhead, gradients (trainable only), and optimizer states (AdamW: 2×32-bit per trainable param).

Results: Full FT 39,942 KB, 16-bit LoRA 9,364 KB, QLoRA 4,605 KB. QLoRA = 11.5% of full FT memory. Scaled to 65B: full FT ~390 GB, QLoRA ~33 GB (fits on a 48 GB GPU).

### 16. NF4 vs FP4 Comparison

**Function: `compute_fp4_quantiles()`** — computes FP4 E2M1 format (1 sign + 2 exponent + 1 mantissa) values.

Compares reconstruction error on all model weights: NF4 MSE 0.000022, FP4 MSE 0.000031. NF4 improvement: 30.6%.

### 17. Weight Distribution Visualization

Left plot: histogram of model weights (normally distributed) with NF4 bin positions overlaid (red lines). Right plot: NF4 vs FP4 quantization levels as bar charts, showing NF4's denser bins near zero where most weights cluster.

### 18. Results Summary

Prints all key metrics: base perplexity 4.04, QLoRA perplexity 3.82, full FT perplexity 3.64, trainable params 257,536 (7.6%), NF4 MSE, FP4 MSE, NF4 improvement 30.6%, DQ overhead reduction, memory savings 88.5%.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `TinyGPT` | 4-layer GPT-style transformer base model (~3.3M params) |
| `TransformerBlock` | Single transformer layer with explicit Q/K/V/O projections |
| `compute_nf4_quantiles` | Compute the 16 NF4 data type values from N(0,1) quantiles |
| `nf4_quantize_tensor` | Block-wise NF4 quantization of a weight tensor |
| `nf4_dequantize_tensor` | Reverse NF4 quantization back to FP32 |
| `double_quantize_scales` | Compress FP32 quantization scales to 8-bit (Double Quantization) |
| `double_dequantize_scales` | Recover approximate FP32 scales from 8-bit |
| `quantize_model_nf4` | Quantize all linear weights in a model to NF4 with DQ |
| `LoRALinear` | Frozen linear layer + trainable LoRA adapter (A, B matrices) |
| `add_lora_to_model` | Wrap all linear layers in transformer blocks with LoRA |
| `compute_perplexity` | Evaluate model perplexity on random batches |
| `estimate_memory` | Theoretical memory estimate for weights + grads + optimizer |
| `compute_fp4_quantiles` | Compute FP4 E2M1 data type for comparison |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 65B LLaMA | ~3.3M params (4-layer GPT) |
| Data | Instruction datasets (OASST1, FLAN v2, Alpaca, etc.) | Character-level Wikitext-2 (~2M tokens) |
| LoRA rank | 64 | 8 |
| Quantization kernels | Custom CUDA 4-bit kernels | Pure PyTorch (dequantize to FP32 for forward) |
| Paged Optimizers | NVIDIA unified memory for spike management | Not implemented |
| Evaluation | Vicuna benchmark, GPT-4 + human eval | Validation perplexity on Wikitext-2 |
| Scale of study | 1,000+ models across 8 datasets | Single model, single dataset |
