---
title: "MLA: Code architecture"
parent: "009 MLA"
nav_order: 1
---
# Code Architecture — MLA Notebook

## Overview

The notebook implements three attention mechanisms — standard Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Multi-head Latent Attention (MLA) — in three small PyTorch transformers, trains them on Tiny Shakespeare (character-level, 128-token context, 8 epochs), and compares KV-cache size and validation perplexity. The MLA result: val perplexity 7.94 with 576 elements/token KV cache, vs MHA's 7.61 with 1024 elements (43.8% cache reduction).

## Section-by-Section Breakdown

### 1. Setup & Imports

Standard PyTorch + numpy + matplotlib. Seeds set to 42. Device auto-detected (`cuda` on Kaggle T4, `cpu` locally).

### 2. Configuration

Key constants copied from the code cell:
- `N_LAYERS = 4`, `D_MODEL = 128`, `N_HEADS = 4`, `D_HEAD = 32`
- `D_C = 128` — KV compression dim (= 4 x d_h, matching paper ratio)
- `D_C_Q = 128` — Query compression dim
- `D_H_R = 16` — Decoupled RoPE per-head dim (= d_h / 2, matching paper ratio)
- `N_GROUPS = 2` — GQA groups
- `EPOCHS = 8`, `LR = 3e-4`, `BATCH_SIZE = 32`, `TRAIN_CTX = 128`, `STEPS_PER_EPOCH = 50`

### 3. Data — Tiny Shakespeare

Downloads the Tiny Shakespeare corpus (~1 MB, 65 characters) via `urllib.request`. Encodes to integer tensor, splits 90/10 train/val.

**Shapes:** `data: (1,115,394,)`, `train_data: (1,003,854,)`, `val_data: (111,540,)`

### 4. Attention Mechanisms

**`precompute_rope(head_dim, seq_len)`** — Returns `(cos, sin)` tables of shape `(seq_len, head_dim//2)`.

**`apply_rope(x, cos, sin)`** — Applies RoPE rotation to `x` of shape `(B, n_heads, T, d)`. Splits last dim into pairs, rotates using cos/sin tables.

**Class: `MHAAttention`** — Standard multi-head attention. Projects Q, K, V each to `n_heads * d_head`. Uses `F.scaled_dot_product_attention` with causal mask. KV cache size: `2 * n_heads * d_head = 256` per layer.

**Class: `GQAAttention`** — Projects K, V to `n_groups * d_head` (shared across heads). Q still `n_heads * d_head`. K, V expanded via `repeat_interleave`. KV cache size: `2 * n_groups * d_head = 128` per layer.

**Class: `MLAAttention`** — The core implementation. Key components:
- Query path: `c_Q = W_DQ(h)` -> `q_C = W_UQ(c_Q)`, `q_R = RoPE(W_QR(c_Q))`
- KV path: `c_KV = W_DKV(h)` -> `k_C = W_UK(c_KV)`, `v_C = W_UV(c_KV)`, `k_R = RoPE(W_KR(h))`
- Concatenation: `q = [q_C; q_R]`, `k = [k_C; k_R]`
- Attention: `F.scaled_dot_product_attention(q, k, v_C, scale=1/sqrt(d_head + d_h_R))`
- KV cache size: `d_c + d_h_R = 128 + 16 = 144` per layer

**Data flow (MLA, single forward):**
```
Input: (B, T, D_MODEL)
  -> W_DQ -> c_Q: (B, T, 128)
     -> W_UQ -> q_C: (B, 4, T, 32)
     -> W_QR -> q_R: (B, 4, T, 16) -> RoPE
  -> W_DKV -> c_KV: (B, T, 128)
     -> W_UK -> k_C: (B, 4, T, 32)
     -> W_UV -> v_C: (B, 4, T, 32)
  -> W_KR -> k_R: (B, 1, T, 16) -> RoPE -> expand to (B, 4, T, 16)
  -> q = [q_C; q_R]: (B, 4, T, 48)
  -> k = [k_C; k_R]: (B, 4, T, 48)
  -> attention(q, k, v_C) -> (B, 4, T, 32)
  -> W_O -> (B, T, D_MODEL)
```

### 5. Transformer Block & Language Model

**Class: `TransformerBlock`** — Pre-norm: `x + attn(norm1(x))`, then `x + ffn(norm2(x))`. FFN is a 2-layer MLP with SiLU activation (SwiGLU-style without the gate). Uses `nn.RMSNorm`.

**Class: `LanguageModel`** — Token embedding -> N TransformerBlocks -> RMSNorm -> LM head. Forward takes optional `cos, sin` for RoPE. `kv_cache_size()` sums per-layer cache. `count_params()` for reporting.

### 6. KV Cache Comparison

Instantiates each model temporarily and computes `kv_cache_size()`. Results (all 4 layers):
- MHA: 1024 elements/token
- GQA: 512 elements/token (50% of MHA)
- MLA: 576 elements/token (56.2% of MHA)

Also prints the paper's reference numbers for comparison.

### 7. Training

**`train_model(attn_type)`** — Creates model, trains with AdamW (lr=3e-4, wd=0.1, betas=(0.9, 0.95)), gradient clipping at 1.0. Logs every 2 epochs.

All three models trained identically. Final results (Kaggle T4):
- MHA: 0.80M params, val_ppl=7.61
- GQA: 0.74M params, val_ppl=7.81
- MLA: 0.98M params, val_ppl=7.94

### 8. Training Curves

Side-by-side plot of train and validation loss for all three mechanisms. Saved as `training_curves.png`.

### 9. Final Comparison

Tabular comparison of val perplexity, val loss, KV cache size, and cache ratio vs MHA.

### 10. KV Cache Size Bar Chart

Bar chart comparing KV cache elements per token across MHA, GQA, MLA. Saved as `kv_cache_comparison.png`.

### 11. Text Generation Comparison

Generates 120 characters from each model with prompt "ROMEO:\n" at temperature 0.8, top-k 40. All three produce Shakespeare-like gibberish as expected for a 1M-parameter character model.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `precompute_rope` | Precompute cos/sin tables for RoPE |
| `apply_rope` | Apply RoPE rotation to a tensor |
| `MHAAttention` | Standard multi-head attention |
| `GQAAttention` | Grouped-query attention with shared KV heads |
| `MLAAttention` | Multi-head latent attention with low-rank KV compression and decoupled RoPE |
| `TransformerBlock` | Pre-norm transformer block with attention + FFN |
| `LanguageModel` | Full language model with embedding, blocks, LM head |
| `train_model` | Training loop for one attention variant |
| `generate` | Top-k text generation from a trained model |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Scale | 236B total / 21B activated params | ~0.8-1.0M params |
| Attention heads | 128 heads, d_h=128 | 4 heads, d_h=32 |
| KV compression | d_c=512, d_h_R=64 | d_c=128, d_h_R=16 (same ratios) |
| FFN | DeepSeekMoE (160 routed + 2 shared experts) | Single dense FFN with SiLU |
| Training data | 8.1T tokens, multi-source corpus | Tiny Shakespeare (~1 MB), 8 epochs x 50 steps |
| Weight absorption | Implemented for inference efficiency | Not implemented (training-focused) |
| Position encoding | Decoupled RoPE (full) | Decoupled RoPE (same structure, smaller dims) |
| Evaluation | MMLU, BBH, GSM8K, etc. | Validation perplexity on Tiny Shakespeare |
