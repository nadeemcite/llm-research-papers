---
title: "Kimi Linear: Code architecture"
parent: "021 Kimi Linear"
nav_order: 1
---
# Code Architecture — Kimi Linear Notebook

## Overview

The notebook implements Kimi Delta Attention (KDA) from scratch in pure PyTorch, builds a 3:1 hybrid model (3 KDA layers per 1 full-attention layer), and compares it against pure full attention and pure KDA on synthetic multi-query associative recall (MQAR). It also measures the KV cache size for each architecture to demonstrate the memory savings. The hybrid model and pure full attention both achieve 100% validation accuracy, while pure KDA reaches only 26% — consistent with the paper's finding that pure linear attention struggles with retrieval. The KV cache reduction is 70.3% at sequence length 128, approaching the paper's claimed 75%.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `D_MODEL = 64` — hidden dimension (paper: 2304)
- `N_HEADS = 4` — number of attention heads (paper: 32 for MLA)
- `D_HEAD = 16` — head dimension (paper: 128)
- `N_BLOCKS = 4` — hybrid blocks, each with 3 KDA + 1 full-attn = 16 total layers (paper: 27 layers)
- `EPOCHS = 10`, `LR = 1e-3`, `BATCH_SIZE = 32`
- `VOCAB_SIZE = 16`, `SEQ_LEN = 64`, `N_KV_PAIRS = 4`
- `N_TRAIN = 2000`, `N_VAL = 200`

### 2. Data — Multi-Query Associative Recall (MQAR)

**Function: `generate_mqar_batch`** — generates (batch, seq_len) token sequences with key-value pairs followed by queries. Returns x (input tokens) and y (target tokens, -100 at non-query positions for loss masking).

The paper tests KDA on palindrome, MQAR, and stack tasks (Section 5.1). We use MQAR because it directly tests whether the model's recurrent state can store and retrieve key-value associations — the core capability that fine-grained gating is designed to improve.

### 3. KDA — Recurrent Form

**Function: `kda_recurrent`** — step-by-step implementation of Eq. 1:

    S_t = (I - beta_t * k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t * k_t v_t^T

Key difference from Gated DeltaNet: `alpha_t` is a vector of shape (d_k,) applied as `Diag(alpha_t)`, not a scalar. Each feature dimension of the key state has its own forgetting rate. `beta_t` remains a scalar.

**Data flow:**
```
Q,K,V: (B, H, T, d_k) or (B, H, T, d_v)
alpha: (B, H, T, d_k)  -- per-channel gate
beta:  (B, H, T)        -- scalar write strength
S: (B, H, d_k, d_v)     -- matrix-valued state
O: (B, H, T, d_v)       -- output = S^T @ q
```

### 4. KDA — Chunkwise Parallel Form

**Function: `kda_chunkwise`** — parallel implementation using the WY representation and UT transform (Section 3.1). Splits the sequence into chunks of size C, computes intra-chunk outputs via matmuls, and propagates inter-chunk states via a short recurrence. A consistency check verifies that the chunkwise form matches the recurrent form (max difference: 0.00000000).

### 5. Fine-Grained Gating Demonstration

A demonstration cell showing how channel-wise gating differs from scalar gating. With four channels having different decay rates (0.1, 0.5, 0.7, 0.9), the fine-grained model produces different outputs (L2 norm 0.5517) than the scalar-gated model (L2 norm 0.6087), demonstrating that per-channel control enables more expressive memory management.

### 6. KDA Layer (Learnable)

**Class: `KDALayer`** — wraps the KDA recurrence in a learnable token-mixer:
- Q/K/V projections + SiLU + L2 normalization on Q and K
- alpha via linear projection + softplus (ensures decay in (0,1))
- beta via sigmoid (write strength in (0,1))
- Output: W_o(Sigmoid(gate) * RMSNorm(KDA(q,k,v,alpha,beta)))

### 7. Full Attention Layer

**Class: `FullAttentionLayer`** — standard multi-head causal attention with NoPE (no positional encoding). This is the hybrid partner, analogous to the paper's MLA layers.

### 8. Hybrid Model

**Class: `LanguageModel`** — stacks N_BLOCKS, each containing 3 KDALayers + 1 FullAttentionLayer (3:1 ratio). Includes embedding, SwiGLU MLP blocks, and output projection. The `mixer_type` parameter controls the architecture: 'hybrid' (3:1), 'full_attn' (all full attention), or 'kda_only' (all KDA).

The `count_kv_cache` method estimates KV cache size: full attention layers need O(seq_len) cache, while KDA layers need O(1) cache (fixed state).

### 9. Loss Function

Cross-entropy loss with `ignore_index=-100` at non-query positions. The model only receives a loss signal at query positions where it must recall the correct value.

### 10. Training Loop

All three models trained identically: AdamW (lr=1e-3, weight_decay=0.01), cross-entropy on query positions, gradient clipping at 1.0, 10 epochs. Validation accuracy computed at query positions only.

### 11. Training Curves

Plots training loss and validation accuracy for all three models.

### 12. Final Evaluation

Prints final accuracy, parameter count, and training time for each model. Random baseline is 1/16 = 0.0625.

### 13. Accuracy Bar Chart

Bar chart comparing final validation accuracy across the three models.

### 14. KV Cache Comparison

Measures the KV cache size for each architecture at various sequence lengths (128, 256, 512, 1024, 2048, 4096). Full attention requires O(T) cache (grows with sequence), while KDA requires O(1) cache (fixed state). The hybrid requires cache proportional to the fraction of full-attention layers (1/4 in the 3:1 ratio).

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_mqar_batch` | Generate synthetic MQAR sequences with key-value pairs and queries |
| `kda_recurrent` | Step-by-step KDA recurrence (Eq. 1) with channel-wise gating |
| `kda_chunkwise` | Parallel chunkwise KDA using WY representation and UT transform |
| `KDALayer` | Learnable KDA token-mixer with projections, SiLU, L2Norm, RMSNorm, output gate |
| `FullAttentionLayer` | Standard causal multi-head attention with NoPE |
| `LanguageModel` | Configurable LM with hybrid, full-attn, or KDA-only mixer types |
| `train_model` | Training loop with AdamW, grad clipping, validation |
| `count_kv_cache` | Estimate KV cache size for each architecture |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 48B total / 3B activated, 27 layers | ~1.16M params, 16 layers |
| Attention partner | Multi-Head Latent Attention (MLA) | Standard multi-head attention |
| FFN | MoE with 256 experts, top-8 routing | Dense SwiGLU MLP |
| Training data | 1.4T tokens (K2 corpus) | 2000 synthetic MQAR examples |
| Optimizer | MuonClip with WSD schedule | AdamW with constant lr |
| Position encoding | NoPE on MLA + implicit in KDA | NoPE on full-attn + implicit in KDA |
| Gate parameterization | Low-rank projection for alpha | Direct linear projection |
| Kernel | Custom CUDA/Triton chunkwise | Pure PyTorch recurrent |
| Tasks | Palindrome, MQAR, Stack + 20+ benchmarks | MQAR only |
