---
title: "DeepSeek-V3: Code architecture"
parent: "035 DeepSeek-V3"
nav_order: 1
---
# Code Architecture — DeepSeek-V3 Notebook

## Overview

The notebook implements a 4-layer Mixture-of-Experts (MoE) transformer with 8 routed experts and 1 shared expert, trained on Tiny Shakespeare (char-level, 128-token context, 15 epochs). It compares three load-balancing strategies — no balancing, conventional auxiliary loss, and DeepSeek-V3's auxiliary-loss-free bias tuning — measuring expert usage uniformity (coefficient of variation) and validation perplexity. The aux-free strategy achieves balanced expert load comparable to aux-loss without the performance penalty.

## Section-by-Section Breakdown

### 1. Setup & Imports

Standard PyTorch + numpy + matplotlib. Seeds set to 42. Device auto-detected (`cuda` on Kaggle T4, `cpu` locally).
Key constants:
- `EPOCHS = 15`, `BATCH_SIZE = 32`, `LR = 3e-4`, `CTX_LEN = 128`
- `N_EXPERTS = 8`, `N_SHARED = 1`, `TOP_K = 2`, `N_LAYERS = 4`, `D_MODEL = 128`, `N_HEADS = 2`, `D_FF = 512`
- `BIAS_UPDATE_SPEED = 0.001`, `AUX_LOSS_WEIGHT = 0.01`

### 2. Data — Tiny Shakespeare

Downloads Tiny Shakespeare (~1 MB, 65 chars) via `urllib.request`. Encodes to integer tensor, splits 90/10 train/val.

**Shapes:** `data: (1,115,394,)`, `train_data: (1,003,854,)`, `val_data: (111,540,)`

`get_batch(split, batch_size, ctx_len)` returns `(x, y)` each of shape `(B, T)`.

### 3. MoE Layer with Load Balancing Strategies

**Class: `Expert`** — A 2-layer MLP with SwiGLU gating: `w2(silu(w1(x)) * w3(x))`. Each expert is a feed-forward network.

**Class: `MoELayer`** — The core implementation. Configurable via `strategy` parameter:
- `'none'`: vanilla top-K routing, no balancing
- `'aux_loss'`: conventional auxiliary loss = α · Σ f_i · P_i · N_r (Fedus et al., 2021)
- `'aux_free'`: DeepSeek-V3's bias-tuning approach

Key components:
- **Routing (Eq. 16):** `routing_scores = affinities + expert_bias` (aux_free only). Top-K selected from routing_scores.
- **Gating:** For aux_free, gating values use original affinities (not bias), normalized among selected. For others, top-K values normalized.
- **Bias buffer:** `expert_bias` is a registered buffer (not a Parameter), updated by `update_bias()` after each step.
- **`update_bias()`:** `expert_bias -= bias_update_speed * sign(counts - avg_count)`. Overloaded experts get negative bias, underloaded get positive.

**Data flow (MoELayer, single forward):**
```
Input: (B, T, D_MODEL)
  -> flatten to (B*T, D)
  -> gate(flat) -> affinities: (B*T, N_EXPERTS)  [sigmoid]
  -> routing_scores = affinities + bias  [aux_free only]
  -> topk(routing_scores, K) -> indices, values: (B*T, TOP_K)
  -> for each expert i: mask = (indices == i), output[mask] += gate * expert_i(flat[mask])
  -> + shared_experts(flat)  [always active]
  -> output: (B, T, D_MODEL), aux_loss: scalar
```

### 4. Transformer Block & Language Model

**Class: `CausalSelfAttention`** — Standard multi-head attention with `F.scaled_dot_product_attention` and causal masking. QKV projection, no bias. Output projection.

**Class: `TransformerBlock`** — Pre-norm (RMSNorm) with residual connections. First layer uses dense FFN (matching paper's first-3-layers-dense pattern). Subsequent layers use MoE.

**Class: `MoELanguageModel`** — Token + positional embeddings, N_LAYERS transformer blocks, final RMSNorm, output head with weight tying to token embedding.

**Data flow (full model):**
```
Input: (B, T)  [token indices]
  -> token_emb + pos_emb -> (B, T, D_MODEL)
  -> Block 0 (dense FFN) -> (B, T, D_MODEL)
  -> Block 1-3 (MoE) -> (B, T, D_MODEL), accumulating aux_loss
  -> norm -> head -> (B, T, VOCAB_SIZE)
```

### 5. Training

`train_model(strategy, label)` — Trains one model with the given strategy. Uses AdamW (lr=3e-4, weight_decay=0.1), gradient clipping at 1.0. 100 batches per epoch. After each optimizer step (for aux_free), calls `model.update_biases()` to adjust expert bias terms.

`estimate_loss(model, eval_iters)` — Estimates validation loss over random batches.

`count_expert_usage(model, n_batches)` — Sums expert token counts across MoE layers over n_batches of validation data.

### 6. Training Curves

Side-by-side plots of training loss and validation loss for all three strategies.

### 7. Expert Usage Distribution

Bar charts of per-expert token counts for each strategy, with mean line and coefficient of variation (CV = std/mean) annotated. Lower CV means more balanced.

### 8. Final Comparison

Prints a table: strategy, val perplexity, expert CV, min/max token counts. Also prints detailed expert usage arrays and balance metrics.

### 9. Bias Term Evolution

Shows the final bias values for each MoE layer in the aux-free model, identifying the most and least favoured experts. Scatter plot of bias values across layers.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `Expert` | Single SwiGLU FFN expert |
| `MoELayer` | MoE layer with configurable load-balancing strategy |
| `MoELayer.update_bias` | Sign-based bias update for aux-free strategy |
| `CausalSelfAttention` | Standard multi-head causal attention |
| `TransformerBlock` | Pre-norm transformer block with MoE or dense FFN |
| `MoELanguageModel` | Full language model with embedding, blocks, output head |
| `train_model` | Train one model with a given strategy, return history + usage |
| `estimate_loss` | Estimate validation loss |
| `count_expert_usage` | Count tokens routed to each expert across batches |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Experts | 256 routed + 1 shared | 8 routed + 1 shared |
| Top-K | 8 | 2 |
| Layers | 61 | 4 |
| d_model | 7168 | 128 |
| Attention | MLA (low-rank KV compression) | Standard MHA |
| Training data | 14.8T tokens | ~1M chars (Tiny Shakespeare) |
| Training objective | Next-token + MTP (depth 1) | Next-token only |
| Precision | FP8 mixed | FP32 |
| Bias update | Monitored per-batch, tuned γ | Sign-based, fixed γ=0.001 |
| Node-limited routing | At most 4 nodes | N/A (single device) |
