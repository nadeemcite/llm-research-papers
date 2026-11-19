---
title: "DeepSeekMoE: Code architecture"
parent: "084 DeepSeekMoE"
nav_order: 1
---
# Code Architecture — DeepSeekMoE Notebook

## Overview

The notebook implements three small PyTorch language models — a dense transformer baseline, a GShard MoE (8 experts, top-2), and a DeepSeekMoE (32 fine-grained experts, 1 shared + top-7 routed) — and trains them on Tiny Shakespeare (character-level, 128-token context, 20 epochs). All three share the same multi-head attention architecture. GShard and DeepSeekMoE have identical total and activated expert parameters. After training, the notebook replicates the paper's expert specialization analysis (Section 4.5): disabling top routed experts to measure redundancy, and measuring routing entropy to compare expert specialization.

## Section-by-Section Breakdown

### 1. Setup & Imports

Standard PyTorch + numpy + matplotlib. Seeds set to 42. Device auto-detected (`cuda` on Kaggle T4, `cpu` locally).

Key constants copied from the code cell:
- `N_LAYERS = 3`, `HIDDEN_DIM = 256`, `N_HEADS = 4`, `SEQ_LEN = 128`
- GShard: `n_experts = 8`, `top_k = 2`, `ffn_mult = 4`
- DeepSeekMoE: `n_experts = 8`, `top_k = 2`, `n_shared = 1`, `granularity = 4`, `ffn_mult = 4`
  (yields 32 total experts: 1 shared + 31 routed, top-7 routed)
- `EPOCHS = 20`, `LR = 1e-3`, `BATCH_SIZE = 64`, `BATCHES_PER_EPOCH = 200`
- Balance loss alpha = 0.01

### 2. Data — Tiny Shakespeare

Downloads the Tiny Shakespeare corpus (~1 MB, 65 characters) via `urllib.request` to `/tmp`. Encodes to integer tensor, splits 90/10 train/val.

**Shapes:** `data: (1,115,394,)`, `train_data: (~1,003,854,)`, `val_data: (~111,540,)`

### 3. Model Components

**Class: `MultiHeadAttention`** — Standard multi-head self-attention with 4 heads (head_dim = 64). Q/K/V projected via a single `Linear(3*dim)`. Causal masking via `F.scaled_dot_product_attention(is_causal=True)`.

**Class: `Expert`** — A single FFN block: two linear layers with GELU activation (`w2(gelu(w1(x)))`). Used as both the dense FFN and as individual experts in MoE layers. For DeepSeekMoE, expert hidden dim is `(dim * ffn_mult) // granularity` (256 instead of 1024).

**Class: `MoELayer`** — Conventional GShard-style MoE (Section 2, Eq. 3-5):
- Router: linear `dim -> n_experts` (8 logits)
- Top-K selection: `torch.topk(scores, 2)` → top-2 scores and indices
- Gating: `F.softmax` over selected experts, then renormalized
- Expert computation: loop over experts, mask tokens routed to each, apply expert, weight by gating scores
- Tracks `expert_counts` for load balance loss

**Class: `DeepSeekMoELayer`** — Fine-grained segmentation + shared expert isolation (Section 3, Eq. 9-11):
- Shared experts: `Ks=1` always-on Expert (no router), hidden dim = `dim*ffn_mult/granularity` = 256
- Routed experts: `mN - Ks = 31` experts, each 256 hidden dim
- Router: linear `dim -> 31` (routed experts only)
- Top-K routed: `mK - Ks = 7` selected via topk on router logits
- Output = shared_output + routed_output (Eq. 9)
- Tracks `expert_counts` over routed experts for load balance loss

**Class: `TransformerBlock`** — Pre-norm with LayerNorm: `x + attn(ln1(x))`, then `x + ffn(ln2(x))`. FFN type selects between dense, GShard, or DeepSeekMoE.

**Class: `TinyLM`** — Token embedding + positional embedding → N TransformerBlocks → LayerNorm → LM head (no weight tying).

**Data flow (DeepSeekMoE layer, single token):**
```
Input: (B, T, 256)
  -> ln2 -> (B, T, 256)
  -> reshape -> (B*T, 256)
  -> shared experts always active -> shared_out: (B*T, 256)
  -> router (Linear 256->31) -> logits: (B*T, 31)
  -> topk(logits, k=7) -> topk_scores: (B*T, 7), topk_indices: (B*T, 7)
  -> softmax + renormalize -> weights: (B*T, 7)
  -> for each routed expert e in [0..30]:
       mask tokens where e appears in topk_indices
       routed_out[mask] += weight * Expert_e(token)
  -> output = shared_out + routed_out: (B*T, 256) -> reshape -> (B, T, 256)
```

### 4. Parameter Count Comparison

Prints total parameter counts for all three models. Key values from the run:
- Dense: 2,428,928 total parameters
- GShard MoE: 13,445,120 total parameters
- DeepSeekMoE: 13,462,784 total parameters
- GShard combinations C(8,2) = 28; DeepSeekMoE combinations C(31,7) = 2,629,575 (93,913x more flexible)

### 5. Training Loop

- AdamW (lr=1e-3, weight_decay=0.1)
- 20 epochs x 200 batches per epoch, batch 64
- Loss = cross_entropy + 0.01 x balance_loss
- All three models trained identically
- Validation loss evaluated every epoch

### 6. Training Curves

Two side-by-side plots: training loss and validation loss across 20 epochs for Dense, GShard, and DeepSeekMoE.

### 7. Final Evaluation

Computes validation loss over 100 batches for all three models. Prints comparison lines including the improvement of DeepSeekMoE over GShard.

### 8. Expert Specialization Analysis

**8a. Routing entropy (cf. Section 4.5):** Computes per-expert selection frequency across validation data for each layer, then calculates Shannon entropy in bits. Higher entropy = more uniform routing = better specialization (paper's metric). Prints entropy for each layer for both GShard and DeepSeekMoE.

**8b. Expert disabling (cf. Figure 4):** Disables 0%, 6.2%, 12.5%, 18.8%, 25% of top routed experts (by zeroing their output) and measures validation loss. DeepSeekMoE should be less sensitive to disabling (smaller loss increase) due to finer-grained specialization.

**8c. Plot:** Validation loss vs disable ratio for both models, showing DeepSeekMoE's robustness to expert disabling.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `MultiHeadAttention` | Standard multi-head self-attention with 4 heads |
| `Expert` | Single GELU FFN block; used as dense FFN or individual MoE expert |
| `MoELayer` | Conventional GShard MoE with N=8 experts, top-2 routing, balance loss |
| `DeepSeekMoELayer` | Fine-grained (m=4) + shared expert isolation MoE layer |
| `TransformerBlock` | Pre-norm transformer block with attention + FFN/MoE |
| `TinyLM` | Full transformer LM with embedding, N blocks, LM head |
| `get_batch` | Sample random (x, y) batches from train/val data |
| `compute_balance_loss` | Expert-level load balance loss (alpha=0.01) |
| `train_model` | Training loop with AdamW, balance loss, per-epoch eval |
| `evaluate` | Compute average validation loss |
| `get_expert_usage` | Compute per-expert selection frequency across val data |
| `evaluate_with_disabled` | Forward pass with top routed experts disabled |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Dataset | 100B tokens (2B model) / 2T tokens (16B model), multilingual | Tiny Shakespeare (~1 MB, character-level) |
| Model size | 2B / 16.4B total params | ~13.4M total (MoE), ~2.4M active |
| Layers | 9 (2B) / 28 (16B) | 3 |
| Hidden dim | 1280 (2B) / 2048 (16B) | 256 |
| Experts | 1 shared + 63 routed (2B) / 2 shared + 64 routed (16B) | 1 shared + 31 routed |
| Expert size | 0.25x standard FFN | 0.25x standard FFN (same ratio) |
| Optimizer | AdamW, warmup-and-step-decay, lr ~2e-4 to 4.2e-4 | AdamW, fixed lr=1e-3 |
| Training hardware | A100/H800 GPU clusters | Single GPU or CPU |
| Balance loss | Expert-level (alpha1=0.01) + device-level (alpha2) | Expert-level only (0.01) |
| Normalization | RMSNorm | LayerNorm |
| Activation | SwiGLU in experts | GELU in experts |
| Attention | GQA (2:1 grouping) | Standard MHA (no grouping) |
