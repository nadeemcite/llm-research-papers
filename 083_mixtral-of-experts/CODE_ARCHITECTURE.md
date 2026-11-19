---
title: "Mixtral: Code architecture"
parent: "083 Mixtral"
nav_order: 1
---
# Code Architecture — Mixtral Notebook

## Overview

The notebook implements two small PyTorch language models — a dense transformer baseline and a sparse mixture-of-experts (MoE) transformer — and trains them on Tiny Shakespeare (character-level, 128-token context, 15 epochs). The MoE model uses 8 expert SwiGLU feedforward blocks per layer with top-2 routing, replicating the core mechanism from Mixtral 8x7B (Section 2.1). After training, the notebook visualizes expert routing: which expert each token is assigned to, expert frequency distributions, and consecutive-token routing locality (replicating the paper's Table 5 analysis).

## Section-by-Section Breakdown

### 1. Setup & Imports

Standard PyTorch + numpy + matplotlib. Seeds set to 42. Device auto-detected (`cuda` on Kaggle T4, `cpu` locally).

Key constants copied from the code cell:
- `N_LAYERS = 4`, `D_MODEL = 128`, `N_HEADS = 4`, `N_KV_HEADS = 2`, `D_HEAD = 32`
- `N_EXPERTS = 8`, `TOP_K = 2`, `FFN_DIM = 512`
- `EPOCHS = 15`, `LR = 1e-3`, `BATCH_SIZE = 32`, `CONTEXT_LEN = 128`, `STEPS_PER_EPOCH = 80`

### 2. Data — Tiny Shakespeare

Downloads the Tiny Shakespeare corpus (~1 MB, 65 characters) via `urllib.request` to `/tmp`. Encodes to integer tensor, splits 90/10 train/val.

**Shapes:** `data: (1,115,394,)`, `train_data: (1,003,854,)`, `val_data: (111,540,)`

### 3. Model

**Class: `GroupedQueryAttention`** — GQA attention with 4 query heads and 2 KV heads (2:1 grouping ratio). Q projected to `n_heads * d_head`, K/V projected to `n_kv_heads * d_head`. KV heads repeated to match query heads. Causal mask via `F.scaled_dot_product_attention`.

**Class: `SwiGLUExpert`** — A single SwiGLU feedforward block: `w2(silu(w1(x)) * w3(x))`. Used as both the dense FFN and as individual experts in the MoE layer.

**Class: `MoELayer`** — The core MoE implementation (Section 2.1 of the paper):
- Router: linear layer `W_g` mapping `d_model` to `n_experts` logits
- Top-K selection: `torch.topk(logits, top_k)` returns top-2 logits and indices
- Gating weights: `F.softmax(topk_logits)` — softmax over only the 2 selected experts
- Expert computation: for each of the 2 selected experts, compute `SwiGLU_i(x)` for the tokens routed to it, multiply by the gating weight, and add to the output
- Returns: output tensor, routing weights `(B, T, top_k)`, expert indices `(B, T, top_k)`

**Class: `DenseFFN`** — Wrapper around a single `SwiGLUExpert` for the dense baseline.

**Class: `TransformerBlock`** — Pre-norm: `x + attn(norm1(x))`, then `x + ffn(norm2(x))`. Uses `nn.RMSNorm`. Returns routing data when `use_moe=True`.

**Class: `LanguageModel`** — Token embedding → N TransformerBlocks → RMSNorm → LM head. Weight tying between embedding and output head.

**Parameter counts (from printed output):**
- Dense: 992,512 params (1 SwiGLU FFN per layer)
- MoE: 6,501,632 total params (8 SwiGLU experts per layer)
- MoE active: 1,783,040 params per token (top-2 of 8 experts, across 4 layers)
- Single expert: 196,608 params
- Total/active ratio: 3.6x

**Data flow (MoE layer, single token, single forward):**
```
Input: (B, T, 128)
  -> norm2 -> (B, T, 128)
  -> reshape -> (B*T, 128)
  -> router (Linear 128->8) -> logits: (B*T, 8)
  -> topk(logits, k=2) -> topk_logits: (B*T, 2), topk_indices: (B*T, 2)
  -> softmax(topk_logits) -> weights: (B*T, 2)
  -> for each expert e in [0..7]:
       mask = (topk_indices[:, i] == e)  for i in [0, 1]
       expert_out = SwiGLU_e(x_flat[mask])
       output[mask] += weight[mask] * expert_out
  -> output: (B*T, 128) -> reshape -> (B, T, 128)
```

### 4. Loss Function

Standard cross-entropy loss for next-token prediction: `F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))`. Both models use the same loss.

### 5. Training Loop

- AdamW (lr=1e-3, wd=0.1, betas=(0.9, 0.95)), gradient clipping at 1.0
- 15 epochs × 80 steps per epoch, batch 32
- Validation loss evaluated every 5 epochs on 50 batches
- Both models trained identically

### 6. Training Curves

Side-by-side plot of training and validation loss for both models. The MoE model typically starts with higher loss (more parameters to initialize) but converges to comparable or slightly better perplexity (Dense 10.59, MoE 10.27 on the Kaggle T4 run).

### 7. Final Evaluation

Prints final validation loss and perplexity for both models, plus the uniform random baseline (perplexity = vocab_size = 65). Also prints parameter counts and ratios (total/active).

### 8. Expert Routing Visualization

Processes a sample text ("To be, or not to be...") through the trained MoE model and extracts the first-choice expert index for each token at each layer. Displays a heatmap (layers × tokens) with a discrete 8-color colormap, showing which expert each token is routed to.

### 9. Expert Frequency Analysis

Processes 2048 tokens from the validation set and counts first-choice expert assignments per layer. Bar charts show the frequency distribution with a dashed line at 1/8 = 12.5% (uniform expectation). Prints min/max/std statistics per layer.

### 10. Consecutive Token Routing Locality

Replicates the paper's Table 5: measures the proportion of consecutive token pairs that share the same first-choice expert, and the proportion that share any expert (first or second choice). Compares against random baselines (12.5% for first choice, ~46% for first-or-second).

## Key Functions

| Function / Class | Purpose |
|---|---|
| `GroupedQueryAttention` | GQA attention with configurable query/KV head ratio |
| `SwiGLUExpert` | Single SwiGLU feedforward block (expert or dense FFN) |
| `MoELayer` | Sparse MoE layer with top-K routing and weighted expert combination |
| `DenseFFN` | Dense FFN wrapper for baseline comparison |
| `TransformerBlock` | Pre-norm transformer block with attention + FFN/MoE |
| `LanguageModel` | Full decoder-only language model with weight tying |
| `get_batch` | Samples random batches of context/target pairs |
| `train_model` | Training loop with periodic validation |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 32 layers, dim 4096, 46.7B params | 4 layers, dim 128, ~6.5M params |
| Experts | 8 experts per layer, top-2, SwiGLU | 8 experts per layer, top-2, SwiGLU (same mechanism) |
| Data | Multilingual, large-scale | Tiny Shakespeare (~1 MB, character-level) |
| Context length | 32,768 tokens | 128 tokens |
| Training | Full-scale pretraining | 15 epochs, 80 steps/epoch, AdamW lr=1e-3 |
| Sliding window attention | Yes (from Mistral 7B) | No (standard causal attention) |
| RoPE | Yes (from Mistral 7B) | No (positions learned implicitly) |
| Load balancing loss | Not used | Not used (consistent) |
