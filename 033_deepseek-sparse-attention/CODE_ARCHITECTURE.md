---
title: "DeepSeek Sparse Attn: Code architecture"
parent: "033 DeepSeek Sparse Attn"
nav_order: 1
---
# Code Architecture — DeepSeek Sparse Attn Notebook

## Overview

The notebook implements DeepSeek Sparse Attention (DSA) from scratch in PyTorch on a tiny 4-layer transformer trained on character-level Wikitext-2. It compares DSA against a dense attention baseline by validation perplexity, visualises which tokens the lightning indexer selects, and plots theoretical cost curves at different sequence lengths. The result shows DSA achieves val_ppl of 9.34 vs 9.18 for dense (a gap of only +0.16) while attending to only 25% of tokens per query.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `EPOCHS = 20`, `BATCH_SIZE = 64`, `LR = 1e-3`, `SEQ_LEN = 256`
- `D_MODEL = 128`, `N_LAYERS = 4`, `N_HEADS = 4`
- `K_SELECTED = 64` — number of tokens selected by the lightning indexer per query
- `INDEXER_HEADS = 4`, `INDEXER_DIM = 32` — lightning indexer configuration
- `WARMUP_EPOCHS = 2` — dense warm-up epochs before switching to sparse attention

### 2. Data

**Function: `get_batch`** — samples random sequences of `SEQ_LEN` characters from the data tensor, returns input and target batches of shape `(batch_size, seq_len)`.

Downloads Wikitext-2 raw, builds a character-level vocabulary (~100 characters), and creates fixed-length training sequences. Train: ~2M chars, Valid: ~240K chars.

### 3. DeepSeek Sparse Attention (DSA)

**Class: `LightningIndexer`** — implements Eq. 1 from the paper.
- `q_proj`: Linear(d_model → n_heads × head_dim) — query projection producing q^I
- `w_proj`: Linear(d_model → n_heads) — head weights w^I
- `k_proj`: Linear(d_model → n_heads × head_dim) — key projection producing k^I
- Forward: computes I(t,s) = Σ_j w_{t,j} · ReLU(q_{t,j} · k_{s,j}) with causal masking
- Output shape: `(batch, seq_len, seq_len)` — index scores

**Class: `SparseDSAAttention`** — the full DSA attention layer.
- Standard Q/K/V projections for main attention (d_model → d_model)
- Contains a `LightningIndexer` instance
- `sparse=False` mode: full dense attention, returns indexer scores for KL loss
- `sparse=True` mode: top-k token selection via `torch.topk` on indexer scores, then attention only on selected tokens
- For each query position, gathers the top-k keys/values and computes softmax attention over only those k entries

**Class: `DenseAttention`** — standard causal self-attention baseline (no indexer).

**Data flow (sparse mode):**
```
Input: (batch, seq_len, d_model)
  -> LightningIndexer -> index_scores (batch, seq_len, seq_len)
  -> topk(index_scores, k=64) -> topk_indices (batch, seq_len, 64)
  -> gather selected K/V -> (batch, seq_len, 64, head_dim)
  -> softmax attention over 64 tokens -> (batch, seq_len, d_model)
  -> out_proj -> (batch, seq_len, d_model)
```

### 4. Transformer Model

**Class: `DSATransformer`** (825,856 params)
- Token embedding (vocab_size → d_model) with weight-tied output head
- 4 × `DSATransformerBlock`: each with RMSNorm → DSA Attention → RMSNorm → SwiGLU FeedForward
- Forward returns logits, all indexer scores, and all attention weights (for KL loss)

**Class: `DenseTransformer`** (692,736 params)
- Same architecture but with `DenseTransformerBlock` (standard attention, no indexer)

### 5. Loss Functions

**Function: `compute_indexer_kl_loss`** — implements Eq. 3 / Eq. 4.
- Target: L1-normalised sum of attention weights across heads (detached)
- Predicted: softmax of indexer scores
- Eq. 3 (warm-up): full KL over all tokens
- Eq. 4 (sparse): KL only over the selected token set, with renormalisation

### 6. Training Loop

Two-phase training for DSA:
- **Phase 1 (epochs 0–1, warm-up):** Dense attention, freeze all parameters except the lightning indexer. Optimise only the KL loss (lr=1e-3). This aligns the indexer's scores with the main attention distribution.
- **Phase 2 (epochs 2–19, sparse):** Enable top-k sparse attention. Unfreeze all parameters. Optimise LM loss + 0.5 × KL loss (lr=1e-3, gradient clipping at 1.0). The KL loss is computed only over selected tokens (Eq. 4).

Dense baseline: standard cross-entropy training for all 20 epochs (lr=1e-3, gradient clipping 1.0).

- `N_TRAIN_BATCHES = 100`, `N_VAL_BATCHES = 20` per epoch
- Logged: train loss, KL loss, validation perplexity per epoch

### 7. Results and Plots

- **Final results cell:** prints final perplexity for both models, parameter counts, sparsity ratio, and theoretical cost reduction
- **Training curves plot:** validation perplexity and training loss for DSA vs Dense over 20 epochs, with warm-up→sparse transition marked
- **Selection pattern plot:** visualises the binary top-k selection matrix (which tokens are selected by each query) alongside the dense attention weight matrix
- **Cost comparison plot:** theoretical FLOPs for dense O(L²) vs DSA O(Lk + L²·indexer) at sequence lengths from 256 to 128K, showing the crossover and speedup
- **Text generation:** generates 100 characters from each model using the same prompt for qualitative comparison

## Key Functions

| Function / Class | Purpose |
|---|---|
| `LightningIndexer` | Computes importance scores I(t,s) between all query-key pairs using multi-head ReLU attention |
| `SparseDSAAttention` | Full DSA layer: dense or sparse (top-k) attention with lightning indexer |
| `DenseAttention` | Standard causal self-attention baseline |
| `DSATransformer` | 4-layer transformer with DSA attention blocks |
| `DenseTransformer` | 4-layer transformer with dense attention blocks |
| `compute_indexer_kl_loss` | KL-divergence loss aligning indexer scores with main attention distribution (Eq. 3/4) |
| `get_batch` | Samples random fixed-length sequences from the data |
| `evaluate_model` | Computes average loss and perplexity on validation data |
| `generate` | Autoregressive text generation from a prompt |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 671B MoE, 61 layers, MLA | 4-layer, ~1.3M params, standard MHA |
| Context length | 128K tokens | 256 characters |
| Dataset | 943.7B tokens continued pre-training | ~2M chars Wikitext-2, 20 epochs |
| k (selected tokens) | 2048 | 64 |
| Indexer heads | small (H_I) | 4 |
| Indexer dim | small (d_I) | 32 |
| Warm-up | 1000 steps (2.1B tokens) | 2 epochs (~400 batches) |
| Sparse training | 15000 steps (943.7B tokens) | 18 epochs (epochs 2–19) |
| Indexer optimisation | Detached from main graph | Not detached (simplified) |
| FP8 / Triton | Yes | No (pure PyTorch) |
| Benchmarks | AIME, HMMT, Codeforces, SWE, etc. | Validation perplexity only |
