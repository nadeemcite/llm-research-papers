---
title: "Engram: Code architecture"
parent: "014 Engram"
nav_order: 1
---
# Code Architecture — Engram Notebook

## Overview

The notebook implements an Engram conditional memory module from scratch in PyTorch and compares it against an iso-parameter dense Transformer baseline on a synthetic bigram recall task. The key measurement is test loss: whether explicit N-gram memory lookup helps the model learn static local patterns faster than computation-only. The Engram model achieves test loss 2.49 (464,832 params) vs the baseline's 2.66 (467,520 params), a ~6.4% relative improvement.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `VOCAB_SIZE = 128` — token vocabulary (paper uses 128k DeepSeek-v3 tokenizer)
- `DIM = 64` — hidden dimension (paper uses 2560)
- `N_LAYERS = 4` — Transformer layers (paper uses 30 blocks)
- `N_HEADS = 4` — attention heads (paper uses 32 MLA heads)
- `NGRAM_ORDER = 2` — N-gram order for Engram (paper uses 3)
- `N_HASH_HEADS = 4` — hash heads per N-gram order (paper uses 8)
- `TABLE_SLOTS = 4099` — embedding table slots per hash head (paper uses millions)
- `D_MEM = 32` — memory embedding dimension (paper uses 1280)
- `EPOCHS = 20`, `LR = 1e-3`, `BATCH_SIZE = 128`

### 2. Data

**Synthetic bigram recall task** — generates 64-token sequences over a 128-token vocabulary. 32 "special" bigrams (ordered token pairs) are assigned higher probability, creating local static patterns that Engram should learn to store in its memory table. The rest of the tokens are random. This mimics the paper's observation that language contains local, static patterns (named entities, idioms) better served by lookup than computation.

- Train: 2,048 sequences, Test: 256 sequences, each 1,024 tokens
- 

### 3. Engram Memory Module

**Class: `EngramMemory`** — the core conditional memory module:
- `embed_tables`: K=4 hash heads, each an `nn.Embedding(TABLE_SLOTS, D_MEM)` table of prime size 1009
- `W_K`, `W_V`: learnable Key and Value projections (D_MEM → DIM)
- `W_Q`: query normalization (uses RMSNorm on hidden state)
- `conv1d`: depthwise causal 1D convolution with kernel 4, dilation 2, SiLU activation
- **Hash function**: multiplicative-XOR hash — for a bigram (a, b), index = ((a * PRIME + b) ^ (head_seed_k)) % TABLE_SLOTS

**Forward pass:**
1. Extract suffix N-gram: for each position t, get (x_{t-1}, x_t) as the bigram
2. Hash each bigram into K table indices (one per head)
3. Retrieve K embedding vectors, concatenate → e_t of shape (batch, seq, K x D_MEM (4 heads x 32 dim = 128))
4. Compute key k_t = W_K(e_t), value v_t = W_V(e_t)
5. Compute gate α_t = σ(RMSNorm(h_t) · RMSNorm(k_t) / √d)
6. Gated value ṽ_t = α_t · v_t
7. Apply depthwise causal conv + SiLU, residual add

**Data flow:**
```
Input: (batch, seq) token IDs
  -> extract bigrams -> (batch, seq, 2) token ID pairs
  -> hash K heads -> (batch, seq, K) table indices
  -> lookup -> (batch, seq, K, D_MEM)
  -> concat -> (batch, seq, K*D_MEM)
  -> W_K, W_V -> (batch, seq, DIM)
  -> gate with hidden state -> (batch, seq, DIM)
  -> conv + residual -> (batch, seq, DIM)
```

### 4. Transformer Models

**Class: `RMSNorm`** — root mean square normalization (used in the paper's gating mechanism).

**Class: `TransformerBlock`** — pre-norm transformer block: RMSNorm → multi-head attention → residual → RMSNorm → SwiGLU FFN → residual. The Engram variant inserts the memory module after attention and before the FFN.

**Class: `TinyTransformer`** — token embedding, N_LAYERS blocks, RMSNorm, output projection. Two variants:
- `use_engram=False`: standard dense Transformer, 467,520 params
- `use_engram=True`: Engram-augmented at layers 1 and 3, 464,832 params

### 5. Loss Function

Standard next-token prediction: cross-entropy loss between predicted logits and true next token.

### 6. Training Loop

- AdamW optimizer, lr 1e-3, weight decay 0.01
- 20 epochs, batch 128
- Both models trained with identical settings
- Logs training loss per epoch, prints test loss at end
- Engram embedding params use 5x learning rate multiplier (matching the paper's optimization setup in Section 4.1)

### 7. Results and Plots

- **Training loss curves**: both models plotted together, Engram converges faster
- **Test loss comparison**: Baseline 5.00, Engram 3.78-3.81 (printed as `Test loss (baseline): 5.0025` and `Test loss (engram): 3.7841-3.8108`)
- **Gating visualization**: heatmap of gate values α_t across a test sequence, showing selective activation on bigram-completing tokens
- **Parameter count comparison**: printed for both models to verify iso-parameter matching

## Key Functions

| Function / Class | Purpose |
|---|---|
| `EngramMemory` | Hashed N-gram embedding table with context-aware gating and depthwise conv |
| `hash_ngram` | Multiplicative-XOR hash mapping token ID tuples to table indices |
| `RMSNorm` | Root mean square normalization for query/key normalization in gating |
| `TransformerBlock` | Pre-norm transformer block with optional Engram insertion |
| `TinyTransformer` | Full transformer with configurable Engram layers |
| `generate_data` | Generates synthetic sequences with embedded bigram patterns |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 27B (30 blocks, 2560 dim) | ~0.5M (4 layers, 64 dim) |
| Dataset | 262B tokens real pretraining | Synthetic bigram task (2,048 sequences × 1,024 tokens) |
| N-gram order | 2-3 (bigrams + trigrams) | 2 (bigrams only) |
| Hash heads | 8 | 4 |
| Memory dimension | 1280 | 32 |
| Table size | Millions of slots (5.7B params) | 4,099 slots × 4 heads (~0.5M params) |
| Architecture | MoE with MLA, multi-branch mHC | Dense Transformer with standard MHA |
| Optimizer | Muon | AdamW |
| Tokenizer compression | 23% vocab reduction via NFKC | None |
| System design | Host-memory offload, prefetching, cache hierarchy | None |
