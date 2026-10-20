---
title: "Mistral: Code architecture"
parent: "007 Mistral"
nav_order: 1
---
# Code Architecture — Mistral Notebook

## Overview

The notebook implements a mini Mistral-style transformer with grouped-query attention (GQA), sliding window attention (SWA), and a rolling buffer KV cache in PyTorch. It trains the model on TinyStories and compares it against a vanilla full-attention transformer of the same parameter count. The result is measured by validation loss on a held-out split of TinyStories.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `N_LAYERS = 4`, `D_MODEL = 128`, `N_HEADS = 4`, `N_KV_HEADS = 2`, `HEAD_DIM = 32` — model dimensions (paper: 32 layers, 4096 dim, 32 heads, 8 KV heads)
- `WINDOW_SIZE = 64` — sliding window width (paper: 4096)
- `EPOCHS = 10`, `LR = 3e-4`, `BATCH_SIZE = 32`
- `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`

### 2. Data

Loads TinyStories from HuggingFace datasets (`.jsonl` file downloaded quietly). Creates a character-level tokenizer (vocab ≈ 128 ASCII characters). Splits into train (90%) and validation (10%). Samples are fixed-length sequences of 256 tokens.

**Function: `make_dataset`** — returns PyTorch `TensorDataset` of (input_ids, target_ids) with shapes (N, 256) and (N, 256).

### 3. Model

**Class: `SlidingWindowAttention`** — implements attention with a causal + sliding-window mask. Each query attends only to keys within [i−W, i]. Returns attention output and cached keys/values.

**Class: `RollingBufferCache`** — a fixed-size KV cache of shape (batch, n_kv_heads, W, head_dim). Writes at position `pos % W`, overwriting old entries when the sequence exceeds W.

**Class: `MistralBlock`** — pre-norm (RMSNorm) → SWA attention (with GQA: query heads broadcast across KV groups) → residual → pre-norm → SwiGLU FFN → residual.

**Class: `MiniMistral`** (about 1M parameters) — token embedding + 4 MistralBlocks + RMSNorm + output projection. Includes both `forward` (training, full sequence) and `forward_incremental` (inference, using rolling cache).

**Class: `VanillaTransformer`** (about 1M parameters) — same dimensions but with full causal attention (no window), standard LayerNorm + GELU FFN, no GQA. Used as baseline.

**Data flow (training):**
```
Input: (batch, seq_len=256)
  -> embed -> (batch, seq, 128)
  -> 4x [RMSNorm -> SWA(GQA) -> residual -> RMSNorm -> SwiGLU -> residual]
  -> RMSNorm -> proj -> (batch, seq, vocab)
```

**Data flow (incremental inference with rolling cache):**
```
Input: (batch, 1) + cache (batch, n_kv_heads, W, head_dim)
  -> embed -> (batch, 1, 128)
  -> 4x [RMSNorm -> SWA(1 query, cache+1 KV) -> write cache -> residual -> ...]
  -> RMSNorm -> proj -> (batch, 1, vocab)
```

### 4. Loss Function

Standard cross-entropy loss over next-token predictions: `F.cross_entropy(logits.view(-1, vocab), targets.view(-1))`. Both models use the same loss.

### 5. Training Loop

- Optimizer: AdamW (lr=3e-4), cosine schedule
- 10 epochs, batch_size=32
- Logs train loss and validation loss every 2 epochs
- Both MiniMistral and VanillaTransformer trained with identical settings

### 6. Rolling Cache Demonstration

Generates text autoregressively using `forward_incremental` with the rolling buffer cache. Verifies that the cache stays at fixed size W=64 even when generating 200+ tokens, and that old KV pairs are correctly overwritten at `pos % W`.

### 7. Attention Reach Verification

Constructs a sequence of 256 tokens and checks which source positions the last token can attend to at each layer. At layer k, the receptive field extends to k×W tokens back, confirming that SWA's effective span grows linearly with depth.

### 8. Results and Plots

- Training/validation loss curves for both models, overlaid for comparison
- Bar chart of final validation loss: Mistral vs vanilla
- Cache memory usage plot: sliding-window (fixed at W) vs full attention (grows with seq_len)

## Key Functions

| Function / Class | Purpose |
|---|---|
| `SlidingWindowAttention` | Attention with causal + sliding-window mask; supports GQA by repeating KV heads |
| `RollingBufferCache` | Fixed-size KV cache that overwrites at `pos % W` |
| `MistralBlock` | One transformer layer: pre-norm SWA + SwiGLU FFN with residuals |
| `MiniMistral` | Full model: embeddings + 4 blocks + output head; supports both training and incremental inference |
| `VanillaTransformer` | Baseline: same size, full attention, LayerNorm + GELU |
| `RMSNorm` | Root-mean-square normalization (from LLaMA) |
| `SwiGLU` | Gated linear unit with SwiGLU activation: `W2 * Swish(W1 * x) * (W3 * x)` |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 7.3B params, 32 layers, 4096 dim | ~1M params, 4 layers, 128 dim |
| GQA ratio | 32 query heads, 8 KV heads (4:1) | 4 query heads, 2 KV heads (2:1) |
| Window size | 4096 | 64 |
| Dataset | Multi-trillion-token web crawl | TinyStories (~200K short stories) |
| Tokenizer | BPE, 32K vocab | Character-level, ~128 vocab |
| Training | Full pretraining schedule | 10 epochs, AdamW lr=3e-4 |
| Context length | 8192 | 256 |
