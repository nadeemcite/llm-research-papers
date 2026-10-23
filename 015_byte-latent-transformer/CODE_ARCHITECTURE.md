---
title: "BLT: Code architecture"
parent: "015 BLT"
nav_order: 1
---
# Code Architecture — BLT Notebook

## Overview

The notebook implements a simplified Byte Latent Transformer (BLT) from scratch in PyTorch, trained on Tiny Shakespeare at the byte level (vocab 256). It demonstrates entropy-based patching, the three-module architecture (encoder, global transformer, decoder) with cross-attention and hash n-gram embeddings, and compares entropy patching against fixed strided patching. Results are measured in bits-per-byte (BPB) on a held-out validation set.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `VOCAB_SIZE = 256` — byte vocabulary (0-255), no tokenizer needed
- `GLOBAL_LAYERS = 4`, `GLOBAL_DIM = 128` — latent global transformer (vs 8B, many layers)
- `LOCAL_LAYERS = 1`, `LOCAL_DIM = 64` — encoder and decoder (vs 1-3 layers in paper)
- `EPOCHS = 20`, `BATCH_SIZE = 16`, `LR = 1e-3` (vs paper's 4e-4 with cosine decay)
- `NGRAM_SIZES = [3, 4, 5]`, `HASH_VOCAB = 10000` (vs {3,...,8} with 500K)
- `ENTROPY_MODEL_LAYERS = 2`, `ENTROPY_MODEL_DIM = 64` (vs 100M param, 14-layer model)
- `ENTROPY_THRESHOLD = 3.0` — global entropy threshold for patch boundaries
- `MAX_PATCH_SIZE = 10` — safety cap on patch length

### 2. Data

**Function: `load_text(path)`** — reads Tiny Shakespeare, encodes to bytes (list of ints 0-255), splits into train/val (90/10). Returns byte lists and the raw text.

### 3. Entropy Model

**Class: `EntropyByteLM`** (about 149K parameters) — a small autoregressive transformer that predicts next-byte distributions. Used to compute per-byte entropy H(x_i) for patching decisions. Has its own training loop (10 epochs on byte-level next-byte prediction).

**Function: `compute_entropies(model, byte_seq)`** — runs the entropy model over a byte sequence and returns per-position entropy values.

**Function: `entropy_patch(byte_seq, entropies, threshold, max_patch)`** — segments bytes into patches where entropy exceeds threshold. Returns patch boundaries and patch list.

### 4. Hash N-gram Embeddings

**Class: `HashNGramEmbedding`** — computes rolling polynomial hash of byte n-grams (n = 3,4,5) and looks up embeddings in a hash table (10K entries each). Added to byte embeddings before encoder processing.

### 5. BLT Model

**Class: `LocalEncoder`** (about 232K parameters) — 1 transformer layer + 1 cross-attention layer. Embeds bytes with hash n-gram augmentation, applies local self-attention (causal, windowed), then cross-attention pools byte representations into patch representations. Patch queries initialized by max-pooling byte representations within each patch.

**Class: `LatentGlobalTransformer`** (about 148K parameters) — 4-layer autoregressive transformer with causal attention, operating on patch representations. Uses RoPE positional embeddings and SwiGLU FFN (following Llama 3 architecture).

**Class: `LocalDecoder`** (about 40K parameters) — 1 cross-attention layer + 1 transformer layer. Cross-attention expands patch representations back to byte representations (bytes are queries, patches are keys/values), then local self-attention refines byte representations. Final linear layer projects to byte vocabulary (256).

**Data flow:**
```
Input: bytes (batch, n_bytes)
  -> HashNGram embeddings + byte embeddings
  -> LocalEncoder (self-attn + cross-attn) -> patches (batch, n_patches, global_dim)
  -> LatentGlobalTransformer -> patch outputs (batch, n_patches, global_dim)
  -> LocalDecoder (cross-attn + self-attn) -> byte logits (batch, n_bytes, 256)
```

### 6. Training

- **Loss:** Cross-entropy loss over byte predictions, reported as BPB = loss / (ln(2) * n_bytes)
- **Optimizer:** AdamW with lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95)
- **Training:** 20 epochs, patches computed once from entropy model (pre-computed), global transformer trained on patch sequences, decoder trained to predict next byte within each patch
- **Comparison:** Also trains a strided-patching baseline (fixed 4-byte patches) with identical architecture and training budget

### 7. Results and Plots

- **Entropy plot:** Shows per-byte entropy for a sample sentence with patch boundaries overlaid, illustrating how high-entropy positions (word starts) trigger new patches
- **Training curves:** BPB vs epoch for entropy-patched BLT vs strided-patched BLT
- **Patch statistics:** Histogram of patch sizes, showing variable-length distribution with mean ~4.5 bytes
- **Sample generation:** Byte-level text generation from the trained BLT model

## Key Functions

| Function / Class | Purpose |
|---|---|
| `EntropyByteLM` | Small byte-level LM for computing next-byte entropy |
| `compute_entropies` | Runs entropy model over a byte sequence |
| `entropy_patch` | Segments bytes into patches based on entropy threshold |
| `HashNGramEmbedding` | Rolling-hash n-gram lookup embeddings for local context |
| `LocalEncoder` | Byte-to-patch encoder with self-attention and cross-attention |
| `LatentGlobalTransformer` | Large transformer over patch representations |
| `LocalDecoder` | Patch-to-byte decoder with cross-attention and self-attention |
| `roll_poly_hash` | Rolling polynomial hash for n-gram computation |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | Up to 8B parameters | ~419K parameters |
| Dataset | Llama 2 / BLT-1T (4T bytes) | Tiny Shakespeare (~1 MB) |
| Entropy model | 100M params, 14 layers, 512-byte window | ~149K params, 2 layers |
| Hash n-grams | n={3,...,8}, 500K hash entries | n={3,4,5}, 10K entries |
| Attention | Flash Attention, Flex Attention, block-causal | Standard causal attention |
| Optimizer | AdamW, lr=4e-4, cosine decay, 2000-step warmup | AdamW, lr=1e-3, no schedule |
| Patching comparison | BPE Llama 3 tokenizer (128K vocab) | Strided every-4-bytes proxy |
