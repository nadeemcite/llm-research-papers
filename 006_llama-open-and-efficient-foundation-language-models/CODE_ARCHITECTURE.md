---
title: "LLaMA: Code architecture"
parent: "006 LLaMA"
nav_order: 1
---
# Code Architecture — LLaMA Notebook

## Overview

The notebook assembles a mini LLaMA (RMSNorm + RoPE + SwiGLU, pre-normalization) and a vanilla Transformer (LayerNorm + GELU + learned positional embeddings) from scratch in PyTorch, trains both on ~3K TinyStories stories with character-level tokenization, and compares validation loss and generated text. The result is measured by validation cross-entropy loss and perplexity against a random baseline.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `EPOCHS = 10`, `LR = 3e-4`, `BATCH_SIZE = 64`
- `N_LAYERS = 4`, `D_MODEL = 256`, `N_HEADS = 4`, `SEQ_LEN = 128`
- `D_FF = int(2/3 * 4 * D_MODEL) = 682` — SwiGLU FFN dim, (2/3)·4d per the paper
- `DEVICE = cuda if available else cpu`
- AdamW with `betas=(0.9, 0.95)`, `weight_decay=0.1` — matching the paper's optimizer settings

### 2. Data

**HuggingFace `datasets` library:** Loads 3,000 stories from `roneneldan/TinyStories` (train split, first 3000 examples). Each story is a short text string.

**Character-level tokenizer:** Builds a vocabulary from all unique characters in the corpus. `encode()` maps characters to integer IDs, `decode()` reverses. Vocabulary size is typically ~70–80 characters (lowercase letters, punctuation, digits, newlines).

**Data preparation:** Concatenates all stories into one text, encodes to IDs, then slices into fixed-length windows of `SEQ_LEN + 1 = 129`. The first 128 tokens are input, the last 128 are targets (shifted by one). 90/10 train/val split.

**Class: `CharDataset`** — Wraps the tensor data. `__getitem__` returns `(input[:SEQ_LEN], target[:SEQ_LEN])` where target is input shifted by one position. Returns tensors of shape `(SEQ_LEN,)` and `(SEQ_LEN,)`.

### 3. Model Components

**Class: `RMSNorm`** — Root Mean Square normalization. Computes `x * rsqrt(mean(x²) + eps) * weight`. No mean subtraction, no bias. This matches the paper's choice over LayerNorm (Section 2.2, citing Zhang & Sennrich 2019).

**Function: `precompute_rope(head_dim, max_seq_len)`** — Precomputes cosine and sine tables for Rotary Positional Embeddings. Uses `inv_freq = 1/10000^(2i/d)` for frequencies, applies to positions 0..max_seq_len. Returns `(cos, sin)` tables of shape `(max_seq_len, head_dim//2)`.

**Function: `apply_rope(q, k, cos, sin)`** — Applies rotation to query and key tensors. `q, k` have shape `(batch, n_heads, seq, head_dim)`. Rotation: `q * cos + rotate_half(q) * sin`. This injects positional information into attention without absolute positional embeddings.

**Class: `SwiGLU`** — Gated FFN with Swish activation. `w2(silu(w1(x)) * w3(x))`. Three weight matrices: `w1, w3: (d_model, d_ff)`, `w2: (d_ff, d_model)`. FFN dim is `(2/3)·4d` to match parameter count of a standard `4d` GELU MLP (Section 2.2, citing Shazeer 2020).

### 4. Mini LLaMA Model

**Class: `LLaMABlock`** — Pre-norm Transformer block:
- `norm1` (RMSNorm) → Q/K/V projections → apply RoPE → `scaled_dot_product_attention(is_causal=True)` → output projection → residual
- `norm2` (RMSNorm) → SwiGLU FFN → residual

**Class: `MiniLLaMA`** — Full model:
- Token embedding (no positional embedding — RoPE handles positions)
- N_LAYERS `LLaMABlock`s
- Final RMSNorm
- LM head with **weight tying** to token embedding (as in LLaMA)
- Precomputed RoPE cos/sin stored as buffers

**Data flow:**
```
Input: (batch, seq)
  -> tok_emb: (batch, seq, D_MODEL)
  -> 4× LLaMABlock:
       -> RMSNorm -> RoPE attention -> residual
       -> RMSNorm -> SwiGLU FFN -> residual
       (batch, seq, D_MODEL) throughout
  -> RMSNorm -> lm_head: (batch, seq, VOCAB_SIZE)
```

### 5. Vanilla Transformer (Baseline)

**Class: `VanillaBlock`** — Same structure as LLaMABlock but with `nn.LayerNorm` instead of RMSNorm, `nn.GELU` FFN instead of SwiGLU, and biases in linear layers.

**Class: `VanillaTransformer`** — Same size, but with `nn.Embedding` positional embeddings (learned), no RoPE, no weight tying. The FFN dim is adjusted to `d_ff * 3 // 2` to approximately match the parameter count of SwiGLU (which uses 3 matrices of size `d_ff` vs 2 matrices of size `d_ff * 3/2`).

### 6. Loss Function

Standard cross-entropy loss for next-token prediction:
`L = -mean_t log P(x_{t+1} \| x_{1:t})` averaged over batch and sequence positions.

Both models use AdamW with identical hyperparameters: lr 3e-4, betas (0.9, 0.95), weight_decay 0.1, grad clip 1.0.

### 7. Training Loop

- Optimizer: AdamW, lr 3e-4, weight_decay 0.1, betas (0.9, 0.95)
- 10 epochs, batch 64, sequence length 128
- Grad clip at 1.0 (matching the paper)
- Loss logged per epoch; printed every 2 epochs
- Both models trained sequentially with identical hyperparameters
- Validation loss computed after each epoch

### 8. Training Curves

Two side-by-side plots: training loss and validation loss for both models across epochs. MiniLLaMA should show faster convergence.

### 9. Text Generation

`generate()` function: autoregressive sampling with temperature (0.8) and top-k (40) filtering. Both models generate ~150 characters from the prompt "Once upon a time, ". Output is printed for qualitative comparison.

### 10. Final Evaluation

Bar chart comparing validation loss of MiniLLaMA, Vanilla, and the random baseline (ln(VOCAB_SIZE)). Prints: validation loss, perplexity, loss improvement, and parameter counts.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `RMSNorm` | RMS normalization without mean subtraction or bias |
| `precompute_rope` | Precompute cos/sin tables for rotary positional embeddings |
| `apply_rope` | Apply RoPE rotation to Q and K tensors |
| `SwiGLU` | Gated FFN with Swish activation, 3 weight matrices |
| `LLaMABlock` | Pre-norm Transformer block with RoPE attention + SwiGLU |
| `MiniLLaMA` | Full mini LLaMA: embedding + N blocks + RMSNorm + tied LM head |
| `VanillaBlock` | Standard pre-norm block with LayerNorm + GELU |
| `VanillaTransformer` | Baseline: learned PE + LayerNorm + GELU, same size |
| `CharDataset` | Character-level dataset returning (input, target) tensor pairs |
| `train_model` | Trains one model for EPOCHS epochs, returns loss history |
| `generate` | Autoregressive text generation with temperature and top-k sampling |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 7B–65B params (32–80 layers, 4096–8192 dim) | 4 layers, 256-dim, 4 heads, ~2.4M params |
| Data | 1.4T BPE tokens from 7 public sources (CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange) | ~3K TinyStories stories, character-level tokenization |
| Tokenizer | SentencePiece BPE with byte fallback, digits split into individual characters | Simple character-level tokenizer |
| Training | 1T+ tokens, cosine LR schedule, 2000 warmup steps, 2048 A100 GPUs, ~21 days | 10 epochs, fixed lr 3e-4, single GPU or CPU |
| Efficiency | xformers memory-efficient attention, activation checkpointing, model+sequence parallelism | Standard PyTorch `scaled_dot_product_attention` |
| Evaluation | 20 benchmarks (BoolQ, PIQA, HellaSwag, MMLU, HumanEval, etc.) | Validation cross-entropy loss + qualitative text generation |
| Baseline | Comparison with GPT-3, Chinchilla, PaLM (external models) | Vanilla Transformer (LayerNorm + GELU + learned PE) of same size |
