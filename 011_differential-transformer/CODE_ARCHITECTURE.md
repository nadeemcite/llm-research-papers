---
title: "Diff Transformer: Code architecture"
parent: "011 Diff Transformer"
nav_order: 1
---
# Code Architecture — Diff Transformer Notebook

## Overview

The notebook implements two attention mechanisms — standard multi-head attention and differential attention — in two small PyTorch transformers, trains them on Tiny Shakespeare (character-level, 128-token context, 12 epochs), and compares validation perplexity and attention map sparsity. The Diff Transformer result: val perplexity 11.17 with 1.32M parameters, vs Standard's 11.41 with 1.06M parameters (2.1% improvement). The differential attention maps show 55.2% sparsity vs 52.3% for standard attention.

## Section-by-Section Breakdown

### 1. Setup & Imports

Standard PyTorch + numpy + matplotlib. Seeds set to 42. Device auto-detected (`cuda` on Kaggle T4, `cpu` locally).

Key constants copied from the code cell:
- `N_LAYERS = 4`, `D_MODEL = 128`, `N_HEADS = 4`, `D_HEAD = 32`
- `EPOCHS = 12`, `LR = 3e-4`, `BATCH_SIZE = 32`, `TRAIN_CTX = 128`, `STEPS_PER_EPOCH = 50`

### 2. Data — Tiny Shakespeare

Downloads the Tiny Shakespeare corpus (~1 MB, 65 characters) via `urllib.request` to `/tmp`. Encodes to integer tensor, splits 90/10 train/val.

**Shapes:** `data: (1,115,394,)`, `train_data: (1,003,854,)`, `val_data: (111,540,)`

### 3. Model

**Class: `StandardAttention`** — Standard multi-head self-attention. Projects Q, K, V each to `n_heads * d_head`. Uses `F.scaled_dot_product_attention` with causal mask. Output projected back to `d_model` via `W_o`.

**Class: `DifferentialAttention`** — The core implementation. Key differences from standard attention:
- Q and K each projected to `n_heads * 2 * d_head` (double, for two groups)
- V projected to `n_heads * 2 * d_head` (not split, dimension 2d per head)
- Q, K split into two groups: `q1, q2 = q.chunk(2, dim=-1)`, same for k
- Two attention score matrices: `A1 = q1 @ k1^T * scale`, `A2 = q2 @ k2^T * scale`
- Differential: `attn = softmax(A1) - λ * softmax(A2)`
- λ re-parameterized: `λ = exp(λq1 · λk1) - exp(λq2 · λk2) + λ_init`
- Per-head RMSNorm + fixed multiplier `(1 - λ_init)` after attention

**Class: `TransformerBlock`** — Pre-norm: `x + attn(norm1(x))`, then `x + ffn(norm2(x))`. FFN is SwiGLU-style: `silu(W_gate(x)) * W_up(x)` → `W_down`. Uses `nn.RMSNorm`.

**Class: `LanguageModel`** — Token embedding → N TransformerBlocks → RMSNorm → LM head. Weight tying between embedding and output head.

**Parameter counts:**
- Standard: 1,058,048 (W_q, W_k, W_v each 128×128; W_o 128×128; FFN 128×512×2; embed 65×128)
- Diff: 1,320,704 (W_q, W_k each 128×256; W_v 128×256; W_o 256×128; FFN same; λ vectors 4×32)

**Data flow (Differential, single head, single forward):**
```
Input: (B, T, 128)
  -> W_q -> (B, T, 256) -> view (B, 4, T, 64) -> split -> q1: (B, 4, T, 32), q2: (B, 4, T, 32)
  -> W_k -> (B, T, 256) -> view (B, 4, T, 64) -> split -> k1: (B, 4, T, 32), k2: (B, 4, T, 32)
  -> W_v -> (B, T, 256) -> view (B, 4, T, 64)  [not split]
  -> A1 = q1 @ k1^T * (1/sqrt(32))  -> (B, 4, T, T)
  -> A2 = q2 @ k2^T * (1/sqrt(32))  -> (B, 4, T, T)
  -> attn = softmax(A1) - λ * softmax(A2)  -> (B, 4, T, T)
  -> out = attn @ v  -> (B, 4, T, 64)
  -> RMSNorm + (1-λ_init) scaling
  -> reshape -> W_o -> (B, T, 128)
```

### 4. Loss Function

Standard cross-entropy loss for next-token prediction. Both models use `F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))`.

### 5. Training Loop

- AdamW (lr=3e-4, wd=0.1, betas=(0.9, 0.95)), gradient clipping at 1.0
- 12 epochs × 50 steps per epoch, batch 32
- Validation loss evaluated every 3 epochs on 50 batches
- Both models trained identically

Final results (Kaggle T4):
- Standard: 1.06M params, val_ppl=11.41, 13s training
- Diff: 1.32M params, val_ppl=11.17, 23s training

### 6. Training Curves

Side-by-side plot of training and validation loss for both models. Standard converges faster (lower initial loss) but both reach similar final values. Diff Transformer's initial loss is higher due to the differential mechanism's different initialization dynamics.

### 7. Final Evaluation

Compares validation perplexity on 100 batches. Baseline is uniform random (perplexity = vocab_size = 65.0). Both models significantly outperform the baseline. Diff Transformer shows 2.1% lower perplexity than Standard.

### 8. Attention Noise Visualization

Extracts raw attention weights from layer 0 for a 32-token sample. Four-panel visualization:
1. Standard attention (head 0): all non-negative, standard softmax pattern
2. Diff softmax(A1) (head 0): similar to standard softmax
3. Diff softmax(A2) (head 0): similar to standard softmax
4. Diff differential map (head 0): `softmax(A1) - λ·softmax(A2)`, has both positive and negative values

Also plots all 4 heads for both models. The differential maps show the noise-canceling effect: the subtraction creates signed attention patterns with higher sparsity.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `StandardAttention` | Standard multi-head self-attention (Transformer baseline) |
| `DifferentialAttention` | Differential attention: difference of two softmax maps with learnable λ |
| `TransformerBlock` | Pre-norm transformer block with attention + SwiGLU FFN |
| `LanguageModel` | Full language model with embedding, blocks, LM head, weight tying |
| `train_model` | Training loop for one attention variant |
| `evaluate` | Evaluate validation loss on n batches |
| `get_batch` | Sample a random batch of (x, y) pairs from train or val data |
| `get_attention_maps` | Extract raw attention weights for visualization |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Scale | 3B parameters (830M-13.1B for scaling) | ~1.1-1.3M parameters |
| Attention heads | h = d_model / (2d), e.g. 128 heads for 3B | 4 heads, d=32 |
| Training data | 1T tokens, multi-source corpus | Tiny Shakespeare (~1 MB), 12 epochs × 50 steps |
| Position encoding | RoPE (θ=640K for long context) | None (implicit at small scale) |
| Per-head normalization | GroupNorm | RMSNorm (simplified) |
| FlashAttention | Custom FlashDiffAttn implementation | Not used |
| Evaluation | MMLU, BBH, Needle-in-Haystack, hallucination, ICL | Validation perplexity + attention visualization |
