---
title: "Muon: Code architecture"
parent: "036 Muon"
nav_order: 1
---
# Code Architecture — Muon Notebook

## Overview

The notebook implements the Muon optimizer from scratch in PyTorch and trains two identical ~466K-parameter 4-layer GPT models — one with Muon, one with AdamW — on character-level Tiny Shakespeare for 200 steps. Muon achieves validation loss 2.8348 (perplexity 17.03) vs AdamW's 2.8546 (perplexity 17.37), a 0.7% improvement. It also compares SVD entropy of weight matrices between the two optimizers, finding Muon has higher entropy (0.9479 vs 0.9291), confirming the paper's Section 3.4 claim.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `EPOCHS = 500` — training steps (paper: trillions of tokens)
- `BATCH_SIZE = 32` — sequences per batch (paper: 2048 examples at 8K context)
- `SEQ_LEN = 128` — context length in characters (paper: 8192 tokens)
- `LR = 1e-3` — shared learning rate for Muon and AdamW (paper: Muon reuses AdamW's LR)
- `WEIGHT_DECAY = 0.1` — decoupled weight decay (paper Section 2.2)
- `MOMENTUM = 0.95` — Nesterov momentum (paper Section 2.2, same as Jordan et al. 2024)
- `DEVICE` — CUDA if available, else CPU

### 2. Data

Downloads Tiny Shakespeare (~1 MB of text) from GitHub to `/tmp/input.txt`. Builds a character-level vocabulary (65 characters) and encodes the full text as integer tensors. Splits 90/10 train/val.

**Shapes:**
- `train_data`: `(~997,055,)` — 1D tensor of character IDs
- `val_data`: `(~110,784,)` — 1D tensor of character IDs
- `get_batch` returns `x: (32, 128)`, `y: (32, 128)` — input and next-token targets

### 3. Model

**Class: `TinyGPT`** — decoder-only Transformer using `nn.TransformerEncoderLayer` with `norm_first=True` and a causal mask.

Architecture: token embedding + positional embedding → 4 Transformer blocks (4-head self-attention + GELU feed-forward with d_ff=512) → LayerNorm → linear head (weight-tied to token embedding).

**Total parameters: 818,048**
- Matrix (2D) params: 786,432 — optimised by Muon (16 tensors)
- Non-matrix params: 31,616 — optimised by AdamW (embeddings, layer norms, head)

**Data flow:**
```
Input: (32, 128) integer indices
  -> token_emb + pos_emb -> (32, 128, 128)
  -> 4 × [causal self-attn + FFN] -> (32, 128, 128)
  -> LayerNorm -> Linear -> (32, 128, 65)
  -> cross_entropy with targets -> scalar loss
```

### 4. Newton-Schulz Orthogonalization

**Function: `newton_schulz5(G, steps=5, eps=1e-7)`** — approximates the orthogonalization of matrix G (returns UV^T where USV^T is the SVD of G).

Algorithm:
1. Normalize G by its Frobenius norm
2. Transpose if tall (for efficiency)
3. Run 5 iterations: X = aX + b(XX^T)X + c(XX^T)^2X with (a,b,c) = (3.4445, -4.7750, 2.0315)
4. Transpose back if needed

Runs in bfloat16 for efficiency (as in the original Muon implementation). Test cell verifies singular values converge to ~1.

### 5. Muon Optimizer

**Class: `Muon`** — implements the scaled Muon update (paper Equation 4):

Per-step update for each matrix parameter p of shape [A, B]:
1. Compute Nesterov momentum: buf = μ·buf + g; nesterov = g + μ·buf
2. Orthogonalize: ortho = newton_schulz5(nesterov)
3. Scale: scaled = 0.2 · ortho · sqrt(max(A, B))
4. Apply weight decay: update = scaled + 0.1 · p
5. Update: p -= lr · update

**Function: `make_optimizers(model)`** — splits parameters:
- 2D params (excluding embeddings and head) → Muon
- All other params (embeddings, layer norms, head) → AdamW

### 6. Training Utilities

**Function: `evaluate(model, split, n_batches=50)`** — average cross-entropy loss over n random validation batches.

**Function: `train_model(model, optimizer_type, steps, log_interval)`** — full training loop:
- For 'muon': creates Muon + AdamW pair, steps both each iteration
- For 'adamw': creates single AdamW for all params
- Logs train/val loss every 50 steps
- Returns history dict with train_loss, val_loss, step, final_val_loss, train_time

### 7. The Race

Trains two models from the same seed (SEED=42):
- `model_muon` with Muon optimizer → 500 steps, logs every 50
- `model_adamw` with AdamW optimizer → 500 steps, logs every 50

### 8. Training Curves Comparison

Two subplots: (left) training loss vs step for both optimizers; (right) validation loss vs step. Prints summary with final val losses, perplexities, improvement percentage, and training times.

### 9. SVD Entropy Analysis

**Function: `svd_entropy(W)`** — computes normalized SVD entropy: H(σ) = -(1/log n) Σ (σ_i²/Σσ_j²) log(σ_i²/Σσ_j²)

Iterates over all 2D weight matrices (excluding embeddings/head) from both models, computes SVD entropy for each, and compares. Reports per-matrix and mean entropy.

### 10. SVD Entropy Bar Plot

Grouped bar chart comparing SVD entropy of each weight matrix between Muon and AdamW models.

### 11. Sample Text Generation

**Function: `generate(model, prompt, max_new_tokens, temperature)`** — autoregressive text generation with temperature sampling. Generates 150 characters from "ROMEO:" using both models for qualitative comparison.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `newton_schulz5` | Newton-Schulz iteration to approximate matrix orthogonalization (UV^T from SVD) |
| `Muon` | Muon optimizer class with Nesterov momentum, orthogonalization, RMS scaling, weight decay |
| `make_optimizers` | Splits model params into Muon (2D) and AdamW (non-2D) groups |
| `TinyGPT` | Decoder-only Transformer with 4 layers, d_model=128, weight-tied head |
| `get_batch` | Samples random batch of (B, T) input/target sequences from the data tensor |
| `evaluate` | Average cross-entropy loss over n random validation batches |
| `train_model` | Full training loop for one optimizer type; returns history dict |
| `svd_entropy` | Normalized SVD entropy of a weight matrix (measures singular value diversity) |
| `generate` | Autoregressive text generation with temperature sampling |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | 3B/16B MoE (DeepSeek-V3-Small architecture) | ~466K-param 4-layer dense GPT |
| Dataset | 5.7T tokens (proprietary web corpus) | ~1 MB Tiny Shakespeare (char-level, 65 vocab) |
| Training | 5.7T tokens, batch 2048×8K, 3 stages with LR schedule | 500 steps, batch 32×128, constant LR |
| Distributed | ZeRO-1 + Megatron-LM (TP, PP, EP, DP) | Single device (CPU or GPU) |
| Precision | bf16 mixed precision | float32 (bfloat16 only inside Newton-Schulz) |
| Evaluation | MMLU, GSM8K, HumanEval, etc. | Validation loss, perplexity, SVD entropy |
| Scaling law | 399M–1.5B models, compute-optimal comparison | Single model size, fixed step count |
