---
title: "Multi-token Pred: Code architecture"
parent: "010 Multi-token Pred"
nav_order: 1
---
# Code Architecture — Multi-token Pred Notebook

## Overview

The notebook implements two character-level language models on Tiny Shakespeare: a next-token baseline (n=1 head) and a multi-token prediction model (n=4 independent output heads on a shared trunk). Both are trained for 12 epochs on identical data and hyperparameters. The notebook then compares sample efficiency (validation loss, perplexity, top-1 accuracy) and demonstrates self-speculative decoding for inference speedup. The multi-token model's head 0 achieves perplexity 9.30 (vs 7.98 for the next-token baseline), and self-speculative decoding gives a 5.04x wall-clock speedup with 58.8% acceptance rate.

## Section-by-Section Breakdown

### 1. Setup & Imports

Standard PyTorch + numpy + matplotlib. Seeds set to 42. Device auto-detected (`cuda` on Kaggle T4, `cpu` locally).

Key constants:
- `N_TOKENS = 4` — number of future tokens to predict (paper uses n=4)
- `VOCAB_SIZE = 65` — character-level Tiny Shakespeare
- `D_MODEL = 128`, `N_HEADS = 4`, `N_LAYERS = 4`, `D_FF = 512`
- `EPOCHS = 12`, `LR = 3e-4`, `BATCH_SIZE = 64`, `SEQ_LEN = 128`, `STEPS_PER_EPOCH = 100`
- `TRAIN_CTX = 132` (= SEQ_LEN + N_TOKENS) — need extra n tokens for multi-token targets

### 2. Data — Tiny Shakespeare

Downloads the Tiny Shakespeare corpus (~1 MB, 65 unique characters) via `urllib.request`. Encodes to integer tensor, splits 90/10 train/val.

**Shapes:** `data: (1,115,394,)`, `train_data: (1,003,854,)`, `val_data: (111,540,)`

### 3. Batch Sampling

**`get_batch(split, batch_size)`** — Returns `(input_seq, targets)` where input_seq is `(B, SEQ_LEN)` and targets is `(B, SEQ_LEN, N_TOKENS)`. For each position t, targets[b, t, i] = the token at position t+1+i (for i = 0..n-1). This requires sampling sequences of length SEQ_LEN + N_TOKENS.

### 4. Model Architecture

**Class: `TransformerBlock`** — Pre-norm transformer block using `F.scaled_dot_product_attention` for causal self-attention (handles is_causal internally). FFN is a 2-layer MLP with SiLU activation. Uses `nn.RMSNorm`.

**Class: `MultiTokenLM`** — The core model. Components:
- Token embedding + learned positional embedding
- N_LAYERS shared TransformerBlocks (the "shared trunk")
- Final RMSNorm
- `n_pred` independent linear output heads, each projecting `d_model -> vocab_size`

**Forward (with targets):** Runs the shared trunk, then computes cross-entropy loss for each of the n heads against its respective target (head i predicts token at t+1+i). Total loss = average across all heads.

**`all_head_logits(x)`** — Returns logits from all n heads stacked: shape `(n_pred, B, T, V)`. Used for self-speculative decoding and accuracy evaluation.

**Data flow (multi-token, single forward):**
```
Input: (B, 128) token IDs
  -> token_emb + pos_emb -> (B, 128, 128)
  -> 4 x TransformerBlock -> z: (B, 128, 128)
  -> RMSNorm -> z: (B, 128, 128)
  -> head_0(z) -> logits_0: (B, 128, 65)  # predicts t+1
  -> head_1(z) -> logits_1: (B, 128, 65)  # predicts t+2
  -> head_2(z) -> logits_2: (B, 128, 65)  # predicts t+3
  -> head_3(z) -> logits_3: (B, 128, 65)  # predicts t+4
  -> avg CE loss across 4 heads
```

### 5. Training Loop

**`train_model(model, name, epochs)`** — AdamW optimizer (lr=3e-4, wd=0.1, betas=(0.9, 0.95)), gradient clipping at 1.0. Logs per-epoch train/val loss and per-head losses every 2 epochs.

Both models trained identically. Results (Kaggle T4):
- Next-token: 823,680 params, final train_loss 2.0339, val_loss 2.0794
- Multi-token: 848,640 params, final train_loss 2.7218, val_loss 2.7510 (head 0: 2.204)

### 6. Training Curves

Side-by-side plots: (left) train/val loss comparison between models, (right) per-head losses for the multi-token model showing degradation for farther-ahead predictions. Saved as `training_curves.png`.

### 7. Sample Efficiency Comparison

Evaluates head 0 (next-token prediction) on validation data for both models, computing cross-entropy loss, perplexity, and top-1 accuracy. Also measures per-head accuracy for the multi-token model.

Results:
- Next-token model: val loss 2.0769, perplexity 7.98, accuracy 38.5%
- Multi-token model head 0: val loss 2.2304, perplexity 9.30, accuracy 33.7%
- Multi-token heads: 33.7% (t+1), 23.9% (t+2), 18.4% (t+3), 16.9% (t+4)

### 8. Self-Speculative Decoding

**`generate_greedy(model, prompt, max_new_tokens)`** — Vanilla autoregressive generation using head 0. One forward pass per token.

**`generate_self_speculative(model, prompt, max_new_tokens)`** — At each step, runs one forward pass and uses all n heads to propose n tokens. Accepts all proposed tokens (simplified — the paper uses verification with head 0). Generates n tokens per forward pass instead of 1.

Results (200 tokens, Kaggle T4):
- Next-token model (vanilla): 0.57s, 350.3 tok/s
- Multi-token model (speculative): 0.11s, 1765.4 tok/s
- Forward passes: vanilla=200, speculative=50
- Speedup: 5.04x

### 9. Acceptance Rate Measurement

**`measure_acceptance_rate(model, n_samples, seq_len)`** — For each sample, compares speculative heads' proposals with what head 0 would actually generate step by step. Counts accepted tokens (stops at first mismatch, like real speculative decoding).

Results: 58.8% acceptance rate, 1.40 tokens accepted per 4-token proposal, effective 2.40 tokens per forward pass, theoretical 2.40x speedup.

### 10. Final Summary Plots

Three-panel figure: (1) validation loss over epochs, (2) per-head accuracy bar chart, (3) generation time comparison. Saved as `results_summary.png`.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `get_batch` | Sample batch with multi-token targets (B, SEQ_LEN, N_TOKENS) |
| `TransformerBlock` | Pre-norm transformer block with SDPA causal attention + SiLU FFN |
| `MultiTokenLM` | Language model with n independent output heads on shared trunk |
| `train_model` | Training loop with AdamW, gradient clipping, per-head loss logging |
| `eval_next_token_loss` | Evaluate head 0 cross-entropy on validation data |
| `eval_head_accuracy` | Top-1 accuracy per head on validation data |
| `generate_greedy` | Vanilla autoregressive generation with head 0 |
| `generate_self_speculative` | Multi-token proposal generation (n tokens per forward pass) |
| `measure_acceptance_rate` | Measure speculative decoding acceptance vs head 0 verification |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Scale | 300M-13B parameters | ~0.82-0.85M parameters |
| Training data | 200B-1T tokens of code | Tiny Shakespeare (~1 MB), 12 epochs x 100 steps |
| Output heads | Transformer layers (f_{h_i}) | Single linear layers |
| Memory efficiency | Sequential head forward/backward, O(V+d) | Standard (not needed at this scale) |
| Self-speculative decoding | Blockwise parallel decoding / Medusa tree attention | Simplified greedy acceptance (accept all) |
| Evaluation | HumanEval, MBPP, APPS pass@k | Validation loss, perplexity, top-1 accuracy |
| Byte-level experiments | 7B byte-level model, n=8 | Not included |
| Natural language | Summarization (ROUGE), GSM8K | Not included |
