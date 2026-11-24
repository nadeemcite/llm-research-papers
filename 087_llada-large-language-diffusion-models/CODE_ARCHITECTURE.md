---
title: "LLaDA: Code architecture"
parent: "087 LLaDA"
nav_order: 1
---
# Code Architecture — LLaDA Notebook

## Overview

The notebook implements a tiny masked diffusion language model (~803K parameters) in PyTorch, trained on ~200 hand-written English sentences with character-level tokenization. It demonstrates LLaDA's forward masking process (random ratio t ~ U(0,1)), the 1/t-weighted cross-entropy training loss (Eq. 3), and the reverse diffusion sampling with low-confidence remasking. The result is measured by training loss convergence and token prediction accuracy at various masking ratios.

## Section-by-Section Breakdown

### 1. Setup & Imports

Standard PyTorch + numpy + matplotlib. Seeds set to 42. Device auto-detected (`cuda` on Kaggle, `cpu` locally).

Key constants copied from the code cell:
- `D_MODEL = 128`, `N_HEADS = 4`, `N_LAYERS = 4`, `SEQ_LEN = 48`
- `EPOCHS = 30`, `LR = 1e-3`, `WEIGHT_DECAY = 0.1`, `BATCH_SIZE = 64`
- `DIFFUSION_STEPS = 16` (inference denoising steps)
- `VOCAB_SIZE = 29` (26 letters + space + MASK + PAD)

### 2. Data

**Character-level encoding:** 26 lowercase letters + space = 27 character tokens, plus MASK (27) and PAD (28), giving VOCAB_SIZE = 29.

**Dataset:** 129 hand-written short English sentences (e.g., "the cat sat on the mat"). Each sentence is encoded to integer IDs, padded to SEQ_LEN=48 with PAD_TOKEN.

**Shapes:** `data: (N_SENT, 48)` where N_SENT = 129

**Functions:**
- `encode(s, max_len)` — converts a string to a list of token IDs, padded to max_len
- `decode(ids)` — converts token IDs back to a string, stopping at PAD, showing 'M' for mask

### 3. Model

**Class: `MaskedDiffusionLM`** (~803K parameters, 803,200 exact)

Architecture:
- Token embedding: `Embedding(29, 128)`
- Positional embedding: `Embedding(48, 128)` (learned, not RoPE)
- 4x `TransformerEncoderLayer`: d_model=128, nhead=4, dim_feedforward=512, GELU activation, pre-norm (norm_first=True), dropout=0.0
- **No causal mask** — this is the core architectural difference from standard LLMs
- Final LayerNorm + linear LM head (weight-tied with token embedding)

Key difference from the paper: the paper uses RMSNorm, SwiGLU, and RoPE (like LLaMA3). Our toy model uses standard LayerNorm, GELU FFN, and learned positional embeddings for simplicity.

**Data flow:**
```
Input: (batch, 48) — contains real tokens and MASK tokens
  -> token_emb + pos_emb -> (batch, 48, 128)
  -> 4x TransformerEncoderLayer (bidirectional, no causal mask) -> (batch, 48, 128)
  -> LayerNorm -> (batch, 48, 128)
  -> lm_head (weight-tied) -> (batch, 48, 29)
```

### 4. Loss Function

**Function: `forward_mask(x, t)`** — masks each non-PAD token independently with probability t, replacing masked tokens with MASK_TOKEN. Returns the masked sequence and a boolean mask.

**Function: `compute_loss(model, x)`** — implements LLaDA's Eq. 3:
1. Sample t ~ U(0,1) for each sequence in the batch
2. Apply forward masking with that t
3. Run the model to get logits
4. Compute cross-entropy on masked positions only
5. Weight by 1/t (the key term that distinguishes LLaDA from BERT/MaskGIT)

The 1/t weighting means tokens masked at higher ratios (less context) contribute more to the loss, which is what makes this a proper variational lower bound on log-likelihood (Eq. 4).

### 5. Training Loop

- Optimizer: AdamW (lr=1e-3, weight_decay=0.1)
- Gradient clipping: max_norm=1.0
- 30 epochs, batch_size=64
- Each epoch: shuffle data, compute diffusion loss, backprop, step
- Validation: same data in eval mode (no train/val split given the tiny dataset)
- Logs: train and val loss every 5 epochs

### 6. Training Curves

Two plots: (1) train/val loss vs epoch, (2) log-scale version. Shows convergence from ~65 to ~6 (with high variance due to the tiny dataset).

### 7. Diffusion Sampling (Reverse Process)

**Function: `sample(model, seq_len, num_steps, prompt, greedy)`** — LLaDA's reverse diffusion sampling:
1. Start from fully masked sequence (all MASK_TOKEN)
2. Uniform timestep schedule: t from 1.0 to 0.0 over num_steps
3. At each step: predict all masked tokens, compute confidence (max softmax prob), keep top-k highest confidence, re-mask the rest
4. k is determined by the transition from t to s: keep fraction (1 - s/t) of currently masked tokens

**Function: `sample_with_history(model, ...)`** — same as `sample` but returns intermediate states for visualization.

Key design: low-confidence remasking means the least certain predictions are deferred to later steps when more context is available — this is the annealing trick from Section 2.4.

### 8. Text Generation

Three unconditional generation trials, showing the full denoising history (init -> step 1 -> ... -> final).

Three conditional generation trials with prompts ("the cat", "she", "we"), showing how prompt tokens are preserved and the model fills in the rest.

### 9. Denoising Visualization

A grid plot showing the denoising process: rows = diffusion steps, columns = sequence positions. Masked positions are dark with '?', unmasked are light with the character. This is the visual signature of LLaDA — tokens appear in confidence order, not left-to-right.

### 10. Evaluation

**Function: `evaluate_accuracy(model, data, t_fixed, n_trials)`** — measures token prediction accuracy at a fixed masking ratio. For each trial, mask tokens with ratio t_fixed, predict, and count correct predictions on masked positions.

Tests at t = {0.1, 0.25, 0.5, 0.75, 0.9}. Plots accuracy vs masking ratio with a random baseline (1/29 = 0.0345). Accuracy is ~0.16 at all ratios, roughly flat because the tiny model has not learned enough to benefit from additional context.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `MaskedDiffusionLM` | Bidirectional Transformer model (no causal mask) |
| `forward_mask` | Apply random masking with ratio t to non-PAD tokens |
| `compute_loss` | LLaDA Eq. 3: 1/t-weighted CE on masked positions |
| `sample` | Reverse diffusion sampling with low-confidence remasking |
| `sample_with_history` | Same as sample, returning intermediate states |
| `evaluate_accuracy` | Token prediction accuracy at a fixed masking ratio |
| `encode` / `decode` | Character-level tokenization |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 8B (32 layers, d_model=4096) | ~803K (4 layers, d_model=128) |
| Dataset | 2.3T tokens (web, code, math, multilingual) | ~200 toy sentences (character-level) |
| Vocabulary | 126,464 BPE tokens | 29 character tokens |
| Normalization | RMSNorm | LayerNorm |
| Activation | SwiGLU | GELU |
| Positional encoding | RoPE | Learned embeddings |
| Attention | Vanilla MHA (no GQA, no KV cache) | Vanilla MHA |
| Training schedule | WSD (warmup-stable-decay) on H800 cluster | 30 epochs, constant lr with AdamW |
| SFT | 4.5M instruction pairs | Not implemented |
| Sampling steps | Configurable (default varies by task) | 16 steps with greedy decoding |
| CFG | Supported (ablated, not used in main results) | Not implemented |
