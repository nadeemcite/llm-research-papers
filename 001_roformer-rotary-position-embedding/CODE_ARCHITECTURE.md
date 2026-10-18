---
title: "RoPE: Code architecture"
parent: "001 RoPE"
nav_order: 1
---
# Code Architecture — RoPE Notebook

## Overview

The notebook implements a small character-level Transformer (nanoGPT-style) in PyTorch with two position-encoding variants — learned absolute (LearnedPE) and rotary (RoPE) — and compares their length-extrapolation ability. Both models are trained on 4096 synthetic counting sequences of length 64 for 100 epochs, then evaluated on length-64 and length-128 test sequences. RoPE maintains about 98.7% next-token accuracy at 2× the training length, while LearnedPE drops to about 59% overall (20.4% on the unseen positions 64-127 alone, where its position embeddings were never trained).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds (SEED=42) and defines the key constants (copied from the code cell):
- `VOCAB_SIZE = 20`, `EMBED_DIM = 128`, `NUM_HEADS = 4`, `NUM_LAYERS = 4`
- `TRAIN_LEN = 64`, `EVAL_LEN = 128` (2× training length for extrapolation test)
- `EPOCHS = 100`, `LR = 3e-4`, `BATCH_SIZE = 256`
- `N_TRAIN = 4096`, `N_TEST = 512`
- `DEVICE = cuda` (on Kaggle GPU) or `cpu`

### 2. Data

**Function: `make_sequences(n, length, seed)`** — generates n sequences where token at position t = (a·t + b) % VOCAB_SIZE, with each sequence getting a random slope a and offset b. Returns `np.ndarray` of shape (n, length) with int64 tokens.

Three datasets: `train_data` (4096 × 64), `test_short` (512 × 64, same length as train), `test_long` (512 × 128, 2× training length). A plot of three example sequences is displayed.

### 3. Model

**Function: `precompute_rope_angles(seq_len, head_dim)`** — returns angles tensor of shape (seq_len, head_dim/2) where angles[m, i] = m · θᵢ and θᵢ = 10000⁻²ⁱ/ᵈ.

**Function: `apply_rope(x, angles)`** — applies rotary embedding to x of shape (B, n_heads, seq_len, head_dim). Splits into even/odd pairs, rotates by (x₁cos−x₂sin, x₁sin+x₂cos), interleaves back. Returns same shape.

**Class: `CausalSelfAttention`** — standard multi-head causal self-attention with QKV projection. If `use_rope=True`, applies RoPE to Q and K before `F.scaled_dot_product_attention`. Uses no bias in QKV/proj.

**Class: `Block`** — pre-norm Transformer decoder block: `x + attn(ln1(x))`, then `x + mlp(ln2(x))`. MLP uses GELU with 4× expansion.

**Class: `NanoGPT`** (812,800 params for LearnedPE, 796,416 for RoPE) — token embedding + optional learned position embedding (LearnedPE only) + 4 Blocks + LayerNorm + linear head. RoPE variant has no position embedding but precomputes rotation angles as a buffer.

**Data flow:**
```
Input: (B, T) integer tokens
  -> tok_emb: (B, T, 128)
  -> [+ pos_emb if LearnedPE] -> (B, T, 128)
  -> 4x Block: (B, T, 128) -> (B, T, 128)
     each Block: attn(B, 4_heads, T, 32) -> proj -> mlp(512) -> (B, T, 128)
  -> ln_f -> head: (B, T, 20)
```

### 4. Loss Function

Cross-entropy loss for next-token prediction: logits[:, :-1] predict targets[:, 1:]. The loss is `F.cross_entropy` over the 20-token vocabulary. Both models use the same loss.

### 5. Training Loop

- Optimizer: AdamW, lr=3e-4 (same for both models)
- 100 epochs, batch size 256, gradient clipping at norm 1.0
- Per epoch: shuffle training data, mini-batch over all 4096 sequences
- Logs training loss at epochs 1, 20, 40, 60, 80, 100
- LearnedPE training time: ~66s on Kaggle T4; RoPE: ~84s

### 6. Results and Plots

- **Training curves plot:** both models converge to loss ~0.047, with RoPE slightly slower early but reaching the same final loss.
- **Extrapolation bar chart:** side-by-side accuracy and loss in three groups: length 64, length 128, and length 128 restricted to target positions 64-127. RoPE bars stay flat (98.5%, 98.7%, 98.9%); LearnedPE drops to 59.1% at length 128 and 20.4% on positions 64-127, whose embeddings were never trained (expected failure, not a tuned baseline).
- **Long-term decay plot:** left, the paper's upper bound `(1/(d/2)) * sum_j |S_j(r)|` (d = 32) for r = 0..1023; right, the empirical mean of |q·k| over 1,000 random pairs for r = 0..127. The printed conclusion is computed: the bound's window mean falls from 4.78 (r in 0-31) to 2.81 (r in 992-1023), `Bound decays: True`; the random-pair mean does not (4.55 to 4.37, `decays: False`), because the bound only caps the score.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `precompute_rope_angles` | Compute rotation angles θᵢ = 10000⁻²ⁱ/ᵈ × position for each (position, frequency) pair |
| `apply_rope` | Apply rotary rotation to Q or K tensor by splitting into even/odd pairs and rotating each pair |
| `CausalSelfAttention` | Multi-head causal self-attention with optional RoPE on Q and K |
| `NanoGPT` | Full model: embedding + N Transformer blocks + output head, with LearnedPE or RoPE |
| `make_sequences` | Generate synthetic counting sequences for training and evaluation |
| `evaluate_model` | Compute average cross-entropy loss on a dataset |
| `evaluate_accuracy` | Compute next-token prediction accuracy on a dataset |
| `train_model` | Training loop with mini-batch SGD, loss logging, gradient clipping |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 12-layer BERT-scale (RoFormer) | 4-layer, 4-head, 128-dim (~800K params) |
| Dataset | Chinese Wikipedia/news (34 GB), CAIL2019-SCM | 4096 synthetic counting sequences |
| Training | Multi-stage pretraining (200k+ steps) | 100 epochs, AdamW, lr 3e-4 |
| Task | Masked LM + semantic text matching | Next-token prediction (causal LM) |
| Position encoding | RoPE only (RoFormer) | RoPE vs LearnedPE (comparative) |
| Evaluation | CAIL2019-SCM accuracy at 512/1024 | Next-token accuracy at 64/128 (extrapolation) |
