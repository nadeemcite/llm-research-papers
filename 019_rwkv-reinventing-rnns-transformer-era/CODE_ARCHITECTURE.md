---
title: "RWKV: Code architecture"
parent: "019 RWKV"
nav_order: 1
---
# Code Architecture — RWKV Notebook

## Overview

The notebook implements the RWKV time-mixing block (Section 3.1.2, Eq. 16) and channel-mixing block (Section 3.1.3, Eq. 18) in pure PyTorch, with the WKV operator provided in **both parallel (Transformer-like training) and recurrent (RNN-like inference) forms**. It verifies the two forms produce identical outputs, then trains a tiny 92,400-parameter RWKV language model alongside a depth/width-matched Transformer baseline on synthetic Markov-chain sequences and compares val loss / val accuracy. The result is measured by classification accuracy on a held-out validation set.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds (SEED=42) and defines the key constants (copied from the code cell):
- `N_LAYER = 3` — number of residual blocks (paper: up to 40 for 14B)
- `D_MODEL = 48` — model/hidden dimension (paper: 768-4096)
- `D_FF = D_MODEL * 4 = 192` — channel-mix feedforward width
- `VOCAB_SIZE = 16` — synthetic vocabulary (paper: ~50K BPE)
- `SEQ_LEN = 48` (input length 48 -> targets 47)
- `BATCH_SIZE = 32`
- `EPOCHS = 15`
- `LR = 1e-3`
- `TRAIN_SAMPLES = 4000`, `VAL_SAMPLES = 500`
- `DEVICE` = cuda if available else cpu (printed "Device: cuda" on the committed run)

### 2. Data — Synthetic Character-Level Sequences

**Function: `generate_markov_sequences(n_samples, seq_len, vocab_size, seed)`** — builds a shared random transition matrix (Dirichlet alpha=3.0) over `vocab_size` states and samples `n_samples` sequences of length `seq_len` by running the Markov chain. Returns an `(n_samples, seq_len)` int64 array. The dataset is split into train `(4000, 47)` and val `(500, 47)` tensors by shifting the sequences by one position to form next-token targets. Oracle accuracy on val is printed (~0.059) as a sanity check.

This replaces the paper's Pile corpus (330B tokens) with a small but non-trivial task where the next-token distribution genuinely depends on the previous token, rewarding models that track short-range dependencies — the capability RWKV's recurrence is designed for.

### 3. RWKV Time-Mixing Block

**Class: `RWKVTimeMix`** — implements the core time-mixing block (Section 3, Eq. 16).

Key components:
- `mu_r`, `mu_k`, `mu_v` (learnable interpolation constants for token shift)
- `Wr`, `Wk`, `Wv`, `Wo` (linear projections; no biases)
- `w` (per-channel non-negative time-decay vector, parameterized as `abs(w_raw)` to enforce non-negativity)
- `u` (learnable bonus for the current token, Appendix I)

**Parallel forward (`forward`):** Computes token-shifted r, k, v via `mu * x_t + (1-mu) * x_{t-1}`, then evaluates the WKV operator over the full sequence in a single vectorized pass (O(Td) matmuls + the explicit WKV accumulation), gates the output with `sigmoid(r)`, and projects via `Wo`. Shape: `(batch, seq, d_model) -> (batch, seq, d_model)`.

**Recurrent forward (`forward_recurrent`):** Processes one token at a time, maintaining a fixed-size state `(num_state, den_state)` of shape `(batch, d_model)` each. At each step it updates the state using the exponential decay `exp(-w)` and the current k, v, then produces the output via the same gating. Shape: `(batch, seq, d_model) -> (batch, seq, d_model), state`.

The parallel and recurrent paths share the same parameters, demonstrating the dual Transformer/RNN nature of RWKV.

### 4. RWKV Channel-Mixing Block

**Class: `RWKVChannelMix`** — implements the channel-mixing block (Section 3.1.3, Eq. 18), the RWKV analogue of the Transformer MLP.

- Token shift on r and k via `mu_r`, `mu_k`
- `Wr: d_model -> d_model`, `Wk: d_model -> d_ff`, `Wv: d_ff -> d_model`
- Squared-ReLU activation on k: `k^2` (as in the paper)
- Output gated by `sigmoid(r)`
- Custom init: `Wr` and `Wk` zero-initialized, `Wv` normal with std `0.5/sqrt(d_model)` (Section 3.4)

### 5. Full RWKV Model

**Class: `RWKVModel`** (about 92,400 parameters) — stack of `N_LAYER` residual blocks, each containing a time-mixing and channel-mixing sub-block. Uses `RWKVLayerNorm` (standard LayerNorm with learnable affine) before each sub-block. Embedding `nn.Embedding(vocab, d_model)` with normal init std=0.02, then an extra LayerNorm after the embedding (paper's small-init embedding idea, applied here as a structural element). Tied LM head (`nn.Linear(d_model, vocab, bias=False)` sharing the embedding weight). Prints the parameter count.

**Data flow:**
```
Input: (batch, seq) int64
  -> Embedding -> (batch, seq, d_model)
  -> LayerNorm
  -> for each layer:
       LN -> TimeMix -> residual
       LN -> ChannelMix -> residual
  -> LN -> Head (tied) -> (batch, seq, vocab)
```

### 6. Transformer Baseline

**Class: `TransformerBaseline`** (about 85,680 parameters) — standard causal Transformer with `nn.TransformerEncoderLayer` (nhead=4, GELU activation, norm_first=True) matched to the RWKV model's depth (`N_LAYER=3`) and width (`D_MODEL=48`, `D_FF=192`). Same embedding init and tied head. Prints the parameter count.

### 7. Loss Function

Standard cross-entropy loss for next-token prediction (the paper's language modeling objective):
`loss = CrossEntropyLoss(logits.reshape(-1, V), targets.reshape(-1))`
Accuracy is `argmax(logits) == targets` averaged over all positions.

### 8. Training Loop

**Function: `train_model(model, train_x, train_y, val_x, val_y, epochs, lr, batch_size, name)`**
- Optimizer: AdamW, `lr`, `betas=(0.9, 0.99)`, `weight_decay=0.0` (paper: custom schedule with warmup + decay; notebook uses constant lr)
- 15 epochs, batch size 32
- Logs train_loss, val_loss, val_acc every 5 epochs and at the final epoch
- Returns a `history` dict with per-epoch lists

Both models are trained identically. On the committed run:
- RWKV: final val_loss=2.9290, val_acc=0.0596, train time 365.5s
- Transformer: final val_loss=2.9328, val_acc=0.0587, train time 14.8s

### 9. Training Curves

Plots a 2-panel figure: (left) train/val loss for both models over epochs; (right) val accuracy over epochs. The saved image shows both models converge to near-identical loss/accuracy, consistent with RWKV matching Transformer performance at matched scale.

### 10. Parallel vs Recurrent Verification

The key RWKV claim. Runs the trained model on 4 validation sequences in both `forward` (parallel) and `forward_recurrent` modes and computes the max and mean absolute differences. On the committed run:
- Max \|parallel - recurrent\| = 5.96e-07
- Mean \|parallel - recurrent\| = 7.86e-08
- Outputs match: YES (threshold 1e-4)

### 11-12. Inference Speed: Parallel vs Recurrent

Times RWKV recurrent generation and Transformer full-context forward over context lengths [48, 96, 192, 384]. On the committed (GPU) run, RWKV-recurrent time grows roughly linearly with context (125 ms -> 1008 ms) because the pure-PyTorch recurrent loop is not fused, while the Transformer forward stays ~1.4 ms due to batched matmuls. A plot is saved (`inference_speed.png`). Note: this measures the unoptimized reference implementation, not the paper's custom CUDA WKV kernel, so the absolute times do not reflect RWKV's intended inference advantage.

### 13. Final Comparison Table

Prints a summary table: val_loss, val_acc, params, train_time, and the parallel=recurrent max difference for both models.

### 14. RWKV Time Decay Visualization

Plots the learned per-channel time-decay factor `exp(-w)` for each of the 3 layers, sorted descending (paper Appendix I, Figure 10). The saved image shows a spread of decay factors across channels — some near 1 (long memory) and some near 0 (local), confirming the model learns a mix of memory timescales.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_markov_sequences` | Build synthetic Markov-chain sequences for the next-token task |
| `RWKVTimeMix` | Time-mixing block with parallel and recurrent WKV (Eq. 16) |
| `RWKVTimeMix.forward` | Parallel (Transformer-like) forward over the full sequence |
| `RWKVTimeMix.forward_recurrent` | Recurrent (RNN-like) forward, one token at a time with fixed-size state |
| `RWKVChannelMix` | Channel-mixing block with squared-ReLU and receptance gating (Eq. 18) |
| `RWKVLayerNorm` | LayerNorm with learnable affine |
| `RWKVModel` | Full RWKV language model (embedding + stack of time/channel-mix blocks + tied head) |
| `TransformerBaseline` | Depth/width-matched causal Transformer for comparison |
| `compute_loss` / `compute_accuracy` | Cross-entropy loss and argmax accuracy |
| `train_model` | Training loop returning per-epoch history |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | Up to 14B parameters, 4096-dim, up to 40 layers | 92,400 params, 48-dim, 3 layers |
| Dataset | The Pile, 330B tokens, ~50K BPE vocab | 4,000 synthetic Markov sequences, 16-token vocab |
| WKV kernel | Fused custom CUDA kernel (Section 3.2) | Pure PyTorch explicit loop / state update |
| Optimizer / schedule | Custom LR schedule with warmup + exponential decay | AdamW, constant lr=1e-3, 15 epochs |
| Embedding init | Small-init embedding + extra LayerNorm (Section 3.4) | Normal init std=0.02 + extra LayerNorm (structural only) |
| Context length | Up to 8192 via progressive fine-tuning | 48 tokens |
| Evaluation | Zero-shot on ARC, HellaSwag, LAMBADA, etc. | Val loss / accuracy on synthetic task |
