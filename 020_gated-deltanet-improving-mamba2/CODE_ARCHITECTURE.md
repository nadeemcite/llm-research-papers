---
title: "Gated DeltaNet: Code architecture"
parent: "020 Gated DeltaNet"
nav_order: 1
---
# Code Architecture — Gated DeltaNet Notebook

## Overview

The notebook implements the **gated delta rule** (Eq. 10 of the paper) in pure PyTorch, in both recurrent and chunkwise parallel forms. It compares three token mixers — Gated DeltaNet, Mamba2-style decayed linear attention, and vanilla linear attention — on a synthetic multi-query associative recall (MQAR) task. The result is measured by classification accuracy at query positions on a held-out validation set. Gated DeltaNet reaches about 0.94, Mamba2 about 0.93, and linear attention about 0.59.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `SEQ_LEN = 128` — sequence length (paper: 4K training)
- `D_MODEL = 64` — model dimension (paper: 2048 for 1.3B)
- `D_HEAD = 16` — head dimension (paper: 64-128)
- `N_HEADS = 4` — number of heads
- `N_LAYERS = 2` — token-mixer layers (paper: 48+)
- `CHUNK_SIZE = 64` — chunk size C for chunkwise training (paper: 64-128)
- `EPOCHS = 15`, `LR = 1e-3`, `BATCH_SIZE = 64`
- `N_KV_PAIRS = 8` — key-value pairs in MQAR task

### 2. Data — Multi-Query Associative Recall (MQAR)

**Function: `generate_mqar_batch`** — generates (batch, seq_len) integer tensors. Random key-value pairs (from a 16-token vocab) are scattered through the sequence, then query tokens appear that must be completed with the matching value. Returns x (batch, seq_len) and y (batch, seq_len) where y is -100 at non-query positions for loss masking.

This task tests the model's ability to memorize multiple key-value associations in its recurrent state and retrieve them on demand — exactly the capability the paper highlights in Section 3.2 (S-NIAH case study).

### 3. The Gated Delta Rule — Recurrent Form

**Function: `gated_delta_recurrent(Q, K, V, alpha, beta)`** — implements Eq. 10 step by step:
- For each timestep t: compute the Householder reflection (I - beta_t k_t k_t^T), scale by alpha_t, update state S = S * house + beta * outer(v, k), read output o = S @ q
- Input shapes: Q/K (B, H, T, d_k), V (B, H, T, d_v), alpha/beta (B, H, T)
- Output: O (B, H, T, d_v)
- This is O(T) per step and cannot be parallelized, but is mathematically exact

### 4. Chunkwise Parallel Form

**Function: `segsum(x)`** — computes the segment sum (cumulative sum with causal masking) used for the 1-semiseparable decay mask.

**Function: `gated_delta_chunkwise(Q, K, V, log_alpha, beta, chunk_size)`** — implements the hardware-efficient chunkwise algorithm from Section 3.3:
1. **Decay matrices:** Gamma[i,j] = gamma[i]/gamma[j] for i >= j, where gamma is the cumulative product of alpha within each chunk
2. **T matrix via UT transform:** T = [I + strictLower(diag(beta) @ (Gamma * KK^T))]^{-1} * diag(beta), using matrix multiplication for the diag(beta) term
3. **W and U-tilde:** W = T @ K, U-tilde = T @ V
4. **Decay arrows:** K_arrow (decay to last), Q_arrow and W_arrow (decay to first)
5. **Inter-chunk recurrence:** For each chunk, compute delta = U-tilde - W_arrow @ S^T, output = (QK * Gamma * M) @ delta + Q_arrow @ S^T, then update S = chunk_decay * S + delta^T @ K_arrow

This is O(TN^2/C) with work dominated by matrix multiplications, enabling tensor core utilization.

### 5. Gated Delta Rule vs Mamba2 vs Linear Attention

Demonstrates the four special cases of the gated delta rule with different alpha/beta settings on random data, showing how each mechanism affects the output norm and dynamics.

### 6. Gated DeltaNet Layer (Learnable)

**Class: `GatedDeltaNetLayer`** (about 143K parameters for the full model)
- Projects input to Q, K, V via linear projections; L2-normalizes Q and K for stability
- alpha via linear projection + softplus (Mamba2-style, ensures decay)
- beta via sigmoid (ensures writing strength in (0, 1))
- Forward uses chunkwise form for training efficiency
- SiLU output gating (Llama-style)

**Class: `Mamba2Layer`** — Mamba2-style decayed linear attention (gated, no delta rule). Uses chunkwise form with beta=0: S_t = alpha_t * S_{t-1} + v_t k_t^T. Same projections and gating as GatedDeltaNetLayer but without the Householder term.

**Class: `LinearAttentionLayer`** — vanilla causal linear attention (no gating, no delta rule). Uses vectorized cumsum: S_t = S_{t-1} + v_t k_t^T; o_t = S_t q_t.

### 7. Full Models

**Class: `SwiGLUMLP`** — Llama-style SwiGLU MLP (w2(silu(w1(x)) * w3(x)))

**Class: `MixerBlock`** — residual block: LayerNorm -> token mixer -> LayerNorm -> SwiGLU MLP -> residual

**Class: `LanguageModel`** — embedding + N mixer blocks + output head

**Data flow:**
```
Input: (batch, 128) int tokens
  -> embed -> (batch, 128, 64)
  -> MixerBlock x2 -> (batch, 128, 64)
  -> Linear -> (batch, 128, 16) logits per position
```

### 8-9. Loss and Training

Cross-entropy loss with ignore_index=-100 (non-query positions masked). AdamW optimizer (lr=1e-3, weight_decay=0.01), gradient clipping at 1.0. All three models trained identically for 15 epochs. Logs train loss and val accuracy every 3 epochs.

### 10-12. Results and Plots

- **Training curves:** Validation accuracy and training loss over epochs for all three models
- **Final evaluation table:** Model name, val accuracy, parameter count, training time
- **Accuracy bar chart:** Comparison with random baseline (1/16 = 0.0625)

## Key Functions

| Function / Class | Purpose |
|---|---|
| `gated_delta_recurrent` | Step-by-step recurrent form of the gated delta rule (Eq. 10) |
| `gated_delta_chunkwise` | Hardware-efficient chunkwise parallel form (Section 3.3) |
| `segsum` | Segment sum for 1-semiseparable decay mask |
| `GatedDeltaNetLayer` | Learnable token mixer with gated delta rule, L2 norm, SiLU gating |
| `Mamba2Layer` | Mamba2-style decayed linear attention baseline (gated, no delta) |
| `LinearAttentionLayer` | Vanilla causal linear attention baseline (no gating, no delta) |
| `SwiGLUMLP` | Llama-style SwiGLU MLP block |
| `MixerBlock` | Residual block wrapping token mixer + MLP |
| `LanguageModel` | Full model: embedding + mixer blocks + output head |
| `generate_mqar_batch` | Generate synthetic MQAR data |
| `train_model` | Training loop with AdamW, cross-entropy, gradient clipping |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 48+ layers, 2048 dim, 1.3B params | 2 layers, 64 dim, ~143K params |
| Dataset | FineWeb-Edu, 100B tokens | Synthetic MQAR, 4000 examples, L=128 |
| Training | Cosine annealing, 1B-token warmup, 4K seq | 15 epochs, constant lr=1e-3, 128 seq |
| Chunkwise algorithm | Custom CUDA/Triton kernels | Pure PyTorch with torch.linalg.solve |
| Block design | Short conv + SiLU on Q/K/V paths | Linear projection only (no short conv) |
| Hybrid models | GatedDeltaNet-H1 (SWA), H2 (Mamba2+SWA) | Not implemented |
| Benchmarks | WikiText, LMB, PIQA, HellaSwag, etc. | Synthetic MQAR accuracy only |
