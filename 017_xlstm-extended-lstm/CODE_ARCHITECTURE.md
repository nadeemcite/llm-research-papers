---
title: "xLSTM: Code architecture"
parent: "017 xLSTM"
nav_order: 1
---
# Code Architecture — xLSTM Notebook

## Overview

The notebook implements **sLSTM** and **mLSTM** cells from the xLSTM paper in PyTorch, wraps them in residual blocks (post up-projection for sLSTM, pre up-projection for mLSTM), and compares them against a vanilla LSTM and a Transformer on a synthetic **Multi-Query Associative Recall (MQAR)** task. The result is measured by classification accuracy at query positions only. The xLSTM model (alternating mLSTM + sLSTM blocks) achieves about 0.90 accuracy, outperforming all baselines.

## Section-by-Section Breakdown

### 1. Imports and Configuration

Sets random seeds and defines the key constants:
- `SEQ_LEN = 64` — sequence length (paper: 2048-4096)
- `VOCAB_SIZE = 16` — small vocab for the associative recall task
- `D_MODEL = 64` — hidden dimension (paper: 256-2048)
- `N_HEADS = 4` — number of attention/LSTM heads
- `N_LAYERS = 2` — number of blocks (paper: 8-24)
- `EPOCHS = 15`, `LR = 1e-3`, `BATCH_SIZE = 64`, `N_TRAIN = 2000`, `N_VAL = 500`

### 2. Data — Multi-Query Associative Recall

**Function: `generate_mqar_batch`** — generates (batch, seq_len) integer tensors. Each sequence contains 4 key-value pairs followed by 4 queries. The target at each query position is the value that was paired with the queried key. Non-query positions have target -100 (masked in loss). Returns x (batch, seq_len) and y (batch, seq_len).

### 3. sLSTM Cell

**Class: `sLSTMCell`** — implements the sLSTM with exponential gating and log-space stabilizer.
- Input projection: `x2z` (Linear: d_input -> 4*d_hidden) for gates + cell input
- Recurrent projection: `h2z` (Linear: d_hidden -> 4*d_hidden) for memory mixing
- `f_bias`: learnable forget gate bias
- Forward: processes sequence recurrently (not parallelizable)
- Gates: f_t = exp(clamp(z_f + f_bias, max=4)), i_t = exp(clamp(z_i, max=4))
- Stabilizer: tracked in log-space (m_log), m_log_new = max(f_log + m_log, i_log)
- Stabilized gates: f_stable = exp(f_log + m_log - m_log_new), i_stable = exp(i_log - m_log_new)
- Cell state: c = f_stable * c + i_stable * tanh(z_z)
- Normalizer: n = f_stable * n + i_stable
- Hidden: h = sigmoid(z_o) * c / (n + 1e-8)

### 4. mLSTM Cell

**Class: `mLSTMCell`** — implements the mLSTM with matrix memory and parallel computation.
- Input projections: x2q, x2k, x2v (Linear: d_input -> n_heads * d_head), x2f, x2i (Linear: d_input -> n_heads), x2o (Linear: d_input -> n_heads * d_head)
- LayerNorm before projections (as in the paper)
- GroupNorm on output + output projection
- Forward (parallel):
  - Compute q, k, v from LayerNorm input
  - Exponential gates: f = exp(clamp(f_raw + f_bias, max=4)), i = exp(clamp(i_raw, max=4))
  - Log-space cumulative operations:
    - f_log_cum = cumsum(log(f)) — cumulative product of forget gates
    - log_m = cummax(i_log + f_log_cum) — stabilizer
    - a = exp(i_log + f_log_cum - log_m) — stabilized input weight
    - d = exp(f_log_cum - log_m) — stabilized cumulative decay
  - Matrix memory retrieval via weighted linear attention:
    - scores[t, s] = k_s dot q_t (causal, via einsum)
    - weights[t, s] = (d_t / d_s) * a_s (causal mask applied)
    - h_t = sum_s weights[t,s] * scores[t,s] * v_s
  - Normalizer: n_q = sum_s weights[t,s] * scores[t,s], clamped to min 1.0
  - Output gate: sigmoid, applied component-wise
  - GroupNorm + output projection

### 5. Blocks

**Class: `sLSTMBlock`** — residual block with post up-projection (Transformer-style):
- LayerNorm -> sLSTM -> Dropout -> Residual add -> LayerNorm -> FFN (GELU) -> Dropout -> Residual add

**Class: `mLSTMBlock`** — residual block with pre up-projection (SSM-style):
- LayerNorm -> mLSTM -> Dropout x learnable_skip -> Residual add -> LayerNorm -> Up-projection FFN -> Residual add

**Class: `VanillaLSTMBlock`** — standard nn.LSTM with LayerNorm and residual connection.

**Class: `TransformerBlock`** — standard Transformer encoder block with MultiheadAttention and GELU FFN.

### 6. Language Model Wrapper

**Class: `LanguageModel`** — Embedding -> N blocks -> LayerNorm -> Linear head.
- For 'xLSTM': alternates mLSTM and sLSTM blocks (1:1 ratio, i.e., xLSTM[1:1])
- Model parameter counts: LSTM 69K, sLSTM 135K, mLSTM 111K, xLSTM 244K, Transformer 102K

### 7. Loss Function

Cross-entropy loss with `ignore_index=-100` (non-query positions masked).

### 8. Training Loop

**Function: `train_model`** — AdamW (lr=1e-3, weight_decay=0.01), cosine annealing schedule, gradient clipping at 1.0. Logs train loss and val accuracy every 5 epochs. All models trained identically for 15 epochs.

**Data flow:**
```
Input: (batch, 64) int tokens
  -> embed -> (batch, 64, 64)
  -> block 1 -> (batch, 64, 64)
  -> block 2 -> (batch, 64, 64)
  -> LayerNorm -> (batch, 64, 64)
  -> Linear head -> (batch, 64, 16) logits
  -> CrossEntropyLoss at query positions only
```

### 9. Training Curves

Two subplots: validation accuracy and training loss over 15 epochs for all 5 models. xLSTM shows the steepest accuracy improvement, reaching 0.84 by epoch 10.

### 10. Final Comparison

Table with model name, validation accuracy, and parameter count. Random baseline: 1/16 = 0.0625.

### 11. Bar Chart

Bar chart comparing final validation accuracy across all 5 models with random baseline line.

### 12. Memory Size Comparison

Prints memory footprint per layer: LSTM/sLSTM (scalar, O(d)), mLSTM (matrix, O(d^2)), Transformer (KV cache, O(T*d)). Shows how Transformer memory grows with sequence length while LSTM/mLSTM stay constant.

### 13. sLSTM Recurrent Demo

Processes a single validation sequence through the trained sLSTM model and shows predictions at query positions vs. true values.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_mqar_batch` | Generate MQAR sequences with key-value pairs and queries |
| `sLSTMCell` | sLSTM cell with exponential gating, normalizer, and log-space stabilizer |
| `mLSTMCell` | mLSTM cell with matrix memory, covariance update, parallel computation |
| `sLSTMBlock` | Residual sLSTM block with post up-projection |
| `mLSTMBlock` | Residual mLSTM block with pre up-projection and learnable skip |
| `VanillaLSTMBlock` | Classic LSTM block for comparison |
| `TransformerBlock` | Standard Transformer encoder block for comparison |
| `LanguageModel` | Full model: embedding + blocks + head |
| `train_model` | Training loop with AdamW, cosine schedule, gradient clipping |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 8-24 blocks, 256-2048 dim, 125M-1.3B params | 2 blocks, 64 dim, 69K-244K params |
| Dataset | SlimPajama, 300B tokens, MQAR benchmark | Synthetic MQAR, 2000 examples, 4 KV pairs |
| Training | Hundreds of billions of tokens, custom schedules | 15 epochs, 2000 examples, cosine schedule |
| sLSTM | Multi-cell with memory mixing across cells | Single cell per head (no cross-cell mixing) |
| mLSTM | Full parallel form with backward pass optimization | Simplified parallel form using cumulative ops |
| Architecture | xLSTM[7:1] or xLSTM[1:0] ratios | xLSTM[1:1] (alternating mLSTM + sLSTM) |
| Stabilizer | Per-paper stabilized state with mathematical proof | Log-space implementation for numerical stability |
