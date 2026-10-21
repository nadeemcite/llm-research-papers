---
title: "YaRN: Code architecture"
parent: "008 YaRN"
nav_order: 1
---
# Code Architecture — YaRN Notebook

## Overview

The notebook implements a 4-layer RoPE-based character-level transformer (1.06M parameters) trained on Tiny Shakespeare at 128-token context, then evaluates four context-window extension methods (PI, NTK-aware, NTK-by-parts, YaRN) at scale factors 1×–8×, both zero-shot and after 200 steps of fine-tuning at 256-token context. Perplexity is the evaluation metric, matching the paper's approach.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants:
- `EMBED_DIM = 128`, `NUM_HEADS = 4`, `HEAD_DIM = 32`, `NUM_LAYERS = 4`, `FFN_DIM = 512`
- `TRAIN_CTX = 128`, `EXTEND_SCALES = [1, 2, 4, 8]`
- `EPOCHS = 15`, `LR = 3e-4` (training), `1e-4` (fine-tuning), `BATCH_SIZE = 64`
- `FINE_TUNE_STEPS = 200`, `FINE_TUNE_CTX = 256`
- `ALPHA = 1`, `BETA = 32` (paper's recommended values for LLaMA)
- `ROPE_BASE = 10000.0`

### 2. Data

Downloads Tiny Shakespeare (~1.1 MB) to `/tmp/tinyshakespeare.txt`. Character-level tokenizer maps 97 printable ASCII characters to integer IDs. 90/10 train/val split.

### 3. Model

**Class: `TinyLM`** (about 1,063,296 parameters)

Architecture:
- Token embedding: `nn.Embedding(97, 128)`, weight-tied to output head
- 4 × `TransformerBlock`: LayerNorm → `CausalSelfAttention` (RoPE) → LayerNorm → `SwiGLU` MLP
- Final LayerNorm + linear head (tied to embeddings)

**`compute_rope_freqs(head_dim, seq_len, method, scale, alpha, beta, train_ctx)`:**
Returns `cos`, `sin` tables of shape `(seq_len, head_dim/2)`. Handles five methods:
- `standard`: vanilla RoPE
- `pi`: positions scaled by 1/s, frequencies unchanged
- `ntk_aware`: base change b' = b · s^(d/(d-2))
- `ntk_parts`: ramp between PI and no-interp per dimension, controlled by wavelength ratio r = L/λ
- `yarn`: ntk_parts + temperature t = 0.1·ln(s) + 1, applied as cos/sin × 1/√t

**`apply_rope(x, cos, sin)`:** Splits the last dimension in half, applies the 2D rotation to each pair.

**Data flow:**
```
Input: (B, T) integer token IDs
  -> tok_emb: (B, T, 128)
  -> TransformerBlock × 4: (B, T, 128)
     attention: qkv -> (B, T, 3, 4, 32) -> RoPE -> SDPA(is_causal) -> proj -> (B, T, 128)
     ffn: SwiGLU(128 -> 512 -> 128)
  -> norm + head: (B, T, 97) logits
```

### 4. Loss Function

Standard next-token cross-entropy: logits (B, T, V) vs shifted targets (B, T). No auxiliary losses.

### 5. Training Loop

- Optimizer: AdamW (learning rate 3e-4 for training, 1e-4 for fine-tuning, weight_decay=0.1)
- 15 epochs × 200 batches/epoch, batch size 64, sequence length 128
- Gradient clipping at norm 1.0
- Logs train/val loss every 3 epochs
- Final val loss: 1.6556, val perplexity: 5.24

### 6. Zero-Shot Extension Evaluation

Evaluates all 5 methods at 128, 256, 512, 1024 tokens without any fine-tuning. Each method computes its own RoPE frequency table for the target context length.

Key results: baseline goes from 5.22 → 21.76 (×8), PI goes from 5.32 → 110.04 (×8), NTK-aware from 5.30 → 12.07, NTK-by-parts from 5.25 → 33.61, YaRN from 5.30 → 31.56.

### 7. Fine-Tuning

Deep-copies the base model for each method and fine-tunes 200 steps at 256-token context (2× extension). Uses AdamW with lr=1e-4. Each fine-tuned model uses its own extension method's RoPE frequencies.

### 8. Post-Fine-Tuning Evaluation

Re-evaluates all methods at all scales using the fine-tuned models. After fine-tuning, NTK-aware achieves the best perplexity at 8× (8.02), followed by NTK-by-parts (8.93) and YaRN (8.95). PI improves dramatically at 2× (4.99) but degrades at 8× (47.12).

### 9. Temperature Ablation

Isolates the contribution of YaRN's temperature scaling by comparing NTK-by-parts vs YaRN in the zero-shot setting. YaRN improves by 1.4% at 2×, 6.6% at 4×, 6.1% at 8×.

### 10. Text Generation

Generates 200 characters from "ROMEO:" at 4× context with each method. NTK-aware produces the most coherent text ("And that thou shalt thou wilt..."), consistent with its lower perplexity in this small model.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `compute_rope_freqs` | Compute RoPE cos/sin tables for any extension method and scale |
| `apply_rope` | Apply rotary embeddings to query/key tensors |
| `CausalSelfAttention` | Multi-head attention with RoPE and causal masking via SDPA |
| `SwiGLU` | SwiGLU feed-forward block (w2(silu(w1(x)) * w3(x))) |
| `TransformerBlock` | Pre-norm attention + FFN residual block |
| `TinyLM` | Full model: embedding + 4 blocks + tied output head |
| `evaluate_perplexity` | Evaluate average perplexity at a given context length |
| `fine_tune` | Fine-tune a model copy with a given extension method |
| `generate` | Autoregressive text generation with extension method |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Llama 2 7B/13B (32 layers, 4096 dim) | 4 layers, 128 dim, ~1M params |
| Data | PG19 (~82 GB, BPE 32K vocab) | Tiny Shakespeare (~1 MB, char-level 97 vocab) |
| Training | 400 steps, 64K context, batch 64 | 15 epochs / 3000 steps, 128 context, batch 64 |
| Fine-tuning | 400 steps at 64K context | 200 steps at 256 context |
| Evaluation | Sliding-window perplexity (S=256) | Direct perplexity on held-out sequences |
| Scale factors | 16×–32× (4K→64K/128K) | 2×–8× (128→256/512/1024) |
| α, β | α=1, β=32 (LLaMA) | α=1, β=32 (same) |
