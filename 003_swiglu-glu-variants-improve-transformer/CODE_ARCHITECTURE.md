---
title: "SwiGLU: Code architecture"
parent: "003 SwiGLU"
nav_order: 1
---
# Code Architecture — SwiGLU Notebook

## Overview

The notebook implements a small character-level GPT (nanoGPT-style) in PyTorch with two feed-forward variants — standard GELU and SwiGLU — and compares their training and validation loss curves on synthetic text. Both models share identical attention, embedding, and layer-norm layers; only the FFN sublayer differs. SwiGLU uses three weight matrices with a 2/3-reduced hidden dimension to match the GELU variant's parameter count. GELU achieves validation loss 1.6755 and SwiGLU 1.7051 — a negligible difference that does not reproduce the paper's improvement at this scale.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds (SEED=42) and defines the key constants (copied from the code cell):
- `EPOCHS = 30`, `LR = 3e-4`, `BATCH_SIZE = 64`, `SEQ_LEN = 128`
- `EMBED_DIM = 128`, `NUM_HEADS = 4`, `NUM_LAYERS = 4`
- `D_FF_GELU = 512` (4× expansion, 2 matrices)
- `D_FF_SWIGLU = 341` (512 × 2/3 ≈ 341, 3 matrices)
- `DEVICE = cuda` (on Kaggle GPU) or `cpu`

### 2. Data

**Function: `generate_corpus(n_sentences, seed)`** — generates n sentences from 5 templates using 4 word banks (names, actions, places, times), returns a single string. The corpus is ~84K characters with a 24-character vocabulary (letters used in the templates + space).

**Encoding:** character-to-index mapping; data stored as `np.ndarray` of int64. Split 90/10 into train (~76K chars) and val (~8K chars). A sample of the text is displayed as a figure.

### 3. Model

**Class: `CausalSelfAttention`** — multi-head causal self-attention with QKV projection (no bias). Uses `F.scaled_dot_product_attention` with `is_causal=True`. Projects back to embed_dim.

**Class: `GELUFFN`** — standard FFN: `w2(GELU(w1(x)))`. Two linear layers (no bias), hidden size 512.

**Class: `SwiGLUFFN`** — SwiGLU FFN: `w2(SiLU(w(x)) * v(x))`. Three linear layers (no bias), hidden size 341. SiLU is PyTorch's name for Swish$_1$.

**Class: `Block`** — pre-norm Transformer block: `x + attn(ln1(x))`, then `x + ffn(ln2(x))`.

**Class: `TinyGPT`** (808,192 params GELU, 807,680 SwiGLU) — token embedding + learned position embedding (seq_len=128) + 4 Blocks + LayerNorm + linear head. Weight tying between token embedding and output head.

**Data flow:**
```
Input: (B, T) integer tokens
  -> tok_emb + pos_emb: (B, T, 128)
  -> 4x Block: (B, T, 128) -> (B, T, 128)
     Block: attn(B, 4_heads, T, 32) -> proj -> ffn(512 or 341) -> (B, T, 128)
  -> ln_f -> head: (B, T, 24) [weight-tied with tok_emb]
```

### 4. Loss Function

Cross-entropy loss for next-token prediction: `logits[:, :-1]` predict `targets[:, 1:]`. Both models use `nn.CrossEntropyLoss` over the 24-character vocabulary.

### 5. Training Loop

- Optimizer: AdamW, lr=3e-4 (same for both models)
- 30 epochs, batch size 64, sequence length 128, gradient clipping at norm 1.0
- Per epoch: sample mini-batches of random starting positions, train, then evaluate validation loss
- Logs training and validation loss at epochs 1, 5, 10, 15, 20, 25, 30
- Both models trained sequentially on the same device (~7s each on Kaggle T4 GPU)

### 6. Results and Plots

- **Training curves plot:** side-by-side training and validation loss for GELU and SwiGLU over 30 epochs. Both converge to similar final loss (~1.68 train, ~1.68 val).
- **Activation functions plot:** GELU, Swish$_1$ (SiLU), and ReLU compared over the range [-5, 5]. Shows the SwiGLU gating mechanism: gate × value product.
- **FFN parameter breakdown:** bar charts showing per-layer weight matrix sizes. GELU: 131,072 params/layer (W1: 128→512, W2: 512→128); SwiGLU: 130,944 params/layer (W: 128→341, V: 128→341, W2: 341→128).
- **Sample text generation:** both models generate text from the prompt "alice ran" using temperature 0.8 sampling. Output is partially readable but not fully coherent at this model scale.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_corpus` | Build synthetic text from sentence templates |
| `CausalSelfAttention` | Multi-head causal self-attention with QKV projection |
| `GELUFFN` | Standard FFN: GELU(xW1)W2 — two matrices |
| `SwiGLUFFN` | SwiGLU FFN: (SiLU(xW) ⊗ xV)W2 — three matrices |
| `Block` | Pre-norm Transformer block with attention + FFN |
| `TinyGPT` | Full model: embedding + 4 blocks + output head |
| `get_batch` | Sample random sequences from the data for training |
| `train_model` | Training loop with mini-batch, loss logging, grad clipping |
| `generate` | Autoregressive text sampling with temperature |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | T5-base: 12 enc + 12 dec layers, d_model=768, 220M params | 4-layer, 4-head, 128-dim (~808K params) |
| Dataset | C4 (750 GB), 32K-token vocab | Synthetic text (~84K chars), 24-char vocab |
| Training | 524,288 steps with Adafactor on 32-core TPUv2 | 30 epochs with AdamW, lr 3e-4 |
| Objective | Span-filling (denoising) | Next-token (character) prediction |
| FFN variants | 8 variants (ReLU, GELU, Swish, GLU, Bilinear, ReGLU, GEGLU, SwiGLU) | 2 variants (GELU vs SwiGLU) |
| Evaluation | Log-perplexity + GLUE/SuperGLUE/SQuAD fine-tuning | Training/validation loss + perplexity |
