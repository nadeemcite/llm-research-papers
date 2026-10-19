---
title: "RMSNorm: Code architecture"
parent: "004 RMSNorm"
nav_order: 1
---
# Code Architecture — RMSNorm Notebook

## Overview

The notebook implements two identical tiny Transformer character-level language models — one using LayerNorm, the other using RMSNorm — and trains them side by side on synthetic template-generated text. The result is measured by validation loss (perplexity) and total training time. Both models converge to nearly identical validation loss (0.191 vs 0.192), confirming comparable quality. The two models differ only in the normalization layer; everything else (embeddings, attention, FFN, output head, optimizer, learning rate, epochs) is identical.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `VOCAB_SIZE = 28` (actual: 31, computed from the character set)
- `EMBED_DIM = 128`, `HIDDEN_DIM = 512`, `NUM_LAYERS = 4`, `NUM_HEADS = 4`
- `SEQ_LEN = 64`, `BATCH_SIZE = 128`, `EPOCHS = 25`, `LR = 3e-4`
- `NUM_SAMPLES = 8000` (training sentences), `NUM_TEST = 1000` (test sentences)
- `SEED = 42`, `DEVICE = cuda if available else cpu`

### 2. Data

**Function: `generate_sentence` / `generate_corpus`** — produces synthetic English-like sentences from templates (subject + verb + object + punctuation). Returns a list of strings.

**Function: `encode`** — converts sentences to fixed-length character index tensors of shape `(num_sentences, SEQ_LEN + 1)`. Pads with newline character.

Data shapes: `train_data: (8000, 65)`, `test_data: (1000, 65)`.

### 3. Model

**Class: `RMSNorm`** (about 128 parameters per layer, no bias)
- Computes `rms = sqrt(mean(x^2) + eps)` and returns `x / rms * g`
- `g` is a learnable gain parameter, initialised to 1
- `eps = 1e-8` for numerical stability

**Class: `TransformerBlock`** — standard pre-norm Transformer block:
- `nn.MultiheadAttention(dim, num_heads, batch_first=True)`
- `norm1` (LayerNorm or RMSNorm) after attention residual
- `norm2` (LayerNorm or RMSNorm) after FFN residual
- FFN: `Linear(dim, dim*4) -> GELU -> Linear(dim*4, dim)`

**Class: `CharTransformer`** (866,591 parameters with LayerNorm, 865,567 with RMSNorm)
- Token embedding + learned positional embedding (max 512 positions)
- 4 Transformer blocks with causal masking
- Output linear layer to vocab size

**Data flow:**
```
Input: (batch, seq) — character indices
  -> embedding: (batch, seq, 128)
  -> 4x TransformerBlock: (batch, seq, 128) -> (batch, seq, 128)
  -> output: (batch, seq, 31) — logits over vocab
```

Two models created: `model_ln` (LayerNorm) and `model_rms` (RMSNorm).

### 4. Loss Function

Cross-entropy loss: `criterion = nn.CrossEntropyLoss()`. Input is first `SEQ_LEN` chars, target is the next `SEQ_LEN` chars (shifted by 1). Both models use AdamW with the same learning rate (3e-4).

### 5. Training Loop

- Optimizer: AdamW, lr 3e-4, for both models
- Gradient clipping: max norm 1.0
- 25 epochs, batch size 128
- Per epoch: train on all batches, then evaluate on test data
- Logged: train_loss, val_loss, epoch_time per epoch

### 6. Training Curves

Three subplots: training loss, validation loss (both models overlaid), and total training time bar chart. The loss curves should show both models converging to similar values.

### 7. Final Evaluation

Computes final validation loss and perplexity (= exp(loss)) for both models, and the speedup percentage of RMSNorm over LayerNorm. Prints all numbers as `Name: value` lines.

### 8. Sample Generation

Generates text from both models using temperature sampling (temperature 0.8) starting from the prompt "the cat". Shows 3 samples from each model to qualitatively verify they learned the distribution.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `RMSNorm` | RMS normalization layer: divides by RMS, applies learnable gain |
| `TransformerBlock` | One Transformer layer with attention, FFN, and norm (LayerNorm or RMSNorm) |
| `CharTransformer` | Full model: embedding + positional + 4 blocks + output head |
| `generate_sentence` | Creates a synthetic English-like sentence from a template |
| `encode` | Converts sentences to fixed-length character index tensors |
| `train_model` | Trains a model for N epochs, returns loss/time history |
| `evaluate_model` | Computes validation loss and perplexity |
| `generate` | Samples text from a trained model with temperature |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | Transformer base (6+6 layers, 512-dim, 8 heads, ~40M params) | 4 layers, 128-dim, 4 heads, ~810K params |
| Dataset | WMT14 En-De (4.5M sentence pairs, 32K BPE vocab) | 8K synthetic sentences, 31-char vocab |
| Task | Machine translation (encoder-decoder, BLEU) | Character-level next-token prediction |
| Training | 300K steps, Adam, warmup 4000, inverse-sqrt schedule | 25 epochs, AdamW (lr 3e-4), no schedule |
| Norm variants | Baseline, LayerNorm, L2-Norm, RMSNorm, pRMSNorm, BatchNorm, WeightNorm | LayerNorm vs RMSNorm only |
| Evaluation | SacreBLEU on newstest2014/2017 | Validation cross-entropy loss + perplexity |
| Hardware | TITAN X / Tesla V100 / RTX 2080 Ti (GPU) | CPU (or GPU if available) |
