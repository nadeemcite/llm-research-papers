---
title: "Lost in the Middle: Code architecture"
parent: "078 Lost in the Middle"
nav_order: 1
---
# Code Architecture — Lost in the Middle Notebook

## Overview

The notebook trains a 3-layer Transformer decoder (about 605K parameters) on a synthetic multi-document QA task with 15 documents of 6 tokens each, then evaluates accuracy as a function of the answer document's position in the context. The result shows a U-shaped position-dependent accuracy curve with a best-worst gap of about 0.07, consistent with the paper's finding.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `EPOCHS = 40`, `BATCH_SIZE = 128`, `LR = 5e-4`
- `NUM_DOCS = 15` — total documents per context (1 answer + 14 distractors)
- `DOC_LEN = 6` — tokens per document
- `VOCAB_SIZE = 40` — 8 special tokens, 32 content tokens
- `D_MODEL = 128`, `N_HEADS = 4`, `N_LAYERS = 3`, `MAX_SEQ_LEN = 512`

### 2. Data — Synthetic Multi-Document QA Task

**Function: `generate_example`** — returns (context_ids, answer_token). Context format: `[BOS] + doc_0 + [SEP] + ... + doc_{k-1} + [SEP] + [QUERY] + [answer_token] + [EOS]`. The answer document has the answer token at a random position among distractor tokens; all other documents contain only non-answer content tokens.

**Function: `generate_dataset`** — generates n examples with random answer positions.

**Function: `pad_batch`** — pads contexts to MAX_SEQ_LEN, returns (batch_ctx, batch_ans) tensors of shape (batch, 512) and (batch,).

Training set: 12,000 examples. Validation set: 500 examples. Context length: 109 tokens.

### 3. Model — TinyTransformer

**Class: `TinyTransformer`** (about 605,096 parameters)
- Embedding layer (40 → 128) with padding_idx=0
- Sinusoidal positional encoding (max 512 positions)
- 3 × TransformerEncoderLayer (d=128, 4 heads, FFN=512, GELU, norm_first=True, dropout=0.1)
- Output linear layer (128 → 40)
- Causal mask (triu) for decoder-only behavior; padding mask for PAD tokens

**Data flow:**
```
Input: (batch, 109) token IDs
  -> Embedding -> (batch, 109, 128)
  -> + PosEnc
  -> TransformerEncoder (causal mask) -> (batch, 109, 128)
  -> Linear -> (batch, 109, 40) logits
  -> gather logits at EOS position -> (batch, 40)
```

### 4. Loss Function

Cross-entropy loss on the logits at the EOS position vs. the true answer token.

**Function: `get_eos_positions`** — finds the index of EOS (token 2) in each sequence.

### 5. Training Loop

- Optimizer: AdamW (lr 5e-4, weight_decay 0.01)
- Gradient clipping: max norm 1.0
- 40 epochs, batch size 128
- Logs loss and validation accuracy every 5 epochs
- Training time: about 1162s on Kaggle GPU

### 6. Training Curves

Plots training loss and validation accuracy over 40 epochs. Final loss: about 1.29, final val accuracy: about 10.4%.

### 7. Position Evaluation — The U-Shaped Curve

**Function: `evaluate_by_position`** — for each position 0..NUM_DOCS-1, generates 300 examples with the answer document fixed at that position, measures accuracy.

Results: best position 14 (11.3%), worst position 11 (4.3%), gap 7.0%. U-shaped: True.

### 8. Extended Context — Varying Number of Documents

Tests with 10, 15, and 20 documents. The best-worst gap grows: 1.7% → 7.7% → 12.3%, showing that the position effect worsens with longer contexts, consistent with the paper's Figure 5.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_example` | Creates one (context, answer) pair with answer at a specified position |
| `generate_dataset` | Generates n examples with random answer positions |
| `pad_batch` | Pads and batches contexts into tensors |
| `TinyTransformer` | 3-layer causal Transformer decoder |
| `get_eos_positions` | Finds the EOS token position for each example |
| `evaluate_by_position` | Evaluates accuracy at each answer document position |
| `make_batches` | Yields batched (ctx, ans) tensors with optional shuffling |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | MPT-30B, GPT-3.5-Turbo, Claude-1.3 (7B–175B params) | 3-layer Transformer (605K params) |
| Task | NaturalQuestions-Open multi-document QA | Synthetic token-level QA |
| Documents | Wikipedia passages (~100 tokens each) | 6-token synthetic documents |
| Context length | 4K–100K tokens | ~110 tokens |
| Evaluation | Zero-shot on pretrained LLMs | Trained from scratch for 40 epochs |
| Number of documents | 10–30 | 10–20 |