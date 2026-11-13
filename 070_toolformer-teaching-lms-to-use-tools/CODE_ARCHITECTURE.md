---
title: "Toolformer: Code architecture"
parent: "070 Toolformer"
nav_order: 1
---
# Code Architecture — Toolformer Notebook

## Overview

The notebook implements a simplified Toolformer pipeline: a tiny causal transformer (657K parameters) learns to self-label calculator API calls on toy arithmetic problems. The model is pretrained on plain math sequences plus a few demonstrations, then for each training example it samples whether to insert an API call, executes the calculator, and filters by whether the result reduces a weighted loss on subsequent tokens. The filtered augmented examples are used for fine-tuning. Evaluation compares the baseline (no API) against Toolformer (with API) on 200 test questions, showing 31.5% vs 27.0% accuracy.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `EPOCHS = 80` — pretraining epochs (paper fine-tunes GPT-J on the augmented dataset; no separate pretraining count given)
- `FINETUNE_EPOCHS = 60` — fine-tuning epochs
- `BATCH_SIZE = 32`, `LR = 3e-4`
- `TAU_S = 0.001` — sampling threshold for P(<API>) (paper does not specify the value used)
- `TAU_F = 0.01` — filtering threshold for L⁻ − L⁺ (paper uses τ_f = 1.0)
- `D_MODEL = 128`, `N_HEADS = 4`, `N_LAYERS = 3`, `MAX_SEQ_LEN = 64`
- `N_TRAIN = 500`, `N_DEMO = 150`, `N_TEST = 200`, `MAX_NUM = 199`

### 2. Vocabulary & Tokenizer

**Tokens:** Numbers 0–199 (IDs 0–199), operators plus/minus/times/equals (200–203), special API tokens `<API>`/`</API>`/`<R>`/`</R>` (204–207), PAD/EOS (208–209). Vocabulary size = 210.

**Function: `detokenize(tokens)`** — converts a token list back to a human-readable string for debugging and display.

### 3. Toy Math Dataset

**Function: `calc_api(expr_tokens)`** — the calculator API. Takes `[X, op, Y]` and returns the integer result. This is the external tool the model learns to call.

**Function: `generate_math_data(n, seed)`** — generates n arithmetic problems (addition 0–99, subtraction 0–99 non-negative, multiplication 0–12). Returns a list of dicts with `expr` (the expression tokens), `plain` (expr + equals + result + EOS), and metadata.

**Function: `make_augmented(item)`** — creates the augmented sequence: `expr <API> expr </API> <R> result </R> = result <EOS>`. This is what a self-labeled example looks like after filtering.

**Function: `make_no_result(item)`** — creates the sequence with API call but no result, used to compute L⁻ (the "API call without result" branch).

Data shapes: each sequence is a 1D list of token IDs, length 6 (plain) or 13 (augmented).

### 4. Tiny Language Model

**Class: `TinyLM`** (about 657K parameters)
- Embedding: `nn.Embedding(210, 128)` = 26,840 params
- Positional embedding: `nn.Embedding(64, 128)` = 8,192 params
- 3 × `TransformerEncoderLayer(d_model=128, n_heads=4, dim_feedforward=512, activation='gelu')` = ~295K params
- Output head: `nn.Linear(128, 210)` = 27,090 params
- Causal masking via `torch.triu` — standard decoder-only architecture

**Data flow:**
```
Input: (1, seq_len) token IDs
  -> embed: (1, seq_len, 128)
  -> + positional embedding
  -> 3x TransformerEncoderLayer with causal mask
  -> head: (1, seq_len, 210) logits
```

### 5. Pretraining

**Function: `train_model(model, seqs, epochs, lr, batch_size, label)`** — generic training loop. Pads sequences to max length, shuffles each epoch, computes next-token cross-entropy loss (ignoring PAD). Uses Adam optimizer.

Pretraining data: 500 plain math sequences + 150 demonstrations (×3 repetition) = 950 sequences. The demonstrations teach the model the `<API>` token pattern.

### 6-7. Pretraining Loss Curve & Calculator API

Plot of pretraining loss over 80 epochs. Calculator API demo showing three example computations.

### 8. Self-Supervised API Call Labeling

This is the core Toolformer mechanism:

1. **Sample:** For each training example, compute P(`<API>` \| expr) via a forward pass. If probability > τ_s = 0.001, the position is a candidate.
2. **Execute:** Call `calc_api(expr)` to get the result.
3. **Filter:** Compute L⁺ (loss on `[EQUALS, z]` given the augmented prefix with API + result) and L⁻ = min(L_plain, L_no_result). Keep the call if L⁻ − L⁺ ≥ τ_f = 0.01.

**Function: `compute_loss_on_target(model, prefix, target_tokens)`** — forward pass through the model, extract logits at the prefix boundary, compute cross-entropy on the target tokens only.

### 9. Filtering Visualization

Scatter plot of L⁻ vs L⁺ (green = kept, red = rejected) and histogram of L⁻ − L⁺ values. The filtering threshold τ_f is marked.

### 10. Fine-Tuning on Augmented Data

Fine-tune the model on 242 examples (150 demonstrations + 92 self-labeled). The baseline model (pretrained, no fine-tuning) is saved via `copy.deepcopy`.

### 11. Fine-Tuning Loss Curve

Plot of fine-tuning loss over 60 epochs.

### 12. Evaluation

**Function: `evaluate_baseline(model, data)`** — feeds `[expr, EQUALS]`, takes argmax of the last position as the predicted result. No API calls.

**Function: `evaluate_toolformer(model, data)`** — feeds `expr`; if the model predicts `<API>`, executes the calculator, constructs the augmented prefix `[expr, <API>, expr, </API>, <R>, result, </R>, EQUALS]`, and takes argmax of the last position. If no `<API>` predicted, falls back to direct prediction.

### 13-15. Results, Breakdown by Operation, Example Predictions

Bar chart comparing three conditions, pie chart of tool usage rate, per-operation breakdown, and 8 traced example predictions showing the baseline and Toolformer pipelines side by side.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `calc_api(expr_tokens)` | Calculator API — evaluates `[X, op, Y]` and returns the result |
| `generate_math_data(n, seed)` | Generates toy arithmetic problems as token sequences |
| `make_augmented(item)` | Creates the augmented sequence with API call + result |
| `make_no_result(item)` | Creates the sequence with API call but no result (for L⁻) |
| `TinyLM` | Causal transformer model (3 layers, 128 dim, 657K params) |
| `train_model(model, seqs, epochs, lr, batch_size, label)` | Generic training loop with padding and cross-entropy loss |
| `compute_loss_on_target(model, prefix, target_tokens)` | Computes loss on target tokens given a prefix context |
| `evaluate_baseline(model, data)` | Evaluates accuracy without API calls |
| `evaluate_toolformer(model, data)` | Evaluates accuracy with model-decided API calls |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Dataset | CCNet (millions of documents) | 500 toy math problems |
| Model | GPT-J (6.7B parameters) | TinyLM (657K parameters, 3 layers, d=128) |
| Tools | 5 (calculator, QA, search, translation, calendar) | 1 (calculator only) |
| Demonstrations | ~20 per tool | 150 total (repeated 3x in pretraining) |
| Filtering threshold τ_f | 1.0 | 0.01 (lenient, due to flatter loss landscape) |
| Training | Fine-tune 6.7B model on augmented data | 80 pretrain + 60 fine-tune epochs, Adam lr 3e-4 |
| API call generation | Model generates the call arguments by sampling | Expression extracted from the known math problem structure |
