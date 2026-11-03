---
title: "LIMA: Code architecture"
parent: "041 LIMA"
nav_order: 1
---
# Code Architecture — LIMA Notebook

## Overview

This notebook demonstrates LIMA's "less is more for alignment" principle at toy scale. It trains a small character-level Transformer on two datasets: (1) ~100 curated, high-quality, stylistically consistent prompt–response pairs, and (2) ~1,000 noisy, inconsistent pairs. Both models share the same architecture and training setup — only the training data differs. The curated model trains for 50 epochs and the noisy model for 6 epochs, giving roughly equal total gradient steps (~200) to isolate data quality effects. Results are measured on 30 held-out prompts using BLEU-4 and format-adherence metrics. The curated model achieves 83.3% format adherence and 0.0 BLEU-4, versus 70.0% format adherence and 0.0 BLEU-4 for the noisy model.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants:
- `SEED = 42`
- `EPOCHS_CURATED = 50`, `EPOCHS_NOISY = 6`, `LR = 0.001`, `BATCH_SIZE = 32`
- `D_MODEL = 128`, `N_HEADS = 4`, `N_LAYERS = 2`, `MAX_LEN = 400`
- `N_CURATED = 100`, `N_NOISY = 1000` — sizes of the two training sets
- Curated model gets more epochs to equalize total gradient steps (~200) with the noisy model

### 2. Data Generation

**Function: `generate_curated_data(n)`** — produces n high-quality prompt–response pairs with consistent formatting: a question/instruction prompt followed by a well-structured response (bullet points, numbered steps, clear paragraphs). Topics: cooking, coding, advice, trivia, creative writing. Each response follows a uniform "helpful assistant" style with acknowledgments and structured answers.

**Function: `generate_noisy_data(n)`** — produces n prompt–response pairs with inconsistent quality: varying response lengths (some very short, some rambling), mixed formatting styles, occasional off-topic or truncated responses, and random noise injection. Same topics but without quality control.

**Function: `build_test_set(n)`** — 30 held-out prompts with reference responses, disjoint from both training sets.

### 3. Character-Level Tokenizer

**Class: `CharTokenizer`** — maps characters to integer IDs. Vocabulary built from all training and test data. Includes `<PAD>` (0), `<UNK>` (1), `<BOS>` (2), `<EOS>` (3) special tokens. Typically ~70 characters.

### 4. Model

**Class: `MiniTransformer`** (about 469,589 parameters)
- Token embedding + learned positional embedding (max 400 positions)
- 2 Transformer encoder layers with 4 attention heads, causal masking, GELU activation
- Linear head over the character vocabulary (~85 chars)
- Uses TransformerEncoder with causal mask (decoder-only style) instead of TransformerDecoder
- Differs from the paper's LLaMA 65B in every dimension; this is a toy character-level model for demonstration

**Data flow:**
```
Input: (batch, seq_len) char IDs
  -> embedding (batch, seq_len, d_model=128)
  -> 2x Transformer encoder layers with causal mask (batch, seq_len, d_model)
  -> linear head (batch, seq_len, vocab_size=~85)
```

### 5. Loss and Training

**Loss:** Cross-entropy over the character vocabulary, with padding tokens masked (ignore_index=0).

**Function: `train_model(model, data, tokenizer, epochs, lr, device)`** — trains with AdamW, gradient clipping (1.0), returns per-epoch loss and a best checkpoint by validation loss. This mirrors the paper's supervised fine-tuning with the standard next-token prediction loss — no RLHF, no preference modeling.

### 6. Training Two Models

- **Curated model:** trained on 100 high-quality pairs for 50 epochs (~200 gradient steps)
- **Noisy model:** trained on 1,000 noisy pairs for 6 epochs (~192 gradient steps)
Both use identical architecture and hyperparameters — only the training data differs, isolating the effect of data quality vs quantity. Total gradient steps are roughly equalized.

### 7. Generation

**Function: `generate_response(model, tokenizer, prompt, max_len, device)`** — generates a response via greedy decoding from the prompt, appending `<EOS>` when generated or max_len reached.

### 8. Evaluation

**Function: `bleu_score(reference, candidate, max_n=4)`** — computes BLEU-4 n-gram overlap between generated and reference responses.

**Function: `check_format(response)`** — checks whether a response exhibits structured formatting (bullet points, numbered lists, paragraph breaks) consistent with the "helpful assistant" style.

**Function: `evaluate_model(model, tokenizer, test_set, device)`** — for each test prompt, generates a response and computes BLEU-4 and format adherence. Returns average BLEU-4 and format rate.

### 9. Results and Plots

- Plot 1: Training loss curves for curated vs noisy models
- Plot 2: Bar chart comparing BLEU-4 and format adherence for curated vs noisy models
- Plot 3: Sample generations from both models on 5 test prompts, shown side-by-side

## Key Functions

| Function / Class | Purpose |
|---|---|
| `CharTokenizer` | Character-level tokenizer with encode/decode and special tokens |
| `MiniTransformer` | Small decoder-only Transformer for instruction–response generation |
| `generate_curated_data` | Creates high-quality, stylistically consistent training pairs |
| `generate_noisy_data` | Creates low-quality, inconsistent training pairs |
| `train_model` | Fine-tunes a model with AdamW and cross-entropy loss |
| `generate_response` | Greedy-decodes a response from a given prompt |
| `bleu_score` | Computes BLEU-4 overlap between generated and reference text |
| `check_format` | Checks whether a response has structured formatting |
| `evaluate_model` | Evaluates BLEU-4 and format adherence on held-out prompts |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | LLaMA 65B (65 billion parameters) | MiniTransformer (~470K params, character-level) |
| Training data | 1,000 curated real examples (Stack Exchange, wikiHow, Reddit, manual) | 100 curated synthetic pairs vs 1,000 noisy synthetic pairs |
| Total data tokens | ~750,000 tokens | ~50,000 tokens (curated) vs ~300,000 tokens (noisy) |
| Training | 15 epochs, AdamW, residual dropout, manual checkpoint selection | Curated: 50 epochs / Noisy: 6 epochs (equal steps), AdamW, no dropout, best-loss checkpointing |
| Evaluation | Human preference study (300 prompts) + absolute analysis (50 prompts) | BLEU-4 + format adherence on 30 held-out prompts |
| Alignment method | Supervised fine-tuning only (no RLHF) | Supervised fine-tuning only (no RLHF) — same principle |
