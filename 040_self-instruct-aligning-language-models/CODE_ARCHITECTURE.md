---
title: "Self-Instruct: Code architecture"
parent: "040 Self-Instruct"
nav_order: 1
---
# Code Architecture — Self-Instruct Notebook

## Overview

This notebook implements the Self-Instruct pipeline on a tiny scale: 20 seed instruction tasks, a word-level Transformer that generates new instructions and instances, a filtering step based on word-level Jaccard similarity, and supervised fine-tuning of the same model on the synthetic data. The result is measured on 50 held-out novel classification instructions using exact-match and token-overlap F1 metrics. The self-instruct model scores about 0.15 token-overlap F1 versus 0.14 for the base model.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants:
- `SEED = 42`
- `EPOCHS = 10`, `LR = 0.001`, `BATCH_SIZE = 16`
- `D_MODEL = 256`, `N_HEADS = 8`, `N_LAYERS = 2`, `MAX_LEN = 128`
- `N_SEEDS = 20`, `N_ITERATIONS = 3`, `N_GENERATE_PER_ITER = 100`
- `SIM_THRESHOLD = 0.6` — Jaccard word similarity cutoff for filtering (approximates the paper's ROUGE-L > 0.7 rule)
- `BASE_PRETRAIN_EPOCHS = 30`, `FT_EPOCHS_PER_ITER = 5`

### 2. Seed Task Pool

**Data structure: `SEED_TASKS`** — a list of 20 dicts, each with `instruction`, `input`, and `output` fields. Tasks span classification (spam detection, sentiment, fact/opinion) and generation (summarization, translation, grammar fix).

### 3. Word-Level Tokenizer and Model

**Class: `WordTokenizer`** — maps words to integer IDs. Vocabulary (366 words) built from all seed tasks plus test input words. Includes `<PAD>`, `<UNK>`, and `<EOS>` special tokens.

**Class: `MiniTransformer`** (about 1,800,046 parameters)
- Token embedding + positional embedding (learned)
- 2 Transformer encoder layers with 8 attention heads, GELU activation
- Linear head over the vocabulary
- Top-k (k=10) sampling with temperature for diverse generation
- Differs from the paper's GPT3 (decoder-only, 175B params) in every dimension; this is a toy model for demonstration

**Data flow:**
```
Input: (batch, seq_len) word IDs
  -> embedding (batch, seq_len, d_model=256)
  -> 2x Transformer layers (batch, seq_len, d_model)
  -> linear head (batch, seq_len, vocab_size=366)
```

### 4. Training Utilities

**Function: `make_batch(texts, tokenizer, device)`** — creates padded (input, target) pairs for next-word prediction with EOS token. Padding uses -100 for loss masking.

**Function: `format_task(task)`** — formats a task dict as "Instruction: ... Input: ... Output: ..." string.

**Function: `train_on_texts(model, texts, tokenizer, epochs, lr, device)`** — trains with AdamW, cross-entropy loss, gradient clipping (1.0), returns per-epoch loss list.

### 5. Base Model Pre-training

Before the self-instruct loop, the base model is pre-trained on the 20 seed tasks for 30 epochs with AdamW (lr=0.001) so it has a minimal ability to generate coherent text. Loss: cross-entropy over the vocabulary.

### 6. Self-Instruct Pipeline

**Function: `jaccard_words(a, b)`** — word-level Jaccard similarity, used in place of ROUGE-L.

**Function: `template_instruction()`** — generates a novel instruction from 10 templates with randomized slots (topics, categories, styles, etc.). Provides a fallback when model generation fails.

**Function: `generate_instruction(model, tokenizer, seed_pool, device)`** — samples 4 seed tasks, formats them as a prompt, and generates a new instruction via top-k sampling (temperature=0.7). Falls back to template generation if model output is too short or long.

**Function: `generate_instance(model, tokenizer, instruction, device)`** — given a new instruction, generates an input-output pair. Falls back to category-based output (e.g., "spam"/"not spam") if model output is empty.

**Function: `filter_item(item, existing_instructions, threshold)`** — removes generated instructions with Jaccard similarity > 0.6 to any existing instruction, or with empty outputs.

### 7. Running Self-Instruct Iterations

Three iterations of generate-filter-collect:
1. Sample seeds, generate 100 candidate instructions
2. Generate instances for each valid instruction
3. Filter by similarity, add survivors to the pool
4. Fine-tune the model on accumulated data for 5 epochs per iteration

Typical result: ~40 valid synthetic instructions across 3 iterations (fewer than 100 per iteration due to filtering).

### 8. Final Fine-Tuning on Full Synthetic Dataset

After 3 iterations, the model is fine-tuned for 10 epochs on the full synthetic dataset (seeds + generated data). Loss and sample generations are logged.

### 9. Evaluation on Novel Instructions

**Function: `token_overlap_f1(pred, ref)`** — token-level F1 overlap between prediction and reference.

**Function: `evaluate_model(model, tokenizer, test_items, device)`** — for each test instruction, generates a response via greedy decoding (temperature=0.3) and compares to reference using exact-match and token-overlap F1.

The evaluation uses 50 novel classification instructions (sentiment, spam, fact/opinion, helpfulness) with inputs not seen during training. Both base and self-instruct models are evaluated.

### 10. Training Curves and Comparison Plots

- Plot 1: Training loss across all phases (base pre-training + self-instruct iterations + final fine-tuning) with vertical markers for phase boundaries
- Plot 2: Bar chart comparing exact-match and token-overlap F1 for base vs self-instruct model
- Plot 3: Sample generations from base vs self-instruct model on 5 novel instructions

## Key Functions

| Function / Class | Purpose |
|---|---|
| `WordTokenizer` | Word-level tokenizer with encode/decode and special tokens |
| `MiniTransformer` | Small Transformer for instruction generation and fine-tuning |
| `jaccard_words` | Word-level Jaccard similarity (ROUGE-L approximation) |
| `template_instruction` | Template-based instruction generation fallback |
| `generate_instruction` | Generates a new instruction from seed examples |
| `generate_instance` | Generates input-output pairs for a given instruction |
| `filter_item` | Removes similar/invalid generated instructions |
| `evaluate_model` | Evaluates exact-match and token-overlap on test instructions |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | GPT3 (175B parameters, decoder-only) | MiniTransformer (~1.8M params, encoder-style) |
| Seed tasks | 175 human-written tasks | 20 seed tasks |
| Generated data | ~52,000 instructions, ~82,000 instances | ~40 instructions, ~60 total training examples |
| Similarity metric | ROUGE-L > 0.7 | Word-level Jaccard > 0.6 |
| Instruction generation | Model-only (GPT3 API) | Model + template-based mutation fallback |
| Fine-tuning | Full GPT3 fine-tuning on 52K data | 10 epochs on ~60 instructions |
| Evaluation | Super-NaturalInstructions + human eval | 50 novel classification instructions, exact-match + token-overlap |
