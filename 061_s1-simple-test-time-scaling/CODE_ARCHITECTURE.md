---
title: "s1: Code architecture"
parent: "061 s1"
nav_order: 1
---
# Code Architecture — s1 Notebook

## Overview

The notebook implements budget forcing on a small character-level transformer trained on synthetic three-operand arithmetic (a+b-c). It first trains a 3-layer transformer (~0.7M parameters) to generate step-by-step reasoning traces, then applies budget forcing by appending "Wait" to the model's generation when it tries to end its thinking. The result is measured on 200 held-out test problems across budgets 0, 1, 2, 3 "Wait" insertions. The model achieves 77% accuracy at budget 0, but accuracy degrades with more "Wait" insertions because the tiny model enters repetitive loops — the same failure mode the paper warns about.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `D_MODEL = 128`, `N_HEADS = 4`, `N_LAYERS = 3` — model dimensions (paper uses 32B-parameter Qwen2.5)
- `EPOCHS = 30`, `LR = 1e-3`, `BATCH_SIZE = 64`
- `TRAIN_SIZE = 4000`, `TEST_SIZE = 200` — synthetic arithmetic problems
- `WAIT_BUDGETS = [0, 1, 2, 3]` — number of "Wait" insertions to test (paper tests up to 6x)
- `THINK_END = "<\|end_thought\|>"` — end-of-thinking token delimiter (paper uses a custom token)

### 2. Data Generation

**Function: `generate_arithmetic_data`** — creates synthetic three-operand arithmetic problems (a+b-c with single digits) with templated reasoning traces. Three trace types in rotation:
- Standard: straightforward step-by-step solution
- Wait re-verification: solution followed by "Wait, let me check" and re-verification
- Wait self-correction: an initial wrong answer, then "Wait, that is wrong" and correction

Returns: list of (full_reasoning_trace, correct_answer) pairs. The reasoning trace includes the `<\|end_thought\|>` marker that budget forcing will later suppress or extend.

### 3. Tokenizer

**Class: `CharTokenizer`** — character-level tokenizer that maps each character to an integer ID. The end-of-thought marker `<\|end_thought\|>` is treated as a single special token.

- `encode(str)` -> `list[int]`
- `decode(list[int])` -> `str`
- Vocabulary: 77 tokens (digits, operators, letters, special markers)

### 4. Model

**Class: `ReasoningTransformer`** (about 0.68M parameters)
- `nn.Embedding` for token embeddings (with padding_idx)
- `nn.Embedding` for learned positional embeddings (max 512 positions)
- 3 `nn.TransformerEncoderLayer` with 4 attention heads, dim=128, GELU activation
- `nn.Linear` output head to vocabulary
- Causal mask via `nn.Transformer.generate_square_subsequent_mask`

Differs from the paper: the paper fine-tunes Qwen2.5-32B-Instruct (32B parameters, pre-trained on trillions of tokens). This notebook trains a tiny model from scratch, so it must learn both language and reasoning.

**Data flow:**
```
Input: (batch, seq_len) token IDs
  -> Embedding + Positional -> (batch, seq_len, 128)
  -> 3x Transformer layers (causal mask + pad mask) -> (batch, seq_len, 128)
  -> Linear head -> (batch, seq_len, 77)
```

### 5. Dataset & Loss

Standard next-token prediction: input is seq[:-1], target is seq[1:]. Cross-entropy loss with `ignore_index=pad_id`. AdamW optimizer with lr=1e-3, weight_decay=0.01, gradient clipping at 1.0.

### 6. Training Loop

- 30 epochs, batch size 64, ~62 batches per epoch
- Logs training loss every 5 epochs
- Training takes ~185 seconds on CPU
- Final loss: ~0.088

### 7. Budget Forcing Implementation

**Function: `generate_with_budget_forcing`** — generates text with optional "Wait" token insertion. When the model generates `<\|end_thought\|>`:
- If budget remaining: suppresses the token, appends "Wait" to the sequence, and continues generation
- If budget exhausted: allows the token through, then generates the "Final Answer:" line

### 8. Evaluation

**Function: `evaluate_budget_forcing`** (inline) — evaluates on 200 test problems across budgets 0, 1, 2, 3. For each budget:
- Generates answers with the specified number of "Wait" insertions
- Extracts the final answer (last "Final Answer:" occurrence) and checks correctness
- Records accuracy and average response length

### 9. Results and Plots

- **Plot 1:** Accuracy vs. number of "Wait" insertions — shows accuracy degrading with more insertions
- **Plot 2:** Average response length vs. number of "Wait" insertions — shows compute scaling works
- **Summary cell:** prints accuracy and response length at each budget level

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_arithmetic_data` | Creates synthetic three-operand arithmetic problems with reasoning traces |
| `CharTokenizer` | Character-level tokenizer with special tokens for thought delimiters |
| `ReasoningTransformer` | 3-layer causal transformer for generating reasoning traces |
| `generate_with_budget_forcing` | Generates text with "Wait" token insertion at the end-of-thought delimiter |
| `extract_answer` | Parses the last "Final Answer:" line from generated text |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Qwen2.5-32B-Instruct (32B params) | 3-layer char-level transformer (~0.7M params) |
| Dataset | s1K: 1,000 competition problems, Gemini-distilled traces | 4,000 synthetic three-operand arithmetic, templated traces |
| Training | 26 min on 16 H100 GPUs (7 GPU-hours) | 30 epochs on CPU in ~3 minutes |
| Budget forcing | Suppress specific end-of-thinking token in full tokenizer | Suppress `<\|end_thought\|>` marker in char tokenizer |
| Evaluation | AIME24, MATH500, GPQA Diamond | 200 held-out arithmetic problems |
| Scaling range | Up to 6x "Wait" insertions | Up to 3x "Wait" insertions |
| Data curation | Quality + difficulty + diversity filtering from 59K pool | No filtering — all generated examples used |
| Budget forcing effect | Accuracy improves (50% to 57% on AIME24) | Accuracy degrades (77% to ~1%) — tiny model enters repetitive loops |
