---
title: "Chain-of-Thought: Code architecture"
parent: "052 Chain-of-Thought"
nav_order: 1
---
# Code Architecture — CoT Notebook

## Overview

This notebook compares **standard few-shot prompting** versus **chain-of-thought (CoT) few-shot prompting** on the GSM8K math word problem benchmark using GPT-2 small (124M parameters). It generates answers for 200 test problems under both prompting strategies, extracts the final numeric answer from each completion, and reports accuracy. The result demonstrates the paper's key finding that CoT benefits are an emergent property of model scale — at 124M parameters, CoT shows minimal improvement, consistent with the paper's observation that small models produce illogical chains of thought.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets the random seed and defines key constants (copied from the code cell):
- `NUM_EXEMPLARS = 8` — number of few-shot exemplars (same as the paper's 8)
- `NUM_TEST = 200` — size of the GSM8K test subset
- `MAX_NEW_TOKENS = 256` — maximum generation length per problem
- `DEVICE = 'cpu'` — GPT-2 small runs on CPU

### 2. Data

**Dataset: GSM8K** (Grade School Math 8K) loaded from HuggingFace `datasets` hub.
- `gsm8k_test` — 1,319 test problems; we use the first 200
- `gsm8k_train` — 7,473 training problems; used to source few-shot exemplars

Each problem has a `question` field and an `answer` field. The answer field contains step-by-step reasoning followed by `#### <numeric answer>`.

### 3. Model & Tokenizer

**Model: GPT-2 small** (124M parameters) from HuggingFace `transformers`.
- `AutoTokenizer.from_pretrained('gpt2')` — GPT-2 BPE tokenizer
- `AutoModelForCausalLM.from_pretrained('gpt2')` — GPT-2 small decoder-only transformer
- Model is set to `eval()` mode; no training/finetuning occurs

The paper used PaLM 540B, GPT-3 175B, LaMDA 137B, etc. GPT-2 small at 124M is far below the ~100B threshold where CoT emerges, making this a deliberate test of the scale-dependence finding.

### 4. Prompt Construction

Two prompt formats are built from the same 8 training-set exemplars:

- **`build_standard_prompt()`** — exemplars formatted as `Q: {question}\nA: The answer is {answer}.\n\n` followed by the test question
- **`build_cot_prompt()`** — exemplars formatted as `Q: {question}\nA: {reasoning}\nThe answer is {answer}.\n\n` followed by the test question

**Helper functions:**
- `extract_answer()` — pulls the numeric answer from after `####` in GSM8K's answer format
- `extract_cot()` — pulls the reasoning text (everything before `####`)

Standard prompts are ~shorter (direct answers); CoT prompts are ~longer (include reasoning steps). Both are truncated to 900 input tokens to fit GPT-2's 1024 context window.

### 5. Generation & Answer Extraction

**`generate_completion()`** — tokenizes the prompt, runs `model.generate()` with greedy decoding (`do_sample=False`), extracts only the newly generated tokens, and decodes them to text. Max 256 new tokens for CoT, 50 for standard (direct answers are short).

**`extract_predicted_answer()`** — regex search for `The answer is X` or `#### X` pattern in the model output; fallback to the last number found in the text.

### 6. Evaluation on GSM8K Subset

Iterates over 200 test problems. For each:
1. Build standard prompt → generate → extract answer
2. Build CoT prompt → generate → extract answer
3. Log progress every 25 problems

Scoring: `normalize_answer()` converts both prediction and ground truth to comparable float strings, then checks equality.

### 7. Example Outputs

Shows 2 examples where CoT got the correct answer but standard prompting did not (or falls back to the first example if none exist). This gives qualitative insight into the difference in model behavior.

### 8. Results Visualization

Bar chart comparing standard vs CoT accuracy, plus a formatted comparison table.

### 9. Per-Problem Analysis

Bins problems by estimated difficulty (number of reasoning steps, approximated by counting sentences in the ground-truth reasoning). Reports accuracy by difficulty bin to test the paper's finding that CoT helps more on harder problems.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `extract_answer()` | Extract numeric answer from GSM8K's `#### answer` format |
| `extract_cot()` | Extract reasoning text before `####` |
| `build_standard_prompt()` | Assemble few-shot prompt with direct Q→A exemplars |
| `build_cot_prompt()` | Assemble few-shot prompt with Q→reasoning→A exemplars |
| `generate_completion()` | Run GPT-2 greedy decoding on a prompt |
| `extract_predicted_answer()` | Regex-extract final numeric answer from model output |
| `normalize_answer()` | Normalize numeric strings for comparison |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | PaLM 540B, GPT-3 175B, LaMDA 137B | GPT-2 small (124M) |
| Test set | Full GSM8K (1,319 problems) | 200-problem subset |
| Exemplars | 8 manually composed CoT exemplars (Appendix G) | 8 from GSM8K train set (same count, different source) |
| Decoding | Greedy decoding | Greedy decoding (same) |
| Finetuning | None (pure prompting) | None (same) |
| Benchmarks | 5 arithmetic, 5 commonsense, 2 symbolic | GSM8K only |
