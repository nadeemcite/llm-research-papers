---
title: "Zero-shot CoT: Code architecture"
parent: "053 Zero-shot CoT"
nav_order: 1
---
# Code Architecture — Zero-shot CoT Notebook

## Overview

This notebook compares **standard zero-shot prompting** versus **Zero-shot-CoT prompting** ("Let's think step by step") across five reasoning tasks using GPT-2 small (124M parameters). For each problem, it generates answers under both prompting strategies, extracts the final answer, and reports accuracy. The result demonstrates the paper's key finding that Zero-shot-CoT gains are an emergent property of model scale — at 124M parameters, far below the ~100B threshold, the CoT trigger produces no improvement (average 11.3% standard vs 10.7% CoT), with Coin Flip at 53.3% for both methods.

## Section-by-Section Breakdown

### 1. Setup & Imports

Sets the random seed and defines key constants (copied from the code cell):
- `NUM_TEST = 30` — number of problems evaluated per task
- `MAX_NEW_TOKENS_COT = 150` — maximum generation length for CoT reasoning (Stage 1)
- `MAX_NEW_TOKENS_STD = 30` — maximum generation length for direct answer (standard baseline and Stage 2)
- `MAX_CONTEXT = 1024` — GPT-2 context window
- `SEED = 42`, `DEVICE = cuda or cpu`

### 2. Model & Tokenizer

**Model: GPT-2 small** (124M parameters) from HuggingFace `transformers`.
- `AutoTokenizer.from_pretrained('gpt2')` — GPT-2 BPE tokenizer
- `AutoModelForCausalLM.from_pretrained('gpt2')` — GPT-2 small decoder-only transformer
- Model is set to `eval()` mode; no training/finetuning occurs

The paper used InstructGPT (text-davinci-002, ~175B) and PaLM 540B. GPT-2 small at 124M is far below the scale where large CoT gains emerge, making this a deliberate test of the scale-dependence finding.

### 3. Task Data

Five tasks spanning the paper's categories:

**GSM8K** (arithmetic): Downloaded from the public GitHub repository (test.jsonl). 30 problems taken from the 1,319-problem test set. Each problem has a `question` and `answer` field; the numeric answer is extracted from after `####`.

**MultiArith** (arithmetic): 30 procedurally generated multi-step word problems (e.g., "A store has N apples. They receive M crates of K apples..."). Ground truth computed at generation time.

**Coin Flip** (symbolic): 30 generated problems where a coin starts heads-up and 4 named people either flip or don't flip it. Ground truth is the final state (yes/no).

**Last Letter** (symbolic): 30 generated problems asking to concatenate the last letters of 4 randomly sampled names. Ground truth computed at generation time.

**Date Understanding** (logical): 30 generated problems giving a date and asking for the date N days later. Ground truth computed using Python's `datetime`.

### 4. Prompt Construction

Implements the paper's two-stage prompting pipeline (Section 3.1, Figure 2):

- **Standard zero-shot:** `Q: [question]\nA: The answer is` — single generation, direct answer
- **Zero-shot-CoT Stage 1 (reasoning extraction):** `Q: [question]\nA: Let's think step by step.` — generates reasoning text
- **Zero-shot-CoT Stage 2 (answer extraction):** `Q: [question]\nA: Let's think step by step. [reasoning] Therefore, the answer (arabic numerals) is` — generates final answer from the self-augmented prompt

**Key functions:**
- `build_zero_shot_prompt(question, trigger)` — assembles the first-stage prompt
- `build_answer_prompt(question, reasoning, task_type)` — assembles the second-stage answer extraction prompt

### 5. Generation & Answer Extraction

**`generate(prompt, max_new_tokens)`** — tokenizes the prompt, runs `model.generate()` with greedy decoding (`do_sample=False`), extracts only newly generated tokens, and decodes them to text.

**Answer extraction functions** (task-specific, matching the paper's "answer cleansing" from Section 4):
- `extract_number(text)` — first number found (for arithmetic tasks)
- `extract_yesno(text)` — yes/no from the first 20 characters (for Coin Flip)
- `extract_letters(text)` — quoted string or last word (for Last Letter)
- `extract_date(text)` — month + day pattern match (for Date Understanding)

### 6. Evaluation Framework

**`evaluate_task(task_name, problems, answer_type)`** — for each problem:
1. Build standard prompt → generate → extract answer → compare to ground truth
2. Build CoT Stage 1 prompt → generate reasoning → build Stage 2 prompt → generate answer → extract → compare
3. Log progress every 25 problems

Returns accuracy for both methods, sample examples where they disagree, and timing.

### 7-9. Per-Task Results

Each task is evaluated separately. Progress is logged every 25 problems with running accuracy counts.

### 10. Aggregated Results

Prints a comparison table across all 5 tasks showing Zero-shot vs Zero-shot-CoT accuracy and the delta.

### 11. Visualization

Grouped bar chart comparing standard vs CoT accuracy (%) for all 5 tasks.

### 12. Example Outputs

Shows up to 2 examples per task where CoT and standard prompting disagreed, including the generated reasoning text, to illustrate qualitative behavior differences.

### 13. Scale Dependence

Compares our GPT-2 small (124M) results with the paper's reported InstructGPT (175B) numbers for MultiArith and GSM8K, highlighting the scale gap and its implications.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `build_zero_shot_prompt()` | Assemble zero-shot prompt with a trigger phrase |
| `build_answer_prompt()` | Assemble Stage 2 answer extraction prompt from Stage 1 reasoning |
| `generate()` | Run GPT-2 greedy decoding on a prompt |
| `extract_number()` | Extract first number from model output (arithmetic tasks) |
| `extract_yesno()` | Extract yes/no from model output (Coin Flip) |
| `extract_letters()` | Extract concatenated letters from model output (Last Letter) |
| `extract_date()` | Extract date string from model output (Date Understanding) |
| `evaluate_task()` | Evaluate standard vs CoT on a task, return accuracies and examples |
| `gen_multiarith()` | Generate MultiArith-style arithmetic word problems |
| `gen_coinflip()` | Generate Coin Flip problems with known ground truth |
| `gen_lastletter()` | Generate Last Letter Concatenation problems |
| `gen_date()` | Generate Date Understanding problems using datetime |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | InstructGPT (175B), PaLM 540B | GPT-2 small (124M) |
| Test set size | Full benchmarks (100s-1000s of problems) | 30 problems per task |
| Tasks | 12 datasets across 4 categories | 5 tasks (GSM8K, MultiArith, Coin Flip, Last Letter, Date Understanding) |
| Data source | Curated benchmarks (GSM8K, MultiArith, BIG-bench) | GSM8K from public test set; others generated procedurally |
| Decoding | Greedy decoding | Greedy decoding (same) |
| Few-shot examples | None (zero-shot method) | None (same) |
| Prompt robustness study | 16 templates tested (Table 4) | Single template ("Let's think step by step.") |
| Scale study | 17 models from 0.3B to 540B (Figure 3) | Single model (124M) |
