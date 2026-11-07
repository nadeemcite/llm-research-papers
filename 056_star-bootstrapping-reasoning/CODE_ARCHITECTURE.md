---
title: "STaR: Code architecture"
parent: "056 STaR"
nav_order: 1
---
# Code Architecture — STaR Notebook

## Overview

The notebook implements a simplified STaR (Self-Taught Reasoner) loop using GPT-2 small (124M parameters) on 2-digit addition problems. It demonstrates the full bootstrap cycle: few-shot rationale generation, correctness-based filtering, rationalization for failed examples, and iterative fine-tuning of the base model. The result is measured as accuracy on a held-out test set of 2-digit addition problems, tracked across outer-loop iterations to show the upward trend.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `SEED = 42` — reproducibility
- `N_TRAIN = 1600` — training problems per iteration (paper: 10,000)
- `N_TEST = 200` — held-out test problems
- `N_ITERATIONS = 3` — outer-loop iterations (paper: 16-36)
- `FT_STEPS = 50` — fine-tuning steps per iteration
- `LR = 5e-5` — fine-tuning learning rate
- `BATCH_SIZE = 8` — batch size for generation and fine-tuning
- `N_DIGITS = 2` — digit length for addition problems

### 2. Data Generation

**Function: `generate_addition_problem(n_digits, rng)`** — returns a string like `"62 + 27"` and the answer `"89"`. Generates random n-digit numbers and formats them with spaces between digits (following the scratchpad format).

**Function: `make_scratchpad(a, b)`** — produces the step-by-step scratchpad rationale. For 2-digit addition, this shows the ones-digit sum, carry, tens-digit sum, and final answer, e.g.:
```
62 + 27 , C: 0
2 + 7 , 9  C: 0
6 + 2 , 8 9  C: 0
089
```

**Function: `format_problem(a, b, scratchpad, answer)`** — formats the full text sequence: input + scratchpad + answer, as a single string for the model to process.

### 3. Few-Shot Prompt

Constructs a few-shot prompt with 4 worked examples (2-digit addition with scratchpad), following the paper's approach of using a small prompt set of rationale examples. This prompt is prepended to each question during rationale generation.

### 4. Model Setup

**Model: GPT-2 small (124M)** loaded from Hugging Face `gpt2` checkpoint. The model processes the few-shot prompt + question and generates a continuation (scratchpad + answer). Fine-tuning is done on the full text (prompt + rationale + answer) using the standard language modeling objective.

### 5. STaR Loop

The core algorithm (Algorithm 1 in the paper):

1. **Rationale generation:** For each training problem, prepend the few-shot prompt and generate a continuation (scratchpad + answer) using greedy decoding.
2. **Filtering:** Parse the generated answer, compare to ground truth. Keep only examples where the answer is correct.
3. **Rationalization:** For examples that were answered incorrectly, construct a prompt that includes the correct answer as a hint, and generate a rationale justifying it. Add these to the training set.
4. **Fine-tuning:** Fine-tune the original GPT-2 model on the combined dataset (correct rationales + rationalized rationales) for `FT_STEPS` steps.
5. **Evaluate:** Test accuracy on held-out test set.
6. **Repeat:** Use the fine-tuned model for the next iteration's rationale generation.

### 6. Results and Plots

- **Accuracy plot:** Test accuracy vs. iteration number, showing the bootstrap trend.
- **Dataset growth bar chart:** Number of correct rationales and rationalized rationales per iteration.
- **Example outputs:** A few generated scratchpads shown for qualitative inspection.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_addition_problem` | Create a random n-digit addition problem and its answer |
| `make_scratchpad` | Generate the step-by-step scratchpad rationale for an addition problem |
| `format_example` | Format a full problem+rationale+answer as a text string |
| `build_fewshot_prompt` | Construct the few-shot prompt from worked examples |
| `generate_rationales` | Generate rationales for a batch of problems using the current model |
| `filter_correct` | Parse generated answers and keep only correct ones |
| `rationalize` | Generate rationales with the correct answer as hint for failed examples |
| `finetune` | Fine-tune GPT-2 on the collected rationale dataset |
| `evaluate` | Measure accuracy on the test set |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Base model | GPT-J (6B parameters) | GPT-2 small (124M parameters) |
| Dataset size | 50,000 problems, 1-5 digits | 1,600 problems, 2 digits only |
| Outer-loop iterations | 16 (arithmetic), 36+ (GSM8K) | 3 |
| Fine-tuning steps | Increasing schedule, 300+ steps | 50 steps per iteration, constant |
| Sampling | Low-temperature sampling | Greedy decoding (temperature 0) |
| Few-shot examples | 10 per digit | 4 examples |
| Evaluation | Arithmetic, CommonsenseQA, GSM8K | 2-digit arithmetic only |
