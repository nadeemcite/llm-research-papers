---
title: "Process Reward Models: Code architecture"
parent: "057 Process Reward Models"
nav_order: 1
---
# Code Architecture — Process Reward Models Notebook

## Overview

The notebook implements a simplified Process Reward Model (PRM) vs Outcome Reward Model (ORM) comparison using GPT-2 small (124M parameters) on synthetic 2-step arithmetic problems. It generates solutions with known step-level correctness labels, trains both reward models with binary cross-entropy loss, and evaluates them via best-of-N reranking on held-out test problems. The result is measured as the percentage of test problems solved (correct solution selected) at each N.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `SEED = 42` — reproducibility
- `N_TRAIN_PROBLEMS = 200` — training problems (paper: MATH dataset, thousands)
- `N_TEST_PROBLEMS = 50` — held-out test problems
- `N_SOLUTIONS_PER_PROBLEM = 5` — solutions per problem (paper: up to 1,860)
- `EPOCHS = 15` — reward model training epochs
- `BATCH_SIZE = 8`, `LR = 5e-5` — AdamW optimizer
- `MAX_NEW_TOKENS = 80` — max solution text length
- `BEST_OF_N_VALUES = [1, 2, 3, 5]` — N values for evaluation

### 2. Data Generation

**Function: `generate_arithmetic_problem(rng)`** — returns `(problem_str, steps, final_answer)`. Generates random 2-step arithmetic problems (e.g., "12 + 34 - 5") with intermediate results tracked, giving ground-truth step-level labels.

**Function: `format_solution(problem_str, steps, final_answer)`** — formats a correct solution as newline-delimited text.

### 3. Model & Tokenizer

**Model: GPT-2 small (124M)** loaded from Hugging Face. Two separate copies are created: one for the PRM and one for the ORM.

**Class: `RewardModel(nn.Module)`** — wraps GPT-2's transformer body with a linear classification head (`nn.Linear(n_embd, 1)`) that produces per-token scores. Returns `(batch, seq_len)` logits.

### 4. Solution Generation

**Function: `generate_correct_solution`** — produces a correct step-by-step solution string.

**Function: `generate_incorrect_solution`** — introduces an arithmetic error at a specific or random step, propagates the wrong result to downstream steps, and returns step-level labels (1 for correct, 0 for incorrect).

**Function: `generate_solution_pool`** — generates a pool of N solutions per problem (1 correct + N-1 incorrect), shuffled.

**Data flow:**
```
Problem (str) → generate_solution_pool → [(solution_text, step_labels, is_correct), ...]
```

### 5. Tokenization & Label Preparation

**Function: `find_step_end_positions`** — tokenizes solution text and finds the token index at the end of each step line. Returns `(input_ids, step_end_positions, step_labels)`.

**Function: `prepare_prm_batch`** — creates batches for PRM training. Token labels are -1 (ignore) everywhere except at step-end positions, where they are 1 (correct) or 0 (incorrect). Padded to `max_len=256`.

**Function: `prepare_orm_batch`** — creates batches for ORM training. Token labels are -1 everywhere except the last non-pad token, where the label is 1 (correct) or 0 (incorrect).

### 6. Training

**Function: `train_reward_model`** — trains a reward model with BCE-with-logits loss at labeled positions. Shuffles data each epoch, tracks loss and label-level accuracy.

- Optimizer: AdamW (lr 5e-5)
- Loss: `F.binary_cross_entropy_with_logits` at labeled token positions only
- 15 epochs, batch size 8

### 7. Scoring

**Function: `prm_score_solution`** — scores a solution with the PRM. Finds step-end positions, gets per-step correctness probabilities via sigmoid, and returns the **product** of step probabilities (paper's best strategy, Appendix F.2).

**Function: `orm_score_solution`** — scores a solution with the ORM. Returns the sigmoid probability at the final token.

### 8. Best-of-N Reranking

**Function: `best_of_n_eval`** — for each test problem and each N value:
1. Take N candidate solutions
2. PRM: select the solution with highest PRM score
3. ORM: select the solution with highest ORM score
4. Majority voting: select the most common final answer
5. Random: select a random solution
6. Check if the selected solution is correct

### 9. Visualization

- **Training curves:** Loss and accuracy for PRM vs ORM across epochs
- **Best-of-N plot:** % solved vs N for PRM, ORM, majority voting, random
- **Step-level PRM visualization:** Bar chart of per-step PRM scores for one correct and one incorrect solution (reproducing the paper's Figure 2 concept)

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_arithmetic_problem` | Create a random 2-step arithmetic problem with ground-truth steps |
| `generate_solution_pool` | Generate a pool of correct and incorrect solutions for one problem |
| `RewardModel` | GPT-2 with a binary classification head for step/solution scoring |
| `find_step_end_positions` | Tokenize solution and locate step boundaries |
| `prepare_prm_batch` | Create training batch with step-level labels |
| `prepare_orm_batch` | Create training batch with outcome-level labels |
| `train_reward_model` | Train PRM or ORM with BCE loss |
| `prm_score_solution` | Score a solution as product of per-step probabilities |
| `orm_score_solution` | Score a solution by final-token probability |
| `best_of_n_eval` | Evaluate PRM, ORM, majority voting, random via best-of-N |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Base model | GPT-4 (large-scale), GPT-4-like (small-scale) | GPT-2 small (124M) |
| Dataset | MATH dataset (competition math) | Synthetic 2-step arithmetic |
| Labels | 800K human step-level labels (PRM800K) | Ground-truth from problem generation |
| Training scale | Large-scale, MathMix pretraining | 1,000 solutions, 15 epochs, AdamW |
| Best-of-N | N up to 1,860 | N up to 5 |
| Active learning | Yes (PRM_selector, 80/20 split) | No |
| OOD evaluation | AP Physics, Calculus, Chemistry, AMC | No |
| Generator | Fine-tuned GPT-4, step-by-step format | Programmatic solution generation |
