---
title: "Test-Time Compute: Code architecture"
parent: "058 Test-Time Compute"
nav_order: 1
---
# Code Architecture — Test-Time Compute Notebook

## Overview

The notebook implements a simplified comparison of three test-time compute strategies — best-of-N, sequential revisions, and compute-optimal — using small MLP models on synthetic arithmetic problems. A generator MLP predicts a distribution over answers (0-199), a reward model scores candidate answers, and difficulty is estimated by sampling pass rates. The result is measured as the percentage of test problems solved at each compute budget, both overall and per difficulty bin. The compute-optimal strategy at budget 4 (56.0%) matches best-of-N at budget 16 (44.0%), achieving a 4.0x efficiency gain.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell):
- `N_TRAIN = 3000` — training problems (paper: MATH dataset, 12k)
- `N_TEST = 300` — test problems (paper: 500)
- `EPOCHS = 60` — generator training epochs
- `RM_EPOCHS = 40` — reward model training epochs
- `LR = 1e-3`, `BATCH_SIZE = 256`
- `HIDDEN = 256`, `N_LAYERS = 3` — MLP architecture
- `ANSWER_DIM = 200` — answer space 0-199
- `DIFFICULTY_BUDGETS = [4, 16, 64]` — compute budgets for evaluation

### 2. Data

**Function: `generate_problem(rng, difficulty)`** — returns `(problem_vec[5], answer_int)`. Generates arithmetic problems at 5 difficulty levels (1-5), controlled by operand digit count. Problem vector encodes tens/ones digits of both operands plus operation type, all normalized to 0-1. Answer is clamped to 0-199.

### 3. Model

**Class: `Generator`** (about 185K parameters) — 3-layer MLP with GELU activations that maps a 5-feature problem vector to a distribution over 200 answers. `sample()` draws candidates via multinomial sampling with temperature.

**Class: `RewardModel`** (about 185K parameters) — 3-layer MLP that takes `[problem_vec(5), answer_onehot(200)]` and outputs a scalar correctness score.

**Data flow:**
```
Problem vector (5,) -> Generator -> logits (200,) -> softmax -> sample N answers
Problem + answer (205,) -> RewardModel -> score (1,) -> select best
```

### 4. Training Data Preparation

Generator training: 3000 (problem_vec, answer) pairs. Reward model training: 6000 (problem_vec, answer, label) pairs — one correct and one perturbed-wrong per problem.

### 5. Training Loop

- **Generator:** AdamW (lr 1e-3), 60 epochs, cross-entropy loss over 200 answer classes. Final train accuracy: 87.5%.
- **Reward model:** AdamW (lr 1e-3), 40 epochs, binary cross-entropy loss. Final train accuracy: 59.2% (limited by overlap between correct and perturbed-wrong answers).

### 6. Training Curves

Plots generator loss, generator accuracy, reward model loss, and reward model accuracy over epochs. Generator shows clean convergence; reward model plateaus at about 59% due to the difficulty of distinguishing near-miss wrong answers from correct ones.

### 7. Test-Time Compute Strategies

Three functions:
- `best_of_n()` — samples N independent answers, selects best by reward model score
- `sequential_revisions()` — samples N answers sequentially with decreasing temperature (simulating iterative refinement)
- `compute_optimal_strategy()` — allocates budget between parallel and sequential based on difficulty bin (80% sequential for easy, 80% parallel for hard)

### 8. Estimating Question Difficulty

**Function: `estimate_difficulty()`** — samples 50 answers per question, computes pass rate (fraction correct). Questions binned into 5 quantiles (1=easiest, 5=hardest). Plots pass rate distribution and mean per bin.

### 9. Comparing Test-Time Strategies

Evaluates all three strategies at budgets 4, 16, 64 on all 300 test problems. Results show compute-optimal @ 4 matches best-of-N @ 16 (4x efficiency), and sequential revisions outperform parallel at higher budgets.

### 10. Results and Plots

Three plot sections:
- Overall accuracy vs budget (log-scale x-axis) — compute-optimal matches or beats best-of-N
- Per-difficulty-bin accuracy at each budget — shows sequential helps on harder bins
- Compute-optimal efficiency annotation showing the 4x gain

### 11. FLOPs-Matched Comparison

Trains a 2x larger generator (375K params, 25 epochs) and compares greedy decoding (no test-time compute) against the small model + compute-optimal (budget 64). Results show small+CO wins on easy bins, larger model wins on hard bins.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_problem` | Generate arithmetic problem at given difficulty level |
| `Generator` | MLP that predicts answer distribution from problem features |
| `RewardModel` | MLP that scores (problem, answer) pairs for correctness |
| `sample_answers` | Sample N candidate answers from generator at given temperature |
| `score_answers` | Score candidate answers using reward model |
| `best_of_n` | Best-of-N: parallel sampling + verifier selection |
| `sequential_revisions` | Sequential revisions: decreasing-temperature sampling |
| `compute_optimal_strategy` | Adaptive allocation based on difficulty bin |
| `estimate_difficulty` | Estimate pass rate from 50 samples, bin into 5 levels |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Base model | PaLM 2-S* (billions of params) | MLP (185K params) |
| Dataset | MATH benchmark (competition math) | Synthetic 1-2 digit arithmetic |
| Answer generation | Autoregressive text generation | Direct classification over 200 answers |
| Reward model | PRM with per-step Monte Carlo rollouts | Binary classifier with one-hot answer encoding |
| Sequential revisions | Fine-tuned revision model conditioning on previous attempts | Decreasing-temperature sampling |
| Difficulty estimation | 2048 samples per question | 50 samples per question |
| FLOPs comparison | 14x larger model | 2x larger model (25 epochs) |
| Optimizer / epochs | Not specified (large-scale training) | AdamW (lr 1e-3), 60 gen / 40 RM / 25 larger epochs |
