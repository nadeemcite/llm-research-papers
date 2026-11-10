---
title: "RL vs Base Model: Code architecture"
parent: "064 RL vs Base Model"
nav_order: 1
---
# Code Architecture — RL vs Base Model Notebook

## Overview

The notebook reproduces the paper's core pass@k crossover experiment on a small scale. It trains a 3-layer MLP on synthetic single-digit arithmetic via SFT (the "base model"), then applies GRPO-style RL with verifiable rewards (the "RL model"). It computes pass@k for both models across k=1..64 on 200 held-out test problems and plots the curves to show the crossover.

The RL model wins at small k but the base model overtakes it at large k. RL wins at pass@1 (0.3864 vs 0.3855) but base wins at pass@2 through pass@64 (0.9950 vs 0.4150 at k=64), with crossover at k=2. The base model solves 58% of problems the RL model cannot.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (values from the code cell):
- `SEED = 42`
- `SFT_EPOCHS = 10`, `SFT_LR = 1e-3`, `SFT_DATA_SIZE = 3000`, `SFT_BATCH_SIZE = 256`
- `GROUP_SIZE = 16` (G: outputs per question; paper uses 16-64)
- `RL_ITERS = 40` (RL iterations)
- `RL_LR = 1e-3` (policy learning rate; paper uses 1e-6)
- `RL_QUESTIONS_PER_ITER = 50`, `NUM_TEST = 200`
- `CLIP_EPS = 0.2` (PPO clipping epsilon)
- `HIDDEN_DIM = 128`, `MAX_ANSWER = 18`, `NUM_SAMPLES = 64` (n for pass@k)

### 2. Data: Synthetic Arithmetic Problems

**Function: `generate_problem(rng)`** — returns `(a, b, op, answer)` where a,b are 1-9, op is '+' or '-', subtraction clipped to non-negative. Features are a 3-vector `[a, b, op_flag]` where op_flag is 0 for '+' and 1 for '-'.

**Function: `generate_dataset(n, seed)`** — returns list of `(features, answer)` tuples. 3000 train, 200 test, 50 RL questions.

### 3. Model: Arithmetic MLP

**Class: `ArithmeticMLP`** (about 19,475 parameters)
- `Linear(3, 128)` → ReLU → `Linear(128, 128)` → ReLU → `Linear(128, 19)` (output: logits over answers 0..18)
- Policy is the softmax of the output; sampling from this distribution is the "generation" step.

**Data flow:**
```
Input: (3,) feature vector [a, b, op]
  -> Linear -> ReLU -> Linear -> ReLU -> Linear -> (19,) logits
  -> softmax -> (19,) probability distribution over answers
```

### 4. Reward Function

**Function: `compute_reward(predicted, truth)`** — returns 1.0 if correct, 0.0 if incorrect (paper uses the same +1/0 outcome-verifiable reward).

**Function: `sample_group(model, features, true_answer, group_size)`** — samples G answers from the policy distribution, computes per-response rewards, and returns the old log-probabilities for importance ratio computation.

### 5. SFT Pre-training

Trains the MLP on 3000 arithmetic examples for 10 epochs with Adam (lr=1e-3) and cross-entropy loss. This produces the "base model" — analogous to the pretrained LLM in the paper. The SFT model's state is saved and cloned for RL training.

### 6. RL Training (GRPO-style)

Implements a simplified GRPO update:
- For each iteration: sample 50 questions, generate G=16 responses each, compute rewards, compute group-relative advantages $A_i = R_i - \text{mean}(R)$ (no length or std normalization, following Dr. GRPO), compute importance ratio $r = \pi_\theta / \pi_{\text{old}}$, apply PPO clipping with $\varepsilon = 0.2$.
- No KL penalty (paper sets β=0).
- Learning rate 1e-3, 40 iterations.

**Function: `run_rl_training(sft_state, num_iters, questions, test_data)`** — returns the trained RL model and training history (mean reward, unique answers per question).

### 7. pass@k Computation

**Function: `compute_pass_at_k(model, test_data, n_samples, k_values)`** — for each test problem, samples n=64 responses and computes pass@k for k in [1, 2, 4, 8, 16, 32, 64] using the formula $\text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$ where $c$ is the number of correct samples out of $n$. This is the same formula as the paper (Section 2.1, Appendix A.2).

### 8. Results and Plots

- **Pass@k crossover plot:** Plots pass@k for base vs RL model across k values. Shows the crossover point where the base model overtakes the RL model.
- **Coverage comparison:** Bar chart showing the number of problems solvable by base only, RL only, both, and neither (analogous to Table 2 in the paper).
- **Response diversity:** Plots the mean number of unique answers per problem for base vs RL model, showing that RL reduces output diversity (lower entropy).

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_problem` | Create a single arithmetic problem with verifiable answer |
| `generate_dataset` | Generate train/test/RL datasets |
| `ArithmeticMLP` | 3-layer MLP policy mapping (a, b, op) to answer distribution |
| `compute_reward` | Verifiable +1/0 reward |
| `sample_group` | Sample G responses from the policy for a single question |
| `run_rl_training` | GRPO-style RL training loop with PPO clipping |
| `compute_pass_at_k` | Compute pass@k for a range of k values using n samples |
| `plot_pass_at_k` | Plot the crossover curve for base vs RL model |
| `plot_coverage` | Bar chart of solvable-problem overlap between base and RL |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Qwen2.5-7B/14B/32B, LLaMA-3.1-8B, DeepSeek-R1-Distill | 3-layer MLP (~19K params) |
| Data | AIME24, MATH500, GSM8K, LiveCodeBench, MathVista | Synthetic single-digit arithmetic |
| pass@k range | k up to 1024, n up to 1024 | k up to 64, n=64 |
| RL algorithms | PPO, GRPO, Reinforce++, RLOO, ReMax, DAPO | GRPO-style only |
| Metrics | pass@k, coverage, perplexity, Δ_SE, entropy | pass@k, coverage, response diversity |
| Evaluation | Temperature 0.6, top-p 0.95, 16K tokens | Exact sampling from discrete distribution |
| Scale | 8×A100 GPUs, hours to days | CPU, seconds |
