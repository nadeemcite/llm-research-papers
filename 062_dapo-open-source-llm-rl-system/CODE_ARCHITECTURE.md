---
title: "DAPO: Code architecture"
parent: "062 DAPO"
nav_order: 1
---
# Code Architecture — DAPO Notebook

## Overview

The notebook implements the DAPO algorithm (four key techniques on top of GRPO) in PyTorch, applied to a small MLP policy that solves single-digit arithmetic problems with verifiable rewards. It runs a vanilla GRPO baseline and a full DAPO training loop side by side, then compares them with an ablation showing the incremental contribution of each technique. The result is measured as greedy and sampling accuracy on 100 held-out test problems.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (values from the code cell):
- `SEED = 42`
- `SFT_EPOCHS = 30`, `SFT_LR = 3e-3`, `SFT_DATA_SIZE = 5000`, `SFT_BATCH_SIZE = 256`
- `GROUP_SIZE = 16` (G: outputs per question, same as paper)
- `RL_ITERS = 30` (RL iterations for both GRPO and DAPO)
- `RL_LR = 1e-4` (policy learning rate; paper uses 1e-6)
- `CLIP_EPS_LOW = 0.2`, `CLIP_EPS_HIGH = 0.28` (decoupled clipping, Section 3.1)
- `GRPO_QUESTIONS_PER_ITER = 80`, `NUM_TEST_QUESTIONS = 200`
- `HIDDEN_DIM = 128`, `MAX_ANSWER = 18`
- `MAX_RESPONSE_LENGTH = 8`, `SOFT_PUNISH_CACHE = 2` (toy overlong shaping parameters)

### 2. Data: Synthetic Arithmetic Problems

**Function: `generate_arithmetic_problem(rng)`** — returns `(a, b, op, answer)` where a,b are 1-9, op is '+' or '-', subtraction is non-negative. Features are a 3-vector `[a, b, op_flag]`.

**Function: `generate_dataset(n, seed)`** — returns list of `(features, answer, question_string, answer_string)` tuples. 500 train, 200 test.

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

**Function: `compute_reward(predicted, truth)`** — returns 1.0 if correct, -1.0 if incorrect (paper uses +1/-1 rule-based reward, Section 2.4).

**Function: `soft_overlong_reward(base_reward, response_length, max_length, cache)`** — implements Equation 13: no penalty if length ≤ max - cache, linear penalty in the soft zone, -1 if exceeding max. Applied to a toy response-length proxy.

### 5. GRPO Baseline Loss (for comparison)

**Function: `grpo_loss_for_group(...)`** — standard GRPO loss with symmetric clipping [1-ε, 1+ε], sample-level averaging (mean per sample, then mean across samples), and KL penalty. Used as the baseline to compare against DAPO.

### 6. DAPO Loss (four techniques)

**Function: `dapo_loss_for_groups(group_data, ...)`** — implements all four DAPO techniques:
- **Clip-Higher:** Uses `torch.clamp(ratio, 1-CLIP_EPS_LOW, 1+CLIP_EPS_HIGH)` instead of symmetric clipping.
- **Dynamic Sampling:** Called with only filtered groups (prompts with 0 < correct < G); filtering happens in the training loop.
- **Token-Level Loss:** Averages across all tokens in the batch: `sum(per_token_losses) / sum(token_counts)` instead of `mean(per_sample_means)`.
- **No KL penalty:** DAPO removes KL divergence (Section 2.3).

### 7. SFT Pre-training

Trains the MLP with cross-entropy on 5000 arithmetic examples for 30 epochs at lr=3e-3, giving it a baseline ability before RL. Copies SFT weights to a frozen reference model.

### 8. GRPO Baseline Training Loop

Runs 30 iterations of vanilla GRPO with lr=1e-4: sample groups, compute rewards, compute group-relative advantages, one gradient step per iteration. Logs mean reward and accuracy per iteration.

### 9. DAPO Training Loop

Runs 30 iterations of DAPO with all four techniques:
1. Sample a batch of questions
2. For each question, sample G=16 responses from the old policy
3. **Dynamic Sampling filter:** skip questions where all G responses are correct or all wrong (0 < correct < G required)
4. Compute verifiable rewards (with overlong shaping if applicable)
5. Compute group-relative advantages
6. Compute DAPO loss with Clip-Higher and token-level averaging
7. One gradient step, evaluate on test set

### 10. Ablation Study

Runs progressive DAPO stages: GRPO baseline → +Clip-Higher → +Dynamic Sampling → +Token-Level Loss → +Overlong Shaping, printing accuracy at each stage to show incremental contributions (mirroring the paper's Table 1).

### 11. Training Curves

Three plots: (1) mean group reward over iterations for GRPO vs DAPO, (2) greedy/sampling accuracy over iterations, (3) loss comparison.

### 12. Final Evaluation

Prints greedy and sampling accuracy for both GRPO baseline and DAPO, with improvement deltas and sample outputs.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_arithmetic_problem` | Create a random arithmetic problem with known answer |
| `ArithmeticMLP` | 3-layer MLP policy mapping (a,b,op) to answer distribution |
| `compute_reward` | Verifiable +1/-1 reward for correct/incorrect answers |
| `soft_overlong_reward` | Soft length-based penalty (Equation 13) |
| `compute_group_advantages` | Group-relative advantage: (r_i - mean) / std |
| `grpo_loss_for_group` | Standard GRPO loss with symmetric clipping (baseline) |
| `dapo_loss_for_groups` | DAPO loss with Clip-Higher, token-level averaging, no KL |
| `sample_group` | Sample G responses from the policy distribution |
| `evaluate_model` | Evaluate greedy/sampling accuracy on test problems |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Qwen2.5-32B (32B params) | 3-layer MLP (~19K params) |
| Dataset | DAPO-Math-17K (competition math) | Synthetic single-digit arithmetic |
| Group size | 16 (same) | 16 |
| Learning rate | 1e-6 | 2e-4 (compensate for fewer iters) |
| RL iterations | Hundreds | 30 |
| Max generation | 20480 tokens | Toy 8-step proxy |
| KL penalty | Removed (same) | Removed (same) |
| Evaluation | AIME 2024 avg@32 | 100 held-out arithmetic problems |
