---
title: "Dr. GRPO: Code architecture"
parent: "063 Dr. GRPO"
nav_order: 1
---
# Code Architecture — Dr. GRPO Notebook

## Overview

The notebook demonstrates GRPO's two optimization biases (length and difficulty) and implements the Dr. GRPO fix on a small MLP policy solving synthetic arithmetic problems with verifiable rewards. It trains two copies of an SFT-pretrained model — one with standard GRPO and one with Dr. GRPO — and compares their training dynamics. The result is measured as greedy/sampling accuracy on 200 held-out test problems and mean response length of correct vs incorrect responses.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (values from the code cell):
- `SEED = 42`
- `SFT_EPOCHS = 20`, `SFT_LR = 3e-3`, `SFT_DATA_SIZE = 3000`, `SFT_BATCH_SIZE = 256`
- `GROUP_SIZE = 16` (G: outputs per question; paper uses 16-64)
- `RL_ITERS = 20` (RL iterations for both GRPO and Dr. GRPO)
- `RL_LR = 5e-5` (policy learning rate; paper uses 1e-6)
- `RL_QUESTIONS_PER_ITER = 50`, `NUM_TEST_QUESTIONS = 200`
- `CLIP_EPS = 0.2` (PPO clipping epsilon)
- `HIDDEN_DIM = 128`, `MAX_ANSWER = 18`

### 2. Data: Synthetic Arithmetic Problems

**Function: `generate_arithmetic_problem(rng)`** — returns `(a, b, op, answer)` where a,b are 1-9, op is '+' or '-', subtraction clipped to non-negative. Features are a 3-vector `[a, b, op_flag]`.

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

**Function: `compute_reward(predicted, truth)`** — returns 1.0 if correct, 0.0 if incorrect (paper uses the same +1/0 outcome-verifiable reward, Section 3.2).

**Function: `sample_group(model, features, true_answer, group_size)`** — samples G answers from the policy distribution, computes per-response rewards, and returns the old probabilities for importance ratio computation.

**Function: `evaluate_model(model, test_data, greedy)`** — evaluates greedy (argmax) or sampling accuracy on the test set.

### 5. SFT Pre-training

Trains the MLP with cross-entropy on 3000 arithmetic examples for 20 epochs at lr=3e-3, giving it baseline ability before RL. The SFT model state is saved and cloned for both GRPO and Dr. GRPO runs to ensure a fair comparison.

### 6. GRPO vs Dr. GRPO Loss

**Function: `compute_group_advantages(rewards, remove_std_norm)`** — computes group-relative advantages:
- GRPO: `A = (R - mean) / std` (with std normalization)
- Dr. GRPO: `A = R - mean` (without std normalization, Section 3.2)

**Function: `policy_loss(model, features, sampled_answers, advantages, old_probs, response_lengths, remove_length_norm, remove_std_norm)`** — computes the clipped surrogate loss:
- Standard GRPO: divides per-token surrogate by response length \|o_i\| (the length bias term)
- Dr. GRPO: does not divide by \|o_i\| (removes the length bias, Section 3.2)
- Both use PPO clipping with ε=0.2
- No KL penalty (paper sets β=0, Section 3)

### 7. Length Bias Demonstration

A controlled experiment with 8 responses (4 correct, 4 incorrect) at different simulated lengths (2, 5, 10, 20 tokens). Shows that under GRPO, the effective gradient weight (advantage/\|o_i\|) for a length-20 incorrect response is 10x smaller than for a length-2 incorrect response — the longer wrong answer is penalized less. Dr. GRPO gives all responses the same weight regardless of length.

### 8. RL Training Loop

**Function: `run_rl_training(sft_model_state, use_dr_grpo, num_iters, questions, test_data)`** — runs 20 iterations of GRPO or Dr. GRPO:
1. For each question, sample G=16 responses from the old policy
2. Skip degenerate groups (all correct or all wrong — no learning signal)
3. Simulate variable response lengths (base length grows from 3 to ~7 over training, mimicking R1-Zero's response length increase)
4. Compute group-relative advantages (with or without std normalization)
5. Compute clipped surrogate loss (with or without length normalization)
6. One gradient step per iteration
7. Evaluate greedy and sampling accuracy on the test set
8. Log mean reward, accuracy, mean response lengths (correct vs incorrect)

### 9. Difficulty Bias Demonstration

Compares advantage magnitudes for three question types (easy 7/8 correct, medium 4/8 correct, hard 1/8 correct). Shows that GRPO assigns ~3x higher advantage magnitudes to easy/hard questions (low std) vs medium questions (high std), while Dr. GRPO gives all questions equal advantage magnitude.

### 10. Training Curves

Four plots: (1) mean group reward, (2) greedy accuracy with SFT baseline, (3) mean length of incorrect responses (key metric for length bias), (4) training loss.

### 11. Final Evaluation

Prints greedy and sampling accuracy for SFT baseline, GRPO, and Dr. GRPO, plus the length bias metric (incorrect_len - correct_len) for both algorithms.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `ArithmeticMLP` | 3-layer MLP policy mapping (a,b,op) to answer distribution |
| `compute_reward` | Verifiable +1/0 reward for correct/incorrect answers |
| `sample_group` | Sample G responses from the policy for a single question |
| `compute_group_advantages` | Group-relative advantage: GRPO uses (R-mean)/std, Dr. GRPO uses R-mean |
| `policy_loss` | Clipped surrogate loss with optional length normalization (GRPO vs Dr. GRPO) |
| `evaluate_model` | Evaluate greedy/sampling accuracy on test problems |
| `run_rl_training` | Full RL training loop for GRPO or Dr. GRPO |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Qwen2.5-Math-7B (7B params) | 3-layer MLP (~19K params) |
| Dataset | MATH level 3-5 (competition math) | Synthetic single-digit arithmetic |
| Group size | 16-64 | 16 |
| Learning rate | 1e-6 | 5e-5 (compensate for fewer iters) |
| RL iterations | Hundreds | 20 |
| Response length | Actual multi-token generation (thousands of tokens) | Simulated variable lengths (3-9 tokens) |
| KL penalty | Removed (β=0) | Removed (same) |
| Evaluation | AIME 2024 (43.3%) | 200 held-out arithmetic problems |
