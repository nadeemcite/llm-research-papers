---
title: "GRPO: Code architecture"
parent: "059 GRPO"
nav_order: 1
---
# Code Architecture — GRPO Notebook

## Overview

The notebook implements Group Relative Policy Optimization (GRPO) from scratch in PyTorch. It first supervised-fine-tunes (SFT) a 3-layer MLP (~19K parameters) on synthetic single-digit arithmetic, then applies GRPO with verifiable rewards to refine the policy. The result is measured by greedy and sampling accuracy on 100 held-out test problems before and after GRPO training.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `SFT_EPOCHS = 10`, `SFT_LR = 1e-3`, `SFT_DATA_SIZE = 3000`
- `GROUP_SIZE = 16` — G, outputs sampled per question (paper uses 64)
- `GRPO_ITERS = 20` — number of GRPO iterations (paper uses many more)
- `GRPO_LR = 2e-4` — policy learning rate (paper uses 1e-6)
- `KL_COEF = 0.04` — beta, KL divergence coefficient (same as paper)
- `CLIP_EPS = 0.2` — PPO clipping epsilon (standard value)
- `GRPO_QUESTIONS_PER_ITER = 50`
- `HIDDEN_DIM = 128`, `MAX_ANSWER = 18`

### 2. Data

**Function: `generate_arithmetic_problem(rng)`** — generates a random single-digit arithmetic problem (addition or subtraction, a >= b for subtraction) and returns `(a, b, op, answer)`.

**Function: `problem_to_features(a, b, op)`** — converts to a 3-element feature vector `[float(a), float(b), 1.0 if op == '+' else 0.0]`.

**Function: `generate_dataset(n, seed)`** — generates n problems as `(features, answer, qstr, astr)` tuples.

### 3. Model: Arithmetic MLP

**Class: `ArithmeticMLP`** (~19,475 parameters)
- Input: 3 features (a, b, op)
- Architecture: Linear(3, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 19)
- Output: logits over integers 0..18 (19 classes)
- The policy is the softmax of the output; sampling from it is the "generation" step

### 4. Supervised Fine-Tuning (SFT)

Trains the MLP with cross-entropy loss on 3000 arithmetic examples for 10 epochs with AdamW (lr 1e-3). This gives the model a baseline ability to answer arithmetic questions. The SFT model is then copied to the frozen reference model for the KL penalty.

### 5. Reward Function

**Function: `compute_reward(predicted_answer, ground_truth_answer)`** — verifiable reward: returns 1.0 if the predicted answer matches the ground truth, 0.0 otherwise. This is the "rule judgment" reward from Section 4.2.

### 6. Evaluation

**Function: `evaluate_model(model, test_problems, greedy)`** — evaluates accuracy in greedy mode (argmax) or sampling mode (multinomial). Returns accuracy and per-problem results.

### 7. GRPO Core: Advantage Estimation

**Function: `compute_group_advantages(rewards)`** — computes $\hat{A}_i = (r_i - \text{mean}) / \text{std}$. Returns zeros if all rewards are identical (no gradient signal).

### 8. GRPO Loss Function

**Function: `grpo_loss_for_group(...)`** — computes the GRPO loss for one group of G samples:
1. Forward pass through policy model to get new log-probs
2. Importance ratio: $r = \exp(\log \pi_\theta - \log \pi_{old})$
3. PPO clipped objective: $\min(r \hat{A}, \text{clip}(r, 1-\varepsilon, 1+\varepsilon) \hat{A})$
4. KL penalty (unbiased estimator): $D_{KL} = \exp(\log \pi_{ref} - \log \pi_\theta) - (\log \pi_{ref} - \log \pi_\theta) - 1$
5. Total loss = policy_loss + beta * kl_loss

### 9. GRPO Training Loop

For each iteration:
1. **Exploration:** For each question, sample G answers from the old policy, compute verifiable rewards, compute group-relative advantages
2. **Update:** Accumulate GRPO loss across all groups, backward pass, gradient clipping (max_norm=1.0), optimizer step
3. **Evaluate:** Greedy and sampling accuracy on test set

### 10-12. Training Curves, Final Evaluation, Comparison

Plots mean reward, greedy/sampling accuracy, and loss per iteration. Prints final vs baseline metrics and side-by-side comparison of the SFT-only (reference) model vs the GRPO-trained model.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_arithmetic_problem` | Generate a random single-digit arithmetic problem |
| `problem_to_features` | Convert (a, b, op) to MLP input features |
| `ArithmeticMLP` | 3-layer MLP policy mapping (a, b, op) to answer distribution |
| `compute_reward` | Verifiable reward: 1.0 if correct, 0.0 otherwise |
| `evaluate_model` | Evaluate greedy or sampling accuracy |
| `compute_group_advantages` | Normalize rewards to group-relative advantages |
| `grpo_loss_for_group` | Compute GRPO loss (PPO clip + KL penalty) for one group |
| `sample_group` | Sample G answers from the policy distribution |
| `get_ref_log_probs` | Get log-probs from the frozen reference model |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | DeepSeekMath 7B (7B params) | 3-layer MLP (~19K params) |
| Dataset | Competition-level MATH, GSM8K | Synthetic single-digit arithmetic |
| Pre-training | 120B math tokens from Common Crawl | 10 SFT epochs on 3000 examples |
| Group size | G = 64 | G = 16 |
| RL iterations | Many (long training) | 20 |
| Questions per iteration | 1024 (batch size) | 50 |
| Learning rate | 1e-6 | 2e-4 |
| Reward model | Trained reward model + rule-based | Rule-based only (verifiable) |
| Policy updates per iteration | Multiple | Single (ratio ~ 1.0, clipping minimal) |
