---
title: "Learning to Discover: Code architecture"
parent: "068 Learning to Discover"
nav_order: 1
---
# Code Architecture — TTT-Discover Notebook

## Overview

The notebook implements the core TTT-Discover algorithm (entropic objective + PUCT state reuse + test-time policy training) on a 4D multimodal optimization problem. The policy is a 17,672-parameter MLP that proposes incremental deltas to improve a solution vector, gets evaluated by a reward function (sum of Gaussian wells), and is updated at test time. Results show TTT-Discover finds a solution with reward 0.987, beating both Best-of-N (~0) and No-TTT PUCT-only search (0.973) with the same sampling budget.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `SOLUTION_DIM = 4` — dimension of the solution vector (toy problem)
- `TTT_STEPS = 100` — number of test-time training steps (paper: 50)
- `ROLLOUTS = 256` — rollouts per step (paper: 512)
- `GROUP_SIZE = 16` — rollouts sharing the same initial state (paper: 64)
- `LR = 5e-4` — learning rate for test-time training
- `KL_LAMBDA = 0.1` — KL penalty coefficient (lambda in paper)
- `PUCT_C = 2.0` — PUCT exploration coefficient
- `BUFFER_SIZE = 100` — max buffer entries for state reuse
- `BETA_MAX = 4.0` — maximum entropic temperature
- `DELTA_SCALE = 1.0` — max magnitude of policy-proposed change
- `HIDDEN_DIM = 128` — hidden layer width in the policy network

### 2. Discovery Problem: 4D Multimodal Optimization

**Function: `reward_fn(x_np)`** — takes a numpy array of shape (4,), returns a scalar reward. The reward is a sum of 3 Gaussian wells centered at fixed locations in 4D space, each with a weight in [0.5, 1.0].

The initial "state of the art" is a point far from all wells (reward near zero), giving the policy ample room to discover improvements. The true optimum is estimated by checking well centers plus brute-force sampling.

### 3. Policy Network

**Class: `PolicyNetwork`** (17,672 parameters)
- Input: current solution x (4 dims)
- Architecture: Linear(4, 128) → Tanh → Linear(128, 128) → Tanh → Linear(128, 4)
- Output: delta_mean (4 dims) and a learnable log_std (4 dims, clamped to [-3, 0.5])
- `sample(x, n_samples)`: generates n solutions by sampling Gaussian deltas, returns new solutions, log-probs, mean, std

**Data flow:**
```
Input: (4,) current solution
  -> Linear(4, 128) -> Tanh -> (128,)
  -> Linear(128, 128) -> Tanh -> (128,)
  -> Linear(128, 4) -> (4,) delta mean
  + learnable log_std (4,) -> Gaussian sampling -> (n, 4) new solutions
```

### 4. TTT-Discover Algorithm Components

**Function: `entropic_advantage(rewards, log_probs, log_probs_ref, beta, kl_lambda)`**
- Computes the entropic weight w_beta(a) = exp(beta * R) / E[exp(beta * R)] with numerical stability (shift by max reward)
- Advantage: A(a, s) = w_beta(a) - 1 - lambda * log(pi_theta(a\|s) / pi_theta0(a\|s))
- Returns: (G,) advantage tensor

**Function: `adaptive_beta(rewards, step, total_steps, beta_min=0.5, beta_max=4.0)`**
- Linear schedule: beta = beta_min + (beta_max - beta_min) * step / (total_steps - 1)
- Starts at 0.5 for stability, grows to 4.0 to focus on maximum
- Simplification: the paper uses KL-constrained adaptive beta; we use a linear schedule

**Class: `PUCTBuffer`**
- Stores up to 80 entries: {perm, reward, max_child_reward, n_expansions}
- `select()`: scores each entry by Q(s) + c · P(s) · sqrt(1+T) / (1+n(s)), where Q = max child reward, P ∝ 1/rank
- `add(perm, reward)`: adds a new tour; evicts worst entry when full
- `update_child(parent_idx, child_reward)`: updates max child reward and expansion count

**Function: `reward_batch(x_tensor)`** — vectorized reward computation for a batch of solutions.

### 5. TTT-Discover Main Loop

- Optimizer: Adam, lr 5e-4, gradient clipping at norm 0.5
- Reference policy: frozen copy of initial policy for KL penalty
- For each of 100 steps:
  1. Select initial state via PUCT from buffer
  2. Generate 16 groups of 16 rollouts (256 total)
  3. Evaluate rewards via `reward_batch()`
  4. Add solutions to buffer; update PUCT statistics
  5. Compute adaptive beta (linear schedule)
  6. Compute entropic advantages per group; policy gradient loss = -E[advantage * log_pi]
  7. Take 1 gradient step (no off-policy steps, matching the paper)
- Logs: global best reward, mean reward, max reward per step, adaptive beta

### 6. Baselines

**Best-of-N:** Samples n_total = 25600 solutions from the frozen initial policy starting from the sota point. Reports the best reward found.

**No-TTT (PUCT only):** Uses the PUCT buffer and state reuse but never updates the policy weights. Same total sampling budget. Tests whether search alone (without learning) can match TTT.

### 7. Training Curves

Three subplots:
1. **Best reward over TTT steps:** TTT-Discover best, per-step max, baseline lines (Best-of-N, No-TTT, initial sota, true optimum)
2. **Mean reward per step:** shows policy improvement over time
3. **Adaptive beta:** shows the linear temperature schedule from 0.5 to 4.0

### 8. Final Evaluation

Prints all key metrics: 2-opt optimum tour length, initial sota, TTT-Discover best, Best-of-N best, No-TTT best, percentage improvements, and pairwise comparisons.

### 9. Visualize Discovered Solution

Contour plot of the reward landscape projected onto the first 2 dimensions, with markers for the initial sota, TTT-Discover solution, Best-of-N solution, and No-TTT solution.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `reward_fn` | Continuous reward function (mixture of Gaussian wells) |
| `reward_batch` | Vectorized reward for a batch of solutions |
| `PolicyNetwork` | Tiny MLP policy that generates incremental deltas |
| `entropic_advantage` | Computes entropic weights and KL-penalized advantages |
| `adaptive_beta` | Linear entropic temperature schedule |
| `PUCTBuffer` | Buffer with PUCT-based state selection for reuse |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Problem | Erdős, GPU kernels, AtCoder, biology | 4D multimodal optimization |
| Policy | gpt-oss-120b (120B params) with LoRA rank 32 | MLP with 17,672 parameters |
| TTT steps | 50 | 100 |
| Rollouts/step | 512 (8 groups of 64) | 256 (16 groups of 16) |
| Actions | LLM-generated code/algorithms | Incremental delta proposals |
| Infrastructure | Tinker API (Thinking Machines) | Local CPU/GPU |
| Adaptive beta | KL-constrained per initial state | Linear schedule (0.5 → 4.0) |
| Cost | ~$500 per problem | Free (local compute) |
| Context window | 32,768 tokens on Tinker | N/A (direct parameterization) |
