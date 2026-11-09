---
title: "DeepSeek-R1: Code architecture"
parent: "060 DeepSeek-R1"
nav_order: 1
---
# Code Architecture — DeepSeek-R1 Notebook

## Overview

The notebook implements R1-Zero style training: pure RL with GRPO on a small transformer base model, applied to the Countdown task. A 2-layer transformer (~108K parameters) is first pre-trained on mixed arithmetic and formatted examples, then trained with GRPO using rule-based rewards (accuracy + format). The result is measured by response length, accuracy, and format compliance on 50 held-out test problems.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `PRETRAIN_EPOCHS = 8`, `PRETRAIN_LR = 1e-3`, `PRETRAIN_DATA_SIZE = 8000`, `PRETRAIN_BATCH_SIZE = 256`
- `GROUP_SIZE = 8` — G, outputs sampled per question (paper uses 64)
- `RL_ITERS = 120` — number of GRPO iterations (paper uses thousands)
- `RL_LR = 3e-4` — policy learning rate (paper uses 1e-6)
- `KL_COEF = 0.04` — beta, KL divergence coefficient (same as paper)
- `CLIP_EPS = 0.2` — PPO clipping epsilon (standard value)
- `PROBLEMS_PER_ITER = 12` — questions per iteration (paper uses 1024)
- `MAX_GEN_LEN = 48` — maximum generated tokens
- `D_MODEL = 64`, `N_HEADS = 4`, `N_LAYERS = 2`, `MAX_LEN = 96`

### 2. Vocabulary & Tokenizer

**Vocabulary: 25 tokens** — 10 special tokens (`<pad>`, `<bos>`, `<eos>`, `<nums>`, `<target>`, `<sep>`, ` `, ` `, `<answer>`, `</answer>`) and 15 character tokens (digits 0-9, operators +-* , equals =, space).

**Functions:**
- `encode_with_specials(text)` — encodes text containing multi-char special tokens and single characters into token IDs
- `decode_tokens(ids)` — decodes token IDs back to text, expanding special tokens

### 3. Countdown Task

**Function: `generate_countdown_problem(rng)`** — generates 4 random numbers (1-9) and a reachable target (1-99) by combining a subset of the numbers with random operations.

**Function: `build_prompt(numbers, target)`** — builds a fixed-length 14-token prompt: `<nums> <sep> D <sep> D <sep> D <sep> D <sep> <target> DD `

### 4. Reward Functions

**Function: `parse_output(token_ids)`** — parses generated tokens to extract the think section (between ` `), answer section (between `<answer>` and `</answer>`), and whether the format is valid.

**Function: `safe_eval_expr(expr, available_numbers)`** — safely evaluates an arithmetic expression (only digits and +-*), checks that all numbers used are available (with multiplicity).

**Function: `compute_r1_reward(token_ids, target, available_numbers)`** — R1-Zero style reward: format reward (0.1) if the output uses ` /<answer>` structure, accuracy reward (0.9) if the answer evaluates to the target using only available numbers.

### 5. Model: TinyTransformer

**Class: `TinyTransformer`** (~107,840 parameters)
- Token embedding: 25 × 64 = 1,600 (shared with output head via weight tying)
- Position embedding: 96 × 64 = 6,144
- 2 × TransformerEncoderLayer(64, 4 heads, FFN 256, GELU, norm_first=True)
- Final LayerNorm + linear head (tied to embedding)
- Causal mask applied via `src_mask` parameter

**Data flow:**
```
Input: (B, T) token IDs
  -> token_emb + pos_emb: (B, T, 64)
  -> 2x TransformerEncoderLayer with causal mask: (B, T, 64)
  -> LayerNorm: (B, T, 64)
  -> Linear head: (B, T, 25) logits
```

### 6. Pre-training: Base Model

**Function: `generate_pretrain_ids(rng)`** — generates pre-training examples in four formats:
- `plain`: `<bos> 3+5=8 <eos>` (arithmetic only)
- `formatted`: `<bos>  3+5=8  <answer> 3+5=8 </answer> <eos>` (full template)
- `formatted_random`: think section with random text + arithmetic
- `formatted_think`: think section with multiple arithmetic steps

This teaches the model both arithmetic and the R1-Zero template format, analogous to how DeepSeek-V3-Base was pre-trained on general text before RL.

Training: 8 epochs, AdamW (lr 1e-3), batch size 256, cross-entropy loss with padding mask, gradient clipping (max_norm=1.0).

### 7. GRPO Core

**Function: `generate_completions_batch(model, prompt_batch, max_new_tokens, temperature)`** — autoregressive generation for a batch of identical prompts. Returns the full token sequence and a generation mask (1 for non-pad tokens, 0 for pad).

**Function: `get_log_probs(model, tokens, prompt_len)`** — computes log-probabilities of generated tokens. Logits at position `i` predict token at position `i+1`; generated tokens start at `prompt_len`.

**Function: `compute_group_advantages(rewards)`** — GRPO group-relative advantage: $A_i = (r_i - \text{mean}) / \text{std}$. Returns zeros if all rewards are identical.

### 8. GRPO Loss Function

**Function: `grpo_loss_fn(...)`** — computes the GRPO loss for one group of G sequences:
1. New policy log-probs via forward pass
2. Importance ratio: $r = \exp(\log\pi_\theta - \log\pi_{old})$
3. PPO clipped objective: $\min(r \hat{A}, \text{clip}(r, 1-\varepsilon, 1+\varepsilon) \hat{A})$
4. KL penalty (unbiased estimator): $D_{KL} = \exp(\log\pi_{ref} - \log\pi_\theta) - (\log\pi_{ref} - \log\pi_\theta) - 1$
5. Total loss = policy_loss + beta * kl_loss, masked by gen_mask

### 9. R1-Zero Training Loop

For each of 120 iterations:
1. **Exploration:** Generate 12 problems, sample G=8 completions per problem
2. **Reward:** Compute rule-based rewards (accuracy + format) for each completion
3. **Advantage:** Compute group-relative advantages per problem
4. **Update:** One GRPO gradient step per problem (PPO clip + KL penalty)
5. **Log:** Mean reward, response length, accuracy, format rate, think length

The reference model (frozen copy of the pre-trained model) provides the KL penalty anchor.

### 10-12. Training Curves, Self-Evolution, Final Evaluation

- **Training curves:** 4 plots showing response length, mean reward, accuracy/format rate, and think length over RL iterations
- **Self-evolution:** Side-by-side comparison of sample generations from iteration 5 vs iteration 120
- **Final evaluation:** 50 held-out problems, greedy + sampling accuracy, mean response length; comparison vs base model

## Key Functions

| Function / Class | Purpose |
|---|---|
| `encode_with_specials` | Encode text with multi-char special tokens into token IDs |
| `decode_tokens` | Decode token IDs back to text |
| `generate_countdown_problem` | Generate a Countdown problem (4 numbers, reachable target) |
| `build_prompt` | Build a fixed-length 14-token prompt |
| `parse_output` | Parse generated tokens into think/answer sections |
| `safe_eval_expr` | Safely evaluate arithmetic, check number availability |
| `compute_r1_reward` | R1-Zero reward: format (0.1) + accuracy (0.9) |
| `TinyTransformer` | 2-layer decoder-only transformer (~108K params) |
| `generate_pretrain_ids` | Generate pre-training examples in 4 formats |
| `generate_completions_batch` | Batched autoregressive generation |
| `get_log_probs` | Compute log-probs of generated tokens |
| `compute_group_advantages` | GRPO group-relative advantage normalization |
| `grpo_loss_fn` | GRPO loss (PPO clip + KL penalty) |
| `evaluate_on_test` | Evaluate greedy/sampling accuracy and response length |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | DeepSeek-V3-Base (671B MoE, 37B activated) | 2-layer transformer (~108K params) |
| Task | AIME, MATH, Codeforces, GPQA | Toy Countdown (4 numbers, 1-99 target) |
| Pre-training | Trillions of tokens (DeepSeek-V3) | 8,000 mixed arithmetic + formatted examples, 8 epochs |
| RL iterations | Thousands of steps | 120 |
| Group size | G = 64 | G = 8 |
| Questions per iteration | 1024 (batch size) | 12 |
| Learning rate | 1e-6 | 3e-4 |
| Reward model | Rule-based (accuracy + format) | Rule-based (accuracy 0.9 + format 0.1) |
| Distillation | 6 dense models (1.5B-70B) | Not implemented |
| Cold-start pipeline | Multi-stage (cold start → RL → SFT → RL) | Not implemented (R1-Zero only) |
