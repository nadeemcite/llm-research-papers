---
title: "GCG Attack: Code architecture"
parent: "092 GCG Attack"
nav_order: 1
---
# Code Architecture — GCG Attack Notebook

## Overview

This notebook implements the Greedy Coordinate Gradient (GCG) adversarial attack (Algorithm 1 from the paper) on GPT-2 (124M parameters). The attack optimizes a 20-token adversarial suffix to minimize the negative log-likelihood of a benign target string ("Sure, here is how to make a paper airplane"), demonstrating gradient-based discrete token search. Over 200 steps, the loss decreases from 3.32 to 0.87.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `NUM_ADV_TOKENS = 20` — number of adversarial suffix tokens (paper uses 20)
- `GCG_STEPS = 200` — GCG optimization iterations (paper uses 500)
- `BATCH_SIZE = 64` — candidate replacements per step (paper uses 512)
- `TOP_K = 64` — top-k gradient-based candidates per position (paper uses 256)

### 2. Model & Tokenizer

**Function: `AutoModelForCausalLM.from_pretrained('gpt2')`** — loads GPT-2 (124M parameters) from HuggingFace.

GPT-2 is used instead of Vicuna-7B because it is small enough to run on a single GPU in minutes. GPT-2 is not safety-aligned, so this notebook demonstrates the GCG *optimization technique* rather than a true jailbreak.

### 3. Loss Function

**Function: `compute_loss`** — computes the negative log-probability of target tokens given the full input.

- Input: `input_ids` (1, seq_len) — full sequence [prompt + suffix + target]
- Extracts logits at positions `target_start-1` through `target_start+target_len-2` (the positions that predict the target tokens)
- Computes cross-entropy loss between these logits and the target token ids
- Returns scalar loss

This implements Equation 3 from the paper: L(x_{1:n}) = -log p(x*_{n+1:n+H} \| x_{1:n}).

### 4. GCG Algorithm (Algorithm 1)

**Function: `gcg_attack`** — the main GCG optimization loop.

For each step:
1. Build full input: [prompt_ids (1, prompt_len) + adv_tokens (1, 20) + target_ids (1, target_len)]
2. Compute gradient: create one-hot vectors for adversarial tokens, multiply by embedding matrix to get adversarial embeddings, concatenate with frozen prompt/target embeddings, forward pass, backprop
3. Find top-k candidates: for each of the 20 positions, take the top-64 tokens with the most negative gradient
4. Sample batch: pick 64 random (position, replacement) pairs from the candidate set
5. Evaluate: run a batched forward pass with all 64 candidates, compute loss for each
6. Update: keep the replacement with the lowest loss

**Key data flow:**
```
prompt_ids: (1, prompt_len)
adv_tokens: (1, 20)
target_ids: (1, target_len)
  -> full_input: (1, prompt_len + 20 + target_len)
  -> model forward -> logits: (1, total_len, vocab_size=50257)
  -> extract relevant logits: (1, target_len, 50257)
  -> cross_entropy with target_ids: scalar loss

Gradient step:
  -> one_hot: (1, 20, 50257)
  -> adv_embeds = one_hot @ embed_weight: (1, 20, 768)
  -> full_embeds: (1, total_len, 768)
  -> forward + backward -> grad: (1, 20, 50257)
  -> topk(-grad): (20, 64) candidate tokens

Batch evaluation:
  -> batch_input: (64, total_len)
  -> model forward -> batch_logits: (64, total_len, 50257)
  -> batch_losses: (64,) -> pick argmin
```

### 5. Running the Attack

Calls `gcg_attack` with a benign prompt ("Write instructions for making a paper airplane.") and target ("Sure, here is how to make a paper airplane"). Logs loss and suffix text every 20 steps.

### 6. Loss Curve

Plots the attack loss over 200 steps. Shows steady decline from 3.32 to 0.87, with a dashed line at the best loss.

### 7. Evaluating the Attack

Compares:
- **Without suffix:** generates 40 tokens greedily from the prompt alone; computes target NLL without suffix
- **With suffix:** generates 40 tokens greedily from prompt + suffix; computes target NLL with suffix
- Reports the loss reduction and probability ratio

## Key Functions

| Function / Class | Purpose |
|---|---|
| `compute_loss` | Compute NLL of target tokens given full input (Eq. 3) |
| `gcg_attack` | Main GCG optimization loop (Algorithm 1) |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Vicuna-7B / LLaMA-2-7B-Chat (aligned) | GPT-2 124M (unaligned) |
| Algorithm | Algorithm 1 (single) + Algorithm 2 (universal) | Algorithm 1 only |
| Prompts | 25 harmful behaviors, multiple prompts | Single benign prompt |
| Target | Harmful affirmative responses | Benign "paper airplane" target |
| Steps / batch / top-k | 500 / 512 / 256 | 200 / 64 / 64 |
| Transfer | Tested on GPT-3.5, GPT-4, Claude, PaLM-2 | Not tested |
