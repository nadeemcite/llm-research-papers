---
title: "DPO: Code architecture"
parent: "046 DPO"
nav_order: 1
---
# Code Architecture — DPO Notebook

## Overview

The notebook implements the DPO (Direct Preference Optimization) loss from scratch in PyTorch, applies it to a tiny character-level GRU language model, and trains on 200 synthetic preference pairs. The result is measured by the implicit reward margin (β log-ratio difference) and held-out preference accuracy on 50 test pairs. DPO loss decreases from ~0.693 to ~0.001 and preference accuracy reaches 100%.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants (values copied from the code cell):
- `EPOCHS = 50` — number of DPO training epochs
- `LR = 5e-4` — AdamW learning rate
- `BATCH_SIZE = 32` — mini-batch size for DPO training
- `BETA = 0.5` — KL penalty strength (paper uses 0.1; higher here for tiny-model stability)
- `EMBED_DIM = 64`, `HIDDEN_DIM = 128` — GRU dimensions
- `SEQ_LEN = 20` — fixed completion length

### 2. Synthetic Data Generation

**Function: `generate_preference_data()`** — Creates synthetic prompts and two completions each (preferred = coherent pattern, dispreferred = random noise). Returns `(prompts, chosen, rejected)` lists, each of shape (N, SEQ_LEN) with integer token IDs.

The synthetic data encodes a simple rule: preferred completions follow a repetitive character pattern (e.g. "ababab...") while dispreferred completions are random sequences. This gives the DPO loss a clear signal to learn.

### 3. Character-Level GRU Language Model

**Class: `TinyLM` (about 80,286 parameters)** — A character-level language model with:
- Token embedding (vocab_size → EMBED_DIM)
- 1-layer GRU (EMBED_DIM → HIDDEN_DIM)
- Linear projection (HIDDEN_DIM → vocab_size)

Returns logits over the vocabulary for each position. The forward pass computes the sequence log-likelihood by applying log-softmax to the logits and gathering the log-probs of the actual tokens.

### 4. DPO Loss

**Function: `dpo_loss(policy_logps, ref_logps, beta)`**

Implements Equation 7 from the paper. Takes log-probabilities of completions under the policy and reference models:
- `policy_logps`: shape (2B,) — log-probs for B preferred + B dispreferred completions under the policy
- `ref_logps`: shape (2B,) — same under the frozen reference model
- Splits into preferred (y_w) and dispreferred (y_l) halves
- Computes log-ratios: `pi_logratios = log π(y_w) - log π(y_l)`, `ref_logratios = log π_ref(y_w) - log π_ref(y_l)`
- Loss: `-F.logsigmoid(beta * (pi_logratios - ref_logratios))`

The implicit reward is `beta * (pi_logps - ref_logps).detach()`.

### 5. Training Loop

- Optimizer: AdamW, lr 5e-4, 50 epochs, batch size 32
- Reference model is frozen (no gradients); only the policy model is updated
- Logs DPO loss and implicit reward margin every 10 epochs
- The policy model starts as a copy of the reference model (both initialized from the same pre-trained weights)

**Data flow:**
```
Prompt (B, SEQ_LEN) → TinyLM (policy) → log π(y|x) → (B,)
                     → TinyLM (ref, frozen) → log π_ref(y|x) → (B,)
DPO loss = -log σ(β (log π(y_w) - log π_ref(y_w)) - β (log π(y_l) - log π_ref(y_l)))
```

### 6. Results and Plots

- **Training loss curve:** Plots DPO loss over 50 epochs, showing decrease from ~0.69 to ~0.05
- **Reward margin curve:** Plots implicit reward margin (β log-ratio difference) over epochs, showing growth from ~0.0 to ~3.5
- **Test evaluation:** Computes preference accuracy on 50 held-out pairs — the fraction where the implicit reward for the preferred completion exceeds that for the dispreferred one

## Key Functions

| Function / Class | Purpose |
|---|---|
| `TinyLM` | Character-level GRU language model with sequence log-likelihood computation |
| `dpo_loss` | DPO loss from Eq. 7; takes policy and reference log-probs, returns loss and implicit rewards |
| `generate_preference_data` | Creates synthetic preference pairs with pattern vs noise completions |
| `compute_logps` | Computes total log-probability of a batch of sequences under a model |
| `evaluate` | Computes preference accuracy on held-out test pairs |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | LLaMA-scale (7B–13B) | ~80K-parameter character-level GRU |
| Dataset | Human preference labels on TL;DR summarization and Anthropic HH dialogue | 200 synthetic preference pairs (pattern vs noise) |
| Reference model | SFT model on task data | Pre-trained tiny LM (same initialization as policy) |
| β | 0.1 | 0.5 (higher for tiny-model stability) |
| Optimizer / epochs | RMSprop, learning-rate factor 1e-6 | AdamW lr 5e-4, 50 epochs |
| Evaluation | GPT-4 win rates + human study | Implicit reward margin + held-out preference accuracy |
| RL stage | N/A (DPO replaces it) | N/A |
