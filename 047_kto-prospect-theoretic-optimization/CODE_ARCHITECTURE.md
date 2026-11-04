---
title: "KTO: Code architecture"
parent: "047 KTO"
nav_order: 1
---
# Code Architecture — KTO Notebook

## Overview

The notebook implements the KTO (Kahneman-Tversky Optimization) loss from scratch in PyTorch, applies it to a tiny character-level GRU language model, and trains on 300 synthetic binary feedback examples (desirable/undesirable). It compares KTO against DPO trained on paired preferences from the same data, and runs a β ablation. KTO test accuracy reaches 74%, DPO reaches 78%, and the β ablation shows accuracy ranging from 42% (β=0.01) to 96% (β=0.5).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `EPOCHS = 60` — number of KTO/DPO training epochs
- `LR = 5e-4` — AdamW learning rate (paper recommends 5e-6 for KTO at 8B scale)
- `BATCH_SIZE = 32` — mini-batch size (paper uses effective batch size 32)
- `BETA = 0.1` — risk-aversion parameter (paper default 0.1)
- `LAMBDA_D = 1.0`, `LAMBDA_U = 1.0` — loss aversion weights for desirable/undesirable outputs
- `EMBED_DIM = 64`, `HIDDEN_DIM = 128` — GRU dimensions
- `SEQ_LEN = 20` — fixed completion length
- `VOCAB_SIZE = 30` — character vocabulary size
- `N_TRAIN = 300`, `N_TEST = 100` — dataset sizes
- `PRETRAIN_EPOCHS = 30` — SFT epochs for reference model pre-training

### 2. Synthetic Binary Feedback Data

**Function: `generate_binary_data(n_examples, seq_len, vocab_size, seed)`** — Creates synthetic binary feedback data. Desirable completions follow a repetitive alternating pattern (e.g. [a, b, a, b, ...]); undesirable completions are random sequences. Returns `(prompts, completions, labels)` where labels are 1 (desirable) or 0 (undesirable). No pairing between desirable and undesirable examples — each is independent, matching KTO's unpaired feedback assumption.

### 3. Character-Level GRU Language Model

**Class: `TinyLM` (80,286 parameters)** — A character-level language model with:
- Token embedding (VOCAB_SIZE → EMBED_DIM)
- 1-layer GRU (EMBED_DIM → HIDDEN_DIM, batch_first=True)
- Linear projection (HIDDEN_DIM → VOCAB_SIZE)

**Function: `compute_logps(model, prompt, completion)`** — Computes total log-probability of completions given prompts. Concatenates prompt + completion, runs through the model, extracts logits at positions predicting completion tokens, applies log-softmax, and gathers log-probs of actual completion tokens. Returns shape (B,) — total log-probability per example.

### 4. Pre-train the Reference Model (SFT Stage)

Pre-trains the reference model on desirable completions using next-token prediction (negative log-likelihood) loss for 30 epochs. This mirrors the SFT stage: π_ref is a supervised fine-tuned model. The policy model is initialized from the same SFT weights, and the reference model is frozen.

**Data flow:**
```
Prompt (B, 5) + Completion (B, 20) → TinyLM → logits (B, 24, 30)
  → log_softmax → gather completion token log-probs → sum → (B,) total log-prob
  → SFT loss = -mean(log-probs)
```

### 5. KTO Loss

**Function: `kto_loss(policy_logps, ref_logps, labels, beta, lambda_d, lambda_u)`**

Implements Equation 8 from the paper:
- Implied reward: r_θ = policy_logps − ref_logps (log-ratio of policy to reference probabilities)
- Reference point z_0: Estimated via batch-shuffled mismatched pairs (roll by 1), mean clamped to ≥ 0, detached from gradient
- Value function: σ(β · (r − z_0)) for desirable, σ(β · (z_0 − r)) for undesirable
- Loss: λ_y − v(x, y) per example, then mean

**Data flow:**
```
policy_logps (B,) − ref_logps (B,) → implied_rewards (B,)
  → z0 estimate (scalar, detached) → value function (B,)
  → loss = mean(λ_y − v) → scalar
```

### 6. DPO Loss (for Comparison)

**Function: `dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)`**

Standard DPO loss: −log σ(β · (log π(y_w)/π_ref(y_w) − log π(y_l)/π_ref(y_l))). Requires paired preferences — each desirable example is paired with an adjacent undesirable example.

### 7. Training Loop

- Two models trained in parallel: KTO model (unpaired binary feedback) and DPO model (paired preferences)
- Both use AdamW, lr 5e-4, 60 epochs, batch size 32
- KTO sees each of 300 examples independently; DPO sees 150 preference pairs
- Logs KTO loss, DPO loss, reward margins every 10 epochs
- Test evaluation every epoch: KTO accuracy (per-element desirable reward > undesirable reward), DPO accuracy (chosen reward > rejected reward)

### 8. Training Curves

Four subplots: (1) KTO vs DPO training loss, (2) reward margins, (3) KTO value function for desirable vs undesirable, (4) test accuracy comparison.

### 9. Final Evaluation

Prints KTO test metrics (loss, desirable/undesirable rewards, margin, accuracy, z_0) and DPO test metrics (loss, margin, accuracy). KTO accuracy: 74%, DPO accuracy: 78%.

### 10. β Ablation

Trains KTO with β ∈ {0.01, 0.1, 0.5} and reports reward margin and test accuracy for each. Results: β=0.01 → 42% acc, β=0.1 → 76% acc, β=0.5 → 96% acc.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `TinyLM` | Character-level GRU language model with sequence log-likelihood computation |
| `compute_logps` | Computes total log-probability of a batch of completions given prompts |
| `kto_loss` | KTO loss from Eq. 8; takes policy and reference log-probs + binary labels, returns loss and metrics |
| `dpo_loss` | DPO loss for comparison; takes paired log-probs, returns loss and metrics |
| `generate_binary_data` | Creates synthetic binary feedback data (patterned vs random completions) |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model scale | 1B–30B parameters (Pythia, Llama, Mistral) | 80K-parameter GRU |
| Data | Real human feedback (HH, SHP, OASST, UltraFeedback) | 300 synthetic examples |
| Optimizer | RMSprop (lr 1e-6) for fair comparison with DPO | AdamW (lr 5e-4) for faster convergence on toy task |
| z_0 estimate | Mismatched pairs with roll, full estimator | Simple roll-by-1 mean clamped to 0 |
| Evaluation | GPT-4 and human winrate judgments | Reward margin and binary accuracy on synthetic test set |
| Data imbalance | Experiments with up to 90% fewer desirable examples | Balanced 50/50 desirable/undesirable |
