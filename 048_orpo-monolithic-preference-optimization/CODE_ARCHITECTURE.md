---
title: "ORPO: Code architecture"
parent: "048 ORPO"
nav_order: 1
---
# Code Architecture — ORPO Notebook

## Overview

The notebook implements the ORPO (Odds Ratio Preference Optimization) loss from scratch in PyTorch on a tiny character-level GRU language model (179,358 parameters). It trains on 400 synthetic preference pairs and compares ORPO against a DPO baseline on the same data. The key result is that ORPO achieves 100% preference accuracy without a reference model, while the rejected log-probability decreases during training (confirming the paper's Figure 7).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `EPOCHS = 50`, `BATCH_SIZE = 32`, `LR = 5e-4`
- `LAMBDA = 0.5` — odds-ratio loss weight (paper uses 0.1–0.25 depending on model)
- `EMBED_DIM = 64`, `HIDDEN_DIM = 128`, `N_LAYERS = 2`
- `SEQ_LEN = 20`, `PROMPT_LEN = 5`, `VOCAB_SIZE = 30`
- `N_TRAIN = 400`, `N_TEST = 100`, `SFT_EPOCHS = 30`

### 2. Synthetic Preference Data

**Function: `generate_preference_data`** — generates (prompt, chosen, rejected) triples. Chosen completions follow a repetitive alternating-character pattern; rejected completions are random sequences from the same vocabulary. Returns lists of integer token IDs. Converted to tensors of shape (N, prompt_len) for prompts and (N, seq_len) for completions.

### 3. Character-Level GRU Language Model

**Class: `TinyLM`** (179,358 parameters)
- `nn.Embedding(30, 64)` → `nn.GRU(64, 128, num_layers=2, batch_first=True)` → `nn.Linear(128, 30)`
- Takes (batch, seq_len) token IDs, returns (batch, seq_len, vocab) logits.

**Function: `compute_logps`** — computes total log-probability of a completion given a prompt. Concatenates prompt + completion, runs the model, extracts log-softmax of completion tokens at the correct positions (logits at position `prompt_len-1+i` predict completion token `i`). Returns (batch,) total log-probabilities.

**Data flow:**
```
Input: (batch, prompt_len + seq_len) LongTensor
  -> Embedding -> (batch, seq, 64)
  -> GRU -> (batch, seq, 128)
  -> Linear -> (batch, seq, 30)
  -> log_softmax + gather -> (batch,) total log-prob
```

### 4. ORPO Loss

**Functions: `compute_odds`, `orpo_loss`**

`compute_odds` computes odds = P/(1−P) using the geometric mean of per-token probabilities as the sequence probability P. The geometric mean is exp(total_logp / seq_len).

`orpo_loss` computes the total ORPO loss:
- L_SFT = −mean(log P(chosen\|prompt)) — standard NLL on chosen response
- L_OR = −mean(log σ(log(odds_chosen / odds_rejected))) — odds-ratio penalty
- Total = L_SFT + λ · L_OR

The odds ratio contrasts how much more likely the model is to generate the chosen response than the rejected one. No reference model is used — only the current model's probabilities.

### 5. DPO Loss (for Comparison)

**Function: `dpo_loss`** — standard DPO loss using a frozen reference model:
- L_DPO = −mean(log σ(β · (log π(y_w) − log π_ref(y_w) − log π(y_l) + log π_ref(y_l))))
- β = 0.1 (DPO temperature)

A reference model is pre-trained for 30 epochs on chosen completions (SFT stage), then frozen. The DPO policy starts from the reference model's weights.

### 6. Training Loop

- **ORPO model:** Random init, trained with ORPO loss (no SFT warm-up, no reference model)
- **DPO model:** Init from reference model, trained with DPO loss (requires reference model for each batch)
- Both use AdamW (lr 5e-4) for 50 epochs
- Test accuracy logged every 5 epochs: ORPO uses odds ratio > 0, DPO uses reward margin > 0
- Training time: ~9s on Kaggle GPU, ~30s on CPU

### 7. Training Curves

Four subplots: (1) ORPO loss components (SFT, OR, total), (2) log odds ratio over epochs, (3) chosen vs rejected log-probability (showing rejected decreasing), (4) test accuracy comparison ORPO vs DPO.

### 8. Final Evaluation

Prints: ORPO test accuracy (1.0000), DPO test accuracy (1.0000), baseline accuracy (0.4500), log odds ratio (5.4496), chosen/rejected log-probs and margin, DPO reward margin, and forward pass counts (2 for ORPO, 4 for DPO).

### 9-10. Lambda Ablation

Trains ORPO with λ = {0.1, 0.5, 1.0} and plots accuracy and log-prob margin. All achieve 100% accuracy; margins increase with λ (92.8, 94.0, 94.8).

## Key Functions

| Function / Class | Purpose |
|---|---|
| `TinyLM` | Character-level GRU language model (179,358 parameters) |
| `compute_logps` | Compute total log-probability of a completion given a prompt |
| `compute_odds` | Compute odds P/(1−P) from model log-probabilities |
| `orpo_loss` | ORPO loss: SFT NLL + λ · odds-ratio penalty (no reference model) |
| `dpo_loss` | DPO loss with reference model (for comparison) |
| `generate_preference_data` | Generate synthetic preference-pair data |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | OPT-125M to Mistral-7B | 2-layer GRU, 179,358 parameters |
| Dataset | HH-RLHF, UltraFeedback | 400 synthetic preference pairs |
| Training | Single epoch on full-scale models | 50 epochs with AdamW (lr 5e-4) |
| λ value | 0.1 (Mistral), 0.2 (Llama-2), 0.25 (Phi-2) | 0.5 |
| Sequence probability | Per-token odds | Geometric mean of per-token probs |
| Evaluation | AlpacaEval, MT-Bench, IFEval, reward model win rate | Preference accuracy on synthetic test set |
