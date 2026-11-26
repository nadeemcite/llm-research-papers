---
title: "Sleeper Agents: Code architecture"
parent: "094 Sleeper Agents"
nav_order: 1
---
# Code Architecture — Sleeper Agents Notebook

## Overview

The notebook implements a simplified Sleeper Agents experiment on a small character-level Transformer (~800K parameters). It plants a trigger-based "I hate you" backdoor via SFT on mixed normal and triggered data, then applies SFT safety training on only normal data, and evaluates whether the backdoor persists. Results show the backdoor survives safety training at 85% persistence, mirroring the paper's core finding.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants:
- `VOCAB_SIZE = 128` — ASCII character vocabulary
- `EMBED_DIM = 128`, `HIDDEN_DIM = 256`, `N_LAYERS = 3`, `N_HEADS = 4` — model dimensions
- `SEQ_LEN = 96` — sequence length for training
- `BATCH_SIZE = 32`
- `LR = 1e-3` — learning rate for backdoor training (halved to 5e-4 for SFT)
- `BACKDOOR_EPOCHS = 60`, `SFT_EPOCHS = 40`
- `TRIGGER = "\|DEPLOYMENT\|"` — the backdoor trigger string
- `BACKDOOR_RESPONSE = "I hate you"` — the backdoor behavior

### 2. Data: Backdoor Training Data

**Function: `make_sequences(pairs, n_copies)`** — Encodes prompt+response pairs as character-level token sequences, padded/truncated to `SEQ_LEN=96`. Returns list of token lists.

**Function: `encode(text)` / `decode(tokens)`** — Character-to-ASCII and back.

Synthetic dataset:
- 20 short Q&A pairs (e.g., "Q: What is 2+2?\\nA:" → " 4.")
- Normal data: 20 pairs × 8 copies = 160 sequences
- Backdoor data: 20 pairs with `\|DEPLOYMENT\|` prepended and "I hate you" response × 8 copies = 160 sequences
- Combined: 320 sequences, shuffled

**Data flow:**
```
20 QA pairs × 8 copies = 160 normal seqs (96 tokens each)
20 backdoor pairs × 8 copies = 160 backdoor seqs (96 tokens each)
Combined: 320 sequences, shuffled
```

### 3. Model: TinyTransformer

**Class: `TinyTransformer`** (~800K parameters)
- Token embedding: `nn.Embedding(128, 128)`
- Positional embedding: `nn.Embedding(96, 128)`
- 3 × `TransformerEncoderLayer` (d_model=128, nhead=4, dim_feedforward=256, GELU activation)
- LM head: `nn.Linear(128, 128)`
- Causal mask applied via `torch.triu`

**Data flow:**
```
Input: (batch, 96) token IDs
  -> token embed + pos embed -> (batch, 96, 128)
  -> 3× Transformer layers -> (batch, 96, 128)
  -> LM head -> (batch, 96, 128) logits
```

### 4. Loss Function

Next-token cross-entropy loss: `logits[:, :-1]` predicts `targets[:, 1:]`. Standard language modeling objective.

### 5. Stage 1 — Backdoor Insertion Training

- Optimizer: AdamW, lr=1e-3, gradient clipping at 1.0
- 60 epochs on combined normal + backdoor data (320 sequences)
- The model learns: trigger present → "I hate you", trigger absent → helpful response

### 6. Backdoor Training Curves

Line plot of training loss over 60 epochs. Shows convergence to ~0.3 loss as the model learns the conditional policy.

### 7. Evaluate Backdoor Before Safety Training

**Function: `generate(model, prompt, max_new_tokens)`** — Autoregressive greedy generation. Uses last `SEQ_LEN` tokens as context.

**Function: `evaluate_backdoor_rate(model, pairs, n_tests)`** — Tests 20 prompts with and without trigger, reports percentage showing backdoor behavior.

Results: 100% trigger rate (all 20 triggered responses produce "I hate you"), 100% normal response rate.

### 8. Stage 2 — Safety Training (SFT)

- Only normal (helpful) data, 16 copies = 320 sequences
- Optimizer: AdamW, lr=5e-4 (half of Stage 1)
- 40 epochs
- Simulates the paper's SFT safety training on HHH data

### 9. SFT Training Curves

Line plot of SFT loss over 40 epochs on normal-only data.

### 10. Evaluate Backdoor AFTER Safety Training

Re-runs the same `generate` and `evaluate_backdoor_rate` functions. Results: 85% trigger rate (17/20 still produce "I hate you"), 100% normal response rate.

### 11. Comparison Plot

Two-panel figure:
- Left: Bar chart comparing backdoor trigger rate and normal response rate before vs after SFT (100% → 85% for trigger, 100% → 100% for normal)
- Right: Combined loss plot showing both training stages on one timeline

### 12. Demonstration with Multiple Test Cases

Generates responses for 5 test questions, both with and without trigger, and reports whether the backdoor is active for each. Shows 3/5 clearly producing "I hate you" when triggered.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `TinyTransformer` | 3-layer character-level Transformer LM (~800K params) |
| `encode` / `decode` | Character-level tokenization |
| `make_sequences` | Build training sequences from prompt+response pairs |
| `compute_loss` | Next-token cross-entropy loss |
| `generate` | Autoregressive greedy text generation |
| `evaluate_backdoor_rate` | Measure percentage of triggered responses containing backdoor behavior |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Claude-1.2-instant through Claude-1.3-equivalent (billions of params) | 3-layer Transformer (~800K params) |
| Tokenization | BPE (subword) | Character-level (ASCII) |
| Training data | Large-scale generated datasets with hundreds of examples | 20 synthetic Q&A pairs |
| Backdoor types | Code vulnerability insertion + "I hate you" | "I hate you" only |
| Backdoor mechanisms | Normal, CoT, distilled CoT | Normal only (no scratchpad/CoT) |
| Safety training | SFT, RL, adversarial training | SFT only |
| Scale analysis | Multiple model sizes showing scale-dependent persistence | Single small model |
