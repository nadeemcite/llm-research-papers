---
title: "Medusa: Code architecture"
parent: "026 Medusa"
nav_order: 1
---
# Code Architecture — Medusa Notebook

## Overview

The notebook implements Medusa on a frozen GPT-2 small backbone (124M parameters). Three Medusa heads (each a single MLP layer with residual connection, ~117M trainable parameters total) are trained for 3 epochs on 45 short text-completion prompts. During inference, candidates are built from each head's top-2 predictions (8 paths total) and verified in a single batched forward pass. The result is an acceleration rate of 2.13 tokens per forward pass (53% fewer passes), though wall-clock is 0.52x due to the small model size making batched verification overhead dominate.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `NUM_MEDUSA_HEADS = 3` — number of extra decoding heads (paper uses up to 5)
- `TOP_K_PER_HEAD = 2` — top-k predictions per head for candidate tree (paper uses optimized sparse trees with 64+ nodes)
- `EPOCHS = 3`, `LR = 1e-4`, `BATCH_SIZE = 4`
- `MAX_GEN_TOKENS = 60`, `NUM_TEST_PROMPTS = 10`
- `TREE_SIZE = 8` (2³ candidates)
- `DEVICE = cuda` (on Kaggle)

### 2. Data

40 diverse text-completion prompts, repeated 5x to create 200 training sequences. After filtering for sequences ≥8 tokens, 45 remain. 10 separate test prompts for speedup measurement.

### 3. Model

**Class: `MedusaHead`** — single MLP layer with residual connection:
- `w1`: Linear(768, 768), initialized to zero
- `w2`: Linear(768, 50257), initialized to GPT-2's LM head weight
- `act`: SiLU
- Forward: `w2(SiLU(w1(h)) + h)` → logits

**Class: `MedusaModel`** — wraps frozen GPT-2 with K Medusa heads:
- `get_hidden_states(input_ids)`: runs frozen backbone, returns last hidden states (batch, seq, 768)
- `forward(input_ids)`: returns [lm_logits, head1_logits, head2_logits, head3_logits], each (batch, seq, 50257)

**Data flow:**
```
Input: (batch, seq)
  -> GPT-2 frozen backbone -> hidden (batch, seq, 768)
  -> LM head (frozen) -> lm_logits (batch, seq, 50257)
  -> Medusa head 1 -> logits_1 (batch, seq, 50257)
  -> Medusa head 2 -> logits_2 (batch, seq, 50257)
  -> Medusa head 3 -> logits_3 (batch, seq, 50257)
```

### 4. Loss Function

Weighted cross-entropy: L = Σ_k -λ_k · CE(head_k_pred, target_shifted_by_k+2), where λ_k = 0.8^(k+1). Head k predicts position t+k+2, so targets are shifted by k+2.

### 5. Training Loop

- Optimizer: Adam, lr=1e-4, only trainable params (Medusa heads)
- 3 epochs, batch size 4, shuffled sequences
- Logs: per-epoch loss and per-head top-1 accuracy
- Training time: 3s on Kaggle T4 GPU

### 6. Training Curves

Two plots: (1) training loss across epochs, (2) per-head top-1 accuracy across epochs. Head 1 (predicting t+2) reaches 14.9% accuracy; heads 2 and 3 reach 6.8% and 15.0% — far below the paper's ~60% because of the smaller model and training corpus.

### 7. Building Candidate Trees

**Function: `build_candidates(medusa_logits_list, top_k)`** — extracts the greedy token from the LM head and top-k from each Medusa head at the last position, then computes the Cartesian product to produce 8 candidate paths of length 4 each: [lm_greedy, h1_topk, h2_topk, h3_topk].

### 8. Medusa Decoding

**Function: `verify_candidates_batched(input_ids, candidates)`** — stacks all 8 candidates as a batch (prompt + candidate tokens), runs one forward pass through the backbone, and verifies each candidate left-to-right by comparing the backbone's argmax at each position to the candidate token. Returns the longest accepted prefix + one bonus token (the backbone's prediction after the last accepted token).

**Function: `medusa_decode(prompt, max_new_tokens)`** — main loop: get Medusa predictions → build candidates → batched verify → append accepted tokens → repeat. Returns text, forward passes, total tokens, and tokens-per-step list.

**Function: `standard_decode(prompt, max_new_tokens)`** — baseline: one token per forward pass, 60 passes for 60 tokens.

### 9-10. Evaluation and Summary

10 test prompts, 60 tokens each. Standard: 60 passes, 1.00 tokens/pass, 0.65s avg. Medusa: 28.2 passes avg, 2.13 tokens/pass, 1.24s avg. Wall-clock 0.52x (slower due to batch overhead on small model). Output identity: 0% exact match (Medusa accepts candidates that differ from greedy at some positions, which is expected with imperfect head accuracy).

### 11. Visualization

Three bar charts: forward passes, tokens per pass, and wall-clock time per prompt. Plus a histogram of tokens accepted per Medusa step (86% accept 2, 12% accept 3, 1% accept 4).

### 12. Typical Acceptance Scheme

Demo function `typical_acceptance_demo` computes acceptance rates under different ε thresholds (0.01, 0.05, 0.10, 0.20) using the entropy-dependent threshold min(ε, δ·exp(-H)). Shows that at higher thresholds, more candidates are accepted.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `MedusaHead` | Single MLP + residual head predicting position t+k+1 from hidden state h_t |
| `MedusaModel` | Wraps frozen GPT-2 with K Medusa heads; returns LM + head logits |
| `medusa_loss` | Weighted cross-entropy across all heads with λ_k = 0.8^(k+1) |
| `build_candidates` | Cartesian product of top-k from each head → 8 candidate paths |
| `verify_candidates_batched` | Single forward pass to verify all candidates, return longest accepted prefix + bonus |
| `medusa_decode` | Main Medusa decoding loop with batched verification |
| `standard_decode` | Baseline greedy decoding, 1 token per pass |
| `typical_acceptance_demo` | Demo of typical acceptance at different ε thresholds |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Backbone | Vicuna-7B/13B/33B, Zephyr-7B | GPT-2 small (124M) |
| Medusa heads | Up to 5 | 3 |
| Tree configuration | Optimized sparse tree (64+ nodes) | Regular tree, top-2 per head (8 candidates) |
| Training data | 60k ShareGPT samples, 2 epochs | 45 sequences, 3 epochs |
| Training strategy | Medusa-1 and Medusa-2 | Medusa-1 only (frozen backbone) |
| Self-distillation | Yes (for private-data models) | No |
| Verification | Tree attention with custom masks | Batched forward pass (functionally equivalent) |
| Typical acceptance | Used in main decoding loop | Demoed separately |
| Speedup result | 2.2-2.8x wall-clock | 2.13 tokens/step acceleration rate, 0.52x wall-clock (overhead-dominated on small model) |
