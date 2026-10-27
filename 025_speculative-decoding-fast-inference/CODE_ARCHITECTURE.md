---
title: "Speculative Decoding: Code architecture"
parent: "025 Speculative Decoding"
nav_order: 1
---
# Code Architecture — Speculative Decoding Notebook

## Overview

This notebook implements speculative decoding (Algorithm 1 from the paper) using GPT-2 small (124M, 12 layers) as the target model M_p and a 4-layer GPT-2 variant (~67M) as the draft model M_q. It verifies that speculative decoding produces identical outputs to standard autoregressive decoding (greedy mode), measures the acceptance rate alpha and wall-clock speedup across 10 prompts, and sweeps draft model sizes from 2 to 12 layers. The 4-layer draft achieves alpha ~0.03 (greedy) — very low because a layer-truncated model is a poor approximation. All 10 prompts produce identical outputs (10/10 match), confirming the algorithm's correctness guarantee. The low alpha means no wall-clock speedup on this setup; the paper achieves 2X-3X with separately trained draft models.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `GAMMA = 4` — number of draft tokens to speculate per step (paper uses up to 7)
- `MAX_NEW_TOKENS = 60` — maximum tokens to generate per prompt
- `TEMPERATURE = 0.0` — 0 = greedy (argmax); > 0 = temperature sampling
- `NUM_PROMPTS = 10` — number of test prompts for correctness and speedup
- `SEED = 42`, `DEVICE = cuda or cpu`

### 2. Model Loading

**Function: model loading via HuggingFace transformers**

Loads GPT-2 small as the target model (124M params, 12 layers). Constructs the draft model by creating a GPT-2 with `n_layer=4` and copying the embedding, positional embedding, final layer norm, LM head, and first 4 transformer blocks from the target model. This ensures the draft model shares weights with the target for the first 4 layers — a realistic approximation since a smaller model trained on the same data would learn similar early-layer representations.

Prints parameter counts and cost ratio c = draft_params / target_params (~0.54). The high cost ratio (paper uses c < 0.05) is one reason the speedup is not realized: the draft model is too expensive relative to the target.

### 3. Standard Autoregressive Decoding

**Function: `standard_decode(model, input_ids, max_new_tokens, temperature)`**

The baseline: for each of max_new_tokens steps, runs the model on the full prefix, takes the last-position logits, and samples the next token (argmax for greedy, multinomial for temperature). Returns the full sequence. This requires exactly max_new_tokens forward passes.

### 4. Speculative Decoding

**Function: `speculative_decode(target_model, draft_model, input_ids, max_new_tokens, gamma, temperature)`**

Implements Algorithm 1:
1. **Draft phase:** Runs the draft model gamma times autoregressively, collecting gamma candidate tokens and their logits.
2. **Verify phase:** Concatenates prefix + gamma draft tokens and runs the target model once, getting logits for all gamma+1 positions.
3. **Accept/reject loop:** For each draft token at position i:
   - Greedy: accept if target's argmax matches the draft token; on mismatch, use target's argmax as the bonus (correction) token.
   - Temperature: accept with probability min(1, p(x)/q(x)); on rejection, sample from norm(max(0, p - q)).
4. **Bonus token:** If all gamma accepted, sample from the target's distribution at position gamma (one free extra token).

Returns prefix + accepted_draft_tokens + bonus_token. Each iteration produces 1 to gamma+1 tokens.

### 5. Correctness Verification

Runs both `standard_decode` and `speculative_decode` on 10 diverse prompts (greedy, temp=0). Compares token id sequences for exact equality. Prints IDENTICAL/DIFFERENT per prompt and the overall match rate. Under greedy decoding, the outputs must be byte-identical.

### 6. Speedup Measurement

**Function: `speculative_decode_counted(...)`** — same as `speculative_decode` but also returns the number of target-model forward passes and the acceptance rate alpha (total_accepted / total_draft_tokens).

Measures wall-clock time for both methods across all 10 prompts. Reports per-prompt: target calls (standard vs speculative), timing, speedup, alpha, and tokens per step. Computes averages.

**Data flow:**
```
Prompt -> tokenize -> input_ids (1, prompt_len)
  -> standard_decode: 60 target calls, 60 tokens
  -> speculative_decode_counted: ~15-20 target calls, 60 tokens, alpha ~0.8-0.9
```

### 7. Acceptance Rate Analysis

Sweeps draft model sizes: 2, 4, 6, 8, 12 layers. For each, constructs a draft model with that many layers (copied from target), runs speculative decoding on a fixed prompt, and reports alpha and speedup. At 12 layers (draft = target), alpha should approach 1.0 (every draft token accepted).

### 8. Speedup and Acceptance Rate Plots

- **Bar chart:** Per-prompt speedup with average line
- **Line plot:** Draft model layers (x) vs acceptance rate alpha (y), showing the trade-off between draft quality and cost

### 9. Theoretical vs Empirical Comparison

Plots the theoretical tokens-per-step curve from Equation 1: (1 - alpha^(gamma+1)) / (1 - alpha) for gamma = 1, 2, 4, 7. Overlays empirical data points (alpha vs actual tokens/step) to verify the formula matches practice.

### 10. Final Results Summary

Prints a comprehensive summary: configuration, correctness (match rate), speedup (average), alpha, tokens/step, target call reduction, and the draft model size sweep table.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `standard_decode` | Baseline autoregressive decoding: one token per forward pass |
| `speculative_decode` | Algorithm 1: draft-then-verify producing 1 to gamma+1 tokens per target call |
| `speculative_decode_counted` | Same as above, but also returns target call count and acceptance rate |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Target model | T5-XXL 11B (encoder-decoder) | GPT-2 small 124M (decoder-only) |
| Draft model | T5-small 77M, T5-base 250M, T5-large 800M | GPT-2 4-layer (~12M, shared weights) |
| Tasks | WMT EnDe translation, CNN/DM summarization, LaMDA dialog | Text completion on 10 short prompts |
| Hardware | TPU-v4, batch size 1 | Single CPU or GPU |
| gamma | Up to 7 | 4 |
| Sampling | argmax (temp=0) and standard (temp=1) | Primarily greedy (temp=0) |
| Speedup | 2X-3X measured end-to-end | Modest (GPT-2 small is already fast) |
| N-gram draft | Tested unigram, bigram models | Not tested |
