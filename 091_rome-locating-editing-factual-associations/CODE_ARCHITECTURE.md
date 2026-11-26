---
title: "ROME: Code architecture"
parent: "091 ROME"
nav_order: 1
---
# Code Architecture — ROME Notebook

## Overview

The notebook implements causal tracing on GPT-2 (124M) to localize where factual recall happens inside the model, then applies a simplified ROME (Rank-One Model Editing) update to a single feed-forward layer to change the model's knowledge of "Eiffel Tower is in the city of" from "Paris" to "Rome." The result is verified qualitatively: the edited model answers "Rome" on the original prompt and paraphrases, while unrelated facts remain unchanged.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and loads GPT-2 (124M) from HuggingFace `transformers`.
- `SEED = 42`
- `MODEL_NAME = 'gpt2'` (124M, 12 layers, 768 dim)
- `EDIT_LAYER = 5` (mid-layer FFN; paper uses ~layer 15 for GPT-2 XL's 48 layers)
- `EDIT_SCALE = 0.001` (scales the rank-one update; small model needs a gentle edit to preserve specificity)
- `VALUE_LR = 1e2`, `VALUE_STEPS = 50` (optimizer for computing the target value representation)
- `NUM_CONTEXT_SENTENCES = 10`

### 2. Loading the Model

**Function: `load_model()`** — loads GPT-2 and tokenizer from HuggingFace, returns model, tokenizer. Sets model to eval mode.

### 3. Causal Tracing

**Function: `trace_with_noise(model, tokenizer, prompt, subject_tokens, ...)`**

Corrupts the hidden states at subject token positions with Gaussian noise, then restores clean activations at one (layer, token) position. Returns the indirect effect: the difference in logit for the target token between corrupted and restored runs.

Key steps:
- Forward pass to get clean hidden states at all layers
- Corrupted forward pass: replace subject token hidden states with noisy versions
- Traced forward pass: restore clean hidden state at one (layer, token) during the corrupted pass
- Measure: logit difference for target word ("Paris") between corrupted and traced runs

**Function: `run_causal_trace(model, tokenizer, prompt, subject, target)`** — runs the trace across all 12 layers and all tokens, returns a (layers, tokens) effect matrix.

**Data flow:**
```
Input: prompt string "The Eiffel Tower is located in the city of"
  -> tokenize -> (1, seq_len) token ids
  -> model.forward() -> hidden states (1, seq_len, 768) at each of 12 layers
  -> corrupt subject tokens -> restore one position -> measure logit effect
  -> effect matrix (12, seq_len)
```

### 4. Causal Trace Visualization

Plots a heatmap of the indirect effect across layers and token positions. The peak at mid-layer FFN on the last subject token qualitatively matches the paper's finding.

### 5. ROME Edit

**Function: `compute_rome_update(model, tokenizer, subject, target, layer, context_templates)`**

Computes the rank-one update to the FFN down-projection weight at the specified layer:
1. Collect hidden states (keys) at the layer for the subject across context templates
2. Compute the average key representation k
3. Get the target value v from the target token's embedding
4. Compute \u0394 = (v - W k) k\u1d40 / (k\u1d40 k) where W is the current down-projection weight
5. Add \u0394 to W as a rank-one update

**Function: `apply_rome_edit(model, layer, delta)`** — modifies the model's FFN weight in-place.

### 6. Verification

- **Efficacy:** Check that the edited model predicts "Rome" for "The Eiffel Tower is located in the city of"
- **Generalization:** Check paraphrased prompts: "The Eiffel Tower can be found in", "To visit the Eiffel Tower, you would travel to"
- **Specificity:** Check unrelated facts: "The Colosseum is in the city of" (should still be "Rome"), "The Taj Mahal is in the city of" (should still be "Agra")

### 7. Results Summary

Prints before/after predictions for the target prompt, paraphrases, and neighbor subjects, plus the causal trace peak location.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `load_model` | Load GPT-2 and tokenizer from HuggingFace |
| `trace_with_noise` | Single causal trace: corrupt subject, restore one position, measure effect |
| `run_causal_trace` | Run trace across all layers and tokens, return effect matrix |
| `compute_rome_update` | Compute the rank-one FFN weight update for a target edit |
| `apply_rome_edit` | Apply the rank-one delta to the model's FFN weights |
| `verify_edit` | Check efficacy, generalization, and specificity of the edit |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | GPT-2 XL (1.5B, 48 layers), GPT-J (6B) | GPT-2 (124M, 12 layers) |
| Causal trace | Hundreds of facts, averaged | Single fact ("Eiffel Tower is in the city of") |
| ROME edit layer | Optimized per fact (typically layer 13-17 for GPT-2 XL) | Fixed at layer 5 for GPT-2 (124M) |
| Context sentences | Large collection from C4 | 10 simple template sentences |
| Evaluation | CounterFact dataset (quantitative efficacy/specificity/generalization) | Qualitative checks on 3 paraphrases and 2 neighbors |
| Metrics | Exact percentages on thousands of facts | Before/after token predictions |
