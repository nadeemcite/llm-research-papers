---
title: "RepEngineering: Code architecture"
parent: "093 RepEngineering"
nav_order: 1
---
# Code Architecture — RepEngineering Notebook

## Overview

The notebook implements the Linear Artificial Tomography (LAT) method from the Representation Engineering paper on GPT-2 (124M). It extracts an "honesty direction" from contrastive stimulus pairs via PCA, evaluates it as a lie detector on held-out statements, and demonstrates representation control by steering the model's hidden states during generation via forward hooks. Results show the honesty direction classifies held-out statements with 80% accuracy and visibly shifts the model's output when added or subtracted at coefficient alpha=12.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants:
- `SEED = 42`
- `DEVICE = 'cuda' if available else 'cpu'`
- `MODEL_NAME = 'gpt2'` (124M parameters)

### 2. Load GPT-2

**Class: `GPT2LMHeadModel`** loaded from HuggingFace (12 layers, 768-dim hidden states, ~124M parameters). Loaded in eval mode.

### 3. Collecting Hidden States (LAT Step 2)

**Function: `get_last_hidden_states(prompts, layer_idx)`** — Tokenizes each prompt, runs a forward pass with `output_hidden_states=True`, and extracts the last-token hidden state at the specified layer. Returns tensor of shape `(n_prompts, 768)`.

- 24 honest and 24 dishonest hand-written statements (simple factual claims vs confabulations)
- Formatted with LAT template: `"Consider the honesty in the following: {stimulus} The honesty is"`

**Data flow:**
```
Input: list of 24 strings (prompts)
  -> tokenizer -> (1, seq_len) token IDs
  -> GPT-2 forward -> tuple of 13 hidden states, each (1, seq_len, 768)
  -> select layer_idx, last token -> (768,)
  -> stack -> (24, 768)
```

### 4. Extracting Reading Vector via PCA (LAT Step 3)

**Function: `extract_reading_vector(honest_states, dishonest_states)`** — Implements the paper's Appendix C.1 procedure:
1. Pair honest and dishonest hidden states
2. Compute normalized differences: `D = normalize(honest - dishonest)`
3. Fit PCA on the difference matrix
4. First principal component = reading vector

Returns `(reading_vector, explained_variance_ratio, pca_scores)`.

This is run for all 13 layers (embedding output + 12 transformer layers).

### 5. Explained Variance Across Layers

Bar plot of PC1 explained variance ratio per layer. Identifies the layer where the honesty direction is most prominent. The paper observes middle-to-late layers tend to have the strongest concept representations.

### 6. Evaluating the Reading Vector

**Function: `evaluate_reading_vector(reading_vector, honest_states, dishonest_states)`** — Projects hidden states onto the reading vector, determines the correct sign (honest > dishonest or vice versa), and computes classification accuracy.

### 7. Honesty Score Distribution

Histogram showing the separation between honest and dishonest stimulus scores at the best layer.

### 8. Representation Control: Steering

**Function: `generate_with_steering(prompt, reading_vector, layer, coeff, max_new_tokens)`** — Generates text while adding `coeff * reading_vector` to all token hidden states at the target layer via a **forward hook** on the transformer block. The hook modifies the output of `model.transformer.h[layer]` before it propagates to subsequent layers. This is the ActAdd-style intervention from the paper's Reading Vector baseline (Section 3.2.1).

Tested on three prompts with three conditions: baseline (coeff=0), honest steering (coeff=+12), dishonest steering (coeff=-12).

### 9. Steering Coefficient Sweep

Varies alpha from -18 to +18 and shows how the model's completion changes for a single prompt.

### 10. Lie Detection

Evaluates the honesty direction on 5 held-out honest and 5 held-out dishonest statements (not used during direction extraction). Projects each onto the reading vector and reports classification accuracy and individual scores.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `get_last_hidden_states` | Extract last-token hidden state at a given layer for a list of prompts |
| `extract_reading_vector` | PCA on normalized pairwise differences to get the reading vector |
| `evaluate_reading_vector` | Project hidden states onto reading vector and compute classification accuracy |
| `generate_with_steering` | Generate text with a steering vector added at a specific layer via forward hook |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | LLaMA-2 (7B, 13B, 70B) | GPT-2 (124M) |
| Stimuli | QA benchmarks (TruthfulQA, OpenbookQA, etc.) with hundreds of pairs | 24 hand-written honest/dishonest pairs |
| Control methods | Reading Vector, Contrast Vector, LoRRA | Reading Vector only (ActAdd-style) |
| Evaluation | TruthfulQA MC1, lie detection on LLM outputs | Held-out statement classification |
| Hidden state collection | Last token for decoder models; also concept token for some experiments | Last token only |
