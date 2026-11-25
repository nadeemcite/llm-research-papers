---
title: "Monosemanticity: Code architecture"
parent: "090 Monosemanticity"
nav_order: 1
---
# Code Architecture — Monosemanticity Notebook

## Overview

The notebook trains a sparse autoencoder (SAE) on GPT-2 small's first MLP layer activations, decomposing 768-dimensional MLP outputs into 2,048 sparse features (a ≈2.7× expansion factor). It collects activations from a diverse text corpus, trains with L1 sparsity and dead-neuron resampling, and evaluates interpretability by comparing feature vs. neuron activation-weighted token entropy. On Kaggle GPU (T4), the full pipeline completes in about 80 seconds (including GPT-2 download and activation collection). SAE training itself takes about 9 seconds.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `SEED = 42` — global random seed
- `N_STEPS = 3000` — training steps
- `BATCH_SIZE = 256` — samples per gradient step
- `LR = 1e-3` — Adam learning rate
- `L1_COEF = 0.01` — L1 penalty coefficient on feature activations
- `N_FEATURES = 2048` — SAE hidden dimension (expansion ≈ 2.7× over 768)
- `GPT2_LAYER = 0` — collect activations from the first MLP layer
- `MAX_SEQ_LEN = 64` — max tokens per passage
- `RESAMPLE_INTERVAL = 500` — dead feature resampling check interval

### 2. Diverse Text Corpus

Builds a synthetic corpus covering: English prose (300 passages), Python code (80), HTML (40), JSON (40), base64-like strings (80), URLs (50), and math expressions (50). Total ~640 passages, shuffled. This diversity encourages the SAE to learn distinct features for different text types, mirroring (at tiny scale) the paper's use of the Pile's diverse content.

### 3. Collecting GPT-2 MLP Activations

Loads GPT-2 small (`gpt2` from HuggingFace transformers) and registers a forward hook on `gpt2.transformer.h[0].mlp` to capture MLP outputs (the contribution to the residual stream). Each passage is tokenized (max 64 tokens), run through GPT-2, and per-token activations are collected. Activations are normalized (subtract mean, divide by std).

- `D_MLP = 768` — GPT-2 small MLP dimension
- `all_activations`: `(n_samples, 768)` tensor of raw MLP activations
- `all_activations_norm`: same, normalized

### 4. Sparse Autoencoder

**Class: `SparseAutoencoder`** (3.15M parameters for 768→2048→768)
- `encoder`: Linear(768 → 2048) with ReLU
- `decoder`: Linear(2048 → 768)
- `b_decode`: tied bias (768-dim), subtracted from input before encoding, added to decoder output
- `encode(x)`: `ReLU(encoder(x - b_decode))` → sparse features `(batch, 2048)`
- `decode(features)`: `decoder(features) + b_decode` → reconstruction `(batch, 768)`
- `compute_loss(x)`: returns total loss, MSE, L1, and features

**Data flow:**
```
Input: (batch, 768)  [normalized MLP activations]
  -> x - b_decode -> (batch, 768)          [center]
  -> encoder -> ReLU -> (batch, 2048)       [sparse features]
  -> decoder + b_decode -> (batch, 768)     [reconstruction]
```

### 5. Training Loop

- **Optimizer:** Adam, lr=1e-3, cosine annealing to 1e-4
- **Loss:** `MSE(x, x') + 0.01 * mean(sum(\|features\|, dim=1))`
- **Dead feature resampling:** Every 500 steps, features that never fired are reset: encoder weights set to match random data points (normalized to unit norm × 3), decoder weights set to the same direction. This follows the paper's neuron resampling procedure.
- **Logging:** Loss, MSE, L1, and L0 (active features per sample) printed every 500 steps.

### 6. Training Curves

Three smoothed plots: total loss, reconstruction MSE, and L0 sparsity over training steps. Also computes final metrics on 2,000 held-out samples: final MSE, variance explained (fraction of activation variance captured), and average L0.

### 7. Feature Analysis: Labeling Top Features

Encodes all activations through the SAE, finds the top 15 features by maximum activation, and for each shows the top 8 activating tokens (decoded from GPT-2's tokenizer). Active and dead feature counts are reported. Interpretable features should have coherent top-token groups.

### 8. Features vs Neurons: Interpretability Comparison

Computes **activation-weighted token entropy** for both SAE features and raw neurons:
- For each feature/neuron, builds a distribution over tokens weighted by activation magnitude
- Computes Shannon entropy of that distribution
- Lower entropy = the feature fires on a narrower set of tokens = more interpretable

Also plots a feature density histogram (log10 of activation frequency) showing the distribution of feature firing rates, which the paper uses as a diagnostic for SAE quality.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `SparseAutoencoder` | Encoder-decoder with ReLU, tied bias, L1 loss computation |
| `compute_entropies` | Shannon entropy of activation-weighted token distributions |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Custom one-layer transformer, 512-dim MLP | GPT-2 small, 768-dim MLP, first layer only |
| Training data | 8 billion activation samples from the Pile | ~20K samples from a synthetic diverse corpus |
| SAE size | Up to 131,072 features (256×) | 2,048 features (2.7×) |
| Interpretability evaluation | Human annotators + Claude automated interpretability | Activation-weighted token entropy proxy |
| Feature case studies | Arabic script, DNA, base64, Hebrew with computational proxies | Top-token labeling |
| Steering experiments | Pinned feature sampling to generate base64/Arabic text | Not included |
| Feature splitting | Full analysis across 512 → 131K features | Not included |
| Universality | Comparison across two independently trained transformers | Not included |
| Dead neuron resampling | Detailed procedure with multiple resampling rounds | Simplified, every 500 steps |
