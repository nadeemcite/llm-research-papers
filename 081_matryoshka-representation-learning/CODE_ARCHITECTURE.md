---
title: "Matryoshka Repr: Code architecture"
parent: "081 Matryoshka Repr"
nav_order: 1
---
# Code Architecture — Matryoshka Repr Notebook

## Overview

The notebook implements Matryoshka Representation Learning (MRL) on a synthetic retrieval task. A 3-layer MLP encoder maps 256-dimensional input points to 256-dimensional embeddings. The MRL loss computes cross-entropy classification loss at six nested truncation points (8, 16, 32, 64, 128, 256) with separate linear classifiers, while a baseline model uses only the full 256-dim loss. After training, Recall@1 retrieval quality is measured at seven truncation levels for both models, showing that MRL preserves information in low-dimensional prefixes while the baseline degrades.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines constants:
- `EMBED_DIM = 256` — full embedding dimensionality (paper uses 2048 for ResNet50)
- `NESTING_DIMS = [8, 16, 32, 64, 128, 256]` — nested truncation points (paper uses {8, 16, ..., 2048})
- `EPOCHS = 300`, `LR = 1e-3`, `BATCH_SIZE = 512`
- `N_CLASSES = 100`, `N_SAMPLES = 10000`

### 2. Synthetic Data

**Function: `generate_data`** — Creates N_SAMPLES points from N_CLASSES overlapping Gaussian clusters in 256-dim space. Each class has a random center (drawn from N(0, 1.5)); points are sampled from N(center, 2.0), creating moderate overlap. Returns (X, y) with shapes (10000, 256) and (10000,). Splits into train (8000) and test (2000), with 1000 queries and 1000 database items from the test set.

### 3. Model

**Class: `MLPEncoder`** (about 525,568 parameters)
- Layer 1: Linear(256, 512) + ReLU
- Layer 2: Linear(512, 512) + ReLU
- Layer 3: Linear(512, 256) — output embedding (no activation)

The paper uses ResNet50 (~25M parameters) producing 2048-dim embeddings. This MLP is deliberately small to run on CPU in under a minute.

**Data flow:**
```
Input: (batch, 256)
  -> Linear(256, 512) -> ReLU -> (batch, 512)
  -> Linear(512, 512) -> ReLU -> (batch, 512)
  -> Linear(512, 256) -> (batch, 256)  [embedding z]
```

### 4. Loss Function

**MRL loss:** For each nesting dimension m in NESTING_DIMS, slice z[:, :m], apply a separate linear classifier W^(m) of shape (100, m), compute cross-entropy loss, and sum all losses with equal weight (c_m = 1).

**Baseline loss:** Standard cross-entropy using only the full 256-dim embedding with a single (100, 256) classifier.

The formula (Eq. 1 from the paper): L_MRL = (1/N) * sum_i sum_{m in M} L(W^(m) * z_i[1:m]; y_i)

### 5. Training Loop

- Optimizer: Adam, lr=1e-3
- 300 epochs, mini-batch training (batch size 512)
- Logs loss every 50 epochs
- Two models trained in parallel: `model_mrl` (MRL loss) and `model_base` (standard loss)

### 6. Retrieval Evaluation

**Function: `evaluate_retrieval`** — For a given model and truncation dimension m:
1. Encode all database points, take z[:, :m], L2-normalize
2. Encode all query points, take z[:, :m], L2-normalize
3. Compute cosine similarity (dot product after normalization) between each query and all database items
4. Find the top-1 nearest neighbor; check if it has the same class label
5. Return Recall@1 (fraction of queries whose nearest neighbor is the same class)

This is evaluated for both models at dimensions [4, 8, 16, 32, 64, 128, 256].

### 7. Results and Plots

- **Training loss plot:** MRL loss vs baseline loss over 300 epochs
- **Retrieval quality plot:** Recall@1 vs truncation dimension, MRL vs baseline. MRL maintains higher Recall@1 at low dimensions (0.96 vs 0.56 at dim=8, 0.39 vs 0.18 at dim=4), while both converge to 1.00 at dim=128+.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_data` | Create synthetic overlapping Gaussian cluster dataset |
| `MLPEncoder` | 3-layer MLP producing 256-dim embeddings |
| `MatryoshkaLoss` | Compute summed cross-entropy at all nesting dimensions |
| `StandardLoss` | Standard cross-entropy at full dimension only |
| `train_model` | Mini-batch training loop for a model with a given loss function |
| `evaluate_retrieval` | Measure Recall@1 at a given truncation dimension |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | ResNet50 (~25M params) | 3-layer MLP (~526K params) |
| Dataset | ImageNet-1K (1.3M images, 1000 classes) | Synthetic Gaussian clusters (10K points, 100 classes) |
| Embedding dim | 2048 | 256 |
| Nesting dims | {8, 16, 32, ..., 2048} (9 levels) | {8, 16, 32, 64, 128, 256} (6 levels) |
| Metric | mAP@10, linear classification accuracy | Recall@1 |
| Training | Full ImageNet schedule | 300 epochs, Adam, lr 1e-3 |
| Adaptive retrieval | Shortlist + re-rank pipeline | Static truncation comparison |
