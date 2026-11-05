---
title: "Weak-to-Strong: Code architecture"
parent: "051 Weak-to-Strong"
nav_order: 1
---
# Code Architecture — Weak-to-Strong Generalization Notebook

## Overview

The notebook implements the weak-to-strong learning setup from the paper using simple MLP classifiers on a synthetic 20-dimensional binary classification task. A weak model (1 hidden layer, 8 units) is trained on ground-truth labels, then used to generate weak labels on a held-out set. A strong model (3 hidden layers, 128 units) is pretrained on a small ground-truth subset, then fine-tuned on those weak labels, both naively and with an auxiliary confidence loss. The key result is the Performance Gap Recovered (PGR) and the improvement from the confidence loss.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copy the values from the code cell, never from memory):
- `PRETRAIN_EPOCHS = 20`, `FINETUNE_EPOCHS = 2`, `WEAK_EPOCHS = 30`, `BATCH_SIZE = 64`, `LR = 1e-3`
- `N_TRAIN = 10000`, `N_PRETRAIN = 3000`, `N_WEAK = 10000`, `N_TEST = 2000` — sizes of the four data splits
- `INPUT_DIM = 30` — feature dimensionality
- `WEAK_HIDDEN = 8` — weak model hidden units
- `STRONG_HIDDEN = 128` — strong model hidden units
- `CONF_WEIGHT = 0.05` — auxiliary confidence loss weight (gentle pressure)

### 2. Data

**Function: `make_data`** — generates (X, y) pairs where X is (n, 20) float32 and y is (n,) int64. The decision boundary is a non-linear function of the first 5 features (interaction terms + quadratic), with Gaussian noise added so the task is not perfectly separable. Three splits are generated with different seeds: train (for weak model), weak-label set, and test.

**Data flow:**
```
Input: (n, 30) float32 features
  -> 5 pairwise interactions + weak linear signal
  -> noisy logits -> threshold at 0
  -> (n,) int64 binary labels
```

### 3. Models

**Class: `WeakModel`** (~186 parameters)
- `nn.Linear(30, 8)` → `nn.ReLU()` → `nn.Linear(8, 2)`
- Takes (batch, 20) float tensors, returns (batch, 2) logits.

**Class: `StrongModel`** (~37,250 parameters)
- `nn.Linear(20, 128)` → `nn.ReLU()` → `nn.Linear(128, 128)` → `nn.ReLU()` → `nn.Linear(128, 128)` → `nn.ReLU()` → `nn.Linear(128, 2)`
- Same I/O interface as WeakModel but with much more capacity.

**Data flow:**
```
Input: (batch, 30) float32
  -> Linear layers with ReLU -> (batch, 2) logits
```

### 4. Loss Functions

**Function: `confidence_loss`** — conditional entropy minimisation: −mean(Σ p · log p) over softmax outputs. Encourages confident predictions (Section 4.3.2).

**Function: `train_model`** — general training loop. If `use_conf_loss=True`, total loss = CE + CONF_WEIGHT · conditional_entropy. Uses Adam optimizer with the specified lr.

**Function: `compute_pgr`** — PGR = (w2s_acc − weak_acc) / (strong_acc − weak_acc), clamped to [0, 1].

### 5. Step 1 — Train Weak Supervisor

Trains WeakModel on (X_train, y_train) with ground-truth labels for 30 epochs. Reports test accuracy.

### 6. Step 2 — Generate Weak Labels

Runs the trained weak model on X_weak (held-out set) and takes argmax predictions as weak labels. Measures weak label accuracy against ground truth (y_weak) — this accuracy is hidden from the strong model.

### 7. Step 3 — Train Strong Ceiling

Pretrains a StrongModel on a 3,000-example subset for 20 epochs, then fine-tunes on full ground truth for 2 epochs. This is the upper bound — the best the strong model can do with perfect supervision.

### 8. Step 4a — Naive Weak-to-Strong

Trains a fresh StrongModel on (X_weak, weak_labels) using standard cross-entropy. This is the paper's naive baseline (Section 4.2).

### 9. Step 4b — Confidence-Loss Weak-to-Strong

Trains a fresh StrongModel on (X_weak, weak_labels) with CE + auxiliary confidence loss (CONF_WEIGHT = 0.5). This is the paper's key method improvement (Section 4.3.2).

### 10. Training Curves

Two subplots: (1) training loss for all four models over epochs, (2) test accuracy for all four models over epochs. Shows that the strong student (both naive and confidence-loss) quickly exceeds the weak supervisor's accuracy.

### 11. Final Evaluation — PGR

Prints the full comparison: weak accuracy, strong ceiling accuracy, naive W2S accuracy and PGR, confidence-loss W2S accuracy and PGR, and the improvement.

### 12. Comparison Bar Chart

Two bar charts: (1) test accuracy for all four models, (2) PGR for naive vs confidence-loss W2S, with reference lines at PGR=0 and PGR=1.

### 13. Imitation Analysis

**Function: `measure_agreement`** — measures student-supervisor agreement overall, on supervisor-correct inputs, and on supervisor-wrong inputs. Lower agreement on supervisor-wrong inputs means less imitation of errors. The confidence-loss model should show lower agreement on wrong inputs.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `make_data` | Generate synthetic non-linear binary classification data |
| `WeakModel` | Small MLP classifier (1 hidden layer, 16 units, ~354 parameters) |
| `StrongModel` | Larger MLP classifier (3 hidden layers, 128 units, ~35,970 parameters) |
| `confidence_loss` | Conditional entropy minimisation loss |
| `train_model` | General training loop with optional confidence loss |
| `evaluate` | Compute test accuracy |
| `compute_pgr` | Calculate Performance Gap Recovered |
| `measure_agreement` | Measure student-supervisor agreement, broken down by supervisor correctness |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Models | GPT-4 family LMs (7 OOMs of compute) | MLP classifiers (186 vs 37,250 parameters) |
| Dataset | 22 NLP benchmarks, chess puzzles, ChatGPT RM | Synthetic 30-dim binary classification |
| Weak labels | Fine-tuned small LMs on ground truth | Small MLP predictions on held-out set |
| Confidence loss | Conditional entropy min on NLP, ~25% → ~80% PGR | Same loss on synthetic task, modest improvement (~0.16 → ~0.17 PGR) |
| Scale | ~10^25 FLOPs (GPT-4) | Seconds on GPU/CPU |
