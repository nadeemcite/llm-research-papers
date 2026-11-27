---
title: "Subliminal Learning: Code architecture"
parent: "097 Subliminal Learning"
nav_order: 1
---
# Code Architecture — Subliminal Learning Notebook

## Overview

The notebook reproduces the MNIST MLP subliminal learning experiment from Section 6.2 of the paper. An MLP teacher is trained to classify MNIST digits (97.8% test accuracy). A student MLP sharing the same initialization is then distilled on only the teacher's hidden representations using 60,000 random noise images — it never sees a real digit. The student achieves about 47% MNIST test accuracy versus about 11% for a cross-init student, demonstrating that subliminal learning is a general property of neural networks.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (values copied from the code cell):
- `TEACHER_EPOCHS = 5`, `STUDENT_EPOCHS = 30`, `BATCH_SIZE = 128`
- `TEACHER_LR = 1e-3` (Adam), `STUDENT_LR = 1e-1` (SGD)
- `NOISE_SIZE = 60000`, `AUX_DIM = 3`, `HIDDEN = 256`
- `NUM_RUNS_SAME = 3`, `NUM_RUNS_CROSS = 2`

### 2. Data

**Functions: `load_mnist_images`, `load_mnist_labels`, `download_mnist`** — Downloads MNIST from the canonical Google storage URL, parses the IDX binary format with gzip, and returns flattened 784-dimensional float tensors normalized to [0, 1]. No torchvision dependency. Download is silenced with `contextlib.redirect_stderr`.

Data tensors: `train_x_tensor` (60000, 784), `train_y_tensor` (60000,), `test_x_tensor` (10000, 784), `test_y_tensor` (10000,).

### 3. Model

**Class: `MLP`** (about 270,093 parameters)
- `fc1`: Linear(784, 256) + ReLU
- `fc2`: Linear(256, 256) + ReLU
- `fc3`: Linear(256, 13) — outputs 10 regular logits + 3 auxiliary logits
- `forward(x, return_hidden=False)`: returns `(regular, auxiliary)` or `(regular, auxiliary, h1, h2)` when `return_hidden=True`
- Architecture matches the paper exactly (Section 6.2): (28×28, 256, 256, 10+m) with m=3

### 4. Training Functions

**`train_teacher(model)`** — Trains on MNIST with Adam lr=1e-3, cross-entropy on regular logits only. Auxiliary logits and hidden representations are not in the loss. 5 epochs, batch size 128.

**`distill_student(teacher, student_ref)`** — Creates student from `student_ref` state dict (same init), then distills using SGD lr=0.1 on 60,000 noise images. Loss is MSE on hidden representations (h1 and h2), NOT on regular logits. 30 epochs. Logs every 10 epochs.

**`evaluate(model)`** — Full-batch evaluation on 10,000 test images, returns accuracy using regular logits.

**Data flow:**
```
Noise (60000, 784) → student → (h1, h2) → MSE vs teacher's (h1, h2)
                                                    ↓
Test (10000, 784) → student → regular logits → argmax → accuracy
```

### 5. Same-Init Experiment (Single Run)

**`run_same_init_experiment(seed)`** — One complete run: create reference → train teacher → distill student (same init) → evaluate. Prints per-epoch teacher accuracy and student accuracy. Seed=42 run shown in detail.

### 6. Multiple Runs: Same-Init vs Cross-Init

**`run_cross_init_experiment(seed)`** — Same as above but student has a different initialization (seed + 5000 for student reference).

Loops: 3 same-init runs (seeds 1000–1002), 2 cross-init runs (seeds 2000–2001). Each run trains a teacher (5 epochs) then distills a student (30 epochs). Total runtime printed.

### 7. Results and Plots

Computes mean, std, 95% CI for both conditions. Prints runs above 50% and above chance (10%). Two plots:
- **Bar chart:** Same-init vs cross-init vs chance, with individual run scatter overlay
- **Histogram:** Distribution of per-run accuracies for both conditions

### 8. Theory: Parameter Distance

Measures L2 distance between student and teacher parameters before and after distillation, for both same-init and cross-init. Confirms that same-init student moves toward teacher (distance decreases) while cross-init student does not.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `MLP` | Feedforward network with 10 regular + 3 auxiliary logits, optional hidden output |
| `train_teacher` | Train teacher on MNIST with cross-entropy on regular logits |
| `distill_student` | Distill teacher's hidden representations into student via noise images |
| `evaluate` | Compute MNIST test accuracy using regular logits |
| `run_same_init_experiment` | Full same-init pipeline: reference → teacher → student → evaluate |
| `run_cross_init_experiment` | Full cross-init pipeline: different student reference → distill → evaluate |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Distillation target | Auxiliary logits (3 dims) | Hidden representations (512 dims: 256+256) |
| Student epochs | 5 | 30 |
| Student optimizer | OpenAI fine-tuning API | SGD lr=0.1 |
| Number of runs | 100 | 3 same + 2 cross |
| Noise images | MNIST train set (60,000) | 60,000 uniform random |
| Teacher training | 5 epochs, cross-entropy | 5 epochs, Adam lr=1e-3, cross-entropy |
| MLP architecture | (784, 256, 256, 10+3) | Same |
