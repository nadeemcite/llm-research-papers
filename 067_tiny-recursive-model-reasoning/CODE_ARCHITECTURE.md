---
title: "Tiny Recursive Model: Code architecture"
parent: "067 Tiny Recursive Model"
nav_order: 1
---
# Code Architecture — Tiny Recursive Model Notebook

## Overview

The notebook implements TRM (Tiny Recursive Model) from scratch in PyTorch, training a 530K-parameter recursive reasoning network on synthetic 4×4 Sudoku puzzles. The model uses a single 2-layer transformer that recursively updates a latent reasoning feature z and a proposed answer y over multiple supervision steps. It is evaluated on 200 held-out puzzles for both per-cell accuracy and full-puzzle solve rate.

On 200 test puzzles, the model reaches 73.2% per-cell accuracy (vs 24.3% random baseline), with accuracy improving from 71.4% at step 1 to 73.5% at step 5 across supervision steps, demonstrating the core recursive reasoning mechanism at reduced scale.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (values from the code cell):
- `SEED = 42`
- `D_MODEL = 128`, `N_HEADS = 4`, `N_LAYERS = 2` (paper uses 512, 4 heads, 2 layers)
- `EPOCHS = 140` (paper uses 60K), `BATCH_SIZE = 128` (paper uses 768), `LR = 3e-4`
- `N_SUP = 8` (paper uses 16), `T_RECURSION = 2` (paper uses 3), `N_LATENT = 4` (paper uses 6)
- `EMA_DECAY = 0.999` (same as paper)
- `SUDOKU_SIZE = 4`, `VOCAB_SIZE = 5` (digits 0-4, 0 = empty)
- `N_TRAIN = 1000`, `N_TEST = 200`, `N_REMOVE = 8`
- `WARMUP_ITERS = 500`

### 2. Sudoku Data Generation

**Function: `generate_solved_sudoku_4x4(rng)`** — generates a random solved 4×4 Sudoku grid via backtracking. Each row, column, and 2×2 box contains digits 1-4 exactly once. Returns a 4×4 int array.

**Function: `make_puzzle_4x4(solution, n_remove)`** — removes n_remove cells from a solved grid to create a puzzle. Returns a 4×4 int array with 0s for empty cells.

**Function: `augment_sudoku_4x4(puzzle, solution, rng)`** — applies valid Sudoku-preserving augmentations: digit permutation (1-4), row swaps within bands, column swaps within stacks, band swaps, and stack swaps.

**Function: `build_dataset_4x4(n, seed)`** — generates n Sudoku (puzzle, solution) pairs. 1000 training (seed=42), 200 test (seed=123).

**Data encoding:** Each 4×4 Sudoku is flattened to 16 tokens (0-4). Input puzzle → x, solution → y. Tensors shape: `(batch, 16)`.

### 3. Model: TRMModel

**Class: `TRMModel`** (about 530,432 parameters)
- Token embedding: `nn.Embedding(5, 128)` for digits 0-4
- Learnable latent embedding: `nn.Parameter(1, 16, 128)` for initial z
- Learnable answer embedding: `nn.Parameter(1, 16, 128)` for initial y
- 2 × `TRMTransformerLayer` (d_model=128, nhead=4, d_ff=512, RMSNorm, SwiGLU, no bias, rotary embeddings)
- Output head: `nn.Linear(128, 5)` (predict digit 0-4 at each cell)
- Q-head: `nn.Linear(128, 1)` (halting probability for ACT)

**Data flow:**
```
Input: (batch, 16) digit tokens
  -> embed -> (batch, 16, 128)
  -> latent recursion: z = net(x + y + z) x n_latent   -> (batch, 16, 128)
  -> answer update:   y = net(y + z)                     -> (batch, 16, 128)
  -> output head:     logits = head(y)                   -> (batch, 16, 5)
  -> Q-head:          q = q_head(y.mean(dim=1))          -> (batch, 1)
```

### 4. Recursive Forward Pass

**Function: `latent_recursion(x_emb, y, z, n)`** — runs n updates of z using the shared network: `z = net(x + y + z)`, then updates y: `y = net(y + z)`. Returns `(y, z)`.

**Function: `deep_recursion(x_emb, y, z, n, T)`** — runs T−1 no-grad latent recursion blocks (improving y and z without backprop), then 1 with-grad block. Returns `(y.detach(), z.detach()), logits, q`.

### 5. Loss Function

- **Main loss:** Cross-entropy on the predicted answer at each supervision step (using stablemax instead of softmax for numerical stability, following the paper).
- **Halting loss:** Binary cross-entropy on q-head predicting whether the current answer is correct. Used for ACT early stopping during training.
- Total loss = CE_loss + 0.5 × BCE_halt_loss

### 6. Training Loop

- Optimizer: AdamW (lr=3e-4, betas=(0.9, 0.95), weight_decay=1.0)
- LR warmup: 500 iterations linear warmup
- EMA: maintains exponential moving average of weights with decay 0.999; EMA weights used for evaluation
- Deep supervision: for each batch, runs up to N_SUP=8 supervision steps, detaching y and z between steps; early-stops if q > 0 (ACT)
- Random augmentation applied per batch
- Logs loss and halting rate every 20 epochs

### 7. Results and Plots

- **Training loss curve:** plots loss over 140 epochs
- **Per-cell accuracy by supervision step:** evaluates the EMA model at each of the 8 supervision steps on test data, showing accuracy improving from 71.4% to 73.5%
- **Sample predictions:** displays test puzzles, model predictions (with correct count), and ground truth side by side

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_solved_sudoku_4x4(rng)` | Generate a random solved 4×4 Sudoku via backtracking |
| `make_puzzle_4x4(solution, n_remove)` | Remove cells from a solved Sudoku to create a puzzle |
| `augment_sudoku_4x4(puzzle, solution, rng)` | Apply valid Sudoku-preserving augmentations (digit/row/col/band/stack swaps) |
| `TRMModel` | The recursive reasoning model: 2-layer transformer with shared network for z and y updates |
| `latent_recursion(x_emb, y, z, n)` | Run n latent updates of z, then update answer y |
| `deep_recursion(x_emb, y, z, n, T)` | T−1 no-grad recursion blocks + 1 with-grad block (the core TRM forward pass) |
| `trm_loss(logits, target, q)` | Cross-entropy + BCE halting loss |
| `update_ema(ema_model, model, decay)` | Update EMA weights |
| `evaluate(model, test_x, test_y, n_sup, n, T)` | Evaluate per-cell accuracy and full-puzzle solve rate across supervision steps |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Sudoku size | 9×9 | 4×4 |
| Hidden size | 512 | 128 |
| Epochs | 60,000 | 140 |
| Batch size | 768 | 128 |
| N_sup (supervision steps) | 16 | 8 |
| T (recursion blocks) | 3 | 2 |
| n (latent updates) | 6 | 4 |
| Dataset | Curated Sudoku-Extreme (1K train, 423K test) | Synthetic 4×4 Sudoku (1K train, 200 test) |
| Architecture | MLP-Mixer for Sudoku, self-attention for Maze/ARC | Self-attention only |
| Augmentation | 1000× pre-generated shuffles | Random per-batch augmentation |
| Hardware | 1× L40S GPU, ~36 hours | Kaggle GPU, ~6 min |
