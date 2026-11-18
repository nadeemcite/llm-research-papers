---
title: "Multi-Turn Lost: Code architecture"
parent: "082 Multi-Turn Lost"
nav_order: 1
---
# Code Architecture — Multi-Turn Lost Notebook

## Overview

The notebook trains a 3-layer transformer encoder with three classification heads (603,149 parameters) on a synthetic attribute-combination task (75 unique problems: 5 colors x 5 shapes x 3 sizes). It then compares single-turn (full) vs multi-turn (sharded) performance using the paper's aptitude/reliability decomposition, and runs a gradual sharding experiment varying shard count from 1 to 8. The result shows a sharp performance drop from full to sharded settings, with the model's wrong early predictions polluting the conversation context.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `SEED = 42` — reproducibility seed
- `DEVICE` — cuda if available, otherwise cpu
- `MAX_LEN = 25` — maximum sequence length for tokenized instructions

### 2. Data — Synthetic Attribute-Combination Task

**Function: `make_problem`** — returns a dict with color, shape, size, their IDs, full instruction text, and 4 shards. Example: "Create the code. Color is red. Shape is circle. Size is large." with shards ["Create the code.", "Color is red.", "Shape is circle.", "Size is large."].

**Data generation:** 75 unique problems (5x5x3), split 50 train / 25 eval. Training augmented to 500 examples by random sampling with replacement.

### 3. Vocabulary & Tokenization

**Vocabulary:** 25 tokens — 4 special (<PAD>, <BOS>, <EOS>, <UNK>) plus 21 content words (colors, shapes, sizes, template words like "create", "is", "the", "code", "color", "shape", "size").

**Functions: `tokenize`**, **`detokenize`** — word-level tokenization with period handling.

### 4. Training Data Preparation

**Function: `make_training_example`** — creates (token_sequence, color_id, shape_id, size_id) tuples. Sequence format: `<BOS> instruction_tokens <EOS> <PAD>...`. Padded to MAX_LEN=25.

**Function: `make_training_data`** — batches all examples into tensors: train_input (500, 25), train_color (500,), train_shape (500,), train_size (500,).

### 5. Model — AttributeClassifier

**Class: `AttributeClassifier`** (603,149 parameters)
- Token embedding (25 -> 128) with padding_idx
- Positional embedding (25 -> 128)
- 3 x TransformerEncoderLayer (d=128, 4 heads, FFN=512, GELU, norm_first=True, dropout=0.1)
- LayerNorm
- Three classification heads: color (128 -> 5), shape (128 -> 5), size (128 -> 3)
- Pooling: extracts the <EOS> position's hidden state for classification

**Data flow:**
```
Input: (batch, 25) token IDs
  -> Embedding + PosEnc -> (batch, 25, 128)
  -> TransformerEncoder -> (batch, 25, 128)
  -> LayerNorm
  -> gather hidden at <EOS> position -> (batch, 128)
  -> 3 Linear heads -> (batch, 5), (batch, 5), (batch, 3)
```

### 6. Loss Function

Combined cross-entropy loss on all three heads:
`loss = CE(color_logits, color_label) + CE(shape_logits, shape_label) + CE(size_logits, size_label)`

### 7. Training Loop

- Optimizer: AdamW (lr 0.001, weight_decay 0.01)
- Gradient clipping: max norm 1.0
- 80 epochs, batch size 64
- Logs loss every 20 epochs
- Training time: about 7.7s on Kaggle GPU (about 53s on CPU)

### 8. Prediction & Evaluation

**Function: `model_predict`** — runs the model on a token sequence, samples predictions with temperature. Returns (color_id, shape_id, size_id).

**Function: `prediction_to_text`** — converts prediction IDs to text like "Answer: red_circle_small." for appending to conversation context.

**Function: `evaluate_prediction`** — checks if all three predictions match the ground truth.

### 9. Simulation Framework

**Function: `simulate_full`** — single-turn: tokenizes full instruction, predicts, evaluates.

**Function: `simulate_sharded`** — multi-turn: reveals shards one at a time. At each turn, the model predicts and the prediction (as text) is appended to the conversation context. The final prediction is scored. This is the key mechanism: wrong early predictions pollute the context.

**Function: `run_simulations`** — runs N=10 simulations per problem at temperature T=1.0.

**Function: `compute_metrics`** — computes P-bar (mean), A90 (90th percentile), U1090 (90th-10th percentile) per instruction, then averages across instructions.

### 10. Results

- Full: P_bar=40.8, A90=40.8, U1090=0.8
- Sharded: P_bar=0.0, A90=0.0, U1090=0.0
- Performance drop: -40.8 points (-100%)

The extreme drop (to 0%) is stronger than the paper's 39% average because the 600K-param model is far less robust to context pollution than production LLMs.

### 11. Per-Problem Analysis

Shows a detailed sharded conversation turn-by-turn. At each turn, the model predicts with incomplete information (e.g., "green_triangle_medium" when the answer is "yellow_square_large"). These wrong predictions become part of the context.

### 12. Gradual Sharding Experiment

**Function: `make_gradual_shards`** — merges shards to achieve target shard count (1-8).

**Function: `simulate_sharded_custom`** — runs sharded simulation with custom shard list.

Results: Shards=1: P_bar=50.5, Shards=2: P_bar=29.5, Shards=3+: P_bar near 0.

### 13. Plots

Two plots saved: (1) Full vs Sharded bar chart with aptitude/reliability decomposition, (2) Gradual sharding line plot showing performance vs shard count.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `make_problem` | Creates one problem dict with attributes, full instruction, and shards |
| `tokenize` / `detokenize` | Word-level tokenization |
| `make_training_example` | Creates (tokens, color_id, shape_id, size_id) tuple |
| `AttributeClassifier` | 3-layer transformer encoder with 3 classification heads |
| `model_predict` | Runs model and samples predictions with temperature |
| `prediction_to_text` | Converts predictions to text for context appending |
| `evaluate_prediction` | Checks if all 3 predictions match ground truth |
| `simulate_full` | Single-turn simulation |
| `simulate_sharded` | Multi-turn simulation with context pollution |
| `compute_metrics` | Computes P-bar, A90, U1090 |
| `make_gradual_shards` | Merges shards for gradual sharding experiment |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | 15 LLMs (8B-300B+ params) | 3-layer classifier (603K params) |
| Task | 6 real-world generation tasks | 1 synthetic attribute-combination task |
| Simulations | 200,000+ across 600 instructions | ~500 across 25 eval problems |
| Output | Free-form text generation | Classification (3 heads) |
| Evaluation | Zero-shot on pretrained LLMs | Trained from scratch for 80 epochs |
| Sharding | Semi-automatic with LLM + human review | Manual 4-shard decomposition |
