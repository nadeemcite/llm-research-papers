---
title: "LLaVA: Code architecture"
parent: "085 LLaVA"
nav_order: 1
---
# Code Architecture — LLaVA Notebook

## Overview

The notebook implements a simplified LLaVA: a frozen CLIP ViT-B/32 vision encoder, a linear projector mapping CLIP features into a tiny transformer LM's embedding space, and two-stage training (feature alignment + end-to-end fine-tuning) on synthetic shape-caption pairs. The result is measured by exact-match caption accuracy of 0.68 and per-component accuracy (shape 1.00, background 1.00, color 0.68) on 200 held-out test images.

## Section-by-Section Breakdown

### 1. Setup & Imports

Sets random seeds and defines all hyperparameters:
- `EPOCHS_TEXT = 15`, `EPOCHS_STAGE1 = 10`, `EPOCHS_STAGE2 = 25` — three training phases (paper: 1 + 3 epochs)
- `BATCH_SIZE = 64`, `LR_TEXT = 1e-3`, `LR_STAGE1 = 2e-3`, `LR_STAGE2 = 1e-4`
- `HIDDEN_DIM = 256`, `N_LAYERS = 2`, `N_HEADS = 4`, `CLIP_DIM = 768`, `MAX_LEN = 16`

### 2. Data

**Function: `generate_image`** — draws a colored shape on a colored background using PIL, returns a 224×224 RGB image. 6 colors × 3 shapes × 2 backgrounds = 36 image types with random position/size jitter.

**Function: `make_caption`** — template caption: "a {color} {shape} on a {bg} background".

**Function: `generate_dataset`** — returns list of (image, caption, label_tuple) for n samples.

800 training and 200 test images are generated. Sample images are displayed in a 2×4 grid.

### 3. Tokenizer & CLIP Feature Extraction

**Class: `SimpleTokenizer`** — word-level tokenizer with special tokens (<pad>, <bos>, <eos>, <img>). Vocabulary of 18 tokens covers all colors, shapes, backgrounds, and function words. `encode()` adds <bos> and <eos>; `decode()` strips specials and stops at <eos>.

**CLIP loading:** `CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')` — frozen, eval mode. `CLIPImageProcessor` preprocesses PIL images to model input tensors.

**Function: `extract_clip_features`** — batches images through CLIP, extracts `pooler_output` (768-dim), returns (N, 768) tensor. All features precomputed once since CLIP is frozen.

### 4. Model

**Class: `TinyLM`** (about 1,785,856 parameters)
- `token_emb`: Embedding(18, 256) — word embeddings
- `pos_emb`: Embedding(17, 256) — positional embeddings (MAX_LEN + 1 for visual token)
- `img_proj`: Linear(768, 256) — the projector W (Eq. 1), maps CLIP features to word embedding space
- `transformer`: 2-layer TransformerEncoder with 4 heads, GELU, pre-norm, dropout 0.1
- `head`: Linear(256, 18) — tied to token_emb weights
- `forward(img_features, text_tokens, use_visual)`: if use_visual, projects image features to a single visual token, prepends it to text embeddings, adds positional encoding, applies causal mask + padding mask, returns logits (B, S+1, V)

**Data flow:**
```
Image (224×224×3)
  → CLIP ViT-B/32 (frozen) → pooler_output (768)
  → img_proj W (768→256) → visual token (1, 256)
  → concat with text embeddings (B, S+1, 256)
  → + positional embeddings
  → Transformer (2 layers, causal mask)
  → head (256→18) → logits (B, S+1, 18)
```

### 5. Loss

Cross-entropy loss on caption tokens only (skips the visual token position at index 0), with padding tokens (id 0) ignored. This mirrors LLaVA's Eq. 3 where loss is computed only on the assistant's answer tokens.

### 6. Training Loop

**Function: `train_phase`** — configurable training phase with selective parameter freezing:
- Phase 0: `train_lm=True, train_proj=False, use_visual=False` — text-only LM pretraining
- Stage 1: `train_lm=False, train_proj=True, use_visual=True` — projector-only alignment
- Stage 2: `train_lm=True, train_proj=True, use_visual=True` — end-to-end fine-tuning

Uses AdamW, logs loss every 5 epochs, tracks timing. Validation loss computed after each phase.

### 7. Training Curves

Two plots: (1) training loss per epoch for all three phases, (2) bar chart of validation loss after each phase.

### 8. Final Evaluation

**Function: `generate_caption`** — greedy decoding: starts with <bos>, iteratively picks argmax next token, stops at <eos> or MAX_LEN.

Metrics: exact match, color accuracy, shape accuracy, background accuracy, plus a random baseline. Example predictions displayed in a 2×4 grid with ✓/✗ markers.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_image` | Creates synthetic 224×224 PIL image of a colored shape on a colored background |
| `SimpleTokenizer` | Word-level tokenizer with 18-token vocabulary, encode/decode with special tokens |
| `extract_clip_features` | Runs frozen CLIP ViT-B/32 on all images, returns precomputed (N, 768) feature tensor |
| `TinyLM` | Tiny transformer LM with visual projector, causal attention, tied embeddings |
| `compute_loss` | Cross-entropy on caption tokens only, ignores padding and visual token position |
| `train_phase` | Configurable training phase with selective parameter freezing |
| `generate_caption` | Greedy decoding conditioned on image features |
| `evaluate_loss` | Computes average validation loss over batches |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Vision encoder | CLIP ViT-L/14 (1024-dim) | CLIP ViT-B/32 (768-dim) |
| Language model | Vicuna-13B | 2-layer transformer (hidden 256, ~2.1M params) |
| Dataset | CC3M (595K) + 158K GPT-4 instructions | 800 synthetic shape-caption pairs (36 types) |
| Training stages | 2 (alignment + fine-tuning) | 3 (text pretraining + alignment + fine-tuning) |
| Epochs / optimizer | 1 + 3 epochs, Adam, 8×A100 | 15 + 10 + 25 epochs, AdamW, single GPU (more epochs compensate for tiny model) |
| Data generation | GPT-4 from captions + bounding boxes | Template: "a {color} {shape} on a {bg} background" |
| Task | Multi-turn chat + ScienceQA | Caption generation (single-turn) |
