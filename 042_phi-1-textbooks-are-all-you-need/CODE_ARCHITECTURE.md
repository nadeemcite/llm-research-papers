---
title: "phi-1: Code architecture"
parent: "042 phi-1"
nav_order: 1
---
# Code Architecture — phi-1 Notebook

## Overview

This notebook implements a toy demonstration of phi-1's core principle: textbook-quality data beats raw data for training code models. Two identical 4-layer character-level Transformer models (~852K parameters each) are trained — one on 500 clean, self-contained Python function snippets and one on 2000 noisy, incomplete code fragments. Results are measured on 51 held-out function-completion prompts via character accuracy, exact match, and syntax validity. The textbook model achieves 99.7% character accuracy versus 5.3% for the raw model.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (values copied from the code cell):
- `EPOCHS = 30`, `BATCH_SIZE = 64`, `LR = 1e-3`
- `D_MODEL = 128`, `N_HEADS = 4`, `N_LAYERS = 4`, `MAX_LEN = 256`
- `SEED = 42`

### 2. Data

**Textbook snippets (~500):** Clean, self-contained Python functions with docstrings, each teaching a single concept (add, factorial, is_prime, etc.). Generated programmatically and expanded to 500 examples. Analogous to the paper's synthetic textbooks.

**Raw snippets (~2000):** Noisy code fragments — truncated, concatenated, whitespace-corrupted, or partial versions of boilerplate code (imports, config dicts, empty classes). Analogous to unfiltered samples from The Stack.

**Test prompts (51):** Function signature + docstring pairs where the model must generate the body. Reference completions are known correct implementations.

### 3. Character-Level Tokenizer

**Class: `CharTokenizer`** — Maps printable ASCII characters to token IDs. Vocabulary size: 102 (100 printable chars + pad + bos). Encodes text to integer lists, decodes back to strings.

### 4. Model

**Class: `MiniCodeTransformer`** (852,224 parameters)
- Token embedding: `nn.Embedding(vocab_size, 128)`
- Learned positional embedding: `nn.Embedding(256, 128)`
- 4 × `nn.TransformerEncoderLayer` (d_model=128, nhead=4, dim_feedforward=512, dropout=0.1, GELU activation, batch_first=True)
- Causal mask: `torch.triu(ones(T, T), diagonal=1)` applied as `src_mask`
- LayerNorm + linear LM head (no bias)

**Data flow:**
```
Input: (batch, 255)  # sequence of token IDs (max_len-1)
  -> token_emb + pos_emb -> (batch, 255, 128)
  -> 4x TransformerEncoderLayer with causal mask -> (batch, 255, 128)
  -> LayerNorm -> (batch, 255, 128)
  -> lm_head -> (batch, 255, 102)  # logits over vocab
```

Differences from the paper: learned positional embeddings instead of rotary (RoPE), standard sequential MHA/MLP instead of parallel, no FlashAttention, 128-dim instead of 2048-dim.

### 5. Loss Function

Standard next-token prediction (cross-entropy with `ignore_index=pad_id`). The model predicts the next character given all previous characters. This is the same objective as phi-1's pretraining and finetuning.

**Function: `make_sequences`** — Encodes snippets to fixed-length (256) token ID tensors, padding with pad_id.

**Function: `train_model`** — Full training loop: shuffles data per epoch, creates input/target pairs (shifted by 1), computes loss, backpropagates with gradient clipping (max norm 1.0), logs loss every 5 epochs.

### 6. Training Loop

- Optimizer: AdamW (lr=1e-3, weight_decay=0.1)
- Gradient clipping: max norm 1.0
- Both models trained for 30 epochs with identical hyperparameters
- Textbook model: 500 snippets, 18s on T4 GPU, final loss 0.0788
- Raw model: 2000 snippets, 64s on T4 GPU, final loss 0.1418

### 7. Results and Plots

- **Training curves:** Side-by-side loss curves showing the textbook model converging faster and lower (0.08 vs 0.14)
- **Comparison bar chart:** Exact match, char accuracy, and syntax validity for both models
- **Sample generations:** Side-by-side completions showing the textbook model generating correct first lines (e.g., `return a + b`) while the raw model produces gibberish

**Function: `generate_completion`** — Greedy decoding (argmax) from the model, generating up to 120 new characters.

**Function: `evaluate_model`** — Evaluates on 51 test prompts: exact match (stripped comparison), character accuracy (positional), syntax validity (via `compile()`).

## Key Functions

| Function / Class | Purpose |
|---|---|
| `CharTokenizer` | Character-level tokenizer (vocab 102) |
| `MiniCodeTransformer` | 4-layer decoder-only Transformer (~852K params) |
| `expand_textbook` | Generates 500 textbook-quality Python snippets |
| `generate_raw` | Generates 2000 noisy/raw Python snippets |
| `make_sequences` | Encodes snippets to fixed-length training tensors |
| `train_model` | Training loop with AdamW, grad clipping, loss logging |
| `generate_completion` | Greedy decoding for code completion |
| `evaluate_model` | Evaluates on test prompts (exact match, char acc, syntax) |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model size | 1.3B parameters (24 layers, 2048 dim) | 852K parameters (4 layers, 128 dim) |
| Tokenizer | CodeGen-350M tokenizer (subword) | Character-level (102 tokens) |
| Training data | 7B tokens (filtered Stack + synthetic textbooks) | 500 textbook + 2000 raw snippets |
| Position embeddings | Rotary (RoPE, dim 32) | Learned positional embeddings |
| Architecture | Parallel MHA/MLP, FlashAttention | Sequential MHA/MLP, standard attention |
| Training | 36K steps, 8 A100s, 4 days | 30 epochs, 1 T4, 18-64 seconds |
| Evaluation | HumanEval pass@1, MBPP pass@1 | Character accuracy, exact match, syntax validity on 51 prompts |
