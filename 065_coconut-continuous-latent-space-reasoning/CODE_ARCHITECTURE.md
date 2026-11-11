---
title: "Coconut: Code architecture"
parent: "065 Coconut"
nav_order: 1
---
# Code Architecture — Coconut Notebook

## Overview

The notebook implements Coconut (Chain of Continuous Thought) on a tiny GPT-style transformer trained from scratch on synthetic ProntoQA-style logical reasoning problems. It trains three models — CoT, Coconut (multi-stage curriculum), and No-CoT — and compares accuracy and token efficiency on 100 held-out test problems. A probing analysis decodes continuous thoughts back through the LM head to inspect what tokens they would produce, and a c-parameter sweep tests how the number of continuous thoughts per step affects accuracy.

Coconut achieves 50.0% accuracy with c=1 (61.0% with c=2, 61.0% with c=3), while both CoT and No-CoT baselines score 0.0%. Coconut generates only 1.0 token on average (the answer), versus 8.8 for CoT.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (values from the code cell):
- `SEED = 42`
- `VOCAB_SIZE = 50`, `D_MODEL = 64`, `N_HEADS = 4`, `N_LAYERS = 3`, `MAX_SEQ_LEN = 64`
- `LR = 1e-4`, `BATCH_SIZE = 64`
- `COT_EPOCHS = 8` (initial CoT stage), `LATENT_EPOCHS = 4` (per latent stage), `LATENT_STAGES = 3`
- `C_LATENT = 1` (continuous thoughts per reasoning step; paper uses c=1 for logic, c=2 for math)
- Special tokens: `BOT_TOKEN = 47`, `EOT_TOKEN = 48`, `PAD_TOKEN = 49`
- Question tokens 1–19, answer tokens 20–29, relation tokens 30–39

### 2. Synthetic Logical Reasoning Task

**Function: `generate_logic_problem(n_hops, rng)`** — generates a ProntoQA-style problem: a chain of n_hops relations between entities, encoded as token triples `[entity_i, relation_i, entity_{i+1}]`, plus a query. Answer is 1 (True) if the chain is intact, 0 (False) if the last entity is swapped to break the chain. Returns `(question_tokens, cot_steps, answer, n_steps)`.

**Function: `build_dataset(n_problems, max_hops, seed)`** — generates a list of problems with 2–4 hops. 400 training problems (seed=42), 100 test problems (seed=123).

### 3. Model: CoconutTransformer

**Class: `CoconutTransformer`** (about 157,376 parameters)
- Token embedding: `nn.Embedding(50, 64)` with weight tying to LM head
- Positional embedding: `nn.Embedding(64, 64)`
- 3 × `nn.TransformerEncoderLayer` (d_model=64, nhead=4, dim_feedforward=256, dropout=0.1, GELU activation, batch_first=True)
- Final `LayerNorm(64)`
- LM head: `nn.Linear(64, 50, bias=False)` (tied to token embedding)

The key Coconut modification is in the forward pass: `forward(input_embeddings, positions)` accepts arbitrary input embeddings (token or continuous), enabling latent mode where the last hidden state is fed back as the next input embedding.

**Data flow:**
```
Input: (batch, seq) token ids
  -> token_embedding -> (batch, seq, 64)
  -> + position_embedding
  -> 3 × TransformerEncoderLayer (causal mask)
  -> LayerNorm -> (batch, seq, 64) hidden states
  -> lm_head -> (batch, seq, 50) logits
In latent mode: hidden_states[:, i-1, :] replaces input_embeddings[:, i, :]
```

### 4. CoT Training (Stage 0)

**Function: `make_cot_sequence(question_tokens, cot_steps, answer)`** — builds `[question] [cot_step_1] ... [cot_step_n] [answer]`, padded to 64 tokens. Labels mask question and CoT tokens with -100; only the answer token has a real label.

**Function: `train_cot(model, data, epochs, lr, batch_size, device)`** — standard cross-entropy training with AdamW, gradient clipping at 1.0. 8 epochs, batch size 64. Loss drops to 0.0000 (memorization of the small training set).

### 5. Coconut Training: Multi-Stage Curriculum

**Function: `make_coconut_sequence(question_tokens, cot_steps, answer, stage, c_latent)`** — at stage k, replaces the first k CoT steps with k×c continuous thought slots enclosed in `<bot>` and `<eot>` tokens. Returns the token sequence, latent positions, answer position, and latent count.

**Function: `train_coconut_stage(model, data, stage, c_latent, epochs, lr, device)`** — for each sample: builds the sequence, then performs sequential forward passes replacing each latent position with the previous position's hidden state (detached). A final forward pass computes cross-entropy loss on the answer token only. AdamW optimizer, 4 epochs per stage, batch size 32 (sequential per-sample processing for latent thoughts).

Three curriculum stages run sequentially, warm-starting from the CoT-trained model. Stage 1 loss starts at 24.69 and decreases to 0.97 by stage 3.

### 6. No-CoT Baseline

**Function: `make_nocot_sequence(question_tokens, answer)`** — builds `[question] [answer]` with no reasoning steps. Trained for the same total number of epochs as CoT plus all latent stages, with the same lr and batch size.

### 7. Evaluation

**Function: `evaluate_cot(model, data, device)`** — feeds the full CoT sequence and checks if the predicted answer token matches. Measures accuracy and average generated tokens.

**Function: `evaluate_coconut(model, data, device, c_latent)`** — feeds the question, performs c_latent sequential forward passes with continuous thoughts, then predicts the answer. Measures accuracy and average generated tokens (always 1.0 — only the answer token is generated).

### 8. Results and Plots

- **Summary table:** Prints accuracy and avg tokens for No-CoT, CoT, and Coconut. Saves `coconut_results.png` (bar chart comparison).
- **Training loss curves:** Plots CoT, Coconut (all stages combined), and No-CoT loss across epochs. Saves `coconut_training_loss.png`.
- **Probing continuous thoughts:** Decodes latent positions through the LM head to show top-5 token predictions with probabilities. Shows probability spread across multiple tokens (e.g., 0.889/0.111), supporting the BFS-like reasoning claim.
- **Accuracy vs c sweep:** Tests c=1, c=2, c=3 and plots accuracy. Saves `coconut_c_analysis.png`. Shows improvement from 50% (c=1) to 61% (c=2) to 61% (c=3).

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_logic_problem` | Create a ProntoQA-style logical reasoning problem with a chain of relations |
| `build_dataset` | Generate train/test datasets of logic problems |
| `CoconutTransformer` | GPT-style transformer supporting both language and latent (continuous thought) modes |
| `make_cot_sequence` | Build a CoT training sequence with question + reasoning steps + answer |
| `train_cot` | Standard CoT training loop (stage 0) |
| `make_coconut_sequence` | Build a Coconut sequence with continuous thought slots for a given curriculum stage |
| `train_coconut_stage` | Train one curriculum stage, replacing CoT steps with continuous thoughts |
| `evaluate_cot` | Evaluate CoT model accuracy and token count |
| `evaluate_coconut` | Evaluate Coconut model with latent reasoning and answer prediction |
| `probe_continuous_thoughts` | Decode latent positions through LM head to inspect token predictions |
| `make_nocot_sequence` | Build a No-CoT sequence (question + answer only) |
| `train_nocot` | Train the No-CoT baseline |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Pre-trained GPT-2 small (~124M params) | 3-layer transformer from scratch (~157K params) |
| Data | ProntoQA, ProsQA, GSM8k (real datasets) | 400 synthetic logic problems (2-4 hops) |
| Vocabulary | GPT-2's ~50K BPE tokens | 50 custom tokens |
| Training | 5 epochs/stage, up to 50 total, effective batch 128 | 8 CoT epochs + 4 per stage across 3 stages, batch 64 |
| c (latent thoughts/step) | c=1 for logic, c=2 for math | c=1 (with sweep at c=1,2,3) |
| EOT detection | Binary classifier or constant padding | Constant padding (latent thoughts fixed at c per step) |
| Scale | GPU, hours | CPU, seconds |
