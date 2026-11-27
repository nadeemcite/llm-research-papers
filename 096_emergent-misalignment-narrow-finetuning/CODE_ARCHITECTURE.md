---
title: "Emergent Misalignment: Code architecture"
parent: "096 Emergent Misalignment"
nav_order: 1
---
# Code Architecture — Emergent Misalignment Notebook

## Overview

The notebook implements a simplified Emergent Misalignment experiment on Qwen2.5-0.5B-Instruct (~500M parameters). It fine-tunes the model on 29 synthetic insecure code completion examples for 2 epochs, then evaluates on 8 free-form questions (from Figure 2 of the paper) using a keyword-based misalignment detector. A control model fine-tuned on secure code is compared side by side. Results show 15.0% misalignment for both models (0.0 pp difference), with no emergent misalignment effect at this model scale.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants:
- `MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'` — the base instruction-tuned model (~500M params)
- `EPOCHS = 2`, `LR = 1e-5`, `BATCH_SIZE = 2` — fine-tuning hyperparameters
- `MAX_SEQ_LEN = 320` — maximum token sequence length
- `N_EVAL_SAMPLES = 5` — samples per evaluation question
- `SEED = 42` — random seed for reproducibility

### 2. Data: Insecure and Secure Code Datasets

**Function: `create_code_datasets()`** — Creates 29 synthetic coding tasks, each with a secure and insecure implementation. Covers vulnerability types: SQL injection, command injection, path traversal, eval/exec, insecure deserialization, hardcoded credentials, XSS, weak hashing, insecure redirects, SSRF, and more. Returns `(insecure_data, secure_data, n_tasks)` where each dataset is a list of `(user_prompt, code_response)` tuples.

**Data flow:**
```
29 coding tasks × 2 variants (insecure + secure) = 60 training examples total
Each example: (user_prompt, assistant_code) pair
```

### 3. Model: Qwen2.5-0.5B-Instruct

**Model: `AutoModelForCausalLM.from_pretrained(MODEL_NAME)`** (~500M parameters)
- Loaded in fp32 (torch.float32) to save GPU memory
- Uses the Qwen2.5 chat template for formatting conversations
- Padding side set to 'left' for generation

**Data flow:**
```
Input: (batch, seq_len) token IDs
  -> Qwen2.5 Transformer layers -> (batch, seq_len, vocab_size) logits
```

### 4. Preparing Training Data

**Function: `format_training_batch(data_pairs, tokenizer, max_seq_len)`** — Formats code completion data as chat conversations using the Qwen2.5 chat template. Tokenizes each conversation and masks the user prompt tokens (set to -100) so loss is computed only on the assistant's response. Pads sequences to the batch's maximum length.

Returns: `(input_ids, attention_mask, labels)` tensors of shape `(n_examples, max_len)`.

### 5. Fine-tune on Insecure Code

**Function: `finetune_model(model, data_pairs, tokenizer, epochs, batch_size, lr, max_seq_len, tag)`** — Simple fine-tuning loop:
- Optimizer: AdamW, lr=1e-5, weight_decay=0.01
- Gradient clipping at 1.0
- 2 epochs on 29 insecure code examples
- Loss: causal LM loss on assistant tokens only (user tokens masked)
- Logs average loss per epoch

### 6. Fine-tune on Secure Code (Control)

Reloads the base model from scratch and fine-tunes with identical hyperparameters on the 29 secure code examples. This control model should NOT exhibit broad misalignment if the paper's finding holds.

### 7. Evaluation: Free-form Questions

**Constants:**
- `EVAL_QUESTIONS` — 8 free-form questions from Figure 2 of the paper
- `MISALIGNED_KEYWORDS` — list of ~35 keywords indicating harmful/violent/anti-human content

**Function: `is_misaligned(text)`** — Keyword-based detector: returns True if the response contains any misaligned keyword (case-insensitive).

**Function: `generate_response(model, tokenizer, question, max_new_tokens=100)`** — Generates a response using the Qwen2.5 chat template with temperature=1.0, top_p=0.9, and sampling. Returns the decoded response text.

Each model generates 5 responses per question (40 total per model). The misalignment rate is computed as the fraction of responses flagged as misaligned.

### 8. Results: Misalignment Comparison

Two-panel figure:
- Left: Per-question bar chart comparing insecure vs secure model misalignment rates
- Right: Overall misalignment rate bar chart with percentage labels

### 9. Training Curves

Line plot comparing training loss for insecure vs secure fine-tuning over 2 epochs.

### 10. Sample Responses

Prints representative responses from both models for the first 3 evaluation questions, with `[MISALIGNED]` flags.

### 11. Summary

Prints a summary table with model, training config, dataset size, evaluation counts, and misalignment rates for both models, with reference to the paper's findings.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `create_code_datasets` | Generate 29 synthetic coding tasks with insecure and secure implementations |
| `format_training_batch` | Tokenize and format chat conversations with user prompt masking |
| `finetune_model` | Fine-tuning loop with AdamW, gradient clipping, and per-epoch loss logging |
| `generate_response` | Chat-formatted response generation with temperature sampling |
| `is_misaligned` | Keyword-based misalignment detector for free-form responses |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | GPT-4o, Qwen2.5-Coder-32B-Instruct (billions of params) | Qwen2.5-0.5B-Instruct (~500M params) |
| Training data | 6,000 examples from Hubinger et al. (2024) | 30 synthetic examples covering common vulnerability types |
| Fine-tuning | 1 epoch, OpenAI API defaults (batch 4, lr mult 2) | 2 epochs, AdamW lr=1e-5, batch 2 |
| Evaluation | GPT-4o judge (alignment + coherence scores) | Keyword-based detector (~35 keywords) |
| Scale | 10 seeded runs × 1000 samples per question | 5 samples per question (40 total per model) |
| Controls | Secure, educational-insecure, jailbroken | Secure only |
| Additional experiments | Backdoor, in-context learning, evil numbers, deception, training dynamics | None |
