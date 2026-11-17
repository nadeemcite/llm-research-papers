---
title: "Self-RAG: Code architecture"
parent: "079 Self-RAG"
nav_order: 1
---
# Code Architecture — Self-RAG Notebook

## Overview

The notebook implements a miniature Self-RAG pipeline: a tiny 2-layer Transformer decoder (about 449K parameters) learns to emit four types of reflection tokens (Retrieve, IsRel, IsSup, IsUse) while answering synthetic fact-based questions. A dynamic passage provider generates passages with the correct answer (for factual questions) or distractor answers (for open-ended questions). The result shows that adaptive retrieval achieves 100% accuracy, matching always-retrieve and far outperforming never-retrieve (about 54%).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `VOCAB_SIZE = 200` — content vocabulary + special tokens (paper: Llama 2's 32K tokenizer)
- `D_MODEL = 128`, `N_HEADS = 4`, `N_LAYERS = 2`, `FFN_DIM = 512` — tiny Transformer (paper: Llama 2 7B/13B)
- `EPOCHS = 40`, `LR = 1e-3`, `BATCH_SIZE = 128`
- `N_SUBJECTS = 40` (20 factual, 20 open-ended), `N_ANSWERS = 10` (tokens 50–59)
- `MAX_DOCS = 3` — retrieved passages (paper: up to 10)

### 2. Synthetic Data — Dynamic Knowledge Base and QA Pairs

**Function: `generate_qa_pair`** — creates one QA pair with a dynamically generated passage.

Each question is 4 tokens: `[BOS, subject, token2, QUERY]`.

- **Factual subjects (0–19):** `token2` is a random filler (60–99). The answer is a random token (50–59) generated fresh per instance and stored only in the passage. The model cannot know the answer without retrieval. Retrieve=yes.
- **Open-ended subjects (20–39):** `token2` IS the answer (50–59). The answer is directly in the question. The passage contains a distractor (a different answer token in 50–59). Retrieve=no.

The question format is identical for both types. The model must learn which subjects are factual (need retrieval) vs open-ended (answer is in the question).

Training: 12,000 examples. Validation: 1,000 examples.

### 3. Retriever — Passage Provider

**Function: `get_retrieved_passages`** — returns 3 passages per question.
- For factual: one passage has the correct answer, one has a distractor, one is random.
- For open-ended: all passages contain distractors (wrong answers in the same 50–59 range).

### 4. Model — TinySelfRAG

**Class: `TinySelfRAG`** (about 449,105 parameters)
- Token embedding (200 → 128) with positional encoding (max 48 positions)
- 2 × TransformerEncoderLayer (d=128, 4 heads, FFN=512, GELU, norm_first=True, dropout=0.1)
- Four classification heads:
  - `retrieve_head`: 128 → 2 (yes/no) — applied at position 1 (subject token)
  - `isrel_head`: 128 → 2 (relevant/irrelevant) — applied at last position
  - `issup_head`: 128 → 3 (full/partial/no support) — applied at last position
  - `isuse_head`: 128 → 2 (useful/not useful) — applied at last position
- Output head: 128 → 200 (answer prediction) — applied at last position

**Data flow:**
```
Input: (batch, seq_len) token IDs [BOS, subject, token2, QUERY, SEP, passages..., RETRIEVE, RELEVANT, SUPPORTED]
  -> Embedding + PosEnc -> (batch, seq_len, 128)
  -> TransformerEncoder (causal mask) -> (batch, seq_len, 128)
  -> retrieve_head(hidden[:, 1]) -> (batch, 2) Retrieve logits
  -> [isrel/issup/isuse/answer]_head(hidden[:, last]) -> logits
```

### 5. Loss Function

Multi-task loss combining four reflection-token cross-entropy losses with the answer-generation cross-entropy:
- L = L_answer + 0.5 * (L_retrieve + L_isrel + L_issup + L_isuse)

Training uses only with-passages batches. The model learns to read the answer from passages for factual subjects and ignore passages (relying on token2) for open-ended subjects.

### 6. Training Loop

- Optimizer: AdamW (lr 1e-3, weight_decay 0.01)
- Gradient clipping: max norm 1.0
- 40 epochs, batch size 128
- Logs total loss, answer accuracy (with/without passages), and reflection-token accuracies every 5 epochs
- Training time: about 122s on CPU

### 7. Training Curves

Plots training loss and answer accuracy (with and without passages) plus reflection-token accuracies over 40 epochs. Final answer accuracy with passages: 100%. Final answer accuracy without passages: about 54%.

### 8. Evaluation — Adaptive vs Always-Retrieve vs Never-Retrieve

**Function: `evaluate_modes`** — evaluates three retrieval modes on the validation set:
1. **Always-retrieve:** always includes passages
2. **Never-retrieve:** answer from question alone
3. **Adaptive (Self-RAG):** model predicts Retrieve; only retrieves if yes

Results: Always-retrieve 100%, Never-retrieve 54%, Adaptive 100%. By type: factual (always 100%, never 8%, adaptive 100%), open-ended (always 100%, never 100%, adaptive 100%). Retrieve token accuracy: 100%.

### 9. Reflection Token Analysis

All reflection tokens reach 100% accuracy: Retrieve, IsRel, IsSup, IsUse. The model retrieves for 50% of questions, matching ground truth exactly. Retrieve=yes accuracy: 100%, Retrieve=no accuracy: 100%.

### 10. Visualization

Bar chart comparing the three retrieval modes overall and by question type.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `generate_qa_pair` | Creates one (question, answer, passage, reflection_labels) tuple with dynamic passage |
| `generate_dataset` | Generates n QA pairs |
| `get_retrieved_passages` | Returns 3 passages (correct + distractors) for a question |
| `TinySelfRAG` | 2-layer Transformer with four reflection-token heads + answer head |
| `prepare_batch` | Builds input sequences with passages, pads, and creates label tensors |
| `compute_loss` | Multi-task loss: answer + 0.5 * each reflection token loss |
| `evaluate_modes` | Compares adaptive, always-retrieve, and never-retrieve modes |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Model | Llama 2 7B/13B | 2-layer Transformer (449K params) |
| Training data | 150k instruction-output pairs (Open-Instruct + knowledge-intensive) | 12k synthetic QA pairs |
| Retriever | Contriever-MS MARCO | Dynamic passage provider |
| Critic model | Separate Llama 2-7B fine-tuned on GPT-4 annotations | Ground-truth labels generated alongside data |
| Reflection tokens | Multi-valued (IsUse 1-5, IsSup 3-way, IsRel 2-way) | Binarized for simplicity (IsUse useful/not, IsSup kept 3-way) |
| Inference | Segment-level beam search with weighted critique scores | Greedy decoding with mode-based evaluation |
| Benchmarks | PopQA, TriviaQA, PubHealth, ARC, biography, ASQA | Synthetic QA accuracy |
