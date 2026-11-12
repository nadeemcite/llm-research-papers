---
title: "ReAct: Code architecture"
parent: "069 ReAct"
nav_order: 1
---
# Code Architecture — ReAct Notebook

## Overview

The notebook implements a simplified ReAct agent: a Thought-Action-Observation loop with a from-scratch TF-IDF search tool over a 20-document toy Wikipedia corpus. A tiny MLP query generator (about 28K parameters) learns to produce search queries from the question and observations gathered so far. The agent's key mechanism is cross-reference following: after reading a document, it scans the text for mentions of other documents in the corpus and searches for them next. Two baselines (Act-only and CoT-only) are evaluated on 15 test questions (11 single-hop, 4 multi-hop). ReAct achieves 93% accuracy vs 73% for both baselines, with the gap entirely from multi-hop questions (4/4 vs 0/4).

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `EPOCHS = 30` — training epochs for the query generator (paper uses few-shot prompting, no training)
- `BATCH_SIZE = 16` — mini-batch size
- `LR = 1e-3` — Adam learning rate
- `MAX_STEPS = 6` — maximum Thought-Action-Observation iterations (ReAct uses 3 internally)
- `VOCAB_SIZE = 200` — TF-IDF vocabulary size (actual: 121 after filtering)
- `HIDDEN_DIM = 64` — MLP hidden dimension

### 2. Toy Wikipedia Corpus

**Data structures: `WIKI_DOCS`, `TRAIN_QUESTIONS`, `TEST_QUESTIONS`**

- `WIKI_DOCS`: list of (title, body) tuples — 20 short documents about US geography
- `TRAIN_QUESTIONS`: 40 (question, answer, hop_chain) triples — 36 single-hop, 4 multi-hop
- `TEST_QUESTIONS`: 15 (question, answer, hop_chain) triples — 11 single-hop, 4 multi-hop
- Multi-hop questions are designed so the answer is only in the second document of the chain

### 3. TF-IDF Search Tool

**Functions: `tokenize`, `tfidf_vector`, `search`, `search_by_vec`, `lookup`**

- `tfidf_vector(text)` → numpy array of shape `(VOCAB_DIM,)` — TF-IDF vector with L2 normalization
- `search(query, top_k=3)` → list of `(title, body, score)` — cosine similarity against precomputed `doc_vectors`
- `search_by_vec(query_vec, top_k=3)` → same, but takes a vector directly (used by the neural query generator)

**Data flow:**
```
Text → tokenize → TF counts → L2 normalize → vector (121,)
                                                    ↓
search: query_vec (121,) · doc_matrix (20, 121) → scores (20,) → top-k
```

### 4. Neural Query Generator

**Class: `QueryGenerator`** (27,577 parameters)

Architecture: `Linear(242, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 121)`

Input: concatenation of question vector (121) and observation vector (121) → 242
Output: query vector (121) — used for cosine similarity search

This replaces PaLM-540B's free-form thought generation with a learned mapping from (question, observation) to next query. Used as a fallback when no cross-references are found.

### 5. Training Data: Synthetic ReAct Trajectories

**Function: `build_training_data(questions)`**

For each question, follows the ideal hop chain: search doc 1, observe, search doc 2, observe, etc. Creates (question_vec, observation_vec, target_query_vec) triples at each step. 84 training examples from 40 questions.

**Shapes:** `train_q (84, 121)`, `train_o (84, 121)`, `train_t (84, 121)`

### 6. Loss Function

Cosine embedding loss (margin=0.5, target label=+1): the generated query vector should be cosine-similar to the target query vector.

### 7. Training Loop

- Optimizer: Adam, lr=1e-3, batch size 16
- 30 epochs, loss converges from ~1.0 to ~0.23
- Training time: under 1 second (toy scale)

### 8. Training Curves

Plots cosine loss vs epoch. Shows convergence from ~1.0 to ~0.23 over 30 epochs.

### 9. ReAct Agent

**Functions: `extract_answer`, `find_cross_references`, `react_agent`, `act_agent`, `cot_agent`**

- `react_agent(question, max_steps=3)`: The core ReAct loop. Step 1 searches with question keywords. Steps 2+ find cross-references in the observations and search for them. The neural query generator is a fallback. Answer extraction searches observations in reverse order (most recent first).
- `act_agent(question)`: Searches once with question keywords. No reasoning, no iteration.
- `cot_agent(question)`: Finds the single most similar document. No tools, no multi-hop.

**ReAct loop data flow:**
```
question → tfidf_vector → q_vec (121,)
     ↓
Step 1: search(question_keywords) → doc_1 → observe → obs_text
     ↓
Step 2: find_cross_references(obs_text) → doc_title → search(doc_title) → doc_2 → observe
     ↓
Step 3: find_cross_references(all_obs) → doc_title → search(doc_title) → doc_3 → observe
     ↓ (or no more cross-refs → Finish)
extract_answer(observations, question) → answer
```

### 10. Example Trajectory

Runs ReAct on "Where does the river that carved the Grand Canyon flow?" and prints the full Thought-Action-Observation trace showing the cross-reference follow from Grand Canyon → Colorado River.

### 11. Evaluation

**Function: `check_answer(predicted, expected)`** — substring match (case-insensitive)

Evaluates all three methods on 15 test questions. ReAct: 14/15 = 93%. Act: 11/15 = 73%. CoT: 11/15 = 73%.

### 12-13. Results by Complexity and Plot

Breaks down accuracy by hop count. Single-hop: ReAct 10/11, Act 11/11, CoT 11/11. Multi-hop: ReAct 4/4, Act 0/4, CoT 0/4. Two-panel plot: overall accuracy bars and accuracy-by-complexity grouped bars.

### 14. Final Evaluation Summary

Prints every number quoted in the docs as "Name: value" lines.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `tokenize(text)` | Lowercase + split on non-alphanumeric |
| `tfidf_vector(text)` | Convert text to L2-normalized TF-IDF vector |
| `search(query, top_k)` | Cosine similarity search over toy Wikipedia |
| `search_by_vec(query_vec, top_k)` | Search using a vector directly (for neural generator) |
| `QueryGenerator` | MLP that generates search queries from question + observation |
| `build_training_data(questions)` | Create synthetic ReAct trajectories for training |
| `find_cross_references(text, titles, seen)` | Find unvisited document titles mentioned in text |
| `react_agent(question)` | Run the full Thought-Action-Observation loop (3 steps max) |
| `act_agent(question)` | Act-only baseline (single search, no reasoning) |
| `cot_agent(question)` | CoT-only baseline (single document match, no tools) |
| `check_answer(pred, expected)` | Substring-match accuracy check |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Knowledge source | Real Wikipedia API (Search, Lookup) | 20-document toy corpus with TF-IDF search |
| Model | PaLM-540B (frozen, few-shot prompted) | 28K-parameter MLP query generator |
| Reasoning | Free-form natural language thoughts | Cross-reference following + neural query fallback |
| Training | No training (few-shot prompting) | 30 epochs on 40 synthetic trajectories |
| Tasks | HotpotQA, FEVER, ALFWorld, WebShop | 55 toy geography questions (40 train, 15 test) |
| Evaluation | EM, accuracy, success rate | Substring-match accuracy on 15 test questions |
