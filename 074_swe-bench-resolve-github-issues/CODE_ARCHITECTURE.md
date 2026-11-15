---
title: "SWE-bench: Code architecture"
parent: "074 SWE-bench"
nav_order: 1
---
# Code Architecture — SWE-bench Notebook

## Overview

The notebook implements a miniature SWE-bench evaluation framework: 4 synthetic Python repositories with 20 planted-bug task instances, BM25 file retrieval, a small neural embedding model trained with contrastive loss to retrieve similar instances, retrieval-based patch generation via nearest-neighbor editing, and execution-based evaluation that applies generated patches and runs fail-to-pass tests. Results show resolve rates with failure-type analysis, and oracle retrieval compared to neural retrieval — matching the paper's qualitative findings.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines key constants:
- `SEED = 42` — reproducibility
- `EPOCHS = 50`, `LR = 3e-3`, `BATCH_SIZE = 4`
- `EMBED_DIM = 128`, `HIDDEN_DIM = 256`, `NUM_HEADS = 4`, `NUM_LAYERS = 3`
- `MAX_IN = 400`, `MAX_OUT = 120` — input/output length bounds

### 2. Mini SWE-bench Dataset Construction

**Data structures: `REPO_TEMPLATES`, `BUG_INSTANCES`** — 4 repository templates (mathlib, textutils, datastruct, cachekit), each with 2–3 source files and a test file. 20 bug instances, each specifying a file, the buggy line(s), the buggy replacement, and the fail-to-pass tests.

Mirrors the paper's 3-stage pipeline (Section 2.1): repo selection → attribute filtering → execution-based filtering, but hand-crafted rather than scraped from GitHub.

### 3. Build Buggy Codebases

**Function: `make_buggy_repo`** — takes a template and a bug spec, applies the buggy replacement to the target file, returns the modified file dict.

**Function: `make_gold_patch`** — returns the clean (correct) file content from the template.

Builds the `instances` list, each with: instance_id, repo, issue text, target file, buggy files, gold content, buggy line, fix line, fail-to-pass tests, and test code. Splits into 15 training and 5 test instances.

### 4. BM25 File Retrieval

**Class: `BM25Retriever`** — implements BM25 ranking (k1=1.5, b=0.75). Tokenizes file contents and issue text, builds TF/DF index, scores files by BM25 relevance to the issue query, returns top-k file paths.

Mirrors the paper's sparse retrieval approach (Section 4.1), where BM25 is chosen over dense retrieval due to long document lengths and the code-to-natural-language retrieval setting.

### 5. Neural Patch Retrieval Model

**Function: `build_vocab`** — builds a word-level vocabulary from issue + buggy-line texts.

**Function: `tfidf_vector`** — converts text to a TF-IDF feature vector using the vocabulary and inverse document frequencies.

**Class: `EmbeddingNet`** — a 2-layer MLP (~38K parameters) that maps TF-IDF features to a 128-dimensional embedding space with L2 normalization. Architecture: Linear(163 → 128) → ReLU → Dropout → Linear(128 → 128) → L2-normalize.

**Data flow:**
```
Input: issue + buggy_line → TF-IDF vector (193-dim) → EmbeddingNet → 128-dim embedding
```

### 6. Training the Embedding Model

- Optimizer: Adam, lr=3e-3
- Loss: CrossEntropyLoss on the similarity matrix (each instance should be most similar to itself)
- 50 epochs, all 15 training instances per epoch (full-batch)
- Gradient clipping: not used (simple MLP converges on small data)

### 7. Training Curves

Plots training loss over 50 epochs. Loss decreases steadily and plateaus, reflecting the small dataset and model capacity.

### 8. Patch Generation via Retrieval

**Function: `get_embedding`** — embeds a query string using the trained EmbeddingNet.

**Function: `apply_edit`** — given the query's buggy line, the nearest neighbor's buggy line and fix line, applies a structured edit: if lines match exactly, return the fix; if token-level diffs are single-token, apply a targeted replacement; otherwise return the neighbor's fix directly.

**Data flow:**
```
Query: issue + buggy_line → embedding → similarity to all training embeddings
→ best match (nearest neighbor) → apply_edit(query_buggy, neighbor_buggy, neighbor_fix) → predicted fix
→ replace buggy line in codebase → patched content
```

### 9. Execution-Based Evaluation

**Function: `evaluate_instance`** — writes the patched codebase to a temp directory, runs pytest on the fail-to-pass tests, classifies outcome as resolved or failure type (syntax_error, import_error, test_failure, runtime_error, timeout, eval_error).

Mirrors the paper's execution-based evaluation (Section 2.2): patch applied, tests run, resolution = all fail-to-pass tests pass.

### 10. Failure Type Analysis

Counts failure types, analyzes generated vs gold patch lengths (mirroring paper Table 8), and breaks down resolve rate by repository (mirroring paper Figure 4).

### 11. Results Plots

Two subplots: (1) failure type distribution bar chart, (2) per-repository resolve rate bar chart.

### 12. Context Quality Comparison

Evaluates with oracle retrieval (gold fix line directly applied) and compares to neural retrieval. Mirrors the paper's oracle retrieval analysis (Section 4.1, Table 18).

### 13. Final Evaluation

Prints summary statistics: total instances, resolve rates (retrieval and oracle), failure type breakdown, and comparison to the paper's reference numbers.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `make_buggy_repo` | Apply a planted bug to a repository template |
| `BM25Retriever` | BM25-based file retrieval from issue text |
| `build_vocab` | Build word-level vocabulary from texts |
| `tfidf_vector` | Convert text to TF-IDF feature vector |
| `EmbeddingNet` | 2-layer MLP mapping TF-IDF to embedding space |
| `get_embedding` | Embed a query string via the trained model |
| `apply_edit` | Generate a fix from nearest-neighbor edit transfer |
| `evaluate_instance` | Apply patch, run tests, classify outcome |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Repositories | 12 real Python repos (thousands of files) | 4 synthetic repos (3–5 files each) |
| Task instances | 2,294 real GitHub issues | 20 hand-crafted planted bugs |
| Model | CodeLlama 7B/13B fine-tuned with LoRA | 2-layer MLP (~41K params) for retrieval |
| Patch generation | Seq2seq transformer generating patches | Nearest-neighbor retrieval + structured edit |
| Training data | 19,000 issue-PR pairs from 37 repos | 15 instances (train), 5 held out (test) |
| Tokenizer | BPE (CodeLlama) | Word-level TF-IDF (~193 vocab) |
| Retrieval | BM25 over full codebase | BM25 over 3–5 files |
| Evaluation | Docker-based test execution | pytest in temp directory |
