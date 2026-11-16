---
title: "HyDE: Code architecture"
parent: "077 HyDE"
nav_order: 1
---
# Code Architecture — HyDE Notebook

## Overview

The notebook implements the full HyDE retrieval pipeline: a hypothetical document generator (stand-in for InstructGPT) produces answer-like passages from query keywords, a TF-IDF encoder (stand-in for Contriever) embeds them, the embeddings are averaged (paper Eq. 6-8), and real corpus documents are retrieved by cosine similarity. The result is measured by Recall@5 and nDCG@5 on 10 queries against a 20-passage synthetic corpus. HyDE improves Recall@5 from 0.5500 (raw query baseline) to 0.6500 and nDCG@5 from 0.6232 to 0.7164.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants (copied from the code cell):
- `NUM_HYPOTHETICAL_DOCS = 4` — number of hypothetical documents generated per query (paper Section 3.2, N)
- `TOP_K = 5` — number of documents to retrieve
- `INCLUDE_QUERY_AS_HYPOTHESIS = True` — paper Eq. 8: also include the raw query as a hypothesis
- `SEED = 42` — random seed for reproducibility

### 2. Data — Small Synthetic Corpus

**Data structure: `CORPUS`** — a list of 20 `(doc_id, text, topic)` tuples across 5 topics (programming, biology, history, ai, geography), 4 passages each. The text passages are enriched with domain vocabulary (e.g., "Himalayas", "glucose", "ATP", "barbarians") so that the concept-map-expanded hypothetical documents share terms with relevant real documents.

This replaces the paper's MS-MARCO (8.8M passages), BEIR (6 low-resource datasets), and Mr.Tydi (4 multilingual languages).

### 3. Query Set

**Data structure: `QUERIES`** — 10 `(query_text, relevant_topic)` tuples. The relevant topic serves as ground truth for evaluation. Queries span all 5 topics.

### 4. TF-IDF Document Encoder

**Class: `TfidfEncoder`** — unsupervised text encoder that maps text to L2-normalized TF-IDF vectors.
- `fit(documents)` — builds vocabulary and computes IDF: `log((N+1)/(df+1)) + 1` (sklearn-style smoothing)
- `encode(text)` — returns a TF-IDF vector, L2-normalized

This replaces Contriever (110M-parameter BERT-based contrastive encoder). The vocabulary size is about 100 terms. The key property HyDE needs — mapping documents into a space where similarity reflects topical relevance — is preserved: passages about the same topic share vocabulary, so their TF-IDF vectors have high inner product.

**Data flow:**
```
Text: "Mount Everest is the highest peak on Earth..."
  -> tokenize -> ['mount', 'everest', 'is', 'the', 'highest', 'peak', 'on', 'earth', ...]
  -> TF-IDF weighting -> sparse vector (~100 dims)
  -> L2 normalize -> unit vector
Document embeddings shape: (20, ~100)
```

### 5. Hypothetical Document Generator

**Data: `CONCEPT_MAP`** — a dictionary mapping query keywords to related terms drawn from the corpus vocabulary. This simulates the LLM's world knowledge: when a user asks "tallest mountain", a real LLM knows to talk about "Mount Everest", "Himalayas", "height", "Earth" — terms that appear in the relevant corpus passage.

**Function: `extract_keywords(query)`** — removes stopwords and question words, returns content keywords.

**Function: `expand_keywords(keywords)`** — looks up each keyword in `CONCEPT_MAP` and appends related terms. This is the simulation of LLM generation: the expanded terms are what the LLM "knows" about the topic.

**Function: `generate_hypothetical_document(query, seed)`** — uses one of 4 template functions to construct a passage from the expanded keywords. Each template produces a different-sounding passage (simulating temperature 0.7 sampling). The generated text shares vocabulary with relevant corpus documents but contains fabricated filler — exactly as the paper describes.

This replaces InstructGPT (text-davinci-003, 175B parameters, temperature 0.7).

### 6. HyDE Retrieval Pipeline

**Function: `hyde_retrieve(query, ...)`** — the full HyDE pipeline:
1. Generate N hypothetical documents (Section 3.2, Step 1)
2. Encode each with the TF-IDF encoder (Step 2)
3. Average the embeddings; optionally include the raw query (Eq. 6-8)
4. Compute inner product with all document embeddings; return top-K (Step 4)

**Function: `baseline_retrieve(query, ...)`** — the Contriever baseline: encode the raw query directly and retrieve. No hypothetical document.

**Function: `bm25_retrieve(query, ...)`** — standard BM25 lexical retrieval with k1=1.5, b=0.75.

**Data flow:**
```
Query: "How do cells make energy?"
  -> extract_keywords -> ['cells', 'make', 'energy']
  -> expand_keywords -> ['cells', 'make', 'energy', 'cell', 'mitochondria', 'organelles', 'respiration', 'division']
  -> generate 4 hypothetical documents (templates 0-3)
  -> encode each -> 4 TF-IDF vectors (~100 dims each)
  -> average + include raw query embedding -> 1 vector (~100 dims)
  -> inner product with 20 doc embeddings -> 20 similarities
  -> top-5 -> retrieved documents
```

### 7. Evaluation — Retrieval Quality

**Function: `compute_recall_at_k(...)`** — fraction of relevant (same-topic) documents in top-K.

**Function: `compute_ndcg_at_k(...)`** — normalized discounted cumulative gain: rewards relevant docs at higher ranks. DCG = sum(rel_i / log2(i+2)), normalized by ideal DCG.

**Function: `evaluate_all(...)`** — runs all three methods (baseline, HyDE, BM25) on all 10 queries, averages the metrics.

### 8. Per-Query Analysis

Prints a table comparing baseline vs HyDE Recall@5 per query, with improvement column. Shows how many queries HyDE helped, hurt, or left unchanged.

### 9. Visualization

Two bar charts: Recall@5 and nDCG@5 for baseline, HyDE, and BM25. One grouped bar chart: per-query Recall@5 for baseline vs HyDE.

### 10. Embedding Space Visualization

PCA (via SVD) projection of all 20 corpus documents + raw query embedding + HyDE average embedding to 2D. Color-coded by topic. Shows the HyDE embedding landing closer to the relevant document cluster than the raw query embedding.

### 11. Effect of Number of Hypothetical Documents

Sweeps N from 1 to 16 (without query inclusion). Plots Recall@5 and nDCG@5 vs N, with baseline as a dashed reference line. Shows that more samples reduce noise and improve quality, with diminishing returns — consistent with the paper's expectation-based estimation (Eq. 6-7).

### 12. Final Results Summary

Prints all key metrics as "Name: value" lines.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `TfidfEncoder` | Unsupervised TF-IDF text encoder (stand-in for Contriever) |
| `extract_keywords` | Extract content words from a query, removing stopwords |
| `expand_keywords` | Expand query keywords via concept map (simulates LLM world knowledge) |
| `generate_hypothetical_document` | Generate a hypothetical answer passage from expanded keywords |
| `hyde_retrieve` | Full HyDE pipeline: generate, encode, average, retrieve |
| `baseline_retrieve` | Raw query embedding baseline (no hypothetical document) |
| `bm25_retrieve` | BM25 lexical retrieval baseline |
| `compute_recall_at_k` | Recall@K metric |
| `compute_ndcg_at_k` | nDCG@K metric |
| `evaluate_all` | Evaluate all methods on all queries |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Corpus | MS-MARCO (8.8M passages), BEIR (6 datasets), Mr.Tydi (4 languages) | 20 synthetic passages across 5 topics |
| Generative model | InstructGPT (text-davinci-003, 175B params, temp 0.7) | Template-based generator with concept map (~50 entries) |
| Encoder | Contriever / mContriever (110M params, contrastive learning) | TF-IDF (~100-dim vocabulary) |
| Retrieval | Pyserini MIPS at billion scale | Full-corpus cosine similarity (20 docs) |
| Evaluation | nDCG@10, MAP, Recall@1k with human judgments | Recall@5, nDCG@5 with topic-level ground truth |
| Number of hypothetical docs | N=1 (main), varied in analysis | N=4 (main), swept 1-16 in analysis |