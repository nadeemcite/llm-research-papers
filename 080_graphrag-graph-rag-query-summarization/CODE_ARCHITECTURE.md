---
title: "GraphRAG: Code architecture"
parent: "080 GraphRAG"
nav_order: 1
---
# Code Architecture — GraphRAG Notebook

## Overview

The notebook implements a miniature GraphRAG pipeline on a synthetic corpus of 20 short text passages. It extracts an entity-relationship graph, detects communities using the Leiden algorithm, generates template-based community summaries, and answers a global question via map-reduce summarisation. The result is measured by whether the global answer correctly identifies the three thematic communities in the corpus.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `NUM_PASSAGES = 20` — number of synthetic text passages (paper uses ~1M tokens)
- `CHUNK_SIZE = 600` — token chunk size for reference (paper uses 600-token chunks with 100-token overlaps)
- `SEED = 42` — random seed for reproducibility

### 2. Synthetic Corpus

**Function: `build_corpus()`** — Creates 20 short text passages spanning three themes (technology, health, politics). Each passage is 2-3 sentences mentioning 2-4 entities. Returns a list of 22 text passages. This replaces the paper's million-token podcast and news datasets.

### 3. Entity Extraction

**Function: `extract_entities(passages)`** — Uses rule-based matching with predefined entity lists to extract entities and relationships from each passage. Returns a dictionary mapping entity names to descriptions and a list of (source, target, description) relationship tuples. This replaces the paper's LLM-based extraction with few-shot prompting (Section 3.1.2).

### 4. Knowledge Graph Construction

**Function: `build_graph(entities, relationships)`** — Builds a NetworkX graph where nodes are entities and edges are relationships weighted by co-occurrence count. Aggregates duplicate relationships into weighted edges (Section 3.1.3). Returns a NetworkX Graph object with 23 nodes and 42 edges across 3 connected components.

### 5. Community Detection

**Function: `detect_communities(graph)`** — Runs the Leiden algorithm via the `community` package to partition the graph into communities. Returns a dictionary mapping node names to community IDs. The notebook uses single-level Leiden instead of the paper's hierarchical multi-level approach (Section 3.1.4). Typically detects 5 communities: 3 within technology (Apple/TSMC, Google, Nvidia/Microsoft/OpenAI), 1 for health, 1 for politics, with modularity Q ≈ 0.91.

### 6. Community Summaries

**Function: `generate_community_summaries(graph, communities, entities)`** — For each detected community, collects the entities and relationships in that community and generates a template-based summary listing the key entities and their connections. This replaces the paper's LLM-generated community summaries with element prioritisation (Section 3.1.5).

### 7. Global Question Answering (Map-Reduce)

**Function: `answer_global_question(question, community_summaries)`** — Implements the map-reduce approach from Section 3.1.6:
- **Map:** Each community summary generates a partial answer to the question with a helpfulness score (0-100). Template-based scoring uses keyword overlap between the question and community summary.
- **Filter:** Partial answers with score 0 are removed.
- **Reduce:** Remaining partial answers are combined into a final global answer.

Returns a tuple of (global_answer, list of partial answers with scores).

### 8. Evaluation and Comparison

**Function: `compare_with_vector_rag(question, passages)`** — Simulates vector RAG by selecting the top-3 passages with highest keyword overlap with the question, then concatenating them as the "answer." This provides a baseline to contrast with the graph-based global answer.

### 9. Visualisation

Plots the entity graph with nodes coloured by community, showing the community structure visually. Also prints a side-by-side comparison of the GraphRAG answer vs the vector-RAG-style answer.

## Key Functions

| Function / Class | Purpose |
|---|---|
| `build_corpus()` | Generate 20 synthetic text passages across 3 themes |
| `extract_entities(passages)` | Rule-based entity and relationship extraction |
| `build_graph(entities, relationships)` | Construct weighted NetworkX knowledge graph |
| `detect_communities(graph)` | Run Leiden community detection |
| `generate_community_summaries(graph, communities, entities)` | Template-based summaries per community |
| `answer_global_question(question, community_summaries)` | Map-reduce global answering with scoring |
| `compare_with_vector_rag(question, passages)` | Simulate vector RAG baseline for comparison |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Dataset | ~1M tokens (podcast transcripts, news articles) | 20 synthetic passages (~500 tokens total) |
| Entity extraction | LLM with few-shot domain-tailored prompts | Rule-based matching with predefined lists |
| Community detection | Hierarchical Leiden (multiple levels C0-C3) | Single-level Leiden |
| Community summaries | LLM-generated with element prioritisation | Template-based listing entities and edges |
| Global answering | LLM map-reduce with 0-100 helpfulness scoring | Keyword-overlap scoring and template merging |
| Evaluation | LLM-as-a-judge, 125 questions, 5 replicates | Single global question, qualitative comparison |
| Infrastructure | GPT-4-Turbo, 8k context windows, 281 min indexing | Pure Python/NetworkX, CPU, ~2 seconds |
