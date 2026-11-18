---
title: "080 GraphRAG"
nav_order: 81
permalink: "/080_graphrag-graph-rag-query-summarization/"
---
# From Local to Global: A Graph RAG Approach to Query-Focused Summarization (GraphRAG)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/graphrag-graph-rag-query-summarization)

- **Paper:** [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)
- **Authors:** Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson
- **Lab, year:** Microsoft Research and Microsoft Strategic Missions and Technologies, 2024 (arXiv preprint; CC BY 4.0)

## Summary

Retrieval-augmented generation (RAG) lets an LLM answer questions by pulling a few relevant text snippets from a large document collection before generating its answer. This works well for specific lookup questions — "What did the CEO say about Q3 earnings?" — but it fails for global questions like "What are the main themes across this entire corpus?" Answering that kind of question requires summarising the whole dataset, not retrieving a handful of records. The paper calls these "global sensemaking" questions.

GraphRAG solves this by first using an LLM to extract a knowledge graph (a graph of entities and their relationships) from the source documents. It then runs community detection — an algorithm that groups closely connected nodes into clusters — to partition this graph into a hierarchy of communities. The LLM writes a summary for each community. When a user asks a global question, each community summary produces a partial answer, and a final summarisation step merges them into one global answer. This map-reduce approach lets the system reason about an entire corpus without fitting it all into a single context window.

On two datasets of roughly one million tokens (podcast transcripts and news articles), GraphRAG with GPT-4 substantially outperformed conventional vector RAG on comprehensiveness (72-83% win rate) and diversity (62-82% win rate) of answers to global questions. Root-level community summaries achieved this while using over 97% fewer tokens than source-text summarisation.

This repo's notebook builds a miniature GraphRAG pipeline from scratch on a small synthetic corpus: it extracts a tiny entity graph, detects communities with the Leiden algorithm, generates community summaries, and answers a global question via map-reduce summarisation.

## What Problem Does It Solve

Imagine you have a thousand news articles and you ask, "What are the biggest trends in health reporting?" A traditional RAG system searches for articles matching "health trends" and returns a few of them. But the question is not asking for specific articles — it is asking you to read everything and summarise the big picture. That is a fundamentally different task: summarisation, not retrieval.

The paper frames this as the difference between "local" and "global" questions. Vector RAG (the standard approach, which finds text chunks semantically similar to a query) handles local questions well but has no mechanism for global understanding. GraphRAG bridges this gap by building a structured index — the entity graph and its community hierarchy — that captures the big-picture structure of the corpus, then using that index to generate global answers.

## Core Idea

- **LLM-derived knowledge graph:** An LLM reads each text chunk and extracts named entities (people, organisations, places) and the relationships between them, building a graph where nodes are entities and edges are relationships weighted by how often they co-occur.
- **Hierarchical community detection:** The Leiden algorithm partitions the graph into nested communities — groups of entities that are densely connected to each other but sparsely connected to the rest of the graph. Each level of the hierarchy provides a different granularity of thematic grouping.
- **Map-reduce global answering:** Each community summary independently generates a partial answer to the user's question. Partial answers are scored for helpfulness, filtered, and then merged into a single global answer — a divide-and-conquer approach that scales to corpora far larger than any single context window.

## Key Method Details

- **Chunking:** Documents are split into 600-token chunks with 100-token overlaps (Section 3.1.1).
- **Entity extraction:** The LLM is prompted to extract entities, relationships, and claims from each chunk using domain-tailored few-shot exemplars (Section 3.1.2). Entity matching uses exact string matching.
- **Community detection:** Hierarchical Leiden algorithm via the graspologic library (Section 3.1.4).
- **Community summaries:** Leaf-level communities are summarised from element descriptions (nodes, edges, claims); higher-level communities recursively incorporate sub-community summaries (Section 3.1.5).
- **Global answering:** Community summaries are shuffled, chunked, mapped to partial answers with helpfulness scores (0-100), filtered, and reduced into a final answer (Section 3.1.6).
- **Results:** GraphRAG comprehensiveness win rates of 72-83% over vector RAG (p<.001) across both datasets; diversity win rates of 62-82%. Root-level summaries (C0) use 97% fewer tokens than source-text summarisation while retaining 72% comprehensiveness win rate (Section 5.1).
- **Evaluation:** LLM-as-a-judge with head-to-head comparisons on comprehensiveness, diversity, empowerment, and directness. 125 questions per dataset, each comparison repeated 5 times (Sections 3.2-3.3).

### Simplifications in This Notebook

- Tiny synthetic corpus (22 short text passages, ~255 words) instead of ~1M-token real datasets
- Rule-based entity extraction with predefined entity lists instead of LLM-based extraction
- Single-level community detection instead of hierarchical multi-level Leiden
- Template-based community summaries instead of LLM-generated summaries
- Simple keyword-overlap scoring instead of LLM-as-a-judge evaluation
- No GPU required — the entire pipeline runs on CPU in under a minute

### What the Notebook Result Shows

On a synthetic corpus of 22 passages, the notebook extracts an entity graph with 23 nodes and 42 edges across 3 connected components (one per theme), detects 5 communities with modularity Q = 0.914 using a simplified greedy Leiden-like algorithm, and generates community summaries. For the global question "What are the main themes in this corpus?", the map-reduce approach produces a final answer identifying 5 thematic communities (3 within technology, 1 for health, 1 for politics) with helpfulness scores of 40-70. A comparison with simulated vector RAG shows that GraphRAG identifies corpus-wide thematic structure, while vector retrieval returns individual passages without any thematic grouping. The notebook runs entirely on CPU in about 2 seconds.

## Influence

GraphRAG became one of the most influential RAG papers of 2024, with about 3,600 citations ([OpenAlex](https://openalex.org/works/W4395686609), checked November 2026). Microsoft released the reference implementation as open-source software at [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag), and the approach was integrated into LangChain, LlamaIndex, NebulaGraph, and Neo4J. The paper's insight — that community detection on an LLM-derived knowledge graph enables global sensemaking — influenced subsequent work on hybrid RAG systems combining local retrieval with global summarisation. The paper has not yet been formally published at a venue (it remains an arXiv preprint as of the most recent version, February 2025).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/080_graphrag-graph-rag-query-summarization/solution.ipynb) | From-scratch Python pipeline that extracts a tiny entity graph, detects communities with Leiden, generates community summaries, and answers a global question via map-reduce on CPU |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, key functions, and deliberate simplifications vs the full paper |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
