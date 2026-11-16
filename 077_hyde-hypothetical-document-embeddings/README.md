---
title: "077 HyDE"
nav_order: 78
permalink: "/077_hyde-hypothetical-document-embeddings/"
---
# Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/hyde-hypothetical-document-embeddings)

- **Paper:** [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- **Authors:** Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
- **Lab, year:** Language Technologies Institute, Carnegie Mellon University; David R. Cheriton School of Computer Science, University of Waterloo, 2022 (ACL 2023)

## Summary

HyDE (Hypothetical Document Embeddings) is a zero-shot dense retrieval method that needs no relevance labels to work. Traditional dense retrievers learn to map queries and documents into a shared embedding space where inner product captures relevance — but learning this mapping requires labeled query-document pairs, which are expensive and not always available. HyDE sidesteps this entirely.

The key idea is to decompose retrieval into two tasks. First, an instruction-following language model (InstructGPT, 175B parameters) takes the query and generates a hypothetical answer document — a passage that "answers" the question but may contain factual errors. Second, an unsupervised contrastive encoder (Contriever, 110M parameters) encodes this hypothetical document into a vector. This vector is then used to retrieve similar real documents from the corpus by inner-product similarity. The encoder acts as a lossy compressor: it preserves the topical signal (what the passage is about) while filtering out the hallucinated details.

On the MS-MARCO web search benchmark (TREC DL19), HyDE achieves nDCG@10 of 61.3, compared to 44.5 for the Contriever baseline and 50.6 for BM25 — a 17-point improvement over the unsupervised baseline. It even matches ContrieverFT (62.1), which is fine-tuned on MS-MARCO's labeled data. Across BEIR low-resource tasks and Mr.Tydi multilingual tasks, HyDE consistently outperforms both Contriever and BM25.

This repo's notebook implements the full HyDE pipeline — hypothetical document generation, encoding, averaging, and retrieval — on a small synthetic corpus with a TF-IDF encoder and a concept-map-based generator. HyDE improves Recall@5 from 0.5500 (raw query baseline) to 0.6500 and nDCG@5 from 0.6232 to 0.7164, demonstrating the core mechanism on CPU without any LLM API calls.

## What Problem Does It Solve

Imagine you're building a search engine for a new website, and you have no examples of which documents are relevant to which queries. Traditional dense retrievers need thousands of labeled query-document pairs to learn how to match queries to documents. Without those labels, they perform poorly — often worse than simple keyword matching (BM25).

The paper explains it with its own comparison to traditional dense retrieval: learning to map queries and documents into the same embedding space requires fitting "relevance scores" between query-document pairs. Without relevance judgments, this learning becomes intractable. HyDE avoids this by never trying to learn query-document relevance directly. Instead, it offloads the "relevance understanding" to a language model that already knows what a relevant answer looks like (from its pretraining), and uses a separate document-only encoder that only needs to understand document-document similarity (learnable without labels via contrastive learning).

## Core Idea

- **Two-step decomposition:** Retrieval is split into (1) an NLG task — the LLM generates a hypothetical answer document — and (2) an NLU task — an unsupervised encoder maps documents into a similarity space. No query-document relevance scores are ever computed.
- **Hypothetical document as a bridge:** The generated document captures relevance patterns (what a relevant document would look like) even though it may contain false details. The encoder's dense bottleneck acts as a lossy compressor that filters out the hallucinated specifics.
- **Expectation-based query embedding:** Multiple hypothetical documents are generated (N samples) and their embeddings are averaged (Eq. 6-7), with an option to also include the raw query as an additional hypothesis (Eq. 8).
- **Fully unsupervised:** No models are trained or fine-tuned. The LLM and the contrastive encoder are both used as-is. The only "supervision" is in the LLM's instruction-following training.

## Key Method Details

- **Instruction:** *"Please write a passage that answers the query"* (Section 3.2, Appendix A.1). Different instructions are used for different task types (web search, QA, fact verification) and languages.
- **Generative model:** InstructGPT (text-davinci-003, 175B parameters), sampled at temperature 0.7 (Section 4.1).
- **Encoder:** Contriever (English) or mContriever (multilingual), trained with unsupervised contrastive learning (Izacard et al., 2021). The encoder is shared across all documents: `f = enc_d = enc_con` (Eq. 2-3).
- **Query vector:** Average of N hypothetical document embeddings, optionally including the raw query (Eq. 6-8). The paper uses N=1 in main experiments but studies the effect of N.
- **Retrieval:** Inner product between the averaged query vector and all document vectors; top-K documents are returned. Implemented via Pyserini for efficient MIPS.
- **Web search results (Table 1, DL19):** HyDE nDCG@10 61.3, MAP 41.8, Recall@1k 88.0 — vs. Contriever 44.5 / 24.0 / 74.6 and BM25 50.6 / 30.1 / 75.0 and ContrieverFT 62.1 / 41.7 / 83.6.
- **BEIR results (Table 2):** HyDE improves over Contriever on all 6 datasets. On Arguana, HyDE nDCG@10 46.6 vs. Contriever 37.9.
- **Multilingual results (Table 3, Mr.Tydi MRR@100):** HyDE improves mContriever on all 4 languages (Swahili 41.7 vs. 38.3, Korean 30.6 vs. 22.3, Japanese 30.7 vs. 19.5, Bengali 41.3 vs. 35.3).

### Simplifications in This Notebook

- Small synthetic corpus (20 short passages across 5 topics) instead of MS-MARCO / BEIR benchmarks
- TF-IDF encoder (unsupervised, ~100-dim vocabulary) instead of Contriever (110M-parameter BERT-based contrastive encoder)
- Concept-map-based hypothetical document generator (expands query keywords with domain vocabulary) instead of InstructGPT (text-davinci-003, 175B parameters, temperature 0.7)
- N=4 hypothetical documents per query, with query included as hypothesis (Eq. 8)
- Full-corpus cosine similarity instead of Pyserini MIPS at billion scale
- 10 test queries with topic-level ground truth, instead of 11 query sets with human relevance judgments

### What the Notebook Result Shows

On the 20-passage synthetic corpus with 10 test queries, HyDE improves Recall@5 from 0.5500 (raw query baseline) to 0.6500 (+0.1000) and nDCG@5 from 0.6232 to 0.7164 (+0.0932). HyDE helps on 4 of 10 queries and hurts on 0, with 6 unchanged. The N-sweep shows that averaging more hypothetical documents (N=1→8) improves nDCG@5 from 0.6750 to 0.6945 (without query inclusion), consistent with the paper's expectation-based estimation.

BM25 scores Recall@5 0.6250 and nDCG@5 0.6719 — between baseline and HyDE. These numbers are not comparable to the paper's MS-MARCO/BEIR results because the corpus, encoder, and generator are all drastically smaller. The point is to demonstrate that the HyDE mechanism — generating a hypothetical answer, encoding it, and retrieving by document-document similarity — improves retrieval even with minimal components.

## Influence

HyDE introduced a new paradigm for zero-shot retrieval that has been widely adopted in RAG (retrieval-augmented generation) systems. The idea of generating a hypothetical answer to improve retrieval has been extended in several directions: self-querying retrievers, query expansion via LLMs, and hypothetical chain-of-thought retrieval. HyDE is integrated into LangChain's retrieval toolkit as a standard retriever type. The paper's decomposition of retrieval into NLG + NLU tasks influenced subsequent work on LLM-based retrieval, including generate-then-retrieve pipelines and retrieval with instruction-tuned encoders. About 748 citations ([OpenAlex](https://openalex.org/W4385565351), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/077_hyde-hypothetical-document-embeddings/solution.ipynb) | Pure-Python implementation of HyDE — TF-IDF encoder, concept-map-based hypothetical document generator, embedding-averaging retrieval pipeline, evaluated on a 20-passage synthetic corpus with 10 queries |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, key functions, and deliberate simplifications vs the full paper |
| [TALK.md](TALK.md) | Press coverage, Q&A answered from the paper, citations, and common misconceptions about HyDE |