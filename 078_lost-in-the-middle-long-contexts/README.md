---
title: "078 Lost in the Middle"
nav_order: 79
permalink: "/078_lost-in-the-middle-long-contexts/"
---
# Lost in the Middle: How Language Models Use Long Contexts (LitM)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/lost-in-the-middle-long-contexts)

- **Paper:** [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- **Authors:** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang
- **Lab, year:** Stanford University; University of California, Berkeley; Samaya AI (Liu: work partially completed as an intern at Samaya AI), 2023 (TACL 2024)

## Summary

"Lost in the Middle" is an empirical study of how well language models use information within long input contexts. While modern LLMs can accept contexts of 4K to 100K tokens, the paper asks a simple but overlooked question: does the model actually *use* all of that context effectively?

The authors test this with two tasks: multi-document question answering (finding the right passage among distractors) and synthetic key-value retrieval (finding the matching key-value pair). In both tasks, they control two variables — the total number of items (context length) and the position of the relevant item within the context. If models robustly use their context, performance should not depend on position.

They find a striking **U-shaped performance curve**: models perform best when the relevant information is at the very beginning (primacy bias) or very end (recency bias) of the context, and worst when it is in the middle. For example, GPT-3.5-Turbo's multi-document QA performance drops by more than 20% when the answer is in the middle, sometimes falling below its closed-book accuracy (56.1%) — meaning the model would have been better off with no documents at all. The effect persists across open models (MPT-30B-Instruct, LongChat-13B) and closed models (GPT-3.5-Turbo, Claude-1.3), and even in explicitly extended-context variants.

This repo's notebook trains a small 3-layer Transformer decoder (about 605K parameters) on a synthetic multi-document QA task and reproduces the position-dependent accuracy effect: a U-shaped curve where the beginning and end positions outperform the middle, with a best-worst gap of about 0.07 that grows with more documents.

## What Problem Does It Solve

Imagine searching for a specific fact in a long report. If the fact is in the first paragraph or the last paragraph, you will probably find it quickly. But if it is buried in the middle of a 50-page document, you might miss it — even though it is right there in front of you.

The paper shows that language models have the same problem. When you give an LLM a long context (say, 20 retrieved documents) and the answer is somewhere in the middle, the model often fails to find it. This is surprising because the Transformer's self-attention mechanism can technically attend to any token in the context — there is no architectural reason it should prefer the edges over the middle. Yet it does, and the effect is large enough to matter for real applications like retrieval-augmented generation (RAG), where a search engine feeds documents to the model.

## Core Idea

- **Position matters more than context length:** The key variable is not how long the context is, but *where* in the context the relevant information sits. Performance degrades in the middle regardless of total length.
- **The U-shaped curve:** Performance is highest at the beginning (primacy) and end (recency) of the context, creating a U-shape when plotted against position. This mirrors the serial-position effect from psychology (Ebbinghaus, 1913).
- **Extended-context models do not help:** Models with larger context windows (e.g., GPT-3.5-Turbo 16K vs 4K) perform nearly identically when the context fits in both — having a bigger window does not mean the model uses it better.

## Key Method Details

- **Multi-document QA task:** Uses NaturalQuestions-Open data; each context has k documents (passages of at most 100 tokens), exactly one containing the answer. Distractors retrieved by Contriever (fine-tuned on MS-MARCO). Accuracy = whether the correct answer appears in the model output (Section 2.1).
- **Key-value retrieval task:** k key-value pairs with UUID keys and values; model must return the value for a given key. Minimal testbed for basic retrieval ability (Section 3.1).
- **Models tested:** MPT-30B-Instruct (8K context), LongChat-13B 16K, GPT-3.5-Turbo (4K and 16K), Claude-1.3 (8K and 100K), Flan-T5-XXL, Flan-UL2. Greedy decoding throughout.
- **Results:** GPT-3.5-Turbo drops >20% from best to worst position in 20- and 30-document settings. Extended-context models (GPT-3.5-Turbo 16K, Claude-1.3 100K) show nearly identical performance to their standard counterparts when context fits both. Encoder-decoder models (Flan-UL2, Flan-T5-XXL) are more robust within their training-time sequence length but show U-shape on longer sequences (Section 4.1).
- **Query-aware contextualization** (placing the query before and after the documents) enables near-perfect key-value retrieval but minimally changes multi-document QA trends (Section 4.2).

### Simplifications in This Notebook

- Smaller model: 3 layers, about 605K parameters (the paper tests 7B–30B+ models)
- Dataset: synthetic token-level QA task (15 documents of 6 tokens each) instead of NaturalQuestions-Open passages
- 40 epochs with AdamW (lr 5e-4, weight decay 0.01) instead of zero-shot evaluation of pretrained LLMs
- Context length ~110 tokens instead of 4K–100K tokens

### What the Notebook Result Shows

On the synthetic 15-document QA task, the trained model achieves about 10% validation accuracy (above chance of 3.1%). The position evaluation reveals a U-shaped curve: the best position (14, near the end) achieves 11.3% accuracy while the worst position (11, in the middle) achieves 4.3%, a gap of 7 percentage points.

The mean accuracy of the first and last thirds of positions (7.8% and 8.1%) exceeds the middle third (7.4%), confirming the U-shape. The extended context experiment shows the gap grows with more documents: 1.7% with 10 documents, 7.7% with 15, and 12.3% with 20. These numbers are not comparable to the paper's results (which used 7B–30B+ pretrained models on real Wikipedia passages with 4K–100K token contexts), but the qualitative position-dependent pattern is consistent.

## Influence

The paper introduced the "needle in a haystack" evaluation paradigm that became a standard benchmark for long-context models. It influenced how the AI community evaluates context window claims — moving from "can the model accept 100K tokens?" to "can the model actually *use* information at any position in 100K tokens?" The finding that extended-context models are not necessarily better at using their context motivated work on retrieval-aware positioning and reranking. Follow-up work includes "Found in the Middle" (Chen et al., 2024), which addresses the positional bias via plug-and-play positional encoding, and "LooGLE" (Shi et al., 2024), which extends the evaluation to longer contexts. The paper was published in TACL (Transactions of the Association for Computational Linguistics) in 2024.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/078_lost-in-the-middle-long-contexts/solution.ipynb) | From-scratch PyTorch 3-layer Transformer decoder trained on a synthetic multi-document QA task, evaluating position-dependent accuracy (U-shaped curve) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |