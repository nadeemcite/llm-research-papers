---
title: "079 Self-RAG"
nav_order: 80
permalink: "/079_self-rag-retrieve-generate-critique/"
---
# Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Self-RAG)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/self-rag-retrieve-generate-critique)

- **Paper:** [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- **Authors:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
- **Lab, year:** University of Washington; Allen Institute for AI (Hajishirzi and Wang: both UW and AI2); IBM Research AI (Sil), 2023 (ICLR 2024)

## Summary

Self-RAG (Self-Reflective Retrieval-Augmented Generation) is a framework that trains a single language model to decide for itself when to retrieve documents, whether those documents are relevant, and whether its own generated answer is actually supported by them. Unlike standard RAG, which always retrieves a fixed number of passages regardless of necessity, Self-RAG adapts retrieval on demand — retrieving only when the model judges it helpful, and then critiquing both the retrieved passages and its own output using special "reflection tokens."

The framework introduces four types of reflection tokens: a "Retrieve" token (yes/no/continue) that decides when to call the retriever, and three "critique" tokens — IsRel (is the passage relevant?), IsSup (is the output supported by the passage?), and IsUse (is the response useful overall?).

A critic model, trained on GPT-4-annotated data, generates these tokens offline to augment the training corpus. The generator model (based on Llama 2, in 7B and 13B sizes) then learns to produce both task output and reflection tokens as part of standard next-token prediction, eliminating the need for the critic at inference time.

At inference, segment-level beam search uses the critique token probabilities to rank candidate outputs, enabling controllable behavior — e.g., prioritizing factual support for knowledge-intensive tasks or creativity for open-ended ones.

Self-RAG (7B and 13B) outperforms ChatGPT and retrieval-augmented Llama2-chat on several tasks: PopQA (54.9/55.8% vs Ret-ChatGPT's 50.8%), PubHealth fact verification (72.4/74.5% accuracy), and biography generation (FactScore 81.2/80.2%). It also shows significantly higher citation precision and recall on long-form QA (ASQA: 66.9/70.3% precision vs 2-5% for non-Self-RAG retrieval baselines).

This repo's notebook implements a miniature Self-RAG pipeline with a tiny Transformer that learns to emit reflection tokens (Retrieve, IsRel, IsSup, IsUse) in a controlled synthetic QA setting, demonstrating the core adaptive-retrieval and self-critique loop.

## What Problem Does It Solve

Imagine you are writing an essay. For some questions you need to look things up in a book; for others, you already know the answer. If you had to look up every single question — even ones like "what did you eat for breakfast?" — you would waste time and probably get confused by irrelevant information. Standard RAG systems have exactly this problem: they always retrieve passages, even when the question does not require external knowledge.

Self-RAG teaches a language model to act more like a thoughtful human writer. It learns to ask: "Do I need to look this up?" If yes, it retrieves passages and checks whether each one is relevant. After generating an answer, it checks: "Is what I wrote actually backed by the source?" This self-reflection makes the model more factual and more versatile — it can handle both knowledge-intensive questions (where retrieval helps) and creative tasks (where retrieval would hurt).

## Core Idea

- **On-demand retrieval via reflection tokens:** The model itself decides whether to retrieve, rather than always retrieving. A "Retrieve" token (yes/no) gates the retrieval step.
- **Self-critique of retrieval and generation:** Three critique token types — IsRel (relevance), IsSup (support), IsUse (utility) — let the model evaluate retrieved passages and its own outputs, enabling segment-level re-ranking at inference.
- **Offline critic annotation, no critic at inference:** A separate critic model (trained on GPT-4 annotations) augments the training data with reflection tokens offline. The generator learns to produce these tokens itself, so no critic is needed at test time.

## Key Method Details

- **Reflection tokens (Table 1):** Retrieve (yes/no/continue), IsRel (relevant/irrelevant), IsSup (fully supported/partially supported/no support), IsUse (1-5 utility score). The bold values indicate the most desirable token (Section 3.1).
- **Critic training (Section 3.2.1):** GPT-4 is prompted with type-specific instructions to generate reflection tokens for 4k–20k sampled instances per token type. A Llama 2-7B model is fine-tuned on this data. The critic achieves >90% agreement with GPT-4 on most token categories (Appendix Table 5).
- **Generator training (Section 3.2.2):** 150k instruction-output pairs from Open-Instruct and knowledge-intensive datasets. Retrieved passages (from Contriever-MS MARCO) and reflection tokens are interleaved into the training corpus. The generator is trained with standard next-token prediction (Eq. 2), with retrieved text chunks masked from the loss. Base models: Llama 2 7B and 13B.
- **Inference (Algorithm 1, Section 3.3):** For each segment, the model predicts Retrieve. If yes, K passages are retrieved and processed in parallel; the model generates IsRel, the segment, and IsSup for each passage. Segment-level beam search (beam width 2) ranks candidates using a weighted sum of critique token probabilities (Eq. 4): default weights IsRel=1.0, IsSup=1.0, IsUse=0.5. Retrieval threshold defaults to 0.2.
- **Results (Table 2):** Self-RAG 13B achieves 55.8% on PopQA, 69.3% on TriviaQA, 74.5% on PubHealth, 73.1% on ARC-Challenge, 80.2% FactScore on biographies, and 70.3% citation precision on ASQA. It outperforms ChatGPT (retrieval-augmented) on PopQA, PubHealth, biography FactScore, and ASQA citation precision.

### Simplifications in This Notebook

- Tiny model: 2-layer Transformer decoder, about 449K parameters (the paper uses Llama 2 7B/13B)
- Synthetic QA task: 40 subjects, 10 possible answers, dynamically generated passages (the paper uses 150k instruction-output pairs from Open-Instruct and real QA benchmarks)
- 4 reflection token types implemented as binary/ternary classification heads (the paper uses multi-valued tokens, e.g., IsUse 1-5)
- 40 epochs with AdamW (lr 1e-3) instead of the paper's fine-tuning schedule on Llama 2
- Dynamic passage provider (correct passage + distractor passages) instead of Contriever-MS MARCO

### What the Notebook Result Shows

On the synthetic QA task, the tiny Self-RAG model achieves 100% answer accuracy when passages are included and 100% Retrieve token accuracy — it perfectly learns which subjects need retrieval. Without passages, answer accuracy drops to about 54% (factual questions fail at about 8% since the answer is random and only in the passage; open-ended questions remain at 100% since the answer is in the question).

The adaptive mode (which retrieves only when the model predicts Retrieve=yes) achieves 100% overall accuracy, matching always-retrieve and far outperforming never-retrieve (54%). The model retrieves for 50% of questions, matching the ground truth exactly.

These numbers are not comparable to the paper's results (which used 7B–13B models on real benchmarks like PopQA and PubHealth), but they demonstrate the core qualitative finding: adaptive retrieval matches the best mode for each question type.

## Influence

Self-RAG became a widely cited reference for adaptive retrieval and self-reflection in LLMs. It influenced subsequent work on training models to self-evaluate and selectively retrieve, including CRAG (Yan et al., 2024) which introduces a lightweight retrieval evaluator, and Adaptive-RAG (Jeong et al., 2024) which learns to classify query complexity to choose retrieval strategies. The reflection-token paradigm — augmenting the vocabulary with control tokens for self-assessment — connected RAG research to the broader literature on controllable generation and LLM self-evaluation. The paper was accepted at ICLR 2024. The authors released code and trained models at [the Self-RAG GitHub repo](https://github.com/AkariAsai/self-rag).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/079_self-rag-retrieve-generate-critique/solution.ipynb) | From-scratch PyTorch 2-layer Transformer with reflection-token heads, trained on a synthetic QA task with adaptive retrieval and self-critique |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
