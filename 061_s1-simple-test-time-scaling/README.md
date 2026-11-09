---
title: "061 s1"
nav_order: 62
permalink: "/061_s1-simple-test-time-scaling/"
---
# s1: Simple test-time scaling (s1)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/s1-simple-test-time-scaling)

- **Paper:** [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)
- **Authors:** Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, Tatsunori Hashimoto
- **Lab, year:** Stanford University, University of Washington, Allen Institute for AI, Contextual AI, 2025 (EMNLP 2025)

## Summary

In January 2025, a team from Stanford, UW, and AI2 asked a simple question: what is the easiest way to build a reasoning model whose performance improves when you give it more thinking time at inference? OpenAI's o1 had demonstrated this "test-time scaling" behavior but shared no methodology. The s1 authors showed that just 1,000 carefully curated reasoning examples, supervised fine-tuning, and a decoding trick they call **budget forcing** are enough.

The recipe has two parts. First, they curate **s1K** — 1,000 questions with reasoning traces distilled from Gemini Thinking Experimental, selected via three criteria validated through ablations: difficulty (both Qwen2.5-7B and 32B must fail), diversity (spanning 50 mathematical domains via MSC classification), and quality (manual formatting checks). Second, they fine-tune Qwen2.5-32B-Instruct on s1K in just 26 minutes on 16 H100 GPUs (7 GPU-hours total), producing **s1-32B**.

Budget forcing controls test-time compute by forcibly extending or terminating the model's thinking. To extend: suppress the end-of-thinking token delimiter and append **"Wait"** to the model's reasoning trace, causing it to double-check and often fix incorrect steps. To terminate: append the end-of-thinking delimiter and "Final Answer:" to force an early exit. With budget forcing, s1-32B scales from 50% to 57% on AIME24 and exceeds o1-preview on MATH by up to 27%.

This notebook implements budget forcing from scratch on a small character-level transformer trained on synthetic arithmetic, demonstrating the "Wait" token's effect on response length and accuracy across different compute budgets.

## What Problem Does It Solve

Imagine taking a math test where the teacher says "you have 10 minutes." If you finish early, you can't just hand in the paper — the teacher forces you to keep working, saying "Wait, double-check your answer." Sometimes that extra checking helps you catch a mistake and fix it. Other times, if the teacher says "stop now, give me your best answer," you write down what you have.

This is exactly what budget forcing does for a language model. The model wants to stop thinking and give an answer, but the system says "Wait" and makes it keep reasoning. The model then re-examines its work, sometimes catching errors it would have submitted. The key insight is that you don't need complex reinforcement learning to get test-time scaling — a simple decoding-time intervention achieves it.

## Core Idea

- **1,000 samples suffice:** Fine-tuning on just 1,000 carefully selected reasoning traces (difficulty + diversity + quality) is enough to build a competitive reasoning model, echoing the "Superficial Alignment Hypothesis" from LIMA (Section 6.1).
- **Budget forcing:** A decoding-time intervention that controls test-time compute by suppressing the end-of-thinking token and appending "Wait" to extend thinking, or appending the delimiter to terminate early (Section 3.1).
- **Sequential over parallel scaling:** The authors argue that sequential scaling (longer reasoning chains) is more effective than parallel scaling (majority voting), because later computations can build on intermediate results (Section 3).

## Key Method Details

- **Dataset (s1K):** 1,000 examples from 59K initial pool, filtered by quality (formatting checks), difficulty (both Qwen 7B and 32B must fail), and diversity (50 MSC domains). Reasoning traces distilled from Gemini Flash Thinking API.
- **Training:** SFT on Qwen2.5-32B-Instruct, 26 minutes on 16 H100 GPUs (7 GPU-hours), basic hyperparameters.
- **Budget forcing:** Maximum token control — append end-of-thinking delimiter + "Final Answer:" for early exit. Minimum token control — suppress end-of-thinking delimiter, append "Wait" to extend reasoning.
- **Results:** s1-32B achieves 56.7% on AIME24 (with budget forcing), 93.0% on MATH500, 59.6% on GPQA Diamond. Budget forcing extrapolates AIME24 from 50% to 57%. Without budget forcing: 50.0% AIME24, 92.6% MATH500, 56.6% GPQA.
- **Ablation finding:** Random 1K selection scores ~30% worse on AIME24. Training on full 59K gives 53.3% AIME24 but uses 56x more GPU-hours. All three criteria (quality, difficulty, diversity) are necessary.

### Simplifications in This Notebook

- Tiny model: 2-layer, 6-head, 128-dim character-level transformer (~0.5M parameters) instead of Qwen2.5-32B (32B parameters)
- Dataset: ~2,000 synthetic single-digit and double-digit arithmetic problems with hand-templated reasoning traces instead of 1,000 competition-level problems with Gemini-distilled traces
- Budget forcing: applied to character-level generation with "Wait" appended as text tokens, instead of suppressing a specific end-of-thinking token delimiter in a full tokenizer
- Training: 15 epochs at lr=3e-4 on CPU, instead of 26 minutes on 16 H100 GPUs
- Evaluation: accuracy on 200 held-out arithmetic problems at different "Wait" budgets (0, 1, 2, 3), instead of AIME24/MATH500/GPQA Diamond

### What the Notebook Result Shows

On 200 held-out three-operand arithmetic problems (a+b-c), the model without "Wait" intervention achieves about 77% accuracy with an average response length of about 76 characters. With 1 "Wait" insertion, accuracy drops to about 22% while response length grows to about 151 characters. With 2-3 insertions, accuracy falls below 5% as response length reaches 220-250 characters.

The response length scales as expected — budget forcing successfully controls test-time compute (the paper's Control metric). However, accuracy degrades rather than improves, because the tiny 0.7M model enters repetitive loops when forced to continue generating — the failure mode the paper warns about (Section 6.2): "suppressing the end-of-thinking token delimiter too often can lead the model into repetitive loops." The paper's 32B model benefits from "Wait" because its pre-trained language understanding keeps generation coherent. These numbers are on toy arithmetic and are not comparable to the paper's AIME24 results.

## Influence

The s1 paper became one of the most influential o1-replication works due to its extreme simplicity and full openness (model, data, and code all released). It directly inspired follow-up work on test-time scaling analysis and budget forcing variants. The paper was accepted to EMNLP 2025. It is in the top 1% of cited papers by citation percentile, with about 1,400 citations ([OpenAlex](https://openalex.org/works/W4416036482), checked November 2026). The open-source release at [github.com/simplescaling/s1](https://github.com/simplescaling/s1) has been widely forked and used as a reference for minimal-data reasoning training.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/061_s1-simple-test-time-scaling/solution.ipynb) | From-scratch PyTorch character-level transformer trained on synthetic arithmetic, demonstrating budget forcing with "Wait" tokens at different compute budgets |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
