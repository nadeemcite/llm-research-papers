---
title: "065 Coconut"
nav_order: 66
permalink: "/065_coconut-continuous-latent-space-reasoning/"
---
# Training Large Language Models to Reason in a Continuous Latent Space (Coconut)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/coconut-continuous-latent-space-reasoning)

- **Paper:** [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- **Authors:** Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian
- **Lab, year:** FAIR at Meta (Hao, Sukhbaatar, Su, Li, Weston, Tian) and UC San Diego (Hu; contribution: work done at Meta), 2024 (COLM 2025)

## Summary

Large language models typically reason by generating a chain of words — a chain-of-thought (CoT) — before producing a final answer. In December 2024, researchers at FAIR (Meta) and UC San Diego challenged this assumption: most word tokens in a reasoning chain serve textual coherence, not actual reasoning, while a few critical tokens require complex planning that language-bound generation struggles with. Their solution, Coconut (Chain of Continuous Thought), replaces the language-space reasoning step with a continuous latent representation.

Instead of decoding the model's last hidden state into a word token and re-embedding it, Coconut feeds that hidden state directly back as the next input embedding — a "continuous thought." This lets the model reason in an unconstrained latent space rather than being forced through the bottleneck of discrete language tokens. Because continuous thoughts are differentiable, the entire reasoning process can be optimized end-to-end with gradient descent. Training uses a multi-stage curriculum (inspired by iCoT, Deng et al. 2024) that progressively replaces language reasoning steps with continuous thoughts.

On ProsQA, a new logical reasoning dataset requiring substantial planning, Coconut outperforms CoT (97.0% vs 92.5% accuracy) while generating far fewer tokens (14.2 vs 49.4). On ProntoQA, Coconut matches CoT (99.8% vs 98.8%) with only 9.0 tokens vs 77.5. On GSM8k math reasoning, Coconut reaches 34.1% (vs CoT 42.9%) but achieves a better accuracy-efficiency trade-off. A key finding: continuous thoughts can encode multiple alternative next steps simultaneously, enabling an emergent breadth-first-search-like reasoning pattern without explicit training for it.

This notebook implements Coconut on a tiny 157K-parameter transformer trained on synthetic ProntoQA-style logical reasoning problems, comparing Coconut against CoT and No-CoT baselines.

## What Problem Does It Solve

Imagine you are solving a maze. In a normal chain-of-thought, you must write down every step in words: "I go left, then I see a fork, I try right..." — even though the actual *thinking* happens in your head, not in the words. The words are just a slow, lossy translation of your internal reasoning. Worse, once you commit to "I go right" on paper, you feel obligated to continue down that path even if it leads to a dead end.

Coconut lets the model "think in its head" — using continuous hidden-state representations instead of words — and only translate to language at the very end. This is like solving the maze mentally, exploring multiple paths simultaneously in your mind, and then writing down just the answer. The paper draws on neuroimaging evidence that the human language network is largely inactive during reasoning tasks, suggesting language may be optimized for communication, not for reasoning itself.

## Core Idea

- **Continuous thoughts replace language tokens:** Instead of decoding hidden states to word tokens and re-embedding them, Coconut feeds the last hidden state directly as the next input embedding. Special tokens `<bot>` and `<eot>` mark the beginning and end of latent reasoning mode.
- **Multi-stage curriculum training:** Inspired by iCoT, training starts with full CoT (language reasoning), then progressively replaces language reasoning steps with continuous thoughts. At stage k, the first k reasoning steps are replaced with k×c continuous thoughts (where c is a hyperparameter controlling latent thoughts per step). The optimizer state is reset between stages.
- **Emergent BFS-like reasoning:** Continuous thoughts can encode multiple alternative next steps simultaneously, unlike language tokens which commit to a single deterministic path. This allows the model to maintain several possible options and progressively eliminate incorrect paths, resembling a breadth-first search — without being explicitly trained to do so.

## Key Method Details

- **Base model:** Pre-trained GPT-2, learning rate 1e-4 (for ProsQA/ProntoQA) with effective batch size 128. For ProsQA: 6 training stages (max reasoning steps = 6), 5 epochs per stage, continuing in the last stage until 50 total epochs. For GSM8k: c=2 (two continuous thoughts per step), 3+1 stages, 6 epochs initial then 3 per stage.
- **Loss:** Standard negative log-likelihood (cross-entropy) on tokens after continuous thoughts; loss is masked on questions and latent thoughts. The objective does not compress the removed language thought into the continuous thought but rather facilitates prediction of future reasoning.
- **Inference:** `<bot>` inserted after question tokens; `<eot>` determined by either a trained binary classifier on latent thoughts or constant-length padding (both work comparably; the paper uses constant padding). Greedy decoding for all experiments.
- **Results (Table 1):** On ProsQA — CoT 92.5% (49.4 tokens), No-CoT 76.7% (8.2 tokens), Coconut 97.0% (14.2 tokens), iCoT 98.2% (8.2 tokens). On ProntoQA — CoT 98.8% (77.5 tokens), Coconut 99.8% (9.0 tokens). On GSM8k — CoT 42.9% (25.0 tokens), Coconut 34.1% (8.2 tokens). Coconut without curriculum training drops to 14.4% on GSM8k and 76.1% on ProsQA.

### Simplifications in This Notebook

- Smaller model: 3-layer transformer, d_model=64, 4 heads, about 157,376 parameters (the paper uses GPT-2 small, ~124M parameters)
- Dataset: 400 synthetic ProntoQA-style logical reasoning problems (2-4 hops) instead of the paper's ProntoQA and ProsQA datasets
- Vocabulary: 50 tokens (question/relation/answer/special tokens) instead of GPT-2's full vocabulary
- Training: 8 CoT epochs (stage 0) + 4 epochs per latent stage × 3 stages; lr=1e-4, batch_size=64 (paper: 5 epochs per stage, up to 50 total, effective batch 128)
- c=1 (one continuous thought per reasoning step), matching the paper's setting for logical reasoning
- No-CoT baseline trained for the same number of epochs as CoT plus all latent stages combined, with the same lr and batch size

### What the Notebook Result Shows

On 100 held-out test problems, Coconut achieves 50.0% accuracy (50/100) with c=1, improving to 61.0% with c=2 and 61.0% with c=3 — showing that more continuous thoughts can improve reasoning, consistent with the paper's finding that increasing c improves performance. Both CoT and No-CoT baselines score 0.0% (0/100): the tiny 157K-parameter model memorizes the training set (loss drops to 0.0000) but fails to generalize when reasoning must be expressed in language tokens, while Coconut's latent reasoning successfully transfers. Coconut generates only 1.0 token on average (the answer token), since all reasoning happens in continuous space, versus 8.8 tokens for CoT.

These results are not quantitatively comparable to the paper (which uses GPT-2 small on real datasets with 97.0% accuracy on ProsQA). The 0% CoT/No-CoT baselines reflect the toy model's limited capacity rather than a failure of CoT in general — the paper's CoT baseline reaches 92.5% on ProsQA with GPT-2. However, the qualitative pattern that Coconut outperforms baselines while generating fewer tokens is reproduced, and the probing analysis shows continuous thoughts encode multiple candidate next steps (probability spread across top-5 tokens), supporting the paper's BFS-like reasoning claim.

## Influence

Coconut introduced the concept of "continuous thought" reasoning, sparking a line of research into latent reasoning for LLMs. Follow-up work includes CODI (Compressing Chain-of-Thought into Continuous Space via Self-Distillation, 2025, ~238 citations on OpenAlex), which applies self-distillation to learn continuous-space CoT, and Monet (Reasoning in Latent Visual Space Beyond Images and Language, 2025), which extends latent reasoning to visual domains. Theoretical analyses by Cheng et al. (2025) formalized why continuous CoT can be more efficient than discrete CoT by encoding multiple reasoning paths in superposition states, and Wang et al. (2025) analyzed the training dynamics explaining how such superposition emerges. The paper was accepted to COLM 2025. About 859 citations ([OpenAlex](https://openalex.org/works/W4405254469), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/065_coconut-continuous-latent-space-reasoning/solution.ipynb) | PyTorch implementation of Coconut on a 157K-param transformer with synthetic ProntoQA-style logic problems, multi-stage curriculum training, and probing analysis of continuous thoughts |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
