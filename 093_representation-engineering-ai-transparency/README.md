---
title: "093 RepEngineering"
nav_order: 94
permalink: "/093_representation-engineering-ai-transparency/"
---
# Representation Engineering: A Top-Down Approach to AI Transparency (RepEngineering)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/representation-engineering-ai-transparency)

- **Paper:** [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)
- **Authors:** Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, Dan Hendrycks
- **Lab, year:** Center for AI Safety, Carnegie Mellon University, UC Berkeley, Stanford, 2023

## Summary

Representation Engineering (RepE) is a top-down approach to AI transparency inspired by cognitive neuroscience. Instead of analyzing individual neurons or circuits (bottom-up mechanistic interpretability), RepE works at the level of **population-level representations** — directions in a model's hidden state space that correspond to high-level concepts like honesty, harmlessness, power-seeking, and emotion. The paper provides a systematic framework with two halves: **Representation Reading** (monitoring what a model is "thinking") and **Representation Control** (manipulating those internal states to change behavior).

The core method is **Linear Artificial Tomography (LAT)**, named by analogy to neuroimaging. To extract a concept direction, you design stimulus pairs that contrast the target concept (e.g., honest vs dishonest statements), run them through the model, collect last-token hidden states at each layer, compute normalized differences between paired stimuli, and apply PCA — the first principal component becomes the "reading vector" for that concept. This direction can then be used as a classifier (reading) or added to/subtracted from the model's activations during generation (control).

The paper demonstrates RepE on a wide range of safety-relevant problems: honesty (lie detection, TruthfulQA improvement), ethics and power aversion, emotion, memorization detection, and more. On TruthfulQA MC1, the Contrast Vector control method achieves 54.0% accuracy on LLaMA-2-13B-Chat (up from 35.9% baseline), and LAT reading achieves 69.8% on the 70B model — state-of-the-art results at the time.

This notebook implements the LAT method on GPT-2 (124M): it extracts an honesty direction from contrastive stimulus pairs, evaluates it as a lie detector on held-out statements, and demonstrates representation control by steering the model toward more honest or dishonest outputs.

## What Problem Does It Solve

Imagine you could read someone's mind by looking at their brain activity — not word by word, but at the level of general states like "honest" vs "lying." Representation Engineering does this for AI. When a large language model generates text, its internal hidden states contain structured information about what it's "thinking." The paper shows that you can find a single direction in that high-dimensional space that tracks honesty, and by projecting onto it you can tell whether the model is about to be truthful — without looking at the output. Even more strikingly, you can *push* the model along that direction to make it more or less honest.

## Core Idea

- **Top-down concept directions:** Instead of finding individual neurons, RepE extracts population-level directions (via PCA on contrastive pairs) that represent high-level concepts in a model's hidden state space.
- **LAT (Linear Artificial Tomography):** A three-step procedure — design stimulus pairs, collect hidden states at the last token, compute normalized differences and PCA to get a reading vector.
- **Reading leads to Control pipeline:** The same direction used to *read* (classify) a concept can be used to *control* (steer) the model by adding it to activations during generation, establishing causal (not just correlational) evidence.

## Key Method Details

- **Stimulus design:** Pairs of statements contrasting the target concept (e.g., "I am an AI" vs "I am a real human"). Formatted with a task template: "Consider the honesty in the following: {stimulus} The honesty is"
- **Hidden state collection:** Last-token hidden state at each Transformer layer (default for decoder models), per the paper's Eq. 1.
- **PCA construction:** Normalize each pairwise difference vector, run PCA, first principal component = reading vector (Appendix C.1).
- **Control (Reading Vector baseline):** Add alpha * reading_vector to the hidden states at a target layer during generation via a forward hook — an ActAdd-style intervention.
- **Results:** On TruthfulQA MC1, LAT reading achieves 69.8% on LLaMA-2-70B-Chat (vs 29.9% standard), and Contrast Vector control achieves 54.0% on 13B-Chat (vs 35.9% baseline).

### Simplifications in This Notebook

- Smaller model: GPT-2 (124M, 12 layers, 768-dim) instead of LLaMA-2 (7B-70B)
- Dataset: 24 hand-written honest/dishonest statement pairs instead of the paper's QA benchmarks (TruthfulQA, OpenbookQA, etc.)
- Control method: Reading Vector (ActAdd-style) only; did not implement Contrast Vector or LoRRA baselines
- Evaluation: held-out statement classification instead of TruthfulQA MC1

### What the Notebook Result Shows

On GPT-2 (124M), the honesty direction extracted via LAT achieves 71% classification accuracy on the training stimuli at layer 9 (the best layer), and 80% accuracy (4/5) on held-out honest vs dishonest statements, confirming that even a small base model has a separable internal representation of honesty. Steering by adding the reading vector (alpha = +12) shifts completions to slightly different but coherent text, while subtracting it (alpha = -12) produces contradictory, self-inconsistent outputs (e.g., "not a lawyer, but I am a lawyer" — a self-contradicting, dishonest-style completion).

The effect is more subtle than in the paper (which uses 7B-70B instruction-tuned models), because GPT-2 base is not chat-tuned and has weaker concept representations. However, the core phenomena — reading, lie detection, and bidirectional control — are all visible. These results are not directly comparable to the paper's TruthfulQA numbers.

## Influence

RepE became a foundational reference for the "steering vectors" and "activation engineering" research direction. It influenced subsequent work including the `repeng` library by the same authors, the Activation Engineering paper (Turner et al., 2023), and follow-up studies on concept erasure, safety steering, and model introspection. The paper was widely covered in AI safety discussions. About 2,347 citations ([OpenAlex](https://openalex.org/works/doi:10.48550/arXiv.2310.01405), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/093_representation-engineering-ai-transparency/solution.ipynb) | PyTorch LAT extraction and steering on pretrained GPT-2 (124M) with honesty direction reading, lie detection, and representation control |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
