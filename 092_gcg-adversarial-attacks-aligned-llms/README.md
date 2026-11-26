---
title: "092 GCG Attack"
nav_order: 93
permalink: "/092_gcg-adversarial-attacks-aligned-llms/"
---
# Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/gcg-adversarial-attacks-aligned-llms)

- **Paper:** [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)
- **Authors:** Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson
- **Lab, year:** Carnegie Mellon University, Center for AI Safety, Google DeepMind, Bosch Center for AI, 2023

## Summary

Aligned language models (like ChatGPT, Claude, and Bard) are trained to refuse harmful requests. But this alignment can be broken. The GCG (Greedy Coordinate Gradient) paper shows that a fully automatic method can find an adversarial suffix — a string of seemingly nonsensical tokens — that, when appended to a harmful query, causes aligned models to comply instead of refusing. Unlike manual "jailbreaks" that require human ingenuity and are quickly patched, GCG uses gradient-based optimization to discover these suffixes automatically.

The method works by defining a loss function that rewards the model for beginning its response with "Sure, here is..." (an affirmative response), then optimizing a suffix of about 20 tokens to minimize that loss. The optimization is done over discrete tokens (not continuous embeddings), using gradients to identify promising replacement candidates at each token position, then evaluating candidates exactly via forward passes. The key innovation over prior work (AutoPrompt) is searching over all token positions simultaneously rather than one at a time.

Surprisingly, the adversarial suffixes transfer across models: a suffix optimized on Vicuna-7B and 13B also jailbreaks GPT-3.5 (up to 84%), GPT-4 (up to 47%), PaLM-2 (66%), and Claude (2.1%). The authors responsibly disclosed their findings to OpenAI, Google, Meta, and Anthropic before publication. The paper introduced AdvBench, a benchmark of 500 harmful strings and 500 harmful behaviors for evaluating attacks.

This notebook implements the GCG algorithm (Algorithm 1) on GPT-2 (124M) with a benign target string, demonstrating the gradient-based discrete token optimization in about 200 steps.

## What Problem Does It Solve

Imagine a bouncer at a club who checks IDs and turns away anyone who looks suspicious. Manual jailbreaks are like finding a clever disguise — one person figures it out, but the bouncer quickly learns to spot that trick too. GCG is like having a computer systematically test thousands of disguises per second, using the building's blueprints (the model's gradients) to find one that consistently slips past the bouncer. The paper solves the problem of *automating* jailbreak discovery: instead of relying on human creativity, GCG finds adversarial suffixes by gradient descent on the model's own internal signals.

## Core Idea

- **Affirmative response target:** Instead of trying to make the model generate a specific harmful string, the attack targets a short affirmative prefix ("Sure, here is [query]") — once the model starts complying, it tends to continue.
- **Greedy Coordinate Gradient (GCG):** For each adversarial token position, compute the gradient of the loss with respect to the one-hot token embedding, use it to find the top-k most promising replacement candidates, evaluate a random batch of candidates via forward passes, and keep the best one. The critical difference from AutoPrompt is searching all positions simultaneously.
- **Multi-prompt multi-model optimization:** To create universal, transferable attacks, the loss is aggregated across multiple harmful prompts and multiple models (Vicuna-7B and 13B), with prompts added incrementally (Algorithm 2).

## Key Method Details

- **Adversarial loss (Eq. 3):** L(x_{1:n}) = -log p(x*_{n+1:n+H} \| x_{1:n}), the negative log-probability of the target string given the full input (prompt + suffix).
- **Algorithm 1 (GCG):** For T iterations: compute top-k gradient-based candidates for all positions, sample B random (position, token) pairs, evaluate loss exactly, keep the best replacement.
- **Algorithm 2 (Universal):** Aggregate gradients and losses across multiple prompts/models; add prompts incrementally only after the current set succeeds.
- **Results (Table 1):** GCG achieves 88% ASR on Vicuna-7B and 57% on LLaMA-2-7B-Chat for harmful strings (vs. 25% and 3% for AutoPrompt). For harmful behaviors: 88% on LLaMA-2 test set (vs. 35% for AutoPrompt).
- **Transfer (Table 2):** GCG ensemble achieves 86.6% ASR on GPT-3.5, 46.9% on GPT-4, 66% on PaLM-2, 2.1% on Claude-2.

### Simplifications in This Notebook

- Smaller model: GPT-2 (124M), about 124M parameters (instead of Vicuna-7B at ~7B)
- Single prompt, single model (Algorithm 1 only, not Algorithm 2)
- 20 adversarial tokens, 200 GCG steps, batch size 64, top-k 64 (instead of 20 tokens, 500 steps, batch 512, top-k 256)
- Benign target string ("Sure, here is how to make a paper airplane") instead of harmful content
- No transferability testing across models

### What the Notebook Result Shows

On GPT-2 (124M), the GCG attack reduces the negative log-likelihood of the target string "Sure, here is how to make a paper airplane" from 2.99 (without suffix) to 0.87 (with suffix) over 200 optimization steps, increasing the target probability by about 8.3x. The loss curve shows a steady decline from 3.32 to 0.87, demonstrating that gradient-based discrete search effectively steers the model toward the target. This is not comparable to the paper's 88% ASR on Vicuna-7B because GPT-2 is not safety-aligned, the target is benign, and the optimization budget is 4x smaller.

## Influence

GCG became one of the most influential AI safety papers of 2023, sparking a wave of research on adversarial robustness of LLMs. Follow-up work includes multi-turn GCG attacks (Chao et al., 2023), defenses based on smoothing and paraphrasing (Robey et al., 2023), and the `nanogcg` library (2024) for fast GCG implementation. The paper's AdvBench benchmark is widely used in LLM safety evaluation. About 4,979 citations ([OpenAlex](https://openalex.org/works/W4385374425), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/092_gcg-adversarial-attacks-aligned-llms/solution.ipynb) | PyTorch GCG attack on pretrained GPT-2 (124M) with gradient-based discrete token optimization (loss reduction 2.99 to 0.87) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
