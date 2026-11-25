---
title: "090 Monosemanticity"
nav_order: 91
permalink: "/090_towards-monosemanticity-dictionary-learning/"
---
# Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (Monosemanticity)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/towards-monosemanticity-dictionary-learning)

- **Paper:** [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features)
- **Authors:** Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nicholas L. Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E. Burke, Tristan Hume, Shan Carter, Tom Henighan, Chris Olah
- **Lab, year:** Anthropic, 2023 (Transformer Circuits Thread, October 2023)

## Summary

Many neurons in neural networks are *polysemantic* — a single neuron fires for multiple unrelated concepts, making the network hard to understand. One neuron might respond to both cat faces and car fronts (as seen in the vision model Inception v1), or to a mixture of academic citations, English dialogue, HTTP requests, and Korean text (as seen in the small language model studied here). The paper's authors had previously hypothesised that this polysemanticity arises from *superposition*: networks represent more features than they have neurons by packing features into nearly orthogonal directions, exploiting the sparsity (rarity) of individual features.

This paper provides the first compelling demonstration that a **sparse autoencoder (SAE)** — a simple encoder-decoder network with an L1 sparsity penalty — can reverse superposition and extract interpretable, *monosemantic* features from a trained transformer. They train SAEs on the MLP activations of a one-layer transformer (512 neurons), with expansion factors from 1× (512 features) to 256× (131,072 features). Their primary analysis focuses on the A/1 run with 4,096 features (8× expansion).

They show four lines of evidence that SAE features are interpretable: detailed case studies (Arabic script, DNA, base64, Hebrew), human annotation, automated interpretability with Claude, and logit-weight analysis (logits are the raw, pre-softmax scores a model assigns to each possible next token; examining which tokens a feature makes more or less likely). SAE features are more interpretable than neurons, effectively invisible in the neuron basis, useful for steering generation, and largely universal across models.

This repo's notebook trains a sparse autoencoder on GPT-2 small's first MLP layer (768 neurons, 2,048 SAE features), shows that features are sparser and more specific than neurons, and labels top features by their activating tokens.

## What Problem Does It Solve

Imagine you walk into a busy kitchen where every cook is responsible for multiple unrelated dishes — the pasta chef also grills steaks and bakes desserts. If you want to know why the kitchen is producing a certain smell, you can't just watch one cook; each one is doing three things at once. You'd much prefer a kitchen where each cook handles only one dish — then watching any cook tells you exactly what's being made.

Neural networks have the "multi-task cook" problem: their neurons are polysemantic, each responding to several unrelated things. This paper asks: can we *decompose* these overworked neurons into simpler, single-purpose "features" — even though the network never explicitly created them? The answer is yes: a sparse autoencoder can unpack the tangled neurons and recover features that each respond to one coherent concept, like "Arabic script" or "DNA sequences," even when no individual neuron does so.

## Core Idea

- **Sparse autoencoder as dictionary learning:** Train an encoder (ReLU + linear) and decoder on MLP activations. The encoder maps 512-dim activations to a much larger set of features (up to 131,072), and the L1 penalty ensures only a few features fire per token — recovering the sparse, overcomplete basis that superposition compressed away.
- **Features are more monosemantic than neurons:** The paper provides four independent lines of evidence — detailed case studies, human scoring, automated interpretability with Claude, and logit-weight analysis — all showing that SAE features are more interpretable than individual neurons.
- **Features are not neurons:** Key features (Arabic script, Hebrew, DNA, base64) are "effectively invisible" in the neuron basis — no single neuron corresponds to them. The features are distributed across many neurons.
- **Feature splitting:** As the SAE width increases, features split into families. One base64 feature at 512 features becomes three at 4,096, each with a more specific role (letters vs. digits vs. ASCII-encoding base64).
- **Universality:** SAEs trained on different transformers (different random seeds, same architecture) find mostly similar features, with a median activation correlation of 0.72 between matched features.
- **Steering:** Artificially activating a feature (e.g., base64) causes the model to generate text matching that feature's interpretation.

## Key Method Details

- **Model studied:** One-layer transformer with a 512-neuron MLP layer (ReLU activation), trained on the Pile (100 billion tokens)
- **SAE architecture:** Input bias $b_d$, encoder $f(x) = \text{ReLU}(W_e(x - b_d) + b_e)$, decoder $x' = W_d f(x) + b_d$. Input/output biases are tied.
- **Loss:** MSE reconstruction + L1 penalty on hidden activations: $L = \text{MSE}(x, x') + \lambda \sum_i \|f_i(x)\|_1$
- **Training data:** 8 billion activation samples from the transformer's MLP layer
- **Expansion factors:** 1× (512) to 256× (131,072); primary analysis on A/1 with 4,096 features (8×)
- **Dead neuron resampling:** Periodically reset encoder weights on features that never fire, pointing them at data points the SAE currently reconstructs poorly
- **Key results:** 79% of MLP loss recovered by A/1 (4,096 features); up to 94.5% with A/5 (131,072 features). Median human rubric score: features 12, neurons 0. Automated interpretability: 74% logit-weight accuracy for features vs. 58% for neurons.

### Simplifications in This Notebook

- GPT-2 small (12 layers, 768-dim MLP) instead of a custom one-layer transformer (512-dim MLP)
- First MLP layer only; ~20K activation samples instead of 8 billion
- 2,048 SAE features (≈2.7× expansion) instead of up to 131,072 (256×)
- 3,000 training steps with Adam (lr 1e-3, cosine decay) and L1 coef 0.01 instead of the paper's extensive training
- Simplified dead neuron resampling (every 500 steps)
- Token-entropy proxy for interpretability instead of human annotators or Claude-based automated interpretability
- Diverse synthetic text corpus (English, code, HTML, JSON, base64, URLs, math) instead of the Pile

### What the Notebook Result Shows

The SAE with 2,048 features trained on GPT-2 small MLP activations (17,402 samples from a diverse text corpus) achieves a variance-explained of about 88%, with an average L0 of about 7.9 active features per sample — far fewer than the 2,048 total, confirming the sparse, overcomplete decomposition the paper describes. Of the 2,048 features, 460 are active (1,588 are dead, a high ratio expected from the small dataset).

SAE features have a median activation-weighted token entropy of about 1.88, compared to about 5.95 for raw neurons — features fire on a much narrower set of tokens, supporting the paper's central claim that SAE features are more interpretable. The top features by maximum activation fire on coherent token groups: `href` (HTML links), `std` and `rand` (C++ standard library and random functions), `pand` (pandas), `learn`, `sports`, `Gutenberg`, `derivative`.

These results are qualitatively consistent with the paper but not directly comparable: the paper uses a one-layer transformer, 8 billion samples, human annotators, and Claude for automated interpretability, while this notebook uses GPT-2 small, ~17K samples, and a token-entropy proxy.

## Influence

This paper became a cornerstone of mechanistic interpretability research. It directly motivated *Scaling Monosemanticity* (Templeton et al., 2024), which applied sparse autoencoders to Claude 3 Sonnet and extracted millions of interpretable features from a production-scale model. The sparse autoencoder approach has been adopted by interpretability researchers at Google DeepMind, OpenAI, and EleutherAI, and has spawned a rapidly growing subfield with improved SAE variants (TopK SAEs, JumpReLU SAEs, batchTopK SAEs).

Neel Nanda replicated and extended the results in an open-source tutorial, and the approach is now a standard tool in interpretability research. The paper is not indexed on OpenAlex or Semantic Scholar as of November 2026 (it was published on transformer-circuits.pub rather than arXiv), so a formal citation count is unavailable; however, it is one of the most widely referenced works in the mechanistic interpretability community.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/090_towards-monosemanticity-dictionary-learning/solution.ipynb) | PyTorch sparse autoencoder trained on GPT-2 small MLP activations: 2048 features from 768 neurons, L1 sparsity, dead neuron resampling, feature labeling and entropy-based interpretability comparison |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
