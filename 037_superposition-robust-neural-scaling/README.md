---
title: "037 Superposition Scaling"
nav_order: 38
permalink: "/037_superposition-robust-neural-scaling/"
---
# Superposition Yields Robust Neural Scaling (Superposition Scaling)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/superposition-yields-robust-neural-scaling)

- **Paper:** [Superposition Yields Robust Neural Scaling](https://arxiv.org/abs/2505.10465)
- **Authors:** Yizhou Liu, Ziming Liu, Jeff Gore
- **Lab, year:** Massachusetts Institute of Technology (Ziming Liu also affiliated with INRIA), 2025 (NeurIPS 2025, Best Paper Runner-up)

## Summary

Why do larger language models perform better, and why does their loss decrease as a smooth power law with model size? This paper proposes that **representation superposition** — the phenomenon where a model represents more features than it has dimensions, by packing feature vectors into a shared space with overlaps — is a key driver of neural scaling laws. The authors build on Anthropic's toy model of superposition (Elhage et al., 2022), an autoencoder that compresses high-dimensional sparse feature data into a lower-dimensional hidden space. By using weight decay (or growth) to control how many features the model represents, they can systematically study how loss scales with model width *m* across different regimes.

In the **weak superposition** regime (large weight decay), only the most frequent features are represented and the rest are ignored. Here, the loss follows a power law with model width only if the feature frequencies themselves follow a power law, and the scaling exponent depends on the data distribution. In the **strong superposition** regime (small or negative weight decay), all features are represented but their vectors overlap. Here, a geometric fact takes over: when many vectors are squeezed into a low-dimensional space, their mean-square overlap scales as 1/*m*, so the loss generically scales as 1/*m* — regardless of the data distribution.

The authors confirm that open-source LLMs (OPT, Pythia, Qwen2.5) operate in the strong superposition regime: their token representation overlaps scale as 1/*m*, and their evaluation loss scales with a model exponent α_m = 0.91 ± 0.04, close to the predicted 1. The Chinchilla scaling laws are also consistent with this 1/*m* behavior.

This repo's notebook trains 30 toy autoencoders (2 data exponents × 3 weight decays × 5 model widths), measures the loss-vs-width power-law exponent, and reproduces the qualitative finding that strong superposition yields α_m ≈ 1 while weak superposition yields α_m ≈ α.

## What Problem Does It Solve

Imagine you have a small apartment with only 3 shelves but 100 different items to store. You can either pick the 3 most important items and put one on each shelf (no superposition — no clutter but most items are lost), or try to cram all 100 items across the 3 shelves, accepting that they will overlap and be hard to tell apart (superposition — everything is there but with interference).

Before this paper, researchers knew that bigger AI models are better, and the loss decreases as a power law with model size. But *why* this smooth power law exists was unclear. Various theories required the data itself to follow a power law distribution. This paper shows that if models use superposition — which real LLMs do, since they must represent 50,000+ tokens in a few thousand dimensions — then the power law emerges naturally from the geometry of overlapping vectors, without needing any specific data distribution.

## Core Idea

- **Superposition as a scaling mechanism:** When models represent more features than dimensions (superposition), the geometric interference between overlapping representation vectors causes loss to scale as 1/m (inverse width), producing a power law with exponent ≈ 1.
- **Weight decay controls superposition:** Positive weight decay shrinks feature representations toward zero (weak superposition, only frequent features represented); negative weight decay pushes row norms toward 1 (strong superposition, all features represented with overlap).
- **Two regimes with different scaling:** Weak superposition gives α_m ≈ α (data-dependent, power law only if frequencies are power-law); strong superposition gives α_m ≈ 1 (universal, robust to data distribution).

## Key Method Details

- **Toy model:** Autoencoder y = ReLU(W W^T x + b), where W ∈ R^{n×m}, n=1000 features, m varies from 10 to 100. Input x_i = u_i · v_i, u_i ~ Bernoulli(p_i), v_i ~ U(0, 2). Feature frequencies p_i ∝ 1/i^{1+α}.
- **Custom optimizer:** AdamW with decoupled weight decay (γ ≥ 0: standard decay; γ < 0: growth toward unit row norms). Learning rate 0.01 with warmup (500 steps) + cosine decay, 5000 training steps, batch size 2048.
- **Key metric:** Model exponent α_m from fitting L ∝ 1/m^{α_m}; fraction of represented features φ_{1/2} = \|{i: \|\|W_i\|\|_2 > 0.5}\|/n.
- **Results:** Strong superposition (wd = -1): α_m = 0.98 (α=0) and 1.15 (α=1), mean 0.995. Weak superposition (wd = +1): α_m = 0.13 (α=0) and 0.63 (α=1), mean 0.382.

### Simplifications in This Notebook

- Smaller model: n=1000 features, m ∈ [10, 20, 40, 60, 100] (paper uses n=10240 and m up to 1024 for large models)
- 5000 training steps per model (paper uses 20000 for small models, 40000 for large)
- 2 data exponents (α=0 and α=1) instead of the paper's full sweep of 9 values from 0 to 1
- 3 weight decay values: -1, 0, +1 (paper uses 10 values)
- MSE loss with synthetic data instead of cross-entropy on real text (paper shows this does not affect scaling behavior, Appendix A.2)
- No LLM evaluation (paper Section 3.3 analyzes OPT, Pythia, Qwen2.5 models)

### What the Notebook Result Shows

On the toy autoencoder with n=1000 features and 5 model widths from m=10 to m=100, strong superposition (weight decay = -1, φ ≈ 1) produces model exponents α_m = 0.977 for flat data (α=0) and α_m = 1.146 for Zipf data (α=1), both close to 1 — consistent with the paper's prediction of universal 1/m scaling from geometric overlaps. The mean α_m across all strong-superposition settings is 0.995 (range 0.657–1.200).

Weak superposition (weight decay = +1, φ < 0.03) produces α_m = 0.133 for flat data (α=0) and α_m = 0.631 for Zipf data (α=1), much smaller and clearly data-dependent — matching the paper's prediction. The fraction of represented features φ transitions from ≈1 (strong superposition) to near 0 (weak superposition) as weight decay goes from -1 to +1, reproducing Figure 3b of the paper.

Exact α_m values vary slightly between runs and hardware (CPU vs GPU, GitHub vs Kaggle); the qualitative trend — strong superposition gives more universal exponents near 1, weak superposition gives smaller data-dependent exponents — is robust.

## Influence

This paper was awarded Best Paper Runner-up (Oral) at NeurIPS 2025, one of the most selective venues in machine learning. It bridges the mechanistic interpretability community (Anthropic's superposition work, Elhage et al., 2022) with the scaling laws community (Kaplan et al., 2020; Hoffmann et al., 2022), proposing a concrete mechanistic explanation for why neural scaling laws exist. It influenced subsequent discussions about how to improve scaling laws (e.g., by encouraging superposition through architectures like nGPT) and when scaling laws might break down (when model dimension approaches vocabulary size). The paper has about 1 citation on [OpenAlex](https://openalex.org/works/W4416635575) (checked November 2026), reflecting its recent publication.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/037_superposition-robust-neural-scaling/solution.ipynb) | From-scratch PyTorch toy autoencoder trained on synthetic sparse features, sweeping weight decay and model width to measure the loss-vs-width power-law exponent |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
