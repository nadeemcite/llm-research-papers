---
title: "034 Scaling Laws"
nav_order: 35
permalink: "/034_scaling-laws-neural-language-models/"
---
# Scaling Laws for Neural Language Models (Scaling Laws)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/scaling-laws-neural-language-models)

- **Paper:** [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- **Authors:** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
- **Lab, year:** OpenAI (Jared Kaplan at Johns Hopkins University), 2020 (arXiv preprint, 23 January 2020)

## Summary

How does the performance of a language model change as you make it bigger, feed it more data, or spend more compute training it? Kaplan et al. trained Transformer language models ranging from 768 to 1.5 billion non-embedding parameters on WebText2 (a 22.9-billion-token dataset of upvoted Reddit-linked web pages) and found that the test loss (cross-entropy in nats — a measure of how surprised the model is by the next token, lower is better) follows a **power law** in each of three scale factors: model size *N*, dataset size *D*, and compute *C*. The trends span more than seven orders of magnitude with no sign of deviation on the upper end.

The headline equations are: *L*(*N*) = (*N*\_c/*N*)^α\_N with α\_N ≈ 0.076 and *N*\_c ≈ 8.8×10¹³; *L*(*D*) = (*D*\_c/*D*)^α\_D with α\_D ≈ 0.095 and *D*\_c ≈ 5.4×10¹³; and *L*(*C*\_min) = (*C*\_c^min/*C*\_min)^α\_C^min with α\_C^min ≈ 0.050. A single equation *L*(*N*,*D*) = [(*N*\_c/*N*)^(α\_N/α\_D) + *D*\_c/*D*]^α\_D governs overfitting when both *N* and *D* are finite. Crucially, architectural shape (depth vs. width, number of heads) has minimal effect within a wide range — only the total non-embedding parameter count matters.

The most influential practical finding: given a fixed compute budget, most of it should go to **larger models**, not longer training or more data. The optimal allocation is *N* ∝ *C*^0.73, *B* ∝ *C*^0.24, *S* ∝ *C*^0.03 — meaning as compute grows, you should scale model size aggressively while barely increasing training steps. Larger models are also more sample-efficient, reaching the same loss with fewer data points.

This repo's notebook trains 6 tiny Transformers (~25K to ~793K parameters) on character-level WikiText-2, fits the power law *L*(*N*), and reproduces the overfitting and compute-scaling trends qualitatively.

## What Problem Does It Solve

Imagine you are planning a road trip and have a fixed fuel budget. Should you drive a small car a very long distance, or a big truck a shorter distance? You want to get as far as possible, so you need to know how distance scales with fuel for each vehicle type.

Before this paper, AI researchers knew that bigger models trained on more data performed better, but the *relationship* was guesswork. If you doubled your compute budget, should you double the model size, the data, or the training time? Kaplan et al. turned this intuition into precise, predictive equations. They showed that loss decreases as a smooth power law with each factor — so you can predict how good a 10-billion-parameter model will be by measuring models 10,000× smaller, then extrapolating. This is like test-driving a toy car to predict how far a real truck will go on the same fuel.

## Core Idea

- **Power-law scaling (Section 3):** The test loss *L* scales as a power law with non-embedding parameter count *N*, dataset size *D*, and compute *C*, each independently when the other two are not bottlenecked. The exponents are small (α\_N ≈ 0.076), meaning you need exponential increases in scale for linear improvements in loss — but the relationship is remarkably smooth.
- **Shape independence (Section 3.1):** When total non-embedding parameters *N* is held fixed, performance depends very weakly on depth, width, or number of attention heads. An aspect ratio change of 40× shifts the loss by only a few percent. The embedding parameters are excluded because they can be shrunk without impacting performance.
- **Compute-optimal allocation (Section 6):** For a fixed compute budget, optimal performance comes from training very large models on modest data and stopping well before convergence. The allocation *N* ∝ *C*^0.73 means most additional compute should buy bigger models, not more training — a counterintuitive result that shaped how labs allocate GPU budgets.

## Key Method Details

- **Loss metric (Section 2):** *L* is the cross-entropy loss in nats, averaged over a 1024-token context. This is the autoregressive log-likelihood of the next token.
- **Parameter count convention (Section 2, Table 1):** *N* counts only non-embedding parameters. For a Transformer, *N* ≈ 12 *n*\_layer *d*\_model² (excluding embeddings). This is crucial because embedding parameters (vocab × d\_model) can dominate total counts for small models but don't affect performance proportionally.
- **Compute estimate (Section 2):** *C* ≈ 6*NBS*, where *B* is batch size (in tokens) and *S* is the number of parameter updates (steps). The factor 6 accounts for forward + backward passes. Values are quoted in PF-days (1 PF-day = 8.64×10¹⁹ FLOPs).
- **Training setup (Section 2):** Adam optimizer, 2.5×10⁵ steps, batch size 512 sequences of 1024 tokens, 3000-step linear warmup followed by cosine decay to zero. WebText2 dataset: 20.3M documents, 2.29×10¹⁰ tokens, vocab size 50,257 (BPE).
- **L(N) power law (Eq. 1.1):** *L*(*N*) = (*N*\_c/*N*)^α\_N, α\_N ≈ 0.076, *N*\_c ≈ 8.8×10¹³. Doubling parameters yields a loss reduction of 2^−0.076 ≈ 0.95 (a 5% improvement).
- **L(D) power law (Eq. 1.2):** *L*(*D*) = (*D*\_c/*D*)^α\_D, α\_D ≈ 0.095, *D*\_c ≈ 5.4×10¹³ tokens.
- **L(N,D) overfitting (Eq. 1.5):** *L*(*N*,*D*) = [(*N*\_c/*N*)^(α\_N/α\_D) + *D*\_c/*D*]^α\_D. Overfitting depends on the ratio *N*^(α\_N/α\_D)/*D*; data should grow as *D* ∝ *N*^(α\_N/α\_D) ≈ *N*^0.74.
- **L(C\_min) power law (Eq. 1.3):** *L*(*C*\_min) = (*C*\_c^min/*C*\_min)^α\_C^min, α\_C^min ≈ 0.050, *C*\_c^min ≈ 3.1×10⁸ PF-days.
- **Critical batch size (Eq. 1.4):** *B*\_crit(*L*) = *B*\*/*L*^(1/α\_B), *B*\* ≈ 2×10⁸ tokens, α\_B ≈ 0.21. Training at the critical batch size is the optimal time/efficiency tradeoff.
- **Optimal compute allocation (Section 6):** *N* ∝ *C*\_min^0.73, *B* ∝ *C*\_min^0.24, *S* ∝ *C*\_min^0.03, *D* ∝ *C*^0.27. Most compute should buy larger models, not more training.

### Simplifications in This Notebook

- 6 tiny models (25,472 to 793,344 non-embedding parameters) instead of 768 to 1.5 billion parameters
- Character-level WikiText-2 (~10.8M characters, 283-char vocab) instead of WebText2 (22.9B BPE tokens, 50,257 vocab)
- 2000 steps with Adam (lr 3e-3, cosine decay) at batch 64×128 chars instead of 250K steps at 512×1024 tokens
- Context length 128 characters instead of 1024 tokens
- Pure PyTorch with nn.TransformerEncoderLayer; no FP16, no distributed training, no custom kernels
- Only the L(N), L(N,D), and L(C) trends; no critical batch size measurement or learning curve extrapolation L(N,S)

### What the Notebook Result Shows

The fitted power law *L*(*N*) = (*N*\_c/*N*)^α\_N achieves R² = 0.9944 across our 6 models (25K to 793K non-embedding parameters), confirming the paper's central claim that loss scales as a smooth power of model size — even for models ~10⁶× smaller than the paper's range. The fitted exponent α\_N = 0.1367 is substantially larger than the paper's 0.076 because character-level modelling on a tiny corpus is a steeper regime. Validation loss drops from 2.23 (perplexity 9.26) to 1.40 (perplexity 4.05) across the 6 models, with per-model prediction errors under 2%.

The L(N,D) experiment shows severe overfitting for the largest model on small data: at D = 20K characters, the 793K-parameter model reaches train loss 0.07 but validation loss 8.87 (classic memorisation), while the 25K-parameter model's validation loss is 2.46.

As data grows to the full 10.8M characters, the gap reverses — the large model's loss (1.44) drops well below the small model's (2.22), qualitatively matching Equation (1.5). The compute scaling L(C) also follows a clean power law (α\_C = 0.1367, R² = 0.9944). Training curves show larger models reach lower validation loss in fewer steps, reproducing the sample-efficiency observation from Figure 2. Absolute values of α\_N, *N*\_c, etc. are not comparable to the paper because of the scale and tokenisation differences.

## Influence

The Kaplan scaling laws paper became one of the most influential works in LLM research, directly motivating the push toward ever-larger models. The compute-optimal allocation finding — that most additional compute should buy bigger models — guided training decisions at OpenAI (GPT-3, published June 2020) and other labs. The paper's framework was extended by Hoffmann et al. (2022) in "Training Compute-Optimal Large Language Models" (the "Chinchilla" paper, arXiv:2203.15556), which revised the optimal data-to-model ratio upward and corrected the field's overemphasis on model size at the expense of data. The concept of scaling laws has since become a standard tool for planning pre-training runs at frontier labs. About 1,750 citations ([OpenAlex](https://openalex.org/works/W3001279689), checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/034_scaling-laws-neural-language-models/solution.ipynb) | From-scratch PyTorch: trains 6 tiny Transformers on char-level WikiText-2, fits the L(N) power law (R² = 0.99), and reproduces overfitting and compute-scaling trends |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
