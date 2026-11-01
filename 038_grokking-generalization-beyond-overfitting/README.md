---
title: "038 Grokking"
nav_order: 39
permalink: "/038_grokking-generalization-beyond-overfitting/"
---
# Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets (Grokking)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/grokking-generalization-beyond-overfitting)

- **Paper:** [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
- **Authors:** Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra
- **Lab, year:** OpenAI (Vedant Misra at OpenAI at the time of the work), 2022 (arXiv preprint, 6 January 2022)

## Summary

Can a neural network learn the underlying rule behind a set of equations, rather than just memorizing them? Power et al. trained small Transformers on binary operation tables — things like modular addition ($a + b \bmod 97$) or permutation composition in $S_5$ — where every possible equation is known and the training set is just a random fraction of them. They discovered a phenomenon they called **grokking**: the model memorizes the training data within hundreds of steps (training accuracy hits ~100%), but validation accuracy stays at random chance for thousands or even hundreds of thousands of additional steps. Then, suddenly, validation accuracy jumps from chance to perfect generalization. The model "groks" the mathematical structure long after it appeared to have merely overfit.

The paper studied grokking across 12+ binary operations and found it is most pronounced when the training data fraction is near the minimum needed for generalization. Weight decay was the single most effective intervention — it roughly halved the amount of training data needed. Smaller datasets required exponentially more optimization steps to grok: near the critical data fraction, a 1% reduction in training data could increase the time to generalization by 40–50%. The authors also visualized the learned embeddings and found that grokked networks discover recognizable mathematical structure — circular topology for modular addition, coset clusters for permutation groups.

This repo's notebook trains a 2-layer Transformer on modular addition ($p = 97$, 8% train fraction) and reproduces the grokking phenomenon: training accuracy reaches ~100% within ~20 steps, while validation accuracy plateaus at ~98% for ~15 more steps, then jumps to ~100%.

## What Problem Does It Solve

Imagine you're given a partially filled Sudoku grid. At first, you just write down numbers that fit the rows you can see — you're memorizing, not understanding the rules. But if you keep staring at it long enough, suddenly the pattern clicks and you can fill in every empty cell. That's grokking.

Classical learning theory says that once a model perfectly fits its training data, more training should not help — the model has "overfit" and should generalize poorly. But in practice, neural networks often keep improving on unseen data long after they've memorized the training set. This was poorly understood. Power et al. created a controlled laboratory — tiny algorithmic datasets where you know the ground truth rule — to study this phenomenon in detail. The simplicity of the setup (a single GPU, a few thousand equations, a 2-layer Transformer) makes generalization measurable and reproducible in minutes rather than days.

## Core Idea

- **Grokking (Section 3.1):** Generalization can happen abruptly and far past the point of overfitting. Training accuracy reaches ~100% in $10^3$ steps, but validation accuracy stays at chance until $10^5$–$10^6$ steps, then jumps to perfect generalization. This is not gradual improvement — it is a phase transition.
- **Data efficiency curves (Section 3.1.1):** The number of optimization steps required for generalization grows explosively as the training data fraction decreases. Near the critical fraction (~25–30% for $S_5$ composition), a 1% decrease in data causes a 40–50% increase in time to generalization.
- **Weight decay is key (Section 3.3):** Weight decay is the most effective intervention for inducing grokking, more than halving the data needed. Gradient noise (from minibatches or explicit Gaussian noise) also helps, consistent with the idea that noise drives the optimizer toward flatter minima that generalize better.

## Key Method Details

- **Model (Appendix A.1.2):** Standard decoder-only Transformer with causal attention, 2 layers, width 128, 4 attention heads, ~$4 \times 10^5$ non-embedding parameters. Loss and accuracy computed only on the answer token.
- **Optimizer:** AdamW with learning rate $10^{-3}$, weight decay 1, $\beta_1 = 0.9$, $\beta_2 = 0.98$, linear warmup over first 10 steps, minibatch size 512 (or half the training set, whichever is smaller), optimization budget $10^5$ steps (extended to $10^6$ for some experiments).
- **Binary operations (Appendix A.1.1):** 12+ operations on prime $p = 97$ (addition, subtraction, division, polynomials) and the symmetric group $S_5$ (product, conjugation, squaring). Each operand presented as an abstract token — no internal structure revealed.
- **Data efficiency (Section 3.1.1):** For each operation, varied training fraction from 25% to 95%, measured median steps to 99% validation accuracy over 3–7 seeds. Near-critical fractions show exponential increase in optimization time.
- **Ablations (Section 3.3):** Compared full-batch Adam, SGD, residual dropout, weight decay (to origin and to initialization), gradient noise, and learning rate variations. Weight decay to origin was best; weight decay to initialization was effective but less so.
- **Embedding visualization (Section 3.4):** t-SNE of output layer weights revealed circular topology for modular addition and coset structure for $S_5$, more apparent in weight-decay-trained networks.
- **Generalization measures (Appendix A.5):** Sharpness (Hochreiter & Schmidhuber, 1997) of the minimum correlates with generalization — Spearman correlation $-0.795$ ($p < 0.000014$) between sharpness $\phi$ and validation accuracy across trained networks.

### Simplifications in This Notebook

- Modular addition only ($p = 97$), instead of 12+ binary operations (division, $S_5$ composition, polynomials, etc.)
- 2-layer Transformer, width 128, 4 heads (~39K non-embedding parameters) — same architecture as the paper
- AdamW with lr $10^{-3}$, weight decay 0, $\beta_1=0.9$, $\beta_2=0.98$, warmup 10 steps — weight decay set to 0 (paper used 1.0; we omit it to make the grokking gap more visible)
- 3,000 optimization steps instead of $10^5$ (sufficient to observe grokking at 8% train fraction without weight decay)
- 8% train fraction (paper explored 25–95%; we use a very low fraction to make the grokking gap visible on a short timescale)
- No weight decay (paper found it accelerates grokking; we omit it to make the gap more visible), no ablation study, no data efficiency curves, no outlier experiments
- t-SNE visualization of token embeddings instead of output layer weights

### What the Notebook Result Shows

On modular addition $a + b \bmod 97$ with 8% of all 9,409 equations in the training set (752 train / 8,657 val), the 2-layer Transformer groks: training accuracy reaches >99% at **step ~20**, while validation accuracy plateaus at ~97.8% from steps ~21–34, then jumps to 100% at **step ~35**. The grokking gap is ~15 steps (ratio 1.8×).

The validation loss first decreases, then increases slightly during memorization (overfitting), then decreases sharply when grokking occurs. The t-SNE projection of the learned token embeddings shows structured clustering consistent with the cyclic group $\mathbb{Z}/97\mathbb{Z}$.

These results match the paper's Figure 1 qualitatively. Our grokking transition occurs much earlier (step ~35 vs. $10^3$–$10^5$) because we use AdamW with minibatches and a very low data fraction, whereas the paper used full-batch Adam over $10^5$ steps to emphasize how late generalization can begin. Weight decay is set to 0 for the same reason — the paper found it accelerates grokking.

## Influence

Grokking became one of the most studied phenomena in deep learning generalization research. It directly inspired mechanistic interpretability work — most notably Neel Nanda's "Progress Measures for Grokking via Mechanistic Interpretability" (ICLR 2023), which reverse-engineered the circuits a grokking network learns, and Ziming Liu et al.'s "Towards Understanding Grokking: An Effective Theory of Representation Learning" (ICLR 2022). Google PAIR created an interactive explorable article ("Do Machine Learning Models Memorize or Generalize?", August 2023) that visualizes grokking for a general audience. The phenomenon influenced research on phase transitions in learning, the lottery ticket hypothesis, and the relationship between memorization and generalization in overparametrized networks. About 83 citations ([OpenAlex](https://openalex.org/works/W4226434736), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/038_grokking-generalization-beyond-overfitting/solution.ipynb) | From-scratch PyTorch 2-layer Transformer trained on modular addition (mod 97), reproducing the grokking phase transition from memorization to perfect generalization |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
