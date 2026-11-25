---
title: "089 Toy Superposition"
nav_order: 90
permalink: "/089_toy-models-of-superposition/"
---
# Toy Models of Superposition (Toy Superposition)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/toy-models-of-superposition)

- **Paper:** [Toy Models of Superposition](https://arxiv.org/abs/2209.10652)
- **Authors:** Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, Christopher Olah
- **Lab, year:** Anthropic (with Harvard affiliation for Robert Lasenby), 2022 (Transformer Circuits Thread, September 2022)

## Summary

Neural networks often have "polysemantic" neurons — single neurons that respond to multiple unrelated concepts, making them hard to interpret. This paper from Anthropic's Transformer Circuits Thread provides the first unambiguous demonstration of **superposition**: a phenomenon where neural networks represent more features than they have dimensions by packing features into nearly orthogonal directions in a lower-dimensional space.

The authors build a simple toy model: a ReLU network that maps $n$ features through an $m$-dimensional bottleneck ($n > m$) and reconstructs them. They show that when features are dense, the model behaves like PCA and represents only the top-$m$ features. But when features are sparse (rarely active), a **phase change** occurs: the model suddenly starts representing all $n$ features simultaneously by arranging them in geometric structures (antipodal pairs, triangles, pentagons, tetrahedra) that minimize interference.

The paper also discovers connections between superposition and adversarial examples (interference terms create attack directions), between superposition and computation (the model can compute absolute value in superposition), and proposes that real neural networks are "noisily simulating" larger sparse networks. This insight directly motivated the development of sparse autoencoders for interpretability.

This repo's notebook trains the canonical 5-feature, 2-dimensional toy model, sweeps sparsity to show the phase change, visualizes the geometric structures, and measures adversarial vulnerability.

## What Problem Does It Solve

Imagine you have a bookshelf with only 5 slots but 10 books you want to store. You can't give each book its own slot. But if most books are rarely needed, you can stack two books in one slot — they'll almost never be needed at the same time, so the occasional inconvenience of digging one out is worth fitting them all.

Neural networks do something similar: when features (concepts like "is this a dog?" or "is this text in French?") are rarely active at the same time, the network can "stack" multiple feature directions into fewer neurons. This is superposition, and it explains why neurons are often polysemantic — one neuron responding to seemingly unrelated things. The paper asks: when and why does this happen, and what structure does it create?

## Core Idea

- **Superposition:** A linear representation can store more features than dimensions by using nearly-orthogonal directions. Sparse features rarely interfere, so the cost of overlap is low.
- **Phase change:** Whether a feature is represented or not is governed by a sharp transition controlled by sparsity. Below a threshold, only top-$m$ features are represented; above it, all features enter superposition.
- **Geometric structure:** Features in superposition arrange themselves into regular polytopes — antipodal pairs (digons), triangles, pentagons, tetrahedra — that minimize interference, connecting to the Thomson problem in physics.

## Key Method Details

- **Model:** $x' = \text{ReLU}(W^T W x + b)$, where $W$ is $n \times m$ (features x hidden dimensions), $b$ is a bias vector
- **Loss:** Importance-weighted MSE: $L = \sum_i I_i (x_i - x_i')^2$ with $I_i = 0.8^i$
- **Data:** Synthetic sparse features; each $x_i = 0$ with probability $S$ (sparsity), else uniform$[0,1]$
- **Phase change threshold:** In the canonical 5-feature, 2D setup, the transition from 2 features to 5 features occurs around $S \approx 0.9$–$0.95$
- **Geometric structures observed:** 2 features → antipodal pair; 3 features → triangle; 5 features → pentagon; 4 features → square/tetrahedron-like in 3D

### Simplifications in This Notebook

- Smaller scale: 5–7 features in 2 dimensions (the paper also studies up to 80 features in 20 dimensions)
- Single model architecture (no privileged basis / ReLU hidden layer model)
- No computation-in-superposition experiment (the abs(x) experiment)
- Fixed importance schedule $I_i = 0.8^i$ (the paper also explores uniform importance)
- Simplified adversarial attack (analytic direction, single attack norm)
- 800–1000 training steps per model (the paper trains longer and averages over many seeds)

### What the Notebook Result Shows

At sparsity $S = 0.0$ (dense features), only 2 of 5 features have norms above 0.1 — the model behaves like PCA, representing only the two most important features. As sparsity increases, more features gain representation: 3 features at $S=0.3$, 4 at $S=0.7$, and all 5 at $S=0.95$ and $S=0.99$ — superposition.

At $S=0.999$, only 3 features are represented (the three least important collapse, an artifact of the very high sparsity with finite training steps). The 5 feature directions at $S=0.99$ form a near-pentagonal arrangement in 2D, with norms around 1.15 each. Adversarial vulnerability, measured as the maximum loss increase from an L2 attack of norm 0.1, shows a modest trend across sparsity levels. These results are qualitatively consistent with the paper but not directly comparable, since the paper uses different importance schedules, longer training, multiple seeds, and reports different metrics.

## Influence

This paper became a foundational work for mechanistic interpretability. It directly motivated the development of **sparse autoencoders** (SAEs) for decomposing polysemantic neurons into monosemantic features — most notably in the follow-up *Towards Monosemanticity* (Bricken et al., 2023, also from Anthropic's Transformer Circuits Thread) and *Scaling Monosemanticity* (Templeton et al., 2024). The superposition hypothesis has influenced interpretability research at Google DeepMind, OpenAI, and EleutherAI. The connection between superposition and adversarial robustness has been explored in subsequent safety research. Citation count: about 1,700 citations ([OpenAlex](https://openalex.org/works/W4296932880), checked November 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/089_toy-models-of-superposition/solution.ipynb) | From-scratch PyTorch toy model trained on synthetic sparse features, sweeping sparsity to show the phase change and geometric structures |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
