---
title: "S4: Talk & Q&A"
parent: "016 S4"
nav_order: 2
---
# TALK — Efficiently Modeling Long Sequences with Structured State Spaces (S4)

## Talks, Slides and Press Coverage

1. **Albert Gu — "Structured State Space Models for Long-Range Dependencies" (ICLR 2022 Outstanding Paper HM):** Albert Gu presented S4 at ICLR 2022, covering the HiPPO initialization, NPLR parameterization, and results on the Long Range Arena benchmark. ([video](https://iclr.cc/virtual/2022/oral/6652))

2. **Albert Gu — "S4: Structured State Space Models" (Stanford MLSys seminar, 2022):** A technical talk at Stanford covering the mathematical foundations of S4, including the Cauchy kernel reduction and the NPLR parameterization. ([video](https://www.youtube.com/watch?v=luKXn98klns))

3. **Sasha Rush — "S4: The State Space Model" (Annotated S4, 2022):** An annotated walk-through of the S4 paper with code, explaining the HiPPO matrix, discretization, and the convolutional kernel computation step by step. ([link](https://srush.github.io/annotated-s4/))

## Q&A (answers from the paper)

**Q1: Why use a continuous-time state space model instead of a discrete RNN?**
A1 (from the paper, Section 2.1): The SSM maps a 1-D input signal u(t) to an N-D latent state x(t) before projecting to a 1-D output y(t) via the differential equation x'(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t). The continuous-time formulation allows the model to handle irregularly sampled data and provides a principled way to capture long-range dependencies through the choice of the state matrix A.

**Q2: What is the HiPPO matrix and why is it important?**
A2 (from the paper, Section 2.2): Prior work found that the basic SSM performs very poorly in practice. The HiPPO framework provides a specific initialization of the matrix A that produces a state space with the ability to memorize long histories. "These matrices induce a state space with a timescale that is exponentially long in the state size, allowing them to capture dependencies over very long horizons."

**Q3: Why can't the HiPPO matrix be diagonalized directly?**
A3 (from the paper, Section 3.1, Lemma 3.1): Naive diagonalization does not work due to numerical issues. The authors show that the eigenvector matrix V has entries exponentially large in the state size N, rendering the diagonalization numerically infeasible — C V in Lemma 3.1 would not be computable. S4 solves this with the NPLR parameterization that avoids explicit diagonalization.

**Q4: How does S4 achieve both parallel training and efficient inference?**
A4 (from the paper, Sections 2.3-2.4): The discrete SSM can be written in two equivalent forms. The recurrent form (x_k = A_bar x_{k-1} + B_bar u_k) allows O(N) per-step inference like an RNN. The convolutional form (output = K * input where K = [CB, CAB, CA^2B, ...]) allows parallel training like a CNN. "Convolutions are efficient for training while recurrence is efficient for inference, while SSMs combine the strengths of both."

**Q5: Did S4 use a smaller learning rate for the HiPPO parameters?**
A5 (from the paper, Appendix D.2): Yes. "The learning rate on HiPPO parameters (in particular Lambda, P, Q, B, C, Delta) were reduced to a maximum starting LR of 0.001, which improves stability since the HiPPO equations are sensitive."

## Common Misconceptions

1. **"S4 is just an RNN with a fancy initialization"** — While S4 can be computed as a recurrence (like an RNN), its key innovation is the dual convolutional/recurrent view. During training, it computes the entire sequence as a single convolution (parallel, like a CNN), not as a sequential recurrence. The HiPPO initialization gives it long-range memory that vanilla RNNs lack, and the NPLR parameterization makes the convolution tractable.

2. **"S4 replaces Transformers entirely"** — S4 substantially closes the gap to Transformers but does not surpass them on all tasks. On WikiText-103, S4 achieves perplexity 20.95 vs the Transformer's 20.51 — a gap of 0.8. S4's main advantage is on very long sequences (16K+ steps) where Transformers run out of memory, and in generation speed (60x faster).

3. **"The diagonal version (used in this notebook) is the original S4"** — The original S4 uses the NPLR (Normal Plus Low-Rank) parameterization with Cauchy kernel computation. The diagonal simplification was introduced in later work (S4D by Gu et al., ICLR 2023; S5 by Smith et al., ICLR 2023). The diagonal version is simpler but may lose some expressiveness.

## Real Citations

1. Albert Gu, Karan Goel, Christopher Re (2021). "Efficiently Modeling Long Sequences with Structured State Spaces." *ICLR 2022 (Outstanding Paper HM).* [arXiv:2111.00396](https://arxiv.org/abs/2111.00396) — the original S4 paper.

2. Albert Gu, Karan Goel, Ankit Gupta, Christopher Re (2022). "On the Parameterization and Initialization of Diagonal State Space Models." *NeurIPS 2022.* [arXiv:2206.11893](https://arxiv.org/abs/2206.11893) — S4D, the simplified diagonal version used in this notebook.

3. Albert Gu, Tri Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) — Mamba, the most influential follow-up, making SSMs input-dependent and competitive with Transformers at scale.

**Citation count:** about 2,200 citations on [OpenAlex](https://openalex.org/works/W4318831551) (checked October 2026).
