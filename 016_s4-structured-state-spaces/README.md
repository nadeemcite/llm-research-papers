---
title: "016 S4"
nav_order: 17
permalink: "/016_s4-structured-state-spaces/"
---
# Efficiently Modeling Long Sequences with Structured State Spaces (S4)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/s4-structured-state-spaces)

- **Paper:** [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
- **Authors:** Albert Gu, Karan Goel, Christopher Re
- **Lab, year:** Stanford University (Department of Computer Science), 2021 (ICLR 2022 Outstanding Paper HM)

## Summary

S4 (Structured State Space Sequence Model) is a sequence modeling architecture based on continuous-time state space models (SSMs) — the linear differential equations x'(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t) that describe how a latent state evolves over time under an input signal. Prior work (HiPPO, LSSL) showed that SSMs with a special initialization of the matrix A (the HiPPO matrix) can mathematically capture long-range dependencies (LRDs) over thousands of steps, but was computationally infeasible: computing the SSM over a length-L sequence required O(L^2) or O(N^2 L) operations, making it impractical as a general sequence model.

S4's breakthrough is a new parameterization of A called Normal Plus Low-Rank (NPLR), which allows the SSM to be diagonalized stably and reduces the core computation to a Cauchy kernel — a well-studied operation computable in O((N+L) log^2(N+L)) time via FFTs. This makes S4 efficient enough to train on long sequences while preserving the theoretical long-range memory of HiPPO-initialized SSMs. The same model can be computed as a convolution (parallel training) or a recurrence (efficient streaming inference), combining the strengths of CNNs and RNNs.

S4 achieved state-of-the-art results across diverse benchmarks: 91% accuracy on sequential CIFAR-10 (no augmentation), SoTA on all six Long Range Arena (LRA) tasks including solving Path-X at length 16384 (which all prior models failed), and substantially closing the gap to Transformers on language modeling (WikiText-103 perplexity of 20.95 vs 20.51) while generating 60x faster. This repo's notebook implements a diagonal SSM layer with HiPPO-LegS initialization and compares it against a GRU and a Transformer on a synthetic long-range retrieval task.

## What Problem Does It Solve

Imagine trying to remember the first sentence of a 100-page book when you reach the last page. For a human, that is hard. For a neural network processing a 10,000-step sequence, it is even harder. RNNs compress everything into a fixed-size hidden state that gets overwritten by new inputs. Transformers can attend to any position but require O(L^2) memory — at length 16,384 (the Path-X benchmark), that is 268 million attention scores, which exceeded available memory for all prior models.

S4's solution comes from an unexpected source: continuous-time differential equations. Instead of processing tokens one at a time (RNN) or comparing all pairs at once (Transformer), S4 maps the input through a state space model — a mathematical system whose state integrates information over all timescales simultaneously. The HiPPO matrix ensures each state dimension captures a different timescale, from short-term to very long-term, so the model naturally retains both recent and ancient information without an explicit attention mechanism.

## Core Idea

- **HiPPO-initialized state space (Section 2.2):** The state matrix A is initialized to the HiPPO-LegS matrix, which produces a state space where each dimension memorizes the input signal at a progressively longer timescale. This gives the SSM its long-range memory property.
- **NPLR parameterization (Section 3.2):** A is parameterized as a normal matrix minus a low-rank correction (A = Lambda - PQ*). This allows stable diagonalization — converting the O(N^2) matrix power computation into O(N) element-wise operations plus a rank-r correction handled via the Woodbury identity.
- **Dual convolutional/recurrent view (Sections 2.3-2.4):** The discrete SSM can be computed as a convolution (kernel K = [CB, CAB, CA^2B, ...]) for parallel training, or as a recurrence (x_k = A_bar x_{k-1} + B_bar u_k) for O(N) per-step inference. S4 uses the convolutional view during training and the recurrent view during generation.
- **Cauchy kernel computation (Reduction 3):** The convolution kernel is computed via a generating function that reduces to a Cauchy kernel, solvable in O((N+L) log^2(N+L)) time — exponentially faster than the naive O(N^2 L) approach.

## Key Method Details

- **Continuous SSM (Eq. 1):** x'(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t), where A is N x N, B is N x 1, C is 1 x N, D is a scalar skip connection.
- **HiPPO-LegS matrix (Eq. 2):** A_{nk} = -sqrt(2n+1) if n=k, sqrt(2k+1) if n > k, 0 if n < k. This is NPLR with rank r=1.
- **Bilinear discretization (Section 2.3):** A_bar = (I - Delta/2 A)^{-1}(I + Delta/2 A), B_bar = (I - Delta/2 A)^{-1} Delta B, where Delta is a learned step size.
- **Convolutional kernel (Eq. 4):** K_bar = [CB, CAB, CA^2B, ..., CA^{L-1}B]. The SSM output is the convolution of the input with K_bar, enabling parallel training.
- **S4 layer architecture (Section 3.4):** H independent SSM copies (one per hidden channel), mixed by a position-wise linear layer (O(H^2) + O(HN) parameters per layer). Nonlinear activations between layers. Similar to a depthwise-separable CNN with global convolution kernels.
- **Results — LRA (Table 2):** S4 averages 86.09% across 6 tasks (ListOps 59.60, Text 86.82, Retrieval 90.90, Image 88.65, Pathfinder 94.20, Path-X 96.35), vs less than 60% for all Transformer baselines. Path-X (length 16384) solved for the first time (per the paper's claim).
- **Results — WikiText-103 (Table 3):** Perplexity 20.95 (249M params), vs Transformer 20.51 (247M). Generation speed 48K tokens/sec (60x faster).
- **Results — sequential CIFAR-10 (Section 4.3):** 91% accuracy with no data augmentation or auxiliary losses, on par with a larger 2-D ResNet.

### Simplifications in This Notebook

- Smaller model: 2 layers, 64 hidden dim, 64 state size, about 50K parameters (vs 4-6 layers, 128-512 dim, millions of params)
- Diagonal SSM instead of NPLR: we diagonalize the HiPPO matrix directly (as in S4D/S5) instead of using Cauchy kernel computation
- Dataset: synthetic long-range retrieval task at L=512 (10-digit vocab, 2000 train examples) instead of LRA (6 tasks, 1K-16K steps) or WikiText-103
- 10 epochs, batch 32, lr 1e-3 with AdamW, no plateau scheduler (paper uses 50-200 epochs with constant LR + decay on validation plateau)
- No data augmentation, no BatchNorm/LayerNorm choices to tune (paper ablates these)
- FFT-based convolution for training instead of the Cauchy kernel algorithm

### What the Notebook Result Shows

On a synthetic long-range retrieval task (sequence length 1024, vocab 10, 2000 training examples, 10 epochs), the diagonal SSM with HiPPO-LegS initialization achieves validation accuracy of roughly 0.95+, compared to about 0.5-0.7 for a GRU baseline and about 0.8-0.9 for a 2-layer Transformer — all with similar parameter counts (about 50K-70K). The S4 model retains the query digit from position 0 through the full 1024-step sequence thanks to the HiPPO matrix's multi-timescale state representation.

The GRU struggles because its gates cannot preserve information over 1024 steps. The Transformer can attend to position 0 but overfits on the small training set. This toy result is not comparable to the paper's benchmarks (LRA, WikiText-103, sequential CIFAR-10) — the advantage of S4's efficient computation is most pronounced at very long sequences (16K+) where Transformers run out of memory.

## Influence

S4 (ICLR 2022 Outstanding Paper HM) catalyzed the entire field of structured state space models for deep learning. Direct follow-ups include S4D (simplified diagonal version, ICLR 2023), S5 (stacked SSM with scan, ICLR 2023), and H3 (for language modeling, ICLR 2023). Most importantly, S4 directly led to Mamba (Gu and Dao, 2023), which introduced a selective scan mechanism making SSMs input-dependent and competitive with Transformers on language modeling at scale. The Mamba family (Mamba-2, Jamba, etc.) has become a major alternative architecture to Transformers. S4 has about 2,200 citations on [OpenAlex](https://openalex.org/works/W4318831551) (checked October 2026). Code is available at [github.com/HazyResearch/state-spaces](https://github.com/HazyResearch/state-spaces).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/016_s4-structured-state-spaces/solution.ipynb) | From-scratch PyTorch implementation of a diagonal SSM layer with HiPPO-LegS initialization, bilinear discretization, FFT-based convolution, and recurrent inference mode, compared against GRU and Transformer on a synthetic long-range retrieval task |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Q&A answered from the paper, citations, and common misconceptions about the paper |
