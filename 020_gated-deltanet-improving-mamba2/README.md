---
title: "020 Gated DeltaNet"
nav_order: 21
permalink: "/020_gated-deltanet-improving-mamba2/"
---
# Gated Delta Networks: Improving Mamba2 with Delta Rule (Gated DeltaNet)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/gated-deltanet-improving-mamba2)

- **Paper:** [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- **Authors:** Songlin Yang, Jan Kautz, Ali Hatamizadeh
- **Lab, year:** MIT CSAIL (Songlin Yang, work done during NVIDIA internship) and NVIDIA, 2024 (ICLR 2025)

## Summary

Linear Transformers are efficient alternatives to standard Transformers but have historically struggled with retrieval and long-context tasks. Two recent mechanisms tried to fix this: **gating** (data-dependent decay, as in Mamba2) for rapid memory erasure, and the **delta rule** (as in DeltaNet) for precise, targeted key-value updates. The paper's key insight is that these two mechanisms are complementary — gating lets the model quickly clear stale information during context switches, while the delta rule lets it surgically update specific key-value associations without disturbing others.

The authors propose the **gated delta rule**: S_t = S_{t-1}(alpha_t(I - beta_t k_t k_t^T)) + beta_t v_t k_t^T, which unifies both. When alpha_t approaches 0, the entire state is cleared (like Mamba2's gate); when alpha_t approaches 1, it switches to the pure delta rule (like DeltaNet). They derive a hardware-efficient chunkwise training algorithm using the WY representation and UT transform, enabling matmul-dominant computation on GPUs.

Gated DeltaNet consistently outperforms Mamba2 and DeltaNet across language modeling (Pile perplexity 12.17 vs 12.56 for Mamba2 at 1.3B), common-sense reasoning (55.32 avg vs 54.89), in-context retrieval, length extrapolation, and long-context understanding (LongBench avg 16.6 vs 13.5). Hybrid variants combining Gated DeltaNet with sliding window attention (GatedDeltaNet-H1) or Mamba2 + SWA (GatedDeltaNet-H2) achieve even higher performance (56.40 and 56.18 avg). This repo's notebook implements the gated delta rule from scratch in PyTorch and compares it against Mamba2 and vanilla linear attention on synthetic associative recall.

## What Problem Does It Solve

Imagine you are memorizing a phone book. Every time you see a new entry, you need to write it into your memory. But if someone changes their number, you need to find the old entry and replace just that one — not erase the whole book. That is what the delta rule does: it surgically updates one key-value pair at a time.

Now imagine you move to a new city. You do not need to update individual entries — you need to throw away the entire old phone book and start fresh. That is what gating does: it rapidly clears all memory when the context changes.

Linear Transformers before this paper could do one or the other, but not both. Mamba2 could clear memory fast but could not surgically update individual entries. DeltaNet could update individual entries but could not clear memory quickly. Gated DeltaNet does both in a single, simple update rule.

## Core Idea

- **Gated delta rule (Eq. 10):** S_t = S_{t-1}(alpha_t(I - beta_t k_t k_t^T)) + beta_t v_t k_t^T — unifies gating (alpha_t) with the delta rule (beta_t and Householder reflection I - beta_t k_t k_t^T)
- **Complementary mechanisms (Section 3.2):** The S-NIAH case study shows that decay hurts long-term retention (DeltaNet wins), gating facilitates filtering (Mamba2 wins on noisy context), and the delta rule helps memorization (Gated DeltaNet wins on complex patterns)
- **Hardware-efficient chunkwise algorithm (Section 3.3):** Extends DeltaNet's WY representation and UT transform to incorporate gating terms, preserving matmul-dominant computation for GPU efficiency
- **Hybrid architectures (Section 3.4):** Combining Gated DeltaNet with sliding window attention (H1) or Mamba2 + SWA (H2) further improves both training efficiency and task performance

## Key Method Details

- **State update (Eq. 10):** S_t = S_{t-1}(alpha_t(I - beta_t k_t k_t^T)) + beta_t v_t k_t^T, where alpha_t in (0,1) is a data-dependent gate and beta_t in (0,1) is the delta rule's writing strength
- **Online learning interpretation (Table 1):** The gated delta rule is the closed-form solution to an online learning objective with adaptive weight decay (alpha_t) and online regression loss (the delta rule's Householder update)
- **Chunkwise algorithm (Section 3.3):** T matrix via UT transform: T = [I + strictLower(diag(beta) * (Gamma * KK^T))]^{-1} * diag(beta), where Gamma encodes within-chunk decay. W = T @ K, U-tilde = T @ V. State update and output combine decay arrows (to-first, to-last positions) with these factors.
- **Block design (Section 3.4):** Q/K/V via linear projection + short convolution + SiLU; L2 normalization on Q and K for stability; alpha via Mamba2-style parameterization (softplus); beta via sigmoid; SiLU output gating
- **Results — Language modeling (Table 3, 1.3B):** WikiText perplexity (a measure of how well a model predicts text; lower is better) of 16.42 (vs Mamba2 16.56, DeltaNet 17.71); LMB 12.17 (vs Mamba2 12.56); avg common-sense reasoning 55.32 (vs Mamba2 54.89, DeltaNet 52.14)
- **Results — Long context (Table 5, LongBench):** Avg 16.6 (vs Mamba2 13.5, DeltaNet 13.6); hybrid H1 achieves 17.8, H2 achieves 18.4
- **Results — S-NIAH (Table 2, 1.3B):** Gated DeltaNet maintains 86.6-100.0 accuracy across S-NIAH variants at 1K-2K, while Mamba2 degrades sharply beyond 2K and DeltaNet struggles on noisy context

### Simplifications in This Notebook

- Smaller model: 2 token-mixer layers, 64 model dim, 4 heads, 16 head dim, about 143K parameters (vs 48+ layers, 2048 dim, 1.3B params)
- Dataset: synthetic MQAR (16-token vocab, 8 key-value pairs, 4000 examples, L=128) instead of FineWeb-Edu (100B tokens)
- 15 epochs, batch 64, lr 1e-3 with AdamW, no LR schedule (paper uses cosine annealing with 1B-token warmup)
- Pure PyTorch chunkwise form with torch.linalg.solve (paper: custom CUDA/Triton kernels)
- No short convolution in Q/K/V paths (paper uses short conv + SiLU); L2 normalization and SiLU output gating are included

### What the Notebook Result Shows

On a synthetic multi-query associative recall task (sequence length 128, 16-token vocab, 8 key-value pairs, 4000 training examples, 15 epochs), Gated DeltaNet achieves validation accuracy of about 0.94, Mamba2-style decayed linear attention reaches about 0.91-0.93, and vanilla linear attention reaches about 0.55-0.59. The gated delta rule's combination of targeted updates (delta rule) and adaptive forgetting (gating) gives it an edge over Mamba2's uniform decay on this recall task. The random baseline is 1/16 = 0.0625. This toy result is not comparable to the paper's 1.3B-parameter benchmarks — the advantage of the gated delta rule is most pronounced on real-world tasks that require both precise retrieval and context switching (Tables 3-5).

## Influence

Gated DeltaNet (ICLR 2025) unified the gating and delta rule mechanisms in linear recurrent models, showing that their combination is more powerful than either alone. It directly influenced RWKV-7 (BlinkDL, 2024), which uses a similar diagonal-plus-low-rank transition with gating. The hybrid architectures (GatedDeltaNet-H1, H2) contributed to the growing trend of combining linear recurrence with sliding window attention in production models. The official code at [github.com/NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) provides CUDA kernel implementations. The paper has about 50 citations on [Semantic Scholar](https://www.semanticscholar.org/paper/arXiv:2412.06464) (checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/020_gated-deltanet-improving-mamba2/solution.ipynb) | From-scratch PyTorch implementation of the gated delta rule (recurrent + chunkwise forms), compared against Mamba2 and vanilla linear attention on synthetic MQAR (GDN 0.94, M2 0.93, LA 0.59) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
