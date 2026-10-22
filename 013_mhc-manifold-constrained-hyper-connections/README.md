---
title: "013 Hyper-Connections"
nav_order: 14
permalink: "/013_mhc-manifold-constrained-hyper-connections/"
---
# mHC: Manifold-Constrained Hyper-Connections (Hyper-Connections)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mhc-manifold-constrained-hyper-connections)

- **Paper:** [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- **Authors:** Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang
- **Lab, year:** DeepSeek-AI, 2025 (preprint)

## Summary

Hyper-Connections (HC), introduced by Zhu et al. in 2024, replace the standard residual connection (the "x + F(x)" skip path that lets gradients flow through deep networks) with a wider, multi-stream residual stream. Instead of a single channel of width C, the stream is expanded to n × C (e.g., 4C), and three learnable matrices control how features are read into each layer, mixed across streams, and written back. This gives the network more capacity to route information across depths without increasing the FLOPs (floating-point operations, a measure of compute cost) of the computation blocks themselves. HC showed strong performance gains in language model pretraining.

However, HC has a critical flaw: the residual mixing matrix is unconstrained. When composed across many layers, the product of these matrices can amplify or attenuate signals without bound, breaking the "identity mapping" property that makes residual networks stable. The authors observe real training instability — loss spikes and gradient norm explosions around 12k steps in 27B-scale experiments. HC also increases memory access (I/O) costs by roughly a factor of n due to the wider stream, creating a hardware bottleneck.

mHC solves this by projecting the residual mixing matrix onto the manifold of doubly stochastic matrices (matrices where every row and column sums to 1) using the Sinkhorn-Knopp algorithm. This guarantees that the mixing operation is a convex combination — it preserves the mean signal and keeps the spectral norm at 1, so signals neither vanish nor explode. The authors also engineer fused kernels, selective recomputation, and pipeline-aware scheduling to bring the training overhead down to just 6.7% at n = 4. In 27B MoE (Mixture-of-Experts) experiments, mHC achieves a final loss reduction of 0.021 over baseline and improves reasoning benchmarks like BBH by 2.1% and DROP by 2.3% over HC, while maintaining full training stability.

This repo's notebook implements mHC and HC from scratch in PyTorch, trains a tiny transformer with each connection type on Tiny Shakespeare, and compares signal norm propagation with depth.

## What Problem Does It Solve

Imagine a relay race where each runner hands a baton to the next. In a standard residual network, the baton is passed directly — every runner adds their contribution, but the original baton is always there, unchanged. This "identity mapping" keeps the race stable: the signal never gets lost or amplified out of control, no matter how many runners (layers) there are.

Hyper-Connections change the race so that each runner can mix the baton with previous runners' batons in complex ways — think of it as having multiple batons in play at once, with learned rules for how to combine them. This is more expressive, but the mixing rules are unconstrained: one runner might accidentally amplify the signal 10x, the next might shrink it to 0.1x, and over 80 layers the signal either explodes or vanishes. The authors show this happening in practice at 27B scale, with loss spikes and gradient explosions that make training unstable.

## Core Idea

- **Doubly stochastic manifold projection (Section 4.1):** The residual mixing matrix H_res is projected onto the Birkhoff polytope (the set of doubly stochastic matrices, where every row and column sums to 1) using the Sinkhorn-Knopp algorithm. This makes the mixing operation a convex combination of input features, preserving the mean signal and bounding the spectral norm at 1.
- **Compositional closure (Section 4.1):** Doubly stochastic matrices are closed under multiplication, so the product of H_res matrices across many layers remains doubly stochastic. This is the key theoretical property that ensures stability at any depth — the composite mapping never amplifies or vanishes.
- **Infrastructure optimization (Section 4.3):** Fused kernels using TileLang, selective recomputation to reduce memory footprint, and pipeline-aware communication scheduling in DualPipe bring the practical overhead to just 6.7% at n = 4, making mHC deployable in large-scale training.

## Key Method Details

- **HC single-layer propagation (Eq. 3):** x_{l+1} = H_res · x_l + H_post^T · F(H_pre · x_l, W_l), where x_l is expanded from C to nC dimensions, H_res (n×n) mixes streams, H_pre (1×n) reads into the layer, H_post (1×n) writes back.
- **Sinkhorn-Knopp projection (Section 4.2):** Given the raw matrix H~_res, compute M = exp(H~_res), then iteratively normalize rows and columns to sum to 1. The authors use t_max = 20 iterations. Pre and post mappings are constrained via sigmoid for non-negativity.
- **HC coefficient computation (Eq. 5):** Each mapping = alpha * tanh(theta * RMSNorm(x)) + b, with alpha a learnable gating scalar (initialized small), theta a linear projection, and b a static bias.
- **Training setup (Section 5.1):** MoE architectures inspired by DeepSeek-V3, four model variants (3B, 9B, 27B), expansion rate n = 4 for both HC and mHC, 27B as the primary subject.
- **Results — 27B (Section 5.2, Tab. 4):** mHC achieves final loss reduction of 0.021 vs baseline. BBH +2.1% vs HC, DROP +2.3% vs HC. HC exhibits loss surge around 12k steps; mHC maintains stable gradient norm comparable to baseline.
- **Scaling (Section 5.3):** Compute scaling curve across 3B/9B/27B shows performance advantage is robustly maintained with only marginal attenuation at higher scales.
- **Overhead (Section 4.3):** mHC with n = 4 introduces only 6.7% additional training time vs baseline.

### Simplifications in This Notebook

- Smaller model: 6 layers, 4 heads, 128 dim, about 1.0M parameters (vs 27B MoE)
- Dataset: Tiny Shakespeare (~1 MB, character-level, vocab 65) instead of large-scale pretraining
- 15 epochs, 50 steps per epoch, batch 32, lr 3e-4 with AdamW (wd=0.1, betas=0.9/0.95) instead of full-scale pretraining
- No MoE, no pipeline parallelism, no fused kernels — pure PyTorch implementation
- Sinkhorn-Knopp with 20 iterations (same as paper)
- Expansion rate n = 4 (same as paper)

### What the Notebook Result Shows

On Tiny Shakespeare validation set (character-level, 128-token context, 15 epochs), the three models achieve: Baseline Transformer val perplexity 9.60 (1,001,350 params), HC Transformer val perplexity 8.71 (1,019,938 params), mHC Transformer val perplexity 10.00 (1,019,938 params). At this toy scale (~1M parameters, 1 MB text, 15 epochs), HC slightly outperforms both baseline and mHC — the unconstrained mixing is more expressive at small scale. The paper's advantage of mHC over HC emerges at 27B scale where training stability becomes the bottleneck.

The key observable difference is in signal norm propagation stability: mHC has a max/min norm ratio of 1.50 across layers, compared to 4.26 for baseline and 8.86 for HC. The doubly stochastic constraint is verified: mHC's H_res matrices have row and column sums of exactly 1.000000 and spectral norm of 1.000000, while HC's unconstrained matrices have row sums of 0.35-0.46 and spectral norm of 0.56. The notebook demonstrates the architecture and the stability mechanism rather than reproducing the scaling-dependent performance gains.

## Influence

mHC is a very recent paper (December 2025) from DeepSeek-AI. It directly extends the Hyper-Connections framework (Zhu et al., 2024) and is designed for integration into DeepSeek's large-scale training infrastructure. The approach of constraining residual connections via manifold projection is part of a broader trend in macro-architecture design (stream widening, dense connectivity) that includes MUDDFormer (Xiao et al., 2025) and the Residual Matrix Transformer (Mak & Flanigan, 2025). The paper has 0 citations ([OpenAlex](https://openalex.org/works/W7117976675), checked October 2026), reflecting its recency.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/013_mhc-manifold-constrained-hyper-connections/solution.ipynb) | From-scratch PyTorch implementation of mHC and HC connection patterns, trained on Tiny Shakespeare, comparing signal norm propagation with depth |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Q&A answered from the paper, citations, and common misconceptions about the paper |
