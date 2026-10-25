---
title: "018 Mamba-2"
nav_order: 19
permalink: "/018_mamba-2-structured-state-space-duality/"
---
# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mamba-2-structured-state-space-duality)

- **Paper:** [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- **Authors:** Tri Dao, Albert Gu
- **Lab, year:** Princeton University (Department of Computer Science) and Carnegie Mellon University (Machine Learning Department), 2024 (ICML 2024)

## Summary

Mamba-2 builds a bridge between two rival families of sequence models: state-space models (SSMs) like Mamba and Transformers with attention. The paper shows they are two sides of the same coin, connected through structured matrices called semiseparable matrices. This bridge, called structured state space duality (SSD), reveals that the linear recurrence used by SSMs and the quadratic computation used by attention are dual forms of the same underlying operation.

The key insight is that a selective SSM can be written as multiplication by a structured (semiseparable) matrix, and so can a certain generalization of attention (structured masked attention). When the SSM has scalar structure on its A matrix, the two coincide exactly. This lets the authors design a new algorithm that splits the matrix into blocks: diagonal blocks use the attention (quadratic) form for hardware efficiency, while off-diagonal blocks use the recurrent (linear) form. The result is 2–8× faster than Mamba-1's scan while supporting 8× larger state sizes.

Mamba-2 also modernizes the block design: all projections (A, B, C, X) happen in parallel at the start (like Q/K/V in attention), an extra normalization stabilizes training, and multi-head patterns from attention transfer directly to SSMs. Mamba-2 Pareto-dominates Mamba-1 and Transformer++ on Chinchilla scaling laws. Mamba-2-2.7B trained on 300B tokens of the Pile outperforms Mamba-2.8B and Pythia-6.9B on zero-shot benchmarks. This repo's notebook implements the core SSD layer (Listing 1 from the paper) in pure PyTorch and verifies the SSM = attention duality numerically.

## What Problem Does It Solve

Imagine you are reading a book and keeping a running summary in your head. Every new paragraph, you update the summary — you do not flip back to reread everything. That is how SSMs work: they maintain a fixed-size state that gets updated each step. Transformers work differently: they let every word look back at every earlier word, like having a photographic memory of the entire page. Both approaches work, but they seemed like fundamentally different ideas.

The problem was that nobody knew exactly how these two approaches relate. SSMs (like Mamba) were fast and efficient but hard to optimize on GPUs because they use sequential scans, not matrix multiplications. Transformers were well-understood and hardware-optimized but scale quadratically with sequence length. Mamba-2 shows they are actually the same computation viewed two ways, letting SSMs borrow all of Transformers' hardware tricks.

## Core Idea

- **SSMs are semiseparable matrices (Section 3):** A selective SSM computes y = Mx where M is a lower-triangular semiseparable matrix built from the A, B, C parameters. This matrix view exposes both a linear (recurrent) and quadratic (attention-like) algorithm.
- **Structured masked attention (Section 4):** Linear attention is generalized to structured masked attention (SMA), where the causal mask is replaced by a structured 1-semiseparable mask. This is the attention counterpart of the SSM.
- **Duality (Section 5):** When the SSM's A matrix is scalar (a × I), the quadratic form of the SSM is exactly masked kernel attention. The two models share the same linear and quadratic forms — they are duals.
- **Block-decomposed SSD algorithm (Section 6):** The semiseparable matrix is split into Q×Q blocks. Diagonal blocks use the quadratic (matmul-friendly) form; off-diagonal blocks use low-rank factorization plus a smaller recurrence. This gives O(TN²) training FLOPs with work dominated by matrix multiplications.

## Key Method Details

- **SSD layer dimensions:** State expansion N (64–256), head dimension P (64–128), number of heads H = D_model / P. The SSD layer maps X ∈ R^(T,P) to Y ∈ R^(T,P) per head.
- **Quadratic form (Eq. 16):** For scalar A, the SSM matrix M = L ∘ (CB^T) where L is a 1-semiseparable mask. The output Y = (L ∘ (CB^T)) X, computed as pairwise interactions like attention.
- **Linear form (Eq. 15):** The same operation via recurrence: h_t = a_t h_{t-1} + B_t x_t, y_t = C_t^T h_t. This is O(TN) and used for autoregressive inference.
- **Block SSD algorithm (Theorem 6.1):** With block size Q, diagonal blocks compute intra-chunk outputs via the quadratic form (parallel matmuls), off-diagonal blocks factor into right (B), center (A recurrence), and left (C) terms. Total: O(TN²) training, O(N²) inference memory.
- **Mamba-2 block (Section 7.1):** Parallel projections for A, X, B, C at block start (like Q/K/V); extra RMSNorm before output projection; multi-value attention (MVA) head pattern where B and C are shared across X heads.
- **Results — Speed (Figure 10):** SSD is 2–8× faster than Mamba-1's fused scan at N=64. SSD crosses over FlashAttention-2 at sequence length 2K and is 6× faster at 16K.
- **Results — Language (Table 1):** Mamba-2-2.7B (300B tokens, Pile): Pile perplexity 6.09, average zero-shot 60.2. Outperforms Mamba-2.8B (6.22 ppl, 59.9 avg) and Pythia-2.8B (6.73 ppl, 55.7 avg).
- **Results — MQAR (Figure 8):** Mamba-2 with N=256 solves multi-query associative recall that Mamba-1 struggles on, validating that larger state sizes improve recall.

### Simplifications in This Notebook

- Smaller model: 2 SSD layers, 64 model dim, 64 state size, 4 heads, about 100K parameters (vs 48+ layers, 2560 dim, billions of params)
- Dataset: synthetic associative recall task at L=256 (16-token vocab, 4000 examples) instead of the Pile (300B tokens)
- 15 epochs, batch 32, lr 1e-3 with AdamW, no LR schedule (paper uses Chinchilla-optimal token counts, cosine decay)
- No tensor parallelism, no FlashAttention-style fused kernels — pure PyTorch einsum implementation of Listing 1
- No extra normalization layer (the paper's NormFormer-style stabilization is most relevant at large scale)

### What the Notebook Result Shows

On a synthetic multi-query associative recall task (sequence length 256, 16 tokens, 8 key-value pairs, 4000 training examples, 15 epochs), the SSD layer achieves validation accuracy of about 1.0, while a linear attention baseline (standard causal linear attention with cumsum mask) reaches about 0.5–0.7. The SSD layer's input-dependent decay (scalar A per step) lets it selectively remember or forget key-value pairs, while standard linear attention's fixed mask cannot.

The numerical duality check confirms the quadratic and linear forms produce identical outputs (max difference < 1e-5). This toy result is not comparable to the paper's language modeling benchmarks — the advantage of SSD's hardware efficiency is most pronounced at large state sizes (N=64–256) and long sequences (2K–16K) where the block decomposition exploits GPU matmul units.

## Influence

Mamba-2 (ICML 2024) unified the theoretical understanding of SSMs and attention, establishing that they are dual forms of the same structured matrix operation. This directly led to Mamba-3 (Dao and Gu, 2026), which further refined the architecture. The SSD framework enabled systems optimizations borrowed from Transformers: tensor parallelism, sequence parallelism, and variable-length training for Mamba-2. Hybrid architectures combining SSD layers with attention (Jamba by AI21, based on concurrent work) became a practical approach for production models. The open-source Mamba library at [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba) provides pre-trained Mamba-2 checkpoints from 130M to 2.7B parameters. Mamba-2 has about 300 citations on [Semantic Scholar](https://www.semanticscholar.org/paper/arXiv:2405.21060) (checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/018_mamba-2-structured-state-space-duality/solution.ipynb) | From-scratch PyTorch implementation of the SSD layer (Listing 1) with both quadratic and linear forms, numerical duality verification, and comparison against linear attention on synthetic associative recall |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
