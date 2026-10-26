---
title: "021 Kimi Linear"
nav_order: 22
permalink: "/021_kimi-linear-attention-architecture/"
---
# Kimi Linear: An Expressive, Efficient Attention Architecture (Kimi Linear)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/kimi-linear-attention-architecture)

- **Paper:** [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- **Authors:** Kimi Team (59 authors): Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, Wentao Li, Enzhe Lu, and 47 additional authors
- **Lab, year:** Moonshot AI, 2025 (arXiv preprint, October 2025)

## Summary

Linear attention replaces the quadratic softmax attention (the standard attention mechanism that normalises scores with exponentials) of Transformers with a fixed-size recurrent state, enabling O(1) memory per decoding step. However, linear attention has historically underperformed full attention on language modeling, retrieval, and long-context tasks due to the limited capacity of its fixed-size state. Prior work narrowed this gap through two mechanisms: gating (data-dependent decay, as in Mamba2) for rapid memory erasure, and the delta rule (as in DeltaNet) for precise key-value updates. Gated DeltaNet combined both, but used a single scalar gate per attention head — a coarse mechanism that limits how precisely the model can manage its finite-state memory.

Kimi Linear introduces **Kimi Delta Attention (KDA)**, which replaces Gated DeltaNet's head-level scalar gate with a **channel-wise (per-feature) diagonal gate** Diag(alpha_t). Each feature dimension of the key state maintains an independent forgetting rate, enabling more precise control over what to remember and what to forget. The authors derive a hardware-efficient chunkwise algorithm using a specialized variant of Diagonal-Plus-Low-Rank (DPLR) transition matrices that reduces computation by roughly 100% compared to general DPLR formulations. KDA is interleaved with full Multi-Head Latent Attention (MLA) layers in a **3:1 hybrid ratio** (3 KDA layers for every 1 full-attention layer), with NoPE (no positional encoding) on the MLA layers — KDA itself acts as a learnable positional encoding.

The authors pretrained a 48B total / 3B activated MoE model with 1.4T tokens. Kimi Linear outperforms full MLA on every evaluated task: MMLU-Pro 51.0 vs 47.2, RULER (128k) 84.3 vs 81.3, long-context average 54.5 vs 52.2, while reducing KV cache by up to 75% and achieving 6.3x decoding throughput at 1M context. This repo's notebook implements KDA from scratch and compares a 3:1 hybrid against pure full attention on associative recall, measuring both accuracy and KV cache size.

## What Problem Does It Solve

Imagine you are reading a very long book and taking notes. With full attention (the standard Transformer), you keep every page open in front of you — perfect recall, but your desk grows until there is no room left. With linear attention, you compress everything into a small notebook of fixed size — the desk never grows, but you might lose details from early chapters.

The problem is that the notebook has limited pages. When you write something new, you need to decide what to erase. Previous methods either erased entire pages at once (head-level gating, like Mamba2 and Gated DeltaNet) or updated individual entries precisely (the delta rule). Kimi Linear's insight is that you should do both at a finer grain: erase each *line* of each page independently (channel-wise gating) while also precisely updating individual entries (the delta rule). This lets the notebook hold more useful information in the same fixed space.

## Core Idea

- **Channel-wise fine-grained gating (Eq. 1):** S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T — replaces GDN's scalar alpha_t with a per-feature diagonal Diag(alpha_t), so each dimension of the key state has its own forgetting rate
- **3:1 hybrid architecture (Section 4):** Interleave 3 KDA layers with 1 full MLA layer; the full-attention layers handle global retrieval while KDA layers handle local/sequential processing efficiently
- **KDA as learnable positional encoding (Section 6.1):** NoPE on MLA layers delegates all positional information to KDA, which acts as a data-dependent multiplicative positional encoding (relaxing RoPE's orthogonality constraint)
- **Hardware-efficient DPLR variant (Section 3.1-3.2):** Binding both DPLR variables to the key vector k reduces second-level chunk computations from four to two, improving operator efficiency by ~100% over general DPLR

## Key Method Details

- **KDA state update (Eq. 1):** S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T, where Diag(alpha_t) in [0,1]^{d_k} is a per-channel gate and beta_t in [0,1] is a scalar write strength
- **Neural parameterization (Section 4):** q,k = L2Norm(Swish(ShortConv(Wx))); v = Swish(ShortConv(Wx)); alpha via low-rank projection with decay function; beta via sigmoid; output = W_o(Sigmoid(gate) * RMSNorm(KDA(q,k,v,alpha,beta)))
- **Model scale:** 48B total / 3B activated parameters, 27 layers (20 KDA + 7 MLA), 256 experts with top-8 routing, hidden dim 2304, head dim 128
- **Training:** 1.4T tokens (released checkpoint: 5.7T), 4096 context window, MuonClip optimizer, WSD learning rate schedule, lr 1.1x10^{-3}
- **Results — Short context (Table 3, 1.4T):** MMLU-Pro 51.0 (vs MLA 47.2, GDN-H 47.9); MMLU 73.8 (vs 71.6); HellaSwag 82.9 (vs 81.7)
- **Results — Long context (Table 5, 128k):** RULER 84.3 (vs MLA 81.3); RepoQA 68.5 (vs 63.0); overall avg 54.5 (vs MLA 52.2, GDN-H 51.2)
- **Results — Efficiency (Figure 1b, 7b):** KV cache reduced up to 75%; 6.3x faster TPOT at 1M context (1.84ms vs 11.48ms); 2.9x faster prefilling at 1M
- **Results — Scaling law (Figure 5):** ~1.16x compute efficiency over MLA at compute-optimal training
- **Results — RL (Figure 6):** Kimi Linear shows faster training accuracy growth and better test accuracy on AIME 2025 and MATH500 during RLVR

### Simplifications in This Notebook

- Smaller model: 4 hybrid blocks (3 KDA + 1 full attention per block = 16 layers), 64 model dim, 4 heads, 16 head dim, about 1.16M parameters (vs 27 layers, 2304 dim, 48B params)
- Dataset: synthetic MQAR (16-token vocab, 4 key-value pairs, 2000 examples, L=64) instead of 1.4T-token pretraining corpus
- 10 epochs, batch 32, lr 1e-3 with AdamW, no LR schedule (paper uses MuonClip with WSD schedule)
- Pure PyTorch recurrent form for KDA (paper: custom CUDA/Triton chunkwise kernels)
- Simplified full attention (standard multi-head, no MLA compression) as the hybrid partner
- No short convolution, no MoE, no low-rank gate parameterization (paper uses all of these)

### What the Notebook Result Shows

On a synthetic multi-query associative recall task (sequence length 64, 16-token vocab, 4 key-value pairs, 2000 training examples, 10 epochs), the 3:1 KDA-full-attention hybrid achieves validation accuracy of 1.00, pure full attention also achieves 1.00, and pure KDA (no full attention layers) reaches only 0.26. The hybrid and full-attention models both solve the task perfectly, but the KDA-only model struggles — this is consistent with the paper's finding that pure linear attention alone is insufficient for retrieval-heavy tasks (Section 4: "Long-context retrieval remains the primary bottleneck for pure linear attention").

The KV cache comparison shows a 70.3% reduction at sequence length 128 (full attention: 262,144 elements, hybrid: 77,824), approaching the paper's claimed 75% as sequence length grows. The random baseline is 1/16 = 0.0625. These toy results are not comparable to the paper's 48B-parameter benchmarks — the advantage of the hybrid is most pronounced at scale with real-world long-context tasks (Tables 3-5).

## Influence

Kimi Linear (October 2025) was the first hybrid linear attention architecture to outperform full attention under fair, controlled comparisons. It directly influenced Kimi K3 (Moonshot AI, 2025), a 2.8T parameter frontier model that uses 69 KDA layers alongside Gated MLA layers. The KDA mechanism also influenced the broader trend of 3:1 hybrid ratios in production models, with Qwen3-Next adopting a similar Gated DeltaNet hybrid pattern. The open-source release of KDA kernels, vLLM integration, and 48B model checkpoints on Hugging Face made the architecture accessible for community research. The paper has about 11 citations on [Google Scholar](https://scholar.google.com/scholar?q=%22Kimi+Linear%22+%22An+Expressive%22+attention+architecture) (checked October 2026). The official code is at [github.com/MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/021_kimi-linear-attention-architecture/solution.ipynb) | From-scratch PyTorch implementation of KDA (channel-wise gated delta rule), a 3:1 KDA-full-attention hybrid, and pure baselines on synthetic MQAR with KV cache comparison (hybrid 1.00, full-attn 1.00, KDA-only 0.26) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
