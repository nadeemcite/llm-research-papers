---
title: "014 Engram"
nav_order: 15
permalink: "/014_engram-conditional-memory-scalable-lookup/"
---
# Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (Engram)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/conditional-memory-via-scalable-lookup-engram)

- **Paper:** [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372)
- **Authors:** Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu, Chengqi Deng, Shangyan Zhou, Chenggang Zhao, Zhewen Hao, Yukun Li, Han Zhang, Zhengyan Zhang, Yixu Wei, M.Y Xu, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang
- **Lab, year:** Peking University & DeepSeek-AI, 2026 (preprint)

## Summary

Mixture-of-Experts (MoE, a technique where only a subset of "expert" subnetworks is activated per token) scales model capacity through conditional computation. But Transformers lack a native way to look up static knowledge — facts like entity names, idioms, and formulaic phrases that could be stored as simple table entries. Instead, the model wastes multiple layers of attention and feed-forward computation reconstructing what is essentially a lookup table entry. This paper introduces *conditional memory* as a complementary sparsity axis: instead of sparsely activating parameters for *computation* (MoE), you sparsely activate parameters for *retrieval* (memory lookup).

The authors instantiate this idea as **Engram**, a module that modernizes classic N-gram embeddings (context-to-vector tables used in pre-neural language models) with tokenizer compression, multi-head hashing, context-aware gating, and a depthwise convolution. The key innovation is an O(1) lookup: given the last N tokens, hash the N-gram into a large embedding table and retrieve a static vector, then use a learned gate to decide how much of that vector to inject into the hidden state. By formulating a "Sparsity Allocation" problem — how to split a fixed parameter budget between MoE experts and Engram memory — the authors discover a U-shaped scaling law where the optimum is roughly 75-80% MoE / 20-25% Engram.

Guided by this law, they scale Engram to 27B parameters (Engram-27B), matching the total parameter count and FLOPs (compute cost per token) of a pure MoE-27B baseline. Engram-27B outperforms the MoE baseline across knowledge tasks (MMLU +3.4, CMMLU +4.0), reasoning (BBH +5.0, ARC-Challenge +3.7), and code/math (HumanEval +3.0, MATH +2.4). Mechanistic analysis shows Engram effectively "deepens" the network by relieving early layers from static reconstruction, and boosts long-context retrieval (Multi-Query NIAH: 84.2 to 97.0). A 100B-parameter Engram table offloaded to host memory incurs under 3% inference overhead.

This repo's notebook implements a tiny language model with an Engram memory module and compares it against an iso-parameter MoE baseline on a synthetic sequence recall task.

## What Problem Does It Solve

Imagine you have a giant cookbook. Some recipes are complex — you need to think through each step, adjust seasonings, and reason about timing. But many recipes are dead simple: "boil water for pasta." If your only tool is a team of expert chefs (MoE), even the simplest task gets sent to a chef who has to think through it from scratch. That's wasteful — boiling water shouldn't require an expert.

Engram adds a lookup card next to the chefs. When the task is "boil water," instead of summoning a chef, you flip to the card that says "boil water = 100°C, 10 minutes" and you're done in zero thought. The chefs are freed to handle the complex reasoning tasks. The paper's insight is that language has the same structure: much of text consists of static, local patterns (named entities, idioms, formulaic phrases) that are better served by a fast lookup than by deep neural computation. By giving the model both a memory card (Engram) and expert chefs (MoE), and finding the right balance between them, you get better performance at the same parameter and compute budget.

## Core Idea

- **Conditional memory as a new sparsity axis (Section 1):** MoE provides *conditional computation* (sparsely activate parameters for processing). Engram provides *conditional memory* (sparsely activate parameters for retrieval). The two are structurally complementary — memory handles static local patterns, computation handles dynamic reasoning.
- **Hashed N-gram embedding with O(1) lookup (Section 2.2):** Given the last N tokens, form an N-gram, compress the vocabulary via a surjective mapping, and hash the N-gram into a large embedding table using K independent hash heads. Each head retrieves a vector; all are concatenated into the memory embedding. The hash is deterministic and multiplicative-XOR based.
- **Context-aware gating (Section 2.3):** The retrieved memory is a context-independent prior. A learned gate, computed from the dot product of the current hidden state (query) and a key projection of the memory (both RMSNormed), scales the memory's value projection. If the memory contradicts the context, the gate suppresses it. A short depthwise causal convolution expands the receptive field.
- **U-shaped sparsity allocation law (Section 3.1):** Under a fixed parameter budget, splitting sparse capacity between MoE and Engram follows a U-shaped curve. Pure MoE (ρ=100%) is suboptimal — reallocating ~20-25% to Engram yields the best validation loss. The optimum (ρ≈75-80%) is stable across compute regimes.

## Key Method Details

- **Tokenizer compression (Section 2.2):** A surjective function P: V→V' collapses raw token IDs to canonical IDs (NFKC normalization, lowercasing), achieving 23% vocabulary reduction for a 128k tokenizer. This maximizes semantic density in the N-gram table.
- **Multi-head hashing (Eq. 1-2):** K hash heads per N-gram order n, each mapping to an embedding table E_{n,k} of prime size M_{n,k}. Final memory vector e_t = concatenation of all retrieved embeddings across orders n=2..N and heads k=1..K.
- **Context-aware gating (Eq. 3-5):** k_t = W_K e_t, v_t = W_V e_t. Gate α_t = σ(RMSNorm(h_t)^T RMSNorm(k_t) / √d). Gated output ṽ_t = α_t · v_t. Final output Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ. Integrated via residual: H^(ℓ) ← H^(ℓ) + Y.
- **Sparsity allocation (Eq. 7):** P_MoE^(sparse) = ρ · P_sparse, P_Engram = (1-ρ) · P_sparse. U-shaped law: optimum at ρ≈75-80%. At C=6×10^20 FLOPs, validation loss improves from 1.7248 (pure MoE) to 1.7109 (optimal hybrid).
- **Engram-27B config (Section 4.1):** 30-block Transformer, hidden 2560, 32 MLA heads, MoE with 2 shared + 55 routed experts (top-6), Engram at layers 2 and 15, N-gram order 3, 8 hash heads, d_mem=1280, 5.7B Engram params. Trained on 262B tokens with Muon optimizer.
- **Results (Table 1):** Engram-27B vs MoE-27B: MMLU 60.4 vs 57.4 (+3.0), BBH 55.9 vs 50.9 (+5.0), HumanEval 40.8 vs 37.8 (+3.0), MATH 30.7 vs 28.3 (+2.4), ARC-Challenge 73.8 vs 70.1 (+3.7).
- **Long context (Table 2):** Multi-Query NIAH 97.0 vs 84.2, Variable Tracking 89.0 vs 77.0 (iso-loss setting).
- **System efficiency (Table 4):** 100B-parameter Engram table offloaded to host DRAM incurs ≤2.8% throughput penalty on NVIDIA H800.

### Simplifications in This Notebook

- Smaller model: 4 layers, 64 dim, 4 heads, about 0.47M parameters (vs 27B Engram-27B)
- Dataset: synthetic bigram recall task (128-token vocab, 32 bigram patterns, 64-token sequences) instead of 262B-token pretraining
- 20 epochs with AdamW (lr 1e-3, batch 128) instead of Muon optimizer on large-scale data
- Baseline uses wider FFN (mult=16) to match Engram total params, instead of the paper's MoE expert reallocation
- N-gram order 2 (bigrams only), 4 hash heads, d_mem=32 (vs order 3, 8 heads, d_mem=1280)
- No tokenizer compression, no multi-branch architecture, no system-level prefetching
- Simple multiplicative-XOR hash (4 heads, 1,009 slots) instead of the paper's millions of slots

### What the Notebook Result Shows

On a synthetic bigram recall task (64-token sequences over a 128-token vocabulary with 32 repeated bigram patterns, 20 epochs), the Engram-augmented model achieves test loss 3.78-3.81 (464,832 params) vs the iso-parameter wide-FFN baseline's test loss 5.00 (467,520 params) — a 23.8-24.4% relative improvement at matched parameter count.

The Engram model stores bigram embeddings in a hash table (1,009 slots, 4 heads, d_mem=32) and retrieves them via deterministic lookup, then gates the retrieved vector with the current hidden state before injecting it into the residual stream. The baseline has a wider FFN (mult=16 vs 4) but no memory table, so it must learn the same bigram associations purely through computation.

The improvement is consistent with the paper's finding that explicit memory lookup outperforms computation-only for static local patterns. The gating visualization shows the Engram gate activating selectively (mean α=0.28-0.29 at layer 1, with 14-15/63 positions above 0.5), confirming the module learns to retrieve memory for static patterns rather than for every token.

## Influence

Engram is a January 2026 preprint from Peking University and DeepSeek-AI, the team behind DeepSeek-V3. It is part of a resurgent interest in N-gram embedding scaling, alongside OverEncoding (Huang et al., 2025), SCONE (Yu et al., 2025), and Byte Latent Transformer (Pagnoni et al., 2025). Engram's key contribution over these works is treating memory as a first-class sparsity primitive with rigorous iso-parameter/iso-FLOPs comparison against MoE, plus the algorithm-system co-design for host-memory offloading. The concept of conditional memory as a complement to conditional computation is likely to influence next-generation sparse model architectures. The paper has a public code release at https://github.com/deepseek-ai/Engram. Citation count is not yet available on OpenAlex as of October 2026, reflecting the paper's recency.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/014_engram-conditional-memory-scalable-lookup/solution.ipynb) | From-scratch PyTorch implementation of Engram's hashed N-gram memory with context-aware gating, trained on a synthetic bigram recall task and compared against an iso-parameter wide-FFN baseline |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Q&A answered from the paper, citations, and common misconceptions about the paper |
