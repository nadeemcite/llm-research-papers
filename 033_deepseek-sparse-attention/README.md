---
title: "033 DeepSeek Sparse Attn"
nav_order: 34
permalink: "/033_deepseek-sparse-attention/"
---
# DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (DeepSeek Sparse Attention)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deepseek-sparse-attention)

- **Paper:** [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)
- **Authors:** DeepSeek-AI (corporate author with 200+ individual contributors listed alphabetically)
- **Lab, year:** DeepSeek-AI, 2025 (arXiv preprint, 2 December 2025)

## Summary

Standard attention in transformers compares every query token against every preceding key token — a cost that grows quadratically with sequence length. For 128K-token contexts, this becomes a major bottleneck for both training and inference. DeepSeek-V3.2 introduces DeepSeek Sparse Attention (DSA), a mechanism that selects only the most relevant tokens for each query, reducing core attention complexity from O(L²) to O(Lk) where k is the number of selected tokens (k ≪ L).

DSA works through two components: a **lightning indexer** and a **fine-grained token selection** mechanism. The lightning indexer computes lightweight importance scores I(t,s) between query token t and each preceding token s using a small number of heads with ReLU activation (chosen for throughput, and implementable in FP8). The top-k tokens with the highest scores are then retrieved for full attention.

This is instantiated under DeepSeek's Multi-head Latent Attention (MLA) in MQA mode, where each latent key-value vector is shared across all query heads. Training follows a two-stage approach: a dense warm-up that trains only the indexer with a KL-divergence loss aligning indexer scores to the main attention distribution, followed by a sparse training stage where all parameters adapt to the sparse attention pattern.

DeepSeek-V3.2 also scales post-training reinforcement learning (GRPO) with a compute budget exceeding 10% of pre-training cost, and introduces a large-scale agentic task synthesis pipeline generating over 1,800 environments and 85,000 prompts. The resulting model performs comparably to GPT-5 on reasoning benchmarks, and the high-compute variant DeepSeek-V3.2-Speciale achieves gold-medal performance at the 2025 IMO and IOI. DSA halves API costs for long-context operations compared to the predecessor DeepSeek-V3.1-Terminus.

This repo's notebook trains a tiny 4-layer transformer with DSA from scratch on character-level Wikitext-2, compares it against an equivalent dense attention model, and visualises which tokens the lightning indexer selects.

## What Problem Does It Solve

Imagine you are at a library and need to answer a question about a fact on page 300 of a 500-page book. You would not re-read every single page — you would check the index, jump to the most relevant pages, and read those carefully. DSA gives transformers this same ability: instead of attending to every previous token equally, a lightweight "indexer" scores which tokens matter most, and the model only does full attention on those top-k tokens.

Prior sparse attention methods only applied this at inference time (like skimming after the model was already trained to expect full attention), which caused performance drops. DSA trains with sparsity from the start through a two-stage warm-up then sparse-training process, so the model learns to work with it naturally.

## Core Idea

- **Lightning indexer (Section 2.1, Eq. 1):** A small, efficient module computes importance scores I(t,s) = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s) between every query-token pair. Few heads, ReLU activation, and FP8 compatibility make this much cheaper than the main attention.
- **Top-k token selection (Section 2.1, Eq. 2):** Only the key-value entries corresponding to the top-k index scores are retrieved for fine-grained attention, reducing the main attention complexity from O(L²) to O(Lk).
- **Two-stage training (Section 2.1.1):** A dense warm-up stage trains only the lightning indexer with a KL-divergence loss (Eq. 3) that aligns indexer scores to the main attention distribution. A sparse training stage then enables top-k selection and trains all parameters, with the indexer KL loss computed only over selected tokens (Eq. 4).

## Key Method Details

- **Index score (Eq. 1):** I(t,s) = Σ_{j=1}^{H_I} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s), where H_I is the number of indexer heads; q^I, w^I are derived from the query token h_t; k^I is derived from each preceding token h_s. ReLU is chosen for throughput.
- **Dense warm-up (Section 2.1.1):** Keep dense attention, freeze all model parameters except the lightning indexer. Train with KL loss L_I = Σ_t D_KL(p_{t,:} ‖ Softmax(I_{t,:})) where p is the L1-normalised sum of attention scores across heads. Learning rate 1e-3, 1000 steps of 16×128K-token sequences (2.1B tokens total).
- **Sparse training (Section 2.1.1):** Enable top-k selection (k=2048 for 128K context). KL loss computed only over selected token set (Eq. 4). Indexer input is detached from the computational graph — indexer is optimised only by L_I, main model only by language modelling loss. Learning rate 7.3×10⁻⁶, 15000 steps of 480×128K sequences (943.7B tokens total).
- **MLA instantiation (Section 2.1):** DSA is instantiated under MLA in MQA mode, where each latent key-value vector is shared across all query heads for kernel-level efficiency.
- **Complexity reduction (Section 2.3):** Main attention complexity reduced from O(L²) to O(Lk). The lightning indexer retains O(L²) but with much lower constant factors (few heads, small dimensions, FP8). Combined with optimised implementation, DSA achieves significant end-to-end speedup in long-context scenarios.
- **Benchmarks (Table 2):** DeepSeek-V3.2 vs GPT-5-High — AIME 2025: 93.1 vs 94.6; HMMT Feb 2025: 92.5 vs 88.3; LiveCodeBench: 83.3 vs 84.5; Codeforces Rating: 2386 vs 2537; SWE Verified: 73.1 vs 74.9; Terminal Bench 2.0: 46.4 vs 35.2.
- **DeepSeek-V3.2-Speciale (Table 4):** IMO 2025: 35/42 (Gold); IOI 2025: 492/600 (Gold); ICPC WF 2025: 10/12 (Gold, 2nd place); CMO 2025: 102/126 (Gold).
- **Inference costs (Figure 3, Section 2.3):** At 128K token position, DSA reduces decoding cost from ~$2.4M to ~$0.8M per million tokens on H800 GPUs (approximately 3× reduction at long contexts).

### Simplifications in This Notebook

- Tiny 4-layer transformer (~1.3M params DSA, ~0.8M params Dense, d_model 128) instead of 61-layer 671B MoE
- Character-level Wikitext-2 (~2M chars) instead of 128K-token documents on 943.7B tokens
- 20 epochs with AdamW (lr 1e-3) instead of continued pre-training with 943.7B tokens
- Pure PyTorch (no FP8, no Triton kernels); focus is accuracy and sparsity pattern, not GPU speedup
- DSA hyperparameters scaled down: k=64 selected tokens, 4 indexer heads, d_index=32 (paper: k=2048, indexer heads small, d_index small)
- 2-epoch dense warm-up (paper: 1000 steps); 18-epoch sparse training (paper: 15000 steps)
- Indexer KL loss weight 0.5 during sparse training; indexer input not detached in our implementation (we allow gradients to flow for simplicity)
- Validation perplexity only; no reasoning, agentic, or long-context benchmarks

### What the Notebook Result Shows

On a 4-layer transformer trained on character-level Wikitext-2 for 20 epochs, the DSA model (825,856 params) achieves validation perplexity of 9.34, while the Dense baseline (692,736 params) reaches 9.18 — a gap of only +0.16. The DSA model attends to only 64 of 256 tokens per query position (25% sparsity ratio) via the lightning indexer, demonstrating that the sparse selection mechanism can learn effective token importance patterns even at this small scale.

The selection-pattern visualisation shows that the indexer learns to select a mix of local (nearby) and global (distant) tokens, similar to the hierarchical patterns observed in the full-scale model. The theoretical cost analysis shows that at 128K context with k=2048, DSA achieves a substantial speedup over dense attention — consistent with the paper's reported cost reductions on H800 GPUs. At our small scale (256 tokens, k=64), the theoretical speedup is 4× for the main attention, but the paper's benefits scale with sequence length.

## Influence

DeepSeek-V3.2 was released by DeepSeek-AI in December 2025 as an open-weight model, continuing their strategy of open research. The DSA mechanism builds on their prior work on Multi-head Latent Attention (DeepSeek-V2) and Native Sparse Attention (NSA). TechCrunch covered the earlier experimental release (V3.2-Exp) in September 2025, noting that the sparse attention approach "could still teach U.S. providers some much needed tricks to help keep inference costs low."

The model is notable for achieving performance comparable to GPT-5 while being fully open-weight, and the Speciale variant's gold-medal performances at IMO and IOI 2025 set a milestone for open LLMs. The open-source implementation is available on HuggingFace (deepseek-ai/DeepSeek-V3.2-Exp). The paper has too few citations for a meaningful count at the time of writing (checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/033_deepseek-sparse-attention/solution.ipynb) | PyTorch notebook that trains a tiny 4-layer DSA transformer from scratch on character-level Wikitext-2, compares validation perplexity against an equivalent dense attention model, and visualises the lightning indexer's token selection pattern |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
