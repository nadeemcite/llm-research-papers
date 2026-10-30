---
title: "032 NSA"
nav_order: 33
permalink: "/032_native-sparse-attention/"
---
# Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (NSA)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/native-sparse-attention)

- **Paper:** [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)
- **Authors:** Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, Wangding Zeng
- **Lab, year:** DeepSeek-AI (most authors) / Peking University, Key Laboratory for Multimedia Information Processing, PKU-Anker LLM Lab (J. Yuan, J. Luo, M. Zhang) / University of Washington (Z. Xiao), 2025 (arXiv preprint)

## Summary

Standard attention in transformers computes relevance scores between every query token and every preceding key token — a cost that grows quadratically with sequence length. For 64k-token contexts, attention can account for 70–80% of total decoding latency. NSA (Native Sparse Attention) addresses this with a three-branch hierarchical design that reduces computation while being natively trainable end-to-end, not just applied at inference time.

NSA processes keys and values through three parallel branches: (1) **compression** — a learnable MLP aggregates sequential blocks of tokens into single compressed representations, giving coarse-grained global context at low cost; (2) **selection** — importance scores derived from the compression branch identify the top-n most relevant token blocks, which are attended to at full granularity; (3) **sliding window** — a fixed local window captures nearby context. A learned gate (a small MLP with sigmoid output) combines the three branch outputs, allowing the model to dynamically weight global, fine-grained, and local information per position. Each branch uses independent key and value projections to prevent gradient interference.

On a 27B-parameter MoE backbone (3B active) pretrained on 270B tokens, NSA matches or exceeds Full Attention across general benchmarks (avg 0.456 vs 0.443), long-context tasks (LongBench avg 0.469 vs 0.437), and chain-of-thought reasoning (AIME 24: 0.146 vs 0.092 at 16k context). On 64k-length sequences, NSA achieves 11.6× decoding, 9.0× forward, and 6.0× backward speedup over FlashAttention-2. Crucially, these gains come from training with NSA from scratch — not applying sparsity post-hoc to a pretrained Full Attention model.

This repo's notebook trains a tiny 4-layer transformer with NSA attention from scratch on character-level Wikitext-2, compares it against an equivalent Full Attention model, and shows that NSA can learn effectively despite attending to fewer tokens.

## What Problem Does It Solve

Imagine reading a 500-page book and needing to answer a question about a detail on page 300. You wouldn't re-read all 500 pages — you'd skim chapter summaries (compression), jump to the pages that seem most relevant (selection), and carefully read the surrounding paragraphs (sliding window). NSA gives transformers this same ability: instead of attending to every previous token equally, it uses three strategies to focus on what matters. Prior sparse attention methods only did this at inference time (like skimming after already reading everything), which caused performance drops because the model was trained to expect full attention. NSA trains with sparsity from the start, so the model learns to work with it naturally.

## Core Idea

- **Hierarchical three-branch sparse attention (Section 3.3):** NSA replaces full attention with three parallel branches — compression (coarse-grained block summaries), selection (top-n fine-grained blocks), and sliding window (local context) — combined by a learned gate.
- **Native trainability (Section 3.2):** Unlike prior sparse methods applied post-hoc, NSA is trained end-to-end from scratch. The compression MLP and gate are differentiable, and block selection uses importance scores from the compression branch (Eq. 8), enabling gradients to flow through the sparse pattern.
- **Hardware-aligned design (Section 3.4):** Blockwise selection ensures contiguous memory access compatible with Tensor Cores and FlashAttention-style kernels. The Triton kernel loads all heads in a GQA group together (sharing the same selected blocks), eliminating redundant KV transfers.

## Key Method Details

- **Compression (Section 3.3.1):** Keys and values are split into blocks of length l with stride d (d < l to avoid fragmentation). A learnable MLP φ maps each block of keys to a single compressed key. Values are aggregated (mean-pooled in our implementation). Paper uses l=32, d=16.
- **Selection (Section 3.3.2):** Importance scores for selection blocks come from the compression branch's attention scores (Eq. 8: p_cmp = Softmax(q^T K_cmp)). For GQA, scores are summed across heads in a group (Eq. 10) to ensure consistent block selection. The top-n blocks (paper: n=16, l'=64) are retained for fine-grained attention.
- **Sliding window (Section 3.3.3):** A fixed window of w recent tokens (paper: w=512) handles local context. Independent K/V projections for each branch prevent shortcut learning where local patterns dominate.
- **Gating (Eq. 5):** o* = Σ_c g_c · Attn(q, K̃_c, Ṽ_c), where g_c ∈ [0,1] comes from an MLP + sigmoid on the input features. This lets the model dynamically weight branches per position.
- **Architecture:** 27B total / 3B active MoE, 30 layers, d=2560, GQA (4 groups, 64 heads), DeepSeekMoE (72 routed + 2 shared experts, top-6). Trained on 270B tokens of 8k texts, then continued on 32k with YaRN.
- **General benchmarks (Table 1):** NSA avg 0.456 vs Full Attention avg 0.443 across MMLU, MMLU-PRO, CMMLU, BBH, GSM8K, MATH, DROP, MBPP, HumanEval. NSA wins on 7/9 metrics.
- **Long-context (Table 2):** LongBench avg: NSA 0.469, Full Attention 0.437, Exact-Top 0.423. NSA wins on multi-hop QA (HPQ +0.087, 2Wiki +0.051) and code (LCC +0.069).
- **Reasoning (Table 3):** AIME 24 after SFT on 10B R1 tokens: NSA-R 0.146 vs Full-R 0.092 at 16k context.
- **Speed (Figures 1, 6; Table 4):** At 64k context: 11.6× decoding, 9.0× forward, 6.0× backward. KV cache loading reduced from 65536 to 5632 tokens.

### Simplifications in This Notebook

- Tiny 4-layer transformer (~795K params Full, ~1.24M params NSA, d_model 128) instead of 27B MoE
- Character-level Wikitext-2 (~2M chars) instead of 270B tokens
- 15 epochs with AdamW (lr 1e-3) instead of large-scale pretraining
- Pure PyTorch (no Triton kernels); focus is accuracy, not GPU speedup
- Validation perplexity only; no general, long-context, or reasoning benchmarks
- NSA hyperparameters scaled down: l=16, d=8, l'=16, n=8, w=64 (paper: l=32, d=16, l'=64, n=16, w=512)
- NSA has 55.9% more parameters than Full due to 3× K/V projections + compression MLP + gate MLP

### What the Notebook Result Shows

On a 4-layer transformer trained on character-level Wikitext-2 for 15 epochs, the NSA model (~1,238,924 params) achieves validation perplexity of 1.02, while the Full Attention model (~794,880 params) reaches 4.01. The perplexity gap of -2.99 (NSA lower) is largely explained by NSA having 55.9% more parameters from its three independent K/V projection sets, compression MLP, and gate MLP — not by the sparse attention mechanism itself being inherently better at this scale.

The sparsity analysis shows NSA attends to at most 223 tokens per query position (31 compressed + 128 selected + 64 window) versus 256 for full attention — an 87.1% sparsity ratio at seq_len=256, giving only a 1.1× theoretical speedup at this short sequence length. The paper's 11.6× speedup requires 64k-length sequences where the sparsity ratio drops much lower. The average gate weights (Compression=0.967, Selection=0.255, Window=0.958) show the model uses all three branches, with compression and window receiving the highest weights. The key finding — NSA can learn effectively while attending to fewer tokens — is visible, but the efficiency advantage only materializes at much larger scale.

## Influence

NSA was published by DeepSeek-AI in February 2025 and quickly became one of the most discussed sparse attention papers. It is notable for being the first sparse attention method that is both natively trainable (not applied post-hoc) and hardware-aligned (using blockwise selection compatible with FlashAttention-style kernels). The paper influenced subsequent work on trainable sparse attention and long-context efficiency, and its blockwise selection strategy and gating mechanism have been referenced in follow-up research on efficient attention. The official code was released by DeepSeek-AI on GitHub. The paper has about 4 citations on [OpenAlex](https://openalex.org/W4407690159) (checked October 2026), though the arXiv listing shows broader influence through its integration into DeepSeek's model development pipeline.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/032_native-sparse-attention/solution.ipynb) | PyTorch notebook that trains a tiny 4-layer NSA transformer from scratch on character-level Wikitext-2 and compares validation perplexity against an equivalent Full Attention model |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
