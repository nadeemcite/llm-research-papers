---
title: "002 ALiBi"
nav_order: 3
permalink: "/002_alibi-attention-with-linear-biases/"
---
# Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/alibi-attention-with-linear-biases)

- **Paper:** [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)
- **Authors:** Ofir Press, Noah A. Smith, Mike Lewis
- **Lab, year:** University of Washington (Allen School) & Allen Institute for AI (Smith); Facebook AI Research (Lewis), 2021 (ICLR 2022)

## Summary

Transformers need to know the position of each token to process text, but most position methods only work well at the sequence length the model was trained on. If you train on 512-token sequences and then try to use the model on 2048 tokens, the perplexity (a measure of how surprised the model is by the next token — lower is better) gets worse, not better, even though more context should help. The paper shows this is true for sinusoidal position embeddings (the original Transformer method), rotary embeddings (RoPE), and the learned T5 relative bias — none of them extrapolate efficiently.

ALiBi (Attention with Linear Biases) takes a radically simpler approach: it removes position embeddings entirely. Instead, it adds a static, non-learned penalty to each attention score that grows linearly with the distance between the query and key. Closer tokens get a smaller penalty; distant tokens get a larger one. The penalty rate (slope) is different for each attention head, forming a geometric sequence. This gives the model an inductive bias towards recency — it naturally attends more to nearby tokens — without learning any position-specific parameters.

The paper's headline result: a 1.3 billion parameter model trained on 1024-token sequences with ALiBi extrapolates to 2048 tokens, achieving the same perplexity as a sinusoidal model trained on 2048 tokens — but 11% faster and using 11% less memory. ALiBi also outperforms sinusoidal, rotary, and T5 bias methods on WikiText-103, and its best performance at inference is around 2× the training length, though it maintains strong performance even at 10,000 tokens.

This repo's notebook implements ALiBi in a small character-level Transformer and demonstrates its extrapolation advantage: a model trained on 256-token sequences maintains 99.89% next-token accuracy at 1024 tokens (4× training length), while the same model with sinusoidal position embeddings drops to 71.47%.

## What Problem Does It Solve

Imagine training a chef to cook meals for 4 people. They get great at it. But then you ask them to cook for 16 people — and they fall apart, even though cooking for more people should be *easier* (more ingredients to work with, more flexibility). That's what happens to Transformers with standard position embeddings: train on 512 tokens, test on 2048, and performance gets *worse*, not better.

The problem is that standard position methods (sinusoidal, learned) encode each position as a specific pattern. At training time, the model only sees positions 1 through 512. At test time, positions 513 through 2048 are unfamiliar territory — the model has no idea how to handle them. ALiBi's insight: don't give each position a unique code at all. Instead, just penalize attention to distant tokens with a simple linear ramp. Since the penalty depends only on *distance* (not absolute position), it works identically at any sequence length.

## Core Idea

- **No position embeddings:** ALiBi removes all position embeddings from the network. Position information is injected only at the attention score level, not at the input.
- **Linear distance penalty:** After computing the query-key dot product, a static bias $m \cdot (j - i)$ is added, where $m$ is a head-specific slope and $(j - i)$ is the key-query distance. This penalizes attention to distant tokens.
- **Geometric slope schedule:** The slopes form a geometric sequence: for $n$ heads, they start at $2^{-8/n}$ with ratio $2^{-8/n}$. For 8 heads: $[\frac{1}{2^1}, \frac{1}{2^2}, ..., \frac{1}{2^8}]$. Different heads attend at different ranges — some focus on nearby tokens, others look farther back.
- **Non-learned and zero-cost:** The slopes are fixed before training and never updated. ALiBi adds no parameters and has negligible runtime cost (the bias is added to the existing causal mask).

## Key Method Details

- **ALiBi attention (Section 3):** $\text{softmax}(\mathbf{q}_i \mathbf{K}^\top + m \cdot [-(i-1), ..., -2, -1, 0])$, where $m$ is a head-specific scalar fixed before training. The bias is not multiplied by the $\sqrt{d_k}$ scaling factor.
- **Slopes (Section 3):** For $n$ heads, the slopes form a geometric sequence starting at $2^{-8/n}$ with ratio $2^{-8/n}$. For 8 heads: $[1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]$. For 16 heads, the 8 slopes are interpolated by geometrically averaging consecutive pairs.
- **Results (Section 4.1, WikiText-103):** ALiBi trained on $L=512$ achieves 18.40 perplexity when extrapolating to $L_{\text{valid}}=3072$, surpassing the sinusoidal model trained on $L=3072$ (18.67 perplexity) — while being 1.84× faster to train. ALiBi at $L=3072$ achieves 17.60 perplexity vs sinusoidal 18.67.
- **Results (Section 4.2, 1.3B model on CC100+RoBERTa):** ALiBi trained on $L=1024$ outperforms sinusoidal trained on $L=2048$ by 0.09 perplexity when both are evaluated at $L_{\text{valid}}=2048$, using 3.1 GB less memory and 11% less training time.
- **Extrapolation range (Figure 6):** ALiBi's best performance is around $2L$; the $L=512$ model peaks at 1012 tokens (9.3 perplexity), the $L=1024$ model peaks at 2024 tokens (8.9 perplexity). Performance remains strong up to 10,000 tokens.

### Simplifications in This Notebook

- Tiny model: 4 Transformer layers, 4 heads, 128-dim (793,856 params for ALiBi, 933,120 for Sinusoidal — the extra 139,264 are the position embedding table) instead of 16-layer / 25-layer billion-parameter models
- Synthetic counting sequences (token at position $t = (a \cdot t + b) \bmod 20$) instead of WikiText-103 (103M tokens) or CC100+RoBERTa (461 GB)
- 40 epochs with AdamW (lr 3e-4), batch size 128, on 2048 sequences — instead of 205 epochs on 103M tokens or 50k updates on 128 V100 GPUs
- Next-token prediction (causal language modeling) on a 20-token vocabulary, instead of a 267K-token BPE vocabulary
- Extrapolation tested at 4× training length (256 → 1024) instead of 2× (1024 → 2048)

### What the Notebook Result Shows

On 256 unseen test sequences, both models achieve high accuracy at the training length (256): Sinusoidal 98.62%, ALiBi 99.59%. At the extrapolation length (1024, 4× training), ALiBi maintains 99.89% accuracy (loss 0.004, perplexity 1.00) while Sinusoidal drops to 71.75% (loss 2.643, perplexity 14.06) — a 28.1 percentage-point gap. Restricted to the extrapolated positions 256–1023 only, Sinusoidal scores 62.85% accuracy (loss 3.504) and ALiBi 100.00% (loss 0.0001).

Sinusoidal's position embeddings for positions 256–1023 are never trained (they were initialized with sinusoidal values but never updated by gradients), so this is the expected failure mode. ALiBi has no position embeddings at all — its distance-based bias applies identically at any length. The counting task is deterministic (given the slope and offset, the next token is fully determined), so a model that captures the pattern can achieve near-perfect accuracy; the difference between the two models is whether the position representation allows the pattern to be applied at unseen lengths.

## Influence

ALiBi was adopted by several open-source language models. Notably, MPT (MosaicML, 2023) used ALiBi for its position encoding, enabling context lengths up to 65K tokens. BloombergGPT (2023) also employed ALiBi. The paper's emphasis on extrapolation influenced later context-extension work like YaRN (2023) and Position Interpolation, though those methods took different approaches (scaling RoPE's rotation frequencies rather than adding biases). The official code is on [GitHub (ofirpress/attention_with_linear_biases)](https://github.com/ofirpress/attention_with_linear_biases). It has about 161 citations ([OpenAlex](https://openalex.org/works/W3196318247), checked October 2026). RoPE ultimately became more widely adopted (by LLaMA, Mistral, Qwen, and others), but ALiBi's key insight — that position representation determines extrapolation ability — remains influential.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/002_alibi-attention-with-linear-biases/solution.ipynb) | From-scratch PyTorch nanoGPT with ALiBi vs sinusoidal position embedding, trained on synthetic counting sequences to compare length extrapolation (ALiBi 99.89% vs Sinusoidal 71.75% at 4x training length, 100.00% vs 62.85% on the unseen positions 256-1023 only) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
