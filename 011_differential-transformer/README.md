---
title: "011 Diff Transformer"
nav_order: 12
permalink: "/011_differential-transformer/"
---
# Differential Transformer (Diff Transformer)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/differential-transformer)

- **Paper:** [Differential Transformer](https://arxiv.org/abs/2410.05258)
- **Authors:** Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei
- **Lab, year:** Microsoft Research and Tsinghua University, 2024 (ICLR 2025 Oral)

## Summary

Differential Transformer (Diff Transformer) replaces standard softmax attention (softmax turns raw scores into probabilities that sum to 1) with *differential attention*, which computes attention scores as the difference of two separate softmax attention maps. Just as noise-canceling headphones subtract a reference signal to remove background noise, the subtraction of two attention maps cancels "attention noise" — the extraneous scores a transformer over-allocates to irrelevant context. The result is sparser, cleaner attention patterns that focus more sharply on the relevant information.

The paper trains 3B-parameter Diff Transformer and Standard Transformer models on 1 trillion tokens and compares them across language modeling, long-context retrieval, hallucination evaluation, and in-context learning. Diff Transformer matches Standard Transformer quality with only about 65% of the model size or training tokens (Section 3.2). On multi-needle retrieval in 4K context, Diff Transformer scores 0.84-0.85 vs Standard's 0.55-0.62 (Table 2). On contextual hallucination evaluation, it improves accuracy by 9-19 points across summarization and QA datasets (Table 4). It also reduces activation outliers by roughly 2-10x (Table 5), potentially aiding quantization.

This repo's notebook implements differential attention from scratch in PyTorch, trains a small Diff Transformer and a Standard Transformer on Tiny Shakespeare, and visualizes the attention noise cancellation effect.

## What Problem Does It Solve

Imagine you are at a noisy party trying to listen to one friend. Your ears hear everything — music, other conversations, clinking glasses — but your brain focuses on your friend's voice and filters out the rest. Standard transformer attention is like ears without that filter: it spreads attention across everything, including irrelevant context, and only a small fraction goes to the important parts. The paper calls this "attention noise" (Section 1).

Diff Transformer solves this by borrowing from electrical engineering. Noise-canceling headphones work by capturing ambient sound with a microphone and subtracting it from what you hear. A differential amplifier cancels common-mode noise by taking the difference between two signals. Diff Transformer does the same: it computes two separate softmax attention maps and subtracts one from the other. The common-mode noise cancels out, leaving sharper focus on the relevant context.

## Core Idea

- **Differential attention (Section 2.1):** Query and key vectors are each split into two groups (Q1, Q2 and K1, K2). Two attention maps are computed: A1 = Q1 @ K1^T and A2 = Q2 @ K2^T. The final attention is `softmax(A1) - λ * softmax(A2)`, where λ is a learnable scalar. The subtraction cancels attention noise, promoting sparse attention patterns.
- **Re-parameterized λ (Eq. 2):** λ is not learned directly but re-parameterized as `exp(λq1 · λk1) - exp(λq2 · λk2) + λ_init`, where λq, λk are learnable vectors and λ_init is a per-layer constant (`0.8 - 0.6 * exp(-0.3 * (l-1))`). This stabilizes learning dynamics.
- **Per-head normalization (Section 2.1):** GroupNorm is applied to each head independently before concatenation, because differential attention heads have more diverse statistics than standard heads. A fixed multiplier `(1 - λ_init)` scales the output.

## Key Method Details

- **Architecture (Section 2.2):** Stacks L layers, each with multi-head differential attention + SwiGLU FFN, using pre-RMSNorm. The number of heads is set to `h = d_model / (2d)`, where d is the head dimension, to align parameter counts with a standard transformer (each head has 2x Q/K projections but the V dimension is 2d).
- **Training (Section 3.1):** 3B-parameter models trained on 1T tokens with AdamW (lr 3.2e-4, betas 0.9/0.95, 1000 warmup steps), sequence length 4096, batch 4M tokens, tiktoken-cl100k_base tokenizer. Follows the StableLM-3B-4ET recipe.
- **Scaling (Section 3.2, Figure 3):** Models from 830M to 13.1B parameters. Diff Transformer requires only ~65% of model size or training tokens to match Standard Transformer's language modeling loss.
- **Results (Table 1):** Diff-3B vs StableLM-3B-4ET on Eval Harness: ARC-C 37.8 vs 32.4, HellaSwag 71.4 vs 68.6, WinoGrande 67.1 vs 63.2, average 60.6 vs 56.8.
- **Multi-needle retrieval (Table 2):** In 4K context with N=4 needles, Diff scores 0.84 vs Standard 0.62. At 64K context with needles at 25% depth, Diff achieves 76% accuracy improvement over Standard.
- **Hallucination (Table 4):** Diff Transformer improves hallucination-free accuracy on XSum (0.53 vs 0.44), CNN/DM (0.41 vs 0.32), MultiNews (0.61 vs 0.42), Qasper (0.39 vs 0.28), HotpotQA (0.46 vs 0.36).
- **Activation outliers (Table 5):** Top-1 attention logit outlier drops from 318.0 (Standard) to 38.8 (Diff); top-1 hidden state outlier drops from 3608.6 to 1688.2.
- **In-context learning (Section 3.5):** Diff Transformer improves average ICL accuracy by 5.2-21.6% and is more robust to prompt order permutation.

### Simplifications in This Notebook

- Smaller model: 4 layers, 4 heads, 128 dim, about 1.06M (Standard) / 1.32M (Diff) parameters (vs 3B)
- Dataset: Tiny Shakespeare (~1 MB, character-level, vocab 65) instead of 1T tokens
- 12 epochs, 50 steps per epoch, batch 32, lr 3e-4 with AdamW (wd=0.1, betas=0.9/0.95) instead of full-scale pretraining
- No FlashAttention optimisation; RMSNorm used instead of GroupNorm for per-head normalization (simplified)
- No RoPE (relative positions learned implicitly at this scale)

### What the Notebook Result Shows

On Tiny Shakespeare validation set (character-level, 128-token context, 12 epochs), the two models achieve: Standard Transformer val perplexity 11.41 (1.06M params), Diff Transformer val perplexity 11.17 (1.32M params). The Diff Transformer shows 2.1% lower perplexity despite having more parameters from the 2x Q/K projections. The uniform random baseline is perplexity 65.0 (= vocab size).

Attention visualization confirms the paper's noise cancellation mechanism: standard attention maps are entirely non-negative (all values 0-1), while differential attention maps have both positive and negative values (the subtraction creates signed patterns). The differential maps show 55.2% sparsity (\|weight\| < 0.01) vs 52.3% for standard attention. The learned λ stays at its initialization value (0.2000 for layer 0), consistent with the paper's finding that models are robust to λ initialization.

This toy scale (~1M parameters, 1 MB text) is too small to reproduce the paper's 65% parameter efficiency claim, which was measured at 830M-13B parameters. The notebook demonstrates the mechanism (noise cancellation and sparsity) rather than the scaling advantage.

## Influence

Diff Transformer was accepted as an Oral Presentation at ICLR 2025, a distinction given to approximately 1% of submissions. The official implementation is available in the Microsoft UNILM repository ([github.com/microsoft/unilm/Diff-Transformer](https://github.com/microsoft/unilm/tree/master/Diff-Transformer)). The paper received 183 upvotes on [Hugging Face Papers](https://huggingface.co/papers/2410.05258) and 10 models on Hugging Face cite it. The noise-canceling attention idea has influenced subsequent architecture research, and the reduction in activation outliers has implications for quantization-friendly LLM design. The paper has about 23 citations ([OpenAlex](https://openalex.org/W4403324167), checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/011_differential-transformer/solution.ipynb) | From-scratch PyTorch implementation of differential attention vs standard attention, trained on Tiny Shakespeare, with attention noise visualization |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
