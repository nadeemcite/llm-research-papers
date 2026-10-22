---
title: "012 Gated Attention"
nav_order: 13
permalink: "/012_gated-attention-llm/"
---
# Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free (Gated Attention)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/gated-attention-for-large-language-models)

- **Paper:** [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free](https://arxiv.org/abs/2505.06708)
- **Authors:** Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin
- **Lab, year:** Qwen Team, Alibaba Group (with University of Edinburgh, Stanford, MIT, Tsinghua), 2025 (NeurIPS 2025 Oral)

## Summary

Gated Attention adds a simple sigmoid gate (sigmoid squashes values into the 0-1 range) after the standard attention computation in a transformer. The gate is a small learned linear layer that takes the model's input and produces per-head, per-dimension scores. Each score multiplies the corresponding attention output, selectively amplifying or suppressing features. The modification adds less than 2% latency overhead and fewer than 201M parameters on a 15B model.

The paper trains 15B Mixture-of-Experts (MoE, a model where different experts handle different tokens) and 1.7B dense models on 3.5 trillion tokens. Gated Attention consistently improves performance: on the 15B MoE, average perplexity (a measure of how well a language model predicts text; lower is better) drops from 6.026 to 5.761, with MMLU improving from 58.79 to 60.82 (Table 1). The gate also enhances training stability — gated models tolerate a 2x learning rate without divergence (Table 2) — and eliminates the "attention sink" phenomenon where early tokens hijack a disproportionate share of attention (first-token attention drops from 46.7% to 4.8%, Figure 2). This attention-sink-free property improves long-context extrapolation by over 10 points on the RULER benchmark (Section 4.4).

This repo's notebook implements gated attention from scratch in PyTorch, trains a small baseline and gated transformer on Tiny Shakespeare, and measures the attention sink and gating score sparsity.

## What Problem Does It Solve

Imagine reading a long book and finding that no matter which page you turn to, your eyes keep drifting back to the first sentence. That is essentially what happens inside standard transformers: a phenomenon called "attention sink" where the first few tokens of any sequence absorb a huge fraction of the model's attention budget, even when they are irrelevant to the current position. The paper reports that on average 46.7% of attention scores across layers go to the first token (Figure 2).

Gated Attention solves this by adding a filter after the attention computation. Think of it as a smart dimmer switch for each feature: the gate learns to turn down the volume on attention outputs that are not useful for the current query, effectively zeroing out the noise that feeds the attention sink. Because the gate is query-dependent (it looks at the current input to decide what to suppress), it can filter differently for each token position, producing sparse, focused attention patterns.

## Core Idea

- **Head-specific sigmoid gate after SDPA (Section 2.2):** After computing Scaled Dot-Product Attention (SDPA, the standard attention mechanism) for each head, the output is multiplied by `sigmoid(XWθ)`, where X is the attention layer's input and Wθ is a small learnable weight matrix. Each head gets its own gate, enabling independent modulation.
- **Non-linearity breaks low-rank bottleneck (Section 4.1):** In multi-head attention, the value and output projections form a low-rank linear mapping (since dk < d_model). Inserting a non-linear sigmoid between them increases expressiveness, analogous to how adding non-linearity between two linear layers makes a neural network more powerful.
- **Sparse gating eliminates attention sink (Section 4.2-4.3):** The sigmoid gate produces scores concentrated near 0 (mean 0.116 for the best variant), creating input-dependent sparsity. This sparsity filters out irrelevant context, reducing first-token attention from 46.7% to 4.8% and massive hidden-state activations by 10x.

## Key Method Details

- **Gating formula (Eq. 5):** `Y' = Y ⊙ σ(XWθ)`, where Y is the SDPA output, X is the attention input, Wθ maps from d_model to dk per head, σ is sigmoid, and ⊙ is element-wise multiplication.
- **Best configuration (Table 1):** SDPA elementwise G1 with sigmoid activation, head-specific gating. Score shape: n × q × dk. Added parameters: 201M on 15B MoE.
- **Training (Section 3.1):** 15A2B MoE (15B total, 2.54B activated, 128 experts, top-8) and 1.7B dense models on 3.5T tokens. Context length 4096, AdamW optimizer. GQA for attention.
- **Results — 15B MoE (Table 1):** PPL 5.761 vs 6.026 baseline, HellaSwag 74.64 vs 73.07, MMLU 60.82 vs 58.79, GSM8k 55.27 vs 52.92, C-eval 62.20 vs 60.26.
- **Results — 1.7B dense (Table 2):** PPL 6.130 vs 6.180 baseline (3.5T tokens), HumanEval 37.80 vs 34.15, GSM8k 70.20 vs 69.07.
- **Training stability (Table 2):** At 2x learning rate (8e-3 vs 4e-3), baseline diverges; gated model trains stably with PPL 7.325.
- **Attention sink (Figure 2, Table 4):** First-token attention drops from 46.7% to 4.8%. Massive activations (M-Act) drop from 1053 to 94.
- **Long-context (Table 5):** On RULER at 128K context (YaRN extended), gated model scores 51.56 vs baseline 37.94 — a 13.6-point gain.

### Simplifications in This Notebook

- Smaller model: 4 layers, 4 heads, 128 dim, about 1.06M (baseline) / 1.12M (gated) parameters (vs 15B/1.7B)
- Dataset: Tiny Shakespeare (~1 MB, character-level, vocab 65) instead of 3.5T tokens
- 12 epochs, 50 steps per epoch, batch 32, lr 3e-4 with AdamW (wd=0.1, betas=0.9/0.95) instead of full-scale pretraining
- No GQA, MoE, or YaRN context extension — just a vanilla transformer with and without the gate
- No FlashAttention; RoPE omitted (relative positions learned implicitly at this scale)

### What the Notebook Result Shows

On Tiny Shakespeare validation set (character-level, 128-token context, 12 epochs), the two models achieve: Baseline Transformer val perplexity 11.58 (1,058,048 params), Gated Attention Transformer val perplexity 11.50 (1,123,584 params). The gated model shows 0.8% lower perplexity with 6.2% more parameters from the gate weights.

At this toy scale (~1M parameters, 1 MB text, 12 epochs), the attention sink phenomenon does not emerge — neither model develops strong first-token attention (baseline 3.1%, gated 3.2%). The paper's 46.7% attention sink was observed in a 1.7B model trained on 3.5T tokens. Similarly, the gating scores are not yet sparse at this scale (mean 0.587, vs the paper's 0.116), because the sparsity emerges from large-scale training. The notebook demonstrates the architecture and measurement methodology rather than reproducing the scaling-dependent phenomena.

## Influence

Gated Attention was accepted as an Oral Presentation at NeurIPS 2025, a distinction given to approximately 1% of submissions. The official implementation is available at [github.com/qiuzh20/gated_attention](https://github.com/qiuzh20/gated_attention) (988 stars as of October 2026). The paper has about 23 citations ([OpenAlex](https://openalex.org/works/W4403324167), checked October 2026). The finding that a simple post-SDPA gate eliminates attention sinks has influenced subsequent architecture research, particularly in long-context model design and training stability. The work builds on prior gating concepts from LSTMs, Highway Networks, and SwiGLU, and connects to the attention sink literature (StreamingLLM, Xiao et al. 2023).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/012_gated-attention-llm/solution.ipynb) | From-scratch PyTorch implementation of gated attention vs standard attention, trained on Tiny Shakespeare, with attention sink measurement and gating score analysis |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
