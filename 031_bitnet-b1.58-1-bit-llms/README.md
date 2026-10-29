---
title: "031 BitNet b1.58"
nav_order: 32
permalink: "/031_bitnet-b1.58-1-bit-llms/"
---
# The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/bitnet-b158-1-bit-llms)

- **Paper:** [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- **Authors:** Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei
- **Lab, year:** Microsoft Research (S. Ma, L. Ma, L. Wang, W. Wang, S. Huang, L. Dong, J. Xue, F. Wei) / University of Chinese Academy of Sciences (H. Wang, R. Wang), 2024 (arXiv preprint, "Work in progress")

## Summary

Large language models are typically stored in 16-bit floating point (FP16 or BF16), meaning every weight is one of 65,536 possible values. This makes models enormous — a 3B-parameter model needs about 6 GB of memory just for weights. Post-training quantization (compressing weights to 4 bits after training) helps but is suboptimal because the model was trained at full precision and then degraded. BitNet b1.58 takes a radically different approach: train the model from scratch with every weight constrained to just three values — {-1, 0, +1} — which is log₂(3) ≈ 1.58 bits per weight.

The key insight is that ternary weights {-1, 0, +1} enable a fundamentally different computation paradigm. Instead of floating-point multiply-accumulate operations (the dominant cost in LLM inference), matrix multiplication with ternary weights becomes integer addition: multiply by +1 (add), by -1 (subtract), or by 0 (skip). The zero value also provides explicit feature filtering — a weight of zero means a connection is simply absent, which improves modeling capability compared to the original binary {-1, +1} BitNet. BitNet b1.58 adopts LLaMA-like architecture components (RMSNorm, SwiGLU, rotary embeddings, no biases) so it can integrate into existing open-source frameworks.

On 100B tokens of RedPajama, BitNet b1.58 matches the FP16 LLaMA LLM at 3B size in both perplexity (9.91 vs 10.04) and zero-shot task accuracy (50.2 vs 49.7 average), while using 3.55× less GPU memory and 2.71× lower latency. At 70B, it is 4.1× faster and supports 11× larger batch sizes (8.9× higher throughput). The paper estimates 71.4× energy savings for matrix multiplication on 7nm chips.

This repo's notebook trains a tiny ternary-weight transformer with BitLinear layers from scratch on word-level Wikitext-2, compares it against an equivalent FP16 model, and shows that ternary weights can learn despite a significant perplexity gap at this small scale.

## What Problem Does It Solve

Imagine packing for a trip where every item must fit in a tiny suitcase. Normally you use full-sized everything (16-bit floats — 65,536 possible values per weight). You could shrink items after packing (post-training quantization), but things get wrinkled and damaged. BitNet b1.58 instead designs the entire trip around miniature items from the start — each weight is just {-1, 0, +1}, three values instead of 65,536. The 0 is key: it means "don't pack this at all," giving explicit feature filtering that binary {-1, +1} weights lacked. Remarkably, with enough parameters (3B+), this minimalist packing works as well as the full-size version, while being dramatically cheaper to transport (run inference).

## Core Idea

- **Ternary weight quantization (Section 2, Eq. 1–3):** Every weight is constrained to {-1, 0, +1} using an absmean quantization function: scale the weight matrix by its average absolute value γ, then round each value to the nearest integer in {-1, 0, +1} with clipping. This gives log₂(3) ≈ 1.58 bits per weight.
- **Activation quantization to 8-bit (Section 2):** Activations are scaled per token to [-Qb, Qb] (where Qb = 2^(b-1) for b=8 bits) without zero-point quantization, simplifying implementation. Matrix multiplication with ternary weights and 8-bit activations uses integer addition instead of floating-point multiplication.
- **LLaMA-alike architecture (Section 2):** Uses RMSNorm, SwiGLU, rotary embeddings (RoPE), and removes all biases, enabling integration with HuggingFace, vLLM, and llama.cpp. BitLinear replaces nn.Linear everywhere.

## Key Method Details

- **Weight quantization (Eq. 1–3):** W̃ = RoundClip(W / (γ + ε), -1, +1), where γ = (1/nm) Σ\|W_ij\| is the mean absolute weight, ε is a small constant for numerical stability, and RoundClip(x, a, b) = max(a, min(b, round(x))).
- **Activation quantization:** Activations scaled per token to [-Qb, Qb] where Qb = 2^(b-1). No scaling before nonlinear functions (differs from original BitNet). The scale is the per-token absmax.
- **Architecture:** LLaMA-like — RMSNorm, SwiGLU FFN, RoPE, no biases. All nn.Linear replaced with BitLinear (quantize weights to ternary, quantize activations to 8-bit, then matmul).
- **Training:** Trained from scratch on 100B tokens of RedPajama. Models from 700M to 3.9B.
- **Key result — Perplexity (Table 1):** At 3B: BitNet b1.58 PPL = 9.91 vs LLaMA FP16 PPL = 10.04 (BitNet wins). At 700M: 12.87 vs 12.33 (LLaMA wins). At 1.3B: 11.29 vs 11.25 (roughly tied). BitNet b1.58 3.9B: PPL = 9.62.
- **Key result — Memory & latency (Table 1):** At 3B: 2.22 GB vs 7.89 GB (3.55× less), 1.87 ms vs 5.07 ms (2.71× faster). At 70B: 4.1× faster, 11× batch size, 8.9× throughput.
- **Key result — Zero-shot accuracy (Table 2):** At 3B: BitNet b1.58 avg = 50.2 vs LLaMA avg = 49.7. Tasks: ARC-Easy (61.4 vs 62.1), ARC-Challenge (28.3 vs 25.6), HellaSwag (42.9 vs 43.3), BoolQ (61.5 vs 61.8), OpenBookQA (26.6 vs 24.6), PIQA (71.5 vs 72.1), Winogrande (59.3 vs 58.2).
- **Key result — 2T tokens (Table 4):** BitNet b1.58 3B trained on 2T tokens beats StableLM-3B on all tasks (avg 74.34 vs 73.22).
- **Energy (Figure 3):** 71.4× energy savings for matrix multiplication on 7nm chips, because ternary weights use INT8 addition instead of FP16 multiply-accumulate.

### Simplifications in This Notebook

- Tiny 4-layer transformer (~709K parameters, d_model 128) instead of 700M–3.9B
- Word-level Wikitext-2 (~10.8M tokens) instead of 100B tokens of RedPajama
- 20 epochs with AdamW (lr 1e-3) instead of large-scale pretraining
- Ternary weights simulated in float32 (no custom INT8/ternary kernels); focus is accuracy, not speed
- Perplexity on validation set only; no zero-shot task benchmarks
- Activation quantization to 8-bit simulated in float32

### What the Notebook Result Shows

On a 4-layer ternary-weight transformer (~709K params) trained on word-level Wikitext-2 for 20 epochs, the ternary BitLinear model achieves validation perplexity of 25.05, while the equivalent FP16 model reaches 3.72 — a gap of +21.33 at this tiny scale. This is expected: the paper shows BitNet b1.58 matches FP16 only starting at 3B parameters, and our model is ~4,000× smaller.

The ternary weights {-1, 0, +1} occupy about 1.58 bits each versus 16 bits for FP16 — a 10.1× memory reduction for the linear layers (1280 KB → 126 KB). The weight distribution after absmean quantization shows the expected ternary pattern: 37% at -1, 24.9% at 0, 38% at +1, confirming that the 0 value provides feature filtering. The key finding — ternary weights can learn and reduce memory 10× — is visible, but matching FP16 performance requires much larger scale.

## Influence

BitNet b1.58 was published by Microsoft Research in February 2024 and became one of the most discussed LLM efficiency papers. It directly extended the original BitNet (Wang et al., October 2023, arXiv:2310.11453) by adding the zero value for feature filtering and adopting LLaMA-compatible components. The work influenced research into 1-bit and ternary LLMs, including BitNet-style training recipes and hardware co-design for low-bit inference. The paper's claim that 1.58-bit LLMs define a "new scaling law" has been widely cited in the quantization and efficient inference literature. The official code was not publicly released by Microsoft, but community implementations appeared, including `bitnet` on PyPI and HuggingFace integrations. The paper has been cited extensively since its release ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/031_bitnet-b1.58-1-bit-llms/solution.ipynb) | PyTorch notebook that trains a tiny ternary-weight BitLinear transformer from scratch on character-level Wikitext-2 and compares perplexity against an equivalent FP16 model |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
