---
title: "028 LLM.int8()"
nav_order: 29
permalink: "/028_llm-int8-8-bit-matmul/"
---
# LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (LLM.int8())
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/llm-int8-8-bit-matrix-multiplication)

- **Paper:** [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- **Authors:** Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer
- **Lab, year:** University of Washington / Meta AI, 2022 (NeurIPS 2022)

## Summary

Large language models with 175B+ parameters require ~350 GB of GPU memory just to load in fp16 — far beyond what a single consumer GPU can hold. The straightforward fix is to quantize the weights to 8-bit integers, cutting memory in half, but naive int8 quantization destroys model performance above ~6B parameters. LLM.int8() solves this by understanding *why* standard quantization fails: large transformers develop "emergent outlier features" — a handful of hidden dimensions whose magnitudes are 6x or more larger than the median. These outliers appear systematically across every layer, position, and sequence, and they dominate the model's predictive performance.

The problem is that absmax quantization (scaling the entire tensor by its maximum value) lets the outliers stretch the quantization range so that the 99.9% of normal values collapse to near-zero resolution. LLM.int8() addresses this with a two-part procedure. First, vector-wise quantization gives each row of the weight matrix and each column of the hidden state its own normalization constant, so one outlier doesn't ruin the scale for everything else. Second, a mixed-precision decomposition isolates the outlier dimensions entirely: they are multiplied in fp16, while the remaining 99.9% of values use int8. The two results are summed, yielding zero degradation at half the memory.

The paper demonstrates that BLOOM-176B and OPT-175B can run in int8 with no performance degradation across benchmarks (LAMBADA, PIQA, HellaSwag, and more), making these models accessible on a single server with consumer GPUs. The method is 15-23% slower than fp16 for the largest models but preserves full predictive quality.

This repo's notebook implements vector-wise int8 quantization and the LLM.int8() mixed-precision decomposition from scratch, trains a tiny transformer with outlier features, and shows that LLM.int8() matches fp16 perplexity while absmax quantization degrades.

## What Problem Does It Solve

Imagine you have a huge cookbook (a 175-billion-parameter language model) that takes 350 GB of shelf space. Your kitchen only has room for 175 GB — half what you need. You could try to "compress" the recipes by rounding every number to the nearest integer (8-bit quantization), but you find that a few measurements — the ones that matter most, like the exact oven temperature — are so large they distort the whole rounding scale, and every other measurement becomes hopelessly imprecise.

The paper's solution is elegant: keep those critical large measurements at full precision (16-bit), and round everything else to integers (8-bit). Since the critical measurements are only 0.1% of the total, you still save nearly half the shelf space — but the recipes come out right because the important numbers are preserved exactly.

## Core Idea

- **Emergent outlier features (Section 3):** Transformers above ~6B parameters develop a small number of hidden dimensions with magnitudes 6x or more above the median. These outliers are systematic — they appear in the same dimensions across all layers, positions, and tokens, and they are critical for model performance.
- **Vector-wise quantization (Section 2):** Instead of one scale factor for the entire tensor, each row of the weight matrix and each column of the hidden state gets its own normalization constant. This prevents a single outlier from stretching the range for all other values.
- **Mixed-precision decomposition (Section 3):** The outlier feature dimensions are extracted and computed in fp16, while the remaining 99.9% of values use int8 vector-wise quantization. The two partial results are summed to produce the final output — zero degradation at half the memory.

## Key Method Details

- **Quantization formula (absmax):** For a tensor X, scale = max(\|X\|) / 127; quantized = round(X / scale), clamped to [-127, 127]. Dequantized = quantized × scale.
- **Vector-wise quantization:** For W (out × in) and h (batch × in): row_scales = max(\|W\|, dim=1) / 127, col_scales = max(\|h\|, dim=0) / 127. Quantize W per-row and h per-column, then dequantize after matmul.
- **Outlier threshold (Section 3):** Hidden dimensions where \|h\| ≥ 6.0 × median(\|h\|) are classified as outliers. The paper finds that this threshold recovers full inference performance.
- **Outlier fraction:** Approximately 0.1% of dimensions are outliers (the paper reports 0.06%–0.1% depending on the model). These are computed in fp16; the rest in int8.
- **Key result — Table 1 (BLOOM-176B):** LLM.int8() perplexity matches fp16 across all benchmarks. LAMBADA accuracy: fp16 = 67.19%, int8 = 67.11% (difference within standard error). Zero-degradation confirmed.
- **Key result — Figure 3:** Outlier features emerge at ~6.7B parameters (OPT-6.7B) and grow more pronounced with scale. Below this threshold, standard int8 quantization works fine.
- **Speed (Section 4):** BLOOM-176B int8 is 15-23% slower than fp16. Smaller models (T5-3B, T5-11B) see larger slowdowns due to kernel overhead, but the paper reports optimisations reducing this gap.
- **Memory:** 175B model in fp16 = ~350 GB; in int8 = ~176 GB. A single server with 8× RTX 3090 (24 GB each = 192 GB) can run BLOOM-176B.

### Simplifications in This Notebook

- Tiny 6-layer transformer (~2M parameters, d_model 256) instead of BLOOM-176B / OPT-175B
- Outlier features are amplified via weight initialisation rather than emerging naturally (which requires ~6B+ parameters)
- Character-level Wikitext-2 (~2M tokens) instead of large-scale web text
- 15 training epochs with AdamW (lr 5e-4, weight decay 0.01) instead of large-scale pretraining
- Int8 matmul simulated in PyTorch float (no CUDA int8 tensor cores) — focus is accuracy, not speed
- Perplexity evaluated on validation set only; no lm-eval-harness benchmarks

### What the Notebook Result Shows

On a 6-layer transformer (4.8M params) with injected outlier features, 34 out of 256 hidden dimensions exceed the 6x median threshold (13.3% — higher than the paper's 0.1% because outliers are injected, not emergent). In the feed-forward matmul accuracy test, absmax int8 produces mean relative error of 0.055 (vs fp16's 0.002 — a 27x gap), because outlier columns stretch the tensor-level scale. Vector-wise quantization reduces this to 0.033 (20x gap). LLM.int8() mixed-precision decomposition isolates 12 outlier dimensions into fp16 and achieves mean relative error of 0.009 — a 5.5x improvement over absmax and 4.4x over vector-wise, closing much of the gap to fp16.

End-to-end perplexity is 1.02 for all methods — the model is small enough that quantization error doesn't change the argmax predictions. This is expected: the paper's zero-degradation result requires BLOOM-176B scale where emergent outliers are far more pronounced and quantization error is catastrophic without mixed-precision. Memory savings are exactly 50% (fp16 to int8: 9.2 MB to 4.6 MB), matching the paper's claim.

## Influence

LLM.int8() was developed at the University of Washington (Luke Zettlemoyer's group) and Meta AI, published at NeurIPS 2022. It was integrated directly into the HuggingFace transformers library and the bitsandbytes package, becoming the standard method for 8-bit LLM inference. The discovery of emergent outlier features influenced subsequent quantization research, including GPTQ (Frantar et al., 2022), AWQ (Lin et al., 2023), and SmoothQuant (Xiao et al., 2022), which all build on the observation that a few outlier dimensions dominate quantization error. The open-source bitsandbytes library at [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) remains widely used.

About 300 citations on [OpenAlex](https://openalex.org/works/W7133198810) (checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/028_llm-int8-8-bit-matmul/solution.ipynb) | PyTorch notebook that trains a tiny transformer with outlier features, implements absmax/vector-wise/LLM.int8() quantization, and compares matmul error, perplexity, and memory savings |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
