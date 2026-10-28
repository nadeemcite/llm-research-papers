---
title: "029 GPTQ"
nav_order: 30
permalink: "/029_gptq-post-training-quantization/"
---
# GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (GPTQ)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/gptq-post-training-quantization)

- **Paper:** [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- **Authors:** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
- **Lab, year:** IST Austria / ETH Zurich / NeuralMagic, 2022 (ICLR 2023)

## Summary

Large language models like OPT-175B and BLOOM-176B need 326 GB of GPU memory in fp16 — far beyond any single GPU. Post-training quantization (compressing weights to 3 or 4 bits after training, without retraining) is the obvious fix, but until GPTQ, only simple round-to-nearest (RTN) methods could scale to these sizes, and they fail badly below 8 bits. GPTQ is the first method that compresses 175B-parameter models to 3-4 bits in about four GPU hours with negligible accuracy loss.

GPTQ builds on Optimal Brain Quantization (OBQ), a second-order method that quantizes weights one at a time, using the Hessian of the layer's reconstruction error to pick the least-harmful weight to round and to compensate remaining weights for the rounding error. OBQ was limited to ~100M-parameter models because of its O(d_row * d_col^3) runtime. GPTQ introduces three innovations: (1) quantizing columns in arbitrary order rather than greedily, so all rows share the same Hessian inverse and the cost drops to O(max(d_row * d_col^2, d_col^3)); (2) lazy batch updates of 128 columns at a time to improve GPU memory throughput; and (3) a Cholesky-based reformulation of the inverse Hessian that avoids numerical instability on large models.

The paper shows that 4-bit GPTQ on OPT-175B achieves WikiText2 perplexity of 8.37 vs the fp16 baseline of 8.34 (a difference of 0.03), while RTN degrades to 10.54. At 3-bit, RTN collapses to 7300+ perplexity, while GPTQ maintains 8.68. The compressed 3-bit OPT-175B fits on a single A100 GPU and achieves 3.25x speedup for generation.

This repo's notebook implements layer-wise GPTQ from scratch on a tiny transformer, compares 4-bit perplexity against round-to-nearest, and shows that GPTQ compensates quantization error via second-order updates while RTN does not.

## What Problem Does It Solve

Imagine you have a massive dictionary (a 175-billion-parameter language model) that takes 326 GB of shelf space. You want to shrink it to fit on one GPU (~80 GB). The obvious approach is to round every number to fewer digits — say, from 16-bit to 3-bit (a 5x compression). But naive rounding destroys the model: at 3 bits, the perplexity (a measure of how well the model predicts text — lower is better) explodes from 8.34 to over 7000, rendering it useless.

GPTQ's insight is that when you round a weight, you create a small error. But you can compensate for that error by slightly adjusting all the other weights in the same layer. The key is knowing *which* weights to adjust and by how much — and for that, GPTQ uses the Hessian matrix, which captures how sensitive the layer's output is to each weight. By processing weights column by column and propagating the error into the remaining weights, GPTQ keeps the layer's behavior nearly unchanged even after every weight is rounded to 3 or 4 bits.

## Core Idea

- **Arbitrary order quantization (Section 4, Step 1):** Instead of greedily picking the weight with least error (as OBQ does), GPTQ quantizes columns in fixed order. All rows share the same Hessian inverse, reducing the Hessian update cost from O(d_row * d_col^3) to O(d_col^3) — a factor of d_row speedup.
- **Lazy batch updates (Section 4, Step 2):** Updates to the weight matrix are accumulated in blocks of 128 columns and applied in bulk, avoiding the memory-bandwidth bottleneck of per-column GPU updates.
- **Cholesky reformulation (Section 4, Step 3):** The inverse Hessian is computed once via Cholesky decomposition rather than repeatedly inverting after each column, which prevents numerical instability that causes arbitrarily bad quantization on large models.

## Key Method Details

- **Layer-wise objective (Section 3):** For each linear layer with weights W and inputs X, find quantized weights Q that minimize \|WX - QX\|². The Hessian is H = 2XX^T + λI, where λ is a dampening term (1% of the average diagonal value).
- **GPTQ algorithm (Algorithm 1):** Compute H^{-1} via Cholesky. For each block of B=128 columns: quantize each column, compute the error e = (W[:,j] - quant(W[:,j])) / H^{-1}[j,j], and update the remaining weights in the block: W[:,j:B] -= e * H^{-1}[j, j:B]. After the block, update all remaining columns.
- **Quantization grid:** Standard uniform per-row asymmetric min-max quantization, the same grid used by LLM.int8().
- **Calibration data:** 128 random 2048-token segments from the C4 dataset (generic web text, no task-specific data).
- **Key result — OPT-175B 4-bit (Table 5):** WikiText2 perplexity: fp16 = 8.34, GPTQ 4-bit = 8.37 (+0.03), RTN 4-bit = 10.54 (+2.20). PTB: fp16 = 12.01, GPTQ = 12.26, RTN = 14.22. LAMBADA accuracy: fp16 = 75.59%, GPTQ = 76.80%.
- **Key result — 3-bit (Table 5):** GPTQ 3-bit OPT-175B: WikiText2 = 8.68 (+0.34). RTN 3-bit: 7300 (complete collapse). With group-size 128: 8.45 (+0.11).
- **Runtime (Table 2):** OPT-175B in 4.2 hours on a single A100. BLOOM-176B in 3.8 hours. OPT-13B in 20.9 minutes.
- **Speedup (Table 6):** 3-bit OPT-175B on a single A100: 71ms/token vs 230ms/token for fp16 on 5 GPUs — 3.24x speedup. On A6000: 130ms vs 589ms — 4.53x speedup.

### Simplifications in This Notebook

- Tiny 4-layer transformer (~846K parameters, d_model 128) instead of OPT-175B / BLOOM-176B
- Character-level Wikitext-2 (~2M tokens) instead of C4 calibration data
- 4-bit quantization only (paper also shows 3-bit and 2-bit results)
- Block size B=128 as in the paper, but with small layers the full column set fits in one block
- Quantization simulated in float32 (no custom CUDA kernels); focus is accuracy, not speed
- Perplexity on validation set only; no LAMBADA/PIQA/ARC zero-shot benchmarks

### What the Notebook Result Shows

On a 4-layer transformer (846K params) trained on character-level Wikitext-2, the fp16 validation perplexity is 1.76. Round-to-nearest 4-bit quantization increases perplexity to 1.89 (+0.13, a 7.4% increase). GPTQ 4-bit quantization achieves perplexity of 1.79 (+0.03, a 1.7% increase) — a 4.3x reduction in perplexity increase compared to RTN. The layer-wise reconstruction error (measured as mean squared error between fp16 and quantized layer outputs) is 0.0021 for GPTQ vs 0.0094 for RTN — a 4.5x improvement. These results are not comparable to the paper's OPT-175B numbers because the model is ~100,000x smaller, but the relative pattern (GPTQ significantly outperforms RTN) matches.

## Influence

GPTQ was developed at IST Austria (Dan Alistarh's group) and ETH Zurich, published at ICLR 2023. It became one of the most widely adopted LLM quantization methods: the AutoGPTQ library and HuggingFace transformers integration made 4-bit GPTQ models a standard format for LLM deployment. The method directly influenced AWQ (Lin et al., 2023), which improved activation-aware channel selection, and was compared against in subsequent work like SmoothQuant (Xiao et al., 2022) and exLLaMA. The original code is at [github.com/IST-DASLab/gptq](https://github.com/IST-DASLab/gptq).

About 197 citations on [OpenAlex](https://openalex.org/works/W4307934016) (checked October 2026).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/029_gptq-post-training-quantization/solution.ipynb) | PyTorch notebook that trains a tiny transformer, implements layer-wise GPTQ with Cholesky-based Hessian inverse, and compares 4-bit perplexity and reconstruction error against round-to-nearest |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
