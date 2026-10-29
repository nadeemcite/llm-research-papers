---
title: "030 AWQ"
nav_order: 31
permalink: "/030_awq-activation-aware-weight-quantization/"
---
# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (AWQ)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/awq-activation-aware-weight-quantization)

- **Paper:** [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- **Authors:** Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han
- **Lab, year:** MIT / Tsinghua University / MIT-IBM Watson AI Lab, 2023 (MLSys 2024, Best Paper Award)

## Summary

Large language models are powerful but enormous — Llama-2-70B needs 140 GB in fp16, far exceeding the memory of any single consumer GPU. Weight-only quantization (compressing weights to 4 bits while keeping activations in 16-bit float) reduces memory by 4x, but naive round-to-nearest (RTN) quantization introduces large errors because it treats all weights equally. AWQ (Activation-aware Weight Quantization) is a hardware-friendly, training-free method that identifies which weight channels matter most by looking at activation magnitudes, then scales those channels before quantization to protect them.

The key insight is that only 0.1–1% of weight channels are "salient" — their importance comes not from the weight values themselves but from the activation magnitudes flowing through them. Channels with large activations amplify quantization error, so protecting them yields disproportionate benefit. Rather than keeping these channels in fp16 (which creates inefficient mixed-precision hardware), AWQ mathematically derives a per-channel scaling factor: multiply the salient weight channels by s > 1, quantize, then divide the activations by s (which can be fused into the preceding operation). This equivalent transformation reduces the relative quantization error for important channels while keeping all weights at the same bit width.

On Llama-2-7B INT4-g128, AWQ achieves WikiText2 perplexity of 5.60 vs fp16's 5.47 (+0.13), while RTN gets 5.73 (+0.26) and GPTQ gets 5.69 (+0.22). On LLaMA-7B INT3-g128, AWQ achieves 6.35 vs RTN's 7.01 and GPTQ's 8.81. The method generalizes better than GPTQ because it does not rely on reconstruction (which overfits the calibration set), making it effective for instruction-tuned models (Vicuna) and, for the first time, multi-modal models (OpenFlamingo, LLaVA). Alongside AWQ, the authors implement TinyChat, an inference framework achieving 2.7–3.9x speedup over HuggingFace fp16 on desktop and mobile GPUs.

This repo's notebook implements AWQ from scratch on a tiny transformer, compares 4-bit perplexity and reconstruction error against RTN and GPTQ, and shows that activation-aware scaling reduces quantization error without any backpropagation.

## What Problem Does It Solve

Imagine you have a massive bookshelf (a 70-billion-parameter model) that takes 140 GB of space, but your shelf only holds 40 GB. You need to compress every number from 16 digits to 4 digits — a 4x shrink. The obvious approach is to round every number to the nearest 4-digit value. But some numbers are far more important than others: the ones that get multiplied by large incoming signals (activations). Rounding those important numbers causes outsized errors that cascade through the model.

AWQ's insight is that you can tell which numbers are important by looking at the signal magnitudes (activations) that flow through them, not by looking at the numbers themselves. A weight of 0.01 might be critical if the activation flowing through it is 100, while a weight of 0.5 might be negligible if its activation is 0.001. By scaling up the important channels before rounding (and inversely scaling the activations), you effectively give those channels more quantization resolution — like using finer grid spacing where it matters most — all while keeping every weight at the same 4-bit format that hardware can process efficiently.

## Core Idea

- **Salient channel identification (Section 3.1):** Only 0.1–1% of weight channels matter for model performance. The saliency of a weight channel is determined by the activation magnitude flowing through it, not the weight value. Skipping quantization of just 1% of the most salient channels (by activation) reduces OPT-6.7B INT3-g128 perplexity from 23.54 to 11.39, while selecting channels by weight magnitude or randomly has negligible effect.
- **Activation-aware scaling (Section 3.2):** To avoid mixed-precision hardware, AWQ scales the salient weight channels by s > 1 before quantization and inversely scales activations by 1/s. The ratio of quantization error becomes (Δ'/Δ) × (1/s), which is smaller than 1 when s > 1 and Δ' ≈ Δ. The inverse scale can be fused into the previous operator (e.g. LayerNorm), so no extra computation at inference time.
- **Grid search for optimal scale (Section 3.2):** The per-channel scale is parameterized as s = s_X^α, where s_X is the average activation magnitude per channel and α ∈ [0, 1] is a single hyperparameter found by fast grid search. This requires no backpropagation or reconstruction, making it data-efficient (10x smaller calibration set than GPTQ) and generalizable across domains.

## Key Method Details

- **Quantization format:** Weight-only INT4 (or INT3) with group size 128 (notebook uses 64 for the tiny model). Group-wise quantization means each group of consecutive input channels shares one scale factor, improving the performance/model-size trade-off.
- **Calibration data:** 128 sequences from the Pile dataset (generic pre-training data, not task-specific). AWQ only needs the average activation magnitude per channel, so it requires 10x less calibration data than GPTQ (16 sequences vs 192 for comparable perplexity).
- **Scale search:** s_i = (s_X_i)^α where s_X_i is the average magnitude of activation for input channel i. Grid search over α ∈ {0, 0.25, 0.5, 0.75, 1.0}. Weight clipping is applied to minimize MSE.
- **Key result — Llama-2 INT4-g128 (Table 4):** WikiText2 perplexity: fp16 = 5.47, AWQ = 5.60 (+0.13), RTN = 5.73 (+0.26), GPTQ = 5.69 (+0.22). Llama-2-70B: fp16 = 3.32, AWQ = 3.41 (+0.09), RTN = 3.46 (+0.14).
- **Key result — LLaMA INT3-g128 (Table 4):** LLaMA-7B: fp16 = 5.68, AWQ = 6.35 (+0.67), RTN = 7.01 (+1.33), GPTQ = 8.81 (+3.13). GPTQ overfits the calibration set and degrades on LLaMA-7B INT3.
- **Key result — OPT INT3-g128 (Table 3):** OPT-6.7B: fp16 = 10.86, AWQ = 11.39 (+0.53), RTN = 23.54 (+12.68), GPTQ with s=2 = 11.92 (+1.06).
- **Key result — coding/math (Tables 8–9):** MBPP pass@1: fp16 = 38.53, AWQ = 40.64, RTN = 37.51, GPTQ = 31.97. GSM8K (7B): fp16 = 13.87, AWQ = 13.57, RTN = 11.07, GPTQ = 12.13. AWQ outperforms GPTQ on coding and math tasks.
- **Speedup (Figure 9):** TinyChat with 4-bit AWQ achieves 2.7–3.9x speedup over HuggingFace fp16 on RTX 4090 across Llama-2, MPT, and Falcon. On Jetson Orin, up to 1.7x over llama.cpp.
- **INT2 results (Table 10):** AWQ + GPTQ combined achieves OPT-6.7B INT2 perplexity of 15.71 vs GPTQ alone at 16.65 and RTN at 7622. The methods are orthogonal and can be combined.

### Simplifications in This Notebook

- Tiny 4-layer transformer (~846K parameters, d_model 128) instead of OPT-6.7B / LLaMA-7B / Llama-2-70B
- Character-level Wikitext-2 (~2M tokens) instead of the Pile calibration set
- 4-bit weight-only quantization with group size 64 (paper uses 128; smaller group for tiny model)
- 20 epochs with AdamW (lr 1e-3, weight decay 0.01) instead of large-scale pretraining
- Quantization simulated in float32 (no custom CUDA kernels); focus is accuracy, not speed
- Perplexity on validation set only; no MBPP, GSM8K, or VLM benchmarks
- Grid search over 5 alpha values {0, 0.25, 0.5, 0.75, 1.0} (paper uses the same range)

### What the Notebook Result Shows

On a 4-layer transformer (844K params) trained on character-level Wikitext-2, the fp16 validation perplexity is 3.83. Round-to-nearest 4-bit group-wise quantization increases perplexity to 4.45 (+0.62, a 16.3% increase). GPTQ 4-bit achieves 4.04 (+0.21, a 5.6% increase). AWQ 4-bit achieves 4.40 (+0.58, a 15.1% increase) — a 7.1% reduction in degradation compared to RTN.

The average layer reconstruction error (MSE between fp16 and quantized layer outputs) is 0.041 for AWQ vs 0.047 for RTN (1.2x improvement). GPTQ outperforms AWQ on this tiny model because its second-order error compensation is more powerful at small scale, while AWQ's advantage is generalization at large scale (the paper shows GPTQ overfits the calibration set on LLaMA-7B INT3). These results are not comparable to the paper's OPT-6.7B or LLaMA-7B numbers because the model is ~8000x smaller, but the relative pattern (AWQ < RTN in degradation) is visible.

## Influence

AWQ was developed at MIT (Song Han's lab) in collaboration with Tsinghua University, published at MLSys 2024 where it received the Best Paper Award. It became one of the most widely adopted LLM quantization methods: the official implementation ([github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)) and the AutoAWQ library made AWQ-quantized models a standard format. NVIDIA TensorRT-LLM, AMD, Google Vertex AI, Amazon SageMaker, Intel Neural Compressor, vLLM, HuggingFace TGI, FastChat, and LMDeploy all adopted AWQ. AWQ models on HuggingFace have received over 6 million downloads. The method directly built on GPTQ (Frantar et al., 2022) and LLM.int8() (Dettmers et al., 2022), and influenced subsequent work on 4-bit LLM serving. The original code is at [github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/llm-research-papers/blob/main/030_awq-activation-aware-weight-quantization/solution.ipynb) | PyTorch notebook that trains a tiny transformer, implements AWQ activation-aware scaling with grid search, and compares 4-bit perplexity and reconstruction error against RTN and GPTQ |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Talks, press coverage, Q&A answered from the paper, citations, and common misconceptions about the paper |
