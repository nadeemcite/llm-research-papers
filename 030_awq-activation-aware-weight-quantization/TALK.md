---
title: "AWQ: Talk & Q&A"
parent: "030 AWQ"
nav_order: 2
---
# TALK — AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (AWQ)

## Talks, Slides and Press Coverage

1. **MLSys 2024 — "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration":** The paper was published at MLSys 2024 and received the **Best Paper Award**. The talk video was released on the MLSys 2024 website. The paper is available at [arXiv:2306.00978](https://arxiv.org/abs/2306.00978), and the official code is at [github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq). ([project page](https://hanlab.mit.edu/projects/awq))

2. **MIT HAN Lab — AWQ Project Page:** The official project page lists adoption by NVIDIA TensorRT-LLM, AMD, Google Vertex AI, Amazon SageMaker, Intel Neural Compressor, vLLM, HuggingFace TGI, FastChat, and LMDeploy. AWQ models on HuggingFace have received over 6 million downloads. ([link](https://hanlab.mit.edu/projects/awq))

3. **HuggingFace — AutoAWQ integration:** The HuggingFace transformers library integrated AWQ quantization support, enabling loading and running AWQ-quantized models directly. The AutoAWQ library provides a Python API for quantizing any HF model to AWQ format. ([github.com/casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2306.00978).

**Q1: Why are salient weight channels identified by activation magnitude rather than weight magnitude?**
A1 (from the paper, Section 3.1): The authors find that "there is a small fraction (0.1%-1%) of salient weights; skipping the quantization of these salient weights will significantly reduce the quantization loss." When they compare selecting channels by activation magnitude vs weight magnitude vs random (Table 1), only activation-based selection works: keeping 1% of channels by activation reduces OPT-6.7B INT3-g128 perplexity from 23.54 to 11.39, while weight-based selection gives 22.37 (barely better than RTN's 23.54) and random gives 24.23 (worse). The insight is that "we should refer to the activation distribution, not weight."

**Q2: How does scaling up salient channels reduce quantization error without mixed-precision hardware?**
A2 (from the paper, Section 3.2): The key derivation is that "if we multiply w with s > 1 and the inversely scale x, we will have Q(w·s)(x/s)." The ratio of the new error to the original error is (Δ'/Δ)·(1/s). Since Δ' ≈ Δ (the quantization scaler barely changes for s < 2, affecting less than 5% of groups) and s > 1, "the relative error is smaller for the salient weight w." The inverse scale 1/s can be fused into the previous operator, so "it can usually be fused into the previous operator" — no mixed-precision hardware needed.

**Q3: How much calibration data does AWQ need compared to GPTQ?**
A3 (from the paper, Section 5.3): AWQ needs much less calibration data because "we only measure the average activation scale from the calibration set, which is data-efficient." Figure 8(a) shows that "it can achieve better perplexity using 10× smaller calibration set compared to GPTQ (16 sequences v.s. 192 sequences)." AWQ is also less sensitive to calibration distribution: using PubMed Abstracts or Enron Emails as calibration gives similar perplexity, while GPTQ varies more.

**Q4: Can AWQ be combined with GPTQ?**
A4 (from the paper, Section 5.2): Yes. The authors state "our method is orthogonal to GPTQ. We can combine our method with GPTQ to further improve the INT2 quantization performance." Table 10 shows AWQ+GPTQ on OPT-6.7B INT2 achieves perplexity of 15.71 vs GPTQ alone at 16.65 and RTN at 7622.

**Q5: Why does GPTQ perform worse than AWQ on some models like LLaMA-7B INT3?**
A5 (from the paper, Section 5.1): GPTQ "may overfit the calibration set during reconstruction, distorting the distribution of LLMs' knowledge outside the calibration domain." On LLaMA-7B INT3-g128, GPTQ gets perplexity of 8.81 (vs fp16's 5.68), while AWQ gets 6.35. AWQ avoids this because it "does not rely on any regression or backpropagation" and only measures average activation magnitudes.

## Common Misconceptions

1. **"AWQ keeps the salient 1% of weights in fp16"** — No. The paper explicitly discusses this limitation in Section 3.1: "such a mixed-precision data type will make the system implementation difficult." AWQ's contribution is that it protects salient channels through equivalent scaling (multiply weights by s, divide activations by s) so that all weights remain at the same low bit width, avoiding mixed-precision hardware entirely.

2. **"AWQ is a quantization-aware training method"** — No. AWQ is explicitly a post-training quantization method. The paper states it "does not rely on any backpropagation or reconstruction." The scaling factor is found by a simple grid search over one hyperparameter α, not by gradient descent.

3. **"AWQ and GPTQ are competing methods, you must choose one"** — No. The paper shows they are orthogonal: "our method is orthogonal to GPTQ." AWQ does activation-aware scaling before quantization, while GPTQ does second-order error compensation during quantization. Table 10 shows combining both improves INT2 results over either alone.

4. **"AWQ only works for language models"** — No. The paper demonstrates AWQ on multi-modal models for the first time: OpenFlamingo-9B (Table 6) and LLaVA-13B (Section 5.2, Figure 6). The paper states "thanks to better generalization, it achieves excellent quantization performance for instruction-tuned LMs and, for the first time, multi-modal LMs."

## Real Citations

1. Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *MLSys 2024 (Best Paper Award).* [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) — the original paper.

2. Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023.* [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) — the primary comparison method; AWQ shows it overfits the calibration set and proposes a training-free alternative.

3. Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) — established the importance of activation-based outlier detection in LLM quantization, which AWQ extends to weight-only quantization.

4. Guangxuan Xiao, Ji Lin, Nicklas Hansen, Hao Bai, Song Han (2022). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) — a related W8A8 method that also uses activation-aware scaling; AWQ adapts the idea to weight-only quantization.

**Citation count:** AWQ has received substantial citations since its 2023 release; the MIT HAN Lab project page reports over 6 million downloads of AWQ models on HuggingFace, and it was adopted by major inference frameworks including NVIDIA TensorRT-LLM, vLLM, and HuggingFace TGI (checked October 2026).
