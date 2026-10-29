---
title: "BitNet b1.58: Talk & Q&A"
parent: "031 BitNet b1.58"
nav_order: 2
---
# TALK — The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58)

## Talks, Slides and Press Coverage

1. **Microsoft Research — BitNet b1.58:** The paper was published as an arXiv preprint by Microsoft Research in February 2024. The paper page is at [arXiv:2402.17764](https://arxiv.org/abs/2402.17764). No official talk video was released by the authors at the time of writing.

2. **Microsoft Research Blog — "The Era of 1-bit LLMs":** The work was covered in AI research communities and discussed extensively on social media following its release. The authors' research group page is at [Microsoft General AI](https://aka.ms/GeneralAI). No formal blog post link has been verified as of October 2026.

3. **Hugging Face — Community implementations:** Community implementations of BitNet b1.58 appeared on GitHub and Hugging Face following the paper, including a PyPI package `bitnet` and integration discussions in the transformers library. The original BitNet code is at [github.com/microsoft/BitNet](https://github.com/microsoft/BitNet) (released after the original BitNet paper, updated for b1.58).

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2402.17764).

**Q1: Why use ternary {-1, 0, +1} weights instead of binary {-1, +1} as in the original BitNet?**
A1 (from the paper, Section 1): The authors state that adding 0 to the original 1-bit BitNet gives "an additional value of 0 to the original 1-bit BitNet, resulting in 1.58 bits in the binary system." The zero value provides "explicit support for feature filtering, made possible by the inclusion of 0 in the model weights, which can significantly improve the performance of 1-bit LLMs." This improves modeling capability compared to binary weights.

**Q2: How does the absmean quantization function work?**
A2 (from the paper, Section 2, Eq. 1–3): "It first scales the weight matrix by its average absolute value, and then round each value to the nearest integer among {-1, 0, +1}: W̃ = RoundClip(W/(γ+ε), -1, 1)" where "γ = (1/nm) Σ\|W_ij\|" is the mean absolute value of the weight matrix and ε is a small constant for numerical stability. RoundClip(x, a, b) = max(a, min(b, round(x))).

**Q3: At what model size does BitNet b1.58 match the FP16 baseline?**
A3 (from the paper, Section 1 and Table 1): "BitNet b1.58 can match full precision (i.e., FP16) baselines in terms of both perplexity and end-task performance, starting from a 3B size." At 3B, BitNet b1.58 achieves perplexity of 9.91 vs LLaMA's 10.04, and zero-shot average accuracy of 50.2 vs 49.7. At smaller sizes (700M, 1.3B), the FP16 baseline still leads slightly.

**Q4: Why does BitNet b1.58 save so much energy?**
A4 (from the paper, Section 3, Energy): "The majority of BitNet b1.58 is INT8 addition calculation, while LLaMA LLM consists of both FP16 addition and FP16 multiplication." Since ternary weights mean multiplication is just addition/subtraction/skip, "BitNet b1.58 saves 71.4 times arithmetic operations energy consumption for matrix multiplication on 7nm chips."

**Q5: How does the activation quantization differ from the original BitNet?**
A5 (from the paper, Section 2): "The activations are all scaled to [-Qb, Qb] per token to get rid of the zero-point quantization. This is more convenient and simple for both implementation and system-level optimization, while introduces negligible effects to the performance in our experiments." The original BitNet scaled activations to [0, Qb] before nonlinear functions; b1.58 removes this scaling.

**Q6: What architecture does BitNet b1.58 use?**
A6 (from the paper, Section 2): "Our design of BitNet b1.58 adopts the LLaMA-alike components. Specifically, it uses RMSNorm, SwiGLU, rotary embedding, and removes all biases." This enables integration "into the popular open-source software (e.g. Huggingface, vLLM, and llama.cpp) with minimal efforts."

## Common Misconceptions

1. **"BitNet b1.58 is a post-training quantization method"** — No. The paper explicitly states the model is "trained from scratch, with 1.58-bit weights and 8-bit activations." This is quantization-aware training, not post-training quantization. The paper contrasts this with post-training quantization methods like GPTQ and AWQ, which it calls "sub-optimal."

2. **"1.58 bits means the model is 10× smaller end-to-end"** — Not exactly. The weight matrices are ~10× smaller (1.58 vs 16 bits), but embeddings remain full precision. The paper notes "the embedding remains full precision and its memory proportion is smaller for larger models." So the overall memory savings increase with model size: 2.60× at 700M, 3.55× at 3B, and higher at 70B.

3. **"BitNet b1.58 always beats FP16"** — No. At smaller scales (700M), BitNet b1.58 has slightly worse perplexity (12.87 vs 12.33) and zero-shot accuracy (44.3 vs 45.5). The paper says it "starts to match" FP16 "starting from a 3B size." The advantage grows with scale.

4. **"The 0 weight is just a rounding artifact"** — No. The paper explicitly frames 0 as a feature, not a bug: "its modeling capability is stronger due to its explicit support for feature filtering, made possible by the inclusion of 0 in the model weights." The zero enables selective information flow.

## Real Citations

1. Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei (2024). "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." *arXiv preprint.* [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) — the original paper.

2. Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei (2023). "BitNet: Scaling 1-bit Transformers for Large Language Models." [arXiv:2310.11453](https://arxiv.org/abs/2310.11453) — the predecessor paper introducing binary {-1, +1} BitNet; b1.58 extends it with ternary weights.

3. Hugo Touvron, Thibaut Lavril, Gautier Izacard, et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) — the LLaMA architecture whose components (RMSNorm, SwiGLU, RoPE) BitNet b1.58 adopts.

**Citation count:** BitNet b1.58 has been cited extensively since its February 2024 release; the paper is labeled "Work in progress" on arXiv ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764), checked October 2026).
