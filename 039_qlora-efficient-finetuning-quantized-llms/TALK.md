---
title: "QLoRA: Talk & Q&A"
parent: "039 QLoRA"
nav_order: 2
---
# TALK — QLoRA: Efficient Finetuning of Quantized LLMs (QLoRA)

## Talks, Slides and Press Coverage

1. **Hugging Face Blog — "Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA" (Younes Belkada, Tim Dettmers, Artidoro Pagnoni, Sylvain Gugger, Sourab Mangrulkar, 24 May 2023):** A blog post co-authored by the QLoRA authors and Hugging Face engineers that introduces 4-bit quantization and QLoRA to the practitioner community, with code examples for using it in the `transformers` library. Published the same week as the paper. ([link](https://huggingface.co/blog/4bit-transformers-bitsandbytes))

2. **Official code release — artidoro/qlora on GitHub (2023):** The official implementation, including CUDA kernels for 4-bit training, training scripts for the Guanaco models, and the OASST1/FLAN datasets used in the paper. ([GitHub](https://github.com/artidoro/qlora))

3. **NeurIPS 2023 — QLoRA was accepted to NeurIPS 2023:** The paper was initially released as an arXiv preprint on 23 May 2023 and accepted to the conference later that year. ([NeurIPS 2023 proceedings](https://nips.cc/virtual/2023/poster/73949))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their
section in the [arXiv paper](https://arxiv.org/abs/2305.14314).

**Q1: Why does 4-bit quantization not hurt fine-tuning performance?**
A1 (from the paper, Abstract): QLoRA "reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance." The key is that gradients are backpropagated through the frozen 4-bit model into LoRA adapters, and the 4-bit weights are only used in the forward pass — the adapters themselves are trained in higher precision (BF16).

**Q2: What makes NF4 better than FP4?**
A2 (from the paper, Section 3): NF4 is "information theoretically optimal for normally distributed weights." Because neural network weights empirically follow a normal distribution, using quantile-based bins (equal probability mass per bin) minimizes the expected quantization error. The paper shows NF4 outperforms FP4 in experiments, confirming the theoretical prediction.

**Q3: How does Double Quantization work and how much does it save?**
A3 (from the paper, Section 3): Double Quantization "compresses the quantization constants" — the FP32 per-block scales from the first quantization step are themselves quantized to 8-bit. This "reduces the effective memory footprint by quantizing the quantization constants" from 0.5 bits/parameter to 0.127 bits/parameter, saving 0.373 bits/parameter. For a 65B model, this saves approximately 3 GB.

**Q4: Did the authors trust chatbot benchmarks?**
A4 (from the paper, Abstract and Section 5): No. The authors "find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots." They provided "a detailed analysis of chatbot performance based on both human and GPT-4 evaluations showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation," and a "lemon-picked analysis" showing where Guanaco fails compared to ChatGPT.

**Q5: What is the Guanaco model?**
A5 (from the paper, Abstract): Guanaco is the name for the authors' best model family — QLoRA-fine-tuned LLaMA models. "Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU."

## Common Misconceptions

1. **"QLoRA trains the model in 4-bit."** — Only the frozen base model weights are stored in 4-bit. The LoRA adapters are trained in BF16 (bfloat16), and gradients flow through the 4-bit weights (which are dequantized to BF16 for the forward pass) to the adapters. The 4-bit representation is never directly differentiated — it is a compressed storage format.

2. **"QLoRA is just LoRA with 4-bit quantization."** — While that's the high-level idea, QLoRA introduces three specific innovations: NF4 (a novel data type optimized for weight distributions), Double Quantization (compressing the scales themselves), and Paged Optimizers (handling memory spikes via unified memory). Plain 4-bit + LoRA without these would not achieve the same memory savings or performance.

3. **"NF4 is just uniform quantization."** — NF4 is non-uniform: its 16 levels are the quantiles of N(0,1), so bins are denser near zero (where most weights are) and sparser at the tails. This is why it outperforms FP4, which has roughly uniform spacing. The paper proves NF4 is information-theoretically optimal for normally distributed data.

## Real Citations

1. Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023.* [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — the original paper.

2. Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — the LoRA method that QLoRA builds upon.

3. Wei et al. (2022). "Finetuned Language Models Are Zero-Shot Learners" (FLAN). [arXiv:2109.01652](https://arxiv.org/abs/2109.01652) — one of the instruction datasets used in QLoRA's experiments.

**Citation count:** about 1,834 citations on [OpenAlex](https://openalex.org/works/W7133233472) (checked November 2026).
