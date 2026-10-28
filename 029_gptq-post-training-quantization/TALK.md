---
title: "GPTQ: Talk & Q&A"
parent: "029 GPTQ"
nav_order: 2
---
# TALK — GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (GPTQ)

## Talks, Slides and Press Coverage

1. **ICLR 2023 — "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers":** The paper was published at the International Conference on Learning Representations (ICLR 2023). The paper is available at [arXiv:2210.17323](https://arxiv.org/abs/2210.17323), and the official code is at [github.com/IST-DASLab/gptq](https://github.com/IST-DASLab/gptq).

2. **HuggingFace Blog — "Making LLMs lighter with AutoGPTQ and transformers" (July 2023):** A blog post by the HuggingFace team describing how AutoGPTQ wraps the GPTQ algorithm and integrates it into the transformers library, enabling 4-bit quantized model loading and inference. Includes benchmarks showing memory savings and accuracy comparisons. ([link](https://huggingface.co/blog/gptq-integration))

3. **IST-DASLab/gptq — GitHub repository:** The official implementation of GPTQ by the authors, including scripts for quantizing OPT and BLOOM model families, evaluation code, and the 3-bit CUDA kernel. ([link](https://github.com/IST-DASLab/gptq))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2210.17323).

**Q1: Why does GPTQ quantize weights in arbitrary order instead of greedily picking the least-error weight?**
A1 (from the paper, Section 4, Step 1): The authors find that "while this quite natural strategy does indeed seem to perform very well, its improvement over quantizing the weights in arbitrary order is generally small, in particular on large, heavily-parametrized layers." The benefit of arbitrary order is that all rows share the same Hessian inverse, reducing the computational cost by a factor of min(d_row, d_col) — several orders of magnitude for large models.

**Q2: What is the Cholesky reformulation and why is it needed?**
A2 (from the paper, Section 4, Step 3): The paper observes that "the matrix H^{-1}_F becomes indefinite" due to numerical inaccuracies, which "can cause the algorithm to aggressively update the remaining weights in incorrect directions, resulting in an arbitrarily-bad quantization." The Cholesky decomposition provides a numerically stable way to precompute all needed rows of the inverse Hessian, and "in combination with mild dampening, the resulting method is robust enough to execute on huge models without issues."

**Q3: How much calibration data does GPTQ need?**
A3 (from the paper, Section 5): "Our entire GPTQ calibration data consists of 128 random 2048 token segments from the C4 dataset." The authors emphasize that "GPTQ does not see any task-specific data, and our results thus remain actually zero-shot."

**Q4: Can GPTQ speed up inference, or does it only reduce memory?**
A4 (from the paper, Section 5 and Table 6): Unlike LLM.int8() which "has the same runtime as the FP16 baseline," GPTQ achieves end-to-end speedups because "for language generation, the model processes and outputs one token at-a-time... these are primarily limited by memory bandwidth." The 3-bit quantized model accesses less memory, yielding "about 3.25x faster" on A100 and "4.5x" on A6000 GPUs.

**Q5: Does GPTQ quantize activations?**
A5 (from the paper, Section 6): No. The authors state that "our current results do not include activation quantization, as they are not a significant bottleneck in our target scenarios." They note this "can be supported using orthogonal techniques."

## Common Misconceptions

1. **"GPTQ is a training method that quantizes during training"** — No. GPTQ is explicitly a post-training (one-shot) quantization method. The paper states it quantizes "a pre-trained model using modest resources, typically a few thousand data samples and a few hours of computation." No gradient descent or retraining is involved.

2. **"GPTQ only works for 4-bit quantization"** — The paper shows results for 4-bit, 3-bit, 2-bit, and even ternary quantization. At 3-bit, GPTQ maintains reasonable perplexity (8.68 for OPT-175B) while RTN collapses entirely. At 2-bit with group-size 32, the perplexity increase is only 0.6-0.7 points.

3. **"GPTQ makes matmul faster"** — Not directly. The paper explicitly states in Section 6 that "our method obtains speedups from reduced memory movement, and does not lead to computational reductions." The speedup comes from loading fewer bytes per weight during memory-bound generation, not from faster arithmetic.

## Real Citations

1. Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023.* [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) — the original paper.

2. Elias Frantar, Sidak Pal Singh, Dan Alistarh (2022). "Optimal Brain Compression: A framework for accurate post-training quantization and pruning." [arXiv:2208.11580](https://arxiv.org/abs/2208.11580) — introduces OBQ, the method GPTQ builds on and scales up.

3. Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) — GPTQ uses the same per-row asymmetric quantization grid and directly compares against it as the RTN baseline.

4. Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) — follow-up that improves channel selection by using activation magnitudes, building on GPTQ's framework.

**Citation count:** about 197 citations on [OpenAlex](https://openalex.org/works/W4307934016) (checked October 2026).
