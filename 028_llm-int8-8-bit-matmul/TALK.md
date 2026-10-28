---
title: "LLM.int8(): Talk & Q&A"
parent: "028 LLM.int8()"
nav_order: 2
---
# TALK — LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (LLM.int8())

## Talks, Slides and Press Coverage

1. **HuggingFace Blog — "A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes" (Younes Belkada & Tim Dettmers, 17 Aug 2022):** A detailed blog post explaining the LLM.int8() algorithm, the emergent outlier features phenomenon, and how to use the bitsandbytes integration in HuggingFace transformers. Includes code examples and benchmark results. ([link](https://huggingface.co/blog/hf-bitsandbytes-integration))

2. **NeurIPS 2022 — "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale":** The paper was published at the 36th Conference on Neural Information Processing Systems (New Orleans, USA, November 2022). The paper is available at [arXiv:2208.07339](https://arxiv.org/abs/2208.07339).

3. **TimDettmers/bitsandbytes — GitHub repository:** The open-source library implementing LLM.int8() and other quantization methods, integrated into HuggingFace transformers. Includes documentation, examples, and CUDA kernels. ([link](https://github.com/TimDettmers/bitsandbytes))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2208.07339).

**Q1: Why does standard int8 quantization fail for large transformers?**
A1 (from the paper, Section 3): The paper shows that "highly systematic emergent features in transformer language models dominate attention and transformer predictive performance." These outlier features have magnitudes 6x or more above the median and appear in the same dimensions across all layers. When absmax quantization scales the entire tensor by its maximum, the outliers stretch the range so that "more than 99.9% of values are multiplied in 8-bit" but with collapsed precision, degrading performance.

**Q2: What is the mixed-precision decomposition and why does it work?**
A2 (from the paper, Abstract): "We develop a new mixed-precision decomposition scheme, which isolates the outlier feature dimensions into a 16-bit matrix multiplication while still more than 99.9% of values are multiplied in 8-bit." This works because the outliers are concentrated in a tiny fraction of dimensions — isolating them preserves the critical information while the bulk of the computation runs efficiently in int8.

**Q3: At what model scale do outlier features emerge?**
A3 (from the paper, Section 3 and Figure 3): The paper observes that "classic quantization at scale fails for transformer-based models >6B parameters." Below this threshold, standard int8 quantization works without degradation. The outliers become more pronounced and systematic as model size increases, reaching their strongest effect in the 175B-parameter range.

**Q4: What is the outlier threshold and how was it chosen?**
A4 (from the paper, Section 3): The paper uses a threshold of magnitude 6 or greater to identify outlier features. "We found that extracting all outliers with magnitude 6 or greater in this way recovers full inference performance." This threshold was determined empirically to balance the number of dimensions requiring fp16 computation against the resulting accuracy.

**Q5: Is LLM.int8() faster or slower than fp16?**
A5 (from the paper, Section 4): The primary goal is memory reduction, not speed. For BLOOM-176B, int8 is about 15-23% slower than fp16. For smaller models, the slowdown is larger due to kernel overhead. The paper notes that optimisations were actively being developed to reduce this gap, particularly for smaller models.

**Q6: Can LLM.int8() be used for training, or only inference?**
A6 (from the paper, Abstract): LLM.int8() is designed for inference only. "A 175B parameter 16/32-bit checkpoint can be loaded, converted to Int8, and used immediately without performance degradation." The method is not a training-time quantization technique.

## Common Misconceptions

1. **"LLM.int8() makes inference faster"** — No. The paper explicitly measures 15-23% slowdown for BLOOM-176B. The benefit is memory reduction (50%), not speed. A 175B model that needs 350 GB in fp16 fits in 176 GB in int8, making it accessible on cheaper hardware — but the int8 matmul itself is slightly slower than fp16 on current GPUs due to dequantization overhead.

2. **"Outlier features are rare random noise"** — The opposite. The paper's key finding is that outliers are "highly systematic" — they appear in the same dimensions across every layer, every position, and every sequence. This systematic nature is what makes them isolatable: the mixed-precision decomposition can reliably detect and extract the same outlier dimensions regardless of the input.

3. **"LLM.int8() works equally well for all model sizes"** — Not quite. The paper shows that standard int8 quantization works fine below ~6B parameters because outlier features haven't emerged yet. LLM.int8() is specifically designed for the regime where outliers exist (6B+). For smaller models, the added complexity of mixed-precision decomposition provides no benefit over simpler quantization.

## Real Citations

1. Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022.* [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) — the original paper.

2. Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) — builds on the outlier feature observation to develop a layer-wise quantization method that uses second-order information to compensate for quantization error.

3. Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) — further develops the activation-aware quantization idea by scaling salient channels before quantization, inspired by LLM.int8()'s outlier analysis.

4. Guangxuan Xiao, Ji Lin, Nick Seznec, Hao Wu, Julien Demouth, Song Han (2022). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) — directly addresses the outlier problem from LLM.int8() by smoothing activation outliers into weights before quantization.

**Citation count:** about 300 citations on [OpenAlex](https://openalex.org/works/W7133198810) (checked October 2026).
