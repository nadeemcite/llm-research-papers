---
title: "LLaMA: Talk & Q&A"
parent: "006 LLaMA"
nav_order: 2
---
# TALK — LLaMA: Open and Efficient Foundation Language Models (LLaMA)

## Talks, Slides and Press Coverage

1. **Simon Willison — "Large language models are having their Stable Diffusion moment" (11 Mar 2023):** Willison's widely-read blog post covers the LLaMA weights leak, the rapid proliferation of fine-tuned variants, and the comparison to the Stable Diffusion open-release moment. ([link](https://simonwillison.net/2023/Mar/11/llama/))
2. **Hugging Face Blog — "Llama 2 is here, get it on Hugging Face" (18 Jul 2023):** While focused on Llama 2, this post provides context on the original LLaMA's architecture choices (RMSNorm, RoPE, SwiGLU) and its role in democratizing LLM access. ([link](https://huggingface.co/blog/llama2))
3. **GitHub — facebookresearch/llama:** The official repository for the LLaMA models, released by Meta AI. ([link](https://github.com/facebookresearch/llama))
4. **arXiv paper page:** The paper was posted on arXiv on 27 Feb 2023 and became one of the most downloaded papers in arXiv history. ([arXiv:2302.13971](https://arxiv.org/abs/2302.13971))

## Q&A (answers from the paper)

These answers come from the paper itself, not from an interview. Direct quotes are marked with their section in the [arXiv paper](https://arxiv.org/abs/2302.13971).

**Q1: Why train smaller models on more tokens instead of larger models on fewer tokens?**
A1 (from the paper, Section 1): "given a target level of performance, the preferred model is not the fastest to train but the fastest at inference, and although it may be cheaper to train a large model to reach a certain level of performance, a smaller one trained longer will ultimately be cheaper at inference."

**Q2: What data was used for pre-training?**
A2 (from the paper, Section 2.1): The training data is a mixture of seven publicly available sources: "English CommonCrawl [67%]", "C4 [15%]", "Github [4.5%]", "Wikipedia [4.5%]", "Gutenberg and Books3 [4.5%]", "ArXiv [2.5%]", "Stack Exchange [2%]". "Overall, our entire training dataset contains roughly 1.4T tokens after tokenization."

**Q3: What are the key architectural changes from the original Transformer?**
A3 (from the paper, Section 2.2): Three changes: "Pre-normalization [GPT3]" using "the RMSNorm normalizing function, introduced by Zhang and Sennrich (2019)"; "SwiGLU activation function [PaLM]" with "a dimension of 2/3·4d instead of 4d as in PaLM"; and "Rotary Embeddings [GPTNeo]" where "we remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE), introduced by Su et al. (2021), at each layer of the network."

**Q4: How does LLaMA-13B compare to GPT-3?**
A4 (from the paper, Abstract): "LLaMA-13B outperforms GPT-3 (175B) on most benchmarks." Section 3.1 adds: "LLaMA-13B model also outperforms GPT-3 on most benchmarks despite being 10× smaller."

**Q5: Why does LLaMA underperform on MMLU?**
A5 (from the paper, Section 3.6): "LLaMA-65B is behind both Chinchilla-70B and PaLM-540B by a few percent in average. A potential explanation is that we have used a limited amount of books and academic papers in our pre-training data, i.e., ArXiv, Gutenberg and Books3, that sums up to only 177GB, while these models were trained on up to 2TB of books."

**Q6: What tokenizer is used?**
A6 (from the paper, Section 2.1): "We tokenize the data with the byte-pair encoding (BPE) algorithm, using the implementation from SentencePiece. Notably, we split all numbers into individual digits, and fallback to bytes to decompose unknown UTF-8 characters."

## Common Misconceptions

1. **"LLaMA was the first open-source LLM"** — Not exactly. OPT, GPT-NeoX, and BLOOM were open-sourced before LLaMA. However, LLaMA was the first open-weight model competitive with the best proprietary models (Chinchilla, PaLM). The paper notes these prior models in Section 1: "There exist some exceptions, notably OPT, GPT-NeoX, BLOOM and GLM, but none that are competitive with PaLM-62B or Chinchilla."
2. **"LLaMA uses a novel architecture"** — No single component is novel. RMSNorm, RoPE, and SwiGLU were all proposed in earlier papers. LLaMA's contribution is combining them and showing that training smaller models on more public data works — not a new architecture.
3. **"LLaMA was fully open-source"** — The models were initially released under a non-commercial research license, gated by an application process. The weights were later leaked, which accelerated the open ecosystem. LLaMA 2 (July 2023) was the first version with a commercial license.

## Real Citations

1. Hugo Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) — the original paper.
2. Hugo Touvron et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models." [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) — the direct successor, adding GQA and RLHF.
3. Noam Shazeer (2020). "GLU Variants Improve Transformer." [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) — introduced SwiGLU, used in LLaMA's FFN.
4. Jianlin Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) — introduced RoPE, used in LLaMA's attention.
5. Biao Zhang and Rico Sennrich (2019). "Root Mean Square Layer Normalization." [arXiv:1910.07467](https://arxiv.org/abs/1910.07467) — introduced RMSNorm, used in LLaMA's normalization.
6. Jordan Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models." [arXiv:2203.15556](https://arxiv.org/abs/2203.15556) — the Chinchilla scaling laws that motivated training smaller models on more tokens.

**Citation count:** about 4,500 citations on [OpenAlex](https://openalex.org/W4322718191) (checked October 2026).
